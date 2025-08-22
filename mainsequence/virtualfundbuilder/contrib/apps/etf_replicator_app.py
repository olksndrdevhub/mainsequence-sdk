import datetime
from enum import Enum
from typing import List, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from mainsequence.client import Portfolio, Asset
from mainsequence.virtualfundbuilder.contrib.data_nodes import TrackingStrategy, TrackingStrategyConfiguration
from mainsequence.virtualfundbuilder.models import PortfolioConfiguration
from mainsequence.virtualfundbuilder.portfolio_interface import PortfolioInterface
from mainsequence.virtualfundbuilder.utils import get_vfb_logger
from pydantic import BaseModel
from mainsequence.virtualfundbuilder.resource_factory.app_factory import (
    HtmlApp,
    register_app,
)



logger = get_vfb_logger()

portfolio_ids = [portfolio.id for portfolio in Portfolio.filter(local_time_serie__isnull=False)]
class ETFReplicatorConfiguration(BaseModel):
    source_asset_category_identifier: str = "magnificent_7"
    etf_to_replicate: str = "XLF"
    in_window: int = 60
    tracking_strategy: TrackingStrategy = TrackingStrategy.LASSO
    tracking_strategy_configuration: TrackingStrategyConfiguration

@register_app()
class ETFReplicatorApp(HtmlApp):
    configuration_class = ETFReplicatorConfiguration

    def run(self) -> str:

        # get a portfolio template
        portfolio_config = PortfolioInterface.load_configuration(configuration_name="market_cap")
        signal_weights_configuration = {
            "etf_ticker": self.configuration.etf_to_replicate,
            "in_window": self.configuration.in_window,
            "tracking_strategy": self.configuration.tracking_strategy,
            "tracking_strategy_configuration": self.configuration.tracking_strategy_configuration,
            "signal_assets_configuration": portfolio_config.portfolio_build_configuration.backtesting_weights_configuration.signal_weights_configuration["signal_assets_configuration"]
        }
        signal_weights_configuration["signal_assets_configuration"].assets_category_unique_id = self.configuration.source_asset_category_identifier

        portfolio_config.portfolio_build_configuration.backtesting_weights_configuration.signal_weights_configuration = signal_weights_configuration
        portfolio_config.portfolio_build_configuration.backtesting_weights_configuration.signal_weights_name = "ETFReplicator"
        portfolio_config.portfolio_markets_configuration.portfolio_name = f"ETFReplicator Portfolio for {self.configuration.etf_to_replicate} using {self.configuration.source_asset_category_identifier}"

        portfolio = PortfolioInterface(portfolio_config_template=portfolio_config.model_dump())

        results = portfolio.run(add_portfolio_to_markets_backend=True)

        if results.empty:
            return "<html><body><h1>No data available to generate ETF replication report.</h1></body></html>"

        # Get original ETF data for comparison
        etf_replicator_signal = portfolio.portfolio_strategy_time_serie.signal_weights
        etf_asset = etf_replicator_signal.etf_asset
        etf_data = etf_replicator_signal.etf_bars_ts.get_df_between_dates(
            start_date=results.index.min(),
            end_date=results.index.max(),
            unique_identifier_list=[etf_asset.unique_identifier]
        )

        # Prepare DataFrames for as-of merge
        replicated_df = results.sort_index().reset_index()[['time_index', 'close']].rename(
            columns={'close': f"Replicated_{self.configuration.etf_to_replicate}"}
        )
        original_df = etf_data.sort_index().reset_index()[['time_index', 'close']].rename(
            columns={'close': self.configuration.etf_to_replicate}
        )

        df_plot = pd.concat([original_df, replicated_df]).sort_values("time_index").ffill()
        df_plot = df_plot.set_index('time_index').dropna()

        if df_plot.empty:
            return "<html><body><h1>Could not align original and replicated portfolio data.</h1></body></html>"

        # Normalize to 100
        df_plot_normalized = (df_plot / df_plot.iloc[0]) * 100

        # Create the Plotly figure
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df_plot_normalized.index,
            y=df_plot_normalized[self.configuration.etf_to_replicate],
            mode='lines',
            name=f"Original: {self.configuration.etf_to_replicate}"
        ))

        fig.add_trace(go.Scatter(
            x=df_plot_normalized.index,
            y=df_plot_normalized[f"Replicated_{self.configuration.etf_to_replicate}"],
            mode='lines',
            name="Replicated Portfolio"
        ))

        fig.update_layout(
            title=f"ETF Replication Performance: {self.configuration.etf_to_replicate} vs. Replicated Portfolio",
            xaxis_title="Date",
            yaxis_title="Normalized Performance (Indexed to 100)",
            legend_title="Series"
        )

        html_chart = fig.to_html(full_html=False, include_plotlyjs='cdn')

        return html_chart


if __name__ == "__main__":
    cfg = ETFReplicatorConfiguration(
        tracking_strategy_configuration=TrackingStrategyConfiguration(),
        source_asset_category_identifier="s&p500_constitutents"
    )
    ETFReplicatorApp(cfg).run()
