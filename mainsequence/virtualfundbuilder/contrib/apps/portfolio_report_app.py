import datetime
import os
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from mainsequence.client import TargetPortfolio
from mainsequence.virtualfundbuilder.utils import get_vfb_logger
from plotly.subplots import make_subplots

from pydantic import BaseModel
from mainsequence.virtualfundbuilder.resource_factory.app_factory import BaseApp, register_app, HtmlApp
import plotly.graph_objects as go

logger = get_vfb_logger()

PortfolioNameEnum = Enum(
    "PortfolioEnum",
    {portfolio.portfolio_name: portfolio.portfolio_ticker for portfolio in TargetPortfolio.filter(is_asset_only=False, local_time_serie__isnull=False)},
    type=str,
)
class PortfolioReportConfiguration(BaseModel):
    report_title: str = "Portfolio Report"
    portfolio_ticker: List[PortfolioNameEnum]
    report_days: int = 365

@register_app()
class PortfolioReport(HtmlApp):
    configuration_class = PortfolioReportConfiguration

    def __init__(self, configuration: PortfolioReportConfiguration):
        logger.info(f"Create portfolio report {configuration}")
        self.configuration = configuration

    def run(self) -> None:
        """
        Generates and saves the portfolio report.

        This method fetches data for each portfolio specified in the configuration,
        plots their closing prices over time using Plotly, and saves the
        resulting figure as a PNG image.
        """
        num_portfolios = len(self.configuration.portfolio_ticker)

        # Create subplots, one for each portfolio
        fig = make_subplots(
            rows=num_portfolios,
            cols=1,
            subplot_titles=[f"Performance for {ticker}" for ticker in self.configuration.portfolio_ticker]
        )

        # Define the start date for the report
        start_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=self.configuration.report_days)

        for i, ticker in enumerate(self.configuration.portfolio_ticker):
            try:

                portfolio = TargetPortfolio.get(portfolio_ticker=ticker)
                data = portfolio.local_time_serie.get_data_between_dates_from_api()
                logger.info(f"Successfully fetched data for portfolio: {ticker}")

                data['time_index'] = pd.to_datetime(data['time_index'])
                report_data = data[data['time_index'] >= start_date].copy()

                # Add a scatter plot for the portfolio's close price
                fig.add_trace(
                    go.Scatter(
                        x=report_data['time_index'],
                        y=report_data['close'],
                        mode='lines',
                        name=f"{portfolio.portfolio_name} ({portfolio.portfolio_ticker})"
                    ),
                    row=i + 1,
                    col=1
                )

            except Exception as e:
                logger.error(f"Could not process portfolio {ticker}. Error: {e}")

        fig.update_layout(
            title_text=self.configuration.report_title,
            height=300 * num_portfolios,
            showlegend=False
        )

        # Update y-axes to show 'Close Price' title
        fig.update_yaxes(title_text="Close Price")
        return fig.to_html(
            include_plotlyjs=True,
            full_html=False,
            config={'responsive': True, 'displayModeBar': False}
        )

if __name__ == "__main__":
    configuration = PortfolioReportConfiguration(
        portfolio_ticker=[list(PortfolioNameEnum)[0].value] # test with a sample portfolio
    )
    PortfolioReport(configuration).run()