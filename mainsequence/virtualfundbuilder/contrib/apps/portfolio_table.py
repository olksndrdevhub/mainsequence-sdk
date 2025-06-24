import datetime
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from mainsequence.client import TargetPortfolio
from mainsequence.virtualfundbuilder.utils import get_vfb_logger
from pydantic import BaseModel
from mainsequence.virtualfundbuilder.resource_factory.app_factory import (
    HtmlApp,
    register_app,
)
import plotly.graph_objects as go

logger = get_vfb_logger()

PortfolioNameEnum = Enum(
    "PortfolioEnum",
    {
        p.portfolio_name: p.portfolio_ticker
        for p in TargetPortfolio.filter(is_asset_only=False, local_time_serie__isnull=False)
    },
    type=str,
)

class PortfolioTableConfiguration(BaseModel):
    report_title: str = "Portfolio Table"
    portfolio_tickers: List[PortfolioNameEnum] = [list(PortfolioNameEnum)[0].value]
    report_days: int = 365 * 5

@register_app()
class PortfolioTable(HtmlApp):
    configuration_class = PortfolioTableConfiguration

    def __init__(self, configuration: PortfolioTableConfiguration):
        logger.info(f"Create portfolio table with configuration {configuration}")
        self.configuration = configuration

    def run(self) -> str:
        """
        Build **one** Plotly table for all requested portfolios
        and return it as a single HTML document.
        Rows   : one per portfolio (max-5 shown)
        Columns: Close | 1-Day % | 7-Day SMA | 30-Day Volatility
        """
        start_date = datetime.datetime.now(
            datetime.timezone.utc
        ) - datetime.timedelta(days=self.configuration.report_days)

        rows = []
        for ticker in self.configuration.portfolio_tickers:
            try:
                portfolio = TargetPortfolio.get(portfolio_ticker=ticker)
                df = portfolio.local_time_serie.get_data_between_dates_from_api()
                df["time_index"] = pd.to_datetime(df["time_index"])
                df = df[df["time_index"] >= start_date].copy()
                df.sort_values("time_index", inplace=True)

                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                # Indicators
                df["pct_change"] = df["close"].pct_change()
                df["sma_7"] = df["close"].rolling(7, min_periods=1).mean()

                latest = df.iloc[-1]
                rows.append(
                    {
                        "Portfolio": f"{portfolio.portfolio_name} ({portfolio.portfolio_ticker})",
                        "Close": latest["close"],
                        "1-Day %": latest["pct_change"],
                        "SMA-7": latest["sma_7"],
                    }
                )
                logger.info(f"Processed metrics for {ticker}")
            except Exception as exc:
                logger.error(f"Failed for {ticker}: {exc}")

        if not rows:
            return "<h3>No data available for requested portfolios.</h3>"

        metrics_df = pd.DataFrame(rows).set_index("Portfolio")

        # Conditional colours for the 1-Day % column
        pct_color = [
            ["#d7f5d2" if v > 0 else "#fbd1d1" for v in metrics_df["1-Day %"]]
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(metrics_df.columns),
                        fill_color="lightgrey",
                        font=dict(size=12, family="Roboto Condensed"),
                        align="center",
                    ),
                    cells=dict(
                        values=[metrics_df[c] for c in metrics_df.columns],
                        format=[".2f", ".2%", ".2f", ".2%"],
                        align="center",
                        fill_color=["white"] * 3 + pct_color,
                    ),
                )
            ]
        )

        fig.update_layout(
            title=self.configuration.report_title,
            margin=dict(l=0, r=0, t=60, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        return fig.to_html(
            include_plotlyjs="cdn",
            full_html=True,
            config={"responsive": True, "displayModeBar": False},
        )


if __name__ == "__main__":
    cfg = PortfolioTableConfiguration()
    PortfolioTable(cfg).run()
