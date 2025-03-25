
from mainsequence.tdag.time_series import TimeSerie, APITimeSerie

from datetime import datetime, timedelta, tzinfo
from typing import Union, List

import pandas as pd
import pytz

from mainsequence.tdag.time_series import TimeSerie
from mainsequence.client import CONSTANTS,Asset
from mainsequence.virtualfundbuilder.enums import ExecutionVenueNames

from mainsequence.virtualfundbuilder.models import VFBConfigBaseModel
from mainsequence.virtualfundbuilder.strategy_factory.signal_factory import WeightsBase, register_signal_class
from mainsequence.virtualfundbuilder.utils import TIMEDELTA


class SymbolWeight(VFBConfigBaseModel):
    execution_venue_symbol: str = CONSTANTS.ALPACA_EV_SYMBOL
    symbol: str
    weight: float

@register_signal_class(register_in_agent=True)
class FixedWeights(WeightsBase, TimeSerie):

    @TimeSerie._post_init_routines()
    def __init__(self, asset_symbol_weights: List[SymbolWeight], *args, **kwargs):
        """
        Args:
            asset_symbol_weights (List[SymbolWeight]): List of SymbolWeights that map asset symbols to weights
        """
        super().__init__(*args, **kwargs)
        self.asset_symbol_weights = asset_symbol_weights

    def maximum_forward_fill(self):
        return timedelta(days=200 * 365)  # Always forward-fill to avoid filling the DB

    def get_explanation(self):
        max_rows = 10
        symbols = [w.symbol for w in self.asset_symbol_weights]
        weights = [w.weight for w in self.asset_symbol_weights]
        info = f"<p>{self.__class__.__name__}: Signal uses fixed weights with the following weights:</p><div style='display: flex;'>"

        for i in range(0, len(symbols), max_rows):
            info += "<table border='1' style='border-collapse: collapse; margin-right: 20px;'><tr>"
            info += ''.join(f"<th>{sym}</th>" for sym in symbols[i:i + max_rows])
            info += "</tr><tr>"
            info += ''.join(f"<td>{wgt}</td>" for wgt in weights[i:i + max_rows])
            info += "</tr></table>"

        info += "</div>"
        return info

    def update(self, latest_value: Union[datetime, None], *args, **kwargs) -> pd.DataFrame:
        if latest_value is not None:
            return pd.DataFrame()  # No need to store more than one constant weight
        latest_value = latest_value or datetime(1985, 1, 1).replace(tzinfo=pytz.utc)

        df = pd.DataFrame([m.model_dump() for m in self.asset_symbol_weights]).rename(columns={'symbol': 'asset_symbol',
                                                                                               'weight': 'signal_weight'})
        df = df.set_index(['asset_symbol', 'execution_venue_symbol'])

        signals_weights = pd.concat(
            [df],

            axis=0,
            keys=[latest_value]
        ).rename_axis(["time_index", "asset_symbol", "execution_venue_symbol"])

        signals_weights = signals_weights.dropna()
        return signals_weights

@register_signal_class(register_in_agent=True)
class MarketCap(WeightsBase, TimeSerie):
    EXECUTION_TO_DATA_PROVIDER_MAP = {
        ExecutionVenueNames.ALPACA: "polygon",
        ExecutionVenueNames.BINANCE_FUTURES: "coingecko",
        ExecutionVenueNames.BINANCE: "coingecko"
    }

    @TimeSerie._post_init_routines()
    def __init__(self, source_frequency: str = "1d", num_top_assets: Union[int, None] = None, *args, **kwargs):
        """
        Signal Weights using weighting by Market Capitalizations.

        Args:
            source_frequency (str): Frequency of market cap source.
            num_top_assets (Optional[int]): Number of largest assets by market cap to use for signals. Leave empty to include all assets.
        """
        super().__init__(*args, **kwargs)
        self.source_frequency = source_frequency
        self.num_top_assets = num_top_assets or 50000
        self.historical_market_cap_ts = APITimeSerie.build_from_unique_identifier(CONSTANTS.data_sources_constants.HISTORICAL_MARKET_CAP)

        self.execution_venue_symbol=self.asset_universe.get_required_execution_venues()
        if len(self.execution_venue_symbol) > 1:
            raise ValueError(
                f"No support to compare MarketCaps of asset universes {self.execution_venue_symbol}")
        self.execution_venue_symbol=ExecutionVenueNames(self.execution_venue_symbol[0])

    def maximum_forward_fill(self):
        return timedelta(days=1) - TIMEDELTA

    def get_explanation(self):
        return (
            f"{self.__class__.__name__}:\n"
            f"The signal strategy leverages market capitalization to construct dynamic weightings for asset selection. By prioritizing assets with higher market value, the signal emphasizes stability and liquidity in its decision-making process.\n"
            f"\n"
            f"Source Frequency:\n"
            f"The signal relies on {self.source_frequency} data updates, ensuring that the weights reflect the most recent market conditions.\n"
            f"\n"
            f"Number of Top Assets:\n"
            f"This strategy focuses on the top {self.num_top_assets} assets, optimizing for performance by selecting the most influential components of the market."
        )

    def update(self, update_statistics):
        """
        Args:
            latest_value (Union[datetime, None]): The timestamp of the most recent data point.

        Returns:
            DataFrame: A DataFrame containing updated signal weights, indexed by time and asset symbol.
        """

        VENUE_MAP={}
        for v in CONSTANTS.BINANCE_VENUES:
            VENUE_MAP[v] = CONSTANTS.ASSET_TYPE_CRYPTO_SPOT

        VENUE_MAP[CONSTANTS.ALPACA_EV_SYMBOL]= CONSTANTS.ASSET_TYPE_CASH_EQUITY

        market_cap_symbol_map=lambda asset:f"{asset.get_spot_reference_asset_symbol()}-*-{VENUE_MAP[asset.execution_venue.symbol]}"

        asset_universe = self.asset_universe
        execution_venue_symbol = self.execution_venue_symbol

        if execution_venue_symbol not in self.EXECUTION_TO_DATA_PROVIDER_MAP: raise ValueError(f"Unknown execution venue symbol {execution_venue_symbol} in market cap")
        if asset_universe.asset_list is None:
            asset_list = Asset.filter_with_asset_class(**asset_universe.asset_filter,
                                                    )
        else:
            asset_list = asset_universe.asset_list

        update_statistics = update_statistics.update_assets(asset_list, init_fallback_date=datetime(2017,1,1).replace(tzinfo=pytz.utc))
        unique_identifier_range_map = {market_cap_symbol_map(a):{"start_date":update_statistics[a.unique_identifier],
                                                                            "start_date_operand":">"}
                                                                            for a in asset_list}

        mc = self.historical_market_cap_ts.get_df_between_dates(unique_identifier_range_map=unique_identifier_range_map,
                                                                great_or_equal=False,)
        if mc.shape[0] == 0:
            return pd.DataFrame()
        asset_map = {a.get_spot_reference_asset_symbol(): a.unique_identifier for a in asset_list}
        mc.index = mc.index.set_levels(
            mc.index.levels[mc.index.names.index("unique_identifier")]
            .str.split('-*-').str[0]
            .map(asset_map),
              # split on '-*-' and take [0]
            level=1
        )

        mc_pivot = mc.reset_index().pivot(
            index="time_index",
            columns=["unique_identifier"],
            values="market_cap"
        )
        mc_pivot = mc_pivot.ffill().bfill()
        assets_excluded = mc_pivot.rank(axis=1, ascending=False) > self.num_top_assets
        mc_pivot[assets_excluded] = 0  # Exclude assets not in the top by setting weight to 0

        mc_pivot = mc_pivot.div(mc_pivot.sum(axis=1), axis=0)
        signal_weights = mc_pivot.stack().rename("signal_weight").to_frame()
        return signal_weights
