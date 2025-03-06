import copy

from mainsequence.tdag.time_series import TimeSerie, WrapperTimeSerie
from datetime import datetime
import pytz

from mainsequence.virtualfundbuilder.strategy_factory.signal_factory import WeightsBase, send_weights_as_position_to_vam
from mainsequence.virtualfundbuilder.utils import (
    reindex_df,
    build_rolling_regression_from_df,
    TIMEDELTA,
    filter_assets
)

from mainsequence.tdag.time_series import ModelList
import pandas as pd
from typing import Dict, Tuple, Union




class LongShortMomentum(WeightsBase, TimeSerie):
    """
    Momentum Strategy for the long/short strategy
    """

    @TimeSerie._post_init_routines()
    def __init__(
            self,
            signal_weights_strategy_config: Union[Dict, None],
            source_frequency: str="15m",
            momentum_window: str="48h",
            momentum_percentile: float=0.15,
            *args, **kwargs
    ):
        """
        Signal Weights using weighting by Market Capitalizations

        Args:
            source_frequency (str): Frequency of market cap source
            momentum_window (str): Timeframe for momentum calculation
            momentum_percentile (str): Top and bottom momentum percentile to include
            signal_weights_strategy_config (str): Strategy for the momentum
        """
        super().__init__(*args, **kwargs)

        self.asset_list = self.asset_universe["binance_futures"]

        self.source_frequency = source_frequency
        self.signal_weights_strategy_config = signal_weights_strategy_config

        self.momentum_window = momentum_window
        self.momentum_percentile = momentum_percentile
        assert self.momentum_percentile <= 0.5

        self._init_signal_weights()

    def _init_signal_weights(self) -> None:
        """
        Configures and initializes data sources necessary for the calculation of signal weights based on the selected strategy.
        """
        time_series_dict = {}
        asset_symbols = {}
        for exchange, asset_list in self.asset_universe.items():
            time_series_dict[exchange] = DeflatedPrices(
                asset_list=asset_list,
                portfolio_type=DeflatedPricesBase.NONE,
                portfolio_config={},
                source=exchange,
                upsample_frequency_id=self.source_frequency,
                bar_frequency_id="1m",
                intraday_bar_interpolation_rule="ffill",
            )
            asset_symbols[exchange] = [a.unique_identifier for a in asset_list]

        self.bars_ts = WrapperTimeSerie(time_series_dict=time_series_dict)
        self.asset_symbols = asset_symbols

        self.market_caps = MarketCap(
            asset_universe=self.asset_universe,
        )

    def maximum_forward_fill(self):
        return pd.Timedelta(self.source_frequency) - TIMEDELTA

    def _get_top_bottom_assets(self, returns: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Identifies the top and bottom performing assets for each time step based on their returns. This is used in momentum strategies
        to distinguish between high and low performers.

        Args:
            returns (pd.DataFrame): A DataFrame containing return data for each asset, indexed by time.

        Returns:
            Tuple[pd.Series, pd.Series]: Two Series indicating the top and bottom assets respectively, indexed by time and asset symbol.
        """
        top_thresholds = returns.quantile(q=1 - self.momentum_percentile, axis=1)
        bottom_thresholds = returns.quantile(q=self.momentum_percentile, axis=1)

        top_coins = pd.DataFrame(
            returns.to_numpy() > top_thresholds.to_numpy().reshape(-1, 1),
            columns=returns.columns, index=returns.index
        ).stack()

        bottom_coins = pd.DataFrame(
            returns.to_numpy() < bottom_thresholds.to_numpy().reshape(-1, 1),
            columns=returns.columns, index=returns.index
        ).stack()
        return top_coins, bottom_coins

    @send_weights_as_position_to_vam
    def update(self, latest_value: Union[datetime, None], *args, **kwargs) -> pd.DataFrame:
        """
        Calculates signal weights for a long/short momentum strategy using market capitalization.
        Long weights are positive while short weights are negative.

        Args:
            latest_value (Union[datetime, None]): Last value calculated

        Returns:
            DataFrame: Data containing calculated momentum weights.
        """

        max_assets_time = [ts.get_last_observation(asset_symbols=self.asset_symbols[exchange], *args, **kwargs)
                           for exchange, ts in self.bars_ts.related_time_series.items()]
        top_date_limit = min(
            [i.index.get_level_values("time_index").min() for i in max_assets_time])  # get the minimum available time

        if latest_value is None:
            latest_value = datetime(year=2018, month=1, day=1).replace(tzinfo=pytz.utc)
        else:
            # only when there are prices enough for the upsample
            upper_range = latest_value + pd.Timedelta(self.source_frequency)
            if top_date_limit < upper_range:
                return pd.DataFrame()

        prices_start_date = latest_value - pd.Timedelta(self.momentum_window)

        prices = self.bars_ts.pandas_df_concat_on_rows_by_key_between_dates(
            start_date=prices_start_date,
            end_date=top_date_limit,
            great_or_equal=True,
            less_or_equal=True,
        )

        prices = prices.reset_index().pivot(index="time_index", columns="asset_symbol", values="open")

        price_index = pd.date_range(start=prices_start_date, end=top_date_limit, freq=self.source_frequency)
        prices = prices[prices.index.isin(price_index)]

        # calculcate returns
        return_offset = int(pd.Timedelta(self.momentum_window) / pd.Timedelta(self.source_frequency))
        returns = prices.pct_change(return_offset).dropna()

        # get correct coins for top/bottom
        top_coins, bottom_coins = self._get_top_bottom_assets(returns)

        # Get market caps
        market_caps = self.market_caps.get_df_greater_than_in_table(latest_value, *args, **kwargs)
        market_caps = filter_assets(df=market_caps, asset_list=self.asset_list)
        market_caps = market_caps.reset_index().pivot(index="time_index", columns="asset_symbol",
                                                      values="signal_weight")

        # align market caps with returns
        market_caps = reindex_df(df=market_caps, start_time=returns.index.min(), end_time=returns.index.max(),
                                 freq=self.source_frequency)

        # filter only relevant market caps for top/bottom coins
        market_caps.index.name = "time_index"
        market_caps = market_caps.stack()
        top_market_caps = market_caps.loc[top_coins].dropna()
        bottom_market_caps = market_caps.loc[bottom_coins].dropna()

        # Create weights from market caps for top and bottom coins
        top_signal_weights = top_market_caps / top_market_caps.groupby("time_index").sum()
        bottom_signal_weights = bottom_market_caps / bottom_market_caps.groupby("time_index").sum()

        # bottom coins have negative weights
        bottom_signal_weights = -bottom_signal_weights

        # prepare for storage
        signal_weights = pd.concat([top_signal_weights, bottom_signal_weights], axis=0).to_frame()
        signal_weights.columns = ["signal_weight"]
        signal_weights = signal_weights[signal_weights.index.get_level_values("time_index") > latest_value]

        self.logger.info(f"{len(signal_weights)} new signal weights have been calculated.")
        return signal_weights


class Beta(TimeSerie):
    """
    Calculate beta values for assets relative to a market index over specified windows for regression and returns.
    """

    @TimeSerie._post_init_routines()
    def __init__(
            self,
            asset_universe: ModelList,
            source_frequency: str,
            regression_window: str,
            return_window: str,
            market_index_name: str,
            *args, **kwargs
    ):
        """
        Initializes the Beta class, setting up the necessary parameters and data sources for beta calculation.

        Args:
            asset_list (ModelList): List of assets for which betas will be calculated.
            source_frequency (str): Frequency at which data is updated and betas are recalculated.
            regression_window (str): The time window used for regression analysis to compute betas.
            return_window (str): The time window over which returns are calculated for beta computation.
            market_index_name (str): Name of the market index used as the benchmark for beta calculations.
        """
        from mainsequence.virtualfundbuilder.config_handling import TemplateFactory

        self.source_frequency = source_frequency

        # TODO support for all exchanges
        self.asset_universe = asset_universe
        self.asset_list = self.asset_universe["binance_futures"]

        self.market_index_name = market_index_name
        self.return_window = return_window

        time_series_dict = {}
        asset_symbols = {}
        for exchange, asset_list in self.asset_universe.items():
            time_series_dict[exchange] = DeflatedPrices(
                asset_list=asset_list,
                portfolio_type=DeflatedPricesBase.NONE,
                portfolio_config={},
                source=exchange,
                upsample_frequency_id=self.source_frequency,
                bar_frequency_id="1m",
                intraday_bar_interpolation_rule="ffill",
            )
            asset_symbols[exchange] = [a.unique_identifier for a in asset_list]

        self.bars_ts = WrapperTimeSerie(time_series_dict=time_series_dict)

        self.regression_window = regression_window
        self.index_portfolio = TemplateFactory.create_market_index(
            index_name=self.market_index_name,
        )
        super().__init__(*args, **kwargs)

    def update(self, latest_value: datetime, *args, **kwargs) -> pd.DataFrame:
        """
        Updates and calculates new beta values based on the most recent asset prices and market index returns.

        Args:
            latest_value (datetime): The timestamp of the latest data point from which to calculate new betas.

        Returns:
            pd.DataFrame: A DataFrame with calculated beta values for each asset, indexed by time and asset symbol.
        """
        if latest_value is None:
            latest_value = datetime(year=2018, month=1, day=1).replace(tzinfo=pytz.utc)
        else:
            if latest_value > (datetime.now(tz=pytz.utc) - pd.Timedelta(self.source_frequency)):
                self.logger.info(f"0 new betas are calculated.")
                return pd.DataFrame()

        price_start_value = latest_value - pd.Timedelta(self.return_window)  # for calculating the coin returns
        price_start_value = price_start_value - pd.Timedelta(self.regression_window)  # for doing regression

        # get coin returns
        prices = self.bars_ts.pandas_df_concat_on_rows_by_key_between_dates(
            start_date=price_start_value,
            end_date=None,  # TODO
            great_or_equal=True,
            less_or_equal=True,
        )

        prices = prices.reset_index().pivot(index="time_index", columns="asset_symbol", values="open")

        return_offset = int(pd.Timedelta(self.return_window) / pd.Timedelta(self.source_frequency))
        assert len(prices) >= return_offset
        coin_returns = prices.pct_change(return_offset).dropna()

        # get the returns for the market index
        index_returns = self.index_portfolio.get_df_greater_than_in_table(price_start_value, *args, **kwargs)["return"].rename(
            "MarketReturns")

        # align coin returns and market returns
        assert len(index_returns) >= 1
        coin_returns = coin_returns.loc[index_returns.index]

        # do regression on index and individual coins returns
        regression_offset = int(pd.Timedelta(self.regression_window) / pd.Timedelta(self.source_frequency))
        results = build_rolling_regression_from_df(
            x=index_returns.values,
            y=coin_returns.values,
            rolling_window=regression_offset,
            column_names=coin_returns.columns
        )

        # prepare for output
        results.index = coin_returns.index[regression_offset - 1:]
        results = results.stack(level=1)
        results.index.names = ["time_index", "asset_symbol"]
        results = results[results.index.get_level_values("time_index") > latest_value]
        return results


class BetaMomentum(WeightsBase, TimeSerie):

    @TimeSerie._post_init_routines()
    def __init__(
            self,
            source_frequency: str,
            momentum_window: str,
            momentum_percentile: float,
            signal_weights_strategy_config: Union[Dict, None],
            *args, **kwargs
    ):
        """
        Initializes a new instance of the SignalWeights class, setting up market data and weight calculations.

        Args:
            asset_list (ModelList): List of assets for which weights will be calculated.
            source_frequency (str): Frequency of data update (e.g., daily, hourly).
            signal_weights_strategy_config (dict): Additional parameters for the weights strategy.
        """

        super().__init__(*args, **kwargs)
        self.asset_list = self.asset_universe["binance_futures"]

        self.source_frequency = source_frequency
        self.signal_weights_strategy_config = signal_weights_strategy_config

        self.momentum_window = momentum_window
        self.momentum_percentile = momentum_percentile
        assert self.momentum_percentile <= 0.5

        self._init_signal_weights()

    def maximum_forward_fill(self):
        return pd.Timedelta(self.source_frequency) - TIMEDELTA

    def _init_signal_weights(self) -> None:
        """
        Configures and initializes data sources necessary for the calculation of signal weights based on the selected strategy.
        """
        self.momentum_strategy_config = self.signal_weights_strategy_config["momentum_strategy_config"]
        self.beta = Beta(
            asset_universe=copy.deepcopy(self.asset_universe),
            source_frequency=self.source_frequency,
            **self.momentum_strategy_config
        )

        self.longshort_signal_weights = LongShortMomentum(
            asset_universe=copy.deepcopy(self.asset_universe),
            source_frequency=self.source_frequency,
            momentum_window=self.momentum_window,
            momentum_percentile=self.momentum_percentile,
            signal_weights_strategy_config=None,
        )

    def update(self, latest_value: Union[datetime, None], *args, **kwargs) -> pd.DataFrame:
        """
        Calculates signal weights in a way that aims to neutralize beta across the portfolio.
        First a long/short strategy is used to get the top and bottom coins and then the weights are adapted for a
        neutral beta.

        Args:
            latest_value (Union[datetime, None]): The most recent timestamp of signal weights.

        Returns:
            pd.DataFrame: A DataFrame with adjusted weights aiming for beta neutrality, indexed by time and asset symbol.
        """
        if latest_value is not None:
            if latest_value > (datetime.now(tz=pytz.utc) - pd.Timedelta(self.source_frequency)):
                self.logger.info(f"0 new signal weights are calculated.")
                return pd.DataFrame()

        # get betas
        betas = self.beta.get_df_greater_than_in_table(latest_value, *args, **kwargs)[["beta"]]
        betas = betas.reset_index().pivot(index="time_index", columns="asset_symbol", values="beta")

        # get long/short signal weights for top and bottom
        top_bottom_weights = self.longshort_signal_weights.get_df_greater_than_in_table(latest_value, *args, **kwargs)[
            "signal_weight"]
        top_bottom_weights = top_bottom_weights[
            top_bottom_weights.index.get_level_values("time_index").isin(betas.index)]
        top_weights = top_bottom_weights[top_bottom_weights > 0].unstack("asset_symbol")
        bottom_weights = top_bottom_weights[top_bottom_weights < 0].unstack("asset_symbol")

        # simple scaling to create equal beta for top and bottom
        betas_pos = (betas * top_weights).sum(axis=1)
        betas_neg = -(betas * bottom_weights).sum(axis=1)

        scale_factor = betas_neg / betas_pos

        # TODO: this code below needs to be adapted, just temoporary fix as beta_pos can get very negative
        scale_factor[scale_factor < 0] = 0
        optimized_top_weights = top_weights.mul(scale_factor, axis=0)

        # join for output
        signal_weights = pd.concat([optimized_top_weights.stack(), bottom_weights.stack()], axis=0).to_frame()
        signal_weights = signal_weights / signal_weights.abs().groupby(
            "time_index").sum()  # scale weights TODO evaluate
        signal_weights.columns = ["signal_weight"]
        return signal_weights




