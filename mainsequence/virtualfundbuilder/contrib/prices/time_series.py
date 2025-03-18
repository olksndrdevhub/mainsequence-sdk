import copy
import pytz
from typing import Union, Dict, List,Literal, Optional
import pandas as pd
import numpy as np
import datetime
import pandas_market_calendars as mcal

from mainsequence.tdag.time_series import TimeSerie, WrapperTimeSerie, ModelList, APITimeSerie,data_source_pickle_path
from mainsequence.client import(CONSTANTS, LocalTimeSeriesDoesNotExist, LocalTimeSerie, DynamicTableDataSource,
                                BACKEND_DETACHED, DataUpdates
                                )
from mainsequence.client import VAM_CONSTANTS as ASSET_ORM_CONSTANTS, ExecutionVenue
from mainsequence.client import HistoricalBarsSource, DoesNotExist, Asset
from mainsequence.tdag.time_series.utils import (
    string_frequency_to_minutes,
    string_freq_to_time_delta,
)
import os

from tqdm import tqdm
from joblib import Parallel, delayed

from mainsequence.virtualfundbuilder.models import AssetsConfiguration, AssetUniverse, AssetFilter
from mainsequence.virtualfundbuilder.utils import logger

FULL_CALENDAR = "24/7"

def get_prices_timeseries(assets_configuration: AssetsConfiguration):
    """
    Creates a Wrapper Timeseries for an asset configuration.
    """
    time_series_dict = {}

    prices_configuration = copy.deepcopy(assets_configuration).prices_configuration
    is_live = prices_configuration.is_live
    prices_configuration_kwargs = prices_configuration.model_dump()
    prices_configuration_kwargs.pop("is_live", None)

    data_mode = ASSET_ORM_CONSTANTS.DATA_MODE_LIVE if is_live else  ASSET_ORM_CONSTANTS.DATA_MODE_BACKTEST

    asset_universe = assets_configuration.asset_universe
    venue_asset_filters_map = asset_universe.get_filters_per_execution_venue()
    for execution_venue, asset_filters in venue_asset_filters_map.items():
        tmp_universe = AssetUniverse(asset_filters=asset_filters)
        time_series_dict[execution_venue] = InterpolatedPrices(
            execution_venue_symbol=execution_venue,
            asset_universe=tmp_universe,
            data_mode=data_mode,
            **prices_configuration_kwargs
        )

    return WrapperTimeSerie(time_series_dict=time_series_dict)



def get_prices_source(source:str, bar_frequency_id,
                      asset_filters: Optional[List[AssetFilter]]=None,
                      is_train: bool = True):
    """
    Returns the appropriate bar time series based on the asset list and source.
    """
    if source in [ASSET_ORM_CONSTANTS.MAIN_SEQUENCE_PORTFOLIOS_EV]:
        raise NotImplementedError
        if is_train:
            backtest_time_series={}
            for asset in asset_list:
                data_source = DynamicTableDataSource.get(asset.backtest_portfolio_data_source_id)
                pp = data_source_pickle_path(data_source.id)
                if os.path.isfile(pp) == False:
                    data_source.persist_to_pickle(pp)
                backtest_time_series[asset.id], pickle_path = TimeSerie.rebuild_and_set_from_local_hash_id(
                    local_hash_id=asset.backtest_portfolio.local_time_serie_hash_id,
                    data_source_id =data_source.id,
                    graph_depth_limit=-1,
                )
            return WrapperTimeSerie(time_series_dict=backtest_time_series)
        else:
            live_time_series = {}
            for asset in asset_list:
                data_source = DynamicTableDataSource.get(asset.live_portfolio_data_source_id)
                pp = data_source_pickle_path(data_source.id)
                if os.path.isfile(pp) == False:
                    data_source.persist_to_pickle(pp)
                live_time_series[asset.id], pickle_path =  TimeSerie.rebuild_and_set_from_local_hash_id(
                                                            local_hash_id=asset.live_portfolio.local_time_serie_hash_id,
                                                            data_source_id =data_source.id,
                                                            graph_depth_limit=-1,
                                                        )

            return WrapperTimeSerie(time_series_dict=live_time_series)

    else:
        data_mode = ASSET_ORM_CONSTANTS.DATA_MODE_BACKTEST if is_train else ASSET_ORM_CONSTANTS.DATA_MODE_LIVE
        try:
            hbs = HistoricalBarsSource.get(execution_venues__symbol=source, data_frequency_id=bar_frequency_id,
                                           data_mode=data_mode, adjusted=True)
        except DoesNotExist as e:
            logger.exception(f"HistoricalBarsSource does not exist for {source} -{bar_frequency_id} {data_mode}")
            raise e
        api_ts = APITimeSerie(data_source_id=hbs.related_local_time_serie.remote_table.data_source.id,
                              local_hash_id=hbs.related_local_time_serie.local_hash_id)
        return api_ts


class UpsampleAndInterpolation:
    """
    Handles upsampling and interpolation of bar data.
    """
    TIMESTAMP_COLS = ["first_trade_time", "last_trade_time", "open_time"]

    def __init__(
            self,
            bar_frequency_id: str,
            upsample_frequency_id: str,
            intraday_bar_interpolation_rule: str,
    ):
        self.bar_frequency_id = bar_frequency_id
        self.upsample_frequency_id = upsample_frequency_id
        self.intraday_bar_interpolation_rule = intraday_bar_interpolation_rule

        rows = string_frequency_to_minutes(self.upsample_frequency_id) / string_frequency_to_minutes(
            self.bar_frequency_id)
        assert rows.is_integer()

        if "days" in self.bar_frequency_id:
            assert bar_frequency_id == self.upsample_frequency_id  # Upsampling for daily bars not implemented

        self.upsample_frequency_td = string_freq_to_time_delta(self.upsample_frequency_id)

    @staticmethod
    def upsample_bars(
            bars_df: pd.DataFrame,
            upsample_frequency_obs: int,
            upsample_frequency_td: object,
            calendar: str,
            open_to_close_time_delta: datetime.timedelta,
            is_portfolio: bool = False
    ) -> pd.DataFrame:
        """
        Upsamples the bars dataframe based on the given parameters.
        For example, it can convert 5-minute bars to 1-minute bars.
        Note that this method works on iloc as the underlying data should be already interpolated so should be completed


        Args:
            bars_df (pd.DataFrame): The bars data to be upsampled.
            upsample_frequency_obs (int): Frequency for upsampling.
            upsample_frequency_td (object): Time delta for upsampling.
            calendar (str): Trading calendar to account for trading hours.
            open_to_close_time_delta (datetime.timedelta): Time delta between open and close.
            is_portfolio (bool): Whether the data is for a portfolio or a single asset.

        Returns:
            pd.DataFrame: The upsampled bars dataframe.
        """
        obs = bars_df.shape[0] / upsample_frequency_obs
        assert obs > 1.0

        trading_halts = calendar != FULL_CALENDAR
        calendar = mcal.get_calendar(calendar)

        full_schedule = calendar.schedule(bars_df["trade_day"].min(), bars_df["trade_day"].max()).reset_index()
        full_schedule["index"] = full_schedule["index"].apply(lambda x: x.timestamp())
        full_schedule = full_schedule.set_index("index").to_dict("index")

        all_dfs = []
        for i in tqdm(range(bars_df.shape[0] - upsample_frequency_obs + 1),
                      desc=f"Upsampling from {bars_df['trade_day'].iloc[0]} to {bars_df['trade_day'].iloc[-1]} for assets {bars_df['unique_identifier'].dropna().unique()}"):
            start = i
            end = i + upsample_frequency_obs
            tmp_df = bars_df.iloc[start:end]

            day_schedule = full_schedule[tmp_df["trade_day"].iloc[0].timestamp()]
            first_available_bar = day_schedule["market_open"] + upsample_frequency_td
            last_available_bar = day_schedule["market_close"]

            if trading_halts and tmp_df.index[-1] < first_available_bar:
                # edge case 1market is close should not upsample to the next day
                continue
            elif trading_halts and tmp_df.index[-1] > last_available_bar:
                continue
            else:
                dollar = tmp_df.vwap * tmp_df.volume
                volume = np.nansum(tmp_df.volume.values)
                vwap = np.nansum(dollar.values) / volume
                close = tmp_df.close.iloc[-1]
                vwap = vwap if not np.isnan(vwap) else close
                new_bar = {
                    "open_time": tmp_df.index[0] - open_to_close_time_delta,
                    "time": tmp_df.index[-1],
                    "volume": volume,
                    "vwap": vwap,
                    "open": tmp_df.open.iloc[0],
                    "close": close,
                }
                if not is_portfolio:
                    new_bar.update({
                        "high": np.nanmax(tmp_df.high.values),
                        "low": np.nanmin(tmp_df.low.values),
                    })

            all_dfs.append(new_bar)

        all_dfs = pd.DataFrame(all_dfs)
        all_dfs["unique_identifier"] = bars_df["unique_identifier"].iloc[0]
        all_dfs = all_dfs.set_index("time")

        return all_dfs

    def get_interpolated_upsampled_bars(
            self,
            calendar: str,
            tmp_df: pd.DataFrame,
            last_observation: Union[None, pd.Series] = None
    ) -> pd.DataFrame:
        """
        Gets interpolated and upsampled bars based on the given parameters.
        First interpolates the data to fill any gaps, then upsamples it to the desired frequency.

        Args:
            calendar (str): Trading calendar for interpolation and upsampling.
            tmp_df (pd.DataFrame): Dataframe containing the bars to be processed.
            last_observation (Union[None, pd.Series], optional): Last observed data to fill gaps.

        Returns:
            pd.DataFrame: Interpolated and upsampled bars dataframe.
        """
        for col in self.TIMESTAMP_COLS:
            try:
                if col in tmp_df.columns:
                    tmp_df[col] = pd.to_datetime(tmp_df[col], utc=True)
            except Exception as e:
                raise e

        if "d" in self.bar_frequency_id:
            tmp_df = interpolate_daily_bars(
                bars_df=tmp_df.copy(),
                interpolation_rule=self.intraday_bar_interpolation_rule,
                calendar=calendar,
                last_observation=last_observation,
            )
        elif "m" in self.bar_frequency_id:
            bars_frequency_min = string_frequency_to_minutes(self.bar_frequency_id)

            # Interpolation to fill gaps
            tmp_df = interpolate_intraday_bars(
                bars_df=tmp_df.copy(),
                interpolation_rule=self.intraday_bar_interpolation_rule,
                calendar=calendar,
                bars_frequency_min=bars_frequency_min,
                last_observation=last_observation,
            )

        if len(tmp_df) == 0:
            return tmp_df

        assert tmp_df.isnull().sum()[["close", "open"]].sum() == 0

        # Upsample to the correct frequency
        if "d" in self.bar_frequency_id:
            all_columns = self.TIMESTAMP_COLS
            upsampled_df = tmp_df
        else:
            upsample_freq_obs = string_frequency_to_minutes(self.upsample_frequency_id) // bars_frequency_min

            if upsample_freq_obs > bars_frequency_min:
                upsampled_df = UpsampleAndInterpolation.upsample_bars(
                    bars_df=tmp_df,
                    upsample_frequency_obs=upsample_freq_obs,
                    upsample_frequency_td=self.upsample_frequency_td,
                    calendar=calendar,
                    is_portfolio=False,
                    open_to_close_time_delta=datetime.timedelta(minutes=bars_frequency_min),
                )
            else:
                upsampled_df = tmp_df
            all_columns = self.TIMESTAMP_COLS + ["trade_day"]

        for col in all_columns:
            if col in upsampled_df.columns:
                upsampled_df[col] = pd.to_datetime(upsampled_df[col]).astype(np.int64).values

        return upsampled_df


def interpolate_daily_bars(
        bars_df: pd.DataFrame,
        interpolation_rule: str,
        calendar: str,
        last_observation: Union[None, pd.Series] = None,
):
    try:
        calendar_instance = mcal.get_calendar(calendar.name)
    except Exception as e:
        raise e

    def rebase_with_forward_fill(bars_df, last_observation):
        try:
            if last_observation is not None:
                if "interpolated" in last_observation.columns:
                    last_observation = last_observation.drop(columns="interpolated")

                bars_df = pd.concat([last_observation, bars_df], axis=0).sort_index()
                if "unique_identifier" in bars_df.columns:
                    bars_df.loc[:, ['unique_identifier']] = bars_df[
                        ['unique_identifier']
                    ].bfill().ffill()

            null_index = bars_df[bars_df.isnull().any(axis=1)].index
            bars_df.close = bars_df.close.ffill()
            bars_df.loc[null_index, "open"] = bars_df.loc[null_index, "close"]
            try:
                bars_df.volume = bars_df.volume.fillna(0)
            except Exception as e:
                raise e
            if "vwap" in bars_df.columns:
                bars_df.vwap = bars_df.vwap.ffill()
            if "trade_count" in bars_df.columns:
                bars_df.trade_count = bars_df.trade_count.fillna(0)

            if len(null_index) > 0:
                if "high" in bars_df.columns:
                    bars_df.loc[null_index, "high"] = bars_df.loc[null_index, "close"]
                    bars_df.loc[null_index, "low"] = bars_df.loc[null_index, "close"]

                bars_df["interpolated"] = False
                bars_df.loc[null_index, "interpolated"] = True

            else:
                bars_df["interpolated"] = False

            if last_observation is not None:
                bars_df = bars_df.iloc[1:]
        except Exception as e:
            raise e

        return bars_df

    # Restrict to calendar types
    restricted_schedule = None
    full_index = bars_df.index

    restricted_schedule = calendar_instance.schedule(bars_df.index.min(),
                                                     bars_df.index.max())  # This needs to be faster
    restricted_schedule = restricted_schedule.reset_index()
    market_type="market_open" #as bars observation date is at max a close we need to restrick the  scheduler to be at max open
   

    restricted_schedule = restricted_schedule.set_index(market_type)
    full_index = bars_df.index.union(restricted_schedule.index)

    bars_df = bars_df.reindex(full_index)

    if interpolation_rule == "None":
        pass
    elif interpolation_rule == "ffill":
        bars_df = rebase_with_forward_fill(bars_df, last_observation=last_observation)
        if last_observation is None:
            bars_df = bars_df.bfill()
    else:
        raise Exception

    if len(bars_df):
        last_observation = bars_df.iloc[[-1]]

    if len(bars_df) == 0:
        return pd.DataFrame()

    bars_df = bars_df[bars_df.index.isin(full_index)]

    null_index = bars_df[bars_df["open_time"].isnull()].index
    if len(null_index) > 0:
        bars_df.loc[null_index, "open_time"] = restricted_schedule.loc[null_index].index

    return bars_df


def interpolate_intraday_bars(
        bars_df: pd.DataFrame,
        interpolation_rule: str,
        bars_frequency_min: int,
        calendar: str,
        last_observation: Union[None, pd.Series] = None,
) -> pd.DataFrame:
    """
    Interpolates intraday bars based on the given parameters. Fills in missing data points in intraday bar data in case of gaps.
    """
    calendar_instance = mcal.get_calendar(calendar.name)

    def build_daily_range_from_schedule(start, end):
        return pd.date_range(start=start, end=end, freq=f"{bars_frequency_min}min")

    def sanitize_today_update(x: pd.DataFrame, date_range):
        today = datetime.datetime.utcnow()
        if day.date() == today.date():
            x.index.name = None
            date_range = [i for i in date_range if i <= x.index.max()]
        return date_range

    def rebase_withoutnan_fill(x, trade_starts, trade_ends):
        date_range = build_daily_range_from_schedule(trade_starts, trade_ends)
        date_range = sanitize_today_update(x=x, date_range=date_range)
        x = x.reindex(date_range)
        return x

    def rebase_with_forward_fill(x, trade_starts, trade_ends, last_observation):
        is_start_of_day = False
        if (x.shape[0] == 1) and x.index[0].hour == 0:
            is_start_of_day = True
        x["interpolated"] = False
        if not is_start_of_day:
            date_range = build_daily_range_from_schedule(trade_starts, trade_ends)
            date_range = sanitize_today_update(x=x, date_range=date_range)
            try:
                x = x.reindex(date_range)

                if last_observation is not None:
                    if "interpolated" in x.columns:
                        last_observation = last_observation.drop(columns="interpolated")
                    x = pd.concat([last_observation, x], axis=0)

                null_index = x[x["close"].isnull()].index
                x.close = x.close.ffill()
                x.loc[null_index, "open"] = x.loc[null_index, "close"]
                x.volume = x.volume.fillna(0)
                x.vwap = x.vwap.ffill()
                if "trade_count" in x.columns:
                    x.trade_count = x.trade_count.fillna(0)
                x["interpolated"] = False
                if len(null_index) > 0:
                    if "high" in x.columns:
                        x.loc[null_index, "high"] = x.loc[null_index, "close"]
                        x.loc[null_index, "low"] = x.loc[null_index, "close"]

                    x.loc[null_index, "interpolated"] = True

                if last_observation is not None:
                    x = x.iloc[1:]
            except Exception as e:
                raise e

        #interpolate any other columns with 0


        return x

    full_index = bars_df.index

    # because index are closes the greates value should be the open time of the last close to do not extra interpolate
    restricted_schedule = calendar_instance.schedule(bars_df.index.min(),
                                                     bars_df.iloc[-1]["open_time"])  # This needs to be faster

    bars_df = bars_df[~bars_df.index.duplicated(keep='first')]  # todo: remove uncessary with indices.

    full_index = bars_df.index.union(restricted_schedule.set_index("market_open").index).union(
        restricted_schedule.set_index("market_close").index)

    restricted_schedule = restricted_schedule.set_index('market_open')

    bars_df = bars_df.reindex(full_index)

    bars_df["trade_day"] = bars_df.index
    bars_df["trade_day"] = bars_df["trade_day"].apply(lambda x: x.replace(hour=0, minute=0, second=0))

    groups = bars_df.groupby("trade_day")
    interpolated_data = []
    restricted_schedule.index = restricted_schedule.index.map(lambda x: x.timestamp())
    restricted_schedule = restricted_schedule.to_dict()
    for day, group_df in tqdm(groups,
                              desc=f"Interpolating bars from {bars_df.index.min()} to {bars_df.index.max()} for assets {bars_df['unique_identifier'].dropna().unique()}"):
        schedule = calendar_instance.schedule(start_date=day, end_date=day)
        if schedule.shape[0] == 0:
            continue
        try:
            trade_starts = schedule["market_open"].iloc[0]
            trade_ends = schedule["market_close"].iloc[0]
        except Exception as e:
            raise e

        group_df = group_df[group_df.index >= schedule["market_open"].iloc[0]]
        group_df = group_df[group_df.index <= schedule["market_close"].iloc[0]]

        if group_df.dropna().shape[0] == 0:
            continue

        if trade_starts < day:
            trade_starts = day
        next_day = day + datetime.timedelta(days=1)
        if trade_ends >= next_day:
            trade_ends = next_day - datetime.timedelta(minutes=1)

        if interpolation_rule == "None":
            tmp_df = rebase_withoutnan_fill(group_df, trade_starts=trade_starts, trade_ends=trade_ends)
        elif interpolation_rule == "ffill":
            tmp_df = rebase_with_forward_fill(
                group_df,
                trade_starts=trade_starts,
                last_observation=last_observation,
                trade_ends=trade_ends
            )
            if last_observation is None:
                tmp_df = tmp_df.bfill()
        else:
            raise Exception

        if len(tmp_df):
            last_observation = tmp_df.iloc[[-1]]
        interpolated_data.append(tmp_df)

    if len(interpolated_data) == 0:
        return pd.DataFrame()

    interpolated_data = pd.concat(interpolated_data, axis=0)
    interpolated_data["trade_day"] = interpolated_data.index
    interpolated_data["trade_day"] = interpolated_data["trade_day"].apply(
        lambda x: x.replace(hour=0, minute=0, second=0)
    )

    return interpolated_data


class InterpolatedPrices(TimeSerie):
    """
    Handles interpolated prices for assets.
    """
    OFFSET_START = datetime.datetime(2017, 7, 20).replace(tzinfo=pytz.utc)
    @TimeSerie._post_init_routines()
    def __init__(
            self,
            execution_venue_symbol:str,
            asset_universe:AssetUniverse,
            bar_frequency_id: str,
            intraday_bar_interpolation_rule: str,
            data_mode: Literal["live", "backtest"],
            upsample_frequency_id: Optional[str] = None,
            asset_filter:Optional[dict] = None,
            local_kwargs_to_ignore: List[str] = ["asset_universe"],
            *args,
            **kwargs
    ):

        """
        Initializes the InterpolatedPrices object.
        """
        assert "d" in bar_frequency_id or "m" in bar_frequency_id, f"bar_frequency_id={bar_frequency_id} should be 'd for days' or 'm for min'"
        self.data_mode=data_mode
        self.asset_universe = asset_universe
        self.interpolator = UpsampleAndInterpolation(
            bar_frequency_id=bar_frequency_id,
            upsample_frequency_id=upsample_frequency_id,
            intraday_bar_interpolation_rule=intraday_bar_interpolation_rule
        )

        self.intraday_bar_interpolation_rule = intraday_bar_interpolation_rule
        self.bar_frequency_id = bar_frequency_id
        self.upsample_frequency_id = upsample_frequency_id



        self.execution_venue_symbol = execution_venue_symbol
        price_source = get_prices_source(
            asset_filters=asset_universe.asset_filters,
            bar_frequency_id=bar_frequency_id,
            source=self.execution_venue_symbol, is_train=True
        )
        self.bars_ts = (
            WrapperTimeSerie(time_series_dict=price_source) if isinstance(price_source, dict) else price_source
        )

        self.set_asset_details()
        super().__init__(*args, **kwargs)

    def get_html_description(self) -> Union[str, None]:
        description = f"""<p>{self.data_mode} Time Serie Instance of {self.__class__.__name__} updating table {self.remote_table_hashed_name} for for <b>train</b> prices in <b>{self.execution_venue_symbol}</b> for backtesting</p>"""
        return description

    def set_asset_details(self):
        asset_list = self.asset_universe.asset_list
        if len(asset_list) == 0:
            raise Exception(f"Asset universe has no assets {self.asset_universe}")

        equal_venue = [a.execution_venue.symbol == self.execution_venue_symbol for a in asset_list]
        assert all(equal_venue), "InterpolatedPrices should have only one type of execution_venue"

        self.asset_calendar_map = {a.unique_identifier: a.calendar for a in asset_list}
        self.asset_list = asset_list

    def _get_required_cores(self, last_observation_map) -> int:
        """
        Determines the required number of cores for processing.
        """
        if len(last_observation_map) == 0:
            required = 1
        else:
            required = min(len(last_observation_map), 20)

        return required

    @property
    def human_readable(self) -> str:
        """
        Returns a human-readable string representation of the object.
        """
        name = f"{self.__class__.__name__} Upsample: {self.upsample_frequency_id} in {self.execution_venue_symbol}"
        return name

    def run_after_post_init_routines(self):
        """
        Use post init routines to configure the time series
        """
        if BACKEND_DETACHED():
            return None
        
        try:
            if self.metadata is None:
                return None
        except LocalTimeSeriesDoesNotExist:
            return None  # first update

        if self.execution_venue_symbol!= ASSET_ORM_CONSTANTS.MAIN_SEQUENCE_PORTFOLIOS_EV:

            if not self.metadata.protect_from_deletion:
                self.local_persist_manager.protect_from_deletion()

            if not self.metadata.open_for_everyone:
                self.local_persist_manager.open_for_everyone()

    def _transform_raw_data_to_upsampled_df(
            self,
            raw_data_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transforms raw data into an upsampled dataframe.
        """
        upsampled_df = []
        full_last_observation = self.get_last_observation()
        last_observation_map = {}

        for unique_identifier in raw_data_df["unique_identifier"].unique():
            if full_last_observation is None:
                last_observation_map[unique_identifier] = None
                continue

            if unique_identifier in full_last_observation.index.get_level_values("unique_identifier").to_list():
                last_obs = full_last_observation.loc[(slice(None), unique_identifier), :].reset_index(
                    ["unique_identifier"], drop=True
                )
                last_obs.index.name = None
                if "open_time" in last_obs.columns:
                    last_obs["open_time"] = pd.to_datetime(last_obs["open_time"], utc=True)
                last_observation_map[unique_identifier] = last_obs
            else:
                last_observation_map[unique_identifier] = None

        def multiproc_upsample(calendar, tmp_df, unique_identifier, last_observation, interpolator_kwargs):
            interpolator = UpsampleAndInterpolation(**interpolator_kwargs)
            df = interpolator.get_interpolated_upsampled_bars(
                calendar=calendar,
                tmp_df=tmp_df,
                last_observation=last_observation_map[unique_identifier]
            )
            df["unique_identifier"] = unique_identifier
            return df

        required_cores = self._get_required_cores(last_observation_map=last_observation_map)
        required_cores = 1
        if required_cores == 1:
            # Single-core processing
            for unique_identifier, df in raw_data_df.groupby("unique_identifier"):
                if df.shape[0] > 0:
                    df = self.interpolator.get_interpolated_upsampled_bars(
                        calendar=self.asset_calendar_map[unique_identifier],
                        tmp_df=df,
                        last_observation=last_observation_map[unique_identifier],
                    )
                    df["unique_identifier"] = unique_identifier
                    upsampled_df.append(df)
        else:
            upsampled_df = Parallel(n_jobs=required_cores)(
                delayed(multiproc_upsample)(
                    calendar=self.asset_calendar_map[unique_identifier],
                    tmp_df=tmp_df,
                    unique_identifier=unique_identifier,
                    last_observation=last_observation_map[unique_identifier],
                    interpolator_kwargs=dict(
                        bar_frequency_id=self.bar_frequency_id,
                        upsample_frequency_id=self.upsample_frequency_id,
                        intraday_bar_interpolation_rule=self.intraday_bar_interpolation_rule,
                    )
                )
                for unique_identifier, tmp_df in raw_data_df.groupby("unique_identifier") if tmp_df.shape[0] > 0
            )

        upsampled_df = [d for d in upsampled_df if len(d) > 0]  # Remove empty dataframes
        if len(upsampled_df) == 0:
            return pd.DataFrame()

        max_value_per_asset = {d.index.max():d.unique_identifier.iloc[0] for d in upsampled_df}
        min_max=min(max_value_per_asset.keys())
        self.logger.info(f"min_max {max_value_per_asset[min_max]} {min_max} max_max {max(max_value_per_asset.keys())}")
        upsampled_df = pd.concat(upsampled_df, axis=0)
        # upsampled_df = upsampled_df[upsampled_df.index <= min_max]
        upsampled_df.volume = upsampled_df.volume.fillna(0)


        upsampled_df.index.name = "time_index"
        upsampled_df = upsampled_df.set_index("unique_identifier", append=True)
        upsampled_df = upsampled_df.sort_index(level=0)

        if upsampled_df.shape[0] == 0:
            upsampled_df = pd.DataFrame()

        return upsampled_df

    def get_upsampled_data(
            self,
            update_statistics: DataUpdates,

    ) -> pd.DataFrame:
        """
        Main method to get upsampled data for prices.
        """
        from mainsequence.virtualfundbuilder.time_series import PortfolioStrategy

        unique_identifier_range_map = {
            unique_identifier: {
                "start_date": last_update,
                "start_date_operand": '>=',
            } for unique_identifier, last_update in update_statistics.update_statistics.items()
        }

        if isinstance(self.bars_ts, WrapperTimeSerie):
            raw_data_df = []
            for k, ts in self.bars_ts.related_time_series.items():
                if isinstance(ts, PortfolioStrategy) == False:
                    raise NotImplementedError
                tmp_data = unique_identifier_range_map[asset_id_map[k]]

                tmp_df = ts.get_df_between_dates(
                    start_date=tmp_data["start_date"],
                    great_or_equal=(tmp_data["start_date_operand"] == ">="),

                )

                tmp_df["unique_identifier"] = asset_id_map[k]
                tmp_df = tmp_df.set_index(["unique_identifier"], append=True)

                tmp_df = tmp_df.rename(columns={"portfolio_minus_fees": "close"})
                tmp_df["vwap"] = tmp_df["close"]
                tmp_df["open"] = tmp_df["close"]
                tmp_df["high"] = tmp_df["close"]
                tmp_df["low"] = tmp_df["close"]
                tmp_df["volume"] = 1.0
                tmp_df["first_trade_time"] = tmp_df.index.get_level_values("time_index")
                tmp_df["last_trade_time"] = tmp_df.index.get_level_values("time_index")
                tmp_df["open_time"] = tmp_df.index.get_level_values("time_index") - string_freq_to_time_delta(
                    ts.assets_configuration.prices_configuration.bar_frequency_id)

                raw_data_df.append(tmp_df[["vwap", "open", "close", "high", "low", "volume", "first_trade_time",
                                           "last_trade_time", "open_time"]])
            raw_data_df = pd.concat(raw_data_df, axis=0)

        else:
            raw_data_df = self.bars_ts.filter_by_assets_ranges(unique_identifier_range_map=unique_identifier_range_map)

        if raw_data_df.shape[0] == 0:
            self.logger.info("New new data to interpolate")
            return pd.DataFrame()

        upsampled_df = self._transform_raw_data_to_upsampled_df(
            raw_data_df.reset_index(["unique_identifier"]),
        )
        return upsampled_df

    def update(
            self,
            update_statistics: DataUpdates
    ) -> pd.DataFrame:
        """
        Updates the series from the source based on the latest value.
        """


        prices = self.get_upsampled_data(
            update_statistics=update_statistics,
        )

        if update_statistics.is_empty()==False:
            TARGET_COLS=['open', 'close', 'high', 'low', 'volume', 'vwap', 'open_time']
            assert prices[[c for c in prices.columns if c in TARGET_COLS]].isnull().sum().sum() == 0

        prices = update_statistics.filter_df_by_latest_value(prices)

        duplicates_exist = prices.reset_index().duplicated(subset=["time_index","unique_identifier"]).any()
        if duplicates_exist:
            raise Exception()

        return prices


class InterpolatedPricesLive():
    """
    Handles interpolated prices for assets.
    """

    @TimeSerie._post_init_routines()
    def __init__(
            self,
            asset_list: ModelList,
            bar_frequency_id: str,
            intraday_bar_interpolation_rule: str,
            upsample_frequency_id: Union[str, None] = None,
            local_kwargs_to_ignore: List[str] = ["asset_list"],
            *args,
            **kwargs
    ):
        """
        Initializes the InterpolatedPricesLive object.
        """
        super().__init__(
            asset_list=asset_list,
            bar_frequency_id=bar_frequency_id,
            intraday_bar_interpolation_rule=intraday_bar_interpolation_rule,
            upsample_frequency_id=upsample_frequency_id,
            *args,
            **kwargs
        )

        rename_price_source_map = {
            "alpaca_testnet": "alpaca",
            "binance_testnet": "binance",
        }

        execution_venue = asset_list[0].execution_venue_symbol

        if execution_venue in rename_price_source_map:
            execution_venue = rename_price_source_map[execution_venue]

        # Bars time series should not be necessary for live execution
        self.table_name = f"{execution_venue}_{asset_list[0].asset_type}_trades"
        self.view_name = f"{execution_venue}_{asset_list[0].asset_type}_bars_1m"

    def get_earliest_value_for_initial_update(self) -> datetime.datetime:
        """
        Get the earliest value for the initial update.
        """
        return datetime.datetime.utcnow().replace(
            tzinfo=pytz.utc, minute=0, second=0, microsecond=0
        ) - datetime.timedelta(hours=24)

    def _get_prices_directly_from_db(self, after_date: datetime.datetime) -> pd.DataFrame:
        """
        Retrieves prices directly from the database after a specified date.
        """
        import psycopg2

        symbols = ",".join([f"'{s}'" for s in self.asset_symbols_filter])
        end_request = datetime.datetime.now(pytz.utc).replace(second=0, microsecond=0)

        QUERY_INSERTION_HEALTH = f"SELECT * FROM {self.table_name} WHERE insertion_time >'{after_date}'"
        QUERY = f"""
            SELECT * FROM {self.view_name}
            WHERE bucket >= '{after_date}'
            AND bucket < '{end_request}' AND symbol IN ({symbols})
        """

        connection_uri=self.data_source.get_connection_uri()

        try:
            with psycopg2.connect(connection_config['connection_details']) as conn:
                with conn.cursor() as cursor:
                    cursor.execute(QUERY_INSERTION_HEALTH)
                    result = cursor.fetchone()
                    if result is None:
                        raise Exception(f"No new trades data in DB found between {after_date} and {end_request}")

                    cursor.execute(QUERY)
                    result = cursor.fetchall()
                    column_names = [desc[0] for desc in cursor.description]

        except psycopg2.OperationalError as e:
            raise e  # we  must raise to avoid interpolation
        except Exception as e:
            raise e

        bars_df = pd.DataFrame(data=result, columns=column_names)
        bars_df = bars_df.rename(columns={"symbol": "key"})
        bars_df.set_index("bucket", inplace=True)
        bars_df["open_time"] = bars_df.index
        bars_df.index = bars_df.index + datetime.timedelta(minutes=1)
        bars_df = bars_df.sort_index()

        cols = ['key', 'low', 'high', 'open', 'vwap', 'close', 'volume', 'last_trade_time', 'first_trade_time',
                'open_time']
        bars_df.index.name = "time"
        bars_df = bars_df[cols]

        for col in ["first_trade_time", "last_trade_time", "open_time"]:
            bars_df[col] = bars_df[col].astype(np.int64)

        bars_df = bars_df.rename(columns={"key": "unique_identifier"})

        return bars_df

    def update(
            self,
            update_statistics:DataUpdates
    ) -> pd.DataFrame:
        """
        Updates the series from the source based on the latest value.
        """
        #
        if self.data_source.data_type!=CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB:
            self.logger.warning(f"This time serie cant be updated with a data source of type {self.data_source.data_type}")
            return pd.DataFrame()
        # If no latest value, start updating from the most recent 24 hours
        if latest_value is None:
            latest_value = self.get_earliest_value_for_initial_update()

        raw_data_df = self._get_prices_directly_from_db(after_date=latest_value)
        if raw_data_df.shape[0] == 0:
            raise Exception("No Prices in the DB")

        if "days" in self.bar_frequency_id:
            raise NotImplementedError

        prices = self._transform_raw_data_to_upsampled_df(raw_data_df, latest_value)

        if latest_value is not None:
            assert prices.isnull().sum().sum() == 0

        if len(prices) and latest_value is not None:
            prices = prices[prices.index.get_level_values(level="time_index") > latest_value]

        return prices




