import datetime

import pandas as pd

from mainsequence.virtualfundbuilder.enums import RebalanceFrequencyStrategyName, PriceTypeNames
from mainsequence.virtualfundbuilder.resource_factory.rebalance_factory import RebalanceStrategyBase, register_rebalance_class


@register_rebalance_class(register_in_agent=True)
class VolumeParticipation(RebalanceStrategyBase):
    """
    This rebalance strategy implies volume participation with no market impact.
    i.e. that the execution price will be vwap and it will never execute more than max_percent_volume_in_bar
    """

    def __init__(
        self,
        rebalance_start: str = "9:00",
        rebalance_end: str = "23:00",
        rebalance_frequency_strategy: RebalanceFrequencyStrategyName = RebalanceFrequencyStrategyName.DAILY,
        max_percent_volume_in_bar: float = 0.01,
        total_notional: float = 50000000,
        *args, **kwargs
    ):
        """
        Initializes the VolumeParticipation strategy.

        Attributes:
            rebalance_start (str): Start time for rebalancing, in "hh:mm" format.
            rebalance_end (str): End time for rebalancing, in "hh:mm" format.
            rebalance_frequency_strategy (RebalanceFrequencyStrategyName): Rebalance frequency.
            max_percent_volume_in_bar (float): Maximum percentage of volume to trade in a bar.
            total_notional (int): Initial notional invested in the strategy.
        """
        self.rebalance_start = rebalance_start
        self.rebalance_end = rebalance_end
        self.rebalance_frequency_strategy = rebalance_frequency_strategy
        self.max_percent_volume_in_bar = max_percent_volume_in_bar
        self.total_notional = total_notional

        super().__init__(*args, **kwargs)

    def apply_rebalance_logic(
            self,
            last_rebalance_weights: pd.DataFrame,
            start_date: datetime.datetime,
            end_date: datetime.datetime,
            signal_weights: pd.DataFrame,
            prices_df: pd.DataFrame,
            price_type: str,
    ) -> pd.DataFrame:
        raise NotImplementedError
        asset_list = list(signal_weights.columns)
        start_time, end_time = pd.Timestamp(self.rebalance_start).time(), pd.Timestamp(self.rebalance_end).time()

        rebalance_dates = self.calculate_rebalance_dates(
            start=start_date,
            end=end_date,
            rebalance_frequency_strategy=self.rebalance_frequency_strategy,
            calendar=self.calendar
        )

        signal_weights["day"] = signal_weights.index.floor("D")

        volume_df = prices_df.reset_index().pivot(index="time_index", columns="asset_symbol", values="volume").fillna(0)
        prices_df = prices_df.reset_index().pivot(index="time_index", columns="asset_symbol", values=price_type).fillna(
            0)

        rebalance_days = np.intersect1d(signal_weights["day"], rebalance_dates)

        rebalance_weights = signal_weights[signal_weights["day"].isin(rebalance_days)]
        rebalance_weights = rebalance_weights[rebalance_weights.index.time >= start_time]
        rebalance_weights = rebalance_weights[rebalance_weights.index.time <= end_time]

        rebalance_weights = rebalance_weights.set_index("day", append=True)

        # calculcate the maximal participation volumes at each rebalancing steps
        max_participation_volume = (volume_df * prices_df) * self.max_percent_volume_in_bar
        max_participation_volume["day"] = max_participation_volume.index.floor("D")
        max_participation_volume = max_participation_volume.set_index("day", append=True)
        max_participation_volume = max_participation_volume[
            max_participation_volume.index.isin(rebalance_weights.index)]

        rebalance_weights = pd.concat(
            objs=[rebalance_weights, max_participation_volume],
            axis=1,
            keys=["weights", "max_dollar_volume"]
        )

        if last_rebalance_weights is not None:
            # get last rebalance weights
            past_rebalance_weight = last_rebalance_weights["weights_current"].reset_index(level="time_index", drop=True)
        else:
            past_rebalance_weight = pd.Series(0, index=asset_list)

        # recursively group by day
        past_day_rebalance_weight = past_rebalance_weight
        new_rebalance_weights = []
        for day, day_df in tqdm(rebalance_weights.groupby("day"), desc="building volume participation"):
            if (day_df.weights.max() - day_df.weights.min()).sum() != 0.0:
                logger.warning("Signal weight in time period changes, using weights at rebalancing start")
                day_df.loc[:, "weights"] = day_df["weights"].iloc[0].to_numpy()

            # calculate changes in backtesting weights
            weights_diff = day_df.weights - past_day_rebalance_weight

            target_dollar_volume = np.abs(weights_diff) * self.total_notional
            cumulative_dollar_volume = day_df.max_dollar_volume.fillna(0).cumsum()
            weighted_volume_multiplier = (cumulative_dollar_volume / target_dollar_volume).replace([np.inf], np.nan)
            weighted_volume_multiplier = weighted_volume_multiplier.fillna(0).map(lambda x: min(1.0, x))

            new_rebalance_weights_day = past_day_rebalance_weight + weights_diff * weighted_volume_multiplier
            new_rebalance_weights.append(new_rebalance_weights_day)
            past_day_rebalance_weight = new_rebalance_weights_day.iloc[-1]

        if len(new_rebalance_weights) == 0:
            logger.info("No new rebalancing weights found - returning empty DataFrame")
            return pd.DataFrame()

        rebalance_weights = pd.concat(new_rebalance_weights, axis=0).reset_index("day", drop=True)
        rebalance_weights_index = rebalance_weights.index

        shifted_rebalance_weights = rebalance_weights.shift(1)
        shifted_rebalance_weights.iloc[0] = past_rebalance_weight
        rebalance_weights = pd.concat(
            objs=[
                rebalance_weights, shifted_rebalance_weights,
                prices_df, prices_df.shift(1),
                volume_df, volume_df.shift(1),
            ],
            keys=["weights_current", "weights_before", "price_current", "price_before", "volume_current",
                  "volume_before"],
            axis=1
        )

        # filter out values with no weights and set NaNs to 0 weight
        rebalance_weights = rebalance_weights.loc[rebalance_weights_index].fillna(0)

        logger.info(f"{len(rebalance_weights)} new rebalancing weights calculated")
        return rebalance_weights

@register_rebalance_class(register_in_agent=True)
class TimeWeighted(RebalanceStrategyBase):
    def __init__(
            self,
            rebalance_start: str = "9:00",
            rebalance_end: str = "23:00",
            rebalance_frequency_strategy: RebalanceFrequencyStrategyName = RebalanceFrequencyStrategyName.DAILY,
            *args, **kwargs
    ):
        """
        Initialize the time weighted rebalance strategy.

        Attributes:
            rebalance_start (str): Start time for rebalancing, in "hh:mm" format.
            rebalance_end (str): End time for rebalancing, in "hh:mm" format.
            rebalance_frequency_strategy (RebalanceFrequencyStrategyName): Rebalance frequency.
            max_percent_volume_in_bar (float): Maximum percentage of volume to trade in a bar.
            total_notional (int): Initial notional invested in the strategy.
        """
        self.rebalance_start = rebalance_start
        self.rebalance_end = rebalance_end
        self.rebalance_frequency_strategy = rebalance_frequency_strategy
        super().__init__(*args, **kwargs)

    def apply_rebalance_logic(
            self,
            last_rebalance_weights: pd.DataFrame,
            start_date: datetime.datetime,
            end_date: datetime.datetime,
            signal_weights: pd.DataFrame,
            prices_df: pd.DataFrame,
            price_type: PriceTypeNames,
    ) -> pd.DataFrame:
        """
        Rebalance weights are set at start_time of rebalancing

        Parameters
        ----------
        signal_weights
        rebalance_dates

        Returns
        -------
        """
        raise NotImplementedError
        # TODO change last_rebalance_weights, as the calculations are wrong if there was a past rebalance during the day (= during the current rebalance window)
        asset_list = list(signal_weights.columns)
        start_time, end_time = pd.Timestamp(self.rebalance_start).time(), pd.Timestamp(self.rebalance_end).time()

        get_time_seconds = lambda x: x.hour * 3600 + x.minute * 60 + x.second

        rebalance_dates = self.calculate_rebalance_dates(
            start=start_date,
            end=end_date,
            rebalance_frequency_strategy=self.rebalance_frequency_strategy,
            calendar=self.calendar
        )

        # get rebalance dates from signal weights
        signal_weights["day"] = signal_weights.index.floor("D")
        rebalance_days = np.intersect1d(signal_weights["day"], rebalance_dates)
        rebalance_weights = signal_weights[signal_weights["day"].isin(rebalance_days)]

        rebalance_weights = rebalance_weights[rebalance_weights.index.time >= start_time]
        rebalance_weights = rebalance_weights[rebalance_weights.index.time <= end_time]

        # get past rebalance dates to calculate rebalance steps
        past_rebalance_weights = rebalance_weights.groupby("day").first().shift()

        # if past backtesting weights exist, get last one; otherwise start at 0
        if last_rebalance_weights is not None:
            # get last rebalance weights
            past_rebalance_weight = last_rebalance_weights["weights_current"].reset_index(level="time_index", drop=True)
        else:
            past_rebalance_weight = pd.Series(0, index=asset_list)

        past_rebalance_weights.index += datetime.timedelta(seconds=get_time_seconds(start_time))
        past_rebalance_weights = past_rebalance_weights.reindex(rebalance_weights.index).ffill()

        # time-based multiplicator during rebalancing
        time_weight = (rebalance_weights.index - rebalance_weights["day"]).dt.total_seconds()
        time_weight = (time_weight - get_time_seconds(start_time)) / (
                get_time_seconds(end_time) - get_time_seconds(start_time))

        rebalance_weights = rebalance_weights.drop(columns=["day"])

        # calculate new backtesting weights as past_weights + time_weight * diff_weights
        diff_weights = rebalance_weights - past_rebalance_weights
        rebalance_weights = past_rebalance_weights + diff_weights.multiply(time_weight, axis=0)

        # when columns are not available weights and prices need to be set to 0
        prices_df = prices_df.reset_index().pivot(index="time_index", columns="asset_symbol",
                                                  values=price_type).ffill().fillna(0)
        valid_columns = rebalance_weights.columns[rebalance_weights.columns.isin(prices_df.columns)]
        if len(valid_columns) != rebalance_weights.shape[1]:
            rebalance_weights = rebalance_weights[valid_columns].copy()
            rebalance_weights = rebalance_weights.divide(rebalance_weights.sum(axis=1), axis=0)

        # make backtesting weights  nan when there is no price
        nan_mask = prices_df.loc[rebalance_weights.index].isna()
        rebalance_weights[nan_mask] = np.nan

        if len(rebalance_weights) == 0:
            logger.info("No new rebalancing weights found - returning empty DataFrame")
            return pd.DataFrame()

        # create backtesting weights before and after rebalancing
        shifted_rebalance_weights = rebalance_weights.shift(1)
        shifted_rebalance_weights.iloc[0] = past_rebalance_weight
        rebalance_weights = pd.concat(
            objs=[shifted_rebalance_weights, rebalance_weights, prices_df, prices_df.shift(1)],
            keys=["weights_before", "weights_current", "price_current", "price_before"],
            axis=1
        )

        logger.info(f"{len(rebalance_weights)} new rebalancing weights calculated")
        return rebalance_weights

@register_rebalance_class(register_in_agent=True)
class ImmediateSignal(RebalanceStrategyBase):
    """
    This rebalance strategy 'immediately' rebalances the weights. This is equivalent to just using the signal weights.
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the immediate rebalance strategy.
        """
        super().__init__(*args, **kwargs)

    def get_explanation(self):
        explanation=f"""<p> This rebalance strategy 'immediately' rebalances the weights. This is equivalent to just using the signal weights. </p>"""
        return  explanation

    def apply_rebalance_logic(
            self,
            last_rebalance_weights: pd.DataFrame,
            signal_weights: pd.DataFrame,
            prices_df,
            price_type: PriceTypeNames,
            *args, **kwargs
    ) -> pd.DataFrame:

        volume_df = prices_df.reset_index().pivot(index="time_index", columns=["unique_identifier"], values="volume")
        prices_df = prices_df.reset_index().pivot(index="time_index", columns=["unique_identifier"], values=price_type.value)

        if last_rebalance_weights is not None:
            #because this function provides backtest weights we can concatenate last observation
            volume_df = pd.concat([last_rebalance_weights.unstack()["volume_current"], volume_df], axis=0)
            prices_df = pd.concat([last_rebalance_weights.unstack()["price_current"], prices_df], axis=0)
            signal_weights = pd.concat([last_rebalance_weights.unstack()["weights_current"], signal_weights], axis=0)
        rebalance_weights  = pd.concat(
            objs=[
                signal_weights,
                signal_weights.shift(1),
                prices_df,
                prices_df.shift(1),
                volume_df,
                volume_df.shift(1)
            ],
            keys=["weights_current", "weights_before", "price_current", "price_before", "volume_current",
                  "volume_before"],
            axis=1
        )

        if last_rebalance_weights is not None:
            # align last rebalance weights
            rebalance_weights = rebalance_weights[
                rebalance_weights.index > last_rebalance_weights.index.get_level_values("time_index")[0]]


        return rebalance_weights