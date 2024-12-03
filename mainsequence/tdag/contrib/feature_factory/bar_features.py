import os
import pandas as pd
import numpy as np
import datetime
import copy
from tqdm import tqdm
from typing import Union
from mainsequence.tdag.config import ogm
import hashlib
import json
import pickle

class FeatureBase:

    def __init__(self, rolling_window: int, time_serie_frequency: datetime.timedelta, *args, **kwargs):
        self.rolling_window = rolling_window
        self.time_serie_frequency = time_serie_frequency
        self.set_latest_observation(pd.DataFrame)
        feat_hash=dict(rolling_window=rolling_window,time_serie_frequency=str(time_serie_frequency))
        feat_hash.update(kwargs)
        dhash = hashlib.md5()
        encoded=json.dumps(feat_hash, sort_keys=True).encode()
        dhash.update(encoded)
        self.feat_hash=dhash.hexdigest()

    def set_latest_observation(self,latest_observation:pd.DataFrame):
        self.latest_observation=latest_observation

    @property
    def feature_name(self):
        return NotImplementedError

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        seconds_needed = self.rolling_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    @property
    def num_threads(self):
        return 1

    @property
    def feature_temp_folder(self):
        folder=ogm.temp_folder+f"/feature_factory/{self.feat_hash}"
        return folder

    def calculate_feature(self, data_df: pd.DataFrame,last_observation) -> pd.DataFrame:
        raise NotImplementedError

    def __repr__(self):
        return self.feature_name



def get_rolling_moving_average(old_mean:float, new_sample:float,
                               last_sample:float,rolling_window:int):
    """
    Calculates Rolling Moving avarega
    :param old_mean:
    :param new_sample:
    :param last_sample:
    :param rolling_windos:
    :return:
    """
    if np.isnan(new_sample).any():
        raise Exception
    return old_mean + (new_sample - last_sample) / rolling_window

# Volume

class SignalQuality(FeatureBase):
    def __init__(self, target_column: str="volume", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert target_column in ["volume"]
        self.target_column = target_column

    @property
    def feature_name(self):
        return self.__class__.__name__ + str(self.rolling_window)

    @property
    def required_columns(self):
        return ["volume"]

    def calculate_feature(self, data_df,last_observation):
        
        x = data_df[self.target_column].applymap(lambda x: 1 if x>0 else 0).rolling(self.rolling_window).sum()/self.rolling_window

        x = pd.concat([x], keys=[self.feature_name], axis=1)
        x.columns.names = ["feature_name", "asset_id"]

        return x
        
        
        
class VolumeRatio(FeatureBase):

    def __init__(self, target_column: str, numerator_window: int, *args, **kwargs):
        """
        calcualtes teh rolling mean
        :param target_column:
        :param denominator_window:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        assert target_column in ["trades", "volume", "dollar"]
        self.target_column = target_column
        self.numerator_window = numerator_window
        assert numerator_window < self.rolling_window

    @property
    def feature_name(self):
        return self.__class__.__name__ + str(self.numerator_window) + "_over_" + str(
            self.rolling_window) + self.target_column

    @property
    def required_columns(self):
        return ["close","volume"]
    
    def calculate_feature(self, data_df,last_observation):
        if self.target_column == "dollar":
            if data_df.close.shape[1]>1:
                new_data=data_df.close * data_df.volume
                new_data = pd.concat([new_data], keys=[self.target_column], axis=1)
                data_df=new_data
            else:
                data_df["dollar"] = data_df.close * data_df.volume

        x = data_df[self.target_column].rolling(self.numerator_window).mean().divide(
            data_df[self.target_column].rolling(self.rolling_window).mean(), axis=0)

        if isinstance(x, pd.Series):
            x = x.to_frame(self.feature_name)
        else:
            x = pd.concat([x], keys=[self.feature_name], axis=1)
            x.columns.names = ["feature_name", "asset_id"]

        x = x.fillna(method="ffill").fillna(1)  # if denominator is 0 will through  NA
        return x






# Location

class ReturnFromMovingAverage(FeatureBase):

    def __init__(self, numerator_column: str, denominator_column: str,
                 numerator_smooth:int=None,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column
        self.numerator_smooth=numerator_smooth

    @property
    def required_columns(self):
        return [self.numerator_column,self.denominator_column]
    @property
    def feature_name(self):

        name=self.__class__.__name__ + str(
            self.rolling_window) + self.numerator_column + "_" + self.denominator_column

        if self.numerator_smooth is not None:
            name=name+f"smooth_{self.numerator_smooth}"

        return name

    def calculate_feature(self, data_df,last_observation):

        x=data_df[self.numerator_column]
        if self.numerator_smooth is not None:
            x=x.rolling(self.numerator_smooth).mean()
        x = x.divide(data_df[self.denominator_column].rolling(self.rolling_window).mean(),
                                                  axis=0) - 1

        x = pd.concat([x], keys=[self.feature_name], axis=1)
        x.columns.names = ["feature_name", "asset_id"]
        return x

class SimpleReturn(FeatureBase):
    def __init__(self, lag:int,return_window:int,
                 target_column:str,
                 *args, **kwargs):
        """

        Parameters
        ----------
        lag :  A positive number represents laggede returns, for example if data frequency is minute , lag is 60
        and reuturn window is 60 tthen we will get the returns t-60/(t-60-60)
        return_window :
        target_column :
        args :
        kwargs :
        """
        super().__init__(*args, **kwargs)

        self.lag = lag
        self.target_column=target_column
        self.return_window=return_window

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        total_window = self.return_window + self.lag
        seconds_needed = total_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)
    @property
    def feature_name(self):
        return f"{self.__class__.__name__}_{self.target_column}rw_{self.return_window}_l{self.lag}"

    @property
    def required_columns(self):
        return [self.target_column]

    def calculate_feature(self, data_df,feat_name=None):
        x=data_df[self.target_column].shift(self.lag).divide(data_df[self.target_column].shift(self.lag+self.return_window)) -1

        feat_name = self.feature_name if feat_name is None else feat_name
        if isinstance(x, pd.Series):
            x = x.to_frame(feat_name)
        else:
            
            x = pd.concat([x], keys=[feat_name], axis=1)
            x.columns.names = ["feature_name", "asset_id"]
        
        
        return x
# momentum Features


class MACD_hist(FeatureBase):
    def __init__(self,fast_window:int,slow_window:int, target_column:str,lag:int,
                 *args, **kwargs):
        """

        Parameters
        ----------
        fast_window :
        slow_window :
        signal_window :
        target_column :
        lag :
        args :
        kwargs :
        """
        super().__init__(*args, **kwargs)

        self.fast_window = fast_window
        self.slow_window = slow_window
        self.target_column=target_column
        self.lag=lag

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        total_window = self.slow_window+self.lag+1
        seconds_needed = total_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    @property
    def feature_name(self):
        return f"macd_line_{self.target_column}_slow_{self.slow_window}_fast_{self.fast_window}"

    @property
    def required_columns(self):
        return [self.target_column]

    def calculate_feature(self, data_df,last_observation):


        fast_ma=data_df[self.target_column].rolling(self.fast_window).mean()
        slow_ma=data_df[self.target_column].rolling(self.slow_window).mean()
        macd_line=(slow_ma-fast_ma)/slow_ma

        macd_line = macd_line.shift(self.lag)
        macd_line = pd.concat([macd_line], keys=[self.feature_name], axis=1)
        macd_line.columns.names = ["feature_name", "asset_id"]
        

        
        return macd_line

class MedianROC(FeatureBase):
    def __init__(self,return_window: int,lag:int,
                 target_column:str,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature=SimpleReturn(lag=lag,return_window=return_window,
                                  target_column=target_column,*args,**kwargs)

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        total_window = self.rolling_window+1
        seconds_needed = total_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)+ self.feature.min_delta_needed

    @property
    def feature_name(self):
        return f"{self.feature.feature_name}_med_rol{self.rolling_window}"

    @property
    def required_columns(self):
        return self.feature.required_columns

    def calculate_feature(self, data_df,last_observation):

        x=self.feature.calculate_feature(data_df=data_df,last_observation=last_observation,
                                         feat_name=self.feature_name,
                                         ).rolling(self.rolling_window).median()

        return x

# Volatility Features

class DayCloseToOpenReturn(FeatureBase):
    """
    Calcualtes return between days from previous day close to actual day open
    """

    @property
    def feature_name(self):
        return self.__class__.__name__

    def calculate_feature(self, data_df,last_observation):

        data_df["day"] = [i.date() for i in data_df.index]
        day_group = data_df.groupby("day")
        new_df = []
        for counter, (day, df) in enumerate(tqdm(day_group, desc="buidling prev de return")):
            if counter == 0:
                prev_day_close = np.nan

            df["prev_day_close"] = prev_day_close
            df["today_open"] = df.open.iloc[0]
            prev_day_close = df.close.iloc[-1]

            new_df.append(df)
        new_df = pd.concat(new_df, axis=0)
        del data_df
        new_df = new_df["today_open"].divide(new_df["prev_day_close"].values, axis=0)
        new_df = new_df.rename(columns={"today_open": self.feature_name})
        return new_df


class VolatilityRatio(FeatureBase):

    def __init__(self, numerator_vol_kwargs: dict, numerator_vol_type: str,
                 denominator_vol_kwargs: dict, denominator_vol_type: str,
                 *args, **kwargs):
        import sys
        thismodule = sys.modules[__name__]
        super().__init__(*args, **kwargs)
        num_kwargs = copy.deepcopy(kwargs)
        num_kwargs.update(numerator_vol_kwargs)
        self.numerator_vol = getattr(thismodule, numerator_vol_type)(**num_kwargs)

        den_kwargs = copy.deepcopy(kwargs)
        den_kwargs.update(denominator_vol_kwargs)
        self.denominator_vol = getattr(thismodule, denominator_vol_type)(**den_kwargs)

    @property
    def feature_name(self):
        return self.__class__.__name__ + self.numerator_vol.feature_name + "/" + self.denominator_vol.feature_name

    def calculate_feature(self, data_df,last_observation):
        numerator = self.numerator_vol.calculate_feature(data_df=data_df.copy(),last_observation=last_observation)
        denominator = self.denominator_vol.calculate_feature(data_df=data_df.copy(),last_observation=last_observation)
        vol_ratio = numerator.divide(denominator.values, axis=0)
        vol_ratio = vol_ratio.rename(columns={numerator.feature_name: self.feature_name})
        vol_ratio = vol_ratio * (self.denominator_vol.rolling_window / self.numerator_vol.rolling_window)
        return vol_ratio


class RogersSatchelVol(FeatureBase):

    @property
    def feature_name(self):
        return self.__class__.__name__ + str(self.rolling_window)

    @property
    def required_columns(self):
        return ["close","high","open","low"]

    def calculate_feature(self, data_df,last_observation):


        vol = np.log(data_df.high.divide( data_df.close,axis=0)) * np.log(data_df.high.divide( data_df.open,axis=0)) + \
              np.log(data_df.low.divide( data_df.close,axis=0)) * np.log(data_df.low.divide(data_df.open,axis=0))

        vol = vol.rolling(self.rolling_window, min_periods=None).mean()
       
        if isinstance(vol, pd.Series):
            vol = vol.to_frame(self.feature_name)
        else:
            vol = pd.concat([vol], keys=[self.feature_name], axis=1)
            vol.columns.names = ["feature_name", "asset_id"]
        return vol

class RealizedVolatilityContinuous(FeatureBase):

    def __init__(self, target_column: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_column = target_column
    @property
    def required_columns(self):
        return [self.target_column]
    @property
    def feature_name(self):
        return self.__class__.__name__ + str(self.rolling_window) + self.target_column

    def calculate_feature(self, data_df,last_observation):
        vol = (np.log(data_df[self.target_column])).diff().rolling(self.rolling_window, min_periods=None).std()
        if isinstance(vol,pd.Series):
            vol = vol.to_frame(self.feature_name)
        else:
            vol=pd.concat([vol],keys=[self.feature_name],axis=1)
            vol.columns.names=["feature_name","asset_id"]
        return vol


# technical features

class RSI(FeatureBase):

    def __init__(self, target_column: str,
                 from_returns=False,return_deflactor=None,
                 smooth_window:int=None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_column = target_column
        self.from_returns=from_returns
        self.return_deflactor=return_deflactor if return_deflactor is not None else 1
        self.smooth_window=smooth_window
    @property
    def required_columns(self):
        return [self.target_column]

    @property
    def feature_name(self):
        name=self.__class__.__name__ + str(self.rolling_window) + self.target_column
        if self.from_returns == True:
            name=name+f"rd_{self.return_deflactor}"
        if self.smooth_window is not None:
            name=name+f"_smooth{self.smooth_window}"

        return name

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        seconds_needed = (self.rolling_window +1)* self.time_serie_frequency.total_seconds()
        if self.from_returns == True:
            seconds_needed=seconds_needed+self.time_serie_frequency.total_seconds()
        if self.smooth_window is not None:
            seconds_needed = seconds_needed + self.time_serie_frequency.total_seconds()*self.smooth_window

        return datetime.timedelta(seconds=seconds_needed)
    
    def calculate_feature(self, data_df,last_observation):


       
        data=data_df[self.target_column]
        if self.from_returns==True:
            data=(1+data/self.return_deflactor).cumprod()

        if self.smooth_window is not None:
            data=data.rolling(self.smooth_window).mean()

        diffs = data.diff(1)
        positive = (diffs > 0).astype(float)
        negatives = (diffs < 0).astype(float)
        pos_avg = positive.rolling(self.rolling_window).sum()
        neg_avg = negatives.rolling(self.rolling_window).sum()
        rsi = 100 * pos_avg / (pos_avg + neg_avg)
        
        
        rsi=rsi.ffill()
        rsi = pd.concat([rsi], keys=[self.feature_name], axis=1)
        
        rsi.columns.names = ["feature_name", "asset_id"]

        return rsi


class RollingDollarValueBars(FeatureBase):

    def __init__(self, window_to_accumulate: int, percent_of_window_to_accumulate: int,
                 only_volume: False, *args, **kwargs):

        self.window_to_accumulate = window_to_accumulate
        self.percent_of_window_to_accumulate = percent_of_window_to_accumulate
        self.only_volume = only_volume
        super().__init__(*args, **kwargs)

        assert self.window_to_accumulate < self.rolling_window


    @staticmethod
    def get_historical_sequences(df:pd.DataFrame, sequence_length:int,target_column:str):
        """
        Get data with historical sequences of desired length
        :param df:
        :return:
        """
        assert sum(["rows_to_seq" in c for c in df.columns])==1
        rows_to_seq_col=[c for c in df.columns if "rows_to_seq" in c ]
        open_time_col=[c for c in df.columns if "open_time" in c ]
        index_name=df.index.name if df.index.name is not None else "index"
        df=df.reset_index()
        df_index = df.index.values
        tmp_bar=df[[target_column]+rows_to_seq_col+[index_name]]
        columns_names=[]
        cumulative_index_backtrack=df_index*0
        for i in range(sequence_length):
            cumulative_index_backtrack=cumulative_index_backtrack+tmp_bar[rows_to_seq_col].values.ravel()
            previous_bar_index=df_index-cumulative_index_backtrack
            previous_bar_index=np.nan_to_num(previous_bar_index)
            tmp_bar=df[[target_column]+rows_to_seq_col+[index_name]+open_time_col].iloc[previous_bar_index]
            col_name=-(i+1)
            columns_names.append(col_name)
            df[col_name]=tmp_bar[target_column].values

        df=df.rename(columns={target_column:0})
        df=df.set_index(index_name)
        columns_names.append(0)
        columns_names.sort()
        df=df[columns_names].add_prefix(f"{target_column}")

        return df


    @property
    def min_delta_needed(self) -> datetime.timedelta:

        total_window = self.rolling_window + self.window_to_accumulate
        seconds_needed = total_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    @property
    def feature_name(self):
        name = self.__class__.__name__ + str(
            self.rolling_window) + f"_{self.window_to_accumulate}_{self.percent_of_window_to_accumulate}"
        if self.only_volume == True:
            name = name + "only_volume"
        return name

    def build_column_index(self, df):
        df_column_index = {}
        columns_list = df.columns.to_list()
        for col in df.columns:
            df_column_index[col] = columns_list.index(col)
        return df_column_index

    def calculate_feature(self, data_df,last_observation):
        # add rolling_helpers

        if self.only_volume == True:
            data_df["dollar_traded"] = data_df["volume"]

        else:
            data_df["dollar_traded"] = data_df["volume"] * data_df["close"]
        data_df["rolling_dollar_value_mean"] = data_df["dollar_traded"].rolling(self.window_to_accumulate).sum()
        data_df["average_rolling_dollar_value_mean"] = data_df["rolling_dollar_value_mean"].rolling(
            self.rolling_window).mean()

        # loop backwards
       
        original_index = data_df.index
        # DEBUG_DATA = data_df.copy()
        data_df.index.name = "time"
        data_df = data_df.reset_index()

        column_map = self.build_column_index(data_df)
        data_df = data_df.values

        def get_bar(row_id: int, data_df, column_map, percent_of_window_to_accumulate):
            last_obs = data_df[row_id, :]
            loop_back = True
            row_id=0 if np.isnan(last_obs[column_map["average_rolling_dollar_value_mean"]]) else row_id
            rolling_high = [last_obs[column_map["high"]]]
            rolling_low = [last_obs[column_map["low"]]]
            row_count = 0
            cum_dollar = last_obs[column_map["dollar_traded"]]
            cum_volume = last_obs[column_map["volume"]]
            while loop_back == True:

                row_count = row_count + 1
                if row_id - row_count < 0:
                    tmp_bar = {"open_time": np.nan, "time": np.nan,
                               "close": np.nan,
                               "rows_to_seq": np.nan,
                               "open": np.nan, "volume": np.nan, "high": np.nan,
                               "low": np.nan,
                               "dollar_traded": np.nan, "vwap": np.nan}
                    loop_back = False
                else:
                    row = data_df[row_id - row_count, :]
                    rolling_high.append(row[column_map["high"]])
                    rolling_low.append(row[column_map["low"]])

                    cum_dollar = cum_dollar + row[column_map["dollar_traded"]]
                    cum_volume = cum_volume + row[column_map["volume"]]

                    if cum_dollar >= last_obs[
                        column_map["average_rolling_dollar_value_mean"]] * percent_of_window_to_accumulate:
                        tmp_bar = {"open_time": row[column_map["time"]], "time": last_obs[column_map["time"]],
                                   "close": last_obs[column_map["close"]],
                                   "rows_to_seq": row_count,
                                   "open": row[column_map["open"]], "volume": cum_volume, "high": np.max(rolling_high),
                                   "low": np.max(rolling_low),
                                   "dollar_traded": cum_dollar, "vwap": cum_dollar / cum_volume}
                        loop_back = False
            return tmp_bar
        new_bars=[]
        for row_id in tqdm(reversed(range(data_df.shape[0])), desc="building dollar bars",
                 total=data_df.shape[0]):
            new_bars.append(get_bar(row_id,data_df, column_map, self.percent_of_window_to_accumulate) )


        new_bars.reverse()
        new_bars = pd.DataFrame(new_bars).set_index("time").dropna()
        new_bars = new_bars.reindex(original_index)

        # import matplotlib.pyplot as plt
        # (new_bars["dollar_traded"] / DEBUG_DATA["average_rolling_dollar_value_mean"]).plot()
        # plt.show()
        new_bars=new_bars.add_prefix(self.feature_name)
        return new_bars

# bar features
class Shadow(FeatureBase):
    def __init__(self, is_upper: bool, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.is_upper = is_upper

    @property
    def feature_name(self):
        return self.__class__.__name__ + "is_upper" + str(self.upper)

    def calculate_feature(self, data_df,last_observation):
        if self.is_upper == True:
            bar_limit = np.maximum(data_df.close.values, data_df.open.values)
            bar_limit = pd.DataFrame(index=data_df.index, columns=["bar_limit"], data=bar_limit)
            shadow = data_df.high / bar_limit - 1
        else:
            bar_limit = np.minimum(data_df.close.values, data_df.open.values)
            bar_limit = pd.DataFrame(index=data_df.index, columns=["bar_limit"], data=bar_limit)
            shadow = data_df.low / bar_limit - 1
        return shadow


class ReturnFromHighLows(FeatureBase):

    def __init__(self, buffer_window: str, numerator_column: str, denominator_column: str,
                 agg_fun: str, *args, **kwargs):
        """

        :param buffer_window: window to calculate the return from, sometimes this is needed to avoid capturing
        a reversal effect.
        :param args:
        :param kwargs:
        """
        assert numerator_column in ["close", "vwap"]
        assert denominator_column in ["high", "low"]
        assert agg_fun in ["max", "min", "mean"]
        super().__init__(*args, **kwargs)
        self.buffer_window = buffer_window
        self.numerator_column = numerator_column
        self.denominator_column = denominator_column
        self.agg_fun = agg_fun

    @property
    def required_columns(self):
        return [self.denominator_column,self.numerator_column]

    @property
    def feature_name(self):
        return self.__class__.__name__ + str(self.rolling_window) + "_buf" + str(
            self.buffer_window) + f"{self.numerator_column}_{self.denominator_column}_{self.agg_fun}"

    @property
    def min_delta_needed(self) -> datetime.timedelta:

        total_window = self.rolling_window + self.buffer_window + 1
        seconds_needed = total_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    def calculate_feature(self, data_df,last_observation):

        denominator = data_df[self.denominator_column].shift(self.buffer_window).rolling(self.rolling_window,
                                                                                         min_periods=self.rolling_window)
        if self.agg_fun == "max":
            denominator = denominator.max()
        elif self.agg_fun == "min":
            denominator = denominator.min()
        elif self.agg_fun == "mean":
            denominator = denominator.mean()
        else:
            raise NotImplementedError

        numerator = data_df[self.numerator_column]
        x = numerator.divide(denominator, axis=0)

        if isinstance(x, pd.Series):
            x = x.to_frame(self.feature_name)
        else:
            x = pd.concat([x], keys=[self.feature_name], axis=1)
            x.columns.names = ["feature_name", "asset_id"]
        return x


class VolumeHistogram(FeatureBase):

    def __init__(self, obs_step=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.obs_step = obs_step
        assert self.obs_step == 1

    @property
    def required_columns(self):
        return ["volume","vwap"]

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        lcm = np.lcm(self.rolling_window, self.obs_step)
        total_window = self.rolling_window + self.rolling_window % lcm
        seconds_needed = total_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    @property
    def num_threads(self):
        
        return int(os.getenv("VOLUME_HIST_CORES",1))

    @property
    def feature_name(self):
        return self.__class__.__name__ + str(self.rolling_window) + f"step_{self.obs_step}"


    def calculate_feature(self, data_df,last_observation):
        from joblib import Parallel, delayed
        import numexpr as ne
        if self.obs_step != 1:
            raise NotImplementedError
        if isinstance(data_df["vwap"], pd.Series):
            raise NotImplementedError
        data_df[data_df.columns[data_df.dtypes.astype(str) == "float64"]] = data_df[
            data_df.columns[data_df.dtypes.astype(str) == "float64"]].astype("float32")
        def build_vol_hist(vwap_df, obs_step,bins, return_vol_hist=False):
            try:

               
                data_row = vwap_df.iloc[-1]

                bins = bins[-1,:]
                binned = np.digitize(vwap_df["vwap"],bins,right=True)
                target_row_bin=binned[-1]
                
                vol_his = vwap_df.groupby(binned)[["volume"]].sum()
                vol_his=vol_his[vol_his.index<=target_row_bin].copy()
               
                vol_his["frequency"] = (vol_his['volume'] / data_row['rolling_sum_volume']).values
                vol_his["cum"] = (vol_his['volume'] / data_row["rolling_sum_volume"]).cumsum().values

                if return_vol_hist == True:
                    bin_idx = np.searchsorted(bins, vwap_df["vwap"].values)
                    vwap_df["frequency"] = vol_his["frequency"].values[bin_idx]
                    vwap_df["cum"] = vol_his["cum"].values[bin_idx]
                    return vwap_df, vol_his
                else:

                    result=vol_his.iloc[-1][["frequency", "cum"]].to_dict()
                    result["index"]=data_row.name

                    return result
            except Exception as e:
                print(e)
                print(vwap_df)
                raise e

        def create_ranges_numexpr(start, stop, N, endpoint=True):
            if endpoint == 1:
                divisor = N - 1
            else:
                divisor = N
            s0 = start[:, None]
            s1 = stop[:, None]
            r = np.arange(N)
            return ne.evaluate('((1.0/divisor) * (s1 - s0))*r + s0')
        def build_vol_hist_col(tmp_df:pd.DataFrame,obs_step:int,rolling_window:int,feature_name:str,
                               col:str):
            """
            Executes column batch
            Parameters
            ----------
            tmp_df : 

            Returns
            -------

            """
            tmp_df.columns = tmp_df.columns.droplevel(1)

            original_index = tmp_df.index
            index_name = "index" if original_index.name is None else original_index.name
            tmp_df = tmp_df.fillna(method="ffill")
            tmp_df = tmp_df.dropna()

            tmp_df["rolling_sum_volume"] = tmp_df["volume"].rolling(rolling_window, min_periods=None).sum()
            tmp_df["rolling_mean_volume"] = tmp_df["volume"].rolling(rolling_window, min_periods=None).mean()
            tmp_df["rolling_vwap_max"] = tmp_df["vwap"].rolling(rolling_window, min_periods=None).max()
            tmp_df["rolling_vwap_min"] = tmp_df["vwap"].rolling(rolling_window, min_periods=None).min()
            

            bins = create_ranges_numexpr(start=tmp_df["rolling_vwap_min"].values,
                                         stop=tmp_df["rolling_vwap_max"].values,
                                         N=20, endpoint=True)

            starts_points_range = [i for i in range(0, tmp_df.shape[0] - rolling_window + 1, )]
            if len(starts_points_range) == 0:
                return pd.DataFrame()

            if obs_step == 1:
                try:
                    all_dfs=[]
                    for i in tqdm(starts_points_range, desc=f" {col} building volume histogram"):
                        last_v=tmp_df.iloc[i:i + rolling_window]
                        if last_v.iloc[-1]["rolling_sum_volume"]>0:
                            val=build_vol_hist(last_v,
                                                      bins=bins[i:i + rolling_window, :],
                                                      obs_step=obs_step)
                            all_dfs.append(val)
                except Exception as e:
                    raise e
                print("histogram built, creating DF")
                all_dfs = pd.DataFrame(all_dfs)
                all_dfs = all_dfs.set_index("index")
                all_dfs.index.name = index_name
                if all_dfs.isnull().sum().sum() != 0:
                    nauls=all_dfs[all_dfs.isnull().sum(axis=1) > 0]
                    raise Exception("Nans in calculated feature")
                assert all_dfs.index[-1] == original_index[-1]
                all_dfs = all_dfs.reindex(original_index).add_prefix(feature_name)
                all_dfs = pd.concat([all_dfs], keys=[col], axis=1)
                all_dfs.columns = all_dfs.columns.swaplevel()
                all_dfs.columns.names = ["feature_name", "asset_id"]

            else:
                raise NotImplementedError

            return all_dfs

        
      


        all_cols=data_df.columns
        target_cols_idx=lambda target_col:all_cols[all_cols.get_level_values(1)==target_col]
        general_input=dict(obs_step=self.obs_step ,rolling_window=self.rolling_window,feature_name=self.feature_name)
        num_threads=self.num_threads
        joint_data=Parallel(n_jobs=num_threads)(delayed(build_vol_hist_col)(tmp_df=data_df[target_cols_idx(col)].dropna(),col=col,**general_input)
                    for col in data_df["vwap"].columns)


        joint_data=pd.concat(joint_data,axis=1)
        joint_data.columns.names = ["feature_name", "asset_id"]
        
        return joint_data


class VolumeProfiler(FeatureBase):
    def __init__(self, step_size:int,
                 resample_window_minutes:int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.step_size = step_size
        self.resample_window_minutes=resample_window_minutes
        os.makedirs(self.feature_temp_folder,exist_ok=True)

    @property
    def required_columns(self):
        return ["volume", "close"]

    @property
    def min_delta_needed(self) -> datetime.timedelta:

        total_window = self.rolling_window 
        seconds_needed = total_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    @property
    def num_threads(self):
        return int(os.getenv("VOLUME_HIST_CORES", 1))

    @property
    def feature_name(self):
        return self.__class__.__name__ + str(self.rolling_window) + f"st_{self.step_size}_rs{self.resample_window_minutes}"

    @property
    def non_relevant_features(self):
        return ["last_calc_obs"]



    def calculate_feature(self, data_df,last_observation=None):
        from joblib import Parallel, delayed
        from scipy import stats,  signal

        def prev_kde(file_path:str):
            if os.path.isfile(file_path)==False:
                return None
            else:
                with open(file_path, 'rb') as handle:
                    data = pickle.load(handle)
                return data


        def save_kde_data(file_path,top_peaks,cummulative_dist,xr,kdy,calc_time,max_close,min_close):
            data=dict(top_peaks=top_peaks,cummulative_dist=cummulative_dist,xr=xr,
                 kdy=kdy,calc_time=calc_time,max_close=max_close,min_close=min_close
                 )

            with open(file_path, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


        def build_volume_profile(data:pd.DataFrame(),rolling_window:int,step_size,resample_window:int,
                                 previous_kde,
                                 name,col:str,file_path):

            data.columns=[c[0] for c in data.columns]
            data["vol_price"] = data["volume"] * data["close"]
            starts_points_range = [i for i in range(0, data.shape[0] - rolling_window + 1, )]
            all_profiles = []
            init_obs=0
            if previous_kde is not None:
                time_lapse=(data.index[rolling_window]-previous_kde["calc_time"]).total_seconds()//60

                if time_lapse<rolling_window and time_lapse>0:
                    top_peaks ,cummulative_dist= previous_kde["top_peaks"] ,previous_kde["cummulative_dist"]
                    kdy, xr = previous_kde["kdy"], previous_kde["xr"]
                    max_close, min_close = previous_kde['max_close'],previous_kde['min_close']
                    init_obs=time_lapse

            for i in tqdm(starts_points_range, desc=f"  building volume profile {col}"):
                tmp_df = data.iloc[i:i + rolling_window]


                last_obs = tmp_df.iloc[-1].close
                last_calc_obs = init_obs % step_size
                if last_calc_obs == 0:
                    # KDE calc

                    data_dist = tmp_df.resample(f"{resample_window}min").sum()
                    data_dist["vwap"] = data_dist["vol_price"] / data_dist["volume"]
                    data_dist["volume"].fillna(0)
                    data_dist["vwap"] = data_dist["vwap"].ffill()
                    data_dist = data_dist.dropna()
                    try:
                        kde = stats.gaussian_kde(data_dist.vwap.values.ravel(), weights=data_dist.volume.values.ravel(),
    
                                                 )
                    except Exception as e:
                        print(e)
                        continue

                    min_close, max_close = data_dist.vwap.min(), data_dist.vwap.max()
                    xr = np.linspace(min_close, max_close, rolling_window)
                    kdy = kde(xr.ravel())

                    peaks, peak_props = signal.find_peaks(kdy)
                    pkx = xr[peaks]
                    pky = kdy[peaks]
                    top_peaks = [(pkx[i], pky[i]) for i in range(pky.shape[0])]

                    cummulative_dist = kdy.cumsum() / kdy.sum()
                    top_peaks.sort(key=lambda tup: tup[1])
                    save_kde_data(top_peaks=top_peaks,cummulative_dist=cummulative_dist,xr=xr,
                                  file_path=file_path,min_close=min_close,max_close=max_close,
                                  kdy=kdy,calc_time=tmp_df.index[-1])
                    # end KDE CALC

                if len(top_peaks) == 0:
                    ratio_from_peaks = np.ones(2)
                elif len(top_peaks) == 1:
                    ratio_from_peaks = np.array([top_peaks[0][0], top_peaks[0][0]]) / last_obs
                else:
                    ratio_from_peaks = np.array([c[0] for c in top_peaks[-2:]]) / last_obs

                if last_obs > max_close:
                    obs_cum = 1.0
                    obs_freq = kdy[-1]
                elif last_obs < min_close:
                    obs_cum = 0
                    obs_freq = kdy[0]
                else:
                    obs_cum = cummulative_dist[(xr > last_obs).argmax()]
                    obs_freq = kdy[(xr > last_obs).argmax()]

                results = dict(cum=obs_cum, freq=obs_freq,
                               peak_r0=ratio_from_peaks[0], peak_r1=ratio_from_peaks[1],
                               time_index=tmp_df.index[-1],
                               )
                init_obs=init_obs+1

                all_profiles.append(results)
            all_profiles=pd.DataFrame(all_profiles).set_index("time_index").add_prefix(name)
            all_profiles=pd.concat([all_profiles],keys=[col],axis=1)
            all_profiles.columns=all_profiles.columns.swaplevel()
            return all_profiles

        all_cols = data_df.columns
        target_cols_idx = lambda target_col: all_cols[all_cols.get_level_values(1) == target_col]
        general_input = dict(step_size=self.step_size, rolling_window=self.rolling_window,
                             resample_window=self.resample_window_minutes,

                             )
        num_threads = self.num_threads
        joint_data = Parallel(n_jobs=num_threads)(
            delayed(build_volume_profile)(data=data_df[target_cols_idx(col)].dropna(),col=col,
                                          previous_kde=prev_kde(f"{self.feature_temp_folder}/{col}.pickle"),
                                          name=self.feature_name,file_path=f"{self.feature_temp_folder}/{col}.pickle",
                                          **general_input)
            for col in data_df["close"].columns)

        joint_data = pd.concat(joint_data, axis=1)
        joint_data.columns.names = ["feature_name", "asset_id"]
        return joint_data


#Specific Features
class CumResRetAr1(FeatureBase):
    def __init__(self,residual_return_window:int,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_return_window = residual_return_window
    @property
    def required_columns(self):
        return ["close"]
    @property
    def x_required_columns(self):
        return [None]

    @property
    def feature_name(self):
        name =f"{self.__class__.__name__}_retw_{self.residual_return_window}_arw_{self.rolling_window} "
        return name

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        window = (self.rolling_window + self.residual_return_window)  # extra to ffill
        seconds_needed = window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    def calculate_feature(self, data_df, *args,**kwarg):
        beta_columns = np.unique([c[0] for c in data_df.columns if "beta" in c[0] and "t_beta" not in c[0]])
        if len(beta_columns) == 0:
            raise Exception("No betas in columns")

        #1 get return
        returns=np.log(data_df.close_asset_y)
        returns=returns-returns.shift(self.residual_return_window)
        betas=data_df[beta_columns[0]][returns.columns]

        benchmark_returns=np.log(data_df.close_asset_x["benchmark"])
        benchmark_returns=benchmark_returns-benchmark_returns.shift(self.residual_return_window)


        returns=returns

#Factor Features

class BetaFactorLoading(FeatureBase):
    """
    This feature makes a rolling regression of beta columns and finds the loading values for the betas on a cross sectional
    regression
    """

    def __init__(self,y_target_column:str,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.y_target_column=y_target_column

    @property
    def required_columns(self):
        return [self.y_target_column]
    
    @property
    def x_required_columns(self):
        return [None]
    
    @property
    def feature_name(self):
        name = self.__class__.__name__ +self.y_target_column+ str(self.rolling_window)
        return name

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        window = (self.rolling_window + 2)  # extra to ffill
        seconds_needed = window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)
    
    
    
    def calculate_feature(self, data_df, last_observation,    x_asset_weights:Union[None,pd.DataFrame]=None,):
        from mainsequence.tdag.time_series import build_rolling_regression_from_df
        beta_columns=np.unique([c[0] for c in data_df.columns if "beta" in c[0] and "t_beta" not in c[0]])
        if len(beta_columns)==0:
            raise Exception("No betas in columns")

        x_col =beta_columns
        y_col = self.y_target_column + "_asset_y"
        x = data_df[x_col]
        x.columns=[c[1] for c in x.columns]
        y = (np.log(data_df[y_col])).diff().fillna(0)
        x=x[y.columns]
        y=y.melt(ignore_index=False).set_index("asset_id",append=True)["value"].to_frame("return")
        x=x.melt(ignore_index=False).set_index("variable",append=True)["value"].to_frame("beta")
        x_index,x_columns=x.index,x.columns
        y_index, y_columns = y.index, y.columns
       
        effective_window=data_df[y_col].shape[1]*self.rolling_window
        all_params = build_rolling_regression_from_df(x=x.values, y=y.values, rolling_window=effective_window,
                                                      column_names=y_columns, threads=1)
        
        all_params = all_params.replace([-np.inf, np.inf], np.nan).fillna(method="ffill")
        all_params.index = y_index[effective_window - 1:]
        all_params=all_params.reset_index().pivot(index="time_index", columns="asset_id")
        
        def name_replace(x):
            if "beta" in x and "t_beta" not in x:
                return f"{self.feature_name}_f"""
            if "intercept" in x and "t_intercept" not in x:
                return f"{self.feature_name}_int"""
            if "rsquared" in x:
                return f"{self.feature_name}_r2"""
            if "t_beta" in x:
                return f"{self.feature_name}_{x.replace('t_beta','t_be')}"""
            if "t_intercept" in x:
                return f"{self.feature_name}_t_int"""
        
        all_params.columns=pd.MultiIndex.from_tuples([(name_replace(c[0]).lower(), c[2]) for c in all_params.columns])
        all_params=all_params.dropna()
        all_params.columns.names = ["feature_name", "asset_id"]

        return all_params
# Cross Features

class SmoothSpread(FeatureBase):
    def __init__(self, target_column: str,
                 *args, **kwargs):
        """


        :param target_column:
        :param beta_shift:
        :param alpha_size:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.target_column=target_column

    @property
    def feature_name(self):
        name = f"SmoothSpread_y_to_x_{self.target_column}_{self.rolling_window}"

        return name

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        seconds_needed = self.rolling_window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)
    def calculate_feature(self, data_df,last_observation):

        x_col = self.target_column + "_asset_x"
        y_col = self.target_column + "_asset_y"
        X=data_df[y_col]-data_df[x_col]
        X=X.rolling(self.rolling_window).mean()

        return X.to_frame(self.feature_name)

class SmoothSpreadChange(FeatureBase):
    def __init__(self, target_column: str,change_window:int,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_column = target_column
        self.change_window=change_window
        self.spread_feature=SmoothSpread(target_column=target_column,
                                         time_serie_frequency=kwargs["time_serie_frequency"],
                                         rolling_window=kwargs["rolling_window"])

    @property
    def feature_name(self):
        return f"SmSpreadChg_sm{self.rolling_window}_c_{self.change_window}_{self.target_column}"
    @property
    def min_delta_needed(self) -> datetime.timedelta:
        seconds_needed = (self.change_window + 1) * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed) + self.spread_feature.min_delta_needed
    def calculate_feature(self, data_df,last_observation):
        X = self.spread_feature.calculate_feature(data_df=data_df,last_observation=last_observation)
        X=X/X.shift(self.change_window)-1
        return X.to_frame(self.feature_name)


class RollingResidualAlphaBeta(FeatureBase):

    def __init__(self, target_column: str, beta_shift=None, alpha_size=None,
                 normalize_beta=False,upsample_regression_minutes:Union[int,None]=None,
                 *args, **kwargs):
        """

        :param target_column:
        :param beta_shift:
        :param alpha_size:
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.target_column = target_column
        self.beta_shift = beta_shift
        self.alpha_size=alpha_size
        self.normalize_beta=normalize_beta
        self.upsample_regression_minutes=upsample_regression_minutes
    @property
    def required_columns(self):
        return [self.target_column]

    @property
    def num_threads(self):
        return int(os.getenv("VOLUME_HIST_CORES", 1))
    @property
    def feature_name(self):
        name = "RollResAB" + str(self.rolling_window) + self.target_column
        if self.beta_shift is not None:
            name = name + f"_be_l{self.beta_shift}"
            raise NotImplementedError #change the naming for r squared
        if self.alpha_size is not None:
            name = name  +f"az_{self.alpha_size}"
        if self.upsample_regression_minutes is not None:
            name=name+f"up{self.upsample_regression_minutes}"
        return name

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        window = (self.rolling_window + 2)#extra to ffill
        if self.beta_shift is not None:
            window = window + self.beta_shift
        if self.alpha_size is not None:
            window=window+ self.alpha_size
        if self.upsample_regression_minutes is not None:
            window=window+ self.upsample_regression_minutes
        seconds_needed = window * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    def calculate_feature(self, data_df,last_observation,
                          x_asset_weights:Union[None,pd.DataFrame]=None,

                          ):

        from mainsequence.tdag.time_series import build_rolling_regression_from_df

        x_col = self.target_column + "_asset_x"
        y_col = self.target_column + "_asset_y"
        x = np.log(data_df[x_col])
        y = np.log(data_df[y_col])

        x_index, x_cols = x.index, x.columns
        y_index, y_cols = y.index, y.columns
        column_names = y.columns


        effective_rolling_window = self.rolling_window
        intercept_adjustment = 1
        original_y = y.copy()

        if self.upsample_regression_minutes >1:
            up_str = f"{self.upsample_regression_minutes} min"
            y = y.resample(up_str).last().fillna(method="ffill")
            x = x.resample(up_str).last().fillna(method="ffill")
            effective_rolling_window = self.rolling_window // self.upsample_regression_minutes
            intercept_adjustment = self.upsample_regression_minutes
            #get only the necesarry last observation window
            if last_observation is not None:
                last_index=last_observation.index.get_level_values(0).unique()[0]
                min_observation=last_index-datetime.timedelta(minutes=self.upsample_regression_minutes*effective_rolling_window)
                
                

        y = y.diff()
        x = x.diff()

        x_regression_index, y_regression_index = x.index, y.index
        x, y = x.values.ravel(), y.values

        threads=5 if self.rolling_window<=60*24*7 else 5
        all_params = build_rolling_regression_from_df(x=x, y=y, rolling_window=effective_rolling_window,
                                                      column_names=column_names,threads=threads)
        all_params=all_params.replace([-np.inf,np.inf],np.nan).ffill()
        all_params.index = y_regression_index[effective_rolling_window - 1:]
        all_params.intercept = all_params.intercept / intercept_adjustment
        all_params = all_params.reindex(y_index).ffill()
        all_params = all_params.iloc[self.rolling_window - 1:]

        x = np.log(data_df[x_col]).diff().loc[all_params.index]
        y = np.log(data_df[y_col]).diff().loc[all_params.index]



        if self.beta_shift is not None:
            all_params.beta = all_params.beta.shift(self.beta_shift)

        if self.alpha_size is not None:
            x = np.log(data_df[x_col])-np.log(data_df[x_col].shift(self.alpha_size))
            y =np.log(data_df[y_col])-np.log(data_df[y_col].shift(self.alpha_size))
            if isinstance(x, pd.DataFrame):
                x = x[x.columns[0]]

        alpha = y.subtract(all_params.intercept+all_params.beta.multiply(x[x_cols[0]], axis=0),axis=0)

        if self.normalize_beta == True:
            raise NotImplementedError

        alpha=pd.concat([alpha],keys=[f"{self.feature_name}_al"],axis=1)
        data = pd.concat([alpha,all_params], axis=1)
        data.columns=pd.MultiIndex.from_tuples([(c[0].replace("beta",f"{self.feature_name}_be")\
                                                 .replace("intercept",f"{self.feature_name}_int")\
                                                 .replace("rsquared",f"{self.feature_name}_r2").lower(),

                                                 c[1]) for c in data.columns])


        data = data.ffill()

        if last_observation is not None:#forwardfill with last observation
            lo_data = last_observation.reset_index().pivot(index="time_index", columns="asset_symbol",
                                                           values=data.columns.get_level_values(
                                                               0).unique())
            last_observation_time=lo_data.index[-1]
            data = data[data.index >= last_observation_time]
            data = pd.concat([lo_data, data], axis=0).ffill()
            data=data[data.index>last_observation_time]
            t_cols=data.columns.get_level_values(1).isin(last_observation.index.get_level_values(1))
            filt_a = data[data.columns[t_cols]]
            assert filt_a.isnull().sum().sum() == 0

        data.columns.names = ["feature_name", "asset_id"]
        #ffill nulls

        return data

class RealizedVariance(FeatureBase):
    def __init__(self, target_column,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_column = target_column


    @property
    def required_columns(self):
        return [self.target_column]

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        seconds_needed = (self.rolling_window +1) * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)

    @property
    def feature_name(self):
        return f"{self.__class__.__name__}_{self.target_column}_{self.rolling_window}"


    def calculate_feature(self, data_df,last_observation):
        y=data_df[self.target_column]
        y = np.log(y).diff()
        y_2=y**2
        y_t_minus_1=y.shift(1)
        y_t_plus_1=y.shift(-1)

        rv=y_2.rolling(self.rolling_window).sum()
        zhou_rv=rv+(y*y_t_minus_1).rolling(self.rolling_window).sum()+ \
                                                        (y * y_t_plus_1).rolling(self.rolling_window).sum()
        zhou_rv=max(rv/10,zhou_rv)
        if isinstance(zhou_rv, pd.Series):
            zhou_rv = zhou_rv.to_frame(self.feature_name)
        else:
            zhou_rv = pd.concat([zhou_rv], keys=[self.feature_name], axis=1)
            zhou_rv.columns.names = ["feature_name", "asset_id"]
        return zhou_rv

class AlphaNormalizedRV(FeatureBase):
    def __init__(self, target_column: int,
                 beta_rolling_window: int,
                 normalize_alfa_window: int = None,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta_rolling_window=beta_rolling_window
        self.target_column=target_column
        self.rv_feat = AlphaRV(target_column=target_column,beta_rolling_window=beta_rolling_window,
                               *args,**kwargs)
        self.normalize_alfa_window = normalize_alfa_window

    @property
    def required_columns(self):
        return [self.target_column]

    @property
    def min_delta_needed(self) -> datetime.timedelta:

        return self.rv_feat.min_delta_needed

    @property
    def feature_name(self):
        name = f"AnormRV{self.normalize_alfa_window}_{self.target_column}{self.rolling_window}_b{self.beta_rolling_window}"
        return name

    def calculate_feature(self, data_df,last_observation):
        x_col = self.target_column + "_asset_x"
        y_col = self.target_column + "_asset_y"
        x = data_df[x_col]
        y = data_df[y_col]
        x=x[x.columns[0]]
        _, _, beta, alpha_rv=self.rv_feat.calculate_feature(data_df=data_df,
                                                            last_observation=last_observation,
                                                            return_inputs=True)

      
        
        new_alpha = y.divide(y.shift(self.normalize_alfa_window)) - 1
        mkt_return = x / x.shift(self.normalize_alfa_window) - 1
        new_alpha = new_alpha - beta.multiply(mkt_return,axis=0)
        new_alpha=new_alpha/np.sqrt(alpha_rv)
        new_alpha=new_alpha.replace([np.inf, -np.inf], np.nan,)
        new_alpha=new_alpha.fillna(0)
        
        new_alpha = pd.concat([new_alpha], keys=[self.feature_name], axis=1)
        new_alpha.columns.names = ["feature_name", "asset_id"]
        return new_alpha

class AlphaRV(FeatureBase):
    def __init__(self, target_column:int,
                  beta_rolling_window:int,
                 include_inputs_var=False,
                 *args, **kwargs):
        """
        calcualtes RV of alpha by RV of its components
        Parameters
        ----------
        target_column :
        upsample_window :
        alpha_rolling_window :
        args :
        kwargs :
        """
        super().__init__(*args, **kwargs)
        self.target_column = target_column
        self.beta_rolling_window = beta_rolling_window
        self.include_inputs_var=include_inputs_var

    @property
    def required_columns(self):
        return [self.target_column]

    @property
    def min_delta_needed(self) -> datetime.timedelta:
        seconds_needed = (self.beta_rolling_window+2 + 1+self.rolling_window) * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed)
    @property
    def feature_name(self):
        name=f"AlphaRV_{self.target_column}{self.rolling_window}_b{self.beta_rolling_window}"
        return name

    def _calc_realized_variance(self,data_y,rolling_window:int):
        y_squared = data_y ** 2
        y_t_minus_1 = data_y.shift(1)
        y_t_plus_1 = data_y.shift(-1)

        rv = y_squared.rolling(rolling_window).sum()
        zhou_rv = rv + \
                  (data_y * y_t_minus_1).rolling(rolling_window).sum() + \
                  (data_y * y_t_plus_1).rolling(rolling_window).sum()
        zhou_rv = zhou_rv.where(zhou_rv > 0, rv)
        # shift always 1
        zhou_rv = zhou_rv.shift(1)
        return zhou_rv
    def calculate_feature(self, data_df,last_observation,return_inputs=False):
        x_col = self.target_column + "_asset_x"
        y_col = self.target_column + "_asset_y"



        x = (np.log(data_df[x_col])).diff()
        y = (np.log(data_df[y_col])).diff()

        beta_rolling_window=self.beta_rolling_window
        rolling_window=self.rolling_window

        if isinstance(x, pd.DataFrame):
            x = x[x.columns[0]]
        x_mean = x.rolling(beta_rolling_window, min_periods=beta_rolling_window).mean()
        y_mean = y.rolling(beta_rolling_window, min_periods=beta_rolling_window).mean()

        x_demean = x - x_mean
        y_demean = y - y_mean

        xy = y_demean.multiply(x_demean, axis=0).rolling(beta_rolling_window, min_periods=beta_rolling_window).sum()
        x_2 = x_demean.multiply(x_demean, axis=0).rolling(beta_rolling_window, min_periods=beta_rolling_window).sum()
        beta = xy.divide(x_2, axis=0).fillna(method="ffill")
        beta = beta.fillna(beta.mean())

        alpha = y.subtract(beta.multiply(x, axis=0), axis=0)

        alpha_rv=self._calc_realized_variance(data_y=alpha,rolling_window=self.rolling_window)
        if return_inputs == True:
            return x, y, beta, alpha_rv


        if isinstance(alpha_rv, pd.Series):
            zhou_rv = alpha_rv.to_frame(self.feature_name)
        else:

            if self.include_inputs_var == True:
                y_rv = self._calc_realized_variance(data_y=y, rolling_window=self.rolling_window)
                x_rv=self._calc_realized_variance(data_y=x, rolling_window=self.rolling_window)
                x_rv=beta.multiply(x_rv,axis=0)


                col_name = lambda x: self.feature_name + x
                zhou_rv = pd.concat([alpha_rv, y_rv, x_rv], keys=[col_name("_al"), col_name("_"), col_name("_b*mrk")],
                                 axis=1)
                zhou_rv.columns.names = ["feature_name", "asset_id"]

            else:
                zhou_rv = pd.concat([zhou_rv], keys=[self.feature_name], axis=1)
                zhou_rv.columns.names = ["feature_name", "asset_id"]




        return zhou_rv

class AlphaRealizedVolatility(FeatureBase):

    def __init__(self, target_column, alpha_rolling_window,
                 normalize_beta=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_column = target_column
        self.alpha_rolling_window = alpha_rolling_window
        self.alpha_beta_feature = RollingResidualAlphaBeta(target_column=target_column,
                                                           rolling_window=alpha_rolling_window,
                                                           normalize_beta=normalize_beta,
                                                           time_serie_frequency=kwargs["time_serie_frequency"]
                                                           )
    @property
    def required_columns(self):
        return [self.target_column]
    @property
    def min_delta_needed(self) -> datetime.timedelta:
        seconds_needed = (self.rolling_window + 1) * self.time_serie_frequency.total_seconds()
        return datetime.timedelta(seconds=seconds_needed) + self.alpha_beta_feature.min_delta_needed

    @property
    def feature_name(self):
        return self.alpha_beta_feature.feature_name + "A_Vol"+ f"_{str(self.rolling_window)}"

    def calculate_feature(self, data_df,last_observation,x_asset_weights=None):
        data_df = self.alpha_beta_feature.calculate_feature(data_df=data_df,last_observation=last_observation,
                                                            x_asset_weights=x_asset_weights)
        
        if isinstance(data_df.columns,pd.MultiIndex):
            feats= data_df.columns.get_level_values("feature_name").unique()
            feats=[c for c in feats if "_al" in c][0]
            alpha = data_df[feats]
        else:
            alpha = data_df[[c for c in data_df.columns if "_al" in c][0]]
        vol = alpha.rolling(self.rolling_window, min_periods=self.rolling_window).std()

        if isinstance(alpha, pd.Series):
            vol = vol.to_frame(self.feature_name)
        else:
            vol = pd.concat([vol], keys=[self.feature_name], axis=1)
            vol.columns.names = ["feature_name", "asset_id"]
        return vol
