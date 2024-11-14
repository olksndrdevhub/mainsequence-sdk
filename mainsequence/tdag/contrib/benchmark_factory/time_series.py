

import pytz
from mainsequence.tdag.time_series import TimeSerie,  ModelList

from mainsequence.tdag.contrib import HistoricalCoinSupply
from mainsequence.vam_client import KafkaExecutionWeights
from mainsequence.tdag.time_series import Rebalancer
from mainsequence.tdag.time_series.utils import  DeflatedPrices
from mainsequence.tdag.time_series import (PortfolioWeightsTimeSerie, PortfolioBarsFromWeight,
                                           BarTimeSerieUpsampleFromPortfolio)

import pandas as pd
import datetime

import sys
from  tqdm import tqdm
from functools import wraps

import  time
def time_func(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print ('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap







class RebalancerPortfolio(PortfolioWeightsTimeSerie):
    """
    This should be used only as a base class to be inherited
    """
    PRICE_LAG_TOLERANCE_SECONDS = 55  # from observation last update cant be more than N minutes ago
    UPDATE_LOOK_BACK_TIME_MINUTES = 0  # clip price observation to N minutes ago to avoid time series that are not yet updated
    MAX_LAG_ON_CREATION=360
    #NOTE: Base Class as is abstract should not be decorated with post init routines
    def __init__(self, asset_list: ModelList, cash_asset: dict, cash_balance: float, frequency_id: str,
                 source: str, rebalance_rule_config: dict,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert isinstance(asset_list,ModelList)



        upsample_frequency_id=frequency_id
        intraday_bar_interpolation_rule="ffill"
        assert frequency_id =="1min"

        symbol_to_search_map={}
        for asset in asset_list:

            symbol_to_search_map[asset.unique_identifier]=asset.symbol_to_asset_map


        self.cash_asset = cash_asset
        symbol_to_search_map[cash_asset.unique_identifier]=cash_asset.symbol_to_asset_map
        self.symbol_to_search_map=symbol_to_search_map
      

        bars_type_config=self.bars_type_config
        self.bars_ts =DeflatedPrices(asset_list=asset_list,bar_frequency_id=frequency_id,
                                            intraday_bar_interpolation_rule=intraday_bar_interpolation_rule,
                                            source=source,
                                            upsample_frequency_id=upsample_frequency_id,**bars_type_config
                                            )
        self.rebalance_rule_config = rebalance_rule_config
        self.cash_balance = cash_balance

    def get_minimum_required_depth_for_update(self):
        """
        Controls the minimum depth that needs to be rebuil
        Returns
        -------

        """
        return 1

    def _assert_prices_are_updated(self, last_weights: pd.Series, observation_time:datetime.datetime,
                                   price_lag_tolerance):
        w = last_weights[last_weights.index.get_level_values(0) == "weights"].droplevel(0)
        w = w[w != 0]
        last_weights = last_weights[last_weights.index.get_level_values(1).isin(w.index.get_level_values("asset_id"))]
        assert last_weights.isnull().sum() == 0
        last_price_update = self.bars_ts.wrapped_latest_index_value
       
        for key, value in last_price_update.items():
            if key in w.index and key!=self.cash_asset.unique_identifier:
                if observation_time- value > datetime.timedelta(
                        seconds=price_lag_tolerance):
                    msg= f"{last_weights[last_weights.index.get_level_values('asset_id')==key].index[0][2]} is not updated on{observation_time} last value is {value}"
                    self.logger.error(msg)
                    # raise ValueError(msg)


    def _update_weights_from_source(self, latest_value, *args, **kwargs):


        # 1 get opens to match time index Todo: Assert Available prices  are on the same tim
        data_df = self.bars_ts.get_inflated_df_greater_than(target_value=latest_value,
                                                            columns=["close"],
                                                            great_or_equal=True)
       
        observation_time = datetime.datetime.now(pytz.UTC)
        
        if data_df.shape[0]==0:
            self.logger.warning("No new prices to build weights, returning empty DF")
            return pd.DataFrame()

    

        # pivot table
        closes =data_df.close

        # build rebalancer signal df
        rebalancer_signal_df = self._produce_rebalancer_signal_df(closes_df=closes,latest_value=latest_value, *args, **kwargs)
        rebalancer = Rebalancer(rules_config=self.rebalance_rule_config, target_index=rebalancer_signal_df.index)
        rebalance_flag = rebalancer.get_rebalance_flag(target_df=rebalancer_signal_df)

        weights_df,execution_signal = self._produce_weights_and_execution_signal(closes_df=closes, rebalance_flag=rebalance_flag,latest_value=latest_value,
                                           rebalancer_signal_df=rebalancer_signal_df)

        weights_df.loc[:, self.cash_asset.unique_identifier ]=(1.0-weights_df.sum(axis=1)).apply(lambda x: max(x,0))
        weights_df = weights_df.round(decimals=4)

        if latest_value is not None:
            weights_df = weights_df[weights_df.index > latest_value]
            execution_signal=execution_signal[execution_signal.index>latest_value]

        if weights_df.shape[0] > 0:
            data_df = data_df.loc[data_df.index.isin(weights_df.index)]
            weights_df=weights_df.fillna(0)

            data_df.columns = data_df.columns.set_levels(data_df.columns.levels[0].str.replace('close', 'prices'), level=0)
            COLS=["prices"]
            data_df=data_df[COLS]
            data_df["prices"]=data_df["prices"].fillna(method="ffill")
            for col in COLS:
                data_df[(col, self.cash_asset.unique_identifier)]=1.0 if col !="volume" else 0

            data_df.columns.names = ["feature", "asset_id"]
            #add extra key
            execution_signal = (weights_df * 0).add(execution_signal ,axis=0)

            weights_df = weights_df.fillna(0)
            execution_signal = execution_signal.fillna(0)

            weights_df = pd.concat([weights_df], axis=1, keys=["weights"])
            weights_df.columns.names = ["feature", "asset_id"]
            execution_signal = pd.concat([execution_signal], axis=1, keys=["execution"])
            execution_signal.columns.names = ["feature", "asset_id"]


            weights_df = pd.concat([weights_df, data_df,execution_signal], axis=1)
            

        else:
            weights_df = pd.DataFrame()

        return weights_df

    def _produce_rebalancer_signal_df(self, closes_df,latest_value, *args, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def _produce_weights_and_execution_signal(self, closes_df: pd.DataFrame,
                         rebalance_flag, rebalancer_signal_df: pd.DataFrame,
                         latest_value: datetime.datetime) -> [pd.DataFrame,pd.DataFrame]:
        """
        Note that weights should be efffective weights at observation time. Therefore anything calculated with closes, should be shifted at least 1 period ahead
        
        :param closes_df: 
        :param rebalance_flag: 
        :param rebalancer_signal_df: 
        :param latest_value: 
        :return: weights_df ,rebalance_flag
        
        """
        raise NotImplementedError

    def update_after_execute(self, executed_weights: KafkaExecutionWeights):
        pass

    # def get_weights(self) -> KafkaExecutionWeights: ##depreeciated
    #     """
    #     for execution
    #     :param orm_session:
    #     :return:
    #     """
    #
    #
    #     ts = self.bars_ts.get_wrapped()
    #     assets = [s.asset for s in ts]
    #     assets.append(self.serialized_cash_asset)
    #
    #     last_data = self.pandas_df.iloc[-1].astype(float)
    #     self.logger.info(f"Rebalancing on weights for {last_data.name}")
    #     w = {asset: {"weight": last_data.loc[("weights", asset.symbol)],
    #                  "price": last_data.loc[("prices", asset.symbol)]} for asset in assets if
    #          last_data.loc[("weights", asset.symbol)] != 0}
    #     weights = HistoricalWeights(weights_date=datetime.datetime.now())
    #     weights.set_from_models(assets_dict=w)
    #
    #     return weights


class RollingVolumeBenchmarkWeights(RebalancerPortfolio):
    @TimeSerie._post_init_routines()
    def __init__(self, limit_assets: int,max_weight:float,rolling_volume_window:int,
                 exclude_stablecoins:bool,*args, **kwargs):
        self.limit_assets = limit_assets
        self.max_weight=max_weight
        self.rolling_volume_window=rolling_volume_window
        self.exclude_stablecoins=exclude_stablecoins
        super().__init__(*args, **kwargs)

    @property
    def bars_type_config(self):
        return dict(rolling_window=None, portfolio_type=None)

    @property
    def human_readable(self):
        return f"{self.__class__.__name__} rolling_volume_window: {self.rolling_volume_window} "

    def _produce_rebalancer_signal_df(self, closes_df,latest_value, *args, **kwargs) -> pd.DataFrame:
        data_df = self.bars_ts.pandas_df_concat_on_rows_by_key

        data_df = data_df.rename(columns={"key": "asset_symbol"}).set_index(["asset_symbol"], append=True)
        if latest_value is not None:
            if any(data_df.index.get_level_values("time") > latest_value):
                # require one month for at least previous rebalance
                data_df = data_df[data_df.index.get_level_values("time") > latest_value - datetime.timedelta(days=self.rolling_volume_window+1)]
            else:
                print("no new volume")
                volume= pd.DataFrame()
        volume = pd.pivot_table(data_df, values="volume", columns="asset_symbol", index="time")

        volume=volume.rolling(self.rolling_volume_window).mean()
        volume=volume.loc[closes_df.index]
        volume=volume*closes_df
        return volume

    def _produce_weights_and_execution_signal(self, closes_df, rebalance_flag, rebalancer_signal_df, latest_value) -> pd.DataFrame:
        volume_df=rebalancer_signal_df
        if latest_value is None:
            # the initial  rebalance may be zero, only is relevant for construction back history
            volume_df = volume_df.fillna(method="bfill")
        # limit to desired assets
        if self.limit_assets is not None:
            volume_df = volume_df.mask(
                volume_df.rank(axis=1, method='min', ascending=False) > self.limit_assets, 0)

        volume_weights = volume_df.divide(volume_df.sum(axis=1), axis=0)

        #limit to max weight exposure
        for i in tqdm(range(10),desc="fixing volume weights"):
            volume_weights=volume_weights.clip(0,self.max_weight)
            volume_weights=volume_weights.div(volume_weights.sum(axis=1),axis=0)

        cash_subtraction = self.cash_balance / volume_weights.shape[
            1] if self.limit_assets is None else self.cash_balance / self.limit_assets
        volume_weights = volume_weights.applymap(lambda x: max(x - cash_subtraction, 0) if x > 0 else 0)
        
        raise NotImplementedError # review shift
        volume_weights = volume_weights.shift(1).fillna(0)

        return volume_weights ,rebalance_flag


class EqualWeight(RebalancerPortfolio):
    @TimeSerie._post_init_routines()
    def __init__(self,  *args, **kwargs):
        """

        :param asset_list:
        :param source:
        :param rebalance_rule_config:
        :param args:
        :param kwargs:
        """
        super().__init__(*args,**kwargs)

    @property
    def bars_type_config(self):
        return dict(portfolio_config={}, portfolio_type="None")
    @property
    def human_readable(self):
        return f"{self.__class__.__name__}  "

    def _produce_rebalancer_signal_df(self, closes_df, latest_value, *args, **kwargs) -> pd.DataFrame:



        signal_df=1/closes_df.fillna(method="ffill").count(axis=1)
        signal_df=((closes_df*0+1).multiply(signal_df,axis=0)).fillna(0)

        return signal_df

    def _produce_weights_and_execution_signal(self, closes_df, rebalance_flag, rebalancer_signal_df,
                                              latest_value) -> [pd.DataFrame,pd.DataFrame]:

        
        return rebalancer_signal_df, rebalance_flag
class CoinMktCapBmrkWts(RebalancerPortfolio):

    @TimeSerie._post_init_routines()
    def __init__(self, limit_assets: int, exclude_stablecoins: bool,
                 *args, **kwargs):
        """

        :param asset_list:
        :param source:
        :param rebalance_rule_config:
        :param args:
        :param kwargs:
        """

        super().__init__(*args, **kwargs)

        self.historical_market_cap_ts = HistoricalCoinSupply(exclude_stablecoins=exclude_stablecoins)



        #historical_market_cap needs to be updates as frequent as the max supply
        self.limit_assets = limit_assets

    @property
    def bars_type_config(self):
        return dict(portfolio_config={},portfolio_type="None")

    @property
    def human_readable(self):

        return f"{self.__class__.__name__} #assets {len( self.bars_ts.related_time_series.keys())}  "

    def _produce_rebalancer_signal_df(self, closes_df,latest_value, *args, **kwargs) -> pd.DataFrame:

        #tickers
        symbols_to_id = {k:c[0] for k,c in self.symbol_to_search_map.items()}
        id_to_symbol={v:k for k,v in symbols_to_id.items()}
        tickers=list(symbols_to_id.values())
        

        closes_df.columns=[symbols_to_id[c] for c in closes_df.columns]
        market_cap_values = self.historical_market_cap_ts.back_market_cap_with_prices(prices_df=closes_df.copy(),
                                                                                        target_column="market_cap")
        closes_df.columns=[id_to_symbol[c] for c in closes_df.columns]
        market_cap_values.columns=[id_to_symbol[c] for c in market_cap_values.columns]
        return market_cap_values

    def _produce_weights_and_execution_signal(self, closes_df, rebalance_flag, rebalancer_signal_df, latest_value) -> pd.DataFrame:
        market_cap_values = rebalancer_signal_df

        if latest_value is None:
            # the initial  rebalance may be zero, only is relevant for construction back history
            market_cap_values = market_cap_values.fillna(method="bfill")

        # limit to desired assets
        if self.limit_assets is not None:
            market_cap_values = market_cap_values.mask(
                market_cap_values.rank(axis=1, method='min', ascending=False) > self.limit_assets, 0)

        market_cap_weights = market_cap_values.divide(market_cap_values.sum(axis=1), axis=0)

        cash_subtraction = self.cash_balance / market_cap_weights.shape[
            1] if self.limit_assets is None else self.cash_balance / self.limit_assets
        market_cap_weights = market_cap_weights.applymap(lambda x:max(x - cash_subtraction,0) if x > 0 else 0)
        market_cap_weights=market_cap_weights.shift(1).fillna(0)

        return market_cap_weights, rebalance_flag


# volume_adjusted_config: Union[None, dict] = None,

class RebalancerPortfolioBars(PortfolioBarsFromWeight):

    ACCEPTED_CLASSES=["CoinMktCapBmrkWts","RollingVolumeBenchmarkWeights","EqualWeight"]

    @TimeSerie._post_init_routines()
    def __init__(self, asset_list: ModelList, cash_asset: dict, cash_balance: float, frequency_id: str,
                 source: str, rebalance_rule_config: dict, class_extra_kwargs: dict, class_name: str,
                 *args, **kwargs):
        assert class_name in self.ACCEPTED_CLASSES
        try:
            ClassName = getattr(sys.modules[__name__], class_name)
        except KeyError:
            ClassName =globals()[class_name]
        except Exception as e:
            raise e
        all_kwargs = dict(asset_list=asset_list, cash_asset=cash_asset, cash_balance=cash_balance,
                          frequency_id=frequency_id,
                          source=source, rebalance_rule_config=rebalance_rule_config)
        all_kwargs.update(class_extra_kwargs)
        all_kwargs.update(kwargs)
        self.weights_ts = ClassName(**all_kwargs)
        self.bar_frequency_id = frequency_id

        super().__init__(*args, **kwargs)



class RebalancerPortBarsUpsample(BarTimeSerieUpsampleFromPortfolio):

    @TimeSerie._post_init_routines()
    def __init__(self, asset_list: ModelList, cash_asset: dict, cash_balance: float, frequency_id: str,
                 source: str, rebalance_rule_config: dict, class_extra_kwargs: dict, class_name: str,
                 upsample_frequency_id: str, intraday_bar_interpolation_rule: str,
                 *args, **kwargs):
        self.upsample_frequency_id = upsample_frequency_id
        self.trading_hours = asset_list[0].trading_hours
        self.intraday_bar_interpolation_rule = intraday_bar_interpolation_rule

        all_kwargs = dict(asset_list=asset_list, cash_asset=cash_asset, cash_balance=cash_balance,
                          frequency_id=frequency_id,
                          source=source, rebalance_rule_config=rebalance_rule_config)

        all_kwargs.update(kwargs)

        self.portfolio_bar_ts = RebalancerPortfolioBars(**all_kwargs, class_extra_kwargs=class_extra_kwargs,
                                                        class_name=class_name)

        super().__init__(*args, **kwargs)

