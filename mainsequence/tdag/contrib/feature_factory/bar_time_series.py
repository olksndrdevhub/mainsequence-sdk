import gc

import pytz
from mainsequence.tdag.time_series import TimeSerie,  ModelList
from mainsequence.tdag.time_series.utils import (dispatch_upsampled_data, string_frequency_to_minutes, dispatch_deflated_prices, DeflatedPricesBase, DeflatedPrices
                                                 )
from mainsequence.tdag.time_series import (get_feature_frequency
                                           )



from mainsequence.tdag.contrib.feature_factory import bar_features as bar_features_module
from typing import Union
import pandas as pd
import numpy as np
import datetime
import copy
from mainsequence.tdag.time_series.time_series import  TimeSerieConfigKwargs
from mainsequence.tdag_client.utils import inflate_json_compresed_column






class FeaturesFactory:

    def __init__(self,features_list:list):
        """

        :param features_dict: {feature_name:feature_kwargs}
        """
        features={}
        for counter,feat in enumerate(features_list):
            assert len(feat.keys())==1
            key=list(feat.keys())[0]
            features[counter]=getattr(bar_features_module,key)(**feat[key])


        self.features=features

    def set_latest_observation(self,latest_observation:pd.DataFrame):
        for f in self.features.keys():
            self.features[f].set_latest_observation(latest_observation)
    
    @property
    def x_required_columns(self):
        required_cols = []
        for feature in self.features.values():
            if hasattr(feature,"x_required_columns") == True:
                required_cols.extend(feature.x_required_columns)
        return list(set(required_cols))
        
    @property
    def required_columns(self):
        required_cols=[]
        for feature in self.features.values():
            required_cols.extend(feature.required_columns)
        return list(set(required_cols))
    @property
    def max_time_delta(self):
        time_deltas=[]
        for feature in self.features.values():
            time_deltas.append(feature.min_delta_needed)
        return np.max(time_deltas)
    @property
    def num_threads(self):
        threads = []
        for feature in self.features.values():
            threads.append(feature.num_threads)
        return -1 if np.min(threads)==-1 else np.max(threads)


    def build_features(self,data_df,logger,upsample_frequency_id,original_latest_value,
                       last_observation:Union[None,pd.DataFrame],
                       extra_data:Union[dict,None]=None):
        """

        :param data_df: dataframe with information up to the  longest feature
        :param logger:
        :return:
        """
        features=[]

        freq_increase=datetime.timedelta(minutes=string_frequency_to_minutes(upsample_frequency_id))
        for feature,feature_instance in self.features.items():
            logger.info(f"Feature {feature}-{feature_instance.feature_name} : Building ... ")
            # reduce dataframe size for features that require less data
            if original_latest_value is not None:
                min_time_delta = feature_instance.min_delta_needed + freq_increase
                start_required_date = original_latest_value - min_time_delta
                tmp_df=data_df[data_df.index>=start_required_date].copy()
            else:
                tmp_df=data_df.copy()
            calc_kwargs=dict(data_df=tmp_df)

            calc_kwargs["last_observation"]=last_observation
            if extra_data is not None:
                calc_kwargs.update(extra_data)

            feat_df=feature_instance.calculate_feature(**calc_kwargs)
            features.append(feat_df)
            logger.info(f"Feature {feature}-{feature_instance.feature_name} : Completed ... ")
        features=pd.concat(features,axis=1)
        logger.info("All Features Built")
        assert features.columns.duplicated().sum()==0

        return features

    def __repr__(self):
        return "/n".join([feature_instance.__repr__()+"rw_" for feature_instance in self.features.values()])



def override_features_with_feature_frequency(input_features:list,frequency_id:str,upsample_frequency_id:str):
    """
    cahnges feature frequency according to upsample
    :param input_features:
    :return:
    """
    feature_frequency = get_feature_frequency(frequency_id=frequency_id,
                                              upsample_frequency_id=upsample_frequency_id)
    new_features = []
    for feat in input_features:
        for feature ,feature_kwargs in feat.items():
            new_feat_kwargs=copy.deepcopy(feature_kwargs)
            new_feat_kwargs["time_serie_frequency"] = feature_frequency
            new_features.append({feature:new_feat_kwargs})
    return new_features

def build_target_time_serie(asset,portfolio_kwargs:dict,ts_kwargs:dict,bar_frequency_id,
                            upsample_frequency_id,intraday_bar_interpolation_rule,source):

    if portfolio_kwargs == None:
        ts= dispatch_upsampled_data(asset=asset, bar_frequency_id=bar_frequency_id,upsample_frequency_id=upsample_frequency_id,
                                                intraday_bar_interpolation_rule=intraday_bar_interpolation_rule,

                                                  source=source)
    else:
        TargetClass, portfolio_kwargs = portofolio_factory(
                                                           portfolio_kwargs=portfolio_kwargs,

                                                           intraday_bar_interpolation_rule=intraday_bar_interpolation_rule,
                                                           upsample_frequency_id=upsample_frequency_id,
                                                           bar_frequency_id=bar_frequency_id)

        portfolio_kwargs.update(**ts_kwargs)
        ts=TargetClass(**portfolio_kwargs)
        if "init_meta" in portfolio_kwargs.keys():
            portfolio_kwargs.pop("init_meta",None)


    return ts,portfolio_kwargs


def build_features_from_upsampled_df(upsampled_ts:Union[TimeSerie,pd.DataFrame],latest_value,
                                     features_factory:FeaturesFactory,logger,
                                     upsample_frequency_id:str,bar_frequency_id:str,extra_data:Union[dict,None]=None,
                                     concatenate_with_source=True,
                                     last_observation:Union[None,pd.DataFrame]=None
                                     )->pd.DataFrame:
    """
    build function for Bar and portfolio Bars
    :param latest_value:
    :param features_factory:
    :param upsampled_df:
    :param upsample_frequency_id:
    :param bar_frequency_id:
    :return:
    """
    upsample_freq_obs = string_frequency_to_minutes(upsample_frequency_id) // string_frequency_to_minutes(
        bar_frequency_id)
    original_latest_value = latest_value
    min_time_delta = features_factory.max_time_delta
    min_time_delta = min_time_delta + datetime.timedelta(
        minutes=string_frequency_to_minutes(upsample_frequency_id))
    if latest_value is not None:
        start_required_date = latest_value - min_time_delta
        if isinstance(upsampled_ts, pd.DataFrame):
            upsampled_df=upsampled_ts
            upsampled_df=upsampled_df[upsampled_df.index>=start_required_date]
        else:
            upsampled_df =upsampled_ts.get_df_greater_than_in_table(target_value=start_required_date,great_or_equal=True)
        if (upsampled_df.shape[0] / string_frequency_to_minutes(upsample_frequency_id)) <=1.0:

            raise Exception (f"Error in data been pulled  start required date{start_required_date} and latest value {latest_value}")
    else:
        if isinstance(upsampled_ts, pd.DataFrame):
            upsampled_df = upsampled_ts
        else:
            upsampled_df=upsampled_ts.get_persisted_ts()
        assert (upsampled_df.shape[0] / string_frequency_to_minutes(upsample_frequency_id)) >1.0
    
    
    features = []
    if upsample_freq_obs >1:
        for start_row in range(upsample_freq_obs):
            tmp_df_upsampled = upsampled_df.iloc[start_row::upsample_freq_obs]
            features_upsampled = features_factory.build_features(data_df=tmp_df_upsampled,
                                                                 logger=logger,
                                                                 last_observation=last_observation,
                                                                 upsample_frequency_id=upsample_frequency_id,
                                                                 original_latest_value=original_latest_value
                                                                 )
            features.append(features_upsampled)
        features = pd.concat(features, axis=0).sort_index()
    else:
        features= features_factory.build_features(data_df=upsampled_df,
                                                  logger=logger,
                                                  extra_data=extra_data,
                                                  last_observation=last_observation,
                                                  upsample_frequency_id=upsample_frequency_id,
                                                  original_latest_value=original_latest_value
                                                  )
    if features.index.duplicated().sum() != 0:
        logger.error("Duplicated values found in features")
        raise Exception

    if concatenate_with_source == True:
        tmp_df = pd.concat([upsampled_df, features], axis=1)
    else:
        tmp_df=features

    if latest_value is not None:
        try:
            tmp_df = tmp_df[tmp_df.index > original_latest_value]
        except Exception as e:
            raise e
    else: # guarantee that there is
        tmp_df=tmp_df[tmp_df.index>tmp_df.index[0]+min_time_delta]

    if upsample_freq_obs>1:
        #add name to feat
        new_cols=pd.MultiIndex.from_tuples([(c[0]+f"_u{upsample_freq_obs}",c[1]) for c in tmp_df.columns],
                                           names=tmp_df.columns.names)
        tmp_df.columns=new_cols
    return tmp_df







def melt_features(logger,data_df: pd.DataFrame, latest_value,last_observation=None):
    assert data_df.columns.names == ["feature_name", "asset_id"]
    if latest_value is not None:
        data_df = data_df[data_df.index > latest_value]
    if data_df.shape[0] == 0:
        return pd.DataFrame()

    data_df = data_df.stack(1)
    data_df.index.names = ["time_index", "asset_symbol"]
    if data_df.isnull().sum().sum() !=0 and latest_value is not None:
        if last_observation is not None:
            if isinstance(data_df.index,pd.MultiIndex)==True:
                previous_symbols=last_observation.index.get_level_values(1).unique()
                logger.error(data_df.isnull().sum())
                assert data_df[data_df.index.get_level_values(1).isin(previous_symbols)].isnull().sum().sum()==0
            else:
                logger.error(data_df.isnull().sum())
                raise Exception
            
            
        
        
    data_df = data_df.dropna()
    data_df.columns=[c.lower() for c in data_df.columns]
    return data_df

#Compounded
def compress_compounded_features(data_df,latest_value,use_short=False):


    if latest_value is None:
        data_df = data_df.dropna(how="all", axis=0)
        assert data_df.iloc[-1].isnull().sum()==0
    else:
        if data_df.isnull().sum().sum() != 0 and data_df.index.min()> datetime.datetime(2020,6,1).replace(tzinfo=pytz.utc):
            raise Exception(f"Nulls in features {data_df.isnull().sum()}")
        # compress features
    compressed_data = []
    to_comp = lambda df, name: pd.Series(data=df.to_dict('records'), index=df.index).to_frame(name)
    for feat_name in data_df.columns.get_level_values("feature_name").unique():
        prefix="jcomp_" if use_short== True else "json_compressed_"
        new_name = f"{prefix}{feat_name}"
        assert len(new_name) < 64
        tmp_df = data_df[feat_name].replace({np.nan: None})

        compressed_data.append(to_comp(tmp_df, new_name))

    compressed_data = pd.concat(compressed_data, axis=1)
    return compressed_data




class CommonalityFeature(TimeSerie):
    WEIGHTS_SCHEMES=["equal"]

    @TimeSerie._post_init_routines()
    def __init__(self,asset_list:ModelList,time_series_config:TimeSerieConfigKwargs,weight_scheme:str,
                 rolling_window:int,column_contains:Union[None,str]=None,
                 local_kwargs_to_ignore=["asset_list"],
                 *args,**kwargs ):

        assert weight_scheme in self.WEIGHTS_SCHEMES

        self.asset_symbols_filter = [a.internal_id for a in asset_list]

        if "bars_type_config_y" in time_series_config.keys():
            TsClass= CompCrossAssetFeat
            self.bars_feature_config=time_series_config["bars_type_config_y"]
        else:
            TsClass=CompAssetFeat
            self.bars_feature_config = time_series_config["bars_type_config"]
        time_series_config["asset_list"]=asset_list
        self.feature_source_ts=TsClass(**time_series_config)
        self.feature_source_bars_prefix = self.feature_source_ts.get_bars_prefix()

        self.weight_scheme=weight_scheme
        self.rolling_window=rolling_window
        self.column_contains=column_contains
        super().__init__(*args, **kwargs)

    def get_minimum_required_depth_for_update(self):
        """
        Controls the minimum depth that needs to be rebuil
        Returns
        -------

        """
        return 1
    @property
    def human_readable(self):

        name = f"Commonality {self.feature_source_ts.human_readable}  {self.rolling_window} "
        return name


    def get_bars_prefix(self):
        return self.feature_source_bars_prefix
    def update_series_from_source(self, latest_value, *args, **kwargs):



        upsample_frequency=self.bars_feature_config["upsample_frequency_id"]
        min_time_delta = datetime.timedelta(minutes=string_frequency_to_minutes(upsample_frequency))

        start_date=latest_value
        if latest_value is not None:
            start_date = start_date - min_time_delta * self.rolling_window*3

        data_df = self.feature_source_ts.get_df_greater_than_in_table(target_value=start_date ,great_or_equal=True ,
                                                             asset_symbols=self.asset_symbols_filter,
                                                             force_db_look=False)

        inflated_features=[]
        target_columns=data_df.columns
        data_df = data_df.reset_index().pivot(columns="asset_symbol", index="time_index")
        if self.column_contains is not None:
            target_columns=[c for c in target_columns if self.column_contains in c]
        for col in target_columns:
            y =data_df[col]

            if self.weight_scheme =="equal":
                x=y.sum(axis=1)/y.count(axis=1) #mean by presetn non nans

            x_mean=x.rolling(self.rolling_window,min_periods=self.rolling_window).mean()
            y_mean=y.rolling(self.rolling_window,min_periods=self.rolling_window).mean()

            x_demean = x - x_mean
            y_demean = y - y_mean

            xy = y_demean.multiply(x_demean, axis=0).rolling(self.rolling_window, min_periods=self.rolling_window).sum()
            x_2 = x_demean.multiply(x_demean, axis=0).rolling(self.rolling_window, min_periods=self.rolling_window).sum()
            beta = xy.divide(x_2, axis=0).fillna(method="ffill")
            # beta = beta.fillna(beta.mean())

            prediction=beta.multiply(x, axis=0)

            intercept = y_mean-beta.multiply(x_mean,axis=0)
            alpha = y -prediction
            commonality=y_demean.rolling(self.rolling_window, min_periods=self.rolling_window).corr(x_demean)**2


            commonality=commonality.dropna(axis=0,how="all")


            target_column = col

            rename = lambda df,name:pd.concat([alpha],keys=[f"cmn_{target_column}_{name}"], axis=1)
            #To json_compressed
            data = pd.concat([rename(alpha,"al"),rename(beta,"be"),rename(intercept,"int"),
                              rename(intercept, "commonality")       ],
                             axis=1)
            data.columns.names = ["feature", "asset_symbol"]
            data = data.melt(ignore_index=False).set_index("asset_symbol", append=True)
            data = pd.pivot_table(data, index=data.index, columns="feature", values="value")
            data.index=pd.MultiIndex.from_tuples([c for c in data.index])
            data.index.names=["time_index","asset_symbol"]





            inflated_features.append(data)
        inflated_features=pd.concat(inflated_features,axis=1)
        if latest_value is not None:
            data = data[data.index.get_level_values(0) > latest_value]
        inflated_features=inflated_features.dropna()
        if inflated_features.shape[0]==0:
            return pd.DataFrame()


        return inflated_features

class CompCrossAssetFeat(TimeSerie):

    @TimeSerie._post_init_routines()
    def __init__(self, asset_list: ModelList, feature: dict,
                 bars_type_config_x:Union[str, None], bars_type_config_y:Union[str, None],
                  *args, **kwargs):

        assert isinstance(asset_list,ModelList)

        self.asset_symbols_filter = [a.internal_id for a in asset_list]

        bars_type_config_x = bars_type_config_x if bars_type_config_x is not None else dict(portfolio_config={},
                                                                                      portfolio_type="None")
        bars_type_config_y = bars_type_config_y if bars_type_config_y is not None else dict(portfolio_config={},
                                                                                            portfolio_type="None")

        self.bars_type_config_y=bars_type_config_y
        self.bars_type_config_x=bars_type_config_x

     
        
        self.number_of_assets = len(asset_list)
        super().__init__(*args, **kwargs)

        init_meta ={} if "init_meta" not in kwargs else kwargs["init_meta"]

        if "bars_ts_x" in init_meta:
            self.bars_ts_x = init_meta["bars_ts_x"]
        else:
            x_kwargs=copy.deepcopy(kwargs)
            x_kwargs.update(bars_type_config_x)

            self.bars_ts_x = dispatch_deflated_prices(asset_list=asset_list, **x_kwargs  )
        if "bars_ts_y" in init_meta:
            self.bars_ts_y = init_meta["bars_ts_y"]
        else:
            y_kwargs = copy.deepcopy(kwargs)
            y_kwargs.update(bars_type_config_y)
            self.bars_ts_y = dispatch_deflated_prices(asset_list=asset_list, **y_kwargs   )
            
            
        try:

            assert self.bars_ts_y.upsample_frequency_id==self.bars_ts_x.upsample_frequency_id
            assert self.bars_ts_y.bar_frequency_id == self.bars_ts_x.bar_frequency_id
            new_features = override_features_with_feature_frequency(frequency_id=bars_type_config_y['bar_frequency_id'], input_features=[feature],
                                                                    upsample_frequency_id=bars_type_config_y['upsample_frequency_id'])
        except Exception as e:
            raise e
        self.features_factory = FeaturesFactory(features_list=new_features)

        self.bars_ts_y_suffix = self.bars_ts_y.suffix
        self.bars_ts_x_suffix = self.bars_ts_x.suffix

    def get_bars_prefix(self):
        
       
        new_name = "_y_"+self.bars_ts_y_suffix+"_x_"+self.bars_ts_x_suffix
        feat_name = self.features_factory.features[list(self.features_factory.features)[0]]
        if "RollResAB" in str(feat_name):
            new_name=new_name+f"_{self.features_factory.features[0].rolling_window}"
        
        
        return new_name
    def get_minimum_required_depth_for_update(self):
        """
        Controls the minimum depth that needs to be rebuil
        Returns
        -------

        """
        return 1

    @property
    def human_readable(self):
        try:
            name = f"{self.__class__.__name__} {self.features_factory.features[list(self.features_factory.features)[0]]}"
        except Exception as e:
            raise e
        if self.bars_type_config_y['upsample_frequency_id'] !="1min":
            name=name+f"Upsample: {self.bars_type_config_y['upsample_frequency_id']}"

        name = name + f"y_{self.bars_type_config_y['portfolio_type']}_x{self.bars_type_config_x['portfolio_type']} {self.bars_type_config_x['portfolio_config']['rolling_window']}"
        return name
    
    def _get_upsampled_data(self,start_required_date:datetime.datetime,
                            required_columns:list,great_or_equal:bool,
                                end_value: datetime.datetime
                            )->pd.DataFrame:

        upsampled_df = self.bars_ts_y.get_df_between_dates(start_date=start_required_date,great_or_equal=great_or_equal,
                                                                 columns=required_columns,
                                                          asset_symbols=self.asset_symbols_filter,
                                                          end_date=end_value,less_or_equal=False,
                                                                 
                                                                 )
        upsampled_df=upsampled_df.reset_index().pivot(index="time_index",columns="asset_symbol")
        upsampled_df.columns.names=["feature","asset_id"]
        upsampled_df.rename(mapper=lambda x: f'{x}_asset_y',
                    axis='columns',
                    level=0,
                    inplace=True)

        return upsampled_df
    
    def _get_x_data(self,required_columns:list,start_required_date:datetime.datetime,
                    end_value:datetime.datetime):
        x_required_columns=required_columns
        if x_required_columns is not None:
            x_required_columns = required_columns + ["last_execution_weights"]
        asset_symbols = self.asset_symbols_filter + ["benchmark"]

        x_asset_df = self.bars_ts_x.get_df_between_dates(start_date=start_required_date,
                                                        columns=x_required_columns,
                                                        asset_symbols=asset_symbols,
                                                        great_or_equal=True, less_or_equal=False,
                                                        end_date=end_value

                                                        )

        x_asset_weights = x_asset_df[["last_execution_weights"]].reset_index().pivot(index="time_index",
                                                                                     values="last_execution_weights",
                                                                                     columns="asset_symbol")
        x_asset_weights = x_asset_weights[[c for c in self.asset_symbols_filter if c in x_asset_weights.columns]]
        
        if required_columns is not None:
            x_asset_df = x_asset_df[x_asset_df.index.get_level_values(1) == "benchmark"][required_columns].reset_index(
                level=1, drop=True)
            x_asset_df.columns = pd.MultiIndex.from_tuples([(f"{c}_asset_x", "bemchmark") for c in x_asset_df.columns])
        else:
            x_asset_df = x_asset_df.reset_index().pivot(index="time_index", columns="asset_symbol")
            x_asset_df.columns.names = ["feature", "asset_id"]
            x_asset_df.rename(mapper=lambda x: f'{x}_asset_x',
                                axis='columns',
                                level=0,
                                inplace=True)





        return x_asset_weights, x_asset_df

    def update_series_from_source(self, latest_value, *args, **kwargs):

        INIT_LATEST=datetime.datetime(2017, 8, 1).replace(tzinfo=pytz.utc)
        feat_name = self.features_factory.features[list(self.features_factory.features)[0]]
        
        
        min_time_delta = self.features_factory.max_time_delta
        required_columns = self.features_factory.required_columns
        
        x_required_columns=required_columns
        if len(self.features_factory.x_required_columns)>0:
            x_required_columns=self.features_factory.x_required_columns
            x_required_columns=None if x_required_columns[0] is None else x_required_columns
        
        
        start_required_date=latest_value
        last_observation=None
        if latest_value is not None:

            min_time_delta = min_time_delta + datetime.timedelta(
                minutes=string_frequency_to_minutes(self.bars_ts_y.upsample_frequency_id))
            start_required_date = latest_value - min_time_delta
            last_observation=self.get_df_greater_than_in_table(target_value=latest_value,great_or_equal=True)
        else:
            latest_value = INIT_LATEST
            start_required_date=latest_value
        # to control first build ram requirements
        next_value =  datetime.datetime.now(pytz.utc)
        if "RollResAB" in str(feat_name):
            next_value = latest_value + datetime.timedelta(days=180)

        

        end_value = None if next_value >= datetime.datetime.now(pytz.utc).replace(hour=0, minute=0,
                                                                                  second=0) else next_value

        x_asset_weights, x_asset_df=self._get_x_data(required_columns=x_required_columns,end_value=end_value,
                                                     start_required_date=start_required_date)

        upsampled_df=self._get_upsampled_data(start_required_date=start_required_date,required_columns=required_columns,
                                              end_value=end_value,     great_or_equal=True)
        
        max_earliest = max(x_asset_df.index.min(), upsampled_df.index.min())
        min_latest = min(x_asset_df.index.max(), upsampled_df.index.max())
        start_required_date = min_latest - min_time_delta
        x_asset_df, upsampled_df = x_asset_df[x_asset_df.index >= max_earliest], upsampled_df[
            upsampled_df.index >= max_earliest]
        x_asset_df, upsampled_df = x_asset_df[x_asset_df.index <= min_latest], upsampled_df[upsampled_df.index <= min_latest]
        if x_asset_df.shape[0] == 0 or upsampled_df.shape[0] == 0 or start_required_date < max_earliest:
            raise Exception(
                f"Not Enough data to calculate features latest {x_asset_df.shape[0]} {upsampled_df.shape[0]} start_required_date {start_required_date} max earliest{max_earliest}")
        upsampled_df=upsampled_df.fillna(method="ffill")# force ffill as prices should be interpolated
        if upsampled_df.isnull().sum().sum()!=0 and end_value is not None:
            #try a higher  start to ffill
            self.logger.warning("Increasing rolling windo")
            if latest_value !=INIT_LATEST:
                min_index=upsampled_df.index.min()
                higher_sample = self._get_upsampled_data(start_required_date=start_required_date-datetime.timedelta(days=10),
                                                         great_or_equal=True,
                                                         end_value=end_value,
                                                         required_columns=required_columns).fillna(method="ffill")
                upsampled_df=pd.concat([higher_sample[higher_sample.index<min_index],upsampled_df],axis=0).fillna(method="ffill")
                upsampled_df=upsampled_df[upsampled_df.index>=min_index].copy()
            

        upsampled_df.columns.names = ["col_name", "asset_id"]
        x_asset_df.columns.names = ["col_name", "asset_id"]
        upsampled_df = pd.concat([upsampled_df,x_asset_df], axis=1)
        del x_asset_df
        gc.collect()

        if upsampled_df.shape[0] == 0:
            return pd.DataFrame()

        x_asset_weights=x_asset_weights.reindex(upsampled_df.index)
        data_df = build_features_from_upsampled_df(upsampled_ts=upsampled_df,
                                                  latest_value=latest_value,
                                                   last_observation=last_observation,
                                                  features_factory=self.features_factory,
                                                  logger=self.logger,
                                                  upsample_frequency_id=self.bars_ts_y.upsample_frequency_id,
                                                  bar_frequency_id=self.bars_ts_y.bar_frequency_id,
                                                  extra_data=dict(x_asset_weights=x_asset_weights),
                                                  concatenate_with_source=False)

        data_df = data_df.replace([-np.inf, np.inf], np.nan)
        data_df=data_df.dropna(how="all")


        if (data_df.isnull().sum().sum())>0:
            self.logger.warning(data_df.isnull().sum())
        data_df=melt_features(logger=self.logger,data_df=data_df,latest_value=latest_value,
                              last_observation=last_observation
                              )


        
        return data_df


class CompAssetFeat(TimeSerie):

    @TimeSerie._post_init_routines()
    def __init__(self,asset_list:ModelList, feature: dict,bars_type_config:dict,
                 *args,**kwargs):

        assert isinstance(asset_list,ModelList)
        self.asset_symbols_filter = [a.internal_id for a in asset_list]
        self.bars_type_config=bars_type_config
        super().__init__(*args, **kwargs)


        new_features = override_features_with_feature_frequency(frequency_id=bars_type_config['bar_frequency_id'], input_features=[feature],
                                                                upsample_frequency_id=bars_type_config['upsample_frequency_id'])
        self.features_factory = FeaturesFactory(features_list=new_features)
        self.bars_type_config=bars_type_config
        self.number_of_assets=len(asset_list)
        init_meta= {} if "init_meta" not in kwargs.keys() else  kwargs["init_meta"]
        if "bars_ts" in init_meta:
            self.bars_ts=init_meta["bars_ts"]
        else:

            self.bars_ts=dispatch_deflated_prices(asset_list=asset_list, **bars_type_config  )



        self.bars_ts_suffix=self.bars_ts.suffix


    def _get_update_groups(self,latest_value):
        
        all_symbols=self.bars_ts.asset_symbols_filter
        single_latest_value = latest_value if latest_value is not None else "None"

        if isinstance(self.bars_ts,DeflatedPrices) ==False:
            self.logger.info("TODO: modify for other benchmarks")
            all_symbols=all_symbols+["benchmark"]

        if self.local_persist_manager.metadata['sourcetableconfiguration']is not None:
            max_per_asset_symbol=self.local_persist_manager.metadata["sourcetableconfiguration"]['multi_index_stats']['max_per_asset_symbol']
            
            new_max_per_asset_symbol={}
            for k in all_symbols:
                if k in max_per_asset_symbol.keys():
                    new_max_per_asset_symbol[k]=self.local_persist_manager.dth.request_to_datetime(max_per_asset_symbol[k])
                else:
                    new_max_per_asset_symbol[k] ="None"
            
            max_per_asset_group=pd.Series(new_max_per_asset_symbol).to_frame("latest_value")
            max_per_asset_group.index.name="asset_symbol"
            max_per_asset_group=max_per_asset_group.reset_index()
        else:
            max_per_asset_group=pd.DataFrame(data=all_symbols,columns=["asset_symbol"])

            max_per_asset_group["latest_value"]=single_latest_value
            
        return max_per_asset_group
        
        
    def get_minimum_required_depth_for_update(self):
        """
        Controls the minimum depth that needs to be rebuil
        Returns
        -------

        """
        return 1


    @property
    def human_readable(self):


        name = f"{self.__class__.__name__} {self.features_factory.features[list(self.features_factory.features)[0]]}"
        if self.bars_type_config['upsample_frequency_id'] != "1min":
            name = name + f"Upsample: {self.bars_type_config['upsample_frequency_id']}"

        if self.bars_type_config['portfolio_type']!=DeflatedPricesBase.NONE:
            name = name + f"{self.bars_type_config['portfolio_type']} {self.bars_type_config['portfolio_config']['rolling_window']}"

        return name

    def get_bars_prefix(self):
        new_name = ""
        if self.bars_type_config["portfolio_type"]!=DeflatedPricesBase.NONE:
            new_name = "_"+self.bars_ts_suffix
        return new_name
        
    def update_series_from_source(self, latest_value, *args, **kwargs):

        
        max_per_asset_group=self._get_update_groups(latest_value=latest_value)
        all_group_data=[]
        for latest_value,assets_in_group in max_per_asset_group.groupby("latest_value"):

            latest_value=latest_value if latest_value!="None" else None

            min_time_delta = self.features_factory.max_time_delta+datetime.timedelta(
            minutes=string_frequency_to_minutes(self.bars_ts.upsample_frequency_id))


            required_columns=self.features_factory.required_columns
            start_date = latest_value

            last_observation=None
            if latest_value is not None:
                start_date = start_date - min_time_delta




            if required_columns is None:
                self.logger.warning(f"************No required columns for {self.human_readable}*************")

            upsampled_df=self.bars_ts.get_df_greater_than_in_table(target_value=start_date,
                                                          symbol_list=assets_in_group["asset_symbol"].to_list(),
                                                          columns=required_columns)
        
            upsampled_df=upsampled_df.reset_index().pivot(index="time_index",columns="asset_symbol",
                                                          values=required_columns)
            upsampled_df.columns.names=["feature","asset_id"]
            if upsampled_df.shape[0]==0:
                all_group_data.append(pd.DataFrame())
                continue 
            upsampled_df=upsampled_df.fillna(method="ffill")
            if upsampled_df.isnull().sum().sum()>0 and latest_value is not None:
                self.logger.warning("BackFilling upsample data ")
                upsampled_df=upsampled_df.fillna(method="bfill")



            data_df=build_features_from_upsampled_df(upsampled_ts=upsampled_df,
                                                      latest_value=latest_value,
                                                      features_factory=self.features_factory,
                                                      logger=self.logger,
                                                      upsample_frequency_id=self.bars_ts.upsample_frequency_id,
                                                      bar_frequency_id=self.bars_ts.bar_frequency_id,
                                                      concatenate_with_source=False,
                                                     last_observation=last_observation,
                                                     )
            del upsampled_df
            gc.collect()
            data_df=data_df.replace([-np.inf,np.inf],np.nan)
            data_df=data_df.dropna(how="all")
            if (data_df.isnull().sum().sum())>0:
                self.logger.warning(data_df.isnull().sum())
            data_df=melt_features(data_df=data_df,latest_value=latest_value ,logger=self.logger)

            all_group_data.append(data_df)
            
        data_df=pd.concat(all_group_data,axis=0)
        
        
        return data_df



class FeatOnFeat(TimeSerie):

    @TimeSerie._post_init_routines()
    def __init__(self, base_feat_config: TimeSerieConfigKwargs, feature: dict,
                 bar_frequency_id: str, upsample_frequency_id=None,
                 *args, **kwargs):
        TsClass = CompCrossAssetFeat if "asset_y_list" in base_feat_config.keys() else CompAssetFeat

        build_base_feat=True
        if "init_meta" in kwargs.keys():
            if "base_feat" in kwargs["init_meta"]:
                self.feature_source_ts= kwargs['init_meta']["base_feat"]
                build_base_feat=False
        if build_base_feat==True:
            self.feature_source_ts = TsClass(**base_feat_config)
        self.feature_source_bars_prefix=self.feature_source_ts.get_bars_prefix()
        self.upsample_frequency_id=upsample_frequency_id

        self.bar_frequency_id=bar_frequency_id
        new_features = override_features_with_feature_frequency(frequency_id=bar_frequency_id, input_features=[feature],
                                                                upsample_frequency_id=upsample_frequency_id)
        self.features_factory = FeaturesFactory(features_list=new_features)
        super().__init__(*args, **kwargs)
    @property
    def human_readable(self):
        name = f"{self.__class__.__name__} {self.features_factory.features[list(self.features_factory.features)[0]]}"
        if self.upsample_frequency_id != "1min":
            name = name + f"Upsample: {self.upsample_frequency_id}"

        name = f"FeatOnFeat {name} on  {self.feature_source_ts.human_readable} "
        return name


    def get_bars_prefix(self):
        return "_"+self.feature_source_bars_prefix
    
    
    def update_series_from_source(self, latest_value, *args, **kwargs):
        min_time_delta = self.features_factory.max_time_delta + datetime.timedelta(
            minutes=string_frequency_to_minutes(self.upsample_frequency_id))
        required_columns = self.features_factory.required_columns
        assert len(required_columns)==1
        start_date = latest_value

        # start_date=datetime.datetime(2023,9,30)
        if latest_value is not None:
            start_date = start_date - min_time_delta
       
        upsampled_df = self.feature_source_ts.get_df_greater_than_in_table(target_value=start_date, great_or_equal=True,

                                                        )
        upsampled_df = inflate_json_compresed_column(upsampled_df[f"json_compressed_{required_columns[0]}"])
        upsampled_df.columns=pd.MultiIndex.from_tuples([(required_columns[0], a) for a in upsampled_df.columns])

        if upsampled_df.shape[0] == 0:
            return pd.DataFrame()
        if upsampled_df.isnull().sum().sum() > 0 and latest_value is not None:
            raise Exception("Nulls in feature")
        data_df = build_features_from_upsampled_df(upsampled_ts=upsampled_df,
                                                   latest_value=latest_value,
                                                   features_factory=self.features_factory,
                                                   logger=self.logger,
                                                   upsample_frequency_id=self.upsample_frequency_id,
                                                   bar_frequency_id=self.bar_frequency_id,
                                                   concatenate_with_source=False)

        if data_df.shape[0] == 0:
            return pd.DataFrame()
        assert data_df.columns.names == ["feature_name", "asset_id"]
        compressed_data = compress_compounded_features(data_df=data_df, latest_value=latest_value,
                                                       use_short=True,
                                                       )

        assert all([len(c)<63 for c in compressed_data.columns])
      
        return compressed_data