import pytz

from mainsequence.tdag import TimeSerie,  ModelList
import pandas as pd
import datetime


class TestFeature2(TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(self, asset_list: ModelList, other_config: str, *args, **kwargs):
        self.asset_list = asset_list
        self.asset_symbols_filter = [a.unique_identifier for a in asset_list]
        super().__init__(*args, **kwargs)

    def update_series_from_source(self, latest_value, **class_arguments):
        update_time = datetime.datetime.now(pytz.utc).replace(microsecond=0, second=0)
        data = pd.DataFrame(index=[update_time], columns=[a.unique_identifier for a in self.asset_list]).fillna(0)
        data.index.name = "time_index"
        data = data.melt(ignore_index=False,var_name="asset_symbol",value_name="feature_1")
        data = data.set_index("asset_symbol", append=True)

        return data

class TestFeature(TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(self, asset_list: ModelList,other_config:str, *args, **kwargs):
        self.asset_list=asset_list
        self.asset_symbols_filter = [a.unique_identifier for a in asset_list]
        self.base_feature=TestFeature2(asset_list=asset_list,other_config=other_config,*args,**kwargs)

        new_code_to=5

        super().__init__(*args, **kwargs)
    def update_series_from_source(self, latest_value, **class_arguments):
        update_time = datetime.datetime.now(pytz.utc).replace(microsecond=0, second=0)
        data=pd.DataFrame(index=[update_time],columns=[a.unique_identifier for a  in self.asset_list]).fillna(0)
        data.index.name = "time_index"
        data = data.melt(ignore_index=False,var_name="asset_symbol",value_name="feature_1")
        data=data.set_index("asset_symbol",append=True)

        return data
