import os

os.environ["VAM_ENDPOINT"]="http://127.0.0.1:8000"
os.environ["VAM_ADMIN_USER"]="admin"
os.environ["VAM_ADMIN_PASSWORD"]="admin"
os.environ["TDAG_ENDPOINT"]="http://127.0.0.1:8001"
os.environ["TDAG_ADMIN_USER"]="admin"
os.environ["TDAG_ADMIN_PASSWORD"]="admin"
os.environ["TDAG_DB_CONNECTION"]="postgresql://postgres:postgres@localhost:5436/timeseries"
os.environ["TDAG_RAY_CLUSTER_ADDRESS"]="ray://localhost:10001"


from mainsequence.tdag import ogm
from mainsequence.tdag import ModelList
from tests.tdag.time_series import (TestFeature)
from mainsequence.vam_client import CONSTANTS, Asset

import unittest


def build_test_feature():
    assets, response = Asset.filter(symbol__in=["BTC", "ETH","BNB"],
                                    asset_type=CONSTANTS.ASSET_TYPE_CRYPTO_SPOT,
                                    currency="USDT",
                                    execution_venue__symbol=CONSTANTS.BINANCE_FUTURES_EV_SYMBOL
                                    )

    ts1=TestFeature(asset_list=ModelList(assets),other_config="test",
                   local_kwargs_to_ignore=["asset_list"]
                   )


    ts2=TestFeature(asset_list=ModelList(assets[1:]),other_config="test",
                   local_kwargs_to_ignore=["asset_list"]
                   )

    return ts1,ts2

class TestTimeSerie(unittest.TestCase):



    def test_feature(self):
        ts1,ts2= build_test_feature()

        from mainsequence.tdag.time_series.update import SchedulerUpdater,get_or_pickle_ts_from_sessions
        import os
        import pandas as pd

        if os.path.isfile(ogm.get_ts_pickle_path(local_hash_id=ts1.local_hash_id)) == False:
            lpm = ts1.local_persist_manager  # call lpm to guarantee ts exist
            pickle_path, ts = get_or_pickle_ts_from_sessions(local_hash_id=ts1.local_hash_id,
                                                             remote_table_hashed_name=ts1.remote_table_hashed_name,
                                                             set_dependencies_df=True,
                                                             return_ts=True
                                                                 )
        if os.path.isfile(ogm.get_ts_pickle_path(local_hash_id=ts2.local_hash_id)) == False:
            lpm = ts2.local_persist_manager  # call lpm to guarantee ts exist
            pickle_path, ts = get_or_pickle_ts_from_sessions(local_hash_id=ts2.local_hash_id,
                                                             remote_table_hashed_name=ts2.remote_table_hashed_name,
                                                             set_dependencies_df=True,
                                                             return_ts=True
                                                             )

        DEBUG_RUN=False
        #update forst time serie
        SchedulerUpdater.debug_schedule_ts(time_serie_hash_id=ts1.local_hash_id,
                                           break_after_one_update=True,
                                           run_head_in_main_process=True,
                                           wait_for_update=False, force_update=True,
                                           debug=DEBUG_RUN,update_tree=True,)
        #update second time serie
        SchedulerUpdater.debug_schedule_ts(time_serie_hash_id=ts2.local_hash_id,
                                           break_after_one_update=True,
                                           run_head_in_main_process=True,
                                           wait_for_update=False, force_update=True,
                                           debug=DEBUG_RUN, update_tree=True, )

        df_1=ts1.get_df_greater_than(None)
        df_2=ts2.get_df_greater_than(None)

        #should return same data
        pd.testing.assert_frame_equal(df_1,df_2)

        #filter by update symmbols should be different
        asset_symbols=[c.unique_identifier for c in ts1.asset_list]
        df_1 = ts1.get_df_greater_than(None,asset_symbols=[c.internal_id for c in ts1.asset_list])
        self.assertListEqual(df_1.index.get_level_values("asset_symbol").unique().tolist(),asset_symbols)

        asset_symbols = [c.unique_identifier for c in ts2.asset_list]
        df_2 = ts1.get_df_greater_than(None, asset_symbols=asset_symbols)
        self.assertListEqual(df_2.index.get_level_values("asset_symbol").unique().tolist(), asset_symbols)




