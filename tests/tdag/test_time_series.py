import os
import dotenv

dotenv.load_dotenv('../../.env')




from mainsequence.tdag import ogm
from mainsequence.tdag import ModelList
from tests.tdag.time_series import (TestFeature)
from mainsequence.vam_client import CONSTANTS, Asset

import unittest


def build_test_feature():
    assets = Asset.filter(symbol__in=["BTC", "ETH","BNB"],
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


        import os
        import pandas as pd



        DEBUG_RUN=True


        ts1.run(debug_mode=DEBUG_RUN,force_update=True)
        ts2.run(debug_mode=DEBUG_RUN,force_update=True)


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



class TestTimeSerieLocalLake(unittest.TestCase):
    def test_feature(self):




        import os
        import pandas as pd
        from mainsequence.tdag.time_series.update.scheduler import (
         set_data_source
        )
        from mainsequence.tdag_client import DynamicTableDataSource
        pickle_storage_path = ogm.pickle_storage_path
        data_source = DynamicTableDataSource.get_or_create_local_data_source(datalake_name="Test Lake")
        with set_data_source(pod_source=data_source, tdag_detached=True) as new_source:
            ts1, ts2 = build_test_feature()
            df_1 = ts1.run_local_update(None)
            df_2 = ts2.get_df_greater_than(None)

            # should return same data
            pd.testing.assert_frame_equal(df_1, df_2)

            # filter by update symmbols should be different
            asset_symbols = [c.unique_identifier for c in ts1.asset_list]
            df_1 = ts1.get_df_greater_than(None, asset_symbols=[c.internal_id for c in ts1.asset_list])
            self.assertListEqual(df_1.index.get_level_values("asset_symbol").unique().tolist(), asset_symbols)

            asset_symbols = [c.unique_identifier for c in ts2.asset_list]
            df_2 = ts1.get_df_greater_than(None, asset_symbols=asset_symbols)
            self.assertListEqual(df_2.index.get_level_values("asset_symbol").unique().tolist(), asset_symbols)




