import unittest

from mainsequence.tdag.time_series.update import rebuild_and_update_from_source
from mainsequence.tdag import AssetORM

from mainsequence.tdag.contrib.benchmark_factory import CoinMktCapBmrkBars

assets_db= AssetORM.test_server_name()



def build_benchmark_ts_by_symbols(symbol_list, cash_asset_symbol, rebalance_config, cash_balance, assets_db,
                                  frequency_id,exclude_stablecoins):


    all_assets = s.query(Asset).filter(Asset.symbol.in_(symbol_list), Asset.currency == "usdt").all()
    cash_asset = s.query(Asset).filter(Asset.symbol == cash_asset_symbol, Asset.currency == "usd").one()

    # ts=MarketCapBenchmarkWeights(asset_list=all_assets,cash_balance=cash_balance,
    #                               cash_asset=cash_asset,frequency_id=frequency_id,
    #                               source="binance",rebalance_rule_config=rebalance_config,
    #                                    local_data_path=BASEDIR+"/data")
    ts = CoinMktCapBmrkBars(asset_list=all_assets, cash_balance=cash_balance,
                                cash_asset=cash_asset, frequency_id=frequency_id,limit_assets=None,
                                source="binance", rebalance_rule_config=rebalance_config,
                               )


    return ts

class TestTimeSerie(unittest.TestCase):



    def test_market_cap_benchmark_weights(self):
        REBALANCE_CONFIG = {"last_weekday_of_month": {"target_day": 4},
                            "at_time_intervals": {"time_intervals": ["12:00", "10:00"]}}

        # build Benchmark
        benchmark_ts = build_benchmark_ts_by_symbols(symbol_list=["BTC", "ETH"], cash_asset_symbol="USDT", exclude_stablecoins=True,
                                                     cash_balance=.02,assets_db=assets_db,frequency_id="1min",
                                                     rebalance_config=REBALANCE_CONFIG)
        rebuild_and_update_from_source(assets_db=assets_db, update_tree=True, update_tree_kwargs={"DEBUG": True},

                                       serie_data_folder=benchmark_ts.data_folder)