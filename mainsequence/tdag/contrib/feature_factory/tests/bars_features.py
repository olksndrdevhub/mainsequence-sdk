from mainsequence.vam_client import Asset

from mainsequence.tdag.time_series.update import rebuild_and_update_from_source
import unittest





class TestTimeSerie(unittest.TestCase):
    LIST_OF_FEATURES = [{"RealizedVolatilityContinuous": {"rolling_window": 100, "target_column": "close"}},
                     {"RSI": {"rolling_window": 100,"target_column":"close"}},
                      {"VolumeRatio":{"target_column":"volume","rolling_window":1000,"numerator_window":100}},
                         {"VolumeRatio": {"target_column": "trades", "rolling_window": 1000, "numerator_window": 100}},
                        {"VolumeRatio": {"target_column": "dollar", "rolling_window": 1000, "numerator_window": 100}},
                        {"ReturnFromMovingAverage": {"numerator_column": "close", "rolling_window": 90, "denominator_column": "vwap"}},
                        {"RogersSatchelVol": {"rolling_window": 100}},

    ]
    DICT_OF_FEATURES = {0: {"RealizedVolatilityContinuous": {"rolling_window": 100, "target_column": "close"}},
                        1: {"RSI": {"rolling_window": 100, "target_column": "close"}},
                        2: {"VolumeRatio": {"target_column": "volume", "rolling_window": 1000,
                                            "numerator_window": 100}},
                        3: {"VolumeRatio": {"target_column": "trades", "rolling_window": 1000,
                                            "numerator_window": 100}},
                        4: {"VolumeRatio": {"target_column": "dollar", "rolling_window": 1000,
                                            "numerator_window": 100}},
                        5: {"ReturnFromMovingAverage": {"numerator_column": "close", "rolling_window": 90,
                                                        "denominator_column": "vwap"}},
                        6: {"RogersSatchelVol": {"rolling_window": 100}},

                        }
    # UNIQUE_FEATURE=[{"VolumeHistogram": {"rolling_window": 100,"obs_step":1}}]
    UNIQUE_FEATURE=[{"RollingDollarValueBars": {"rolling_window": (60//1)*24*30,"window_to_accumulate":(60//1)*24,
                                                "only_volume":False,
                                                "percent_of_window_to_accumulate":.1}}]


    CROSS=[{"RollingResidualAlphaBeta": {"rolling_window": int(1 * 60 * 8), "target_column": "close"}},]


    def test_feature_debug_aggregator(self):
        return None
        # clean_broken_updates_in_db()
        # clean_broken_ts_from_db()



        asset =Asset.get(symbol = "BTC")
        # ts=AssetBarFeatAggWrap(local_data_path=ogm.temp_folder, asset=asset, source="binance",
        #
        #                                bar_frequency_id="1min", intraday_bar_interpolation_rule="ffill",
        #                                upsample_frequency_id="15min",
        #                                dict_of_features=self.DICT_OF_FEATURES,
        #                               )
        ts=AssetBarFeatureAggregator( asset=asset, source="binance",

                                       bar_frequency_id="1min", intraday_bar_interpolation_rule="ffill",
                                       upsample_frequency_id="60min",
                                       features=[self.DICT_OF_FEATURES[0]],
                                      )

        ts.set_relation_tree()  # force set relation tree ans ts is not build from 0
        ts.update_local(update_tree=False, )

    def test_compound_feature_ts(self):
        from mainsequence.tdag.contrib.feature_factory.bar_time_series import CompCrossAssetFeat
        from mainsequence.tdag.time_series import TimeSerie
        btc,_ = Asset.filter(symbol="BTC",execution_venue__symbol="binance",currency="USDT")
        eth,_ = Asset.filter(symbol="ETH",execution_venue__symbol="binance",currency="USDT")
        cash_asset,_ = Asset.filter(symbol="USDT", currency="USD",execution_venue__symbol="binance")
        asset_list=[btc[0],eth[0]]
        # ts=CompoundAssetFeat(asset_list=asset_list,source="binance",
        #                                bar_frequency_id="1min", intraday_bar_interpolation_rule="ffill",
        #                                upsample_frequency_id="1min",
        #                      feature=self.DICT_OF_FEATURES[0]
        #
        #
        #                      )
        portfolio_kwargs = TimeSerieConfigKwargs(asset_list=asset_list,
                                                 cash_asset=cash_asset[0],
                                                 cash_balance=.02,
                                                 source="binance",
                                                 rebalance_rule_config={"last_weekday_of_month": {"target_day": 4},
                                                     "at_time_intervals": {"time_intervals": ["12:00", "10:00"]}},
                                                 class_extra_kwargs=dict(limit_assets=None, exclude_stablecoins=True, ),
                                                 class_name="CoinMktCapBmrkWts",
                                                 )
        port_ts=TimeSerie.rebuild_from_configuration(hash_id="rebalancerportbarsupsample_b44a2bcb6d94a273039d653438bf1043")
        ts=CompCrossAssetFeat(asset_y_list=asset_list,source="binance",bar_frequency_id="1min",intraday_bar_interpolation_rule="ffill",
                              upsample_frequency_id="1min",feature=self.CROSS[0],init_meta={"asset_x_ts":port_ts},
                              portfolio_x_kwargs=portfolio_kwargs,portfolio_x_name="bmrk",
                              )

    def test_cross_features_ts(self):

        return None

        REBALANCE_CONFIG = {"last_weekday_of_month": {"target_day": 4},
                            "at_time_intervals": {"time_intervals": ["12:00", "10:00"]}}

        btc =Asset.get(symbol = "BTC")
        eth= Asset.get(symbol = "ETH")
        cash_asset = Asset.get(symbol = "USDT", currency = "usd")
        portfolio_kwargs = dict(asset_list=[btc,eth],
                                                 cash_asset=cash_asset, cash_balance=.05, limit_assets=None,
                                                 exclude_stablecoins=True,
                                                 source="binance", rebalance_rule_config=REBALANCE_CONFIG
                                                 )
        ts=CrossBarFeatureAgg(asset_y=btc,portfolio_y_type=None,portfolio_y_kwargs=None,portfolio_y_name=None,
                                     asset_x=None,portfolio_x_type="coinmarketcap_benchmark",portfolio_x_kwargs=portfolio_kwargs, portfolio_x_name="btcethb",
                                     bar_frequency_id="1min",source="binance",intraday_bar_interpolation_rule="ffill",
                                     upsample_frequency_id="15min",features=[{"AlphaRealizedVolatility": {"rolling_window": 100, "target_column": "close"}}],

                                    )
        ts.set_relation_tree()  # force set relation tree ans ts is not build from 0
        ts = rebuild_and_update_from_source(serie_data_folder=ts.data_folder,
                                            update_tree=True,  update_tree_kwargs={"DEBUG": True})

    def test_feature_factory_benchmark(self):
        return None
        REBALANCE_CONFIG = {"last_weekday_of_month": {"target_day": 4},
                            "at_time_intervals": {"time_intervals": ["12:00", "10:00"]}}

        asset = s.query(Asset).filter(Asset.symbol == "BTC").one()
        cash_asset= s.query(Asset).filter(Asset.symbol == "USDT",Asset.currency=="usd").one()
        portfolio_kwargs=TimeSerieConfigKwargs(asset_list=[asset],
                                               cash_asset=cash_asset,cash_balance=.05,limit_assets=None,
                                               exclude_stablecoins=True,
                                               source="binance",rebalance_rule_config=REBALANCE_CONFIG
        )
        ts=PortfolioBarFeatureAgg( asset=asset, source="binance",
                                       bar_frequency_id="1min", intraday_bar_interpolation_rule="ffill",
                                       upsample_frequency_id="15min",features=[self.LIST_OF_FEATURES[0]],

                                              portfolio_type="coinmarketcap_benchmark",
                                              portfolio_kwargs=portfolio_kwargs
                                              )
        ts.set_relation_tree()  # force set relation tree ans ts is not build from 0
        ts=rebuild_and_update_from_source(serie_data_folder=ts.data_folder,
            update_tree=False,update_tree_kwargs={"DEBUG":True})



    def test_feature_factory(self):
        return None
        clean_broken_updates_in_db()
        clean_broken_ts_from_db()


        asset = s.query(Asset).filter(Asset.symbol == "BTC").one()

        ts = AssetBarFeatureAggregator(asset=asset, source="binance",
                                       bar_frequency_id="1min", intraday_bar_interpolation_rule="ffill",
                                       upsample_frequency_id="1min",
                                       features=self.UNIQUE_FEATURE,
                                      )
        ts.set_relation_tree() #force set relation tree ans ts is not build from 0
        ts.update_local(update_tree=False,)


        #only for volume
        from mainsequence.tdag.contrib.feature_factory.bar_features import RollingDollarValueBars
        sequences=RollingDollarValueBars.get_historical_sequences(ts.pandas_df,sequence_length=3,target_column="close")

        a=5