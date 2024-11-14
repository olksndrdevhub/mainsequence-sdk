

from mainsequence.tdag.time_series.utils import  dispatch_bar_time_serie
from mainsequence.tdag import  AssetORM
from mainsequence.tdag.contrib.feature_factory.bar_time_series import interpolate_intraday_bars



orm = AssetORM(assets_db="DEEPMACHINE")
asset_list = []
asset_tickers = ["SPY", "QQQ"]

for ticker in asset_tickers:
    asset = s.query(Asset).filter(Asset.symbol == ticker).one()
    asset_list.append(asset)


asset=asset_list[0]
time_serie=dispatch_bar_time_serie(source="polygon",asset=asset_list[0],frequency_id="1min")
# time_serie.update_local(update_tree=False)
time_serie.get_ts_as_pandas()
day_start,day_ends=asset.trading_hours

#map columns to right names


bars_df=time_serie.pandas_df[["close","volume","high","low","open","vwap","trades"]]
a=interpolate_intraday_bars(bars_df=time_serie.pandas_df,day_start=day_start,day_end=day_ends,
                            interpolation_rule="ffill",bars_frequency_min=1)

a=5