import pytz
import pandas as pd
import datetime
import numpy as np
np.NaN = np.nan
import dotenv
dotenv.load_dotenv('../../.env')

from mainsequence.tdag import TimeSerie, ModelList, APITimeSerie
from mainsequence.client.models_tdag import DataUpdates
from mainsequence.client.models_vam import Asset
from mainsequence.client.models_helpers import MarketsTimeSeriesDetails
from mainsequence.client import MARKETS_CONSTANTS



class PricesFromApi(TimeSerie):
    """
    A basic time series example tracking BTC and ETH price updates.

    Simulation periods:
      - If no update statistics are provided, simulate data from 30 days before now
        until 20 days before now.
      - Otherwise, simulate data per asset from one hour after its last update until
        yesterday at midnight (UTC).
    """
    OFFSET_START = datetime.datetime(2018, 1, 1, tzinfo=pytz.utc)
    CPUS=1
    GPUS=0

    @TimeSerie._post_init_routines()
    def __init__(self,asset_list:ModelList, *args, **kwargs):
        """
        Initialize the SimpleCryptoFeature time series.

        Args:
            asset_list (ModelList): List of asset objects.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.asset_list = asset_list
        ts_details=MarketsTimeSeriesDetails.get(unique_identifier="alpaca_1d")
        self.prices_ts=APITimeSerie(data_source_id=ts_details.related_local_time_serie.remote_table.data_source,
                                    local_hash_id=ts_details.related_local_time_serie.local_hash_id
                                    )
        super().__init__(*args, **kwargs)


    def update(self, update_statistics: DataUpdates)->pd.DataFrame:

        self.prices_ts.get_df_between_dates()



        return data

if __name__ == "__main__":
    assets = Asset.filter(symbol=["NVDA","MSFT"], execution_venue__symbol=MARKETS_CONSTANTS.ALPACA_EV_SYMBOL)
    ts = PricesFromApi(asset_list=assets)
    ts.run(debug_mode=True, force_update=True)