import pytz
import pandas as pd
import datetime
import numpy as np
np.NaN = np.nan
import dotenv
dotenv.load_dotenv('../../.env')

from mainsequence.tdag import TimeSerie, ModelList
from mainsequence.client.models_tdag import DataUpdates
from mainsequence.client.models_vam import Asset
from mainsequence.client import MARKETS_CONSTANTS


def data_from_api(asset):
    return pd.DataFrame(index=[datetime.datetime.now().replace(tzinfo=pytz.utc,minute=0,second=0,microsecond=0)],
                        data=[100],columns=["price"]
                        )

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


    @TimeSerie._post_init_routines()
    def __init__(self, *args, **kwargs):
        """
        Initialize the SimpleCryptoFeature time series.

        Args:
            asset_list (ModelList): List of asset objects.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

        super().__init__(*args, **kwargs)


    def update(self, update_statistics: DataUpdates)->pd.DataFrame:

        asset=Asset.filter(ticker="NVDA")
        data=data_from_api(asset)
        return data


interpolated_prices = lambda x: x
calcualte_risk_measurement = lambda x: x

class PricesInterpolation(TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(self, *args, **kwargs):

        self.data_time_series=PricesFromApi(*args, **kwargs)
    def update(self, update_statistics: DataUpdates)->pd.DataFrame:
        prices=interpolated_prices(self.data_time_series)
        return prices

class RiskMeasurement(TimeSerie):
    @TimeSerie._post_init_routines()
    def __init__(self, *args, **kwargs):
        self.prices_time_series = PricesInterpolation(*args, **kwargs)

    def update(self, update_statistics: DataUpdates) -> pd.DataFrame:
        prices = calcualte_risk_measurement(self.prices_time_series)
        return prices

if __name__ == "__main__":
    risk_measurement = RiskMeasurement()
    risk_measurement.run(debug_mode=True,force_update=True)