import datetime
import pytz
import pandas as pd
from fredapi import Fred
import os
import dotenv

dotenv.load_dotenv('../../.env')  # Load environment variables from .env
from mainsequence.tdag import TimeSerie, ModelList
from mainsequence.client import DataUpdates
from mainsequence.client import Asset,Calendar, ExecutionVenue
from mainsequence.client import MARKETS_CONSTANTS

class FREDTimeSerie(TimeSerie):
    """
    Fetches macro/financial data from FRED for each string in fred_symbols,
    where each string is a valid FRED series ID.
    The FRED API key is retrieved from the environment variable `FRED_API_KEY`.
    """

    OFFSET_START = datetime.datetime(1970, 1, 1, tzinfo=pytz.UTC)
    CPUS = 1
    GPUS = 0

    @TimeSerie._post_init_routines()
    def __init__(self, unique_identifiers: list[str],
                 local_kwargs_to_ignore=["unique_identifiers"],

                 *args, **kwargs):
        """
        :param fred_symbols: List of FRED series IDs, e.g., ["DGS10", "DEXUSEU", ...]
        """
        self.unique_identifiers = unique_identifiers
        super().__init__(*args, **kwargs)

    def update(self, update_statistics: DataUpdates):
        """
        Pulls data for each FRED series in fred_symbols.
        Uses update_statistics to track each symbol’s last update date,
        then fetches new data from FRED.

        Returns a DataFrame with a MultiIndex: (time_index, unique_identifier).
        """

        # 1) Load the FRED API key from environment
        fred_api_key = os.environ["FRED_API_KEY"]
        self.fred = Fred(api_key=fred_api_key)

        now_utc = datetime.datetime.now(pytz.utc)


        df_list = []
        for ui in self.self.asset_list:
            # 3) Retrieve the last update date from update_statistics
            last_update = update_statistics[ui]  # a datetime object

            # Fetch data from the day after last_update up to now
            observation_start = (last_update + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            observation_end = now_utc.strftime('%Y-%m-%d')

            try:
                series_data = self.fred.get_series(ui, observation_start, observation_end)
            except Exception as e:
                print(f"Failed to fetch FRED series {ui}: {e}")
                continue

            if series_data.empty:
                continue

            # 4) Convert the returned Series into a DataFrame
            series_data = series_data.reset_index()
            series_data.columns = ['time_index', 'feature_1']
            series_data['unique_identifier'] = ui

            df_list.append(series_data)

        if not df_list:
            return pd.DataFrame()

        # 5) Concatenate results & set a MultiIndex of (time_index, unique_identifier)
        data = pd.concat(df_list, axis=0)
        data["time_index"] = pd.to_datetime(data["time_index"], errors="coerce")  # stays the same if already OK

        #always needs to be UTC
        data["time_index"] = data["time_index"].dt.tz_localize("UTC")

        data = data.set_index(['time_index', 'unique_identifier'])

        return data


# Example Test Script
def test_fred_time_serie():
    """
    Example usage of FREDTimeSerie without needing an asset_list.
    Provide a list of valid FRED series IDs in `fred_symbols`.
    The API key is loaded from environment variable `FRED_API_KEY`.
    """
    # Two example FRED series: with synthethic assets ont in the database



    assets = [
        Asset(
              **{"can_trade": False,
                "execution_venue": ExecutionVenue(name="US Treasury Market",
                                                  symbol="US Treasury Market"
                                                  ),
                "delisted_datetime": None,
                "unique_identifier": "DGS10",
                "real_figi": False,
                "figi": "123456789ASG",
                "composite": "123456789ASG",
                "ticker": "US10YR",
                "security_type": "Treasury Note",
                "security_type_2": "Government Bond",
                "security_market_sector": "Fixed Income",
                "share_class": None,
                "exchange_code": "XUSA",
                "name": "U.S. Treasury Note 10-Year Constant Maturity",
                "main_sequence_share_class": None}

              ),
        Asset(

              **{
                  "can_trade": False,
                  "execution_venue": ExecutionVenue(name="Forex Market",
                                                  symbol="Forex Market"
                                                  ),
                  "delisted_datetime": None,
                  "unique_identifier": "DEXUSEU",
                  "real_figi": False,
                  "figi": "123456789ASD",
                  "composite": "123456789ASD",
                  "ticker": "EURUSD",
                  "security_type": "Currency Pair",
                  "security_type_2": "Foreign Exchange",
                  "security_market_sector": "Forex",
                  "share_class": None,
                  "exchange_code": "XFXM",
                  "name": "Euro / U.S. Dollar Currency Pair",
                  "main_sequence_share_class": None
              }
              ),
    ]
    #Warninig the time series will run but this assets will not exist in the Backend
    # Instantiate the TimeSerie
    fred_ts = FREDTimeSerie(asset_list=ModelList(assets))

    # Call the update method
    fred_ts.run(debug_mode=True, force_update=True)

if __name__ == "__main__":
    test_fred_time_serie()
