import datetime
import pytz
import pandas as pd
from fredapi import Fred
import os
import dotenv

dotenv.load_dotenv('../../.env')  # Load environment variables from .env
from mainsequence.tdag import TimeSerie, ModelList
from mainsequence.tdag_client.models import DataUpdates
from mainsequence.vam_client import Asset,Calendar, ExecutionVenue
from mainsequence import VAM_CONSTANTS

class FREDTimeSerie(TimeSerie):
    """
    Fetches macro/financial data from FRED for each string in fred_symbols,
    where each string is a valid FRED series ID.
    The FRED API key is retrieved from the environment variable `FRED_API_KEY`.
    """

    SIM_OFFSET_START = datetime.timedelta(days=365)  # For example, 1-year fallback

    @TimeSerie._post_init_routines()
    def __init__(self, asset_list: list[str], *args, **kwargs):
        """
        :param fred_symbols: List of FRED series IDs, e.g., ["DGS10", "DEXUSEU", ...]
        """
        self.asset_list = asset_list
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

        # 2) Update the last-update statistics for these symbols,
        #    using a fallback date if no prior data is found
        update_statistics = update_statistics.update_assets(
            self.asset_list, init_fallback_date=now_utc - self.SIM_OFFSET_START
        )

        df_list = []
        for asset in self.asset_list:
            # 3) Retrieve the last update date from update_statistics
            last_update = update_statistics[asset.unique_identifier]  # a datetime object

            # Fetch data from the day after last_update up to now
            observation_start = (last_update + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            observation_end = now_utc.strftime('%Y-%m-%d')

            try:
                series_data = self.fred.get_series(asset.unique_identifier, observation_start, observation_end)
            except Exception as e:
                print(f"Failed to fetch FRED series {asset.unique_identifier}: {e}")
                continue

            if series_data.empty:
                continue

            # 4) Convert the returned Series into a DataFrame
            series_data = series_data.reset_index()
            series_data.columns = ['time_index', 'feature_1']
            series_data['unique_identifier'] = asset.unique_identifier

            df_list.append(series_data)

        if not df_list:
            return pd.DataFrame()

        # 5) Concatenate results & set a MultiIndex of (time_index, unique_identifier)
        data = pd.concat(df_list, axis=0)
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
    dummy_asset_kwargs=dict(

            asset_type=VAM_CONSTANTS.ASSET_TYPE_INDEX,
            can_trade= False,
            calendar= Calendar(name="FredCalendar"),
            execution_venue=ExecutionVenue(symbol="FRED",name="FRED EV")
    )
    assets = [
        Asset(symbol="US10YR", unique_identifier="DGS10",name="US10YR",**dummy_asset_kwargs),
        Asset(symbol="EURUSD_FX", unique_identifier="DEXUSEU",name="EURUSD_FX",**dummy_asset_kwargs),
    ]

    # Instantiate the TimeSerie
    fred_ts = FREDTimeSerie(asset_list=ModelList(assets))

    # Call the update method
    df = fred_ts.update(DataUpdates())
    print(df)


if __name__ == "__main__":
    test_fred_time_serie()
