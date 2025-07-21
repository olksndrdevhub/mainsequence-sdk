"""
================================================================================
Instructions for Implementing a `mainsequence.tdag.TimeSerie` Subclass
================================================================================

This file serves as a reference for creating custom time series. To create a new,
functioning time series, you must subclass `TimeSerie` and implement its abstract
methods.

---
### 1. Code Style and Documentation Requirements (IMPORTANT)

All generated code MUST adhere to the following quality standards:

-   **Docstrings**: Every class and method **MUST** have a clear, concise docstring
    explaining its purpose. Methods should include sections for arguments (`Args:`)
    and return values (`Returns:`).
-   **Type Hinting**: All function and method signatures **MUST** use Python type hints
    for all parameters and return values (e.g., `def my_method(self, assets: List[Asset]) -> pd.DataFrame:`).
-   **Inline Comments**: Complex or non-obvious lines of code **MUST** be explained with
    inline comments (`# ...`). Explain the "why" behind the code, not just the "what".
-   **Readability**: Keep functions focused on a single responsibility. Break down
    long, complex methods into smaller, private helper methods where appropriate.

---
### 2. The `__init__` Method
The constructor defines the unique identity of the time series and its dependencies.

-   **Uniqueness**: The arguments passed to `__init__` are hashed to create a unique ID.
    Instantiating a class with the same arguments will always result in the same time series object.
-   **Dependencies**: To declare a dependency on another `TimeSerie`, you **MUST** instantiate it
    within `__init__` and assign it to an instance attribute (e.g., `self.my_dependency = OtherTimeSerie()`).
    The framework will **automatically scan all instance attributes** to find these dependencies
    and build the execution graph.
-   **Special Arguments**: The following arguments are special and are **NOT** included in the hash:
    - `init_meta`: For temporary, runtime-only options.
    - `build_meta_data`: For controlling backend table creation.
    - `local_kwargs_to_ignore`: A list of argument names to exclude from the *local* hash.

---
### 3. The `update` Method (Required)
This is the core method containing the business logic for generating data.

-   **Purpose**: **MUST** be implemented. Its job is to fetch, calculate, or generate new data points.
-   **Input (`update_statistics`)**: This object tells you the timestamp of the last successful
    update for each asset, allowing you to fetch data incrementally.
    -   **Asset Universe**: When looping through assets, you **MUST** iterate over `update_statistics.asset_list`.
        This ensures consistency with the assets prepared for the current run.
    -   **Timestamps**: Use `update_statistics[unique_id]` to get the latest timestamp for a specific asset.

-   **DataFrame Requirements**: The method **MUST** return a pandas DataFrame that adheres to the
    following strict format:
    -   **Index**: Must be a `pandas.MultiIndex` with the names `("time_index", "unique_identifier")`.
    -   **`time_index` Level**: Must contain `datetime.datetime` objects with UTC timezone (`tzinfo=pytz.utc`).
    -   **Column Names**: All column names **MUST** be lowercase and **have a maximum length of 63 characters**.
    -   **Column Values**: No column (other than the index) may contain Python `datetime` objects.
        If you need to store a timestamp in a column, it **MUST** be converted to an integer
        (e.g., a UNIX epoch timestamp).

---
### 4. Optional Hooks (Override as Needed)
These methods have default behaviors but can be overridden for customization.

-   `get_table_metadata(self) -> Optional[ms_client.TableMetaData]`:
    -   **Purpose**: [OPTIONAL] Implement this hook to add human identifier and description to a table.
    -   **Returns**: A configured `ms_client.TableMetaData` object with the following key attributes:
        -   `identifier` (str): A human-readable, public ID for your time series (e.g., "my_daily_rsi_feature"). **Must be unique across all tables**
        -   `data_frequency_id` (ms_client.DataFrequency): The data frequency (e.g., `ms_client.DataFrequency.one_d`).
        -   `description` (str): A user-friendly description of the table.

-   `get_asset_list(self) -> List[Asset]`:
    -   **Purpose**: [OPTIONAL] Implement to dynamically define the list of assets this time series should process during an update. Useful for fetching all assets in a category.
    -   **Returns**: A list of `Asset` objects.

-   `get_column_metadata(self) -> List[ColumnMetaData]`:
    -   **Purpose**: [OPTIONAL] Implement to provide rich descriptions for your data columns. This metadata is used in documentation and user interfaces.
    -   **Returns**: A list of `ColumnMetaData` objects.

-   `_run_post_update_routines(self, ...)`:
    -   **Purpose**: [OPTIONAL] Implement to run custom logic *after* an update is finished. Useful for logging, cleanup, or registering the time series in an external system.
"""
import json

# Imports should be at the top of the file
import numpy as np
np.NaN = np.nan # Fix for a pandas-ta compatibility issue
from mainsequence.tdag import TimeSerie
from mainsequence.client.models_tdag import UpdateStatistics, ColumnMetaData
import mainsequence.client as ms_client
from typing import Union, Optional, List, Dict
import datetime
import pytz
import pandas as pd

class SimulatedPricesManager:

    def __init__(self,owner:TimeSerie):
        self.owner = owner

    @staticmethod
    def _get_last_price(obs_df: pd.DataFrame, unique_id: str, fallback: float) -> float:
        """
        Helper method to retrieve the last price for a given unique_id or return 'fallback'
        if unavailable.

        Args:
            obs_df (pd.DataFrame): A DataFrame with multi-index (time_index, unique_identifier).
            unique_id (str): Asset identifier to look up.
            fallback (float): Value to return if the last price cannot be retrieved.

        Returns:
            float: Last observed price or the fallback value.
        """
        # If there's no historical data at all, return fallback immediately
        if obs_df is None:
            return fallback

        # Try to slice for this asset and get the last 'feature_1' value
        try:
            slice_df = obs_df.xs(unique_id, level="unique_identifier")["feature_1"]
            return slice_df.iloc[-1]
        except (KeyError, IndexError):
            # KeyError if unique_id not present, IndexError if slice is empty
            return fallback
    def update(self, update_statistics: UpdateStatistics)->pd.DataFrame:
        """
        Mocks price updates for assets with stochastic lognormal returns.

        - If update_statistics is empty, simulate data from (now - 30 days) to (now - 20 days),
          floored to the nearest minute.
        - Otherwise, for each unique identifier in update_statistics.update_statistics,
          simulate new data starting one hour after its last update until yesterday at midnight (UTC),
          using the last observed price as the starting price. The last observation is not duplicated.

        Returns:
            pd.DataFrame: A DataFrame with a multi-index (time_index, unique_identifier)
                          and a single column 'feature_1' containing the simulated prices.
        """
        import numpy as np

        initial_price = 100.0
        mu = 0.0  # drift component for lognormal returns
        sigma = 0.01  # volatility component for lognormal returns

        df_list = []

        # Get the latest historical observations; assumed to be a DataFrame with a multi-index:
        # (time_index, unique_identifier) and a column "feature_1" for the last observed price.
        last_observation = self.owner.get_last_observation()
        # Define simulation end: yesterday at midnight (UTC)
        yesterday_midnight = datetime.datetime.now(pytz.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) - datetime.timedelta(days=1)
        # Loop over each unique identifier and its last update timestamp.
        for unique_id, last_update in update_statistics.update_statistics.items():
            # Simulation starts one hour after the last update.
            start_time = last_update + datetime.timedelta(hours=1)
            if start_time > yesterday_midnight:
                continue  # Skip if no simulation period is available.
            time_range = pd.date_range(start=start_time, end=yesterday_midnight, freq='H')
            if len(time_range) == 0:
                continue
            # Use the last observed price for the asset as the starting price.
                # Get or fallback to initial_price
            last_price = self._get_last_price(
                obs_df=last_observation,
                unique_id=unique_id,
                fallback=initial_price
            )

            random_returns = np.random.lognormal(mean=mu, sigma=sigma, size=len(time_range))
            simulated_prices = last_price * np.cumprod(random_returns)
            df_asset = pd.DataFrame({unique_id: simulated_prices}, index=time_range)
            df_list.append(df_asset)

        if df_list:
            data = pd.concat(df_list, axis=1)
        else:
            return pd.DataFrame()

        # Reshape the DataFrame into long format with a multi-index.
        data.index.name = "time_index"
        data = data.melt(ignore_index=False, var_name="unique_identifier", value_name="feature_1")
        data = data.set_index("unique_identifier", append=True)
        return data



    def get_column_metadata(self):
        from mainsequence.client.models_tdag import ColumnMetaData
        columns_metadata = [ColumnMetaData(column_name="feature_1",
                                           dtype="float",
                                           label="Feature 1",
                                           description=(
                                               "Feature calculated like XXX "
                                           )
                                           ),


                            ]
        return columns_metadata


class SingleIndexTS(TimeSerie):
    """
       A simple time series with a single index, generating random numbers daily.
       This serves as a basic, non-asset-based example.
       """
    OFFSET_START = datetime.datetime(2024, 1, 1, tzinfo=pytz.utc)

    def update(self, update_statistics: ms_client.UpdateStatistics):
        today_utc = datetime.datetime.now(tz=pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)

        if update_statistics.last_observation.empty:
            start_date = self.OFFSET_START
        else:
            last_time = update_statistics.last_observation.index.max()
            start_date = last_time + datetime.timedelta(days=1)

        if start_date > today_utc:
            return pd.DataFrame()

        num_days = (today_utc - start_date).days + 1
        time_index = pd.date_range(start=start_date, periods=num_days, freq='D', tz=pytz.utc)
        random_values = np.random.rand(num_days)

        df = pd.DataFrame({'random_number': random_values}, index=time_index)
        df.index.name = 'time_index'
        return df

    def get_column_metadata(self):
        """
        Add MetaData information to the TimeSerie Table
        Returns:

        """
        from mainsequence.client.models_tdag import ColumnMetaData
        columns_metadata = [ColumnMetaData(column_name="random_number",
                                           dtype="float",
                                           label="Random Number ",
                                           description=(
                                               "A random number with no meaning whatsoever"
                                           )
                                           ),

                            ]
        return columns_metadata


    def get_table_metadata(self,update_statistics:ms_client.UpdateStatistics)->ms_client.TableMetaData:
        """


        """
        MARKET_TIME_SERIES_UNIQUE_IDENTIFIER = "simple_time_serie_example"
        mts=ms_client.TableMetaData(unique_identifier=MARKET_TIME_SERIES_UNIQUE_IDENTIFIER,
                                               data_frequency_id=ms_client.DataFrequency.one_d,
                                               description="This is a simulated prices time serie from asset category",
                                               )


        return mts



class SimulatedPrices(TimeSerie):
    """
    Simulates price updates for a specific list of assets provided at initialization.
    """
    OFFSET_START = datetime.datetime(2024, 1, 1, tzinfo=pytz.utc)

    def __init__(self, asset_list: List,
                 local_kwargs_to_ignore=["asset_list"],

                 *args, **kwargs):
        """
        Initialize the SimpleCryptoFeature time series.

        Args:
            asset_list (ModelList): List of asset objects.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.asset_list = asset_list
        self.asset_symbols_filter = [a.unique_identifier for a in asset_list]
        super().__init__(local_kwargs_to_ignore=local_kwargs_to_ignore,*args, **kwargs)

    def update(self,update_statistics:ms_client.UpdateStatistics):
        update_manager=SimulatedPricesManager(self)
        df=update_manager.update(update_statistics)
        return df
    def get_column_metadata(self):
        """
        Add MetaData information to the TimeSerie Table
        Returns:

        """
        from mainsequence.client.models_tdag import ColumnMetaData
        columns_metadata = [ColumnMetaData(column_name="feature_1",
                                           dtype="float",
                                           label="Feature 1",
                                           description=(
                                               "Simulated Feature 1"
                                           )
                                           ),


                            ]
        return columns_metadata

    def get_table_metadata(self,update_statistics:UpdateStatistics)->ms_client.TableMetaData:
        """
        REturns the market time serie unique identifier, assets to append , or asset to overwrite
        Returns:

        """
        MARKET_TIME_SERIES_UNIQUE_IDENTIFIER = "simulated_prices_from_category"
        mts=ms_client.TableMetaData(identifier=MARKET_TIME_SERIES_UNIQUE_IDENTIFIER,
                                               data_frequency_id=ms_client.DataFrequency.one_d,
                                               description="This is a simulated prices time serie from asset category",
                                               )


        return mts

class CategorySimulatedPrices(TimeSerie):
    """
        Simulates price updates for all assets belonging to a specified category.
        This demonstrates using a hook (`get_asset_list`) to dynamically define the asset universe.
        It also shows a dependency on another TimeSerie (`SingleIndexTS`).
        """
    OFFSET_START = datetime.datetime(2024, 1, 1, tzinfo=pytz.utc)

    def __init__(self, asset_category_id:str,
                 local_kwargs_to_ignore=["asset_category_id"],
                 *args, **kwargs):
        """
        Initialize the SimpleCryptoFeature time series.

        Args:
            asset_list (ModelList): List of asset objects.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.asset_category_id = asset_category_id

        #this time serie has a dependency to include dependencies is just enough to declared them in the init method
        self.simple_ts=SingleIndexTS()

        super().__init__(local_kwargs_to_ignore=local_kwargs_to_ignore,*args, **kwargs)

    def get_asset_list(self):
        """[Hook] Dynamically fetches the list of assets from a category."""

        asset_category=ms_client.AssetCategory.get(unique_identifier=self.asset_category_id)
        asset_list=ms_client.Asset.filter(id__in=asset_category.assets)
        return asset_list
    def update(self,update_statistics:ms_client.UpdateStatistics):
        """
              Simulates prices and then adds a random number from its dependency,
              demonstrating how to combine data from multiple sources.
              """
        update_manager=SimulatedPricesManager(self)
        data=update_manager.update(update_statistics)
        if data.empty:
            return pd.DataFrame()
        #an example of a dependencies calls
        historical_random_data=self.simple_ts.get_df_between_dates(start_date=update_statistics.max_time_index_value)
        number=historical_random_data["random_number"].iloc[-1]
        return data+number

    def get_column_metadata(self):
        """
        Add MetaData information to the TimeSerie Table
        Returns:

        """
        from mainsequence.client.models_tdag import ColumnMetaData
        columns_metadata = [ColumnMetaData(column_name="feature_1",
                                           dtype="float",
                                           label="Feature 1",
                                           description=(
                                               "Simulated Feature 1"
                                           )
                                           ),

                            ]
        return columns_metadata

    def get_table_metadata(self,update_statistics)->ms_client.TableMetaData:
        """

        """
        MARKET_TIME_SERIES_UNIQUE_IDENTIFIER = "simulated_prices_from_category"
        meta=ms_client.TableMetaData(identifier=MARKET_TIME_SERIES_UNIQUE_IDENTIFIER,
                                               data_frequency_id=ms_client.DataFrequency.one_d,
                                               description="This is a simulated prices time serie from asset category",
                                               )


        return meta




# Mocking UpdateStatistics and Running the Test


class TAFeature(TimeSerie):
    """
       A derived time series that calculates a technical analysis feature from another price series.
       """

    def __init__(self, asset_list: List[ms_client.Asset], ta_feature_config:List[dict],
                 local_kwargs_to_ignore=["asset_list","ta_feature_config"],
                 *args, **kwargs):
        """
        Initialize the TA feature time series.

        Args:
            asset_list (ModelList): List of asset objects.
            ta_feature (str): Name of the TA indicator to calculate ("SMA", "RSI", "EMA", etc.).
            ta_length (int, optional): Lookback period for the indicator. Default is 14.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.asset_list = asset_list
        self.asset_symbols_filter = [a.unique_identifier for a in asset_list]
        self.ta_feature_config = ta_feature_config


        # Instantiate the base price simulation.
        # Replace this import with the actual location of SimulatedCryptoPrices if needed.

        self.prices_time_serie = SimulatedPrices(asset_list=asset_list,
                                                 local_kwargs_to_ignore=local_kwargs_to_ignore,
                                                 *args, **kwargs)
        super().__init__(local_kwargs_to_ignore=local_kwargs_to_ignore,*args, **kwargs)

    def update(self, update_statistics: UpdateStatistics) -> pd.DataFrame:
        """
        [Core Logic] Calculates a technical analysis feature based on a dependent price series.

        This method serves as a template for creating a derived time series. It follows a
        three-step process:
        1.  **Prepare a request**: It determines the required historical data for each asset,
            ensuring a sufficient lookback window for the TA calculation.
        2.  **Fetch data**: It retrieves the necessary price data from its dependency
            (self.prices_time_serie) using a single, efficient call.
        3.  **Calculate and reshape**: It transforms the data, applies the specified TA
            indicator, and formats the result into the required long-format DataFrame.

        Args:
            update_statistics: An object containing the last update timestamp for each asset.
                               This is used to calculate the precise data range needed.

        Returns:
            A DataFrame with a ("time_index", "unique_identifier") multi-index and a
            single column containing the calculated TA feature values.
        """
        SIMULATION_OFFSET_START = datetime.timedelta(days=50)
        import pandas_ta as ta
        import pytz
        #when using more than 2 indices in a multiindex table it is extremely importnat to make all the filters
        #accross all levels to keep consistency in the update
        update_statistics.filter_assets_by_level(
                                                 level=2,
                                                 filters=[json.dumps(c) for c in self.ta_feature_config])

        # --- Step 1: Prepare the Data Request ---
        # We need to fetch not just new data, but also a "lookback" window of older data
        # to ensure the TA indicator can be calculated correctly from the very first new point.
        rolling_window = datetime.timedelta(days=np.max([a["length"] for a in self.ta_feature_config]).item()+1)  #Fetch the max of the required features
        asset_range_descriptor = {}

        for asset in update_statistics.asset_list:
            # For each asset, find its last update time. If it's a new asset,
            # fall back to the global offset defined on the class.
            last_update = update_statistics.get_asset_earliest_multiindex_update(asset)- SIMULATION_OFFSET_START


            # The start date for our data request is the last update time minus the lookback window.
            start_date_for_asset = last_update - rolling_window
            asset_range_descriptor[asset.unique_identifier] = {
                "start_date": start_date_for_asset,
                "start_date_operand": ">="  # Use ">=" to include the start of the window
            }

        # --- Step 2: Fetch Data From Dependency ---
        # This is a key pattern for derived features. We use a data access method from the base
        # TimeSerie class (via the DataAccessMixin) to query our dependency.
        #
        # `get_ranged_data_per_asset` is highly efficient because it takes a dictionary
        # that specifies a *different* start date for each asset, allowing us to get all
        # the data we need in a single call.
        prices_df = self.prices_time_serie.get_ranged_data_per_asset( range_descriptor=asset_range_descriptor  )

        if prices_df.empty:
            self.logger.warning("Base price data was empty for the requested range.")
            return pd.DataFrame()

        # --- Step 3: Calculate the TA Feature and Reshape the Data ---
        # The pandas-ta library expects data in a "wide" format, where each column
        # is a different time series. We pivot the data to create this structure.
        prices_pivot = (
            prices_df
            .reset_index()
            .pivot(index="time_index", columns="unique_identifier", values="feature_1")
        )



        all_features=[]
        for feature_config in self.ta_feature_config:
            feature_name=json.dumps(feature_config)
            feature_kind=feature_config["kind"].lower()
            feature_config.pop("kind")
            func = getattr(ta, feature_kind)
            ta_results = prices_pivot.apply(lambda col: func(col, **feature_config))

            # The result needs to be cleaned and reshaped back to the required "long" format.
            ta_long = (
                ta_results
                .dropna(how="all")  # Drop rows at the start where the indicator is all NaN
                .reset_index()
                .melt(id_vars="time_index", var_name="unique_identifier", value_name="feature_value")
                .dropna(subset=["feature_value"])  # Drop any remaining individual NaNs
                .set_index(["time_index", "unique_identifier"])
            )
            # pull out the existing levels as arrays
            times = ta_long.index.get_level_values('time_index')
            assets = ta_long.index.get_level_values('unique_identifier')

            # make a constant array of your feature-name
            feat = [feature_name] * len(ta_long)

            midx = pd.MultiIndex.from_arrays(
                [times, assets, feat],
                names=['time_index', 'unique_identifier', 'feature_name']
            )
            ta_long.index = midx

            all_features.append(ta_long)

        all_features=pd.concat(all_features,axis=0)

        return all_features




def test_simulated_prices():
    from mainsequence.client import Asset
    from mainsequence.client import MARKETS_CONSTANTS

    assets = Asset.filter(
        ticker__in=["NVDA", "APPL"],

        execution_venue__symbol=MARKETS_CONSTANTS.MAIN_SEQUENCE_EV
    )
    ts = SimulatedPrices(asset_list=assets)
    ts_2=CategorySimulatedPrices(asset_category_id="external_magnificent_7")

    ts_0=SingleIndexTS()
    # ts_0.run(debug_mode=True,force_update=True)
    # ms_client.SessionDataSource.set_local_db() #run on duck
    # ts.run(debug_mode=True,force_update=True)
    # ts_2.run(debug_mode=True,force_update=True)

    ts = TAFeature(asset_list=assets, ta_feature_config=[dict(kind="SMA", length=28),
                                                         dict(kind="SMA", length=21),
                                                         dict(kind="RSI", length=21)

                                                         ]

                   )

    ts.run(debug_mode=True,
           update_tree=True,
           force_update=True,
           )


if __name__ == "__main__":
    test_simulated_prices()

