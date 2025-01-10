
import os
import pandas as pd
from typing import Union, List
import s3fs
import yaml

import json
from functools import lru_cache
import psutil
import datetime
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow as pa
import pytz
from tqdm import tqdm
import shutil
from mainsequence.tdag_client.utils import set_types_in_table

TIME_PARTITION = "TIME_PARTITION"





def memory_usage_exceeds_limit(max_usage_percentage):
    """
    Checks if the current memory usage exceeds the given percentage of total memory.
    """
    memory_info = psutil.virtual_memory()
    used_memory_percentage = memory_info.percent
    return used_memory_percentage > max_usage_percentage

@lru_cache(maxsize=50)
def read_full_data(file_path, filters=None, use_s3_if_available=False, max_memory_usage=80):
    " Cached access to static datalake file "
    if memory_usage_exceeds_limit(max_memory_usage):
        # Clear cache if memory exceeds the limit
        read_full_data.cache_clear()

    data = read_parquet_from_lake(file_path=file_path, filters=filters, use_s3_if_available=use_s3_if_available)
    data = data.drop(columns=[TIME_PARTITION])
    return data

def read_parquet_from_lake(
        file_path: str,
        use_s3_if_available: bool=False,
        filters: Union[list,None]=None,
        s3_storage_options: Union[dict,None]=None,
):
    extra_kwargs = {'filters': filters} if filters is not None and len(filters) > 0 else {}

    if use_s3_if_available:
        s3_path, storage_options = self._get_storage_options(file_path)
        data = pd.read_parquet(s3_path, engine='pyarrow', storage_options=storage_options, **extra_kwargs)
    else:
        def parse_filters(filters):
            """
            Parse nested filter structure into a PyArrow expression.
            """
            combined_expression = None
            for group in filters:
                group_expression = None
                for column, operator, value in group:
                    field = ds.field(column)
                    # Map operators to PyArrow expressions
                    if operator == "=":
                        expr = field == value
                    elif operator == ">=":
                        expr = field >= value
                    elif operator == "<=":
                        expr = field <= value
                    elif operator == ">":
                        expr = field > value
                    elif operator == "<":
                        expr = field < value
                    elif operator == "in":
                        expr = field.isin(value)
                    else:
                        raise ValueError(f"Unsupported operator: {operator}")

                    group_expression = expr if group_expression is None else group_expression & expr

                combined_expression = group_expression if combined_expression is None else combined_expression | group_expression

            return combined_expression

        filter_expression = parse_filters(filters) if filters else None

        dataset = ds.dataset(file_path, format="parquet", partitioning="hive")
        scanner = ds.Scanner.from_dataset(
            dataset,
            filter=filter_expression,
        )

        data = scanner.to_table().to_pandas()
        data=data.sort_index(level=0)
    return data


def get_extrema_per_asset_symbol_from_sidecars(file_path: str, extrema: str = "max"):
    import operator

    """
    Walk through 'file_path', find each 'summary.json' sidecar file, and:
      - If 'BY_ASSET_VENUE' exists and is non-empty, aggregate those extrema
        values in global_summary (shape {asset_symbol: {venue_symbol: GLOBAL_EXTREMA}}).
      - Independently track the sidecar's __GLOBAL__[extrema] for fallback if needed.

    Args:
        file_path (str): Root directory to walk and find 'summary.json' sidecar files.
        extrema (str, optional): Which value to compute, "max" or "min". Default is "max".

    Returns:
        global_extrema, global_summary

        Where:
            global_extrema: the single highest/lowest time_index across all sidecars
                            (from either BY_ASSET_VENUE or fallback to __GLOBAL__),
                            as a UTC datetime (None if not found).
            global_summary: a dict of shape:
              {
                asset_symbol: {
                  venue_symbol: GLOBAL_EXTREMA_TIME_INDEX_AS_DATETIME
                },
                ...
              }
              or {} if no sidecar had BY_ASSET_VENUE data.
    """
    if extrema not in ("max", "min"):
        raise ValueError("Parameter 'extrema' must be either 'max' or 'min'.")

    # Decide which comparison operator and key to use
    def is_more_extreme(new_val, old_val):
        if extrema == "max":
            return new_val > old_val
        else:
            return new_val < old_val
    sidecar_extrema_values = []  # collects the sidecar-wide fallback extrema (from __GLOBAL__)
    global_summary = {}

    # Recursively walk through all subdirectories
    for dirpath, dirnames, filenames in os.walk(file_path):
        if "summary.json" not in filenames:
            continue  # skip folders without a sidecar

        sidecar_path = os.path.join(dirpath, "summary.json")
        with open(sidecar_path, "r") as f:
            partition_summary = json.load(f)

        # 1) If "BY_ASSET_VENUE" is present and not empty, merge it into global_summary
        if "BY_ASSET_VENUE" in partition_summary and partition_summary["BY_ASSET_VENUE"]:


            by_asset_venue = partition_summary["BY_ASSET_VENUE"]
            for asset_symbol, venues_dict in by_asset_venue.items():
                if asset_symbol not in global_summary:
                    global_summary[asset_symbol] = {}
                for venue_symbol, time_stats in venues_dict.items():
                    # Convert the 'extrema' value from timestamp to datetime
                    # e.g., time_stats["max"] if extrema == "max"; or time_stats["min"] if extrema == "min"
                    timestamp_value = time_stats.get(extrema)
                    if timestamp_value is None:
                        # If the sidecar doesn't have the requested key, skip
                        continue
                    partition_value = datetime.datetime.fromtimestamp(timestamp_value, tz=pytz.utc)

                    current_value = global_summary[asset_symbol].get(venue_symbol)
                    # If there's no current value or partition_value is more extreme
                    # (depending on 'max' or 'min'), update
                    if current_value is None or is_more_extreme(partition_value, current_value):
                        global_summary[asset_symbol][venue_symbol] = partition_value



        # 2) Also gather the sidecar's __GLOBAL__[extrema] for fallback
        if "__GLOBAL__" in partition_summary:
            sidecar_value = partition_summary["__GLOBAL__"].get(extrema, None)
            if sidecar_value is not None:
                sidecar_extrema_values.append(datetime.datetime.fromtimestamp(sidecar_value, tz=pytz.utc))

    # ---------------------------------------------------------------
    # Compute the single global_extrema from the data we have
    # ---------------------------------------------------------------
    global_extrema = None

    # (A) If global_summary is NOT empty, compute the extrema by scanning all asset/venue
    if global_summary:
        for asset_symbol, venues_dict in global_summary.items():
            for venue_symbol, dt_value in venues_dict.items():
                if global_extrema is None or is_more_extreme(dt_value, global_extrema):
                    global_extrema = dt_value
    else:
        # (B) If global_summary is empty, fall back to the single most extreme from __GLOBAL__
        if sidecar_extrema_values:
            # if extrema == "max", use max(); if extrema == "min", use min()
            aggregator = max if extrema == "max" else min
            global_extrema = aggregator(sidecar_extrema_values)
        else:
            # No sidecars or none had __GLOBAL__ 'extrema'
            global_extrema = None  # or raise an exception, depending on your logic

    return global_extrema, global_summary




class DataLakeInterface:

    def __init__(self,data_lake_source:"PodLocalLake",logger):
        from mainsequence.tdag.config import TDAG_PATH
        self.base_path = f"{TDAG_PATH}/data_lakes"
        self.data_source = data_lake_source
        self.date_range_folder = (
            f"{int(self.data_source.related_resource.datalake_start.timestamp() * 1_000_000) if self.data_source.related_resource.datalake_start is not None else 'None'}_"
            f"{int(self.data_source.related_resource.datalake_end.timestamp() * 1_000_000) if self.data_source.related_resource.datalake_end is not None else 'None'}"
        )

        self.logger=logger
        self.s3_endpoint_url = None
        self.s3_secure_connection = False
        self.s3_data_lake=None
        if self.s3_data_lake:
            from minio import Minio
            s3_endpoint_url = os.environ.get("S3_ENDPOINT_URL", None)
            if s3_endpoint_url is None:
                raise Exception("S3_ENDPOINT_URL must be set if using S3 data lake")

            if s3_endpoint_url.startswith("https://"):
                self.s3_secure_connection = True

            self.s3_endpoint_url = s3_endpoint_url.replace("http://", "").replace("https://", "").strip("/")

            self.minio_client = Minio(
                self.s3_endpoint_url,  # The MinIO server address and port
                access_key=os.environ.get("S3_ACCESS_KEY_ID"),  # Replace with your MinIO access key
                secret_key=os.environ.get("S3_SECRET_ACCESS_KEY"),  # Replace with your MinIO secret key
                secure=self.s3_secure_connection
            )
            self.bucket_name = "tdag"
            if not self.minio_client.bucket_exists(self.bucket_name):
                self.minio_client.make_bucket(self.bucket_name)
        self._create_base_path()

    @staticmethod
    def build_time_and_symbol_filter(
            start_date: Union[datetime.datetime, None] = None,
            great_or_equal: bool = True,
            less_or_equal: bool = True,
            end_date: Union[datetime.datetime, None] = None,
            asset_symbols: Union[list, None] = None):
        """
        Build hashable parquet filters based on the parameters.

        Args:
            metadata (dict): Metadata dictionary, not used for filtering here but included for extensibility.
            start_date (datetime.datetime, optional): Start date for filtering.
            great_or_equal (bool): Whether the start date condition is `>=` or `>`.
            less_or_equal (bool): Whether the end date condition is `<=` or `<`.
            end_date (datetime.datetime, optional): End date for filtering.
            asset_symbols (list, optional): List of asset symbols to filter on.

        Returns:
            tuple: Hashable parquet filters for use with pandas or pyarrow.
        """
        # Define the filters
        filters = []

        # Add asset_symbols filter (OR condition across symbols)
        if asset_symbols:
            asset_symbol_filter = ('asset_symbol', 'in', tuple(asset_symbols))
            filters.append((asset_symbol_filter))

        # Add time_index filter for start_date
        if start_date:
            start_date_op = '>=' if great_or_equal else '>'
            start_date_filter = ('time_index', start_date_op, start_date)
            filters.append((start_date_filter))

        # Add time_index filter for end_date
        if end_date:
            end_date_op = '<=' if less_or_equal else '<'
            end_date_filter = ('time_index', end_date_op, end_date)
            filters.append((end_date_filter))

        # Return filters as a hashable tuple of tuples
        return (tuple(filters),)

    def filter_by_assets_ranges(self,table_name:str,
                                asset_ranges_map:dict,

    ):
        """

        :param table_name:
        :param asset_ranges_map:
        :return:
        """
        filters = tuple(
            tuple(
                (
                    ('asset_symbol', '=', key),
                    ('time_index', conditions['start_date_operand'], conditions['start_date'])
                    if 'start_date' in conditions and conditions['start_date'] is not None else None,
                    ('time_index', '<=', conditions['end_date'])
                    if 'end_date' in conditions and conditions['end_date'] is not None else None,
                )
            )
            for key, conditions in asset_ranges_map.items()
        )

        # Remove None values from each filter group
        filters = tuple(
            tuple(condition for condition in group if condition is not None)
            for group in filters
        )

        data=self.query_datalake(table_name=table_name,filters=filters)

        return data

    def get_data_file_path_for_table(self,table_name):
        return self.get_file_path_for_table(table_name)+"/data"
    def get_side_cars_path_for_table(self,table_name):
        return self.get_file_path_for_table(table_name) + "/side_car"
    def get_file_path_for_table(self,table_name):



        file_path = self.get_file_path(data_lake_name=self.data_source.related_resource.datalake_name,
                                                           date_range_folder=self.date_range_folder,
                                                           table_hash=table_name, )

        return file_path

    def lake_exists(self,table_name):
        file_path = self.get_file_path_for_table(table_name)
        return self.lake_exists(file_path)

    def query_datalake(
            self,
            table_name:str,
            filters: Union[list,None] = None,

    ) -> pd.DataFrame:
        """
        Queries the data lake for time series data.

        If the table_hash is in nodes_to_persist, it retrieves or creates the data.
        Otherwise, it updates the series from the source.

        Args:
            ts: The time series object.
            latest_value: The latest timestamp to query from.
            symbol_list: List of symbols to retrieve data for.
            great_or_equal: Boolean flag for date comparison.
            update_tree_kwargs: Dictionary of kwargs for updating the tree.

        Returns:
            pd.DataFrame: The queried data.
        """
        self.logger.debug(f"Data Lake Read {table_name} ")
        data = read_full_data(file_path=self.get_data_file_path_for_table(table_name=table_name),
                              filters=filters,
        )
        data=data.sort_index(level=data.index.names[0])
        return data

    def persist_datalake(self, data: pd.DataFrame,overwrite:bool,table_name,time_index_name:str,
                         index_names:list):
        """
        Partition per week , do not partition per asset_symbol as system only allows 1024 partittions
        Args:
            data:

        Returns:

        """
        if overwrite == False:
            raise NotImplementedError
        self.logger.debug(f"Persisting  datalake {self.data_source.related_resource.datalake_name}")
        file_path = self.get_file_path_for_table(table_name=table_name)

        iso_calendar = data[time_index_name].dt.isocalendar()
        data[TIME_PARTITION] = iso_calendar['year'].values.astype(str) + '-W' + iso_calendar[
            'week'].values.astype(str)
        partition_cols = [TIME_PARTITION]
        data = data.sort_values(by=time_index_name)
        data.set_index(index_names, inplace=True)


        data_path=self.get_data_file_path_for_table(table_name=table_name)
        side_car_dir=self.get_side_cars_path_for_table(table_name=table_name)
        os.makedirs(side_car_dir, exist_ok=True)
        self.write_to_parquet(data=data, partition_cols=partition_cols, data_path=data_path,
                              upsert=os.path.exists(data_path),time_index_name=time_index_name,
                              )

    def table_exist(self, table_name: str):
        file_path = self.get_data_file_path_for_table(table_name=table_name)
        s3_data_lake=False
        if s3_data_lake:
            bucket_name, object_key = file_path.split("/", 1)

            objects = self.minio_client.list_objects(bucket_name, prefix=object_key, recursive=True)
            for obj in objects:

                return True  # Object exists

            return False  # If an exc
        else:
            return os.path.exists(file_path)

    def get_table_schema(self,table_name):
        data_path = self.get_data_file_path_for_table(table_name)
        if not self.table_exist(table_name):
            return None

        dataset = ds.dataset(data_path, format="parquet")

        return dataset.schema

    def get_parquet_latest_value(self,table_name):

        data_path = self.get_data_file_path_for_table(table_name)
        if not self.table_exist(table_name):
            return None, None
        side_cars_path=self.get_side_cars_path_for_table(table_name=table_name)
        last_index_value,last_multiindex=get_extrema_per_asset_symbol_from_sidecars(side_cars_path,extrema="max")


        return last_index_value, last_multiindex
    def get_lake_earliest_value(self,table_name):
        global_min, asset_min = get_extrema_per_asset_symbol_from_sidecars(side_cars_path, extrema="min")

        return global_min, asset_min

    def _create_base_path(self):
        if self.s3_data_lake:

            pass
        else:
            os.makedirs(os.path.dirname(self.base_path), exist_ok=True)

    def get_file_path(self, data_lake_name: str, date_range_folder: str, table_hash: str):
        if self.s3_data_lake:
            file_path = f"{self.bucket_name}/{data_lake_name.replace(' ', '_').lower()}/{date_range_folder}/{table_hash}"
        else:
            file_path = os.path.join(self.base_path, data_lake_name, date_range_folder,
                                     f"{table_hash}")
        return file_path

    def _get_storage_options(self, file_path):
        protocol = "https" if self.s3_secure_connection else "http"

        s3_path = f"s3://{file_path}"
        storage_options = {
            'key': os.environ.get("S3_ACCESS_KEY_ID"),  # Replace with your MinIO access key
            'secret': os.environ.get("S3_SECRET_ACCESS_KEY"),  # Replace with your MinIO secret key
            'client_kwargs': {
                'endpoint_url': f"{protocol}://{self.s3_endpoint_url}"  # Dynamically set protocol
            }
        }
        return s3_path, storage_options


    def get_parquet_data_set(self,file_path):
        if self.s3_data_lake:
            s3_path, storage_options = self._get_storage_options(file_path)
            s3 = s3fs.S3FileSystem(**storage_options)

            data_set= pq.ParquetDataset(file_path, filesystem=s3)
        else:
            data_set= ds.dataset(file_path, format="parquet", partitioning="hive")
        return data_set

    def write_to_parquet(self, data: pd.DataFrame, data_path: str, partition_cols: list, upsert: bool,
                         time_index_name:str):

        overwrite_carts=True
        data[TIME_PARTITION] = data[TIME_PARTITION].astype('str')
        if self.s3_data_lake:
            # Write directly to S3
            s3_path, storage_options = self._get_storage_options(data_path)
            data.to_parquet(s3_path, partition_cols=partition_cols, engine='pyarrow', storage_options=storage_options)
        else:
            # Local filesystem
            os.makedirs(os.path.dirname(data_path), exist_ok=True)

            if upsert:
                self.upsert_to_parquet(new_data_table=data, file_path=data_path)
                if data.shape[0]==0:
                    overwrite_carts=False #todo this could be done per  partitioned
            else:

                data.to_parquet(data_path, partition_cols=partition_cols, engine='pyarrow',
                                use_dictionary=False  # Disable dictionary encoding
                                )

        self._build_sidecar_partition_summary_stats(data_path,time_index_name,overwrite_carts)

    def _build_sidecar_partition_summary_stats(self,file_path,time_index_name,overwrite_carts=False):
        """
               Walks through the directory tree under `file_path`, finds folders of the form
               TIME_PARTITION=..., and if they do NOT already contain a summary.json sidecar,
               computes min/max of time_index grouped by (asset_symbol, execution_venue_symbol).
               It then writes out a summary.json file into that partition folder.
               """
        import re
        from pathlib import Path
        # Pattern to detect a partition directory named TIME_PARTITION=XXXX...
        partition_dir_pattern = re.compile(r"TIME_PARTITION=")

        for (dirpath, dirnames, filenames) in os.walk(file_path):
            # Check if this directory looks like a time-partitioned folder
            if not partition_dir_pattern.search(dirpath):
                continue  # skip folders that aren't named like TIME_PARTITION=...

            sidecar_dir=Path(dirpath).parent.parent / "side_car" / Path(dirpath).stem
            if os.path.isdir(sidecar_dir) ==True and overwrite_carts ==False:

                continue

            os.makedirs(sidecar_dir, exist_ok=True)

            # Optionally, check if there are any parquet files in this directory
            # to confirm it's a valid partition folder
            # For example:
            parquet_files = [f for f in filenames if f.endswith(".parquet")]
            if not parquet_files:
                # If there are no parquet files here, skip
                print(f"No parquet files in {dirpath}, skipping.")
                continue

            print(f"Processing partition folder: {dirpath}")

            # 1. Read the partition data with Spark
            #    (Spark can handle reading the entire directory if there are multiple .parquet files)
            dataset = ds.dataset(dirpath, format="parquet")

            # 2. Convert the entire partition into a single PyArrow Table,
            #    then to a Pandas DataFrame
            table = dataset.to_table()
            df = table.to_pandas()

            if df.empty:
                print(f"Partition {dirpath} is empty, skipping summary.")
                continue

            # 3. Group by (asset_symbol, execution_venue_symbol) and compute min/max of time_index
            self._build_partition_summary_sidecar(df=df,dirpath=sidecar_dir,time_index_name=time_index_name,
                                                  overwrite_carts=overwrite_carts)

    def _build_partition_summary_sidecar(self,df,dirpath,time_index_name,overwrite_carts):
        is_multin_index=isinstance(df.index, pd.MultiIndex)
        df=df.reset_index()
        summary_dict = {"BY_ASSET_VENUE":{}}
        if is_multin_index:
            per_group_dict = {}

            grouped = (

                df.groupby(["asset_symbol", "execution_venue_symbol"])["time_index"]
                .agg(["min", "max"])
            )  # This returns a Pandas DataFrame with index=(asset_symbol, exec_venue) and columns=[min, max]

            # 4. Convert the result into a nested dictionary structure:
            #    { asset_symbol : { execution_venue_symbol : { "min": ..., "max": ... } } }

            for (asset, venue), row in grouped.iterrows():
                if asset not in per_group_dict:
                    per_group_dict[asset] = {}
                per_group_dict[asset][venue] = {
                    "min": row["min"].timestamp(),
                    "max": row["max"].timestamp()
                }
            summary_dict["BY_ASSET_VENUE"]=per_group_dict
        #add to summary dict the global_max and global_min

        global_min = df[time_index_name].min().timestamp()
        global_max = df[time_index_name].max().timestamp()
        summary_dict["__GLOBAL__"] = {
            "min": global_min,
            "max": global_max
        }

        sidecar_path = os.path.join(dirpath, "summary.json")
        if overwrite_carts and os.path.exists(sidecar_path):
            os.remove(sidecar_path)
        # 5. Write out summary.json in the same directory

        with open(sidecar_path, "w") as f:
            # default=str helps if min/max are Timestamp objects
            json.dump(summary_dict, f)

    def upsert_to_parquet(self, new_data_table: pd.DataFrame,
                          file_path: str):
        # Identify unique partitions
        unique_partitions = new_data_table[TIME_PARTITION].unique()

        # Load existing dataset
        existing_dataset = ds.dataset(file_path, format="parquet", partitioning="hive")

        # Determine index columns
        if isinstance(new_data_table.index, pd.MultiIndex):
            index_names = list(new_data_table.index.names)
        else:
            index_name = new_data_table.index.name if new_data_table.index.name else "index"
            index_names = [index_name]

        for partition_val in unique_partitions:
            # Filter the new data for this partition
            partition_df = new_data_table[new_data_table[TIME_PARTITION] == partition_val].copy()
            partition_df.reset_index(inplace=True)  # Ensure index columns are present as normal columns

            # Filter existing dataset for this partition
            partition_filter = ds.field(TIME_PARTITION) == partition_val
            existing_partition_data = existing_dataset.to_table(filter=partition_filter)

            # If we have existing data, ensure schema alignment
            if existing_partition_data.num_rows > 0:
                existing_cols = existing_partition_data.schema.names

                # Ensure all existing columns are in new DataFrame
                missing_cols = [col for col in existing_cols if col not in partition_df.columns]
                if missing_cols:
                    raise ValueError(f"Missing columns in new data for partition {partition_val}: {missing_cols}")

                # Reorder and convert new DataFrame to match existing schema
                partition_df = partition_df[existing_cols]
                new_partition_data = pa.Table.from_pandas(partition_df, preserve_index=False)
                new_partition_data = new_partition_data.cast(existing_partition_data.schema)

                # Create combined keys for deduplication
                def combined_key_array(table: pa.Table, idx_cols: list[str]) -> pa.Array:
                    str_arrays = [table[col].cast(pa.string()) for col in idx_cols]
                    if len(str_arrays) == 1:
                        return str_arrays[0]
                    combined = str_arrays[0]
                    for col_arr in str_arrays[1:]:
                        combined = pa.compute.binary_join_element_wise(
                            combined, pa.array(["-"] * len(col_arr)), col_arr
                        )
                    return combined

                existing_keys = combined_key_array(existing_partition_data, index_names)
                new_keys = combined_key_array(new_partition_data, index_names)

                in_filter = pa.compute.is_in(existing_keys, value_set=new_keys)
                inverted_filter = pa.compute.invert(in_filter)
                existing_partition_filtered = existing_partition_data.filter(inverted_filter)

            else:
                # No existing data, just use the new data's schema
                new_partition_data = pa.Table.from_pandas(partition_df, preserve_index=False)
                # Create empty table with the same schema for easy concatenation
                existing_partition_filtered = pa.Table.from_arrays(
                    [pa.array([], type=field.type) for field in new_partition_data.schema],
                    schema=new_partition_data.schema
                )

            merged_data = pa.concat_tables([existing_partition_filtered, new_partition_data])

            # Write out the merged data
            partition_dir_name = f"{TIME_PARTITION}={partition_val}"
            partition_path = os.path.join(file_path, partition_dir_name)
            temp_partition_path = partition_path + "_tmp"

            # Clean up temp directory if exists
            if os.path.exists(temp_partition_path):
                shutil.rmtree(temp_partition_path)

            os.makedirs(temp_partition_path, exist_ok=True)

            # Write a single parquet file inside the temp partition directory
            temp_file = os.path.join(temp_partition_path, "part-0.parquet")

            pq.write_table(merged_data, temp_file,
                           use_dictionary={TIME_PARTITION: False},  # <--- ensure string, no dictionary
                           )

            # Replace old partition directory
            if os.path.exists(partition_path):
                shutil.rmtree(partition_path)
            os.rename(temp_partition_path, partition_path)

        print("Upsert operation complete.")