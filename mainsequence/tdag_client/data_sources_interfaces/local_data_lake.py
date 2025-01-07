
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
        use_s3_if_available: bool,
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

    return data.sort_index()




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
            table_name:str, index_names=list,
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
        data = read_full_data(file_path=self.get_file_path_for_table(table_name=table_name),
                              filters=filters,
        )
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
        self.write_to_parquet(data=data, partition_cols=partition_cols, file_path=file_path,
                              upsert=self.table_exist(table_name)
                              )

    def table_exist(self, table_name: str):
        file_path = self.get_file_path_for_table(table_name=table_name)
        s3_data_lake=False
        if s3_data_lake:
            bucket_name, object_key = file_path.split("/", 1)

            objects = self.minio_client.list_objects(bucket_name, prefix=object_key, recursive=True)
            for obj in objects:

                return True  # Object exists

            return False  # If an exc
        else:
            return os.path.exists(file_path)

    def get_parquet_latest_value(self,table_name):

        file_path = self.get_file_path_for_table(table_name)
        if not self.table_exist(table_name):
            return None, None
        self.logger.warning("IMPROVE SPEED READ FROM STATS")
        data_set = self.get_parquet_data_set(file_path)
        time_index_column_name = data_set.schema.pandas_metadata['index_columns'][0]

        partitions = data_set.partitioning
        time_partition = partitions.dictionaries[0]
        partition_names = partitions.schema.names

        time_partition = sorted(
            time_partition.to_pylist(),
            key=lambda x: (int(x.split('-W')[0]), int(x.split('-W')[1]))
        )

        filtered_dataset = read_parquet_from_lake(
            file_path=file_path,
            use_s3_if_available=False,
            filters=[[(TIME_PARTITION, "=", time_partition[-1])]]
        )
        last_multiindex = {}
        if "asset_symbol" in filtered_dataset.index.names:
            # Get the latest index value for each asset and execution venue
            last_multiindex = (
                filtered_dataset.index
                .to_frame(index=False)  # Convert index to DataFrame without resetting
                .groupby(["asset_symbol", "execution_venue_symbol"], observed=True)
                [time_index_column_name]
                .max()
            )
            last_multiindex = {
                asset: {ev: timestamp}
                for (asset, ev), timestamp in last_multiindex.items()
            }

            # Get the latest time index value across all assets and execution venues
            last_index_value = filtered_dataset.index.get_level_values('time_index').max()
        else:
            last_index_value = filtered_dataset.index.max()

        if len(last_multiindex) == 0:
            last_multiindex = None

        assert last_index_value
        return last_index_value, last_multiindex

    def _create_base_path(self):
        if self.s3_data_lake:

            pass
        else:
            os.makedirs(os.path.dirname(self.base_path), exist_ok=True)

    def get_file_path(self, data_lake_name: str, date_range_folder: str, table_hash: str):
        if self.s3_data_lake:
            file_path = f"{self.bucket_name}/{data_lake_name.replace(' ', '_').lower()}/{date_range_folder}/{table_hash}.parquet"
        else:
            file_path = os.path.join(self.base_path, data_lake_name, date_range_folder,
                                     f"{table_hash}.parquet")
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

    def write_to_parquet(self, data: pd.DataFrame, file_path: str, partition_cols: list, upsert: bool):
        data[TIME_PARTITION] = data[TIME_PARTITION].astype('str')
        if self.s3_data_lake:
            # Write directly to S3
            s3_path, storage_options = self._get_storage_options(file_path)
            data.to_parquet(s3_path, partition_cols=partition_cols, engine='pyarrow', storage_options=storage_options)
        else:
            # Local filesystem
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if upsert:
                self.upsert_to_parquet(new_data_table=data, file_path=file_path)
            else:

                data.to_parquet(file_path, partition_cols=partition_cols, engine='pyarrow',
                                use_dictionary=False  # Disable dictionary encoding
                                )

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