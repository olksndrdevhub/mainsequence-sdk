
import os
import pandas as pd
from typing import Union, List
import s3fs
import yaml
import pyarrow.parquet as pq
import json
from functools import lru_cache
import psutil
import datetime
import pyarrow.dataset as ds
import pytz

TIME_PARTITION = "TIME_PARTITION"



def memory_usage_exceeds_limit(max_usage_percentage):
    """
    Checks if the current memory usage exceeds the given percentage of total memory.
    """
    memory_info = psutil.virtual_memory()
    used_memory_percentage = memory_info.percent
    return used_memory_percentage > max_usage_percentage

@lru_cache(maxsize=50)
def read_full_data(file_path,index_names,filters=None, use_s3_if_available=False, max_memory_usage=80):
    " Cached access to static datalake file "
    # logger.debug("in data lake read")
    if memory_usage_exceeds_limit(max_memory_usage):
        # Clear cache if memory exceeds the limit
        read_full_data.cache_clear()
        # logger.info("Cache cleared due to high memory usage.")

    data = read_parquet_from_lake(file_path=file_path,filters=filters,use_s3_if_available=use_s3_if_available)
    data = data.drop(columns=[TIME_PARTITION]).set_index([c for c in index_names]).sort_index(level=index_names[0])
    return data

def read_parquet_from_lake(file_path: str,use_s3_if_available:bool,filters:Union[list,None]=None,
                           s3_storage_options:Union[dict,None]=None,
                           ):

    extra_kwargs={} if filters is None else {'filters':filters}
    if use_s3_if_available:
        s3_path,storage_options=self._get_storage_options(file_path)
        data = pd.read_parquet(s3_path, engine='pyarrow', storage_options=storage_options,**extra_kwargs)
    else:
        data = pd.read_parquet(file_path,**extra_kwargs)
    return data




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



    def filter_by_assets_ranges(self,table_name:str,
                                index_names:list,
                                asset_ranges_map:dict):
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

        df=self._query_datalake(table_name,filters=filters,index_names=index_names)
        return df

    def get_file_path_for_table(self,table_name):



        file_path = self.get_file_path(data_lake_name=self.data_source.related_resource.datalake_name,
                                                           date_range_folder=self.date_range_folder,
                                                           table_hash=table_name, )

        return file_path

    def lake_exists(self,table_name):
        file_path = self.get_file_path_for_table(table_name)
        return self.lake_exists(file_path)
    def _query_datalake(
            self,
            table_name:str, index_names=list,
            start_date: Union[datetime.datetime, None] = None,
            symbol_list: Union[List[str], None] = None,
            great_or_equal: bool=False, less_or_equal: bool = True, end_value: Union[datetime.datetime, None] = None,
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



        self.logger.debug(f"Read {table_name} ")
        data = read_full_data(file_path=self.get_file_path_for_table(table_name=table_name),
                              filters=filters,index_names=tuple(index_names),
                              )

        # Todo pass this filter to parquet read
        if symbol_list is not None:
            data = data[data.index.get_level_values("asset_symbol").isin(symbol_list)]
        filter_index = data.index.get_level_values("time_index") if len(data.index.names) > 1 else data.index
        if start_date is not None:
            if great_or_equal:
                data = data[filter_index >= start_date]
            else:
                data = data[filter_index > start_date]

        if end_value is not None:
            if less_or_equal:
                data = data[filter_index <= end_value]
            else:
                data = data[filter_index < end_value]

        return data
    def persist_datalake(self, data: pd.DataFrame,overwrite:bool,table_name,time_index_name:str,
                         index_names:list):
        """
        Partition per week , do not partition per asset_symbol as system only allows 1024 partittions
        Args:
            data:

        Returns:

        """
        if overwrite==False:
            raise NotImplementedError
        self.logger.debug(f"Persisting  datalake {self.data_source.related_resource.datalake_name}")
        file_path = self.get_file_path_for_table(table_name=table_name)



        iso_calendar = data[time_index_name].dt.isocalendar()
        data[TIME_PARTITION] = iso_calendar['year'].values.astype(str) + '-W' + iso_calendar[
            'week'].values.astype(str)
        partition_cols = [TIME_PARTITION]
        data = data.sort_values(by=time_index_name)

        self.write_to_parquet(data=data, partition_cols=partition_cols, file_path=file_path)






    def lake_exists(self, file_path: str):
        if self.s3_data_lake:
            bucket_name, object_key = file_path.split("/", 1)

            objects = self.minio_client.list_objects(bucket_name, prefix=object_key, recursive=True)
            for obj in objects:

                return True  # Object exists

            return False  # If an exc
        else:
            return os.path.exists(file_path)

    def _get_parquet_latest_value(self):
        file_path = self.get_file_path()
        data_set = self.get_parquet_data_set(file_path)
        time_index_column_name = data_set.schema.pandas_metadata["index_columns"][0]

        partitions = data_set.partitioning
        time_partition = partitions.dictionaries[0]
        partition_names = partitions.schema.names

        time_partition = sorted(
            time_partition.to_pylist(),
            key=lambda x: (int(x.split('-W')[0]), int(x.split('-W')[1]))
        )

        filtered_dataset = self.read_parquet_from_lake(file_path=file_path,
                                                                           filters=[(TIME_PARTITION, "=",
                                                                                     time_partition[-1])])
        multiindex_df = filtered_dataset.reset_index()

        # Get the latest index value for each asset and execution venue
        last_multiindex = multiindex_df.groupby(["asset_symbol", "execution_venue_symbol"], observed=True)[
            time_index_column_name].max()
        last_multiindex = {
            asset: {ev: timestamp}
            for (asset, ev), timestamp in last_multiindex.items()
        }

        # Get the latest time index value across all assets and execution venues
        last_index_value = multiindex_df.set_index(
            ["time_index", "asset_symbol", "execution_venue_symbol"]).index.get_level_values('time_index').max()

        if len(last_multiindex) == 0:
            last_multiindex = None

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
            data_set=pq.ParquetDataset(file_path)
        return data_set


    def write_to_parquet(self, data: pd.DataFrame, file_path: str, partition_cols: list):
        """

        Parameters
        ----------
        data

        Returns
        -------

        """
        if self.s3_data_lake:
            s3_path, storage_options = self._get_storage_options(file_path)
            data.to_parquet(s3_path, partition_cols=partition_cols, engine='pyarrow',
                            storage_options=storage_options)
        else:
            os.makedirs(os.path.dirname(file_path, ), exist_ok=True)
            data.to_parquet(file_path, partition_cols=partition_cols, engine='pyarrow')