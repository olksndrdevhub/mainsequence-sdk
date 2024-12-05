import pandas as pd
import datetime
from typing import Union, List,Dict
import pyarrow.dataset as ds
import pytz
import shutil
import os
import yaml
import pyarrow.parquet as pq
import json
from functools import lru_cache
import psutil
import s3fs
from mainsequence.tdag.config import (
    ogm, logging_folder,
    DEFAULT_RETENTION_POLICY)
from mainsequence.tdag.utils import read_yaml
from mainsequence.tdag.logconf import console_logger, create_logger_in_path

from mainsequence.tdag_client import (DynamicTableHelpers, TimeSerieNode, TimeSerieLocalUpdate,
                                      LocalTimeSeriesDoesNotExist,
                                      DynamicTableDoesNotExist, DynamicTableDataSource)

import structlog
from mainsequence.tdag.logconf import get_tdag_logger

logger = get_tdag_logger()
def get_persist_manager(local_hash_id,remote_table_hashed_name:Union[str,None], *args, **kwargs):
    persist_manager = TimeScaleLocalPersistManager(local_hash_id=local_hash_id,
                                                   logger=None,persist_parquet=False,
                                                   class_name=None,
                                                   remote_table_hashed_name=remote_table_hashed_name,
                                                   description=""
                                                   )

    return persist_manager


class RetentionPolicyHelper:
    TIME_MAP = {"day": "days", "days": "days", "d": "days", "minute": "minutes"}

    def __init__(self, retention_policy: str):
        self.retention_policy = retention_policy
        self.set_retention_policy_td()

    def set_retention_policy_td(self):
        td = self.retention_policy.split()
        td_kwargs = {self.TIME_MAP[td[1]]: int(td[0])}
        td = datetime.timedelta(**td_kwargs)
        self.retention_policy_td = td

    def data_lower_bound(self, target_data):
        """
        Calculates data lower bound to target date according to retention policy
        :param target_data:
        :return:
        """
        return target_data - self.retention_policy_td


class DataLakeInterface:
    def __init__(self,use_s3_if_available:bool):
        from mainsequence.tdag.config import TDAG_PATH
        self.base_path = f"{TDAG_PATH}/data_lakes"
        self.s3_data_lake = use_s3_if_available

        self.s3_endpoint_url = None
        self.s3_secure_connection = False
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

    def lake_exists(self, file_path: str):
        if self.s3_data_lake:
            bucket_name, object_key = file_path.split("/", 1)

            objects = self.minio_client.list_objects(bucket_name, prefix=object_key, recursive=True)
            for obj in objects:

                return True  # Object exists

            return False  # If an exc
        else:
            return os.path.exists(file_path)

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

    def read_parquet_from_lake(self, file_path: str,filters:Union[list,None]=None):

        extra_kwargs={} if filters is None else {'filters':filters}
        if self.s3_data_lake:
            s3_path,storage_options=self._get_storage_options(file_path)
            data = pd.read_parquet(s3_path, engine='pyarrow', storage_options=storage_options,**extra_kwargs)
        else:
            data = pd.read_parquet(file_path,**extra_kwargs)
        return data

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

class DataLakePersistManager:
    """
    A class to manage data persistence in a local data lake.

    This class handles the storage and retrieval of time series data in a local file system,
    organized by date ranges and table hashes.
    """
    TIME_PARTITION = "TIME_PARTITION"
    def __init__(self, data_lake_name:str, start_latest_value:datetime.datetime,
                 table_hash:str, nodes_to_get_from_db:list,use_s3_if_available:bool,
                 end_latest_value:datetime.datetime):
        """
        Initializes the DataLakePersistManager with configuration from environment variables.
        """
        self.use_s3_if_available=use_s3_if_available
        self.data_lake_interface = DataLakeInterface(use_s3_if_available=use_s3_if_available)

        self.datalake_name = data_lake_name
        self.logger = create_logger_in_path(logger_file=f"{logging_folder}/{self.datalake_name}.log",
                                          logger_name=self.datalake_name)
        self.table_hash = table_hash
        self.start_latest_value = start_latest_value
        self.end_latest_value = end_latest_value
        self.nodes_to_get_from_db = nodes_to_get_from_db
        str_string = start_latest_value.timestamp() if start_latest_value is not None else None
        end_str = end_latest_value.timestamp() if end_latest_value is not None else None
        self.date_range_folder = f"{str_string}_{end_str}"
        self.already_introspected = self.set_introspection(introspection=False)


    @staticmethod
    def _memory_usage_exceeds_limit(max_usage_percentage):
        """
        Checks if the current memory usage exceeds the given percentage of total memory.
        """
        memory_info = psutil.virtual_memory()
        used_memory_percentage = memory_info.percent
        return used_memory_percentage > max_usage_percentage

    @staticmethod
    @lru_cache(maxsize=50)
    def _read_full_data(file_path,use_s3_if_available, max_memory_usage=80):
        " Cached access to static datalake file "
        # logger.debug("in data lake read")
        if DataLakePersistManager._memory_usage_exceeds_limit(max_memory_usage):
            # Clear cache if memory exceeds the limit
            DataLakePersistManager._read_full_data.cache_clear()
            logger.info("Cache cleared due to high memory usage.")
        
        data_lake_interface=DataLakeInterface(use_s3_if_available=use_s3_if_available)
        data = data_lake_interface.read_parquet_from_lake(file_path)
        data = data.drop(columns=[DataLakePersistManager.TIME_PARTITION]).sort_index()
        return data

    
    def set_introspection(self, introspection:bool):
        """
        This methos is critical as it control the level of introspection and avouids recursivity\
        This happens for example when TimeSeries.update_series_from_source(*,**):
        TimeSeries.update_series_from_source(latest_value,*,**):
            self.get_latest_value() <- will incurr in a circular refefence using local data late
        Args:
            introspection:

        Returns:

        """
        self.logger.debug(f"Setting introspection for {self.table_hash}")
        self.introspection = introspection

    def lake_exists(self):
        file_path = self.get_file_path()
        return self.data_lake_interface.lake_exists(file_path)

    def get_file_path(self):
        file_path = self.data_lake_interface.get_file_path(data_lake_name=self.datalake_name,
                                               date_range_folder=self.date_range_folder,table_hash=self.table_hash,)

        return file_path


    def get_latest_value(self, ts, asset_symbols:list, *args, **kwargs)->[datetime.datetime, Dict[str, datetime.datetime]]:
        """
        Returns: [datetime.datetime,Dict[str, datetime]]
        """
        if self.introspection:
            return None, None

        file_path = self.get_file_path()
        last_index_value, last_multiindex = None, None
        if not os.path.exists(file_path):
            # create the object
            _ = self._query_datalake(
                ts=ts,
                symbol_list=asset_symbols,
                great_or_equal=True,
                start_date=None,
                *args,**kwargs
            )

        last_index_value, last_multiindex = self._get_parquet_latest_value()
        return last_index_value, last_multiindex

    def get_df_greater_than_in_table(self, ts: object,
                            latest_value: pd.Timestamp,
                            symbol_list: List[str],
                            great_or_equal: bool,
                            *args,
                            **kwargs):
        data = self._query_datalake(
            ts=ts,
            start_date=latest_value,
            symbol_list=symbol_list,
            great_or_equal=great_or_equal,
            *args,
            **kwargs
        )
        return data

    def get_df_between_dates(self, ts, symbol_list:List[str],
                             great_or_equal: bool, less_or_equal: bool ,
                             end_date: datetime.datetime, start_date: datetime.datetime,
                             ):
        return self._query_datalake(ts=ts, symbol_list=symbol_list, great_or_equal=great_or_equal,
                                   less_or_equal=less_or_equal, start_date=start_date, end_date=end_date)

    def _get_parquet_latest_value(self):
        file_path = self.get_file_path()
        data_set = self.data_lake_interface.get_parquet_data_set(file_path)
        time_index_column_name = data_set.schema.pandas_metadata["index_columns"][0]

        partitions = data_set.partitioning
        time_partition = partitions.dictionaries[0]
        partition_names = partitions.schema.names

        time_partition = sorted(
            time_partition.to_pylist(),
            key=lambda x: (int(x.split('-W')[0]), int(x.split('-W')[1]))
        )

        filtered_dataset = self.data_lake_interface.read_parquet_from_lake(file_path=file_path,
                                                                           filters= [(self.TIME_PARTITION, "=", time_partition[-1])])
        multiindex_df = filtered_dataset.reset_index()

        # Get the latest index value for each asset and execution venue
        last_multiindex = multiindex_df.groupby(["asset_symbol", "execution_venue_symbol"], observed=True)[time_index_column_name].max()
        last_multiindex = {
            asset: {ev: timestamp}
            for (asset, ev), timestamp in last_multiindex.items()
        }

        # Get the latest time index value across all assets and execution venues
        last_index_value = multiindex_df.set_index(["time_index", "asset_symbol", "execution_venue_symbol"]).index.get_level_values('time_index').max()

        if len(last_multiindex) == 0:
            last_multiindex = None

        return last_index_value, last_multiindex

    def _persist_datalake(self, data:pd.DataFrame):
        """
        Partition per week , do not partition per asset_symbol as system only allows 1024 partittions
        Args:
            data:

        Returns:

        """
        self.logger.debug(f"Persisting  datalake {self.datalake_name}")
        file_path = self.get_file_path()
        if len(data.index.names) > 1:
            iso_calendar = data.index.get_level_values("time_index").isocalendar().astype(str)
        else:
            iso_calendar = data.index.isocalendar()
        data[self.TIME_PARTITION] = iso_calendar['year'].values.astype(str) + '-W' + iso_calendar[
            'week'].values.astype(str)
        partition_cols = [self.TIME_PARTITION]
        data = data.sort_index(level="time_index")

        self.data_lake_interface.write_to_parquet(data=data,partition_cols=partition_cols, file_path=file_path)

    def _query_datalake(
            self,
            ts: object,
            start_date: pd.Timestamp,
            symbol_list: List[str],
            great_or_equal: bool,less_or_equal:bool=True,end_value:Union[datetime.datetime,None]=None,
            *args,
            **kwargs
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
        if self.table_hash in self.nodes_to_get_from_db and not self.lake_exists():
            # Nodes should be queried directly from the DB
            self.logger.debug(f"Persisting {self.table_hash} directly from DB in to datalake {self.datalake_name}")
            data = ts.get_df_between_dates(
                start_date=self.start_latest_value,
                end_date=self.end_latest_value,
                data_lake_force_db_look=True,
                asset_symbols=symbol_list
            )
            if len(data) == 0:
                raise ValueError(f"Data for {self.table_hash} empty")
            self._persist_datalake(data=data)

        elif not self.lake_exists():
            self.logger.debug(f"Persisting {self.table_hash} locally generated to datalake {self.datalake_name}")
            self.set_introspection(True)
            data = ts.update_series_from_source(latest_value=self.start_latest_value, *args, **kwargs)
            if len(data) == 0:
                raise ValueError(f"Results data after update_series_from_source for {self.table_hash} empty")
            self._persist_datalake(data=data)

        else:
            self.logger.debug(f"Read {self.table_hash} from datalake {self.datalake_name}")
            data = DataLakePersistManager._read_full_data(file_path=self.get_file_path(),use_s3_if_available=self.use_s3_if_available)

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

class ParquetLocalPersistManager:

    def __init__(self, data_folder: str, retention_policy: Union[None, str],logger:Union[structlog.BoundLogger,None]):
        self.data_folder = data_folder
        self.logger=logger
        if logger is None:
            self.logger = create_logger_in_path(logger_name="parquet_persist_manager", application_name="tdag",
                                                    logger_file=f'{ogm.get_logging_path()}/parquet_persist_manager.log',
                                                    )

        self.config_file = self.data_folder + "/config.yaml"
        self.meta_file = self.data_folder + "/meta_data.yaml"
        self.local_parquet_file = self.data_folder + "/time_series_data.parquet"
        if retention_policy is not None:
            self.retention_policy_helper = RetentionPolicyHelper(retention_policy=retention_policy)

    @property
    def build_configuration(self):
        try:
            conf = read_yaml(self.config_file)
            return conf
        except:
            raise

    @property
    def meta_data(self):
        meta_data = read_yaml(self.meta_file)
        return meta_data

    @property
    def persist_size(self):
        return os.path.getsize(self.local_parquet_file) / (1024 * 1024)

    def get_df_greater_than_in_table(self, target_value: datetime.datetime,time_index_name:str, great_or_equal: bool, columns=None,
                            asset_symbols:Union[None,list]=None,

                            ):
        filtered_data = self.read_parquet_dates_between_dates(file_path=self.local_parquet_file,
                                                              start_date=target_value,time_index_name=time_index_name,
                                                              great_or_equal=great_or_equal, less_or_equal=False,
                                                              asset_symbols=asset_symbols,
                                                              end_date=None,)
        if columns is not None:
            filtered_data = filtered_data[columns]
        return filtered_data

    def get_df_between_dates(self, start_date: datetime.datetime, end_date: datetime.datetime,
                             time_index_name:str, asset_symbols:Union[None,list]=None,
                             great_or_equal=True, less_or_equal=True,columns=None):
        filtered_data = self.read_parquet_dates_between_dates(file_path=self.local_parquet_file,
                                                              start_date=start_date,
                                                              great_or_equal=great_or_equal,
                                                              less_or_equal=less_or_equal,
                                                              end_date=end_date,
                                                              time_index_name=time_index_name,
                                                              asset_symbols=asset_symbols,
                                                              )
        if columns is not None:
            filtered_data = filtered_data[columns]
        return filtered_data

    # def set_meta(self, key, value):
    #     """
    #     Sets Meta data key
    #     :param key:
    #     :param value:
    #     :return:
    #     """
    #     write_key_to_yaml(path=self.meta_file, key=key, key_dict=value)

    def local_persist_exist_set_config(self, config):
        e = self.local_persist_exist()
        if e == False:
            self.set_initial_config(config_kwargs=config)

    def local_persist_exist(self):

        return os.path.isdir(self.data_folder)

    def set_initial_config(self, config_kwargs):
        # time series is not locally persisted
        os.makedirs(self.data_folder, exist_ok=True)
        try:
            with open(self.config_file, 'w') as f:
                data = yaml.dump(config_kwargs, f)
        except Exception as e:
            os.remove(self.config_file)

    def time_serie_exist(self):
        return os.path.isdir(self.local_parquet_file)

    def get_persisted_ts(self):

        persisted_df = pd.read_parquet(self.local_parquet_file)

        idx = persisted_df.index
        if isinstance(idx, pd.MultiIndex):
            target_level = len(idx.names) - 1
            pandas_df = persisted_df.sort_index(level=target_level)
        else:
            pandas_df = persisted_df.sort_index()
        pandas_df = pandas_df.drop(columns=["PART_YEAR", "PART_MONTH"])

        return pandas_df

    def _get_partition_column_names(self, pandas_df):
        """
        Gets the parittion column list from dataframe
        :param pandas_df:
        :return:
        """
        columns = ["PART_YEAR", "PART_MONTH"]
        if isinstance(pandas_df.columns, pd.MultiIndex):
            number_of_column_idx = len(pandas_df.columns.names)
            append_str = ", ".join(
                ["''" if c < number_of_column_idx - 2 else "'')" for c in range(number_of_column_idx - 1)])

            columns = [f"('{c}', " + append_str for c in columns]
        return columns

    def add_partition_columns(self, pandas_df: pd.DataFrame, idx):

        pandas_df["PART_MONTH"] = idx.month
        pandas_df["PART_YEAR"] = idx.year
        columns = self._get_partition_column_names(pandas_df=pandas_df)



        return pandas_df, columns

    def parse_data_update_if_persisted(self, temp_df, latest_value):
        # concatenate with parquet partition
        rewrite_parquet = self.get_df_greater_than_in_table(
            latest_value.replace(day=1, hour=0, minute=0, second=0, microsecond=0),
            great_or_equal=True,
        )
        temp_df = pd.concat([rewrite_parquet, temp_df], axis=0)
        return temp_df

    def persist_updated_data(self, temp_df
                             , overwrite=True):

        idx = temp_df.index
        asset_symbol_idx=None
        if isinstance(idx, pd.MultiIndex):
            assert idx.dtypes.iloc[0] == "datetime64[ns, UTC]","index first level name must be 'time_inde' "
            assert idx.names[1]=="asset_symbol","index second level name must be 'asset_symbol' "
            assert idx.names[
                       2] == "execution_venue_symbol", "index third level name must be 'execution_venue_symbol' "
            temp_df = temp_df.sort_index(level=0)
            idx = idx.get_level_values(level=0)

            limit = self.retention_policy_helper.data_lower_bound(target_data=idx.max())
            new_df = temp_df[idx >= limit].copy()

            idx = new_df.index.get_level_values(level=0)
            asset_symbol_idx= new_df.index.get_level_values(level=1)
            execution_venue_idx = new_df.index.get_level_values(level=2)


        else:
            assert idx.dtype == "datetime64[ns, UTC]"
            if idx.duplicated().sum() != 0:
                raise ValueError("Duplicated values in index")
            limit = self.retention_policy_helper.data_lower_bound(target_data=idx.max())
            new_df = temp_df[idx >= limit].copy()
            idx = new_df.index
        
        new_df, partition_columns = self.add_partition_columns(pandas_df=new_df, idx=idx)

        new_df = self._filter_by_retention_policy(temp_df=new_df)
        skip = False

        if overwrite == True:
            try:
                if asset_symbol_idx is not None:
                    new_df["new_time"]=new_df.index.get_level_values(0).floor("us")
                    new_df=new_df.reset_index().drop(columns=["time_index"]).rename(columns={"new_time":"time_index"}).set_index(["time_index","asset_symbol","execution_venue_symbol"])
                    partition_columns.append("asset_symbol")
                    partition_columns.append("execution_venue_symbol")

                    total_partitions = new_df.reset_index()[partition_columns].nunique().prod()
                    if total_partitions > 1024 - 12 * 20:
                        partition_columns = [c for c in partition_columns if c != "asset_symbol"]
                    
                else:
                    new_df.index = new_df.index.floor("us")
                new_df=new_df[~new_df.index.duplicated(keep='first')]
                new_df.to_parquet(self.local_parquet_file, partition_cols=partition_columns, use_legacy_dataset=False,
                                   existing_data_behavior="overwrite_or_ignore",
                                   basename_template="part-{i}.parquet'")
            except Exception as e:
                logger.error(new_df)
                logger.error(new_df.iloc[-1])
                raise e
            return None

    def _filter_by_retention_policy(self, temp_df: pd.DataFrame) -> pd.DataFrame:
        from mainsequence.tdag_client import JSON_COMPRESSED_PREFIX
        try:
            old_data = pd.read_parquet(self.local_parquet_file)
            shutil.rmtree(self.local_parquet_file)
            for col in temp_df.columns:
                if any([j in col for j in JSON_COMPRESSED_PREFIX]) == True:
                    temp_df[col] = temp_df[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
                    try:
                        old_data[col]= old_data[col].apply(lambda x: json.dumps(x) if isinstance(x, dict) else x)
                    except Exception as e:
                        raise e
        except FileNotFoundError:
            old_data = pd.DataFrame()
        temp_df = pd.concat([old_data, temp_df], axis=0)

        time_idx = temp_df.index
        time_idx = time_idx.get_level_values(level=0) if isinstance(time_idx,pd.MultiIndex) else time_idx
        limit = self.retention_policy_helper.data_lower_bound(target_data=time_idx.max())
        temp_df = temp_df[time_idx >= limit]
        time_idx = temp_df.index
        time_idx = time_idx.get_level_values(level=0) if isinstance(time_idx,pd.MultiIndex) else time_idx
        if isinstance(temp_df.index, pd.MultiIndex) == False:
            if time_idx.duplicated().sum() != 0:
                temp_df = temp_df[~time_idx.duplicated(keep='first')]

        return temp_df

    def flush_local_persisted(self, flush_only_time_series, ):
        self.logger.warning("DELETING LOCAL PARQUET")
        if self.local_persist_exist()== True:
            flush_sub_folders = True if flush_only_time_series == False else False
            for path, directories, files in os.walk(self.data_folder):
                for d in directories:
                    if flush_sub_folders is True:
                        shutil.rmtree(path + "/" + d)
                for f in files:
                    if f not in ["config.yaml", "meta_data.yaml"]:
                        os.remove(path + "/" + f)

    @staticmethod
    def read_parquet_dates_between_dates(file_path: str, time_index_name:str,start_date: Union[None, datetime.datetime],
                                         end_date: Union[None, datetime.datetime],
                                         great_or_equal: bool = True, less_or_equal: bool = True,
                                         columns=None,asset_symbols:Union[list,None]=None
                                         ):
        """
        read local parquet file data by pushing a filter  on the time index, this avoids loading all the
            data and just reading from column filter
        :param start_date:
        :param end_time:
        :return:
        """
        import operator
        from pyarrow.lib  import ArrowInvalid
        convert_to_utz = lambda x: x.replace(tzinfo=pytz.utc) if x.tzinfo is None else x
        start_date, end_date = convert_to_utz(start_date) if start_date is not None else None, convert_to_utz(
            end_date) if end_date is not None else None
        pandas_meta_data = ds.dataset(file_path).schema.pandas_metadata
        column_indices = pandas_meta_data["column_indexes"]
        number_of_column_idx = len(column_indices)
        symbol_filter=[]
        if asset_symbols is not None:
            symbol_filter = ("asset_symbol", "in", asset_symbols)
        if start_date is None:
            filtered_data = pd.read_parquet(file_path)
        else:
            parquet_join_filters =[(time_index_name, ">=", start_date)]
            parquet_join_filters.append(symbol_filter)

            try:

                try:

                    filtered_data = pd.read_parquet(file_path, filters=[[c for c in parquet_join_filters if len(c)!=0]])
                except ArrowInvalid as e:
                    parquet_join_filters = [("__index_level_0__", ">=", start_date)]
                    parquet_join_filters.append(symbol_filter)
                    filtered_data = pd.read_parquet(file_path, filters=[[c for c in parquet_join_filters if len(c)!=0]])

            except Exception as e:
                raise e
        idx = filtered_data.index
        if isinstance(idx, pd.MultiIndex):
            target_level = 0
            idx = idx.get_level_values(level=target_level)
            filter_kwargs = dict(level=target_level)
        else:
            filter_kwargs = {}
        great_operator = operator.ge if great_or_equal == True else operator.gt
        less_operator = operator.le if less_or_equal == True else operator.lt
        if start_date is not None and end_date is None:
            filtered_data = filtered_data[great_operator(idx, start_date)]
        elif start_date is None and end_date is not None:
            filtered_data = filtered_data[less_operator(idx, end_date)]
        elif start_date is None and end_date is None:
            pass
        else:
            filtered_data = filtered_data[great_operator(idx, start_date) & less_operator(idx, end_date)]
        filtered_data = filtered_data.sort_index(**filter_kwargs)

        if number_of_column_idx > 1:
            filtered_data = filtered_data.drop(columns=["PART_YEAR", "PART_MONTH"], level=0)
        else:
            filtered_data = filtered_data.drop(columns=["PART_YEAR", "PART_MONTH"])

        return filtered_data

    @staticmethod
    def read_ts_persisted_parquet(local_parquet_file_path: str):
        pandas_df = pd.read_parquet(local_parquet_file_path)
        idx = pandas_df.index
        if isinstance(idx, pd.MultiIndex):
            target_level = len(idx.names) - 1
            pandas_df = pandas_df.sort_index(level=target_level)
        else:
            pandas_df = pandas_df.sort_index()
        pandas_df = pandas_df.drop(columns=["PART_YEAR", "PART_MONTH"])
        return pandas_df

    def get_earliest_value(self):
        from pyarrow.lib import ArrowInvalid
        if self.time_serie_exist() is True:
            latest_value = []
            dataset = pq.ParquetDataset(self.local_parquet_file)
            schema = dataset.partitioning.schema
            column_index=1
            if "asset_symbol" in dataset.partitioning.schema.pandas_metadata["index_columns"]:
                if "asset_symbol" not in dataset.partitioning.schema.names:
                    column_index=2
            for path, subdirs, files in os.walk(self.local_parquet_file):
                for name in files:
                    if ".parquet" in name:
                        try:
                            meta_data = pq.read_metadata(os.path.join(path, name))
                        except ArrowInvalid as e:
                            logger.warning("Error reading parquet for latest value. Deleting local persisted")
                            self.flush_local_persisted(flush_only_time_series=True)
                            raise e
                        except Exception as e:
                            raise e
                        statistics = min(
                            [meta_data.row_group(r).column(meta_data.num_columns - column_index).statistics.min for r in
                             range(meta_data.num_row_groups)])
                        # statistics = meta_data.row_group(meta_data.num_row_groups - 1).column(meta_data.num_columns - 1).statistics
                        # latest_value.append(statistics.max)
                      
                        latest_value.append(statistics)
            try:
                latest_value = min(latest_value)
            except Exception as e:
                self.logger.exception("Error reading parquet for latest value. Deleting local persisted")
                self.flush_local_persisted(flush_only_time_series=False)
                raise e

        else:
            latest_value = None

        return latest_value

    def get_latest_value(self):
        from pyarrow.lib import ArrowInvalid
        if self.time_serie_exist() is True:
            latest_value = []
            dataset = pq.ParquetDataset(self.local_parquet_file)
            schema=dataset.partitioning.schema
            column_index=1
            if "asset_symbol" in dataset.partitioning.schema.pandas_metadata["index_columns"]:
                if "asset_symbol" not in dataset.partitioning.schema.names:
                    column_index=2
            for path, subdirs, files in os.walk(self.local_parquet_file):
                for name in files:
                    if ".parquet" in name:
                        try:
                            meta_data = pq.read_metadata(os.path.join(path, name))

                        except ArrowInvalid as e:
                            self.logger.exception(f"Error reading parquet {path} {name} for latest value. Deleting local persisted")
                            self.flush_local_persisted(flush_only_time_series=False)
                            raise e
                        except Exception as e:
                            raise e
                        
                        statistics = max(
                            [meta_data.row_group(r).column(meta_data.num_columns - column_index).statistics.max for r in
                             range(meta_data.num_row_groups)])
                        # statistics = meta_data.row_group(meta_data.num_row_groups - 1).column(meta_data.num_columns - 1).statistics
                        # latest_value.append(statistics.max)
                        latest_value.append(statistics)
            try:
                latest_value = max(latest_value)
            except Exception as e:
                self.logger.exception("Error reading parquet for latest value. Deleting local persisted")
                self.flush_local_persisted(flush_only_time_series=False)
                raise e

        else:
            latest_value = None

        return latest_value




def verify_parquet_consistency(metadata:dict,source_table_configuration:dict,
                               logger,check_duplicated=True):
    """

    Returns
    -------

    """

    hash_id=metadata["local_hash_id"]

    asset_symbols=None
    if "asset_list" in metadata["build_configuration"]:
        asset_symbols=[a['symbol'] for a in metadata["build_configuration"]['asset_list']['model_list']]

    local_retention_policy=DEFAULT_RETENTION_POLICY
    data_folder = f"{ogm.time_series_folder}/schedulers_local/{local_retention_policy['scheduler_name']}/{hash_id}"
    local_parquet_manager = ParquetLocalPersistManager(data_folder=data_folder,logger=logger,
                                                            retention_policy=local_retention_policy[
                                                                "retention_policy_time"])

    last_index_value=source_table_configuration['last_time_index_value']
    dth = DynamicTableHelpers()
    if last_index_value is not None :
        last_index_value = dth.request_to_datetime(last_index_value)
        parquet_latest_value=local_parquet_manager.get_latest_value()
        if parquet_latest_value is not None and check_duplicated==True:
            data=pd.read_parquet(local_parquet_manager.local_parquet_file)
            if data.index.duplicated().sum()>0:
                local_parquet_manager.flush_local_persisted(flush_only_time_series=False)
                parquet_latest_value=None
                logger.warning(f"Parquet file not in sync contains duplicated flushing {hash_id}")
        if parquet_latest_value is not None:
            if last_index_value == parquet_latest_value: 
                return None
            
            if abs((last_index_value - parquet_latest_value).total_seconds())<.1:
                return None
            #not equal floush parquet
            
            logger.warning(f"Parquet file not in sync deleting {hash_id}")
            local_parquet_manager.flush_local_persisted(flush_only_time_series=False)

        #build local parquet
        logger.info(f"Building Local Parquet for local_hash_id {hash_id}")
        target_value = local_parquet_manager.retention_policy_helper.data_lower_bound(last_index_value)



        temp_df =    dth.get_data_by_time_index(start_date=target_value, 
                                                # direct_to_db=True,
                                                metadata=metadata['remote_table'],)
        if asset_symbols is not None:
            bmrks=[c for c in temp_df.index.get_level_values("asset_symbol").unique() if "benchmark" in c]
            asset_symbols=list(set(asset_symbols+bmrks))
            temp_df = temp_df[temp_df.index.get_level_values("asset_symbol").isin(asset_symbols)]

        if temp_df is None:
            return None
        try:
            local_parquet_manager.persist_updated_data(temp_df=temp_df, 
                                                            overwrite=True)
        except Exception as e:
            local_parquet_manager.flush_local_persisted(flush_only_time_series=False)
            raise Exception("Could not persist local parquet")



class TimeScaleLocalPersistManager:
    """
    Main Controler to interacti with TimeSerie ORM
    """




    @staticmethod
    def batch_data_persisted(hash_id_list: list):

        exist = {}
        dth = DynamicTableHelpers()
        in_db, _ = dth.exist(hash_id__in=hash_id_list)

        for t in hash_id_list:

            if t in in_db:
                exist[t] = True
            else:
                exist[t] = False

        return exist

    def __init__(self, local_hash_id: str,remote_table_hashed_name:Union[str,None],class_name:str,    persist_parquet:bool,
                 logger,description:str, human_readable: Union[str, None] = None,metadata:Union[dict,None]=None,
                 local_metadata:Union[dict,None]=None
                
                ):

        self.local_hash_id = local_hash_id
        self.description = description
        if local_metadata is not None and metadata is None:
            #query remote hash_id
            metadata = local_metadata["remote_table"]


        self.remote_table_hashed_name = remote_table_hashed_name
        self.logger = logger if logger is not None else console_logger("timescale_persist_manager", application_name="tdag")
        self.dth = DynamicTableHelpers(logger=logger)
        self.remote_build_metadata = {}
        self.table_model_loaded = False
        self.human_readable = human_readable if human_readable is not None else local_hash_id

        self.persist_parquet = persist_parquet
        self.class_name = class_name
        if self.local_hash_id is not None:
            self.synchronize_metadata(meta_data=metadata,
                                      local_metadata=local_metadata,
                                      class_name=class_name)
            self.local_retention_policy = DEFAULT_RETENTION_POLICY
            self._set_local_parquet_manager()
            # self._persist_local_parquet_manager(None, None)

    

    def depends_on_connect(self,new_ts:"TimeSerie"):
        """
        Connects a time Serie as relationship in the DB
        Parameters
        ----------
        new_ts :

        Returns
        -------

        """
        try:
            human_readable=new_ts.local_persist_manager.metadata['human_readable']
        except KeyError:
            human_readable=new_ts.human_readable
        self.dth.depends_on_connect(source_hash_id=self.metadata["hash_id"],
                                    target_hash_id=new_ts.remote_table_hashed_name,
                                    source_local_hash_id=self.local_metadata["local_hash_id"],
                                    target_local_hash_id=new_ts.local_hash_id,

                                    target_class_name=new_ts.__class__.__name__,
                                    target_human_readable=human_readable
                                    )


    def display_mermaid_dependency_diagram(self):
        from IPython.core.display import display, HTML, Javascript

        response=TimeSerieLocalUpdate.get_mermaid_dependency_diagram(local_hash_id=self.local_hash_id)
        from IPython.core.display import display, HTML, Javascript
        mermaid_chart=response.get("mermaid_chart")
        metadata=response.get("metadata")
        # Render Mermaid.js diagram with metadata display
        html_template = f"""
        <div class="mermaid">
        {mermaid_chart}
        </div>
        <div id="metadata-display" style="margin-top: 20px; font-size: 16px; color: #333;"></div>
        <script>
            // Initialize Mermaid.js
            if (typeof mermaid !== 'undefined') {{
                mermaid.initialize({{ startOnLoad: true }});
            }}

            // Metadata dictionary
            const metadata = {metadata};

            // Attach click listeners to nodes
            document.addEventListener('click', function(event) {{
                const target = event.target.closest('div[data-graph-id]');
                if (target) {{
                    const nodeId = target.dataset.graphId;
                    const metadataDisplay = document.getElementById('metadata-display');
                    if (metadata[nodeId]) {{
                        metadataDisplay.innerHTML = "<strong>Node Metadata:</strong> " + metadata[nodeId];
                    }} else {{
                        metadataDisplay.innerHTML = "<strong>No metadata available for this node.</strong>";
                    }}
                }}
            }});
        </script>
        """

        return mermaid_chart


    def get_all_local_dependencies(self):
        depth_df=TimeSerieLocalUpdate.get_all_dependencies(hash_id=self.local_hash_id)
        return depth_df

    def set_ogm_dependencies_linked(self):
        
        TimeSerieLocalUpdate.set_ogm_dependencies_linked(hash_id=self.local_hash_id)

    def _set_local_parquet_manager(self):
        """
        Sets the local parquet manager
        Returns
        -------

        """

        data_folder = f"{ogm.time_series_folder}/schedulers_local/{self.local_retention_policy['scheduler_name']}/{self.local_hash_id}"
        self.local_parquet_manager = ParquetLocalPersistManager(data_folder=data_folder,logger=self.logger,
                                                                retention_policy=self.local_retention_policy[
                                                                    "retention_policy_time"])

    @property
    def update_details(self):
        if "localtimeserieupdatedetails" in self.local_metadata.keys():
            return self.local_metadata['localtimeserieupdatedetails']
        return None
    @property
    def source_table_configuration(self):
        if "sourcetableconfiguration" in self.metadata.keys():
            return self.metadata['sourcetableconfiguration']
        return None
    def update_source_informmation(self, git_hash_id:str, source_code:str):
        """

        Args:
            git_hash_id:
            source_code:

        Returns:

        """

        self.metadata = self.dth.patch(metadata=self.metadata, time_serie_source_code_git_hash=git_hash_id,
                            time_serie_source_code=source_code,)

    def set_last_index_value(self):
        return TimeSerieLocalUpdate.set_last_update_index_time(metadata=self.local_metadata)


    def synchronize_metadata(self, meta_data:Union[dict,None], local_metadata:Union[dict,None],
                             set_last_index_value: bool = False,
                             class_name:Union[str,None]=None
                             ):
        """
        forces a synchronization between table and metadata
        :return:
        """
        #start with remote metadata
        if set_last_index_value == True:
            TimeSerieLocalUpdate.set_last_update_index_time(metadata=self.local_metadata)
        if meta_data is  None or local_metadata is None: #avoid calling 2 times the DB
            meta_data={}
            try:
                local_metadata={}# set to empty in case not exist
                local_metadata= TimeSerieLocalUpdate.get(local_hash_id=self.local_hash_id)
                if len(local_metadata)==0:
                    raise LocalTimeSeriesDoesNotExist
            except LocalTimeSeriesDoesNotExist:
                # could be localmetadata is none but table could exist
                try:
                    meta_data = self.dth.get(hash_id=self.remote_table_hashed_name)
                except DynamicTableDoesNotExist:
                    pass
        
        if len(local_metadata) != 0:
            self.local_build_configuration = local_metadata["build_configuration"]
            self.local_build_metadata =  local_metadata["build_meta_data"]
            self.local_metadata = local_metadata
            
            #metadata should always exist
            meta_data = local_metadata["remote_table"]

        if len(meta_data) != 0:    
            remote_build_configuration, remote_build_metadata = meta_data["build_configuration"], meta_data[
                "build_meta_data"]
            self.remote_build_configuration = remote_build_configuration
            self.remote_build_metadata = remote_build_metadata
            self.metadata = meta_data

    def add_tags(self,tags:list):
        if any([t not in self.local_metadata["tags"] for t in tags]) == True:
            TimeSerieLocalUpdate.add_tags(tags=tags, local_metadata=self.local_metadata)

    def destroy(self, delete_only_table: bool):
        self.dth.destroy(metadata=self.metadata, delete_only_table=delete_only_table)

    @property
    def persist_size(self):
        try:
            return self.metadata['table_size']
        except KeyError:
            return 0

    def parse_data_update_if_persisted(self, temp_df, latest_value):
        return temp_df

    def time_serie_exist_in_db(self):
        return self.dth.time_serie_exist_in_db(self.remote_table_hashed_name)

    def metadata_registered_in_db(self):
        return self.dth.get(hash_id=self.remote_table_hashed_name)

    def time_serie_exist(self):
        """
        Verifies if time series exist in DB
        Returns
        -------

        """
        return self.time_serie_exist_in_db()

    def get_persisted_ts(self):
        """
        full Request of the persisted data should always default to DB
        :return:
        """

        persisted_df = self.dth.get_data_by_time_index(metadata=self.metadata)

        return persisted_df

    def get_df_greater_than_in_table(self, target_value: datetime.datetime, great_or_equal: bool, force_db_look=False,
                            symbol_list:Union[list,None]=None,
                            columns: Union[list, None] = None
                            ):
        """

        Parameters
        ----------
        target_value
        great_or_equal

        Returns
        -------

        """

        time_serie_exist = self.local_parquet_manager.time_serie_exist() if hasattr(self,
                                                                                    "local_parquet_manager") else False
        earliest_value = None
        if time_serie_exist == True and target_value is not None and force_db_look == False:
            try:
                earliest_value = self.local_parquet_manager.get_earliest_value()
                if target_value > earliest_value:
                    # latest_value = self.get_latest_value() # do not check on quering
                    parquet_latest_value = self.local_parquet_manager.get_latest_value()

                    filtered_data = self.local_parquet_manager.get_df_greater_than_in_table(target_value=target_value,
                                                                                   great_or_equal=great_or_equal,
                                                                                   columns=columns,
                                                                                   asset_symbols=symbol_list,
                                                                                   time_index_name=self.metadata['sourcetableconfiguration']['time_index_name']
                                                                                   )
                    return filtered_data
            except Exception as e:

                self.logger.exception(
                    f"{self.remote_table_hashed_name} {self.local_parquet_manager.local_parquet_file} error getting data {earliest_value}")
        # if start date is no after earlier local retetion default ot DB

        filtered_data = self.dth.get_data_by_time_index(start_date=target_value, metadata=self.metadata,
                                                        columns=columns,
                                                        asset_symbols=symbol_list,
                                                        great_or_equal=great_or_equal, )
        if force_db_look == False:
            self.logger.warning(
                f"""Data is not been pulled from local storage, review  storage policy to improve performace {target_value}
                time_serie_exist: {time_serie_exist}
                force_db_look :{force_db_look}
                earliest_value in parquet :{earliest_value}
                local retention policy : {self.local_retention_policy}
                traceback 
                """)

        return filtered_data

    def filter_by_assets_ranges(self, asset_ranges_map: dict, force_db_look: bool):

        if force_db_look:

            assert "asset_symbol" in self.metadata["sourcetableconfiguration"]["index_names"],"Table does not contain asset_symbol column"
            connection_config=DynamicTableDataSource.get_data_source_connection_details(self.metadata["data_source"]["id"])

            df=self.dth.filter_by_assets_ranges(table_name=self.metadata['hash_id'],asset_ranges_map=asset_ranges_map,

                                                connection_config=connection_config)
            df["time_index"]=pd.to_datetime(df["time_index"])
            df=df.set_index(self.metadata["sourcetableconfiguration"]["index_names"])
            
            
        else:
            raise NotImplementedError
        return df
    def get_df_between_dates(self, start_date, end_date, force_db_look=False, great_or_equal=True,
                                      less_or_equal=True,direct_to_db=False,
                                      asset_symbols: Union[list, None] = None,
                                      columns: Union[list, None] = None):
        return self._get_df_between_dates_from_db(start_date, end_date, force_db_look=force_db_look, great_or_equal=great_or_equal,
                                      less_or_equal=less_or_equal,direct_to_db=direct_to_db,
                                      asset_symbols=asset_symbols,
                                      columns = columns)

    def get_data_source_connection_details(self,override_id:Union[int,None]=None):
        from mainsequence.tdag_client import DynamicTableDataSource
        override_id=override_id or self.metadata['data_source']["id"]

        return DynamicTableDataSource.get_data_source_connection_details(connection_id=override_id)

    def _get_df_between_dates_from_db(self, start_date, end_date, force_db_look=False, great_or_equal=True,
                                      less_or_equal=True,direct_to_db=False,
                                      asset_symbols: Union[list, None] = None,
                                      columns: Union[list, None] = None
                                      ):
        """

        Parameters
        ----------
        start_date
        end_date

        Returns
        -------

        """

        force_db_look = True if start_date is None else force_db_look
        if self.local_parquet_manager.time_serie_exist() == True and force_db_look == False:
            earliest_date, latest_value = self.local_parquet_manager.get_earliest_value(), self.local_parquet_manager.get_latest_value()
            use_local_end_date=True if end_date is None else end_date <= latest_value
            if start_date >= earliest_date and use_local_end_date==True:
                filtered_data = self.local_parquet_manager.get_df_between_dates(start_date=start_date,
                                                                                end_date=end_date,
                                                                                great_or_equal=great_or_equal,
                                                                                less_or_equal=less_or_equal,
                                                                                columns=columns,
                                                                                asset_symbols=asset_symbols,
                                                                                time_index_name=self.metadata[
                                                                                    'sourcetableconfiguration'][
                                                                                    'time_index_name']
                                                                                )
                return filtered_data
        # if start date is no after earlier local retetion default ot DB
        if "id" not in self.metadata.keys():
            raise Exception(f"No id in f{self.metadata}")
        filtered_data = self.dth.get_data_by_time_index(metadata=self.metadata, start_date=start_date,
                                                        end_date=end_date, great_or_equal=great_or_equal,
                                                        less_or_equal=less_or_equal,direct_to_db=direct_to_db,
                                                        asset_symbols=asset_symbols,
                                                        columns=columns
                                                        )
        self.logger.warning(
            f"Data is not been pulled from local storage, review  storage policy to improve performace {start_date} - {end_date}")
        return filtered_data

    def patch_build_configuration(self,local_configuration:dict,remote_configuration:dict):
        """

        Args:
            local_configuration:
            remote_configuration:

        Returns:

        """

        remote_build_metadata = remote_configuration[
            "build_meta_data"] if "build_meta_data" in remote_configuration.keys() else {}
        remote_configuration.pop("build_meta_data", None)
        kwargs = dict(hash_id=self.remote_table_hashed_name,
                      build_configuration=remote_configuration, build_meta_data=remote_build_metadata)

        local_build_metadata = local_configuration[
            "build_meta_data"] if "build_meta_data" in local_configuration.keys() else {}
        local_configuration.pop("build_meta_data", None)
        local_metadata_kwargs = dict(local_hash_id=self.local_hash_id,
                               build_configuration=local_configuration, build_meta_data=local_build_metadata,
                               remote_table__hash_id=self.remote_table_hashed_name)


        TimeSerieNode.patch_build_configuration(remote_table_patch=kwargs,local_table_patch=local_metadata_kwargs)

    def local_persist_exist_set_config(self, local_configuration:dict, remote_configuration:dict,data_source:dict,
                                       time_serie_source_code_git_hash:str, time_serie_source_code:str):
        """
        This method runs on initialization of the TimeSerie class. We also use it to retrieve the table if
        is already persisted
        :param config:

        :return:
        """
        remote_build_configuration = None
        if hasattr(self, "remote_build_configuration"):
            remote_build_configuration, remote_build_metadata = self.remote_build_configuration, self.remote_build_metadata

        remote_table_exist = True
        if remote_build_configuration is None:
            #create remote table
            remote_table_exist = False
            try:

                # table may not exist but
                remote_build_metadata = remote_configuration["build_meta_data"] if "build_meta_data" in remote_configuration.keys() else {}
                remote_configuration.pop("build_meta_data", None)
                kwargs = dict(hash_id=self.remote_table_hashed_name,
                              time_serie_source_code_git_hash=time_serie_source_code_git_hash,
                              time_serie_source_code=time_serie_source_code,
                              build_configuration=remote_configuration,
                              data_source=data_source.model_dump(),
                              build_meta_data=remote_build_metadata)
                if self.human_readable is not None:
                    kwargs["human_readable"] = self.human_readable
                # node_kwargs={"hash_id":self.remote_table_hashed_name,
                #                                                  "source_class_name":self.class_name,
                #                                                  "human_readable": self.human_readable,
                #
                #                                                  }


                # kwargs["source_class_name"]=self.class_name
                self.metadata = self.dth.create(metadata_kwargs=kwargs)

                #after creating metadata always delete local parquet manager even if not exist
                self.delete_local_parquet()

            except Exception as e:
                self.logger.exception(f"{self.remote_table_hashed_name} Could not set meta data in DB for P")
                raise e
        # check if link to local update exists

        local_table_exist = self._verify_local_ts_exists(local_configuration=local_configuration)

        return remote_table_exist,local_table_exist

    def _verify_local_ts_exists(self,local_configuration:Union[dict,None]=None):
        """
        Verifies that the local time serie exist in ORM
        Parameters
        ----------
        local_configuration

        Returns
        -------

        """
        local_table_exist=True
        local_build_configuration = None
        if hasattr(self, "local_build_configuration"):
            local_build_configuration, local_build_metadata = self.local_build_configuration, self.local_build_metadata
        if local_build_configuration is None:
            local_table_exist=False
            local_update = TimeSerieLocalUpdate.filter(local_hash_id=self.local_hash_id)
            if len(local_update) == 0:
                local_build_metadata = local_configuration[
                    "build_meta_data"] if "build_meta_data" in local_configuration.keys() else {}
                local_configuration.pop("build_meta_data", None)
                metadata_kwargs=dict(local_hash_id=self.local_hash_id,
                              build_configuration=local_configuration, build_meta_data=local_build_metadata,
                              remote_table__hash_id=self.metadata['hash_id'],
                                     description=self.description
                                     )
                if self.human_readable is not None:
                    metadata_kwargs["human_readable"] = self.human_readable

                node_kwargs = {"hash_id": self.local_hash_id,
                               "source_class_name": self.class_name,
                               "human_readable": self.human_readable,

                               }

                local_metadata = TimeSerieLocalUpdate.create(metadata_kwargs=metadata_kwargs,
                                                             node_kwargs=node_kwargs
                                              )
                self.local_build_configuration = local_metadata["build_configuration"]
                self.local_build_metadata = local_metadata["build_meta_data"]
                self.local_metadata = local_metadata
            else:
                local_metadata=local_update
            self.local_metadata=local_metadata
        return   local_table_exist
    def _verify_insertion_format(self,temp_df):
        """
        verifies that data frame is properly configured
        Parameters
        ----------
        temp_df :

        Returns
        -------

        """
        if self.remote_table_hashed_name!=self.local_hash_id:

            assert temp_df.index.names==["time_index","asset_symbol"] or  temp_df.index.names==["time_index","asset_symbol","execution_venue_symbol"]
        if isinstance(temp_df.index,pd.MultiIndex)==False:
            # assert temp_df.index.name is not None
            pass
    def persist_updated_data(self, temp_df: pd.DataFrame,
                             update_tracker:Union[object,None]=None,
                             overwrite=False):
        """
        Main update time series function, it is called from TimeSeries class
        Parameters
        ----------
        temp_df
        latest_value
        session

        Returns
        -------

        """
         #Todo Remove local parquet verification
        # #1 Operations with local parquet file
        # self._verify_insertion_format(temp_df)
        # try:
        #     self.local_parquet_manager.persist_updated_data(temp_df=temp_df,
        #                                                     overwrite=True)
        # except Exception as e:
        #     self.local_parquet_manager.flush_local_persisted(flush_only_time_series=False)
        #     raise e
        #
        # min_value=self.local_parquet_manager.get_earliest_value()
        #
        # idx=temp_df.index.get_level_values(0)
        # if idx.min()>min_value and update_tracker is not None:
        #     #set update complete before peristence on timescale. This will help speed the update process
        #     update_tracker.set_end_of_execution(hash_id=self.local_hash_id,error_on_update=False)
        #2 insert into time scale
        self.add_data_to_timescale(temp_df=temp_df,overwrite=overwrite)
        update_tracker.set_end_of_execution(hash_id=self.local_hash_id, error_on_update=False)
       

    def add_data_to_timescale(self, temp_df: pd.DataFrame,historical_update_id:Union[int,None]=None, overwrite=False):
        """
        Main Insertion Method to Time Scale
        Parameters
        ----------
        temp_df :
        historical_update_id :
        overwrite :

        Returns
        -------

        """

        self.local_metadata = self.dth.upsert_data_into_table(
                                                        metadata=self.metadata,
                                                        local_metadata=self.local_metadata,
                                                        data=temp_df,
                                                        historical_update_id=historical_update_id,
                                                        overwrite=overwrite
                                                        )

    def upsert_data(self, data_df: pd.DataFrame):

        self.add_data_to_timescale(temp_df=data_df, overwrite=True)


    def local_persist_exist(self):
        return self.time_serie_exist()

    def get_latest_value(self, asset_symbols:list) -> [datetime.datetime,Dict[str, datetime.datetime]]:

        metadata= self.dth.get(hash_id=self.remote_table_hashed_name)

        last_index_value,last_multiindex=None,None
        if "sourcetableconfiguration" in metadata.keys():
            if metadata['sourcetableconfiguration'] is not None:
                last_index_value=metadata['sourcetableconfiguration']['last_time_index_value']
                if last_index_value is None:
                    return last_index_value,last_multiindex
                last_index_value=self.dth.request_to_datetime(last_index_value)
                if metadata['sourcetableconfiguration']['multi_index_stats'] is not None:
                    last_multiindex=metadata['sourcetableconfiguration']['multi_index_stats']['max_per_asset_symbol']
                    if last_multiindex is not None:
                        last_multiindex={symbol:{ev:self.dth.request_to_datetime(v) for ev,v in ev_dict.items()} for symbol,ev_dict in last_multiindex.items()}

        if asset_symbols is not None and last_multiindex is not None:
            last_multiindex = {asset: value for asset, value in last_multiindex.items() if asset in asset_symbols}

        return last_index_value,last_multiindex

    def get_earliest_value(self) -> datetime.datetime:
        earliest_value = self.dth.get_earliest_value(hash_id=self.remote_table_hashed_name)
        return earliest_value

    def get_full_source_data(self, engine="pandas"):
        """
        Returns full stored data, uses multiprocessing to achieve several queries by rows and speed
        :return:
        """

        from joblib import Parallel, delayed
        from tqdm import tqdm

        metadata = self.dth.get_configuration(hash_id=self.remote_table_hashed_name)
        earliest_obs = metadata["sourcetableconfiguration"]["last_time_index_value"]
        latest_value = metadata["sourcetableconfiguration"]["earliest_index_value"]

        ranges = list(pd.date_range(earliest_obs, latest_value, freq="1 m"))

        if earliest_obs not in ranges:
            ranges = [earliest_obs] + ranges

        if latest_value not in ranges:
            ranges.append(latest_value)

        def get_data(ranges, i, metadata, dth):

            s, e = ranges[i], ranges[i + 1]
            tmp_data = dth.get_data_by_time_index(start_date=s, end_date=e, metadata=self.metadata,
                                                  great_or_equal=True, less_or_equal=False)

            tmp_data = tmp_data.reset_index()
            return tmp_data

        dfs = Parallel(n_jobs=10)(
            delayed(get_data)(ranges, i, self.remote_table_hashed_name, engine) for i in tqdm(range(len(ranges) - 1)))
        dfs = pd.concat(dfs, axis=0)
        dfs = dfs.set_index(self.metadata["table_config"]["index_names"])
        return dfs


    def set_policy(self, interval: str,overwrite:bool,comp_type:str):
        if self.metadata is not None:
            retention_policy_config = self.metadata["retention_policy_config"]
            compression_policy_config=self.metadata["compression_policy_config"]
            if comp_type =="retention":
                if retention_policy_config is None or overwrite == True:
                    status = self.dth.set_retention_policy(interval=interval, metadata=self.metadata)
            if comp_type=="compression":
                if compression_policy_config is None or overwrite == True:
                    status = self.dth.set_compression_policy(interval=interval, metadata=self.metadata)
        else:
            self.logger.warning("Retention policy couldnt be set as TS is not yet persisted")
    def set_policy_for_descendants(self,policy,comp_type:str,exclude_ids:Union[list,None]=None,extend_to_classes=False):
        self.dth.set_policy_for_descendants(hash_id=self.remote_table_hashed_name,pol_type=comp_type,policy=policy,
                                            exclude_ids=exclude_ids,extend_to_classes=extend_to_classes)



    def delete_after_date(self, after_date: str):
        self.dth.delete_after_date(metadata=self.metadata, after_date=after_date)

    def delete_local_parquet(self):
        self.local_parquet_manager.flush_local_persisted(flush_only_time_series=False)

  







    def update_details_exist(self):
        """

        Returns
        -------

        """
        exist= TimeSerieLocalUpdate.update_details_exist(local_metadata=self.local_metadata)
        return exist
    def build_update_details(self,source_class_name):
        """

        Returns
        -------

        """

        update_kwargs=dict(source_class_name=source_class_name,
                           local_metadata=self.local_metadata,
                           )


        metadatas=self.dth.build_or_update_update_details(metadata=self.metadata,
                                                **update_kwargs)

        self.metadata = metadatas["metadata"]
        self.local_metadata = metadatas["local_metadata"]

    def patch_update_details(self,local_hash_id=None,
                             **kwargs):
        """
        Patch update details ofr related_table
        Parameters
        ----------
        hash_id :
        kwargs :

        Returns
        -------

        """
        if local_hash_id is not None:
            kwargs["use_local_hash_id"]=local_hash_id
            metadata=self.dth.build_or_update_update_details(metadata=self.metadata,**kwargs)
            return metadata
        kwargs["local_metadata"]=self.local_metadata
        metadatas=self.dth.build_or_update_update_details(metadata=self.metadata,**kwargs)
        self.metadata=metadatas["metadata"]
        self.local_metadata=metadatas["local_metadata"]

    def patch_table(self,**kwargs):
        self.dth.patch(metadata=self.metadata, **kwargs)

    def protect_from_deletion(self,protect_from_deletion=True):
        self.dth.patch(metadata=self.metadata, protect_from_deletion=protect_from_deletion)

    def open_for_everyone(self,open_for_everyone=True):
        self.dth.patch(metadata=self.metadata, open_for_everyone=open_for_everyone)

    def set_start_of_execution(self,**kwargs):
        return self.dth.set_start_of_execution(metadata=self.metadata,**kwargs)
    def set_end_of_execution(self,**kwargs):
        return self.dth.set_end_of_execution(metadata=self.metadata, **kwargs)
    def reset_dependencies_states(self,hash_id_list):
        return self.dth.reset_dependencies_states(metadata=self.metadata, hash_id_list=hash_id_list)
    def get_pending_nodes(self,table_id_list:list, filter_by_update_time:bool ):

        """

        Parameters
        ----------
        filter_by_update_time :

        Returns
        -------

        """
        request_kwargs=dict(table_id_list=table_id_list,   filter_by_update_time=filter_by_update_time)
        data=self.dth.get_pending_nodes(metadata=self.metadata,**request_kwargs)
        return data["dependecies_updated"],data['pending_nodes'],data["next_rebalances"],data["error_on_dependencies"]