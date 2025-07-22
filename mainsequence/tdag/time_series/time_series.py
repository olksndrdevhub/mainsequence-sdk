import datetime
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Callable
import json
import time
import traceback
import pytz
import inspect
import logging
import copy
import cloudpickle
from dataclasses import asdict
from mainsequence.client import  Scheduler
from mainsequence.instrumentation import tracer
from mainsequence.tdag.config import (
    ogm
)
import tempfile
import structlog.contextvars as cvars
from mainsequence.logconf import logger

from mainsequence.tdag.time_series.persist_managers import PersistManager, APIPersistManager
from mainsequence.client.models_tdag import (DataSource,
                                             UpdateStatistics, UniqueIdentifierRangeMap, ColumnMetaData,      )


from abc import ABC

from typing import Union

from mainsequence.client import LocalTimeSerie,  CONSTANTS, \
    DynamicTableDataSource, AssetTranslationTable

from functools import wraps

import mainsequence.client as ms_client
import mainsequence.tdag.time_series.run_operations as run_operations
import mainsequence.tdag.time_series.build_operations as build_operations


def get_data_source_from_orm() -> Any:
    from mainsequence.client import SessionDataSource
    if SessionDataSource.data_source.related_resource is None:
        raise Exception("This Pod does not have a default data source")
    return SessionDataSource.data_source

def get_latest_update_by_assets_filter(asset_symbols: Optional[list], last_update_per_asset: dict) -> datetime.datetime:
    """
    Gets the latest update timestamp for a list of asset symbols.

    Args:
        asset_symbols: A list of asset symbols.
        last_update_per_asset: A dictionary mapping assets to their last update time.

    Returns:
        The latest update timestamp.
    """
    if asset_symbols is not None:
        last_update_in_table = np.max([timestamp for unique_identifier, timestamp in last_update_per_asset.items()
                                       if unique_identifier in asset_symbols
                                       ])
    else:
        last_update_in_table = np.max(last_update_per_asset.values)
    return last_update_in_table



def last_update_per_unique_identifier(unique_identifier_list: Optional[list],
                                      last_update_per_asset: dict) -> datetime.datetime:
    """
    Gets the earliest last update time for a list of unique identifiers.

    Args:
        unique_identifier_list: A list of unique identifiers.
        last_update_per_asset: A dictionary mapping assets to their last update times.

    Returns:
        The earliest last update timestamp.
    """
    if unique_identifier_list is not None:
        last_update_in_table = min(
            [t for a in last_update_per_asset.values() for t in a.values() if a in unique_identifier_list])
    else:
        last_update_in_table = min([t for a in last_update_per_asset.values() for t in a.values()])
    return last_update_in_table







class DataAccessMixin:
    """A mixin for classes that provide access to time series data."""

    def __repr__(self) -> str:
        try:
            local_id = self.local_time_serie.id
        except:
            local_id = 0
        repr = self.__class__.__name__ + f" {os.environ['TDAG_ENDPOINT']}/local-time-series/details/?local_time_serie_id={local_id}"
        return repr

    def get_logger_context_variables(self) -> Dict[str, Any]:
        return dict(local_hash_id=self.local_hash_id,
                    local_hash_id_data_source=self.data_source_id,
                    api_time_series=self.__class__.__name__ == "APITimeSerie")

    @property
    def logger(self) -> logging.Logger:
        """Gets a logger instance with bound context variables."""
        # import structlog.contextvars as cvars
        # cvars.bind_contextvars(local_hash_id=self.local_hash_id,
        #                      local_hash_id_data_source=self.data_source_id,
        #                      api_time_series=True,)
        global logger
        if hasattr(self, "_logger") == False:
            cvars.bind_contextvars(**self.get_logger_context_variables() )
            self._logger = logger

        return self._logger
    @staticmethod
    def set_context_in_logger(logger_context: Dict[str, Any]) -> None:
        """
        Binds context variables to the global logger.

        Args:
            logger_context: A dictionary of context variables.
        """
        global logger
        for key, value in logger_context.items():
            logger.bind(**dict(key=value))

    def unbind_context_variables_from_logger(self) -> None:
        cvars.unbind_contextvars(*self.get_logger_context_variables().keys())

    def get_df_between_dates(
        self,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        unique_identifier_list: Optional[list] = None,
        great_or_equal: bool = True,
        less_or_equal: bool = True,
        unique_identifier_range_map: Optional[UniqueIdentifierRangeMap] = None,
    ) -> pd.DataFrame:
        """
        Retrieve rows from this TimeSerie whose `time_index` (and optional `unique_identifier`) fall within the specified date ranges.

        **Note:** If `unique_identifier_range_map` is provided, **all** other filters
        (`start_date`, `end_date`, `unique_identifier_list`, `great_or_equal`, `less_or_equal`)
        are ignored, and only the per-identifier ranges in `unique_identifier_range_map` apply.

        Filtering logic (when `unique_identifier_range_map` is None):
          - If `start_date` is provided, include rows where
            `time_index > start_date` (if `great_or_equal=False`)
            or `time_index >= start_date` (if `great_or_equal=True`).
          - If `end_date` is provided, include rows where
            `time_index < end_date` (if `less_or_equal=False`)
            or `time_index <= end_date` (if `less_or_equal=True`).
          - If `unique_identifier_list` is provided, only include rows whose
            `unique_identifier` is in that list.

        Filtering logic (when `unique_identifier_range_map` is provided):
          - For each `unique_identifier`, apply its own `start_date`/`end_date`
            filters using the specified operands (`">"`, `">="`, `"<"`, `"<="`):
            {
              <uid>: {
                "start_date": datetime,
                "start_date_operand": ">=" or ">",
                "end_date": datetime,
                "end_date_operand": "<=" or "<"
              },
              ...
            }

        Parameters
        ----------
        start_date : datetime.datetime or None
            Global lower bound for `time_index`. Ignored if `unique_identifier_range_map` is provided.
        end_date : datetime.datetime or None
            Global upper bound for `time_index`. Ignored if `unique_identifier_range_map` is provided.
        unique_identifier_list : list or None
            If provided, only include rows matching these IDs. Ignored if `unique_identifier_range_map` is provided.
        great_or_equal : bool, default True
            If True, use `>=` when filtering by `start_date`; otherwise use `>`. Ignored if `unique_identifier_range_map` is provided.
        less_or_equal : bool, default True
            If True, use `<=` when filtering by `end_date`; otherwise use `<`. Ignored if `unique_identifier_range_map` is provided.
        unique_identifier_range_map : UniqueIdentifierRangeMap or None
            Mapping of specific `unique_identifier` keys to their own sub-filters. When provided, this is the sole filter applied.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing rows that satisfy the combined time and identifier filters.
        """
        return self.local_persist_manager.get_df_between_dates(
            start_date=start_date,
            end_date=end_date,
            unique_identifier_list=unique_identifier_list,
            great_or_equal=great_or_equal,
            less_or_equal=less_or_equal,
            unique_identifier_range_map=unique_identifier_range_map,
        )



    def get_ranged_data_per_asset(self, range_descriptor: Optional[UniqueIdentifierRangeMap]) -> pd.DataFrame:
        """
        Gets data based on a range descriptor.

        Args:
            range_descriptor: A UniqueIdentifierRangeMap object.

        Returns:
            A DataFrame with the ranged data.
        """
        return  self.get_df_between_dates(unique_identifier_range_map=range_descriptor)

    def filter_by_assets_ranges(self, asset_ranges_map: dict) -> pd.DataFrame:
        """
               Filters data by asset ranges.

               Args:
                   asset_ranges_map: A dictionary mapping assets to their date ranges.

               Returns:
                   A DataFrame with the filtered data.
               """
        return self.local_persist_manager.filter_by_assets_ranges(asset_ranges_map)



    def get_last_observation(self) -> Optional[pd.DataFrame]:
        """
        Gets the last observation from the time series.

        Args:
            unique_identifier_list: An optional list of unique identifiers to filter by.

        Returns:
            A DataFrame with the last observation, or None if not found.
        """
        update_statistics = self.get_update_statistics()
        if update_statistics.is_empty():
            return None
        # todo request specific endpoint
        return self.get_df_between_dates(
            start_date=update_statistics.max_time_index_value,
            great_or_equal=True,
        )

    @property
    def last_observation(self) -> Optional[pd.DataFrame]:
        """The last observation(s) in the time series."""
        return self.get_last_observation()





class APITimeSerie(DataAccessMixin):


    @classmethod
    def build_from_local_time_serie(cls, source_table: "LocalTimeSerie") -> "APITimeSerie":
        return cls(data_source_id=source_table.data_source.id,
                   source_table_hash_id=source_table.hash_id
                   )

    @classmethod
    def build_from_unique_identifier(cls, unique_identifier: str) -> "APITimeSerie":
        from mainsequence.client import MarketsTimeSeriesDetails
        tdag_api_data_source = MarketsTimeSeriesDetails.get(unique_identifier=unique_identifier)
        ts = cls(
            data_source_id=tdag_api_data_source.source_table.data_source,
            source_table_hash_id=tdag_api_data_source.source_table.hash_id
        )
        return ts

    def __init__(self,
                 data_source_id: int, source_table_hash_id: str,
                 data_source_local_lake: Optional[DataSource] = None):
        """
        Initializes an APITimeSerie.

        Args:
            data_source_id: The ID of the data source.
            local_hash_id: The local hash ID of the time series.
            data_source_local_lake: Optional local data source for the lake.
        """
        if data_source_local_lake is not None:
            assert data_source_local_lake.data_type in CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE, "data_source_local_lake should be of type CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE"

        self.data_source_id = data_source_id
        self.source_table_hash_id = source_table_hash_id
        self.data_source = data_source_local_lake
        self._local_persist_manager: APIPersistManager = None

    @property
    def is_api(self):
        return True

    @staticmethod
    def _get_local_hash_id(hash_id):
        return "API_"+f"{hash_id}"
    @property
    def local_hash_id(self):
        return  self._get_local_hash_id(hash_id=self.source_table_hash_id)

    def __getstate__(self) -> Dict[str, Any]:
        """Prepares the state for pickling."""
        state = self.__dict__.copy()
        # Remove unpicklable/transient state specific to APITimeSerie
        names_to_remove = [
            "_local_persist_manager", # APIPersistManager instance
        ]
        cleaned_state = {k: v for k, v in state.items() if k not in names_to_remove}
        return cleaned_state

    @property
    def local_persist_manager(self) -> Any:
        """Gets the local persistence manager, initializing it if necessary."""
        if self._local_persist_manager is None:
            self._set_local_persist_manager()
            self.logger.debug(f"Setting local persist manager for {self.source_table_hash_id}")
        return self._local_persist_manager

    def set_relation_tree(self) -> None:
        pass  # do nothing  for API Time Series

    def _verify_local_data_source(self) -> None:
        """Verifies and sets the local data source from environment variables if available."""
        pod_source = os.environ.get("POD_DEFAULT_DATA_SOURCE", None)
        if pod_source != None:
            from mainsequence.client import models as models
            pod_source = json.loads(pod_source)
            ModelClass = pod_source["tdag_orm_class"]
            pod_source.pop("tdag_orm_class", None)
            ModelClass = getattr(models, ModelClass)
            pod_source = ModelClass(**pod_source)
            self.data_source = pod_source

    def build_data_source_from_configuration(self, data_config: Dict[str, Any]) -> DataSource:
        """
        Builds a data source object from a configuration dictionary.

        Args:
            data_config: The data source configuration.

        Returns:
            A DataSource object.
        """
        ModelClass = DynamicTableDataSource.get_class(data_config['data_type'])
        pod_source = ModelClass.get(data_config["id"])
        return pod_source

    def _set_local_persist_manager(self) -> None:
        self._verify_local_data_source()
        self._local_persist_manager = APIPersistManager(source_table_hash_id=self.source_table_hash_id, data_source_id=self.data_source_id)
        metadata = self._local_persist_manager.metadata

        assert metadata is not None, f"Verify that the table {self.source_table_hash_id} exists "



    @property
    def pickle_path(self) -> str:
        pp = data_source_dir_path(self.data_source_id)
        path = f"{pp}/{self.PICKLE_PREFIFX}{self.local_hash_id}.pickle"
        return path

    @classmethod
    def get_pickle_path(cls, source_table_hash_id: str, data_source_id: int) -> str:
        return f"{ogm.pickle_storage_path}/{data_source_id}/{cls.PICKLE_PREFIFX}{cls._get_local_hash_id()}.pickle"

    def persist_to_pickle(self, overwrite: bool = False) -> Tuple[str, str]:
        path = self.pickle_path
        # after persisting pickle , build_hash and source code need to be patched
        self.logger.debug(f"Persisting pickle")

        pp = build_operations.data_source_pickle_path(self.data_source_id)
        if os.path.isfile(pp) == False or overwrite == True:
            self.data_source.persist_to_pickle(pp)

        if os.path.isfile(path) == False or overwrite == True:
            with open(path, 'wb') as handle:
                cloudpickle.dump(self, handle)
        return path, path.replace(ogm.pickle_storage_path + "/", "")

    def get_update_statistics(self, asset_symbols: Optional[list] = None) -> Tuple[Optional[datetime.datetime], Optional[Dict[str, datetime.datetime]]]:
        """
        Gets update statistics from the database.

        Args:
            asset_symbols: An optional list of asset symbols to filter by.

        Returns:
            A tuple containing the last update time for the table and a dictionary of last update times per asset.
        """
        last_update_in_table, last_update_per_asset = self.local_persist_manager.get_update_statistics(
            remote_table_hash_id=self.remote_table_hashed_name,
            asset_symbols=asset_symbols, time_serie=self)
        return UpdateStatistics(
        max_time_index_value=last_update_in_table,
        max_time_per_identifier=last_update_per_asset or {}
    )

    def get_earliest_updated_asset_filter(self, unique_identifier_list: list,
                                          last_update_per_asset: dict) -> datetime.datetime:
        """
        Gets the earliest last update time for a list of unique identifiers.

        Args:
            unique_identifier_list: A list of unique identifiers.
            last_update_per_asset: A dictionary mapping assets to their last update times.

        Returns:
            The earliest last update timestamp.
        """
        if unique_identifier_list is not None:
            last_update_in_table = min(
                [t for a in last_update_per_asset.values() for t in a.values() if a in unique_identifier_list])
        else:
            last_update_in_table = min([t for a in last_update_per_asset.values() for t in a.values()])
        return last_update_in_table

    def update(self, *args, **kwargs) -> pd.DataFrame:
        self.logger.info("Not updating series")
        pass

class TimeSerie(DataAccessMixin,ABC):
    """
    Base TimeSerie class
    """
    OFFSET_START = datetime.datetime(2018, 1, 1, tzinfo=pytz.utc)

    def __init__(
            self,
            init_meta: Optional[build_operations.TimeSerieInitMeta] = None,
            build_meta_data: Union[dict, None] = None,
            local_kwargs_to_ignore: Union[List[str], None] = None,
            *args,
            **kwargs):
        """
        Initializes the TimeSerie object with the provided metadata and configurations. For extension of the method

        This method sets up the time series object, loading the necessary configurations
        and metadata.

        Each TimeSerie instance will create a table in the Main Sequence Data Engine by uniquely hashing
        the arguments with exception of:

        - init_meta
        - build_meta_data
        - local_kwargs_to_ignore

        Each TimeSerie instance will create a local_hash_id and a LocalTimeSerie instance in the Data Engine by uniquely hashing
        the same arguments as the table but excluding the arguments inside local_kwargs_to_ignore


        allowed type of arguments can only be str,list, int or  Pydantic objects inlcuding lists of Pydantic Objects.

        The OFFSET_START property can be overridend and markts the minimum date value where the table will insert data

        Parameters
        ----------
        init_meta : dict, optional
            Metadata for initializing the time series instance.
        build_meta_data : dict, optional
            Metadata related to the building process of the time series.
        local_kwargs_to_ignore : list, optional
            List of keyword arguments to ignore during configuration.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """


        self.init_meta = init_meta

        if build_meta_data is None:
            build_meta_data = {"initialize_with_default_partitions": True}

        if "initialize_with_default_partitions" not in build_meta_data.keys():
            build_meta_data["initialize_with_default_partitions"] = True

        self.build_meta_data = build_meta_data
        self.local_kwargs_to_ignore = local_kwargs_to_ignore

        self.pre_load_routines_run = False
        self._data_source: Optional[DynamicTableDataSource] = None # is set later
        self._local_persist_manager: Optional[PersistManager] = None

        self._scheduler_tree_connected = False

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called when TimeSerie is subclassed.
        It automatically wraps the subclass's __init__ method to add post-init routines.
        """
        super().__init_subclass__(**kwargs)

        # Get the original __init__ from the new subclass
        original_init = cls.__init__

        @wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            # --- This is the logic from the old decorator ---

            # 1. Call the original __init__ of the subclass first
            original_init(self, *args, **kwargs)

            # 2. Capture all arguments passed to create the final config
            # We inspect the original_init to find what arguments it was called with.
            sig = inspect.signature(original_init)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            final_kwargs = bound_args.arguments
            final_kwargs.pop('self', None)  # Remove self
            final_kwargs.pop('args', None)  # Remove self
            final_kwargs.pop('kwargs', None)  # Remove self

            # 3. Run the post-initialization routines
            logger.info(f"Running post-init routines for {self.__class__.__name__}")

            # Add import path to the config kwargs
            final_kwargs["time_series_class_import_path"] = {
                "module": self.__class__.__module__,
                "qualname": self.__class__.__qualname__
            }

            config=build_operations.create_config(kwargs=final_kwargs,
                                                  ts_class_name=self.__class__.__name__
                                                  ) # Assuming these methods exist
            for field_name, value in asdict(config).items():
                setattr(self, field_name, value)

            # 7. Final setup
            self.set_data_source()
            logger.bind(local_hash_id=self.hashed_name)


            self.run_after_post_init_routines()

            #requirements for graph update
            self.dependencies_df: Optional[pd.DataFrame] = None
            self.depth_df: Optional[pd.DataFrame] = None

            self.scheduler : Optional[Scheduler] = None
            self.update_details_tree :Optional[Dict[str,Any]] =None




            build_operations.patch_build_configuration(time_serie_instance=self)

            logger.info(f"Post-init routines for {self.__class__.__name__} complete.")

        # Replace the subclass's __init__ with our new wrapped version
        cls.__init__ = wrapped_init


    @property
    def is_api(self):
        return False
    @property
    def hash_id(self) -> str:
        """The remote table hash name."""
        return self.remote_table_hashed_name

    @property
    def data_source_id(self) -> int:
        return self.data_source.id

    @property
    def local_hash_id(self) -> str:
        return self.hashed_name

    @property
    def data_source(self) -> DataSource:
        return self.persistence.data_source

    @property
    def local_time_serie(self) -> LocalTimeSerie:
        """The local time series metadata object."""
        return self.local_persist_manager.local_metadata

    @property
    def metadata(self) -> "DynamicTableMetaData":
        return self.local_persist_manager.metadata
    @property
    def table(self)->"DynamicTableMetaData":
        return self.local_persist_manager.metadata

    @property
    def local_persist_manager(self) -> PersistManager:
        if self._local_persist_manager is None:
            self.logger.debug(f"Setting local persist manager for {self.hash_id}")
            self._set_local_persist_manager(local_hash_id=self.local_hash_id,
                                            )
        return self._local_persist_manager

    def persist_to_pickle(self, overwrite: bool = False) -> Tuple[str, str]:
        """
        Persists the TimeSerie object to a pickle file.

        Args:
            overwrite: If True, overwrites the existing pickle file.

        Returns:
            A tuple containing the full path and the relative path of the pickle file.
        """
        import cloudpickle
        path = build_operations.get_pickle_path_from_time_serie(time_serie_instance=self)
        # after persisting pickle , build_hash and source code need to be patched
        self.logger.debug \
            (f"Persisting pickle and patching source code and git hash for {self.hash_id}")
        build_operations.update_git_and_code_in_backend(time_serie_instance=self)

        pp = build_operations.data_source_pickle_path(self.data_source.id)
        if os.path.isfile(pp) == False or overwrite == True:
            self.data_source.persist_to_pickle(pp)

        if os.path.isfile(path) == False or overwrite == True:
            if overwrite == True:
                self.logger.warning("overwriting pickle")

            self._local_persist_manager=None #delete the local persist manager because it has a ThreadLock

            dir_, fname = os.path.split(path)
            fd, tmp_path = tempfile.mkstemp(prefix=fname, dir=dir_ or None)
            os.close(fd)
            try:
                with open(tmp_path, 'wb') as handle:
                    cloudpickle.dump(self, handle)
                # atomic replace (works on POSIX and Windows Python ≥ 3.3)
                os.replace(tmp_path, path)
            except Exception:
                # clean up temp file on error
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise

        return path, path.replace(ogm.pickle_storage_path + "/", "")

    def _set_local_persist_manager(self, local_hash_id: str,
                                  local_metadata: Union[None, dict] = None,

                                  ) -> None:
        """
        Initializes the local persistence manager for the time series. It sets up
        the necessary configurations and checks for existing metadata. If the metadata doesn't
        exist or is incomplete, it sets up the initial configuration and builds the update details.

        Args:
           hashed_name : str
               The local hash ID for the time series.
           remote_table_hashed_name : str
               The remote table hash name for the time series.
           local_metadata : Union[None, dict], optional
               Local metadata for the time series, if available.
        """
        self._local_persist_manager = PersistManager.get_from_data_type(
            local_hash_id=local_hash_id,
            class_name=self.__class__.__name__,
            local_metadata=local_metadata,
            data_source=self.data_source
        )

    #Necessary passhtrough methods

    @property
    def data_source(self) -> Any:
        if self._data_source is not None:
            return self._data_source
        else:
            raise Exception("Data source has not been set")

    def set_data_source(self,
                        data_source: Optional[object] = None) -> None:
        """
        Sets the data source for the time series.

        Args:
            data_source: The data source object. If None, the default is fetched from the ORM.
        """
        if data_source is None:
            self._data_source = get_data_source_from_orm()
        else:
            self._data_source = data_source

    def verify_and_build_remote_objects(self) -> None:
        """
        Verifies and builds remote objects by calling the persistence layer.
        This logic is now correctly located within the BuildManager.
        """
        # Use self.owner to get properties from the TimeSerie instance
        owner_class = self.__class__
        time_serie_source_code_git_hash = build_operations.get_time_serie_source_code_git_hash(owner_class)
        time_serie_source_code = build_operations.get_time_serie_source_code(owner_class)

        # The call to the low-level persist manager is encapsulated here
        self.local_persist_manager.local_persist_exist_set_config(
            remote_table_hashed_name=self.remote_table_hashed_name,
            local_configuration=self.local_initial_configuration,
            remote_configuration=self.remote_initial_configuration,
            remote_build_metadata=self.remote_build_metadata,
            time_serie_source_code_git_hash=time_serie_source_code_git_hash,
            time_serie_source_code=time_serie_source_code,
            data_source=self.data_source,
        )
    def set_relation_tree(self):

        """Sets the node relationships in the backend by calling the dependencies() method."""

        if self.local_persist_manager.local_metadata is None:
            self.verify_and_build_remote_objects()  #
        if self.local_persist_manager.is_local_relation_tree_set():
            return
        declared_dependencies = self.dependencies() or {}

        for name, dependency_ts in declared_dependencies.items():
            self.logger.info(f"Connecting dependency '{name}'...")

            # Ensure the dependency itself is properly initialized
            dependency_ts.verify_and_build_remote_objects()

            is_api = dependency_ts.is_api
            self.local_persist_manager.depends_on_connect(dependency_ts, is_api=is_api)

            # Recursively set the relation tree for the dependency
            dependency_ts.set_relation_tree()

        self.local_persist_manager.set_ogm_dependencies_linked()
    def get_update_statistics(self) -> UpdateStatistics:
        return self.local_persist_manager.get_update_statistics()

    def run(
            self,
            debug_mode: bool,
            *,
            update_tree: bool = True,
            force_update: bool = False,
            update_only_tree: bool = False,
            remote_scheduler: Optional[object] = None
    ):
        """
        Starts the execution of the time series by delegating to the run_operations.
        """
        return run_operations.run(
            running_time_serie=self,
            debug_mode=debug_mode,
            update_tree=update_tree,
            force_update=force_update,
            update_only_tree=update_only_tree,
            remote_scheduler=remote_scheduler
        )


    # --- Optional Hooks for Customization ---
    def run_after_post_init_routines(self) -> None:
        pass

    def get_minimum_required_depth_for_update(self) -> int:
        """
        Controls the minimum depth that needs to be rebuilt.
        """
        return 0

    def get_table_metadata(self,update_statistics:ms_client.UpdateStatistics)->Optional[ms_client.TableMetaData]:
        """Provides the metadata configuration for a market time series.

         """


        return None

    def get_column_metadata(self) -> Optional[List[ColumnMetaData]]:
        """
        This Method should return a list for ColumnMetaData to add extra context to each time series
        Examples:
            from mainsequence.client.models_tdag import ColumnMetaData
        columns_metadata = [ColumnMetaData(column_name="instrument",
                                          dtype="str",
                                          label="Instrument",
                                          description=(
                                              "Unique identifier provided by Valmer; it’s a composition of the "
                                              "columns `tv_emisora_serie`, and is also used as a ticker for custom "
                                              "assets in Valmer."
                                          )
                                          ),
                            ColumnMetaData(column_name="currency",
                                           dtype="str",
                                           label="Currency",
                                           description=(
                                               "Corresponds to  code for curries be aware this may not match Figi Currency assets"
                                           )
                                           ),

                            ]
        Returns:
            A list of ColumnMetaData objects, or None.
        """
        return None

    def get_asset_list(self) -> Optional[List["Asset"]]:
        """
        Provide the list of assets that this TimeSerie should include when updating.

        By default, this method returns `self.asset_list` if defined.
        Subclasses _must_ override this method when no `asset_list` attribute was set
        during initialization, to supply a dynamic list of assets for update_statistics.

        Use Case:
          - For category-based series, return all Asset unique_identifiers in a given category
            (e.g., `AssetCategory(unique_identifier="investable_assets")`), so that only those
            assets are updated in this TimeSerie.

        Returns
        -------
        list or None
            - A list of asset unique_identifiers to include in the update.
            - `None` if no filtering by asset is required (update all assets by default).
        """
        if hasattr(self, "asset_list"):
            return self.asset_list

        return None

    def _run_post_update_routines(self, error_on_last_update: bool, update_statistics: UpdateStatistics) -> None:
        """ Should be overwritten by subclass """
        pass

    @abstractmethod
    def dependencies(self) -> Dict[str, Union["TimeSerie", "APITimeSerie"]]:
        """
        Subclasses must implement this method to explicitly declare their upstream dependencies.

        Returns:
            A dictionary where keys are descriptive names and values are the TimeSerie dependency instances.
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, update_statistics: UpdateStatistics) -> pd.DataFrame:
        """
        Fetch and ingest only the new rows for this TimeSerie based on prior update checkpoints.

        UpdateStatistics provides the last-ingested positions:
          - For a single-index series (time_index only), `update_statistics.max_time` is either:
              - None: no prior data—fetch all available rows.
              - a datetime: fetch rows where `time_index > max_time`.
          - For a dual-index series (time_index, unique_identifier), `update_statistics.max_time_per_id` is either:
              - None: single-index behavior applies.
              - dict[str, datetime]: for each `unique_identifier` (matching `Asset.unique_identifier`), fetch rows where
                `time_index > max_time_per_id[unique_identifier]`.

        Requirements:
          - `time_index` **must** be a `datetime.datetime` instance with UTC timezone.
          - Column names **must** be all lowercase.
          - No column values may be Python `datetime` objects; if date/time storage is needed, convert to integer
            timestamps (e.g., UNIX epoch in seconds or milliseconds).

        After retrieving the incremental rows, this method inserts or upserts them into the Main Sequence Data Engine.

        Parameters
        ----------
        update_statistics : UpdateStatistics
            Object capturing the previous update state. Must expose:
              - `max_time` (datetime | None)
              - `max_time_per_id` (dict[str, datetime] | None)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing only the newly added or updated records.
        """
        raise NotImplementedError



class WrapperTimeSerie(TimeSerie):
    """A wrapper class for managing multiple TimeSerie objects."""

    def __init__(self, translation_table: AssetTranslationTable, *args, **kwargs):
        """
        Initialize the WrapperTimeSerie.

        Args:
            time_series_dict: Dictionary of TimeSerie objects.
        """
        super().__init__(*args, **kwargs)

        def get_time_serie_from_markets_unique_id(table_identifier: str) -> TimeSerie:
            """
            Returns the appropriate bar time series based on the asset list and source.
            """
            from mainsequence.client import DoesNotExist
            try:
                metadata = ms_client.DynamicTableMetaData.get(identifier=table_identifier)

            except DoesNotExist as e:
                logger.exception(f"HistoricalBarsSource does not exist for {market_time_serie_unique_identifier}")
                raise e
            api_ts = APITimeSerie(
                data_source_id=metadata.data_source,
                source_table_hash_id=source_table.hash_id
            )
            return api_ts

        translation_table = copy.deepcopy(translation_table)

        self.api_ts_map = {}
        for rule in translation_table.rules:
            if rule.markets_time_serie_unique_identifier not in self.api_ts_map:
                self.api_ts_map[rule.markets_time_serie_unique_identifier] = get_time_serie_from_markets_unique_id(
                    table_identifier=rule.markets_time_serie_unique_identifier)

        self.translation_table = translation_table

    def get_ranged_data_per_asset(self, range_descriptor: Optional[UniqueIdentifierRangeMap]) -> pd.DataFrame:
        """
        Gets data based on a range descriptor.

        Args:
            range_descriptor: A UniqueIdentifierRangeMap object.

        Returns:
            A DataFrame with the ranged data.
        """
        return self.get_df_between_dates(unique_identifier_range_map=range_descriptor)
    
    def get_df_between_dates(
            self,
            start_date: Optional[datetime.datetime] = None,
            end_date: Optional[datetime.datetime] = None,
            unique_identifier_list: Optional[list] = None,
            great_or_equal: bool = True,
            less_or_equal: bool = True,
            unique_identifier_range_map: Optional[UniqueIdentifierRangeMap] = None,
    ) -> pd.DataFrame:
        """
        Retrieves a DataFrame of time series data between specified dates, handling asset translation.

        Args:
            start_date: The start date of the data range.
            end_date: The end date of the data range.
            unique_identifier_list: An optional list of unique identifiers to filter by.
            great_or_equal: Whether to include the start date.
            less_or_equal: Whether to include the end date.
            unique_identifier_range_map: An optional map of ranges for unique identifiers.

        Returns:
            A pandas DataFrame with the requested data.
        """
        if (unique_identifier_list is None) == (unique_identifier_range_map is None):
            raise ValueError(
                "Pass **either** unique_identifier_list **or** unique_identifier_range_map, but not both."
            )

        if unique_identifier_list is not None:
            wanted_src_uids = set(unique_identifier_list)
        else:  # range‑map path
            wanted_src_uids = set(unique_identifier_range_map.keys())

        if not wanted_src_uids:
            return pd.DataFrame()

        # evaluate the rules for each asset
        from mainsequence.client import Asset
        assets = Asset.filter(unique_identifier__in=list(wanted_src_uids))
        asset_translation_dict = {}
        for asset in assets:
            asset_translation_dict[asset.unique_identifier] = self.translation_table.evaluate_asset(asset)

        # we grouped the assets for the same rules together and now query all assets that have the same target
        translation_df = pd.DataFrame.from_dict(asset_translation_dict, orient="index")
        try:
            grouped = translation_df.groupby(
                ["markets_time_serie_unique_identifier", "execution_venue_symbol", "exchange_code"],
                dropna=False
            )
        except Exception as e:
            raise e

        data_df = []
        for (mkt_ts_id, target_execution_venue_symbol, target_exchange_code), group_df in grouped:
            # get the correct TimeSerie instance from our pre-built map
            api_ts = self.api_ts_map[mkt_ts_id]

            # figure out which assets belong to this group
            grouped_unique_ids = group_df.index.tolist()
            source_assets = [
                a for a in assets
                if a.unique_identifier in grouped_unique_ids
            ]

            # get correct target assets based on the share classes
            main_sequence_share_classes = [a.main_sequence_share_class for a in assets]
            asset_query = dict(
                execution_venue__symbol=target_execution_venue_symbol,
                main_sequence_share_class__in=main_sequence_share_classes
            )
            if not pd.isna(target_exchange_code):
                asset_query["exchange_code"] = target_exchange_code

            target_assets = Asset.filter(**asset_query)

            target_asset_unique_ids = [a.main_sequence_share_class for a in target_assets]
            if len(main_sequence_share_classes) > len(target_asset_unique_ids):
                self.logger.warning(f"Not all assets were found in backend for translation table: {set(main_sequence_share_classes) - set(target_asset_unique_ids)}")

            if len(main_sequence_share_classes) < len(target_asset_unique_ids):
                self.logger.warning(f"Too many assets were found in backend for translation table: {set(target_asset_unique_ids) - set(main_sequence_share_classes)}")

            # create the source-target mapping
            source_asset_share_class_map = {}
            for a in source_assets:
                if a.main_sequence_share_class in source_asset_share_class_map:
                    raise ValueError(f"Share class {a.main_sequence_share_class} cannot be duplicated")
                source_asset_share_class_map[a.main_sequence_share_class] = a.unique_identifier

            source_target_map = {}
            for a in target_assets:
                main_sequence_share_class = a.main_sequence_share_class
                source_unique_identifier = source_asset_share_class_map[main_sequence_share_class]
                source_target_map[source_unique_identifier] = a.unique_identifier

            target_source_map = {v: k for k, v in source_target_map.items()}
            if unique_identifier_range_map is not None:
                # create the correct unique identifier range map
                unique_identifier_range_map_target = {}
                for a_unique_identifier, asset_range in unique_identifier_range_map.items():
                    if a_unique_identifier not in source_target_map.keys(): continue
                    target_key = source_target_map[a_unique_identifier]
                    unique_identifier_range_map_target[target_key] = asset_range

                if not unique_identifier_range_map_target:
                    self.logger.warning(
                        f"Unique identifier map is empty for group assets {source_assets} and unique_identifier_range_map {unique_identifier_range_map}")
                    continue

                tmp_data = api_ts.get_df_between_dates(
                    unique_identifier_range_map=unique_identifier_range_map_target,
                    start_date=start_date,
                    end_date=end_date,
                    great_or_equal=great_or_equal,
                    less_or_equal=less_or_equal,
                )
            else:
                tmp_data = api_ts.get_df_between_dates(
                    start_date=start_date,
                    end_date=end_date,
                    unique_identifier_list=list(source_target_map.values()),
                    great_or_equal=great_or_equal,
                    less_or_equal=less_or_equal,
                )

            if tmp_data.empty:
                continue

            tmp_data = tmp_data.rename(index=target_source_map, level="unique_identifier")
            data_df.append(tmp_data)

        if not data_df:
            return pd.DataFrame()

        data_df = pd.concat(data_df, axis=0)
        return data_df

    def update(self, update_statistics):
        """ WrapperTimeSeries does not update """
        pass


build_operations.serialize_argument.register(TimeSerie, build_operations._serialize_timeserie)
build_operations.serialize_argument.register(APITimeSerie, build_operations._serialize_api_timeserie)
