import pandas as pd
import datetime
from typing import Union, List, Dict, Optional, Tuple
import os
from mainsequence.logconf import logger


from mainsequence.client import (LocalTimeSerie, UniqueIdentifierRangeMap,
                                 LocalTimeSeriesDoesNotExist,
                                 DynamicTableDoesNotExist, DynamicTableDataSource, TDAG_CONSTANTS as CONSTANTS, DynamicTableMetaData,
                                 DataUpdates, DoesNotExist)

from mainsequence.client.models_tdag import DynamicTableHelpers, LocalTimeSerieUpdateDetails
import json
import threading
from concurrent.futures import Future
from .. import  future_registry


class APIPersistManager:
    """
    Manages persistence for time series data accessed via an API.
    It handles asynchronous fetching of metadata to avoid blocking operations.
    """

    def __init__(self, data_source_id: int, local_hash_id: str):
        """
        Initializes the APIPersistManager.

        Args:
            data_source_id: The ID of the data source.
            local_hash_id: The local hash identifier for the time series.
        """
        self.data_source_id: int = data_source_id
        self.local_hash_id: str = local_hash_id

        logger.debug(f"Initializing Time Serie {self.local_hash_id}  as APITimeSerie")

        # Create a Future to hold the local metadata when ready.
        self._local_metadata_future = Future()
        # Register the future globally.
        future_registry.add_future(self._local_metadata_future)
        # Launch the REST request in a separate, non-daemon thread.
        thread = threading.Thread(target=self._init_local_metadata,
                                  name=f"LocalMetadataThread-{self.local_hash_id}",
                                  daemon=False)
        thread.start()


    @property
    def local_metadata(self) -> LocalTimeSerie:
        """Lazily block and cache the result if needed."""
        if not hasattr(self, '_local_metadata_cached'):
            # This call blocks until the future is resolved.
            self._local_metadata_cached = self._local_metadata_future.result()
        return self._local_metadata_cached

    @property
    def metadata(self) -> DynamicTableMetaData:
        """Returns the remote table metadata associated with the local time series."""
        return self.local_metadata.remote_table


    def _init_local_metadata(self) -> None:
        """
        Performs the REST request to fetch local metadata asynchronously.
        Sets the result or exception on the future object.
        """
        try:
            result = LocalTimeSerie.get_or_none(local_hash_id=self.local_hash_id,
                                                remote_table__data_source__id=self.data_source_id,
                                                include_relations_detail=True
            )
            self._local_metadata_future.set_result(result)
        except Exception as exc:
            self._local_metadata_future.set_exception(exc)
        finally:
            # Remove the future from the global registry once done.
            future_registry.remove_future(self._local_metadata_future)

    def get_df_between_dates(self,
                             start_date: Optional[datetime.datetime],
                             end_date: Optional[datetime.datetime],
                             great_or_equal: bool = True,
                             less_or_equal: bool = True,
                             unique_identifier_list: Optional[List[str]] = None,
                             columns: Optional[List[str]] = None,
                             unique_identifier_range_map: Optional[UniqueIdentifierRangeMap] = None
                             ) -> pd.DataFrame:
        """
        Retrieves a DataFrame from the API between specified dates.

        Args:
            start_date: The start date for the data range.
            end_date: The end date for the data range.
            great_or_equal: Whether the start date is inclusive.
            less_or_equal: Whether the end date is inclusive.
            unique_identifier_list: A list of unique identifiers to filter by.
            columns: A list of columns to retrieve.
            unique_identifier_range_map: A map of unique identifiers to their specific date ranges.

        Returns:
            A pandas DataFrame with the requested data.
        """
        filtered_data = self.local_metadata.get_data_between_dates_from_api(
                                                        start_date=start_date,
                                                        end_date=end_date, great_or_equal=great_or_equal,
                                                        less_or_equal=less_or_equal,
                                                        unique_identifier_list=unique_identifier_list,
                                                        columns=columns,
                                                        unique_identifier_range_map=unique_identifier_range_map
        )

        if len(filtered_data) == 0:
            logger.info(f"Data from {self.local_hash_id} is empty in request ")
            logger.debug(
                f"Calling get_data_between_dates_from_api with arguments: "
                f"start_date={start_date}, end_date={end_date}, "
                f"great_or_equal={great_or_equal}, less_or_equal={less_or_equal}, "
                f"unique_identifier_list={unique_identifier_list}, columns={columns}, "
                f"unique_identifier_range_map={unique_identifier_range_map}"
            )
            return filtered_data

        #fix types

        stc = self.local_metadata.remote_table.sourcetableconfiguration
        filtered_data[stc.time_index_name] = pd.to_datetime(filtered_data[stc.time_index_name],
                                                            utc=True
                                                            )
        for c, c_type in stc.column_dtypes_map.items():
            if c!=stc.time_index_name:
                if c_type=="object":
                    c_type="str"
                filtered_data[c]=filtered_data[c].astype(c_type)
        filtered_data=filtered_data.set_index(stc.index_names)
        return filtered_data

    def filter_by_assets_ranges(self, unique_identifier_range_map: UniqueIdentifierRangeMap,
                                time_serie: "TimeSerie") -> pd.DataFrame:
        """
        Filters data by asset ranges.

        Args:
            unique_identifier_range_map: A map of assets to their date ranges.
            time_serie: The parent TimeSerie object.

        Returns:
            A pandas DataFrame containing the filtered data.
        """
        df = self.get_df_between_dates(start_date=None, end_date=None, unique_identifier_range_map=unique_identifier_range_map)
        return df

class PersistManager:
    def __init__(self,
                 data_source: DynamicTableDataSource,
                 local_hash_id: str,
                 description: Optional[str] = None,
                 class_name: Optional[str] = None,
                 metadata: Optional[Dict] = None,
                 local_metadata: Optional[LocalTimeSerie] = None
                 ):
        """
        Initializes the PersistManager.

        Args:
            data_source: The data source for the time series.
            local_hash_id: The local hash identifier for the time series.
            description: An optional description for the time series.
            class_name: The name of the TimeSerie class.
            metadata: Optional remote metadata dictionary.
            local_metadata: Optional local metadata object.
        """
        self.data_source: DynamicTableDataSource = data_source
        self.local_hash_id: str = local_hash_id
        if local_metadata is not None and metadata is None:
            # query remote hash_id
            metadata = local_metadata.remote_table
        self.description: Optional[str] = description
        self.logger = logger

        self.table_model_loaded: bool = False
        self.class_name: Optional[str] = class_name

        # Private members for managing lazy asynchronous retrieval.
        self._local_metadata_future: Optional[Future] = None
        self._local_metadata_cached: Optional[LocalTimeSerie] = None
        self._local_metadata_lock = threading.Lock()
        self._metadata_cached: Optional[DynamicTableMetaData] = None

        if self.local_hash_id is not None:
            self.synchronize_metadata(local_metadata=local_metadata)

    def synchronize_metadata(self, local_metadata: Optional[LocalTimeSerie]) -> None:
        if local_metadata is not None:
            self.set_local_metadata(local_metadata)
        else:
            self.set_local_metadata_lazy(force_registry=True, include_relations_detail=True)

    @classmethod
    def get_from_data_type(cls, data_source: DynamicTableDataSource, *args, **kwargs) -> 'PersistManager':
        """
        Factory method to get the correct PersistManager based on data source type.

        Args:
            data_source: The data source object.

        Returns:
            An instance of a PersistManager subclass.
        """
        data_type = data_source.related_resource_class_type
        if data_type in CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB:
            return TimeScaleLocalPersistManager(data_source=data_source, *args, **kwargs)
        else:
            return TimeScaleLocalPersistManager(data_source=data_source, *args, **kwargs)

    def set_local_metadata(self, local_metadata: LocalTimeSerie) -> None:
        """
        Caches the local metadata object for lazy queries

        Args:
            local_metadata: The LocalTimeSerie object to cache.
        """
        self._local_metadata_cached = local_metadata

    @property
    def local_metadata(self) -> LocalTimeSerie:
        """Lazily block and retrieve the local metadata, caching the result."""
        with self._local_metadata_lock:
            if self._local_metadata_cached is None:
                if self._local_metadata_future is None:
                    # If no future is running, start one.
                    self.set_local_metadata_lazy(force_registry=True)
                # Block until the future completes and cache its result.
                local_metadata = self._local_metadata_future.result()
                self.set_local_metadata(local_metadata)
            return self._local_metadata_cached

            # Define a callback that will launch set_local_metadata_lazy after the remote update is complete.
    @property
    def metadata(self) -> Optional[DynamicTableMetaData]:
        """
        Lazily retrieves and returns the remote metadata.
        """
        if self.local_metadata is None:
            return None
        if self.local_metadata.remote_table is not None:
            if self.local_metadata.remote_table.sourcetableconfiguration is not None:
                if self.local_metadata.remote_table.build_meta_data.get("initialize_with_default_partitions",True) == False:
                    if self.local_metadata.remote_table.data_source.related_resource_class_type in CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB:
                        self.logger.warning("Default Partitions will not be initialized ")

        return self.local_metadata.remote_table

    @property
    def local_build_configuration(self) -> Dict:
        return self.local_metadata.build_configuration

    @property
    def local_build_metadata(self) -> Dict:
        return self.local_metadata.build_meta_data

    def set_local_metadata_lazy_callback(self, fut: Future) -> None:
        """
        Callback to handle the result of an asynchronous task and trigger a metadata refresh.
        """
        try:
            # This will re-raise any exception that occurred in _update_task.
            fut.result()
        except Exception as exc:
            # Optionally, handle or log the error if needed.
            # For example: logger.error("Remote build update failed: %s", exc)
            raise exc
        # Launch the local metadata update regardless of the outcome.
        self.set_local_metadata_lazy(force_registry=True)

    def set_local_metadata_lazy(self, force_registry: bool = True, include_relations_detail: bool = True) -> None:
        """
        Initiates a lazy, asynchronous fetch of the local metadata.

        Args:
            force_registry: If True, forces a refresh even if cached data exists.
            include_relations_detail: If True, includes relationship details in the fetch.
        """
        with self._local_metadata_lock:
            if force_registry:
                self._local_metadata_cached = None
            # Capture the new future in a local variable.
            new_future = Future()
            self._local_metadata_future = new_future
            # Register the new future.
            future_registry.add_future(new_future)

        def _get_or_none_local_metadata():
            """Perform the REST request asynchronously."""
            try:
                result = LocalTimeSerie.get_or_none(
                    local_hash_id=self.local_hash_id,
                    remote_table__data_source__id=self.data_source.id,
                    include_relations_detail=include_relations_detail
                )
                if result is None:
                    self.logger.warning(f"TimeSeries {self.local_hash_id} with data source {self.data_source.id} not found in backend")
                new_future.set_result(result)
            except Exception as exc:
                new_future.set_exception(exc)
            finally:
                # Remove the future from the global registry once done.
                future_registry.remove_future(new_future)

        thread = threading.Thread(target=_get_or_none_local_metadata,
                                  name=f"LocalMetadataThreadPM-{self.local_hash_id}",
                                  daemon=False)
        thread.start()



    def depends_on_connect(self, new_ts: "TimeSerie", is_api: bool) -> None:
        """
        Connects a time series as a relationship in the DB.

        Args:
            new_ts: The target TimeSerie to connect to.
            is_api: True if the target is an APITimeSerie.
        """
        if not is_api:
            self.local_metadata.depends_on_connect(
                                        source_local_hash_id=self.local_metadata.local_hash_id,
                                        target_local_hash_id=new_ts.local_hash_id,
                                        target_class_name=new_ts.__class__.__name__,
                                        source_data_source_id=self.data_source.id,
                                        target_data_source_id=new_ts.data_source.id
                                        )
        else:
            try:
                self.local_metadata.depends_on_connect_remote_table(
                    source_hash_id=self.metadata.hash_id,
                    source_local_hash_id=self.local_metadata.local_hash_id,
                    source_data_source_id=self.data_source.id,
                    target_data_source_id=new_ts.data_source_id,
                    target_local_hash_id=new_ts.local_hash_id
                )
            except Exception as exc:
                raise exc


    def display_mermaid_dependency_diagram(self) -> str:
        """
        Generates and returns an HTML string for a Mermaid dependency diagram.

        Returns:
            An HTML string containing the Mermaid diagram and supporting Javascript.
        """
        from IPython.core.display import display, HTML, Javascript

        response = TimeSerieLocalUpdate.get_mermaid_dependency_diagram(local_hash_id=self.local_hash_id,
                                                                       data_source_id=self.data_source.id
                                                                       )
        from IPython.core.display import display, HTML, Javascript
        mermaid_chart = response.get("mermaid_chart")
        metadata = response.get("metadata")
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

    def get_all_dependencies_update_priority(self) -> pd.DataFrame:
        """
        Retrieves a DataFrame of all dependencies with their update priority.

        Returns:
            A pandas DataFrame with dependency and priority information.
        """
        depth_df = self.local_metadata.get_all_dependencies_update_priority()
        return depth_df

    def set_ogm_dependencies_linked(self) -> None:
        self.local_metadata.set_ogm_dependencies_linked()

    @property
    def update_details(self) -> Optional[LocalTimeSerieUpdateDetails]:
        """Returns the update details associated with the local time series."""
        return self.local_metadata.localtimeserieupdatedetails

    @property
    def run_configuration(self) -> Optional[Dict]:
        """Returns the run configuration from the local metadata."""
        return self.local_metadata.run_configuration

    @property
    def source_table_configuration(self) -> Optional[Dict]:
        """Returns the source table configuration from the remote metadata."""
        if "sourcetableconfiguration" in self.metadata.keys():
            return self.metadata['sourcetableconfiguration']
        return None

    def update_source_informmation(self, git_hash_id: str, source_code: str) -> None:
        """
        Updates the source code and git hash for the remote table.
        """
        self.local_metadata.remote_table = self.metadata.patch(
            time_serie_source_code_git_hash=git_hash_id,
            time_serie_source_code=source_code,
        )

    @staticmethod
    def batch_data_persisted(hash_id_list: List[str]) -> Dict[str, bool]:
        """
        Checks for the existence of multiple time series in the database.

        Args:
            hash_id_list: A list of remote table hash IDs.

        Returns:
            A dictionary mapping each hash ID to a boolean indicating its existence.
        """
        exist = {}
        dth = DynamicTableHelpers()
        in_db, _ = dth.exist(hash_id__in=hash_id_list)

        for t in hash_id_list:

            if t in in_db:
                exist[t] = True
            else:
                exist[t] = False

        return exist

    def add_tags(self, tags: List[str]) -> None:
        """Adds tags to the local time series metadata if they don't already exist."""
        if any([t not in self.local_metadata.tags for t in tags]) == True:
            self.local_metadata.add_tags(tags=tags)

    def destroy(self, delete_only_table: bool):
        self.dth.destroy(metadata=self.metadata, delete_only_table=delete_only_table)

    @property
    def persist_size(self) -> int:
        """Returns the size of the persisted table, or 0 if not available."""
        try:
            return self.metadata['table_size']
        except KeyError:
            return 0

    def time_serie_exist(self) -> bool:
        """Checks if the remote metadata for the time series exists."""
        if hasattr(self, "metadata"):
            return True
        return False

    def patch_build_configuration(self, local_configuration: dict, remote_configuration: dict,
                                  remote_build_metadata: dict) -> None:
        """
        Asynchronously patches the build configuration for the remote and local tables.

        Args:
            local_configuration: The build configuration for the local time series.
            remote_configuration: The build configuration for the remote table.
            remote_build_metadata: The build metadata for the remote table.
        """
        # This ensures that later accesses to local_metadata will block for the new value.
        with self._local_metadata_lock:
            self._local_metadata_future = Future()
            future_registry.add_future(self._local_metadata_future)

        kwargs = dict(
                      build_configuration=remote_configuration, )


        local_metadata_kwargs = dict(local_hash_id=self.local_hash_id,
                               build_configuration=local_configuration,
                              )

        patch_future = Future()
        future_registry.add_future(patch_future)

        # Define the inner helper function.
        def _patch_build_configuration():
            """Helper function for patching build configuration asynchronously."""
            try:
                # Execute the patch operation; this method is expected to return a LocalTimeSerie-like instance.
                result = DynamicTableMetaData.patch_build_configuration(
                    remote_table_patch=kwargs,
                    data_source_id=self.data_source.id,
                    build_meta_data=remote_build_metadata,
                    local_table_patch=local_metadata_kwargs,
                )
                patch_future.set_result(True) #success
            except Exception as exc:
                patch_future.set_exception(exc)
            finally:
                # Once the operation is complete (or errors out), remove the future from the global registry.
                future_registry.remove_future(result)

        thread = threading.Thread(
            target=_patch_build_configuration,
            name=f"PatchBuildConfigThread-{self.local_hash_id}",
            daemon=False
        )
        thread.start()

        patch_future.add_done_callback(self.set_local_metadata_lazy_callback)


    def local_persist_exist_set_config(
            self,
            remote_table_hashed_name: str,
            local_configuration: dict,
            remote_configuration: dict,
            data_source: DynamicTableDataSource,
            time_serie_source_code_git_hash: str,
            time_serie_source_code: str,
            remote_build_metadata: dict,
    ) -> None:
        """
        Ensures local and remote persistence objects exist and sets their configurations.
        This runs on TimeSerie initialization.
        """
        remote_build_configuration = None
        if hasattr(self, "remote_build_configuration"):
            remote_build_configuration = self.remote_build_configuration

        if remote_build_configuration is None:
            logger.debug(f"remote table {remote_table_hashed_name} does not exist creating")
            #create remote table

            try:

                # table may not exist but
                remote_build_metadata = remote_configuration["build_meta_data"] if "build_meta_data" in remote_configuration.keys() else {}
                remote_configuration.pop("build_meta_data", None)
                kwargs = dict(hash_id=remote_table_hashed_name,
                              time_serie_source_code_git_hash=time_serie_source_code_git_hash,
                              time_serie_source_code=time_serie_source_code,
                              build_configuration=remote_configuration,
                              data_source=data_source.model_dump(),
                              build_meta_data=remote_build_metadata)


                dtd_metadata = DynamicTableMetaData.get_or_create(**kwargs)
                remote_table_hash_id = dtd_metadata.hash_id
            except Exception as e:
                self.logger.exception(f"{remote_table_hashed_name} Could not set meta data in DB for P")
                raise e
        else:
            self.set_local_metadata_lazy(force_registry=True, include_relations_detail=True)
            remote_table_hash_id = self.metadata.hash_id

        local_table_exist = self._verify_local_ts_exists(remote_table_hash_id=remote_table_hash_id, local_configuration=local_configuration)


    def _verify_local_ts_exists(self, remote_table_hash_id: str,
                                local_configuration: Optional[Dict] = None) -> None:
        """
        Verifies that the local time series exists in the ORM, creating it if necessary.
        """
        local_build_configuration = None
        if self.local_metadata is not None:
            local_build_configuration, local_build_metadata = self.local_build_configuration, self.local_build_metadata
        if local_build_configuration is None:

            logger.debug(f"local_metadata {self.local_hash_id} does not exist creating")
            local_update = LocalTimeSerie.get_or_none(local_hash_id=self.local_hash_id,
                                                       remote_table__data_source__id=self.data_source.id)
            if local_update is None:
                local_build_metadata = local_configuration[
                    "build_meta_data"] if "build_meta_data" in local_configuration.keys() else {}
                local_configuration.pop("build_meta_data", None)
                metadata_kwargs = dict(
                    local_hash_id=self.local_hash_id,
                    build_configuration=local_configuration,
                    remote_table__hash_id=remote_table_hash_id,
                    description=self.description,
                    data_source_id=self.data_source.id
                )

                local_metadata = LocalTimeSerie.get_or_create(**metadata_kwargs,)


            else:
                local_metadata = local_update

            self.set_local_metadata(local_metadata=local_metadata)


    def _verify_insertion_format(self, temp_df: pd.DataFrame) -> None:
        """
        Verifies that a DataFrame is properly configured for insertion.
        """
        if isinstance(temp_df.index,pd.MultiIndex)==True:
            assert temp_df.index.names==["time_index", "asset_symbol"] or  temp_df.index.names==["time_index", "asset_symbol", "execution_venue_symbol"]

    def build_update_details(self, source_class_name: str) -> None:
        """
        Asynchronously builds or updates the update details for the time series.
        """
        update_kwargs=dict(source_class_name=source_class_name,
                           local_metadata=json.loads(self.local_metadata.model_dump_json())
                           )
        # This ensures that later accesses to local_metadata will block for the new value.
        with self._local_metadata_lock:
            self._local_metadata_future = Future()
            future_registry.add_future(self._local_metadata_future)

        # Create a future for the remote update task and register it.
        future = Future()
        future_registry.add_future(future)

        def _update_task():
            try:
                # Run the remote build/update details task.
                self.local_metadata.remote_table.build_or_update_update_details(**update_kwargs)
                future.set_result(True)  # Signal success
            except Exception as exc:
                future.set_exception(exc)
            finally:
                # Unregister the future once the task completes.
                future_registry.remove_future(future)

        thread = threading.Thread(
            target=_update_task,
            name=f"BuildUpdateDetailsThread-{self.local_hash_id}",
            daemon=False
        )
        thread.start()

        # Attach the callback to the future.
        future.add_done_callback(self.set_local_metadata_lazy_callback)

    def patch_table(self, **kwargs) -> None:
        """Patches the remote metadata table with the given keyword arguments."""
        self.metadata.patch( **kwargs)

    def protect_from_deletion(self, protect_from_deletion: bool = True) -> None:
        """Sets the 'protect_from_deletion' flag on the remote metadata."""
        self.metadata.patch( protect_from_deletion=protect_from_deletion)

    def open_for_everyone(self, open_for_everyone: bool = True) -> None:
        """Sets the 'open_for_everyone' flag on local, remote, and source table configurations."""
        if not self.local_metadata.open_for_everyone:
            self.local_metadata.patch(open_for_everyone=open_for_everyone)

        if not self.metadata.open_for_everyone:
            self.metadata.patch(open_for_everyone=open_for_everyone)

        if not self.metadata.sourcetableconfiguration.open_for_everyone:
            self.metadata.sourcetableconfiguration.patch(open_for_everyone=open_for_everyone)

    def set_start_of_execution(self,**kwargs):
        return self.dth.set_start_of_execution(metadata=self.metadata,**kwargs)

    def set_end_of_execution(self,**kwargs):
        return self.dth.set_end_of_execution(metadata=self.metadata, **kwargs)

    def reset_dependencies_states(self,hash_id_list):
        return self.dth.reset_dependencies_states(metadata=self.metadata, hash_id_list=hash_id_list)


    def get_update_statistics(self, asset_symbols: List[str],
                              remote_table_hash_id: str, time_serie: "TimeSerie"
                              ) -> Tuple[Optional[datetime.datetime], Optional[Dict[str, datetime.datetime]]]:
        """
        Gets update statistics for the time series.

        Returns:
            A tuple containing the last update timestamp for the table and a dictionary of
            last update timestamps per asset.
        """
        metadata = self.local_metadata.remote_table

        last_update_in_table, last_update_per_asset = None, None

        if metadata.sourcetableconfiguration is not None:
            last_update_in_table = metadata.sourcetableconfiguration.last_time_index_value
            if last_update_in_table is None:
                return last_update_in_table, last_update_per_asset
            if metadata.sourcetableconfiguration.multi_index_stats is not None:
                last_update_per_asset = metadata.sourcetableconfiguration.multi_index_stats['max_per_asset_symbol']
                if last_update_per_asset is not None:
                    last_update_per_asset = {unique_identifier: DynamicTableHelpers.request_to_datetime(v) for unique_identifier, v in last_update_per_asset.items()}

        if asset_symbols is not None and last_update_per_asset is not None:
            last_update_per_asset = {asset: value for asset, value in last_update_per_asset.items() if asset in asset_symbols}

        return last_update_in_table, last_update_per_asset

    def persist_updated_data(self, temp_df: pd.DataFrame, historical_update_id: Optional[int],
                             update_tracker: Optional[object] = None,
                             overwrite: bool = False) -> None:
        """
        Main update time series function, called from the TimeSerie class.

        Args:
            temp_df: DataFrame with the data to persist.
            historical_update_id: ID of the historical update record.
            update_tracker: The update tracker object.
            overwrite: Whether to overwrite existing data.
        """
        self._local_metadata_cached = DynamicTableHelpers.upsert_data_into_table(
            local_metadata=self.local_metadata,
            data=temp_df,
            data_source=self.data_source,

        )

    def get_persisted_ts(self):
        """
        full Request of the persisted data should always default to DB
        :return:
        """

        persisted_df = self.dth.get_data_by_time_index(metadata=self.metadata)
        return persisted_df

    def filter_by_assets_ranges(self, asset_ranges_map: dict, time_serie: "TimeSerie") -> pd.DataFrame:
        """
        Filters data by asset ranges using the DynamicTableHelpers.

        Args:
            asset_ranges_map: A dictionary mapping assets to their date ranges.
            time_serie: The parent TimeSerie object.

        Returns:
            A pandas DataFrame with the filtered data.
        """
        if isinstance(self, DataLakePersistManager):
            self.verify_if_already_run(time_serie)
        df = self.dth.filter_by_assets_ranges(metadata=self.metadata, asset_ranges_map=asset_ranges_map,
                                              data_source=self.data_source, local_hash_id=time_serie.local_hash_id)

        return df

    def get_earliest_value(self, remote_table_hash_id: str) -> Optional[datetime.datetime]:
        """Gets the earliest timestamp value from the table."""
        earliest_value = self.dth.get_earliest_value(hash_id=remote_table_hash_id)
        return earliest_value

    def get_df_between_dates(self,
                             start_date: Optional[datetime.datetime],
                             end_date: Optional[datetime.datetime],
                             great_or_equal: bool = True,
                             less_or_equal: bool = True,
                             unique_identifier_list: Optional[List[str]] = None,
                             columns: Optional[List[str]] = None,
                             unique_identifier_range_map: Optional[UniqueIdentifierRangeMap] = None
                             ) -> pd.DataFrame:
        """
        Retrieves a DataFrame from the data source between specified dates.
        """
        filtered_data = self.data_source.get_data_by_time_index(local_metadata=self.local_metadata,
                                                        start_date=start_date,
                                                        end_date=end_date,
                                                        great_or_equal=great_or_equal,
                                                        less_or_equal=less_or_equal,
                                                        unique_identifier_list=unique_identifier_list,
                                                        columns=columns,
                                                        unique_identifier_range_map=unique_identifier_range_map
                                                        )

        return filtered_data


class TimeScaleLocalPersistManager(PersistManager):
    """
    Main Controler to interacti with TimeSerie ORM
    """

    def get_full_source_data(self,remote_table_hash_id, engine="pandas"):
        """
        Returns full stored data, uses multiprocessing to achieve several queries by rows and speed
        :return:
        """

        from joblib import Parallel, delayed
        from tqdm import tqdm

        metadata = self.dth.get_configuration(hash_id=remote_table_hash_id)
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
            delayed(get_data)(ranges, i, remote_table_hash_id, engine) for i in tqdm(range(len(ranges) - 1)))
        dfs = pd.concat(dfs, axis=0)
        dfs = dfs.set_index(self.metadata["table_config"]["index_names"])
        return dfs


    def set_policy(self, interval: str, overwrite: bool, comp_type: str) -> None:
        """
        Sets a retention or compression policy on the time series table.

        Args:
            interval: The interval for the policy (e.g., '7 days').
            overwrite: If True, overwrites an existing policy.
            comp_type: The type of policy ('retention' or 'compression').
        """
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

    def set_policy_for_descendants(self,remote_table_hash_id:str,
                                   policy,comp_type:str,exclude_ids:Union[list,None]=None,extend_to_classes=False):
        self.dth.set_policy_for_descendants(hash_id=remote_table_hash_id,pol_type=comp_type,policy=policy,
                                            exclude_ids=exclude_ids,extend_to_classes=extend_to_classes)

    def delete_after_date(self, after_date: str):
        self.dth.delete_after_date(metadata=self.metadata, after_date=after_date)

    def get_table_schema(self,table_name):
        return self.metadata["sourcetableconfiguration"]["column_dtypes_map"]



