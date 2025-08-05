import pandas as pd
import datetime
from typing import Union, List, Dict, Optional, Tuple
import os
from mainsequence.logconf import logger


from mainsequence.client import (LocalTimeSerie, UniqueIdentifierRangeMap,
                                 LocalTimeSeriesDoesNotExist,
                                 DynamicTableDoesNotExist, DynamicTableDataSource, TDAG_CONSTANTS as CONSTANTS, DynamicTableMetaData,
                                 UpdateStatistics, DoesNotExist)

from mainsequence.client.models_tdag import  LocalTimeSerieUpdateDetails
import mainsequence.client as ms_client
import json
import threading
from concurrent.futures import Future
from .. import  future_registry
from mainsequence.instrumentation import tracer, tracer_instrumentator
import inspect
import hashlib

def get_data_node_source_code(DataNodeClass: "DataNode") -> str:
    """
    Gets the source code of a DataNode class.

    Args:
        DataNodeClass: The class to get the source code for.

    Returns:
        The source code as a string.
    """
    global logger
    try:
        # First try the standard approach.
        source = inspect.getsource(DataNodeClass)
        if source.strip():
            return source
    except Exception:
        logger.warning \
            ("Your TimeSeries is not in a python module this will likely bring exceptions when running in a pipeline")
    from IPython import get_ipython
    # Fallback: Scan IPython's input history.
    ip = get_ipython()  # Get the current IPython instance.
    if ip is not None:
        # Retrieve the full history as a single string.
        history = "\n".join(code for _, _, code in ip.history_manager.get_range())
        marker = f"class {DataNodeClass.__name__}"
        idx = history.find(marker)
        if idx != -1:
            return history[idx:]
    return "Source code unavailable."

def get_data_node_source_code_git_hash(DataNodeClass: "DataNode") -> str:
    """
    Hashes the source code of a DataNode class using SHA-1 (Git style).

    Args:
        DataNodeClass: The class to hash.

    Returns:
        The Git-style hash of the source code.
    """
    data_node_class_source_code = get_data_node_source_code(DataNodeClass)
    # Prepare the content for Git-style hashing
    # Git hashing format: "blob <size_of_content>\0<content>"
    content = f"blob {len(data_node_class_source_code)}\0{data_node_class_source_code}"
    # Compute the SHA-1 hash (Git hash)
    hash_object = hashlib.sha1(content.encode('utf-8'))
    git_hash = hash_object.hexdigest()
    return git_hash


class APIPersistManager:
    """
    Manages persistence for time series data accessed via an API.
    It handles asynchronous fetching of metadata to avoid blocking operations.
    """

    def __init__(self, data_source_id: int, storage_hash: str):
        """
        Initializes the APIPersistManager.

        Args:
            data_source_id: The ID of the data source.
            update_hash: The local hash identifier for the time series.
        """
        self.data_source_id: int = data_source_id
        self.storage_hash: str = storage_hash

        logger.debug(f"Initializing Time Serie {self.storage_hash}  as APIDataNode")

        # Create a Future to hold the local metadata when ready.
        self._metadata_future = Future()
        # Register the future globally.
        future_registry.add_future(self._metadata_future)
        # Launch the REST request in a separate, non-daemon thread.
        thread = threading.Thread(target=self._init_metadata,
                                  name=f"ApiMetaDataThread-{self.storage_hash}",
                                  daemon=False)
        thread.start()


    @property
    def metadata(self) -> DynamicTableMetaData:
        """Lazily block and cache the result if needed."""
        if not hasattr(self, '_metadata_cached'):
            # This call blocks until the future is resolved.
            self._metadata_cached = self._metadata_future.result()
        return self._metadata_cached

    def _init_metadata(self) -> None:
        """
        Performs the REST request to fetch local metadata asynchronously.
        Sets the result or exception on the future object.
        """
        try:
            result = DynamicTableMetaData.get_or_none(storage_hash=self.storage_hash,
                                                data_source__id=self.data_source_id,
                                                include_relations_detail=True
            )
            self._metadata_future.set_result(result)
        except Exception as exc:
            self._metadata_future.set_exception(exc)
        finally:
            # Remove the future from the global registry once done.
            future_registry.remove_future(self._metadata_future)

    def get_df_between_dates(self, *args, **kwargs) -> pd.DataFrame:
        """
        Retrieves a DataFrame from the API between specified dates.

        Returns:
            A pandas DataFrame with the requested data.
        """
        filtered_data = self.metadata.get_data_between_dates_from_api(*args, **kwargs)

        # fix types
        stc = self.metadata.sourcetableconfiguration
        filtered_data[stc.time_index_name] = pd.to_datetime(filtered_data[stc.time_index_name], utc=True)
        for c, c_type in stc.column_dtypes_map.items():
            if c != stc.time_index_name:
                if c_type == "object":
                    c_type = "str"
                filtered_data[c] = filtered_data[c].astype(c_type)
        filtered_data = filtered_data.set_index(stc.index_names)

        return filtered_data


class PersistManager:
    def __init__(self,
                 data_source: DynamicTableDataSource,
                 update_hash: str,
                 description: Optional[str] = None,
                 class_name: Optional[str] = None,
                 metadata: Optional[Dict] = None,
                 local_metadata: Optional[LocalTimeSerie] = None
                 ):
        """
        Initializes the PersistManager.

        Args:
            data_source: The data source for the time series.
            update_hash: The local hash identifier for the time series.
            description: An optional description for the time series.
            class_name: The name of the DataNode class.
            metadata: Optional remote metadata dictionary.
            local_metadata: Optional local metadata object.
        """
        self.data_source: DynamicTableDataSource = data_source
        self.update_hash: str = update_hash
        if local_metadata is not None and metadata is None:
            # query remote storage_hash
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

        if self.update_hash is not None:
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
                    update_hash=self.update_hash,
                    remote_table__data_source__id=self.data_source.id,
                    include_relations_detail=include_relations_detail
                )
                if result is None:
                    self.logger.warning(f"TimeSeries {self.update_hash} with data source {self.data_source.id} not found in backend")
                new_future.set_result(result)
            except Exception as exc:
                new_future.set_exception(exc)
            finally:
                # Remove the future from the global registry once done.
                future_registry.remove_future(new_future)

        thread = threading.Thread(target=_get_or_none_local_metadata,
                                  name=f"LocalMetadataThreadPM-{self.update_hash}",
                                  daemon=False)
        thread.start()



    def depends_on_connect(self, new_ts: "DataNode", is_api: bool) -> None:
        """
        Connects a time series as a relationship in the DB.

        Args:
            new_ts: The target DataNode to connect to.
            is_api: True if the target is an APIDataNode
        """
        if not is_api:
            self.local_metadata.depends_on_connect(
                                       target_time_serie_id=new_ts.local_time_serie.id
                                        )
        else:
            try:

                self.local_metadata.depends_on_connect_to_api_table(
                    target_table_id=new_ts.local_persist_manager.metadata.id
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

        response = ms_client.TimeSerieLocalUpdate.get_mermaid_dependency_diagram(update_hash=self.update_hash,
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

    def get_mermaid_dependency_diagram(self) -> str:
        """
        Displays a Mermaid.js dependency diagram in a Jupyter environment.

        Returns:
            The Mermaid diagram string.
        """
        from IPython.display import display, HTML

        mermaid_diagram = self.display_mermaid_dependency_diagram()

        # Mermaid.js initialization script (only run once)
        if not hasattr(display, "_mermaid_initialized"):
            mermaid_initialize = """
                   <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                   <script>
                       function initializeMermaid() {
                           if (typeof mermaid !== 'undefined') {
                               console.log('Initializing Mermaid.js...');
                               const mermaidDivs = document.querySelectorAll('.mermaid');
                               mermaidDivs.forEach(mermaidDiv => {
                                   mermaid.init(undefined, mermaidDiv);
                               });
                           } else {
                               console.error('Mermaid.js is not loaded.');
                           }
                       }
                   </script>
                   """
            display(HTML(mermaid_initialize))
            display._mermaid_initialized = True

        # HTML template for rendering the Mermaid diagram
        html_template = f"""
               <div class="mermaid">
               {mermaid_diagram}
               </div>
               <script>
                   initializeMermaid();
               </script>
               """

        # Display the Mermaid diagram in the notebook
        display(HTML(html_template))

        # Optionally return the raw diagram code for further use
        return mermaid_diagram

    def get_all_dependencies_update_priority(self) -> pd.DataFrame:
        """
        Retrieves a DataFrame of all dependencies with their update priority.

        Returns:
            A pandas DataFrame with dependency and priority information.
        """
        depth_df = self.local_metadata.get_all_dependencies_update_priority()
        return depth_df

    def set_ogm_dependencies_linked(self) -> None:
        self.local_metadata.patch(ogm_dependencies_linked=True)

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



    def add_tags(self, tags: List[str]) -> None:
        """Adds tags to the local time series metadata if they don't already exist."""
        if any([t not in self.local_metadata.tags for t in tags]) == True:
            self.local_metadata.add_tags(tags=tags)

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


        local_metadata_kwargs = dict(update_hash=self.update_hash,
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
            name=f"PatchBuildConfigThread-{self.update_hash}",
            daemon=False
        )
        thread.start()

        patch_future.add_done_callback(self.set_local_metadata_lazy_callback)


    def local_persist_exist_set_config(
            self,
            storage_hash: str,
            local_configuration: dict,
            remote_configuration: dict,
            data_source: DynamicTableDataSource,
            time_serie_source_code_git_hash: str,
            time_serie_source_code: str,
            remote_build_metadata: dict,
    ) -> None:
        """
        Ensures local and remote persistence objects exist and sets their configurations.
        This runs on DataNode initialization.
        """
        remote_build_configuration = None
        if hasattr(self, "remote_build_configuration"):
            remote_build_configuration = self.remote_build_configuration

        if remote_build_configuration is None:
            logger.debug(f"remote table {storage_hash} does not exist creating")
            #create remote table

            try:

                # table may not exist but
                remote_build_metadata = remote_configuration["build_meta_data"] if "build_meta_data" in remote_configuration.keys() else {}
                remote_configuration.pop("build_meta_data", None)
                kwargs = dict(storage_hash=storage_hash,
                              time_serie_source_code_git_hash=time_serie_source_code_git_hash,
                              time_serie_source_code=time_serie_source_code,
                              build_configuration=remote_configuration,
                              data_source=data_source.model_dump(),
                              build_meta_data=remote_build_metadata)


                dtd_metadata = DynamicTableMetaData.get_or_create(**kwargs)
                storage_hash = dtd_metadata.storage_hash
            except Exception as e:
                self.logger.exception(f"{storage_hash} Could not set meta data in DB for P")
                raise e
        else:
            self.set_local_metadata_lazy(force_registry=True, include_relations_detail=True)
            storage_hash = self.metadata.storage_hash

        local_table_exist = self._verify_local_ts_exists(storage_hash=storage_hash, local_configuration=local_configuration)


    def _verify_local_ts_exists(self, storage_hash: str,
                                local_configuration: Optional[Dict] = None) -> None:
        """
        Verifies that the local time series exists in the ORM, creating it if necessary.
        """
        local_build_configuration = None
        if self.local_metadata is not None:
            local_build_configuration, local_build_metadata = self.local_build_configuration, self.local_build_metadata
        if local_build_configuration is None:

            logger.debug(f"local_metadata {self.update_hash} does not exist creating")
            local_update = LocalTimeSerie.get_or_none(update_hash=self.update_hash,
                                                       remote_table__data_source__id=self.data_source.id)
            if local_update is None:
                local_build_metadata = local_configuration[
                    "build_meta_data"] if "build_meta_data" in local_configuration.keys() else {}
                local_configuration.pop("build_meta_data", None)
                metadata_kwargs = dict(
                    update_hash=self.update_hash,
                    build_configuration=local_configuration,
                    remote_table__hash_id=storage_hash,
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
            name=f"BuildUpdateDetailsThread-{self.update_hash}",
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





    def get_df_between_dates(self, *args, **kwargs) -> pd.DataFrame:
        """
        Retrieves a DataFrame from the data source between specified dates.
        """
        filtered_data = self.data_source.get_data_by_time_index(
            local_metadata=self.local_metadata,
            *args, **kwargs
        )
        return filtered_data

    def set_column_metadata(self,
                            columns_metadata: Optional[List[ms_client.ColumnMetaData]]
                            ) -> None:
        if self.metadata:
            if self.metadata.sourcetableconfiguration != None:
                if self.metadata.sourcetableconfiguration.columns_metadata is not None:
                    if columns_metadata is None:
                        self.logger.info(f"get_column_metadata method not implemented")
                        return

                    self.metadata.sourcetableconfiguration.set_or_update_columns_metadata(
                        columns_metadata=columns_metadata)

    def set_table_metadata(self,
                           table_metadata: ms_client.TableMetaData,
                           ):
        """
        Creates or updates the MarketsTimeSeriesDetails metadata in the backend.

        This method orchestrates the synchronization of the time series metadata,
        including its description, frequency, and associated assets, based on the
        configuration returned by `_get_time_series_meta_details`.
        """
        if not (self.metadata):
            self.logger.warning("metadata not set")
            return

        # 1. Get the user-defined metadata configuration for the time series.
        if table_metadata is None:
            return

        # 2. Get or create the MarketsTimeSeriesDetails object in the backend.
        source_table_id = self.metadata.patch(**table_metadata.model_dump())

    def delete_table(self) -> None:
        if self.data_source.related_resource.class_type == "duck_db":
            from mainsequence.client.data_sources_interfaces.duckdb import DuckDBInterface
            db_interface = DuckDBInterface()
            db_interface.drop_table(self.metadata.storage_hash)

        self.metadata.delete()

    @tracer.start_as_current_span("TS: Persist Data")
    def persist_updated_data(self,
                             temp_df: pd.DataFrame, overwrite: bool = False) -> bool:
        """
        Persists the updated data to the database.

        Args:
            temp_df: The DataFrame with updated data.
            update_tracker: The update tracker object.
            overwrite: If True, overwrites existing data.

        Returns:
            True if data was persisted, False otherwise.
        """
        persisted = False
        if not temp_df.empty:
            if overwrite == True:
                self.logger.warning(f"Values will be overwritten")

            self._local_metadata_cached = self.local_metadata.upsert_data_into_table(
                data=temp_df,
                data_source=self.data_source,

            )


            persisted = True
        return persisted

    def get_update_statistics_for_table(self) -> ms_client.UpdateStatistics:
        """
        Gets the latest update statistics from the database.

        Args:
            unique_identifier_list: An optional list of unique identifiers to filter by.

        Returns:
            A UpdateStatistics object with the latest statistics.
        """
        if isinstance(self.metadata, int):
            self.set_local_metadata_lazy(force_registry=True, include_relations_detail=True)

        if self.metadata.sourcetableconfiguration is None:
            return ms_client.UpdateStatistics()

        update_stats = self.metadata.sourcetableconfiguration.get_data_updates()
        return update_stats

    def is_local_relation_tree_set(self) -> bool:
        return self.local_metadata.ogm_dependencies_linked



    def update_git_and_code_in_backend(self,time_serie_class) -> None:
        """Updates the source code and git hash information in the backend."""
        self.update_source_informmation(
            git_hash_id=get_data_node_source_code_git_hash(time_serie_class),
            source_code=get_data_node_source_code(time_serie_class),
        )

class TimeScaleLocalPersistManager(PersistManager):
    """
    Main Controler to interacti with backend
    """
    def get_table_schema(self,table_name):
        return self.metadata["sourcetableconfiguration"]["column_dtypes_map"]



