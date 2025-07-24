
# Standard Library Imports
import gc
import time
import datetime
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Library Imports
import numpy as np
import pandas as pd
import structlog.contextvars as cvars
import pytz
# Internal Project Imports



# Client and ORM Models
import mainsequence.client as ms_client
from mainsequence.client import (
    LocalTimeSerie,
    LocalTimeSerieUpdateDetails,
    Scheduler,
    UpdateStatistics
)

# Instrumentation and Logging
from mainsequence.instrumentation import (
    tracer,
    tracer_instrumentator,
    TracerInstrumentator
)
from mainsequence.instrumentation.utils import Status, StatusCode

# TDAG Core Components and Helpers
from mainsequence.tdag.time_series import build_operations
from mainsequence.tdag.time_series.update.utils import (
    UpdateInterface,
    wait_for_update_time
)

# Custom Exceptions
class DependencyUpdateError(Exception):
    pass


class UpdateRunner:
    """
       Orchestrates the entire update process for a TimeSerie instance.
       It handles scheduling, dependency resolution, execution, and error handling.
       """

    def __init__(self, time_serie: "TimeSerie", debug_mode: bool = False, force_update: bool = False,
                 update_tree: bool = True, update_only_tree: bool = False,
                 remote_scheduler: Optional[Scheduler] = None):
        from mainsequence.tdag.time_series.update.ray_manager import RayUpdateManager
        self.ts = time_serie
        self.logger = self.ts.logger
        self.debug_mode = debug_mode
        self.force_update = force_update
        self.update_tree = update_tree
        self.update_only_tree = update_only_tree
        if self.update_tree:
            self.update_only_tree = False

        self.remote_scheduler = remote_scheduler
        self.scheduler: Optional[Scheduler] = None
        self.update_tracker: Optional[UpdateInterface] = None
        self.update_actor_manager: Optional[RayUpdateManager] = None

    def _setup_scheduler(self) -> None:
        """Initializes or retrieves the scheduler and starts its heartbeat."""
        if self.remote_scheduler:
            self.scheduler = self.remote_scheduler
            return

        name_prefix = "DEBUG_" if self.debug_mode else ""
        self.scheduler = ms_client.Scheduler.build_and_assign_to_ts(
            scheduler_name=f"{name_prefix}{self.ts.local_time_serie.id}",
            time_serie_ids=[self.ts.local_time_serie.id],
            remove_from_other_schedulers=True,
            running_in_debug_mode=self.debug_mode
        )
        self.scheduler.start_heart_beat()

    def _pre_update_routines(self, local_metadata: Optional[dict] = None) -> Tuple[Dict[int,ms_client.LocalTimeSerie], Any]:
        """
        Prepares the TimeSerie and its dependencies for an update by fetching the
        latest metadata for the entire dependency graph.

        Args:
            local_metadata: Optional dictionary with metadata for the head node,
                            used to synchronize before fetching the full tree.

        Returns:
            A tuple containing a dictionary of all local metadata objects in the
            tree (keyed by ID) and the corresponding state data.
        """
        # 1. Synchronize the head node and load its dependency structure.
        self.ts.local_persist_manager.synchronize_metadata(local_metadata=local_metadata)
        self.ts.set_relation_tree()

        # The `load_dependencies` logic is now integrated here.
        if self.ts.dependencies_df is None:
            depth_df = self.ts.local_persist_manager.get_all_dependencies_update_priority()
            self.ts.depth_df = depth_df
            if not depth_df.empty:
                self.ts.dependencies_df = depth_df[
                    depth_df["local_time_serie_id"] != self.ts.local_time_serie.id].copy()
            else:
                self.ts.dependencies_df = pd.DataFrame()

        # 2. Connect the dependency tree to the scheduler if it hasn't been already.
        if not self.ts._scheduler_tree_connected and self.update_tree:
            self.logger.debug("Connecting dependency tree to scheduler...")
            if not self.ts.depth_df.empty:
                all_ids = self.ts.depth_df["local_time_serie_id"].to_list() + [self.ts.local_time_serie.id]
                self.scheduler.in_active_tree_connect(local_time_series_ids=all_ids)
            self.ts._scheduler_tree_connected = True

        # 3. Collect all IDs in the dependency graph to fetch their metadata.
        # This correctly initializes the list, fixing the original bug.
        if not self.ts.depth_df.empty:
            all_ids_in_tree = self.ts.depth_df["local_time_serie_id"].to_list()
        else:
            all_ids_in_tree = []

        # Always include the head node itself.
        all_ids_in_tree.append(self.ts.local_time_serie.id)

        # 4. Fetch the latest metadata for the entire tree from the backend.
        update_details_batch = dict(
            error_on_last_update=False,
            active_update_scheduler_id=self.scheduler.id,
            active_update_status="Q"  # Assuming queue status is always set here
        )

        all_metadatas_response = ms_client.LocalTimeSerie.get_metadatas_and_set_updates(
            local_time_series_ids=all_ids_in_tree,
            update_details_kwargs=update_details_batch,
            update_priority_dict=None
        )

        # 5. Process and return the results.
        state_data = all_metadatas_response['state_data']
        local_metadatas_list = all_metadatas_response["local_metadatas"]
        local_metadatas_map = {m.id: m for m in local_metadatas_list}

        self.ts.scheduler = self.scheduler
        self.ts.update_details_tree = {key: v.run_configuration for key, v in local_metadatas_map.items()}

        return local_metadatas_map, state_data

    def _setup_execution_environment(self) -> Dict[int, LocalTimeSerie]:
        from mainsequence.tdag.time_series.update.ray_manager import RayUpdateManager

        """Sets up distributed actors and gathers pre-update state."""
        self.update_actor_manager = RayUpdateManager(scheduler_id=self.scheduler.id, skip_health_check=True)
        self.ts.update_actor_manager = self.update_actor_manager  # Assign to TS if it needs direct access

        local_metadatas, state_data = self._pre_update_routines()

        self.update_tracker = UpdateInterface(
            head_hash=self.ts.local_hash_id,
            logger=self.logger,
            state_data=state_data,
            debug=self.debug_mode,
            scheduler_id=self.scheduler.id,
            trace_id=None,
        )
        self.ts.update_tracker = self.update_tracker  # Assign to TS if it needs direct access
        return local_metadatas

    def _start_update(self, local_time_series_map: Dict, use_state_for_update: bool):
        """Orchestrates a single TimeSerie update, including pre/post routines."""
        historical_update = self.ts.local_persist_manager.local_metadata.set_start_of_execution(
            active_update_scheduler_id=self.scheduler.id
        )
        update_statistics = historical_update.update_statistics
        must_update = historical_update.must_update or self.force_update

        # 👇 THIS IS THE ADDED LINE
        # Ensure metadata is fully loaded with relationship details before proceeding.
        self.ts.local_persist_manager.set_local_metadata_lazy(include_relations_detail=True)

        # The TimeSerie defines how to scope its statistics
        update_statistics = self.ts._set_update_statistics(update_statistics)

        error_on_last_update = False
        try:
            if must_update:
                self.logger.info(f"Update required for {self.ts}.")
                self._update_local(
                    update_statistics=update_statistics,
                    local_time_series_map=local_time_series_map,
                    overwrite_latest_value=historical_update.last_time_index_value,
                    use_state_for_update=use_state_for_update
                )
            else:
                self.logger.info(f"Already up-to-date. Skipping update for {self.ts}.")
        except Exception as e:
            error_on_last_update = True
            raise e
        finally:
            self.ts.local_persist_manager.local_metadata.set_end_of_execution(
                historical_update_id=historical_update.id,
                error_on_update=error_on_last_update
            )

            # 👇 IT IS ALSO IMPORTANT TO RE-SYNC AT THE END
            # Always set last relations details after the run completes.
            self.ts.local_persist_manager.set_local_metadata_lazy(include_relations_detail=True)

            self.ts.run_post_update_routines(error_on_last_update=error_on_last_update,
                                             update_statistics=update_statistics)
            self.ts.local_persist_manager.set_column_metadata(columns_metadata=self.ts.get_column_metadata())
            self.ts.local_persist_manager.set_table_metadata(
                table_metadata=self.ts.get_table_metadata(update_statistics=update_statistics))

        return error_on_last_update

    def _validate_update_dataframe(self, df: pd.DataFrame) -> None:
        """
        Performs a series of critical checks on the DataFrame before persistence.

        Args:
            df: The DataFrame returned from the TimeSerie's update method.

        Raises:
            AssertionError or Exception if any validation check fails.
        """
        # Check for infinite values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Check that the time index is a UTC datetime
        time_index = df.index.get_level_values(0)
        if not pd.api.types.is_datetime64_ns_dtype(time_index) or str(time_index.tz) !=str(datetime.timezone.utc)  :
            raise TypeError(f"Time index must be datetime64[ns, UTC], but found {time_index.dtype}")

        # Check for forbidden data types and enforce lowercase columns
        for col, dtype in df.dtypes.items():
            if not isinstance(col, str) or not col.islower():
                raise ValueError(f"Column name '{col}' must be a lowercase string.")
            if "datetime64" in str(dtype):
                raise TypeError(f"Column '{col}' has a forbidden datetime64 dtype.")
    @tracer.start_as_current_span("UpdateRunner._update_local")
    def _update_local(
            self,
            update_statistics: UpdateStatistics,
            local_time_series_map: Dict[int, Any],
            overwrite_latest_value: Optional[datetime.datetime],
            use_state_for_update: bool,
    ) -> Optional[bool]:
        """
        Calculates, validates, and persists the data update for the time series.
        """
        # 1. Handle dependency tree update first

        if self.update_tree:
            self._verify_tree_is_updated(local_time_series_map, use_state_for_update)
            if self.update_only_tree:
                self.logger.info(f'Dependency tree for {self.ts} updated. Halting run as requested.')
                return None

        # 2. Execute the core data calculation
        with tracer.start_as_current_span("Update Calculation") as update_span:

            # 👇 THIS IS THE CORRECTED LOGIC
            # Add specific log message for the initial run
            if not update_statistics:
                self.logger.info(f"Performing first-time update for {self.ts}...")
            else:
                self.logger.info(f'Calculating update for {self.ts}...')

            try:
                # Call the business logic defined on the TimeSerie class
                temp_df = self.ts.update(update_statistics)

                # If the update method returns no data, we're done.
                if temp_df.empty:
                    self.logger.warning(f"No new data returned from update for {self.ts}.")
                    return False

                # In a normal run, filter out data we already have.
                if overwrite_latest_value is None:
                    temp_df = update_statistics.filter_df_by_latest_value(temp_df)

                # If filtering left nothing, we're done.
                if temp_df.empty:
                    self.logger.info(f"No new data to persist for {self.ts} after filtering.")
                    return False

                # Validate the structure and content of the DataFrame
                self._validate_update_dataframe(temp_df)

                # Persist the validated data
                self.logger.info(f'Persisting {len(temp_df)} new rows for {self.ts}.')
                persisted = self.ts.local_persist_manager.persist_updated_data(
                    temp_df=temp_df,
                    update_tracker=self.update_tracker,
                    overwrite=(overwrite_latest_value is not None)
                )
                update_span.set_status(Status(StatusCode.OK))
                self.logger.info(f'Successfully updated {self.ts}.')
                return persisted

            except Exception as e:
                self.logger.exception("Failed during update calculation or persistence.")
                update_span.set_status(Status(StatusCode.ERROR, description=str(e)))
                raise e

    @tracer.start_as_current_span("UpdateRunner._verify_tree_is_updated")
    def _verify_tree_is_updated(
            self,
            local_time_series_map: Dict[int, Any],
            use_state_for_update: bool,
    ) -> None:
        """
        Ensures all dependencies in the tree are updated before the head node.

        This method checks if the dependency graph is defined in the backend and
        then delegates the update execution to either a sequential (debug) or
        parallel (production) helper method.

        Args:
            local_time_series_map: A map of local time series objects in the tree.
            use_state_for_update: If True, uses the current state for the update.
        """
        # 1. Ensure the dependency graph is built in the backend
        if not self.ts.local_persist_manager.is_local_relation_tree_set():
            self.logger.info("Dependency tree not set. Building now...")
            start_time = time.time()
            self.ts.set_relation_tree()
            self.logger.debug(f"Tree build took {time.time() - start_time:.2f}s.")

        # 2. Get the list of dependencies to update
        dependencies_df = self.ts.dependencies_df
        if dependencies_df.empty:
            self.logger.debug("No dependencies to update.")
            return

        # 3. Build a map of dependency instances if needed for debug mode
        update_map = {}
        if self.debug_mode and use_state_for_update:
            declared_dependencies = self.ts.dependencies() or {}
            update_map = self._get_update_map(declared_dependencies,
                                              logger=self.logger
                                              )

        # 4. Delegate to the appropriate execution method
        self.logger.info(f"Starting update for {len(dependencies_df)} dependencies...")

        dependencies_df = dependencies_df[dependencies_df["source_class_name"] != "WrapperTimeSerie"]

        if self.debug_mode:
            self._execute_sequential_debug_update(dependencies_df, update_map,
                                                  local_time_series_map)
        else:
            # Filter out WrapperTimeSerie as they don't have a backend table to update
            deps_to_run = dependencies_df[dependencies_df["source_class_name"] != "WrapperTimeSerie"]
            if not deps_to_run.empty:
                self._execute_parallel_distributed_update(deps_to_run, local_time_series_map)

        self.logger.debug(f'Dependency tree evaluation complete for {self.ts}.')


    def _get_update_map(self,declared_dependencies: Dict[str, 'TimeSerie'],
                       logger: object,
                       dependecy_map: Optional[Dict] = None) -> Dict[
        Tuple[str, int], Dict[str, Any]]:
        """
        Obtains all TimeSerie objects in the dependency graph by recursively
        calling the dependencies() method.

        This approach is more robust than introspecting class members as it relies
        on an explicit declaration of dependencies.

        Args:
            time_serie_instance: The TimeSerie instance from which to start the dependency traversal.
            dependecy_map: An optional dictionary to store the dependency map, used for recursion.

        Returns:
            A dictionary mapping (local_hash_id, data_source_id) to TimeSerie info.
        """
        # Initialize the map on the first call
        if dependecy_map is None:
            dependecy_map = {}

        # Get the explicitly declared dependencies, just like set_relation_tree

        for name, dependency_ts in declared_dependencies.items():
            key = (dependency_ts.local_hash_id, dependency_ts.data_source_id)

            # If we have already processed this node, skip it to prevent infinite loops
            if key in dependecy_map:
                continue
            if dependency_ts.is_api == True:
                continue

            # Ensure the dependency is initialized in the persistence layer
            dependency_ts.local_persist_manager

            logger.debug(f"Adding dependency '{name}' to update map.")
            dependecy_map[key] = {"is_pickle": False, "ts": dependency_ts}
            declared_dependencies = dependency_ts.dependencies() or {}
            # Recursively call get_update_map on the dependency to traverse the entire graph
            self._get_update_map(declared_dependencies=declared_dependencies,
                           logger=logger,
                           dependecy_map=dependecy_map)

        return dependecy_map

    def _execute_sequential_debug_update(
            self,
            dependencies_df: pd.DataFrame,
            update_map: Dict[Tuple[str, int], Dict],
            local_time_series_map:Dict[int, ms_client.LocalTimeSerie]
    ) -> None:
        """Runs dependency updates sequentially in the same process for debugging."""
        self.logger.info("Executing dependency updates in sequential debug mode.")
        # Sort by priority to respect the DAG execution order
        sorted_priorities = sorted(dependencies_df["update_priority"].unique())


        for priority in sorted_priorities:
            priority_df = dependencies_df[dependencies_df["update_priority"] == priority]
            # Sort by number of upstreams to potentially optimize within a priority level
            sorted_deps = priority_df.sort_values("number_of_upstreams", ascending=False)



            for _, ts_row in sorted_deps.iterrows():
                key = (ts_row["local_hash_id"], ts_row["data_source_id"])
                ts_to_update = None
                try:
                    if key in update_map:
                        ts_to_update = update_map[key]["ts"]
                    else:
                        # If not in the map, it must be rebuilt from storage
                        ts_to_update, _ = build_operations.rebuild_and_set_from_local_hash_id(
                            local_hash_id=key[0], data_source_id=key[1]
                        )

                    if ts_to_update:
                        self.logger.debug(f"Running debug update for dependency: {ts_to_update.local_hash_id}")
                        # Each dependency gets its own clean runner
                        dep_runner = UpdateRunner(
                            time_serie=ts_to_update,
                            debug_mode=True,
                            update_tree=False,  # We only update one node at a time
                            force_update=self.force_update,
                            remote_scheduler=self.scheduler,
                        )
                        dep_runner._setup_scheduler()
                        dep_runner.update_tracker = self.update_tracker

                        dep_runner._start_update(
                            local_time_series_map=local_time_series_map,
                            use_state_for_update=False,
                        )
                except Exception as e:
                    self.logger.exception(f"Failed to update dependency {key[0]}")
                    raise e  # Re-raise to halt the entire process on failure

    # This code is a method within the UpdateRunner class.
    # Assumes 'ms_client', 'tracer_instrumentator', and 'DependencyUpdateError' are imported.

    @tracer.start_as_current_span("UpdateRunner._execute_parallel_distributed_update")
    def _execute_parallel_distributed_update(
            self,
            dependencies_to_run_df: pd.DataFrame,
            local_time_series_map: Dict[int, "LocalTimeSerie"]
    ) -> None:
        """
        Launches and manages a parallel, distributed update for a set of dependencies.

        This method sorts the dependencies, prepares and launches remote execution
        tasks (e.g., via Ray), and then waits for their completion, handling
        any errors that arise from the distributed run.

        Args:
            dependencies_to_run_df: A DataFrame of time series dependencies to update.
            local_time_series_map: A map of local time series objects keyed by ID.

        Raises:
            DependencyUpdateError: If any of the distributed tasks fail.
        """
        # 1. Prepare tasks, prioritizing any pre-loaded time series
        telemetry_carrier = tracer_instrumentator.get_telemetry_carrier()
        pre_loaded_ts_ids = [t.hash_id for t in self.scheduler.pre_loads_in_tree]

        # Sort all dependencies by priority and structure
        sorted_deps = dependencies_to_run_df.sort_values(
            ["update_priority", "number_of_upstreams"], ascending=[True, False]
        )
        # Separate and move pre-loaded tasks to the front of the queue
        pre_load_df = sorted_deps[sorted_deps["local_time_serie_id"].isin(pre_loaded_ts_ids)]
        other_deps_df = sorted_deps[~sorted_deps["local_time_serie_id"].isin(pre_loaded_ts_ids)]
        tasks_df = pd.concat([pre_load_df, other_deps_df], axis=0)

        # 2. Launch all dependency updates as distributed tasks
        futures_ = []
        for _, data_row in tasks_df.iterrows():
            local_ts_id = data_row['local_time_serie_id']

            # Get the specific run configuration for this dependency
            run_configuration = local_time_series_map[local_ts_id].run_configuration

            # Prepare arguments for the remote task
            kwargs_update = dict(
                local_time_serie_id=local_ts_id,
                local_hash_id=data_row['local_hash_id'],
                data_source_id=data_row['data_source_id'],
                telemetry_carrier=telemetry_carrier,
                scheduler_id=self.scheduler.id
            )
            task_options = dict(
                num_cpus=run_configuration.required_cpus,
                name=f"ts_update_{local_ts_id}",
                max_retries=run_configuration.retry_on_error
            )

            p = self.update_actor_manager.launch_update_task(
                task_options=task_options,
                kwargs_update=kwargs_update
            )

            # p = self.update_actor_manager.launch_update_task_in_process( **task_kwargs  )
            # continue
            # logger.warning("REMOVE LINES ABOVE FOR DEBUG")

            futures_.append(p)

            # are_dependencies_updated, all_dependencies_nodes, pending_nodes, error_on_dependencies = self.update_tracker.get_pending_update_nodes(
            #     hash_id_list=list(all_start_data.keys()))
            # self.are_dependencies_updated( target_nodes=all_dependencies_nodes)
            # raise Exception

        # 3. Wait for results and check for errors from the distributed run
        tasks_with_errors = self.update_actor_manager.get_results_from_futures_list(futures=futures_)
        if tasks_with_errors:
            raise DependencyUpdateError(f"Update stopped due to errors in Ray tasks: {tasks_with_errors}")

        # 4. Final verification: Check the database status for all tasks in the batch
        updated_ids = tasks_df["local_time_serie_id"].astype(str).to_list()
        dependencies_update_details = ms_client.LocalTimeSerieUpdateDetails.filter(
            related_table__id__in=updated_ids
        )

        failed_ids = [
            detail.related_table.id for detail in dependencies_update_details if detail.error_on_last_update
        ]

        if failed_ids:
            raise DependencyUpdateError(f"Update stopped after children reported errors: {failed_ids}")

    # This method is the primary public method of the UpdateRunner class.
    # Assumes other private methods (_setup_scheduler, _setup_execution_environment, etc.)
    # and necessary imports are defined within the class.

    def run(self) -> None:
        """
        Executes the full update lifecycle for the time series.

        This is the main entry point for the runner. It orchestrates the setup
        of scheduling and the execution environment, triggers the core update
        process, and handles all error reporting and cleanup.
        """
        # Initialize tracing and set initial flags
        tracer_instrumentator = TracerInstrumentator()
        tracer = tracer_instrumentator.build_tracer()
        error_to_raise = None

        # 1. Set up the scheduler for this run
        try:

            self.ts.verify_and_build_remote_objects()#needed to start sch
            self._setup_scheduler()
            cvars.bind_contextvars(scheduler_name=self.scheduler.name, head_local_ts_hash_id=self.ts.local_hash_id)

            # 2. Start the main execution block with tracing
            with tracer.start_as_current_span(f"Scheduler Head Update: {self.ts.local_hash_id}") as span:
                span.set_attribute("time_serie_local_hash_id", self.ts.local_hash_id)
                span.set_attribute("remote_table_hashed_name", self.ts.remote_table_hashed_name)
                span.set_attribute("head_scheduler", self.scheduler.name)

                # 3. Prepare the execution environment (Ray actors, dependency metadata)
                local_time_series_map = self._setup_execution_environment()
                self.logger.debug("Execution environment and dependency metadata are set.")

                # 4. Wait for the scheduled update time, if not forcing an immediate run
                if not self.force_update:
                    wait_for_update_time(
                        local_hash_id=self.ts.local_hash_id,
                        data_source_id=self.ts.data_source.id,
                        logger=self.logger
                    )

                # 5. Trigger the core update process
                self._start_update(
                    local_time_series_map=local_time_series_map,
                    use_state_for_update=True
                )

        except DependencyUpdateError as de:
            self.logger.error("A dependency failed to update, halting the run.", error=de)
            error_to_raise = de
        except TimeoutError as te:
            self.logger.error("The update process timed out.", error=te)
            error_to_raise = te
        except Exception as e:
            self.logger.exception("An unexpected error occurred during the update run.")
            error_to_raise = e
        finally:
            # 6. Clean up resources
            # Stop the scheduler heartbeat if it was created by this runner
            if self.remote_scheduler is None and self.scheduler:
                self.scheduler.stop_heart_beat()

            # Clean up temporary attributes on the TimeSerie instance
            if hasattr(self.ts, 'update_tracker'):
                del self.ts.update_tracker
            if hasattr(self.ts, 'update_actor_manager'):
                del self.ts.update_actor_manager

            gc.collect()

        # 7. Re-raise any captured exception after cleanup
        if error_to_raise:
            raise error_to_raise