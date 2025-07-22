from typing import Tuple, Optional, Dict, Any, Union
from mainsequence.client import UpdateStatistics, LocalTimeSerieUpdateDetails
from mainsequence.instrumentation import tracer, tracer_instrumentator
import mainsequence.client as ms_client
import structlog.contextvars as cvars
import datetime
import pandas as pd
import time
import numpy as np
from mainsequence.instrumentation import tracer, tracer_instrumentator
import mainsequence.tdag.time_series.build_operations as build_operations

class DependencyUpdateError(Exception):
    pass


def get_update_map( declared_dependencies: Dict[str,'TimeSerie'],
                    logger:object,
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

        # Ensure the dependency is initialized in the persistence layer
        dependency_ts.local_persist_manager

        logger.debug(f"Adding dependency '{name}' to update map.")
        dependecy_map[key] = {"is_pickle": False, "ts": dependency_ts}
        declared_dependencies=dependency_ts.dependencies() or {}
        # Recursively call get_update_map on the dependency to traverse the entire graph
        get_update_map(declared_dependencies=declared_dependencies,
                       logger=logger,
                       dependecy_map=dependecy_map)

    return dependecy_map

def load_dependencies(time_serie_instance :"TimeSerie") -> None:
    """Fetches and sets the dependencies DataFrame."""
    if time_serie_instance.dependencies_df is None:  # Lazy loading
        time_serie_instance.logger.debug("Initializing dependency data...")
        depth_df = time_serie_instance.local_persist_manager.get_all_dependencies_update_priority()
        time_serie_instance.depth_df = depth_df

        if not depth_df.empty:
            # Filter out the owner itself from the dependency list
            time_serie_instance.dependencies_df = depth_df[
                depth_df["local_time_serie_id"] != time_serie_instance.local_persist_manager.local_metadata.id].copy()
        else:
            time_serie_instance.dependencies_df = pd.DataFrame()


def pre_update_setting_routines(run_time_serie:"TimeSerie", scheduler: "Scheduler", set_time_serie_queue_status: bool, update_tree: bool,
                                 local_metadata: Optional[dict] = None) -> Tuple[Dict, Any]:
    """
    Routines to execute before an update.

    Args:
        scheduler: The scheduler object.
        set_time_serie_queue_status: Whether to set the queue status.
        update_tree: Whether to update the tree.
        metadata: Optional remote metadata.
        local_metadata: Optional local metadata.

    Returns:
        A tuple containing the local metadata map and state data.
    """
    # reset metadata
    run_time_serie.local_persist_manager.synchronize_metadata(local_metadata=local_metadata)
    run_time_serie.set_relation_tree()


    update_priority_dict = None
    # build priority update

    load_dependencies(time_serie_instance=run_time_serie)

    if not run_time_serie._scheduler_tree_connected and update_tree:
        run_time_serie.logger.debug("Connecting dependency tree to scheduler...")
        # only set once
        all_local_time_series_ids_in_tree = []

        if not run_time_serie.depth_df.empty:
            all_local_time_series_ids_in_tree = run_time_serie.depth_df["local_time_serie_id"].to_list()
            if update_tree == True:
                scheduler.in_active_tree_connect(local_time_series_ids=all_local_time_series_ids_in_tree + [run_time_serie.local_persist_manager.local_metadata.id])
            run_time_serie._scheduler_tree_connected = True

    depth_df = run_time_serie.depth_df.copy()
    # set active tree connections

    if not depth_df.empty > 0:
        all_local_time_series_ids_in_tree = depth_df[["local_time_serie_id"]].to_dict("records")
    all_local_time_series_ids_in_tree.append({"local_time_serie_id":run_time_serie.local_persist_manager.local_metadata.id})

    update_details_batch = dict(error_on_last_update=False,
                                active_update_scheduler_uid=scheduler.uid)
    if set_time_serie_queue_status == True:
        update_details_batch['active_update_status'] = "Q"
    all_metadatas = ms_client.LocalTimeSerie.get_metadatas_and_set_updates(local_time_series_ids=[i["local_time_serie_id"] for i in all_local_time_series_ids_in_tree],

                                                       update_details_kwargs=update_details_batch,
                                                       update_priority_dict=update_priority_dict,
                                                       )
    state_data, local_metadatas, source_table_config_map = all_metadatas['state_data'], all_metadatas[
        "local_metadatas"], all_metadatas["source_table_config_map"]
    local_metadatas = {m.id: m for m in local_metadatas}

    run_time_serie.scheduler = scheduler

    run_time_serie.update_details_tree = {key: v.run_configuration for key, v in local_metadatas.items()}
    return local_metadatas, state_data




def set_update_statistics(time_serie_instance: "TimeSerie",
                          update_statistics: UpdateStatistics) -> UpdateStatistics:
    """
    Default method to narrow down update statistics un local time series,
    the method will filter using asset_list if the attribute exists as well as the init fallback date
    :param update_statistics:
    :return:
    """
    # Filter update_statistics to include only assets in self.asset_list.

    asset_list = time_serie_instance.get_asset_list()
    time_serie_instance._setted_asset_list = asset_list

    update_statistics = update_statistics.update_assets(
        asset_list, init_fallback_date=time_serie_instance.OFFSET_START
    )
    return update_statistics


@tracer.start_as_current_span("Starting TS Update")
def start_time_serie_update(
        run_time_serie: "TimeSerie",
        update_tracker: object, debug_mode: bool,
        update_tree: bool = False,
        local_time_series_map: Optional[Dict[str, "LocalTimeSerie"]] = None,
        update_only_tree: bool = False, force_update: bool = False,
        use_state_for_update: bool = False) -> bool:
    """
    Main update method for a TimeSerie that interacts with the graph node.

    Args:
        update_tracker: The update tracker object.
        debug_mode: Whether to run in debug mode.
        update_tree: Whether to update the entire dependency tree.
        local_time_series_map: A map of local time series.
        update_only_tree: If True, only updates the dependency tree structure.
        force_update: If True, forces an update.
        use_state_for_update: If True, uses the current state for the update.

    Returns:
        True if there was an error on the last update, False otherwise.
    """
    running_time_serie = run_time_serie
    try:
        local_time_serie_historical_update = running_time_serie.local_persist_manager.local_metadata.set_start_of_execution(
            active_update_scheduler_uid=update_tracker.scheduler_uid)
    except Exception as e:
        raise e

    latest_value, must_update = local_time_serie_historical_update.last_time_index_value, local_time_serie_historical_update.must_update
    update_statistics = local_time_serie_historical_update.update_statistics
    error_on_last_update = False

    if force_update == True or update_statistics.max_time_index_value is None:
        must_update = True

    # Update statistics and build and rebuild localmetadata with foreign relations
    running_time_serie.local_persist_manager.set_local_metadata_lazy(include_relations_detail=True)
    update_statistics = set_update_statistics(time_serie_instance=running_time_serie,
                                              update_statistics=update_statistics)

    try:
        if must_update == True:

            update_local(
                running_time_serie=running_time_serie,
                update_tree=update_tree, debug_mode=debug_mode,
                overwrite_latest_value=latest_value,
                local_time_series_map=local_time_series_map,
                update_tracker=update_tracker, update_only_tree=update_only_tree,
                use_state_for_update=use_state_for_update, update_statistics=update_statistics
            )

            running_time_serie.local_persist_manager.local_metadata.set_end_of_execution(
                historical_update_id=local_time_serie_historical_update.id,
                error_on_update=error_on_last_update)
        else:
            running_time_serie.logger.info("Already updated, waiting until next update time")

    except Exception as e:
        error_on_last_update = True
        raise e
    finally:
        running_time_serie.local_persist_manager.local_metadata.set_end_of_execution(
            historical_update_id=local_time_serie_historical_update.id,
            error_on_update=error_on_last_update)
        # always set last relations details
        running_time_serie.local_persist_manager.set_local_metadata_lazy(include_relations_detail=True,
                                                                         )
        running_time_serie._run_post_update_routines(error_on_last_update=error_on_last_update,
                                                     update_statistics=update_statistics
                                                     )

        running_time_serie.local_persist_manager.set_column_metadata(
                                               columns_metadata=running_time_serie.get_column_metadata()
                                               )
        running_time_serie.local_persist_manager.set_table_metadata(
                                              table_metadata=running_time_serie.get_table_metadata(
                                                  update_statistics=update_statistics)
                                              )

    return error_on_last_update


def set_actor_manager(
        time_serie_instance: "TimeSerie",
        actor_manager: object) -> None:
    """
    Sets the actor manager for distributed updates.

    Args:
        actor_manager: The actor manager object.
    """
    time_serie_instance.update_actor_manager = actor_manager


def setup_scheduler(running_time_serie: "TimeSerie", debug_mode: bool,
                    remote_scheduler: Optional[object]) -> "Scheduler":
    from mainsequence.client import Scheduler

    """Initializes or retrieves the scheduler and starts its heartbeat."""
    if remote_scheduler is None:
        name_prefix = "DEBUG_" if debug_mode else ""
        scheduler = Scheduler.build_and_assign_to_ts(
            scheduler_name=f"{name_prefix}{running_time_serie.local_hash_id}_{running_time_serie.data_source.id}",
            local_hash_id_list=[(running_time_serie.local_hash_id, running_time_serie.data_source.id)],
            remove_from_other_schedulers=True,
            running_in_debug_mode=debug_mode
        )
        scheduler.start_heart_beat()
        return scheduler
    return remote_scheduler


def setup_execution_environment(running_time_serie: "TimeSerie", scheduler: "Scheduler", debug_mode: bool,
                                update_tree: bool) -> Tuple[
    object, dict]:
    from mainsequence.tdag.time_series.update.utils import UpdateInterface, wait_for_update_time
    from mainsequence.tdag.time_series.update.ray_manager import RayUpdateManager
    """Sets up distributed actors and gathers pre-update state."""
    distributed_actor_manager = RayUpdateManager(scheduler_uid=scheduler.uid, skip_health_check=True)
    set_actor_manager(time_serie_instance=running_time_serie,
                      actor_manager=distributed_actor_manager)

    local_time_series_map, state_data = pre_update_setting_routines(
        run_time_serie=running_time_serie,
        scheduler=scheduler,
        set_time_serie_queue_status=False,
        update_tree=update_tree
    )

    update_tracker = UpdateInterface(
        head_hash=running_time_serie.local_hash_id,
        logger=running_time_serie.logger,
        state_data=state_data,
        debug=debug_mode,
        scheduler_uid=scheduler.uid,
        trace_id=None,
    )
    return update_tracker, local_time_series_map


def execute_core_update(
        running_time_serie: "TimeSerie",
        update_tracker: object, force_update: bool, **kwargs):
    """Waits if necessary, then starts the main update process."""
    from mainsequence.tdag.time_series.update.utils import wait_for_update_time

    if not force_update:
        wait_for_update_time(
            local_hash_id=running_time_serie.local_hash_id,
            data_source_id=running_time_serie.data_source.id,
            logger=running_time_serie.logger,
            force_next_start_of_minute=False
        )

    # This is the final delegation to the BuildManager to start the update
    return start_time_serie_update(
        run_time_serie=running_time_serie,
        update_tracker=update_tracker,
        force_update=force_update,
        **kwargs
    )


def run(
        running_time_serie: "TimeSerie",
        debug_mode: bool,
        *,
        update_tree: bool = True,
        force_update: bool = False,
        update_only_tree: bool = False,
        remote_scheduler: Union[object, None] = None
):
    """

    Args:
        debug_mode: if the time serie is run in debug mode the DAG will be run node by node in the same process
        update_tree: if set to False then only the selected time series will be run, default is True
        force_update: Force an update even if the time serie schedule does not require an update
        update_only_tree: If set to True then only the dependency graph of the selected time serie will be updated
        remote_scheduler:
    """
    from mainsequence.instrumentation import TracerInstrumentator
    from mainsequence.tdag.time_series.update.utils import UpdateInterface, wait_for_update_time
    from mainsequence.tdag.time_series.update.ray_manager import RayUpdateManager
    import gc
    global logger
    if update_tree:
        update_only_tree = False

    # set tracing
    tracer_instrumentator = TracerInstrumentator()
    tracer = tracer_instrumentator.build_tracer()
    error_on_update = None
    # 1 Create Scheduler for this time serie

    scheduler = setup_scheduler(running_time_serie=running_time_serie,
                                debug_mode=debug_mode, remote_scheduler=remote_scheduler)
    cvars.bind_contextvars(scheduler_name=scheduler.name, head_local_ts_hash_id=running_time_serie.local_hash_id)

    error_to_raise = None
    with tracer.start_as_current_span(f"Scheduler TS Head Update ") as span:
        span.set_attribute("time_serie_local_hash_id", running_time_serie.local_hash_id)
        span.set_attribute("remote_table_hashed_name", running_time_serie.remote_table_hashed_name)
        span.set_attribute("head_scheduler", scheduler.name)

        # 2 add actor manager for distributed
        distributed_actor_manager = RayUpdateManager(scheduler_uid=scheduler.uid,
                                                     skip_health_check=True
                                                     )
        try:
            update_tracker, local_map = setup_execution_environment(running_time_serie=running_time_serie,
                                                                    scheduler=scheduler, debug_mode=debug_mode,
                                                                    update_tree=update_tree)

            set_actor_manager(actor_manager=distributed_actor_manager, time_serie_instance=running_time_serie)
            running_time_serie.logger.debug("state set with dependencies metadatas")

            running_time_serie.update_tracker = update_tracker
            execute_core_update(
                running_time_serie=running_time_serie,
                update_tracker=update_tracker,
                debug_mode=debug_mode,
                force_update=force_update,
                update_tree=update_tree,
                update_only_tree=update_only_tree,
                local_time_series_map=local_map,
                use_state_for_update=True
            )
            del running_time_serie.update_tracker
            gc.collect()
        except TimeoutError as te:
            running_time_serie.logger.error("TimeoutError Error on update")
            error_to_raise = te
        except DependencyUpdateError as de:
            running_time_serie.logger.error("DependecyError on update")
            error_to_raise = de
        except Exception as e:
            running_time_serie.logger.exception(e)
            error_to_raise = e

    if remote_scheduler == None:
        scheduler.stop_heart_beat()
    if error_to_raise != None:
        raise error_to_raise


@tracer.start_as_current_span("Verify time series tree update")
def verify_tree_is_updated(
        run_time_serie: "TimeSerie",
        local_time_series_map: Dict[int, "LocalTimeSerie"],
        debug_mode: bool,
        use_state_for_update: bool = False
) -> None:
    """
    Verifies that the dependency tree is updated.

    Args:
        local_time_series_map: A map of local time series objects.
        debug_mode: Whether to run in debug mode.
        use_state_for_update: If True, uses the current state for the update.
    """
    # build tree
    if run_time_serie.local_persist_manager.is_local_relation_tree_set() == False:
        start_tree_relationship_update_time = time.time()
        run_time_serie.set_relation_tree(time_serie_instance=run_time_serie)
        run_time_serie.logger.debug(
            f"relationship tree updated took {time.time() - start_tree_relationship_update_time} seconds ")

    else:
        run_time_serie.logger.debug("Tree is not updated as is_local_relation_tree_set== True")

    update_map = {}
    if use_state_for_update == True:
        declared_dependencies = run_time_serie.dependencies() or {}

        update_map = get_update_map(declared_dependencies=declared_dependencies,
                                                     logger=run_time_serie.logger)

    run_time_serie.logger.debug(
        f"Updating tree with update map {list(update_map.keys())} and dependencies {run_time_serie.dependencies_df['local_hash_id'].to_list()}")

    if debug_mode == False:
        tmp_ts = run_time_serie.dependencies_df.copy()
        if tmp_ts.shape[0] == 0:
            run_time_serie.logger.debug("No dependencies in this time serie")
            return None
        tmp_ts = tmp_ts[tmp_ts["source_class_name"] != "WrapperTimeSerie"]

        if tmp_ts.shape[0] > 0:
            execute_parallel_distributed_update(tmp_ts=tmp_ts,
                                                local_time_series_map=local_time_series_map,
                                                )
    else:
        updated_uids = []
        if run_time_serie.dependencies_df.shape[0] > 0:
            unique_priorities = run_time_serie.dependencies_df["update_priority"].unique().tolist()
            unique_priorities.sort()

            local_time_series_list = run_time_serie.dependencies_df[
                run_time_serie.dependencies_df["source_class_name"] != "WrapperTimeSerie"
                ][["local_hash_id", "data_source_id"]].values.tolist()
            for prioriity in unique_priorities:
                # get hierarchies ids
                tmp_ts = run_time_serie.dependencies_df[
                    run_time_serie.dependencies_df["update_priority"] == prioriity].sort_values(
                    "number_of_upstreams", ascending=False).copy()

                tmp_ts = tmp_ts[tmp_ts["source_class_name"] != "WrapperTimeSerie"]
                tmp_ts = tmp_ts[~tmp_ts.index.isin(updated_uids)]

                # update on the same process
                for row, ts_row in tmp_ts.iterrows():

                    if (ts_row["local_hash_id"], ts_row["data_source_id"]) in update_map.keys():
                        ts = update_map[(ts_row["local_hash_id"], ts_row["data_source_id"])]["ts"]
                    else:
                        try:

                            ts, _ = build_operations.rebuild_and_set_from_local_hash_id(
                                local_hash_id=ts_row["local_hash_id"],
                                data_source_id=ts_row["data_source_id"]
                            )

                        except Exception as e:
                            run_time_serie.logger.exception(
                                f"Error updating dependency {ts_row['local_hash_id']} when loading pickle")
                            raise e

                    try:
                        #todo: remove
                        ts, _ = build_operations.rebuild_and_set_from_local_hash_id(
                            local_hash_id=ts_row["local_hash_id"],
                            data_source_id=ts_row["data_source_id"]
                        )
                        error_on_last_update = start_time_serie_update(run_time_serie=ts,
                                                                       debug_mode=debug_mode,
                                                                       update_tree=False,
                                                                       update_tracker=run_time_serie.update_tracker
                                                                       )


                    except Exception as e:
                        run_time_serie.logger.exception(f"Error updating dependencie {ts.local_hash_id}")
                        raise e
                updated_uids.extend(tmp_ts.index.to_list())
    run_time_serie.logger.debug(f'Dependency Tree evaluated for  {run_time_serie}')


@tracer.start_as_current_span("Execute distributed parallel update")
def execute_parallel_distributed_update(run_time_serie: "TimeSerie", tmp_ts: pd.DataFrame,
                                        local_time_series_map: Dict[int, "LocalTimeSerie"]) -> None:
    """
    Executes a parallel distributed update of dependencies.

    Args:
        tmp_ts: A DataFrame of time series to update.
        local_time_series_map: A map of local time series objects.
    """

    telemetry_carrier = tracer_instrumentator.get_telemetry_carrier()

    pre_loaded_ts = [t.hash_id for t in run_time_serie.scheduler.pre_loads_in_tree]
    tmp_ts = tmp_ts.sort_values(["update_priority", "number_of_upstreams"], ascending=[True, False])
    pre_load_df = tmp_ts[tmp_ts["local_time_serie_id"].isin(pre_loaded_ts)].copy()
    tmp_ts = tmp_ts[~tmp_ts["local_time_serie_id"].isin(pre_loaded_ts)].copy()
    tmp_ts = pd.concat([pre_load_df, tmp_ts], axis=0)

    futures_ = []

    local_time_series_list = run_time_serie.dependencies_df[
        run_time_serie.dependencies_df["source_class_name"] != "WrapperTimeSerie"
        ]["local_time_serie_id"].values.tolist()

    for counter, (uid, data) in enumerate(tmp_ts.iterrows()):
        local_time_serie_id = data['local_time_serie_id']
        data_source_id = data['data_source_id']
        local_hash_id = data['local_hash_id']

        kwargs_update = dict(local_time_serie_id=local_time_serie_id,
                             local_hash_id=local_hash_id,
                             data_source_id=data_source_id,
                             telemetry_carrier=telemetry_carrier,
                             scheduler_uid=run_time_serie.scheduler.uid
                             )

        update_details = run_time_serie.update_details_tree[local_time_serie_id]
        run_configuration = local_time_series_map[local_time_serie_id].run_configuration
        num_cpus = run_configuration.required_cpus

        task_kwargs = dict(task_options={"num_cpus": num_cpus,
                                         "name": f"local_time_serie_id_{local_time_serie_id}",

                                         "max_retries": run_configuration.retry_on_error},
                           kwargs_update=kwargs_update)

        p = run_time_serie.update_actor_manager.launch_update_task(**task_kwargs)

        # p = self.update_actor_manager.launch_update_task_in_process( **task_kwargs  )
        # continue
        # logger.warning("REMOVE LINES ABOVE FOR DEBUG")

        futures_.append(p)

        # are_dependencies_updated, all_dependencies_nodes, pending_nodes, error_on_dependencies = self.update_tracker.get_pending_update_nodes(
        #     hash_id_list=list(all_start_data.keys()))
        # self.are_dependencies_updated( target_nodes=all_dependencies_nodes)
        # raise Exception

    tasks_with_errors = run_time_serie.update_actor_manager.get_results_from_futures_list(futures=futures_)
    if len(tasks_with_errors) > 0:
        raise DependencyUpdateError(f"Update Stop from error in Ray in tasks {tasks_with_errors}")
    # verify there is no error in hierarchy. this prevents to updating next level if dependencies fails

    dependencies_update_details = LocalTimeSerieUpdateDetails.filter(
        related_table__id__in=tmp_ts["local_time_serie_id"].astype(str).to_list())
    ts_with_errors = []
    for local_ts_update_details in dependencies_update_details:
        if local_ts_update_details.error_on_last_update == True:
            ts_with_errors.append(local_ts_update_details.related_table.id)
    # Verify there are no errors after finishing hierarchy
    if len(ts_with_errors) > 0:
        raise DependencyUpdateError(f"Update Stop from error in children \n {ts_with_errors}")


@tracer.start_as_current_span("TimeSerie.update_local")
def update_local(running_time_serie: "TimeSerie", update_tree: bool, update_tracker: object, debug_mode: bool,
                 update_statistics: UpdateStatistics,
                 local_time_series_map: Optional[dict] = None,
                 overwrite_latest_value: Optional[datetime.datetime] = None, update_only_tree: bool = False,
                 use_state_for_update: bool = False,
                 *args, **kwargs) -> Optional[bool]:
    """
    Performs a local update of the time series data.

    Args:
        update_tree: Whether to update the dependency tree.
        update_tracker: The update tracker object.
        debug_mode: Whether to run in debug mode.
        update_statistics: The data update statistics.
        local_time_series_map: A map of local time series objects.
        overwrite_latest_value: An optional timestamp to overwrite the latest value.
        update_only_tree: If True, only updates the dependency tree structure.
        use_state_for_update: If True, uses the current state for the update.

    Returns:
        True if data was persisted, False otherwise.
    """
    from mainsequence.instrumentation.utils import Status, StatusCode
    persisted = False
    if update_tree == True:

        verify_tree_is_updated(debug_mode=debug_mode, run_time_serie=running_time_serie,
                               local_time_series_map=local_time_series_map,
                               use_state_for_update=use_state_for_update,
                               )
        if update_only_tree == True:
            running_time_serie.logger.info(f'Local Time Series  {running_time_serie} only tree updated')
            return None

    with tracer.start_as_current_span("Update Calculation") as update_span:

        if overwrite_latest_value is not None:  # overwrite latest values is passed form def_update method to reduce calls to api
            latest_value = overwrite_latest_value

            running_time_serie.logger.info(
                f'Updating Local Time Series for  {running_time_serie}  since {latest_value}')
            temp_df = running_time_serie.update(update_statistics)

            if temp_df.shape[0] == 0:
                # concatenate empty

                running_time_serie.logger.info(f'Local Time Series Nothing to update  {running_time_serie}  updated')
                return False

            for col, ddtype in temp_df.dtypes.items():
                if "datetime64" in str(ddtype):
                    running_time_serie.logger.info(f"WARNING DATETIME TYPE IN {running_time_serie}")
                    raise Exception(f"""Datetime in {col}
                                        {temp_df}""")
            running_time_serie.logger.info(f'Persisting Time Series for  {running_time_serie}  since {latest_value} ')

        else:
            if not update_statistics:
                running_time_serie.logger.info(f'Updating Local Time Series for  {running_time_serie}  for first time')
            try:
                temp_df = running_time_serie.update(update_statistics)
                temp_df = update_statistics.filter_df_by_latest_value(temp_df)
                temp_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

            if not temp_df.empty:
                lvl0 = temp_df.index.get_level_values(0)
                is_dt64_utc = str(lvl0.dtype) == "datetime64[ns, UTC]"
                assert is_dt64_utc, f"Time index must be datetime64[ns, UTC] ({lvl0} is used)"
            else:
                running_time_serie.logger.warning(f"Time Series {running_time_serie} does not return data from update")

            for col in temp_df.columns:
                assert col.islower(), f"Error Column '{col}': Column names must be lower case"

            for col, ddtype in temp_df.dtypes.items():
                if "datetime64" in str(ddtype):
                    running_time_serie.logger.info(f"WARNING DATETIME TYPE IN {running_time_serie}")
                    raise Exception

        try:

            # verify index order is correct
            overwrite = True if overwrite_latest_value is not None else False
            persisted = running_time_serie.local_persist_manager.persist_updated_data(temp_df=temp_df,
                                                                            update_tracker=update_tracker,
                                                                            overwrite=overwrite)

            update_span.set_status(Status(StatusCode.OK))
        except Exception as e:
            running_time_serie.logger.exception("Error updating time serie")
            update_span.set_status(Status(StatusCode.ERROR))
            raise e
        running_time_serie.logger.info(f'Local Time Series  {running_time_serie}  updated')

        return persisted
