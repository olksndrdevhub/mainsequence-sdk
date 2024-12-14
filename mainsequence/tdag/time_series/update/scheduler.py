import os
import datetime
import pytz
import time

from mainsequence.tdag_client import (Scheduler, TimeSerieLocalUpdate, SourceTableConfigurationDoesNotExist,
                                      LocalTimeSerieNode
                                      )
from .update_methods import (get_or_pickle_ts_from_sessions, update_remote_from_hash_id,
                             update_remote_from_hash_id_local)
from mainsequence.tdag.config import bcolors
from typing import Union, List
import ray
import pandas as pd
from mainsequence.tdag.time_series.time_series import DependencyUpdateError, TimeSerie
import gc
from mainsequence.tdag.logconf import get_tdag_logger, create_logger_in_path

from contextlib import contextmanager

logger = get_tdag_logger()
TDAG_RAY_CLUSTER_ADDRESS = os.getenv("TDAG_RAY_CLUSTER_ADDRESS")
NAMESPACE = "time_series_update"

USE_PICKLE = os.environ.get('TDAG_USE_PICKLE', True)
USE_PICKLE = USE_PICKLE in ["True", "true", True]


class RayUpdateManager:

    def __init__(self, scheduler_uid,
                 skip_health_check=False,
                 local_mode=False):

        self.scheduler_uid = scheduler_uid

        if skip_health_check == False:
            self.is_node_healthy = self.check_node_is_healthy_in_ip()
            if self.is_node_healthy:
                self.verify_ray_is_initialized(local_mode)

    # Node health interactions

    def verify_ray_is_initialized(self, local_mode=False):
        from mainsequence import tdag
        import os
        from mainsequence.tdag.config import Configuration
        if ray.is_initialized() == False:
            self.check_node_is_healthy_in_ip()
            ray_address = TDAG_RAY_CLUSTER_ADDRESS
            env_vars = {

                "RAY_PROFILING": "0", "RAY_event_stats": "0",
                "RAY_BACKEND_LOG_LEVEL": "error", }

            for c in Configuration.OBLIGATORY_ENV_VARIABLES:
                env_vars[c] = os.environ.get(c)
            for c in Configuration.OPTIONAL_ENV_VARIABLES:
                if c in os.environ:
                    env_vars[c] = os.environ.get(c)

            kwargs = dict(address=ray_address, namespace=NAMESPACE,
                          local_mode=local_mode,
                          # log_to_driver=False,
                          runtime_env={"env_vars": env_vars,
                                       "py_modules": [tdag]
                                       },
                          )  # Todo add ray cluster configuration

            ray.init(**kwargs)

    def shutdown_manager(self):
        if ray.is_initialized() == True:
            ray.shutdown()

    def check_node_is_healthy_in_ip(self) -> bool:
        return True  # todo get function out oof experimental
        healthy = False
        api_address = configuration.conf.distributed_config["ray"]["head_node_ip"]
        try:
            all_nodes = list_nodes()
        except ConnectionError:
            return False

        for n in list_nodes():
            if api_address == n["node_ip"] and n["state"] == "ALIVE":
                healthy = True
        return healthy

    # misc helpers
    def get_results_from_futures_list(self, futures: list) -> list:
        """
        should be a list of futures objects ray.remote()
        Args:
            futures ():

        Returns:

        """
        ready, unready = ray.wait(futures, num_returns=1)
        tasks_with_errors = []
        while unready:
            # logger.debug(ready)
            # logger.debug(unready)
            try:
                ray.get(ready)
            except Exception as e:
                logger.error(e)
                tasks_with_errors.append(ready)
            ready, unready = ray.wait(unready, num_returns=1)

        return tasks_with_errors

    # launch methods helpers to work with Actors
    def launch_update_task(self, kwargs_update: dict, task_options: dict):
        # update_remote_from_hash_id(**kwargs_update)
        # return  None
        future = update_remote_from_hash_id.options(**task_options).remote(**kwargs_update)
        return future

    def launch_update_task_in_process(self, kwargs_update: dict, task_options: dict):
        update_remote_from_hash_id_local(**kwargs_update)


class TimeSerieHeadUpdateActor:
    TRACE_ID = "NO_TRACE"

    def __init__(self, local_hash_id: str, data_source_id: int, scheduler: Scheduler, wait_for_update, debug,
                 update_tree, update_extra_kwargs, remote_table_hashed_name: str,

                 ):
        """

        Parameters
        ----------
        hash_id :
        scheduler :
        wait_for_update :
        debug :
        update_tree :
        update_extra_kwargs :
        """
        self.update_tree = update_tree
        self.remote_table_hashed_name = remote_table_hashed_name
        self.ts = self._load_time_serie(local_hash_id=local_hash_id, remote_table_hashed_name=remote_table_hashed_name,
                                        data_source_id=data_source_id,
                                        scheduler=scheduler)
        self.scheduler = scheduler
        self.wait_for_update = wait_for_update

        self.update_extra_kwargs = update_extra_kwargs
        self.local_hash_id = local_hash_id
        self.debug = debug

    def _load_time_serie(self, local_hash_id: str,
                         remote_table_hashed_name: str, data_source_id: int,
                         scheduler) -> "TimeSerie":
        from mainsequence.tdag.time_series import TimeSerie

        distributed_actor_manager = RayUpdateManager(scheduler_uid=scheduler.uid,
                                                     skip_health_check=True
                                                     )

        if USE_PICKLE == False:
            ts = TimeSerie.rebuild_from_configuration(local_hash_id=local_hash_id,
                                                      remote_table_hashed_name=remote_table_hashed_name,
                                                      data_source=data_source_id,
                                                      )
        else:
            _ = get_or_pickle_ts_from_sessions(local_hash_id=local_hash_id,
                                               remote_table_hashed_name=remote_table_hashed_name,
                                               set_dependencies_df=True,
                                               data_source_id=data_source_id,
                                               )
            pickle_path = TimeSerie.get_pickle_path(local_hash_id,data_source_id)
            if os.path.isfile(pickle_path) == True:
                ts = TimeSerie.load_from_pickle(pickle_path=pickle_path)
            else:
                ts = TimeSerie.rebuild_from_configuration(local_hash_id=local_hash_id,
                                                          remote_table_hashed_name=remote_table_hashed_name,
                                                          data_source=data_source_id,
                                                          )

            local_metadatas, state_data = ts.pre_update_setting_routines(scheduler=scheduler,
                                                                         set_time_serie_queue_status=False,
                                                                         update_tree=self.update_tree)

            ts.set_state_with_sessions(
                graph_depth=0,
                graph_depth_limit=0,
                include_vam_client_objects=False,
                local_metadatas=local_metadatas
            )
            ts.logger.info("state set with dependencies metadatas")

        ts.set_actor_manager(actor_manager=distributed_actor_manager)

        return ts

    def run_one_step_update(self, force_update=False,
                            force_next_start_of_minute=False, update_only_tree=False, ):
        """
        Main update Method for a time serie Head
        Returns
        -------

        """
        from mainsequence.tdag.instrumentation import TracerInstrumentator
        from mainsequence.tdag.config import configuration
        from .utils import UpdateInterface
        error_on_update = False
        tracer_instrumentator = TracerInstrumentator(
            configuration.configuration["instrumentation_config"]["grafana_agent_host"])
        tracer = tracer_instrumentator.build_tracer("tdag_head_distributed", __name__)
        with tracer.start_as_current_span(f"Scheduler TS Head Update ") as span:
            span.set_attribute("time_serie_local_hash_id", self.local_hash_id)
            span.set_attribute("remote_table_hashed_name", self.remote_table_hashed_name)
            span.set_attribute("head_scheduler", self.scheduler.name)

            try:
                all_metadatas, state_data = self.ts.pre_update_setting_routines(update_tree=self.update_tree,
                                                                                set_time_serie_queue_status=True,
                                                                                scheduler=self.scheduler)  # set scheduler before wait to use waiting time
                # build udpate Tracker
                update_tracker = UpdateInterface(head_hash=self.local_hash_id, trace_id=self.TRACE_ID,
                                                 logger=self.ts.logger,
                                                 state_data=state_data, debug=self.debug,
                                                 )
                self.ts.update_tracker = update_tracker

                if self.wait_for_update == True and force_update == False:
                    SchedulerUpdater.wait_for_update_time(local_hash_id=self.local_hash_id,
                                                          logger=self.ts.logger,
                                                          force_next_start_of_minute=force_next_start_of_minute)



                error_on_update = self.ts.update(debug_mode=self.debug, raise_exceptions=True,force_update=force_update,
                                                 update_tree=self.update_tree, update_only_tree=update_only_tree,
                                                 metadatas=all_metadatas, update_tracker=self.ts.update_tracker,
                                                 )

                # self.ts.patch_update_details(update_pid=0, active_update_status=False)
                try:
                    self.ts.local_persist_manager.set_last_index_value()
                except SourceTableConfigurationDoesNotExist as de:
                    self.ts.logger.warning("Last index value not set as source table configuration does not exist")

                del self.ts.update_tracker
                gc.collect()

            except TimeoutError:
                self.ts.logger.error("TimeoutError Error on update")
                error_on_update = True
            except DependencyUpdateError as e:
                self.ts.logger.error("DependecyError on update")
                error_on_update = True
            except Exception as e:
                self.ts.logger.exception(e)
                error_on_update = True
        return error_on_update


@ray.remote(num_cpus=1, )
class TimeSerieHeadUpdateActorDist(TimeSerieHeadUpdateActor):
    ...


@contextmanager
def set_data_lake(pod_source, override_all: bool = False):
    """

    :param override_all:
    :return:
    """
    vars = ["POD_DEFAULT_DATA_SOURCE", "POD_DEFAULT_DATA_SOURCE_FORCE_OVERRIDE"]
    original_values = {k: os.environ.get(k, None) for k in vars}

    # Override the environment variables
    os.environ["POD_DEFAULT_DATA_SOURCE"] = pod_source.model_dump_json()
    os.environ["POD_DEFAULT_DATA_SOURCE_FORCE_OVERRIDE"] = str(override_all)

    try:
        yield pod_source
    finally:
        # Restore the original environment variables
        for key, value in original_values.items():
            if value is None:  # Variable was not originally set
                del os.environ[key]
            else:
                os.environ[key] = value


class LocalDataLakeScheduler:
    @staticmethod
    def setup_datalake(
            datalake_end="2024-09-01 00:00:00",
            datalake_name="Data Lake",
            datalake_start="2022-08-01 00:00:00",
            nodes_to_get_from_db="",
            persist_logs_to_file=False,
            use_s3_if_available=True,
            specific_file_name="datalake_configuration.yaml",
            output_dir=None
    ):
        """
        Sets up the data lake configuration by creating a YAML file with the provided parameters.

        Parameters:
        - datalake_end (str): The end date of the data lake.
        - datalake_name (str): The name of the data lake.
        - datalake_start (str): The start date of the data lake.
        - nodes_to_get_from_db (str): Nodes to retrieve from the database.
        - persist_logs_to_file (bool): Whether to persist logs to a file.
        - use_s3_if_available (bool): Whether to use S3 if available.
        - specific_file_name (str): The name of the YAML configuration file.
        - output_dir (str): The directory where the YAML file will be saved.

        Returns:
        - str: The path to the created YAML configuration file.
        """
        import tempfile

        # Use the system's temporary directory if no output directory is specified
        if output_dir is None:
            output_dir = tempfile.gettempdir()

        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # YAML content for the data lake configuration
        import textwrap
        yaml_content = textwrap.dedent(f"""\
               datalake_end: {datalake_end}
               datalake_name: {datalake_name}
               datalake_start: {datalake_start}
               nodes_to_get_from_db: {nodes_to_get_from_db}
               persist_logs_to_file: {str(persist_logs_to_file).lower()}
               use_s3_if_available: {str(use_s3_if_available).lower()}
           """)

        # Full path to the YAML configuration file
        yaml_file_path = os.path.join(output_dir, specific_file_name)

        # Write the YAML content to the file
        with open(yaml_file_path, 'w') as yaml_file:
            yaml_file.write(yaml_content)

        print(f"YAML configuration file created at: {yaml_file_path}")

        return yaml_file_path

    @staticmethod
    def local_datalake_run(
            head_ts: TimeSerie,
            nodes_to_get_from_db=Union[List[str], None],
    ) -> pd.DataFrame:
        """
                Execute a full data tree run to retrieve the portfolio data with specified end node hashes. Utilized the
                local data lake, which greatly increases the speed due to not storing the node results in TSORM.

                Args:

                Returns:
                    A DataFrame containing the portfolio weights calculated up to the specified end nodes.
                """
        self.logger = create_logger_in_path(logger_name="scheduler_data_lake" + head_ts.local_hash_id,
                                            logger_file=f"{logging_folder}/schedulers/{head_ts.local_hash_id}.log",
                                            scheduler_name=scheduler.name

                                            )
        if nodes_to_get_from_db is None:
            logger.debug("Automatically inferring datalake nodes")
            head_ts.set_dependencies_df()
            nodes_to_get_from_db = find_ts_recursively(head_ts,
                                                       nodes_to_get_from_db)
            nodes_to_get_from_db = list(set([ts.local_hash_id for ts in nodes_to_get_from_db]))
        if not nodes_to_get_from_db:  # Handle the case where no nodes are found
            logger.warning(f" No Ts find with {nodes_to_get_from_db} data lake cant start  with empty origin node")
            return pd.DataFrame()

        self.logger.debug(f"TS data will be stored in data lake for: {nodes_to_get_from_db}")
        # TODO
        # for all nodes in ts, check if new data in remote tables is needed and update it before
        start_time = time.time()
        logger.debug("Start data run")
        df = head_ts.get_df_greater_than_in_table(None)
        minutes, seconds = divmod(elapsed_time, 60)
        self.logger.info(f"Run finished full data run in {int(minutes)} minutes and {seconds:.2f} seconds.")
        return df


class SchedulerUpdater:
    ACTOR_WAIT_LIMIT = 80  # seconds to wait before sending task to get scheduled

    @classmethod
    def debug_schedule_ts(cls, time_serie_hash_id: str,
                          data_source_id: int,
                          debug: bool, update_tree: bool, wait_for_update=True,
                          raise_exception_on_error: bool = True,
                          break_after_one_update=True, force_update=False, run_head_in_main_process=False,
                          update_extra_kwargs=None, update_only_tree=False, name_suffix=None

                          ):

        # get_remote_hash_id

        new_scheduler = Scheduler.initialize_debug_for_ts(local_hash_id=time_serie_hash_id,
                                                          data_source_id=data_source_id,
                                                          name_suffix=name_suffix
                                                          )

        logger.info(f"Scheduler ID {new_scheduler.name}")
        try:

            updater = cls(scheduler=new_scheduler)
            updater.start(debug=debug, update_tree=update_tree, break_after_one_update=break_after_one_update,
                          run_head_in_main_process=run_head_in_main_process,
                          raise_exception_on_error=raise_exception_on_error,
                          wait_for_update=wait_for_update,
                          update_extra_kwargs=update_extra_kwargs,
                          force_update=force_update, update_only_tree=update_only_tree,
                          )

            new_scheduler.delete()

        except KeyboardInterrupt:

            new_scheduler.delete()
            raise KeyboardInterrupt
        except Exception as e:

            new_scheduler.delete()
            raise e

    @classmethod
    def start_from_uid(cls, uid: str, *args, **kwargs):
        scheduler = Scheduler.get(uid=uid)
        scheduler_updater = cls(scheduler=scheduler)
        scheduler_updater.start(*args, **kwargs)

    def __init__(self, scheduler: Scheduler):

        from mainsequence.tdag.config import logging_folder

        self.node_scheduler = scheduler

        self.logger = create_logger_in_path(logger_name="scheduler_" + self.node_scheduler.uid,
                                            logger_file=f"{logging_folder}/schedulers/{self.node_scheduler.uid}.log",
                                            scheduler_name=scheduler.name

                                            )

    @staticmethod
    def get_time_to_wait_from_hash_id(local_hash_id: str,data_source_id:int):
        from mainsequence.tdag_client import DynamicTableHelpers
        dth = DynamicTableHelpers()
        local_metadata = TimeSerieLocalUpdate.get(local_hash_id=local_hash_id,
                                                  data_source_id=data_source_id
                                                  )
        time_to_wait, next_update = SchedulerUpdater._get_node_time_to_wait(local_metadata=local_metadata)

        if next_update is None:
            next_update = datetime.datetime(1985, 1, 1).replace(tzinfo=pytz.utc)
        else:
            next_update = dth.request_to_datetime(next_update)
        return time_to_wait, next_update

    @staticmethod
    def _get_node_time_to_wait(local_metadata):

        next_update = local_metadata["localtimeserieupdatedetails"]["next_update"]
        time_to_wait = 0.0
        if next_update is not None:
            time_to_wait = (pd.to_datetime(next_update) - datetime.datetime.now(pytz.utc)).total_seconds()
            time_to_wait = max(0, time_to_wait)
        return time_to_wait, next_update

    @staticmethod
    def wait_for_update_time(local_hash_id, logger, force_next_start_of_minute=False):

        time_to_wait, next_update = SchedulerUpdater.get_time_to_wait_from_hash_id(local_hash_id=local_hash_id)
        if time_to_wait > 0 and force_next_start_of_minute == False:

            logger.info(f"Scheduler Waiting for ts update time at {next_update} {time_to_wait}")
            time.sleep(time_to_wait)
        else:
            time_to_wait = max(0, 60 - datetime.datetime.now(pytz.utc).second)
            logger.info(f"Scheduler Waiting for ts update at start of minute")
            time.sleep(time_to_wait)
        if force_next_start_of_minute == True:
            logger.info(f"Forcing Next Udpdate at start of minte")

    def _clear_idle_scheduled_tree(self, *args, **kwargs):
        pass

    def _node_scheduler_heart_beat(self):
        """
        Timestamps a heart-beat from scheduler
        Returns
        -------

        """

        self.node_scheduler.last_heart_beat = datetime.datetime.now(pytz.utc)
        self.node_scheduler.save()

    def _run_update_task(self, running_distributed_heads: bool, uid_to_wait: str, actors_map: dict,
                         task_hex_to_uid: dict, force_update: bool, force_next_start_of_minute: bool,
                         update_only_tree: bool) -> dict:
        actor_handle = actors_map[uid_to_wait]['actor_handle']
        if running_distributed_heads == True:
            task_handle = actor_handle.run_one_step_update.remote(force_next_start_of_minute=force_next_start_of_minute,
                                                                  update_only_tree=update_only_tree)
        else:
            # will automatically block execution
            force_next_update = actor_handle.run_one_step_update(force_update=force_update,
                                                                 update_only_tree=update_only_tree,
                                                                 force_next_start_of_minute=force_next_start_of_minute)
            task_handle = None
            actors_map[uid_to_wait]["force_next_update"] = force_next_update
        actors_map[uid_to_wait]["task_handle"] = task_handle
        if running_distributed_heads == True:
            task_hex_to_uid[task_handle.task_id().hex()] = uid_to_wait

        return actors_map

    def _build_actor_handle(self, running_distributed_heads: bool, update_init_kwargs: dict, actor_options: dict,
                            local_time_serie_uid: str, actors_map: dict):

        if running_distributed_heads == True:
            actor_handle = TimeSerieHeadUpdateActorDist.options(**actor_options).remote(**update_init_kwargs)
        else:
            actor_handle = TimeSerieHeadUpdateActor(**update_init_kwargs)

        force_next_update = False if local_time_serie_uid not in actors_map.keys() else actors_map[local_time_serie_uid][
            'force_next_update']
        actors_map[local_time_serie_uid] = dict(actor_handle=actor_handle, actor_options=actor_options,
                                         task_handle=None, force_next_update=force_next_update,
                                         update_init_kwargs=update_init_kwargs
                                         )

        return actors_map

    def _actor_launcher_manager(self, wait_list, sequential_update: bool,
                                running_distributed_heads, actors_map,
                                task_hex_to_uid, force_update=False, update_only_tree=False,
                                launch_backoff_wait: Union[None, float] = None):
        new_wait_list = {}

        time_to_wait, actor_launched = [], False
        for uid_to_wait, wait_details in wait_list.items():
            force_next_start_of_minute = False
            if actors_map[uid_to_wait]['force_next_update'] == True:
                wait_list[uid_to_wait]['next_update'] = datetime.datetime.now(pytz.utc)
                wait_details['next_update'] = wait_list[uid_to_wait]['next_update']
                force_next_start_of_minute = True
            tmp_wait = (wait_details['next_update'] - datetime.datetime.now(pytz.utc)).total_seconds()
            remote_table_hashed_name = wait_details['remote_table_hashed_name']
            time_to_wait.append(max(tmp_wait, 0))
            if tmp_wait < self.ACTOR_WAIT_LIMIT or force_update == True:
                # in debug and in run head in main process this is blocked
                actors_map = self._run_update_task(running_distributed_heads=running_distributed_heads,
                                                   uid_to_wait=uid_to_wait,
                                                   task_hex_to_uid=task_hex_to_uid,
                                                   actors_map=actors_map,
                                                   force_update=force_update,
                                                   force_next_start_of_minute=force_next_start_of_minute,
                                                   update_only_tree=update_only_tree
                                                   )
                if running_distributed_heads == False:
                    next_update = self._get_next_update_loop(local_hash_id=wait_details["local_hash_id"],
                                                             data_source_id=wait_details["data_source_id"])
                    new_wait_list[uid_to_wait] = dict(next_update=next_update,
                                                          remote_table_hashed_name=remote_table_hashed_name)
                    actors_map[uid_to_wait]["task_handle"] = None
                    sequential_update = False  # do not wait for other heads as it is been runn in main process
                if launch_backoff_wait is not None:
                    self.logger.info(f"Waiting {launch_backoff_wait} to start next update")
                    time.sleep(launch_backoff_wait)
                if sequential_update == True:
                    # wait until task is done
                    self.logger.info(f"{bcolors.BOLD} Sequential update waiting for {uid_to_wait} {bcolors.ENDC}")
                    ready, unready = ray.wait([actors_map[uid_to_wait]["task_handle"]], num_returns=1,
                                              timeout=60 * 15)
                    if len(ready) > 0:
                        task_hex_to_uid, tmp_wait_list = self._evaluate_read_task(ready=ready,
                                                                                      task_hex_to_uid=task_hex_to_uid,
                                                                                      wait_list=wait_list)

                        new_wait_list[uid_to_wait] = tmp_wait_list[uid_to_wait]
                        actors_map[uid_to_wait]["task_handle"] = None
            else:
                new_wait_list[uid_to_wait] = dict(next_update=wait_details['next_update'],
                                                      remote_table_hashed_name=remote_table_hashed_name)
                actors_map[uid_to_wait]["task_handle"] = None

        return new_wait_list, actors_map

    def _build_scheduler_actors_if_not_exist(self, actors_map: dict, wait_list: dict, update_tree: bool,
                                             wait_for_update: bool, debug: bool, update_extra_kwargs: dict,
                                             running_distributed_heads: bool, first_launch: bool,
                                             ):
        """
        Queries ts_orm scheduler and build update actors if not exist
        Returns
        -------

        """

        from mainsequence.tdag_client import DynamicTableHelpers
        dth = DynamicTableHelpers()
        self.refresh_node_scheduler()
        local_to_remote = {t.uid: {"updates_to":t.updates_to,"local_hash_id":t.hash_id} for t in self.node_scheduler.schedules_to}
        target_ts = self.node_scheduler.schedules_to

        # verify target_ts_in actors
        all_hds, uids_to_remove = target_ts, []
        for active_uid in actors_map.keys():
            if active_uid not in all_hds:
                self.logger.info(
                    f"{bcolors.WARNING}Killing and removing actor {active_uid} from scheduler{bcolors.ENDC}")
                # 1 kill actor
                try:
                    ray.kill(actors_map[active_uid]['actor_handle'])
                except Exception as e:
                    raise e
                uids_to_remove.append(active_uid)
                # 3 remove ts from wait_list
                if active_uid in wait_list.keys():
                    wait_list.pop(active_uid, None)
        for h in uids_to_remove:
            actors_map.pop(h, None)

        # Modify
        self.logger.info(
            f"{bcolors.OKBLUE} TODO: Implement clear logic for  update and remove schedulers{bcolors.ENDC}")

        for head_ts in target_ts:
            uid = head_ts.uid

            if uid in actors_map.keys():
                continue
            metadata = dth.get(hash_id=local_to_remote[uid]["updates_to"].hash_id,
                               data_source__id=local_to_remote[uid]["updates_to"].data_source_id,
                               )
            update_details = {"distributed_num_cpus": 1}
            if "dynamictableupdatedetails" in metadata.keys():
                raise NotImplementedError
                update_details = metadata["dynamictableupdatedetails"]

            update_init_kwargs = dict(local_hash_id=head_ts.hash_id,
                                      remote_table_hashed_name=local_to_remote[uid]["updates_to"].hash_id,
                                      data_source_id=local_to_remote[uid]["updates_to"].data_source_id,
                                      scheduler=self.node_scheduler,

                                      wait_for_update=wait_for_update, debug=debug, update_tree=update_tree,
                                      update_extra_kwargs=update_extra_kwargs,
                                      )

            actor_options = {"num_cpus": update_details['distributed_num_cpus'],
                             "name": f"{head_ts.hash_id}_{head_ts.data_source_id}",
                             "get_if_exists": True, "max_task_retries": 0, "max_concurrency": 2, "max_restarts": 0}

            actors_map = self._build_actor_handle(running_distributed_heads=running_distributed_heads,
                                                  local_time_serie_uid=head_ts.uid,
                                                  actor_options=actor_options, update_init_kwargs=update_init_kwargs,
                                                  actors_map=actors_map
                                                  )
            if first_launch == True and running_distributed_heads == True:
                time.sleep(30)
            _, nu = SchedulerUpdater.get_time_to_wait_from_hash_id(local_hash_id=head_ts.hash_id,data_source_id=head_ts.data_source_id)
            wait_list[uid] = {"next_update": nu,
                                  "remote_table_hashed_name": local_to_remote[uid]["updates_to"].hash_id,
                                  "data_source_id": local_to_remote[uid]["updates_to"].data_source_id,
                              "local_hash_id": local_to_remote[uid]["local_hash_id"]
                                  }

            self.logger.info(
                f"Actor for head ts with local hash_id {head_ts} built and added to update loop")
        return actors_map, target_ts, wait_list

    def _get_next_update_loop(self, local_hash_id: str,data_source_id:int):
        """
        loop until query succesfull to avoid breaking the  udpates
        Parameters
        ----------
        hash_id

        Returns
        -------

        """
        error_in_request = True
        while error_in_request == True:
            try:
                _, next_update = SchedulerUpdater.get_time_to_wait_from_hash_id(
                    local_hash_id=local_hash_id,data_source_id=data_source_id)
                error_in_request = False
            except Exception as e:
                self.logger.exception("Error getting wait time")
                time.sleep(60)
        return next_update

    def refresh_node_scheduler(self):
        error_in_request = True
        while error_in_request == True:
            try:
                scheduler = Scheduler.get(name=self.node_scheduler.name)
                self.node_scheduler = scheduler
                error_in_request = False
            except SchedulerDoesNotExist as e:
                self.logger.exception(f"Scheduler {scheduler.name} does not exist - shutting down")
                exit()
            except Exception as e:
                self.logger.exception("Error getting refreshing scheduler")
                time.sleep(60)

    def _evaluate_read_task(self, ready: list, task_hex_to_uid: dict, wait_list: dict):
        """
        makes evaluation of task if its ready
        Returns
        -------

        """
        for ready_task in ready:
            task_hex = ready_task.task_id().hex()
            local_hash_id = task_hex_to_uid[task_hex]
            try:
                task_answer = ray.get(ready_task)
            except ray.exceptions.RayTaskError as e:
                self.logger.exception(f"{local_hash_id} Ray Task Error")
            except ray.exceptions.RayActorError as e:
                self.logger.exception(f"{local_hash_id} Actor Error rebuilding actor")
                actor_details = actors_map[local_hash_id]
                actors_map = self._build_actor_handle(
                    running_distributed_heads=True,
                    actor_options=actor_details['actor_options'],
                    update_init_kwargs=actor_details['update_init_kwargs'],
                    actors_map=actors_map, local_hash_id=local_hash_id,
                )

            # rerun ready task
            task_hex_to_uid.pop(task_hex, None)
            remote_table_hashed_name = wait_list[local_hash_id]['remote_table_hashed_name']
            next_update = self._get_next_update_loop(local_hash_id=local_hash_id)
            wait_list[local_hash_id] = dict(next_update=next_update, remote_table_hashed_name=remote_table_hashed_name)

        return task_hex_to_uid, wait_list

    def _print_failed_updates(self):
        """
        Call scheduler time series TREE with detail
        Returns:

        """
        logger.info("Not implemented failed updates")
        # global_queue = GlobalTimeSerieQueueActor.options(max_concurrency=10,
        #                                                  get_if_exists=True,
        #                                                  lifetime="detached",
        #                                                  name="global_ts_queue").remote()
        # hash_id_map = ray.get(global_queue.get_update_queue.remote())
        # logger.info(pd.DataFrame({k: v for k, v in hash_id_map.items() if v['error_on_last_update'] == True}).T)

    def _scheduler_heart_beat_patch(self):
        from mainsequence.tdag_client.utils import get_network_ip
        try:
            self.node_scheduler.patch(is_running=True,
                                      running_process_pid=os.getpid(),
                                      running_in_debug_mode=self._debug_mode,
                                      last_heart_beat=datetime.datetime.utcnow().timestamp(),
                                      api_address=get_network_ip(),
                                      api_port=self._api_port
                                      )
        except Exception as e:
            logger.error(e)

    def start(self, debug=False, update_tree: Union[bool, dict] = True, break_after_one_update=False,
              wait_for_update=True,
              raise_exception_on_error=False,
              update_extra_kwargs: Union[None, dict] = None, run_head_in_main_process=False, force_update=False,
              sequential_update=False, update_only_tree=False, api_port: Union[int, None] = None
              ):
        """

        Parameters
        ----------
        debug :
        update_tree :
        break_after_one_update :
        wait_for_update :
        raise_exception_on_error :
        update_extra_kwargs :
        run_head_in_main_process :

        Returns
        -------

        """

        self._debug_mode = debug
        self._api_port = api_port

        self._scheduler_heart_beat_patch()

        update_extra_kwargs = {} if update_extra_kwargs is None else update_extra_kwargs

        _ = RayUpdateManager(scheduler_uid=self.node_scheduler.uid)  # on initi verifys ray is running
        running_distributed_heads = True
        if run_head_in_main_process == True or debug == True:
            running_distributed_heads = False

        actors_map, wait_list = {}, {}

        first_launch = True
        try:
            task_hex_to_uid = {}

            ready, unready = ray.wait([actor_details["task_handle"] for actor_details in actors_map.values()
                                       if actor_details["task_handle"] is not None
                                       ],
                                      num_returns=1)
            unready = True if len(unready) == 0 else unready

            while unready:
                task_hex_to_uid, wait_list = self._evaluate_read_task(ready=ready,
                                                                          task_hex_to_uid=task_hex_to_uid,
                                                                          wait_list=wait_list)

                # requery DB looking for new TS added to scheduler
                actors_map, target_ts, wait_list = self._build_scheduler_actors_if_not_exist(
                    actors_map=actors_map, update_tree=update_tree,
                    wait_for_update=wait_for_update, debug=debug,
                    update_extra_kwargs=update_extra_kwargs,
                    running_distributed_heads=running_distributed_heads,
                    wait_list=wait_list,
                    first_launch=first_launch,
                )
                first_launch = False

                # wait in this process to avoid  calling extra actors on long waiting task
                wait_list, actors_map = self._actor_launcher_manager(
                    running_distributed_heads=running_distributed_heads,
                    wait_list=wait_list, task_hex_to_uid=task_hex_to_uid,
                    sequential_update=sequential_update, force_update=force_update,
                    update_only_tree=update_only_tree,
                    actors_map=actors_map)
                if break_after_one_update == True:
                    break
                ts_updating = [c for c in target_ts if c not in wait_list.keys()]

                self.logger.info(f"""{bcolors.OKBLUE}Waiting for tasks to finish in scheduler {self.node_scheduler.uid}"
                                     timeseries updating {ts_updating}
                                     timeseries waiting:
                                     {pd.Series(wait_list).to_frame('next_update')} 
                                     {bcolors.ENDC}
                                     """
                                 )
                self._print_failed_updates()

                if len(ts_updating) == 0:
                    self._scheduler_heart_beat_patch()
                    time.sleep(30)
                ready, unready = ray.wait([actor_details["task_handle"] for actor_details in actors_map.values()
                                           if actor_details["task_handle"] is not None
                                           ], timeout=60,
                                          num_returns=1)
                unready = True if len(unready) == 0 else unready



        except KeyboardInterrupt as ki:
            self.node_scheduler.patch(is_running=False, running_process_pid=0, )
            self._clear_idle_scheduled_tree(node_scheduler=self.node_scheduler)
            raise KeyboardInterrupt
        except Exception as e:
            self.node_scheduler.patch(is_running=False, running_process_pid=0, )
            self.logger.exception(e)
            self._clear_idle_scheduled_tree(node_scheduler=self.node_scheduler)
            time.sleep(60)

            if raise_exception_on_error == True:
                raise e
