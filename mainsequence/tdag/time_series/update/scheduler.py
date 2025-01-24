import os
import datetime
import pytz
import time

from ray.util.client.common import ClientActorHandle

from mainsequence.tdag_client import (Scheduler, SourceTableConfigurationDoesNotExist,
                                      LocalTimeSerie, SchedulerDoesNotExist
                                      )

from .utils import get_time_to_wait_from_hash_id
from .ray_manager import RayUpdateManager
from mainsequence.tdag.config import bcolors
from typing import Union, List
import ray
import pandas as pd
from mainsequence.tdag.time_series.time_series import DependencyUpdateError, TimeSerie
import gc
from mainsequence.logconf import  logger

from contextlib import contextmanager
from mainsequence.tdag_client import LocalDiskSourceLake, DynamicTableDataSource




USE_PICKLE = os.environ.get('TDAG_USE_PICKLE', True)
USE_PICKLE = USE_PICKLE in ["True", "true", True]





class TimeSerieHeadUpdateActor:
    TRACE_ID = "NO_TRACE"

    def __init__(self, local_hash_id: str, data_source_id: int, scheduler: Scheduler,  debug,
                 update_tree, update_extra_kwargs, remote_table_hashed_name: str,

                 ):
        """

        Parameters
        ----------
        hash_id :
        scheduler :
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
            ts, pickle_path=rebuild_and_set_from_local_hash_id(local_hash_id=local_hash_id,data_source_id=data_source_id,
                                                               graph_depth_limit=0)

        ts.set_actor_manager(actor_manager=distributed_actor_manager)

        return ts

    def run_one_step_update(self, force_update=False,
                            update_only_tree=False, ):
        """
        Main update Method for a time serie Head
        Returns
        -------

        """
        from mainsequence.tdag.instrumentation import TracerInstrumentator
        from mainsequence.tdag.config import configuration
        from .utils import UpdateInterface
        error_on_update = False


        try:
            self.ts.run(debug_mode=self.debug,update_tree=self.update_tree,update_only_tree=update_only_tree,
                        force_update=force_update,remote_scheduler=self.scheduler

                        )
        except Exception as e:

            error_on_update = True
        return error_on_update

@ray.remote(num_cpus=1, )
class TimeSerieHeadUpdateActorDist(TimeSerieHeadUpdateActor):
    ...


@contextmanager
def set_data_source(pod_source=None, tdag_detached=False, override_all: bool = False):
    """

    :param override_all:
    :return:
    """
    if pod_source is not None:

        vars = ["POD_DEFAULT_DATA_SOURCE", "POD_DEFAULT_DATA_SOURCE_FORCE_OVERRIDE", "BACKEND_DETACHED"]
        original_values = {k: os.environ.get(k, None) for k in vars}

        # Override the environment variables
        os.environ["POD_DEFAULT_DATA_SOURCE"] = pod_source.model_dump_json()
        os.environ["POD_DEFAULT_DATA_SOURCE_FORCE_OVERRIDE"] = str(override_all)
        os.environ["BACKEND_DETACHED"] = str(tdag_detached)

        try:
            yield pod_source
        except Exception as e:

            raise
        finally:
            # Restore the original environment variables
            if pod_source is None: return
            for key, value in original_values.items():
                if value is None:  # Variable was not originally set
                    del os.environ[key]
                else:
                    os.environ[key] = value
    else:
        # default data source for pod
        try:
            yield DynamicTableDataSource.get_default_data_source_for_token()
        except Exception as e:
            raise




class SchedulerUpdater:
    ACTOR_WAIT_LIMIT = 80  # seconds to wait before sending task to get scheduled

    @classmethod
    def debug_schedule_ts(cls, time_serie_hash_id: str,
                          data_source_id: int,
                          debug: bool, update_tree: bool,
                          raise_exception_on_error: bool = True,
                          break_after_one_update=True, force_update=False, run_head_in_main_process=False,
                          update_extra_kwargs=None, update_only_tree=False, name_suffix=None,

                          ):

        # get_remote_hash_id

        new_scheduler = Scheduler.initialize_debug_for_ts(local_hash_id=time_serie_hash_id,
                                                          data_source_id=data_source_id,
                                                          name_suffix=name_suffix,

                                                          )


        try:

            updater = cls(scheduler=new_scheduler)
            updater.logger.info(f"Scheduler ID {new_scheduler.name}")
            updater.start(debug=debug, update_tree=update_tree, break_after_one_update=break_after_one_update,
                          run_head_in_main_process=run_head_in_main_process,
                          raise_exception_on_error=raise_exception_on_error,
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
        global logger
        self.node_scheduler = scheduler
        logger=logger.bind(   scheduler_name=scheduler.name)
        self.logger = logger







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
            )
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
                                             debug: bool, update_extra_kwargs: dict,
                                             running_distributed_heads: bool, first_launch: bool,
                                             ):
        """
        Queries ts_orm scheduler and build update actors if not exist
        Returns
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
                if isinstance(actors_map[active_uid]['actor_handle'], ClientActorHandle):
                    ray.kill(actors_map[active_uid]['actor_handle'])

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

                                      debug=debug, update_tree=update_tree,
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
            _, nu = get_time_to_wait_from_hash_id(local_hash_id=head_ts.hash_id,data_source_id=head_ts.data_source_id)
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
                _, next_update = get_time_to_wait_from_hash_id(
                    local_hash_id=local_hash_id,data_source_id=data_source_id)
                error_in_request = False
            except Exception as e:
                self.logger.exception("Error getting wait time")
                time.sleep(60)
        return next_update

    def refresh_node_scheduler(self):
        from mainsequence.tdag_client.models import BACKEND_DETACHED
        if BACKEND_DETACHED()==True:
            return None
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



    def start(self, debug=False, update_tree: Union[bool, dict] = True, break_after_one_update=False,
              raise_exception_on_error=False,
              update_extra_kwargs: Union[None, dict] = None, run_head_in_main_process=False, force_update=False,
              sequential_update=False, update_only_tree=False, api_port: Union[int, None] = None,

              ):
        """
            Parameters
            ----------
            debug : bool, optional
                If True, all dependencies of a time series run in the same process.
                Defaults to False.
            update_tree : bool or dict, optional
                If True, updates the tree of dependent tasks.
            break_after_one_update : bool, optional
                If True, the process stops after the first update cycle. Defaults to False.
            raise_exception_on_error : bool, optional
                If True, raises an exception on encountering an error during execution.
                Otherwise, errors are handled silently. Defaults to False.
            update_extra_kwargs : dict or None, optional
                Additional parameters (if any) to pass along when updating. Defaults to None.
            run_head_in_main_process : bool, optional
                If True, each "head" time series is run in the main scheduler process. Useful for debugging.
                Defaults to False.
            force_update : bool, optional
                If True, forces an update run even if it's not required. Defaults to False.
            sequential_update : bool, optional
                If True, runs each "head" time series one by one instead of in parallel.
                Defaults to False.
            update_only_tree : bool, optional
                If True, only the dependency tree is updated without fully processing
                every step. Defaults to False.
            api_port : int or None, optional
                The port on which any exposed APIs should run. If None, no API is exposed.
                Defaults to None.
        """

        from mainsequence.tdag_client import CONSTANTS as TDAG_CONSTANTS
        self._debug_mode = debug
        self._api_port = api_port

        # ---------------------------------------------------------

        self.node_scheduler.start_heart_beat()

        update_extra_kwargs = {} if update_extra_kwargs is None else update_extra_kwargs
        if debug ==False:
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

                if self.node_scheduler.updates_halted == True:
                    self.logger.info("Scheduler Updates have been halted")
                    time.sleep(60)
                    continue

                task_hex_to_uid, wait_list = self._evaluate_read_task(ready=ready,
                                                                      task_hex_to_uid=task_hex_to_uid,
                                                                      wait_list=wait_list
                )

                # requery DB looking for new TS added to scheduler
                actors_map, target_ts, wait_list = self._build_scheduler_actors_if_not_exist(
                    actors_map=actors_map, update_tree=update_tree,
                    debug=debug,
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
                ts_updating = [c.hash_id for c in target_ts if c.uid not in wait_list.keys()]

                self.logger.info(f"""{bcolors.OKBLUE}Waiting for tasks to finish in scheduler {self.node_scheduler.uid}"
                                     timeseries updating {ts_updating}
                                     timeseries waiting:
                                     {pd.Series(wait_list).to_frame('next_update')} 
                                     {bcolors.ENDC}
                                     """
                                 )

                if len(ts_updating) == 0:
                    time.sleep(30)

                ready, unready = ray.wait([actor_details["task_handle"] for actor_details in actors_map.values()
                                           if actor_details["task_handle"] is not None
                                           ], timeout=60,
                                          num_returns=1)
                unready = True if len(unready) == 0 else unready


        except KeyboardInterrupt as ki:
            self.node_scheduler.patch(is_running=False, running_process_pid=0, )
            self._clear_idle_scheduled_tree(node_scheduler=self.node_scheduler)
            self.node_scheduler.stop_heart_beat()
            raise KeyboardInterrupt
        except Exception as e:
            self.node_scheduler.patch(is_running=False, running_process_pid=0, )
            self.logger.exception(e)
            self.node_scheduler.stop_heart_beat()
            self._clear_idle_scheduled_tree(node_scheduler=self.node_scheduler)
            time.sleep(60)

            if raise_exception_on_error == True:
                raise e
        self.node_scheduler.stop_heart_beat()

        self.logger.info("Scheduler is stopping")