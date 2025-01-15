import os
import datetime
import pytz
import time

from mainsequence.tdag_client import LocalTimeSerieUpdateDetails,request_to_datetime
from mainsequence.tdag.config import bcolors, configuration
from typing import Union
from mainsequence.tdag.instrumentation import tracer
import ray
from mainsequence.tdag_client import CONSTANTS

from mainsequence.tdag.time_series import TimeSerie
from .models import StartUpdateDataInfo
from concurrent.futures import ThreadPoolExecutor
from mainsequence.tdag.logconf import get_tdag_logger

logger = get_tdag_logger()
EXECUTION_HEAD_WEIGHT = float(os.environ.get('EXECUTION_HEAD_WEIGHT', 60 * 60 * 8))



def update_remote_from_hash_id_local(
                               execution_start: datetime.datetime, telemetry_carrier: str,
                               start_update_data: dict,
                               local_hash_id: str,data_source_id:int,
                               local_metadatas: Union[dict, None],

                               ):
    """

    Args:
        in_update_tree_node_uid ():
        update_tree_kwargs ():
        execution_start ():
        telemtry_carrier ():
        update_priority ():
        hash_id ():


    Returns:

    """
    from mainsequence.tdag.time_series.update.update_methods import rebuild_with_session
    from mainsequence.tdag.instrumentation import SpanKind, TraceContextTextMapPropagator, TracerInstrumentator
    import psutil
    import gc

    tracer_instrumentator = TracerInstrumentator(configuration.configuration["instrumentation_config"]["grafana_agent_host"])
    tracer = tracer_instrumentator.build_tracer("tdag", __name__)

    prop = TraceContextTextMapPropagator()
    ctx = prop.extract(carrier=telemetry_carrier)

    with tracer.start_as_current_span(f"Distributed Task : Main Time Serie Update", context=ctx,
                                      kind=SpanKind(4)) as span:
        span.set_attribute("time_serie_hash_id", local_hash_id)

        _, _ = rebuild_with_session(
            local_hash_id=local_hash_id,
            update_tree=False,
            execution_start=execution_start,
            update=True,
            data_source_id=data_source_id,
            start_update_data=start_update_data,
            local_metadatas=local_metadatas,

        )

    if psutil.virtual_memory().percent >= .75:
        gc.collect()


@ray.remote(max_calls=20)  # max calls helps controlls how many task are lanuched on one actor ot do not press memory
def update_remote_from_hash_id(*args,**kwargs  ):
    local_hash_id=update_remote_from_hash_id_local(*args,**kwargs)

    return local_hash_id





@tracer.start_as_current_span("Update with session")
def update_with_session(session, time_serie, update_tree,
                        ):
    logger.info(f'{time_serie} loaded and ready to update')

    time_serie.update( raise_exceptions=True, update_tree=update_tree,
                      assets_db=session.tdag_orm_db_name)



class TimeSerieUpdater:

    def __init__(self,time_serie: TimeSerie, start_update_data: StartUpdateDataInfo, update_tree: bool,
                 execution_start: datetime.datetime, ):
        self.time_serie=time_serie
        self.start_update_data=start_update_data
        self.update_tree=update_tree
        self.execution_start=execution_start





    def check_if_dependencies_are_updated(self,time_out_of_request:int):
        """

        Args:
            dependencies_update_details:

        Returns:dependencies_updates,raise_error_from_childs

        """
        if len(self.start_update_data.direct_dependencies_ids) == 0:
            return None
        # Filter and use timeout
        dependencies_update_details = LocalTimeSerieUpdate.verify_if_direct_dependencies_are_updated(
            id=time_serie.local_metadata["id"],time_out=time_out_of_request,
        )


        return dependencies_update_details["updated"], dependencies_update_details["error_on_update_dependencies"]


    def do_step_update(self)->bool:



        from mainsequence.tdag import configuration
        from .utils import UpdateInterface
        import logging


        ts_update_details = self.time_serie.update_details
        ignore_timeout = configuration.configuration["time_series_config"]["ignore_update_timeout"]
        execution_timeout_seconds = ts_update_details['execution_timeout_seconds'] if ts_update_details[
                                                                                          'execution_timeout_seconds'] > 0 else EXECUTION_HEAD_WEIGHT

        time_out_of_request=min(execution_timeout_seconds,60*5)
        are_dependencies_updated,error_on_dependencies=self.check_if_dependencies_are_updated(time_out_of_request)


        update_tracker = UpdateInterface(head_hash=None, trace_id=None,
                                         logger=self.time_serie.logger,
                                         state_data=None, debug=False)

        self.time_serie._run_pre_load_routines()
        with tracer.start_as_current_span("Waiting for depencencies update") as waiting_span:

            while are_dependencies_updated == False:
                # verify_scheduler is not stale

                time.sleep(.1)

                execution_timeout_seconds = ts_update_details['execution_timeout_seconds'] if ts_update_details[
                                                                                                  'execution_timeout_seconds'] > 0 else EXECUTION_HEAD_WEIGHT
                passed_seconds = (datetime.datetime.utcnow().replace(tzinfo=pytz.utc) - self.execution_start).total_seconds()
                if  self.time_serie.source_table_configuration is None or ignore_timeout == True:
                    execution_timeout_seconds = passed_seconds + 10
                if passed_seconds > execution_timeout_seconds:
                    next_rebalances =  self.time_serie.next_rebalances if hasattr( self.time_serie, "next_rebalances") else None
                    error_message = f"{bcolors.OKCYAN}Head time Serie timeout exceeded {execution_timeout_seconds} pending: {next_rebalances}{bcolors.ENDC}"
                    self.time_serie.logger.error(error_message)

                    update_tracker.set_end_of_execution(local_hash_id= self.time_serie.hashed_name, error_on_update=True,
                                                        data_source_id=self.time_serie.data_source.id,
                                                        error_message=error_message
                                                        )
                    logging.shutdown()
                    self.should_stop = True

                if error_on_dependencies == True:  # or active_updates==0:
                    error_message = f"{bcolors.OKCYAN}Error on Dependencies Update not started{bcolors.ENDC}"
                    self.time_serie.logger.error(error_message)
                    update_tracker.set_end_of_execution(local_hash_id= self.time_serie.hashed_name, error_on_update=True,
                                                        data_source_id=self.time_serie.data_source.id,
                                                        error_message=error_message
                                                        )
                    logging.shutdown()
                    self.should_stop = True

                are_dependencies_updates, error_on_dependencies = self._check_if_dependencies_are_updated()

        self.time_serie.update( raise_exceptions=True, update_tree=self.update_tree,
                          start_update_data=self.start_update_data, update_tracker=update_tracker,

                          )
        self.should_stop = True

    def get_consumers(self, Consumer, channel):

        exchange = Exchange(CONSTANTS.TDAG_EXCHANGE_UPDATE, type='direct')
        queue_position = Queue(f"time_serie_{self.time_serie.hashed_name}", exchange)
        consumers= [
            Consumer(queues=[queue_position],
                     accept=['pickle', 'json'],
                     callbacks=[self.consume_time_series_update_message]),
        ]
        channel.basic_qos(prefetch_size=0, prefetch_count=1, a_global=False)
        return consumers
    def consume_time_series_update_message(self, body: dict, message):
        """

        Args:
            body:
            message:

        Returns:

        """

        message.ack()

        if body["related_table__local_hash_id"] not in self.dependencies_status:
            return None
        target_ts=self.dependencies_status[body["related_table__local_hash_id"]]
        if  request_to_datetime(body["last_update"])>request_to_datetime(target_ts['last_update']):
            self.dependencies_status[body["related_table__local_hash_id"]]=body
            self.time_serie.logger.debug(f'dependency {body["related_table__local_hash_id"]} updated')




@tracer.start_as_current_span(" get_or_pickle_ts_from_sessions")
def get_or_pickle_ts_from_sessions(local_hash_id: str,data_source_id:int,
                                   set_dependencies_df=False,
                                   ts: Union[object, None] = None,
                                   return_ts=False
                                   ):
    from mainsequence.tdag.time_series.time_series import TimeSerie
    from mainsequence.tdag import ogm

    pickle_path = ogm.get_ts_pickle_path(local_hash_id=local_hash_id)
    if os.path.isfile(pickle_path) == False or os.stat(pickle_path).st_size == 0:
        # rebuild time serie and pickle
        if ts is None:
            ts = TimeSerie.rebuild_from_configuration(local_hash_id=local_hash_id,
                                                      data_source=data_source_id,
                                                      )
        if set_dependencies_df == True:
            ts.set_relation_tree()

        full_path, relative_path = ts.persist_to_pickle()
        ts.logger.info(f"ts {local_hash_id} pickled ")
    if return_ts == True:
        relative_path = pickle_path.replace(ogm.pickle_storage_path + "/", "")
        if ts is None:
            ts = TimeSerie.load_and_set_from_pickle(pickle_path=pickle_path)

        return relative_path, ts

    return local_hash_id


@tracer.start_as_current_span(" get_pickled_ts_path")
def get_or_pickle_ts(hash_id: str, update_priority: int, scheduler_uid: str,
                     in_update_tree_node_uid: int,
                     ):
    from mainsequence.tdag import ogm
    pickle_path = ogm.get_ts_pickle_path(hash_id=hash_id)
    if os.path.isfile(pickle_path) == False:
        ts, pickle_path = rebuild_with_session(local_hash_id=hash_id,
                                               update_tree=False,  update=False,
                                               scheduler_uid=scheduler_uid, execution_start=None,
                                               update_priority=update_priority,
                                               in_update_tree_node_uid=in_update_tree_node_uid)

        a = 5

    return pickle_path


@tracer.start_as_current_span("Rebuild with session")
def rebuild_with_session(local_hash_id:str,data_source_id:int,
                         update_tree:bool, execution_start: Union[datetime.datetime, None],

                         update=True,
                         local_metadatas: Union[dict, None] = None,
                         start_update_data: Union[dict, None] = None,
                         ):
    """
     Rebuild and updates time serie in a

    :param hash_id:
    :param update_tree:
    :param update:
    :param scheduler_uid:
    :return:
    """

    from mainsequence.tdag.time_series.time_series import TimeSerie
    # from mainsequence.tdag_client import DynamicTableHelpers

    if start_update_data is None:
        raise NotImplementedError

    USE_PICKLE = True
    if USE_PICKLE == False:
        ts = TimeSerie.rebuild_from_configuration(local_hash_id=local_hash_id,
                                                  data_source=data_source_id

                                                  )
        pickle_path = None
    else:
        _ = get_or_pickle_ts_from_sessions(local_hash_id=local_hash_id,
                                           data_source_id=data_source_id,
                                           )
        pickle_path = TimeSerie.get_pickle_path(local_hash_id=local_hash_id,
                                                data_source_id=data_source_id,
                                                )
        ts = TimeSerie.load_from_pickle(pickle_path=pickle_path)
        ts.set_state_with_sessions(include_vam_client_objects=False,
                                   graph_depth_limit=0.0,
                                   graph_depth=0.0,
                                   local_metadatas=local_metadatas,
                                   )

    ts.logger.info(f'{ts.local_hash_id} {data_source_id} loaded and ready to update')
    # graph_node = ts.graph_node
    # graph_node.set_update_pid(process_pid)

    if update == True:

        try:
            assert execution_start is not None
            ts = TimeSerieUpdater(time_serie=ts, update_tree=update_tree,
                                execution_start=execution_start,
                                start_update_data=start_update_data
                                )
            ts.do_step_update()
        except Exception as e:
            ts.logger.exception(f'{e} ')

    return ts, pickle_path


@tracer.start_as_current_span("Rebuild and update from source")
def rebuild_and_update_from_source(local_hash_id,
                                   execution_start: datetime.datetime,
                                   update_tree=False,
                                   in_update_tree_node_uid=None,

                                   ):
    """
    rebuilds a time series from configuration and updates it
    :param hash_id:
    :param update_tree:
    :return:
    """

    ts = rebuild_with_session(local_hash_id=local_hash_id,
                              in_update_tree_node_uid=in_update_tree_node_uid,

                              execution_start=execution_start,
                              update_tree=update_tree, )

    return ts




def persist_relationships(local_hash_id: str):
    """

    Parameters
    ----------
    time_serie_hash_id :

    Returns
    -------

    """
    from mainsequence.tdag.time_series import TimeSerie
    from mainsequence.tdag_client import TimeSerieLocalUpdate

    local_ts = TimeSerieLocalUpdate.get(local_hash_id=local_hash_id)
    _ = get_or_pickle_ts_from_sessions(local_hash_id=local_hash_id,
                                       remote_table_hashed_name=local_ts["remote_table"]['hash_id'],
                                       set_dependencies_df=True
                                       )
    pickle_path = TimeSerie.get_pickle_path(local_hash_id)
    ts = TimeSerie.load_from_pickle(pickle_path=pickle_path)
    ts.local_persist_manager.synchronize_metadata(meta_data=None, local_metadata=None)
    ts.set_relation_tree()

