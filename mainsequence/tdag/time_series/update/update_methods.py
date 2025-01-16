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

        ts, pickle_path = TimeSerie.rebuild_and_set_from_local_hash_id(
            local_hash_id=local_hash_id,graph_depth_limit=0,
            data_source_id=data_source_id,
        )
        ts.logger.info(f'{ts.local_hash_id} {data_source_id} loaded and ready to update')

        try:
            assert execution_start is not None
            ts_updater = TimeSerieUpdater(time_serie=ts, update_tree=False,
                                  execution_start=execution_start,
                                  start_update_data=start_update_data
                                  )
            ts_updater.do_step_update()
        except Exception as e:
            ts.logger.exception(f'{e} ')
        
        
        
        

    if psutil.virtual_memory().percent >= .75:
        gc.collect()


@ray.remote(max_calls=20)  # max calls helps controlls how many task are lanuched on one actor ot do not press memory
def update_remote_from_hash_id(*args,**kwargs  ):
    """
    Ray wrapper for session update
    :param args:
    :param kwargs:
    :return:
    """
    local_hash_id=update_remote_from_hash_id_local(*args,**kwargs)

    return local_hash_id


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
            return True,False
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

   








