import socket
from typing import Union
from mainsequence.mainsequence_client import LocalTimeSerie, LocalTimeSeriesHistoricalUpdate
import datetime
import pytz
import logging
from mainsequence.logconf import logger
import time
import pandas as pd

def wait_for_update_time(local_hash_id, data_source_id, logger, force_next_start_of_minute=False):
    time_to_wait, next_update = get_time_to_wait_from_hash_id(local_hash_id=local_hash_id,
                                                                               data_source_id=data_source_id)
    if time_to_wait > 0 and force_next_start_of_minute == False:

        logger.info(f"Scheduler Waiting for ts update time at {next_update} {time_to_wait}")
        time.sleep(time_to_wait)
    else:
        time_to_wait = max(0, 60 - datetime.datetime.now(pytz.utc).second)
        logger.info(f"Scheduler Waiting for ts update at start of minute")
        time.sleep(time_to_wait)
    if force_next_start_of_minute == True:
        logger.info(f"Forcing Next Udpdate at start of minte")

def get_node_time_to_wait(local_metadata):

    next_update = local_metadata.localtimeserieupdatedetails.next_update
    time_to_wait = 0.0
    if next_update is not None:
        time_to_wait = (pd.to_datetime(next_update) - datetime.datetime.now(pytz.utc)).total_seconds()
        time_to_wait = max(0, time_to_wait)
    return time_to_wait, next_update
def get_time_to_wait_from_hash_id(local_hash_id: str,data_source_id:int):
    
    local_metadata = LocalTimeSerie.get_or_none(local_hash_id=local_hash_id,
                                              data_source_id=data_source_id
                                              )
    time_to_wait, next_update = get_node_time_to_wait(local_metadata=local_metadata)

    
    return time_to_wait, next_update

class UpdateInterface:
    """
    Helper class to avoid calling ray in other modules
    """

    def __init__(self, trace_id: Union[str, None], head_hash: Union[str, None],
                 logger: logging.Logger,scheduler_uid:str,
                 state_data: Union[str, None], debug=False):
        self.debug = debug
        self.trace_id = trace_id
        self.head_hash = head_hash
        self.updating_in_tree = []
        self.logger=logger



        self.state_data = state_data
        self.scheduler_uid=scheduler_uid
        self.last_historical_update={}

    def _patch_update(self, hash_id,data_source_id:int, update_kwargs: dict):
        update_kwargs["hash_id"] = hash_id
        update_kwargs["data_source_id"] = data_source_id
        update_details = self.dth.patch_local_update_details(**update_kwargs)

    def _sanitize_state_date(self, state_data: dict):
        """
        Calls API with latest state datte for timeseries
        Parameters
        ----------
        state_data :

        Returns
        -------

        """


        dth = DynamicTableHelpers()
        new_state_data = {}
        for key, value in state_data.items():
            new_state_data[key] = value
            if new_state_data[key]['last_time_index_value'] is not None:
                new_state_data[key]['last_time_index_value'] = dth.request_to_datetime(
                    new_state_data[key]['last_time_index_value'])
            if new_state_data[key]['next_update'] is not None:
                new_state_data[key]['next_update'] = dth.request_to_datetime(
                    new_state_data[key]['next_update'])
        return new_state_data

    def _assert_next_update(self, local_hash_id: str,data_source_id:int):

        execution_time = datetime.datetime.utcnow().replace(tzinfo=pytz.utc)
        if self.state_data[(local_hash_id,data_source_id)]["last_time_index_value"] is None:
            return True

        next_update = self.state_data[(local_hash_id,data_source_id)]["next_update"]
        error_on_last_update = self.state_data[(local_hash_id,data_source_id)]["error_on_last_update"]
        if next_update <= execution_time or error_on_last_update is True:
            must_update = True

        else:
            must_update = False
        return must_update

    def get_ts_in_update_queue(self, hash_id_list: list):
        update_queue = []

        return update_queue





    def set_end_of_execution(self, local_time_serie_id: str,
                             error_on_update=False,):

        TimeSerieLocalUpdate.set_end_of_execution(local_time_serie_id=local_time_serie_id,
                                                  historical_update_id=self.last_historical_update[local_time_serie_id].id,
                                                  error_on_update=error_on_update)





def start_scheduler_api(scheduler_uid: int,
                    scheduler_kwargs: Union[dict,None]=None,port: Union[int, None]=None,
                    host="0.0.0.0", reload=False):
    import uvicorn
    from .api import app
    from mainsequence.tdag.time_series.update.utils import is_port_free

    port =port if port is not None else get_available_port()
    # set state parameters
    app.state.scheduler_uid = scheduler_uid
    app.state.host = host
    app.state.port = port
    app.state.scheduler_kwargs=scheduler_kwargs

    assert is_port_free(port=port)
    uvicorn.run(app, host=host, port=port, reload=reload)



def is_port_free(port: int) -> bool:
    """Check if the port is free on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('', port))  # Bind to all interfaces on the specified port
            return True  # If bind succeeds, the port is free
        except OSError:
            return False  # If bind fails, the port is in use

def get_available_port(port_range: tuple[int, int]=(8000,8090)) -> int:
    """Check if the given port is free, and if not, find an available port within the range."""

    for p in range(port_range[0], port_range[1] + 1):
        if is_port_free(p):
            logger.info(f"Using port {p}.")
            return p
    raise RuntimeError(f"Could not find a free port in the range {port_range}.")




