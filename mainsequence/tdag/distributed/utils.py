
from mainsequence.instrumentation import tracer_instrumentator, tracer
from mainsequence.tdag import ogm
from mainsequence.logconf import logger
import json
import datetime
import requests




def build_ts_updater_serve_request(pickle_path:str,update_priority:int,scheduler_id:int,
                 in_update_tree_node_uid:int,update_tree_kwargs:dict,execution_start:datetime.datetime,
                      ):
    # make request
    telemetry_carrier = tracer_instrumentator.get_telemetry_carrier()
    telemetry_carrier = str(json.dumps(telemetry_carrier)).replace(" ", "")

    query = dict(scheduler_id=scheduler_id, telemetry_carrier=telemetry_carrier,
                 execution_start=execution_start.timestamp(),
                 in_update_tree_node_uid=in_update_tree_node_uid,
                 update_tree_kwargs=json.dumps(update_tree_kwargs),
                 update_priority=update_priority)

    query["pickle_path"] = pickle_path

    return query

def get_request_status_from_query(query:dict,request_url):
    """

    :param query:
    :type query:
    :return:
    :rtype:
    """
    try:
        status = requests.get(request_url, params=query)

        status = status.json()
    except Exception as e:
        logger.error(e)
        raise e
    if status["error"] == 1:
        if status["error_type"] == str(TimeoutError):
            raise TimeoutError
        else:
            raise Exception(status["error_message"])
    return status

@tracer.start_as_current_span("Distributed: predict_from_serve url")
def do_remote_update_by_hash_id(hash_id: str, update_priority: int, scheduler_id: str,
                                in_update_tree_node_uid: int, update_tree_kwargs: dict,
                                execution_start: datetime.datetime):
    pickle_path = ogm.get_ts_pickle_path(hash_id=hash_id)
    query = build_ts_updater_serve_request(pickle_path=pickle_path, update_priority=update_priority,
                                           scheduler_id=scheduler_id,
                                           in_update_tree_node_uid=in_update_tree_node_uid,
                                           update_tree_kwargs=update_tree_kwargs
                                           , execution_start=execution_start)
    status = get_request_status_from_query(query=query,request_url=TS_UPDATER_URL)
    return status