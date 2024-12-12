
from .utils import (TDAG_ENDPOINT, read_sql_tmpfile, direct_table_update,  is_process_running,get_network_ip,
CONSTANTS,
    DATE_FORMAT, get_authorization_headers, AuthLoaders, make_request, get_tdag_client_logger)
import copy
import datetime
import pytz
import requests
import pandas as pd
import numpy as np
import json
from typing import Union
import structlog
import time



from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from typing import Optional, List, Dict, Any
_default_data_source = None  # Module-level cache


JSON_COMPRESSED_PREFIX = ["json_compressed", "jcomp_"]



logger = get_tdag_client_logger()

loaders=AuthLoaders()



class AlreadyExist(Exception):
    pass



if "dth_ws" not in locals():
    DTH_WS = None



def request_to_datetime(string_date: str):
    if "+" in string_date:
        string_date = datetime.datetime.fromisoformat(string_date.replace("T", " ")).replace(tzinfo=pytz.utc)
        return string_date
    try:
        date = datetime.datetime.strptime(string_date, "%Y-%m-%dT%H:%M:%S.%fZ").replace(
            tzinfo=pytz.utc)
    except ValueError:
        date = datetime.datetime.strptime(string_date, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=pytz.utc)
    return date


def serialize_to_json(kwargs):
    new_data = {}
    for key, value in kwargs.items():
        new_value = copy.deepcopy(value)
        if isinstance(value, datetime.datetime):
            new_value = str(value)

        new_data[key] = new_value
    return new_data

get_ts_node_url=lambda root_url:  root_url + "/ogm/api/time_serie"
get_scheduler_node_url=lambda root_url:  root_url + "/ogm/api/scheduler"
get_multi_index_node_url=lambda root_url:  root_url + "/orm/api/multi_index_metadata"
get_continuous_agg_multi_index=lambda root_url: root_url+ "/orm/api/cont_agg_multi_ind"
get_dynamic_table_metadata=lambda root_url:  root_url + "/orm/api/dynamic_table"
get_time_serie_local_update_url=lambda root_url:  root_url + "/ogm/api/local_time_serie"
get_time_serie_local_update_table_url=lambda root_url:  root_url + "/orm/api/time_serie_local_update"
get_local_time_serie_update_details=lambda root_url: root_url + "/orm/api/local_update_details"
get_local_time_serie_historical_update_url=lambda root_url:  root_url + "/orm/api/lts_historical_update"
get_dynamic_table_data_source=lambda root_url: root_url + "/orm/api/dynamic_table_data_source"
get_chat_yaml_url=lambda root_url:  root_url + "/tdag-gpt/api/chat_yaml"
get_signal_yaml_url=lambda root_url:  root_url + "/tdag-gpt/api/signal_yaml"
get_chat_object_url=lambda root_url: root_url + "/tdag-gpt/api/chat_object"

class BaseTdagPydanticModel(BaseModel):
    tdag_orm_class: str = None  # This will be set to the class that inherits

    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Set orm_class to the class itself
        cls.tdag_orm_class = cls.__name__


class BaseObject:
    LOADERS = loaders
    @classmethod
    def build_session(cls):
        from requests.adapters import HTTPAdapter, Retry
        s = requests.Session()
        s.headers.update(cls.LOADERS.auth_headers)
        retries = Retry(total=2, backoff_factor=2, )
        s.mount('http://', HTTPAdapter(max_retries=retries))
        return s

    @property
    def s(self):
        s = self.build_session()
        return s



    @classmethod
    def create(cls, *args, **kwargs):
        url = cls.ROOT_URL + "/"
        payload = {"json": serialize_to_json(kwargs)}
        s=cls.build_session()
        r = make_request(s=s,loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        instance=cls(**r.json())
        return instance

    def delete(self):
        url = self.ROOT_URL + "/destroy/"
        payload = {"json":{"uid":self.uid}}
        s = self.build_session()
        r = make_request(s=s,loaders=self.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    def patch(self,*args,**kwargs):
        url = self.ROOT_URL + "/update/"
        payload = {"json": {"uid": self.uid,"patch_data":serialize_to_json(kwargs)}}
        s = self.build_session()
        r = make_request(s=s,loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")


class SchedulerDoesNotExist(Exception):
    pass
class LocalTimeSeriesDoesNotExist(Exception):
    pass
class DynamicTableDoesNotExist(Exception):
    pass
class SourceTableConfigurationDoesNotExist(Exception):
    pass


class TimeSerieNode(BaseTdagPydanticModel,BaseObject):
    uid: str
    hash_id: str
    data_source_id: int
    source_class_name: str
    human_readable: str
    creation_date: datetime.datetime
    relation_tree_frozen: bool



    @classmethod
    @property
    def ROOT_URL(cls):
        return get_ts_node_url(TDAG_ENDPOINT)

    @classmethod
    def get_all_dependencies(cls, hash_id):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/get_all_dependencies"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, )
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df

    @classmethod
    def delete_with_relationships(cls, *args, **kwargs):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/delete_with_relationships/"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload={"json": kwargs})
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        return r.json()

    @classmethod
    def get_max_depth(cls, hash_id, timeout=None):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/get_max_depth"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r.json()["max_depth"]

    @classmethod
    def get_upstream_nodes(cls, hash_id):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/get_upstream_nodes"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, )
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df

    @classmethod
    def depends_on_connect(cls, source_hash_id: str, target_hash_id: str, target_class_name: str,
                           source_local_hash_id: str,
                           target_local_hash_id: str,
                           source_data_source_id: id,
                           target_data_source_id: id,
                           target_human_readable: str):
        """
        Connects and build relationship
        Parameters
        ----------
        source_hash_id :
        target_hash_id :
        target_class_name :
        target_human_readable :

        Returns
        -------

        """
        s = cls.build_session()
        url = cls.ROOT_URL + "/depends_on_connect/"
        payload = dict(json={"source_hash_id": source_hash_id,
                             "target_hash_id": target_hash_id, "target_class_name": target_class_name,
                             "source_local_hash_id": source_local_hash_id, "target_local_hash_id": target_local_hash_id,
                             "target_human_readable": target_human_readable,
                             "source_data_source_id": source_data_source_id,
                             "target_data_source_id": target_data_source_id,
                             })
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")

    @classmethod
    def set_policy_for_descendants(cls, hash_id, policy, pol_type, exclude_ids, extend_to_classes):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/set_policy_for_descendants/"
        payload = dict(json={"policy": policy,
                             "pol_type": pol_type,
                             "exclude_ids": exclude_ids,
                             "extend_to_classes": extend_to_classes,
                             })
        r = make_request(s=s, loaders=cls.LOADERS, r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    @classmethod
    def create(cls, *args, **kwargs):
        url = cls.ROOT_URL + "/"
        payload = {"json": kwargs}
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            if r.status_code == 409:
                raise AlreadyExist(r.text)
            else:
                raise Exception(r.text)
        data = r.json()
        instance, metadata = cls(**data["node"]), data["metadata"]
        return instance, metadata

    @classmethod
    def remove_head_from_all_schedulers(cls, hash_id):
        url = cls.ROOT_URL + f"/{hash_id}/remove_head_from_all_schedulers/"
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="PATCH", url=url, )
        if r.status_code != 200:
            raise Exception(r.text)

    @classmethod
    def patch_build_configuration(cls, remote_table_patch: dict,
                                  build_meta_data: dict, data_source_id: int,
                                  local_table_patch: dict) -> "TimeSerieLocalUpdate":
        """

        Args:
            remote_table_patch:
            local_table_patch:

        Returns:

        """

        url = cls.ROOT_URL + "/patch_build_configuration"
        payload = {"json": {"remote_table_patch": remote_table_patch, "local_table_patch": local_table_patch,
                            "build_meta_data": build_meta_data, "data_source_id": data_source_id,
                            }}
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(r.text)
        return TimeSerieLocalUpdate(**r.json())
class LocalTimeSerieNode(BaseTdagPydanticModel,BaseObject):
    hash_id: str
    uid: str
    data_source_id: int
    updates_to: TimeSerieNode


class Scheduler(BaseTdagPydanticModel,BaseObject):
    uid: str
    name: str
    is_running: bool
    running_process_pid: Optional[int]
    running_in_debug_mode: bool
    host: Optional[str]
    api_address: Optional[str]
    api_port: Optional[int]
    pre_loads_in_tree: List[str]  # Assuming this is a list of strings
    in_active_tree: List[LocalTimeSerieNode]    # Assuming this is a list of strings
    schedules_to: List[LocalTimeSerieNode]


    @classmethod
    @property
    def ROOT_URL(cls):
        return  get_scheduler_node_url(TDAG_ENDPOINT)

    @classmethod
    def get(cls, *args,**kwargs):

        url = cls.ROOT_URL
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload={"params": kwargs})
        if r.status_code == 404:
            raise SchedulerDoesNotExist
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        results = r.json()
        if len(results) > 1:
            logger.warning(f"Warning: more than 1 scheduler returned for query with kwargs {kwargs}")
        instance = cls(**results[0])
        return instance

    @classmethod
    def filter(cls,payload:Union[dict,None]):
        url = cls.ROOT_URL
        payload={} if payload is None else {"params":payload}

        s = cls.build_session()
        r = make_request(s=s,loaders=cls.LOADERS, r_type="GET", url=url, payload=payload)
        if r.status_code ==404:
            raise SchedulerDoesNotExist
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        instance = [cls(**i.json()) for i in r.json()]
        return instance
    @classmethod
    def get_scheduler_for_ts(cls,hash_id:str):
        s=cls.build_session()
        url=cls.ROOT_URL + "/get_scheduler_for_ts"
        payload = dict(params={"hash_id":hash_id})
        r = make_request(s=s, loaders=cls.LOADERS,r_type="GET", url=url, payload=payload)
        if r.status_code ==404:
            raise SchedulerDoesNotExist(r.text)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        scheduler=cls(**r.json())
        return scheduler
    @classmethod
    def initialize_debug_for_ts(cls,local_hash_id:str,
                                data_source_id:int,
                                name_suffix:Union[str,None]=None):
        s = cls.build_session()
        url = cls.ROOT_URL + "/initialize_debug_for_ts/"
        payload = dict(json={"local_hash_id":local_hash_id,"name_suffix":name_suffix,
                             "data_source_id":data_source_id
                             })
        r = make_request(s=s, loaders=cls.LOADERS,r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        scheduler = cls(**r.json())
        return scheduler

    @classmethod
    def build_and_assign_to_ts(cls, scheduler_name: str, local_hash_id_list: list, delink_all_ts=False,
                               remove_from_other_schedulers=True):
        s = cls.build_session()
        url = cls.ROOT_URL + "/build_and_assign_to_ts/"
        payload = dict(json={
            "scheduler_name": scheduler_name,
            "delink_all_ts": delink_all_ts,
            "hash_id_list": local_hash_id_list,
            "remove_from_other_schedulers": remove_from_other_schedulers
        })
        r = make_request(s=s,loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        scheduler = cls(**r.json())
        return scheduler

    def in_active_tree_connect(self,hash_id_list:list):
        s = self.build_session()
        url = self.ROOT_URL + f"/{self.uid}/in_active_tree_connect/"
        payload = dict(json={"hash_id_list": hash_id_list})
        r = make_request(s=s, loaders=self.LOADERS,r_type="PATCH", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")

    def assign_to_scheduler(self,hash_id_list:list):
        s = self.build_session()
        url = self.ROOT_URL + f"/{self.uid}/assign_to_scheduler/"
        payload = dict(json={"hash_id_list": hash_id_list})
        r = make_request(s=s,loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        return Scheduler(**r.json())

    def is_scheduler_running_in_process(self):
        # test call
        if self.is_running == True and hasattr(self, "api_address"):
            # verify  scheduler host is the same
            if self.api_address == get_network_ip() and is_process_running(self.running_process_pid) == True:
                return True
        return False



class ContinuousAggregateMultiIndex(BaseObject):
    ROOT_URL = get_continuous_agg_multi_index(TDAG_ENDPOINT)

    def __init__(self, *args, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def get_or_create(cls, time_out:Union[None,float],*args, **kwargs):
        s = cls.build_session()
        url = cls.ROOT_URL + "/get_or_create/"
        payload = dict(json=kwargs)
        r = make_request(s=s,loaders=cls.LOADERS, r_type="POST", url=url, payload=payload,time_out=time_out)
        created = False
        if r.status_code not in [200, 201]:
            raise Exception(f"{r.text()}")
        mi_metadata = cls(**r.json())
        if r.status_code == 201:
            created = True

        return mi_metadata, created


    def patch(self,time_out, *args, **kwargs):
        url = self.ROOT_URL + f"/{self.id}/"
        payload = {"json": serialize_to_json(kwargs)}
        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload,time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
    
class MultiIndexTableMetaData(BaseObject):
    ROOT_URL=get_multi_index_node_url(TDAG_ENDPOINT)

    def __init__(self, *args, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)
    @classmethod
    def get_or_create(cls,time_out:Union[None,int]=None,*args,**kwargs,):
        s = cls.build_session()
        url = cls.ROOT_URL + "/get_or_create/"
        payload = dict(json=kwargs)
        r = make_request(s=s,loaders=cls.LOADERS, r_type="POST", url=url, payload=payload,time_out=time_out)
        created=False
        if r.status_code not in [200,201]:
            raise Exception(f"{r.text()}")
        mi_metadata = cls(**r.json())
        if r.status_code ==201:
            created=True
        
        return mi_metadata,created

    def patch(self,time_out:Union[None,int]=None, *args, **kwargs,):
        url = self.ROOT_URL + f"/{self.id}/"
        payload = {"json": serialize_to_json(kwargs)}
        s = self.build_session()
        r = make_request(s=s,loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload,time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")


class RunConfiguration(BaseTdagPydanticModel,BaseObject):

    local_time_serie_update_details_id:Optional[int]=None
    retry_on_error: int = 0
    seconds_wait_on_retry: float = 50
    required_cpus: int = 1
    required_gpus: int = 0
    execution_time_out_seconds: float = 50
    update_schedule: str = "*/1 * * * *"

    @classmethod
    @property
    def ROOT_URL(cls):
        return None

class  LocalTimeSerieUpdateDetails(BaseObject):
    ROOT_URL = get_local_time_serie_update_details(TDAG_ENDPOINT)

    @staticmethod
    def _parse_parameters_filter(parameters):

        for key, value in parameters.items():
            if "__in" in key:
                assert isinstance(value, list)
                parameters[key] = ",".join(value)
        return parameters
    
    @classmethod
    def filter(cls, *args,**kwargs):
        url = cls.ROOT_URL
        payload = {} if kwargs is None else {"params": cls._parse_parameters_filter(kwargs)}

        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload=payload)
        if r.status_code == 404:
            raise SchedulerDoesNotExist
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        instance = [i for i in r.json()]
        return instance



class TimeSerieLocalUpdate(BaseObject):
    ROOT_URL = get_time_serie_local_update_url(TDAG_ENDPOINT)
    LOCAL_UPDATE_URL=get_time_serie_local_update_table_url(TDAG_ENDPOINT)
    LOCAL_TIME_SERIE_HISTORICAL_UPDATE=get_local_time_serie_historical_update_url(TDAG_ENDPOINT)
    def __init__(self, *args, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def add_tags(cls, local_metadata,tags:list, timeout=None):

        base_url = cls.LOCAL_UPDATE_URL
        s = cls.build_session()
        payload = {"json": {"tags":tags}}
        # r = self.s.get(, )
        url = f"{base_url}/{local_metadata['id']}/add_tags/"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="PATCH", url=url,
                         payload=payload, 
                         time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.json()}")
        return r.json()
    
    @classmethod
    def update_details_exist(cls, local_metadata,timeout=None):

        base_url = cls.LOCAL_UPDATE_URL
        s = cls.build_session()

        # r = self.s.get(, )
        url = f"{base_url}/{local_metadata['id']}/update_details_exist/"
        r = make_request(s=s, loaders=cls.LOADERS,r_type="GET", url=url,time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.json()}")
        return r.json()

    @classmethod
    def filter_by_hash_id(cls, local_hash_id_list: list,timeout=None):
        s = cls.build_session()
        base_url = cls.LOCAL_UPDATE_URL
        url = f"{base_url}/filter_by_hash_id/"
        payload = {"json": {"local_hash_id__in": local_hash_id_list}, }
        r = make_request(s=s, loaders=cls.LOADERS,r_type="POST", url=url, payload=payload,time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"{r.text}")
        all_metadatas = {m["local_hash_id"]: m for m in r.json()}
        return all_metadatas
    
    @classmethod
    def set_start_of_execution(cls, metadata, **kwargs):
        s = cls.build_session()
        base_url = cls.LOCAL_UPDATE_URL

        payload = {"json": kwargs}
        # r = self.s.patch(, **payload)
        url = f"{base_url}/{metadata['id']}/set_start_of_execution/"
        r = make_request(s=s, loaders=cls.LOADERS,r_type="PATCH", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        result = r.json()
        if result["last_time_index_value"] is not None:
            result["last_time_index_value"] = self.request_to_datetime(result["last_time_index_value"])
        return result

    @classmethod
    def set_end_of_execution(cls, metadata,timeout=None, **kwargs):
        s = cls.build_session()
        url = cls.LOCAL_UPDATE_URL + f"/{metadata['id']}/set_end_of_execution/"

        payload = {"json": kwargs}
        # r = self.s.patch(, **payload)
        r = make_request(s=s, loaders=cls.LOADERS,r_type="PATCH", url=url, payload=payload,time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request ")
    @classmethod
    def batch_set_end_of_execution(cls,update_map:dict,timeout=None):
        s = cls.build_session()
        url=f"{cls.LOCAL_UPDATE_URL}/batch_set_end_of_execution/"
        payload = {"json": {"update_map":update_map}}
        r = make_request(s=s, loaders=cls.LOADERS,r_type="PATCH",url=url, payload=payload,time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request ")

    @classmethod
    def set_last_update_index_time(cls,metadata,timeout=None):
        s = cls.build_session()
        url = cls.LOCAL_UPDATE_URL + f"/{metadata['id']}/set_last_update_index_time/"
        r = make_request(s=s, loaders=cls.LOADERS,r_type="GET", url=url, time_out=timeout)
        
        if r.status_code == 404:
            raise  SourceTableConfigurationDoesNotExist
        
        if r.status_code != 200:
            raise Exception(f"{metadata['local_hash_id']}{r.text}")
        return r

    @classmethod
    def set_last_update_index_time_from_update_stats(cls, metadata,
                                                     max_per_asset_symbol:dict,
                                                     last_time_index_value:float,
                                                     timeout=None):
        s = cls.build_session()
        url = cls.LOCAL_UPDATE_URL + f"/{metadata['id']}/set_last_update_index_time_from_update_stats/"
        payload = {"json": {"last_time_index_value": last_time_index_value,"max_per_asset_symbol":max_per_asset_symbol}}
        r = make_request(s=s, loaders=cls.LOADERS,payload=payload, r_type="GET", url=url, time_out=timeout)

        if r.status_code == 404:
            raise SourceTableConfigurationDoesNotExist

        if r.status_code != 200:
            raise Exception(f"{metadata['local_hash_id']}{r.text}")
        return r


    @staticmethod
    def serialize_for_json(kwargs):
        return serialize_to_json(kwargs)
    @classmethod
    def create_historical_update(cls, *args, **kwargs):
        """

        :param args:
        :type args:
        :param kwargs:
        :type kwargs:
        :return:
        :rtype:
        """
        s = cls.build_session()
        base_url = cls.LOCAL_TIME_SERIE_HISTORICAL_UPDATE
        data = cls.serialize_for_json(kwargs)
        payload = {"json": data, }
        r = make_request(s=s, loaders=cls.LOADERS,r_type="POST", url=f"{base_url}/", payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.url} {r.text}")

    @classmethod
    def get_mermaid_dependency_diagram(cls, local_hash_id,data_source_id,desc=True,timeout=None)->dict:
        """

        :param local_hash_id:
        :return:
        """
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{local_hash_id}/dependencies_graph_mermaid?desc={desc}&data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url,
                         time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r.json()

    #Node Updates
    @classmethod
    def get_all_dependencies(cls, hash_id,data_source_id,timeout=None):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/get_all_dependencies?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df

    @classmethod
    def get_all_dependencies_update_priority(cls, hash_id,data_source_id, timeout=None):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/get_all_dependencies_update_priority?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df

    @classmethod
    def get_max_depth(cls, hash_id, data_source_id,timeout=None):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/get_max_depth?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r.json()["max_depth"]

    @classmethod
    def get_upstream_nodes(cls, hash_id,data_source_id,timeout=None):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/get_upstream_nodes?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df

    @classmethod
    def create(cls, timeout=None,*args, **kwargs):
        url = cls.ROOT_URL + "/"
        payload = {"json": serialize_to_json(kwargs)}
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload, time_out=timeout)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        instance = cls(**r.json())
        return r.json()

    @classmethod
    def get(cls, local_hash_id,data_source_id:int):

        result = cls.filter(**{"local_hash_id": local_hash_id,
                               "remote_table__data_source__id":data_source_id,
                               "detail": True})
        if len(result) > 1:
            raise Exception("More than 1 return")
        if len(result)==0:
            return {}
        return result[0]

    @classmethod
    def filter(cls, *args,**kwargs):
        url = cls.LOCAL_UPDATE_URL
        
        payload = {} if kwargs is None else {"params": kwargs}

        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload=payload)
        if r.status_code == 404:
            raise LocalTimeSeriesDoesNotExist
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        return r.json()
    @classmethod
    def set_ogm_dependencies_linked(cls,hash_id,data_source_id):
        s = cls.build_session()
        url = cls.ROOT_URL + f"/{hash_id}/set_ogm_dependencies_linked?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, )
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r



class TimeSerie(BaseObject):
    """
    Main Methods of a standard time serie by hash_id
    """
    ROOT_URL = get_dynamic_table_metadata(TDAG_ENDPOINT)

    def __init__(self, *args, **kwargs):

        for key, value in kwargs.items():
            setattr(self, key, value)

    def patch(self, time_out: Union[None, int] = None, *args, **kwargs, ):
        url = self.ROOT_URL + f"/{self.id}/"
        payload = {"json": serialize_to_json(kwargs)}
        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload, time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
    @classmethod
    def get(cls, hash_id):

        result = cls.filter(payload={"hash_id": hash_id, "detail": True})
        if len(result)!=1:
            raise Exception("More than 1 return")
        return cls(**result[0])
    @classmethod
    def filter(cls, payload: Union[dict, None]):
        url = cls.ROOT_URL
        payload = {} if payload is None else {"params": payload}

        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload=payload)
        if r.status_code == 404:
            raise SchedulerDoesNotExist
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r.json()
    @classmethod
    def patch_by_hash(cls,hash_id:str,*args,**kwargs):
        metadata=cls.get(hash_id=hash_id)
        metadata.patch(*args,**kwargs)
    local_time_serie_update_details_id:Optional[int]=None
    retry_on_error: int = 0
    seconds_wait_on_retry: float = 50
    required_cpus: int = 1
    required_gpus: int = 0
    execution_time_out_seconds: float = 50
    update_schedule: str = "*/1 * * * *"

class ChatObject(BaseTdagPydanticModel,BaseObject):
    related_chat:  Optional[int] = Field(None, description="Primary key")
    created_at: Optional[datetime.datetime] =None  # Auto-generated at creation time
    object_type: str
    objects_path: str
    class Config:
        use_enum_values = True  # This ensures that enums are stored as their values (e.g., 'TEXT')

    def __str__(self):
        return f"{self.object_type} object created at {self.created_at}"

    @classmethod
    @property
    def ROOT_URL(cls):
        return get_chat_object_url(TDAG_ENDPOINT)



class DataSource(BaseTdagPydanticModel,BaseObject):
    id: Optional[int] = Field(None, description="The unique identifier of the Local Disk Source Lake")
    organization: Optional[int] = Field(None, description="The unique identifier of the Local Disk Source Lake")
    class_type:str

class DynamicTableDataSource(BaseTdagPydanticModel,BaseObject):
    id:int
    data_type:str
    related_resource:Optional[DataSource]=None
    class Config:
        use_enum_values = True  # This ensures that enums are stored as their values (e.g., 'TEXT')
    def __str__(self):
        return f"{self.object_type} object created at {self.created_at}"

    @classmethod
    @property
    def ROOT_URL(cls):
        return get_dynamic_table_data_source(TDAG_ENDPOINT)
    @classmethod
    def get_default_data_source_for_token(cls):
        global _default_data_source
        if _default_data_source is not None:
            return _default_data_source  # Return cached result if already set
        url = cls.ROOT_URL+"/get_default_data_source_for_token"

        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload={})

        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        _default_data_source = cls(**r.json())  # Cache the result
        return _default_data_source
    @classmethod
    def get_data_source_connection_details(cls,connection_id):
        url = cls.ROOT_URL + f"/{connection_id}/get_data_source_connection_details"

        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload={})

        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        data = r.json()
        return cls.get_class(data["data_type"])(**data)
    @classmethod
    def get(cls,id):
        url = cls.ROOT_URL + f"/{id}/"

        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload={})

        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")


        data=r.json()
        return cls.get_class(data["data_type"])(**data)
    @staticmethod
    def get_class(data_type):
        CLASS_FACTORY = {CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE: LocalDiskSourceLake,
                         CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB: TimeScaleDBDataSource
                         }
        return CLASS_FACTORY[data_type]


class PodLocalLake(DataSource):
    id: Optional[int] = Field(None, description="The unique identifier of the Local Disk Source Lake")
    in_pod: int = Field(..., description="The ID of the related Pod Source")
    datalake_name: str = Field(..., max_length=255, description="The name of the data lake")
    datalake_end: Optional[datetime.datetime] = Field(None, description="The end time of the data lake")
    datalake_start: Optional[datetime.datetime] = Field(None, description="The start time of the data lake")
    nodes_to_get_from_db: Optional[Dict] = Field(None, description="Nodes to retrieve from the database as JSON")
    persist_logs_to_file: bool = Field(False, description="Whether to persist logs to a file")
    use_s3_if_available: bool = Field(False, description="Whether to use S3 if available")


class LocalDiskSourceLake(DynamicTableDataSource):
    related_resource:PodLocalLake
    data_type:str=CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE

    @classmethod
    def get_or_create(cls, *args,**kwargs):
        url = cls.ROOT_URL + "/get_or_create/"
        for field in ['datalake_start', 'datalake_end']:
            if field in kwargs and kwargs[field] is not None:
                kwargs[field] = int(kwargs[field].timestamp())
        kwargs["data_type"]=CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload={"json":kwargs})

        if r.status_code not in  [200,201]:
            raise Exception(f"Error in request {r.text}")
        return cls(**r.json())

class TimeScaleDBDataSource(DynamicTableDataSource):
    related_resource: PodLocalLake
    data_type: str = CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB


class BaseYamlModel(BaseTdagPydanticModel,BaseObject):

    id: Optional[int] = Field(None, description="Primary key")
    created_at: Optional[datetime.datetime] = Field(None, description="Timestamp when the object was created")
    related_chat: Optional[int] = Field(None, description="ID of the related HistoricalChat object")
    yaml_content: Optional[str] = Field(None, description="Path to the YAML content file")
    name: str = Field(..., max_length=255, description="Name of the YAML entry")
    received_by_fund_builder: bool = Field(False, description="Indicates if received by fund builder")

    @staticmethod
    def _parse_parameters_filter(parameters):

        for key, value in parameters.items():
            if "__in" in key:
                assert isinstance(value, list)
                parameters[key] = ",".join(value)
        return parameters

    def add_linked_portfolio_time_series(self, time_out=None, *args, **kwargs):
        url = self.ROOT_URL + f"/{self.id}/add_linked_portfolio_time_series/"
        payload = {"json": serialize_to_json(kwargs)}
        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload, time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    def upload_datapoints(self, time_out=None, *args, **kwargs):
        url = self.ROOT_URL + f"/{self.id}/upload_datapoints/"
        portfolio_results = kwargs.get('portfolio_results')
        if portfolio_results is None:
            raise Exception("portfolio_results is required")

        # Reset index to include index in the data
        portfolio_results = portfolio_results.reset_index()

        # Convert datetime columns to strings to ensure JSON serialization
        for col in portfolio_results.columns:
            if pd.api.types.is_datetime64_any_dtype(portfolio_results[col]):
                portfolio_results[col] = portfolio_results[col].astype(str)

        # Convert DataFrame to list of dictionaries
        datapoints = portfolio_results.to_dict(orient='records')

        # Prepare the payload with a key that requests.post() accepts
        payload = {'json': {'datapoints': datapoints}}

        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="POST", url=url, payload=payload, time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    def patch(self, time_out=None, *args, **kwargs):
        url = self.ROOT_URL + f"/{self.id}/"
        payload = {"json": serialize_to_json(kwargs)}
        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload, time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    @classmethod
    def filter(cls, time_out=None, *args, **kwargs):
        base_url = cls.ROOT_URL
        params = cls._parse_parameters_filter(parameters=kwargs)

        request_kwargs = {"params": params, }
        url = f"{base_url}/"
        if "pk" in kwargs:
            url = f"{base_url}/{kwargs['pk']}/"
            request_kwargs = {}

        r = make_request(s=cls.build_session(), loaders=cls.LOADERS, r_type="GET", url=url, payload=request_kwargs,
                         time_out=time_out)

        if r.status_code != 200:
            if r.status_code == 401:
                raise Exception("Unauthorized please add credentials to environment")
            elif r.status_code == 500:
                raise Exception("Server Error")
            else:
                return {}, r
        else:
            serialized = [r.json()] if "pk" in kwargs else r.json()
            new_serialized = []

            for q in serialized:
                q["tdag_orm_class"] = cls.__name__
                try:
                    new_serialized.append(cls(**q))
                except Exception as e:
                    raise e

            return new_serialized, r


class ChatYamls(BaseYamlModel):
    linked_portfolio_time_series_ids: Optional[List[int]] = Field(
        None, description="IDs of the related LocalTimeSerie objects"
    )
    live_portfolio_details: Optional[Dict[str, Any]] = Field(
        None, description="JSON details of the live portfolio"
    )
    backtest_portfolio_details: Optional[Dict[str, Any]] = Field(
        None, description="JSON details of the backtest portfolio"
    )

    @classmethod
    @property
    def ROOT_URL(cls):
        return get_chat_yaml_url(TDAG_ENDPOINT)

class SignalYamls(BaseYamlModel):
    code: Optional[str] = Field(
        None, description="Path to the generated code file"
    )

    @classmethod
    @property
    def ROOT_URL(cls):
        return get_signal_yaml_url(TDAG_ENDPOINT)

def register_strategy(json_payload:dict, timeout=None):
    url = TDAG_ENDPOINT + "/tdag-gpt/register_strategy/"
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    s.headers.update(loaders.auth_headers)
    retries = Retry(total=2, backoff_factor=2)
    s.mount('http://', HTTPAdapter(max_retries=retries))

    r = make_request(s=s, r_type="POST", url=url, payload={"json": json_payload},
                     loaders=loaders, time_out=timeout)
    return r



def register_default_configuration(json_payload:dict, timeout=None):
    url = TDAG_ENDPOINT + "/tdag-gpt/register_default_configuration/"
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    s.headers.update(loaders.auth_headers)
    retries = Retry(total=2, backoff_factor=2)
    s.mount('http://', HTTPAdapter(max_retries=retries))

    r = make_request(s=s, r_type="POST", url=url, payload={"json": json_payload},
                     loaders=loaders, time_out=timeout)
    return r

def create_configuration_for_strategy(json_payload: dict, timeout=None):
    url = TDAG_ENDPOINT + "/tdag-gpt/create_configuration_for_strategy/"
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    s.headers.update(loaders.auth_headers)
    retries = Retry(total=2, backoff_factor=2)
    s.mount('http://', HTTPAdapter(max_retries=retries))

    r = make_request(s=s, r_type="POST", url=url, payload={"json": json_payload},
                     loaders=loaders, time_out=200)
    return r


class DynamicTableHelpers:
    ROOT_URL = TDAG_ENDPOINT +"/orm/api"
    DATE_FORMAT = DATE_FORMAT
    LOADERS = loaders
    def __init__(self,authorization_headers_kwargs:Union[dict, None] = None,
                 time_series_orm_db_connection: Union[str, None] = None,
                 time_series_orm_root: Union[str, None] = None,
                 TSLU:Union[TimeSerieLocalUpdate,None]=None,
                 logger:Union[structlog.BoundLogger,None]=None
                 ):


        if time_series_orm_root is not None:
            if "orm/api" not in time_series_orm_root:
                time_series_orm_root=self.get_orm_root_from_base_url(time_series_orm_root)
            self.ROOT_URL = time_series_orm_root

        if authorization_headers_kwargs is not None:
            self.LOADERS = AuthLoaders(**authorization_headers_kwargs,
                                       time_series_orm_token_url=self.rest_token_auth_url)

        self.time_series_orm_db_connection = time_series_orm_db_connection

        self.TimeSerieLocalUpdate=TimeSerieLocalUpdate if TSLU==None else TSLU
        self.time_series_orm_uri_db_connection=None
        self.logger=logger if logger is not None else structlog.stdlib.get_logger()

    def set_TSLU(self,TSLU:TimeSerieLocalUpdate):
        self.TimeSerieLocalUpdate=TSLU
    
    def set_time_series_orm_uri_db_connection(self,uri:str):
        self.time_series_orm_uri_db_connection=uri
    
    def make_request(self,r_type:str,url:str,payload:Union[dict,None]=None,
                     timeout: Union[float, None] = None
                     ):
       r=make_request(s=self.s,r_type=r_type,url=url,payload=payload,
                      loaders=self.LOADERS,time_out=timeout)
       return r


    @property
    def s(self):
        from requests.adapters import HTTPAdapter, Retry
        s = requests.Session()
        s.headers.update(self.LOADERS.auth_headers)
        retries = Retry(total=2, backoff_factor=2,)
        s.mount('http://', HTTPAdapter(max_retries=retries))
        return s
    @staticmethod
    def serialize_for_json(kwargs):
       return serialize_to_json(kwargs)

    @staticmethod
    def _parse_parameters_filter(parameters):

        for key, value in parameters.items():
            if "__in" in key:
                assert isinstance(value, list)
                parameters[key] = ",".join(value)
        return parameters

    @staticmethod
    def request_to_datetime(string_date: str):
        return request_to_datetime(string_date)

    def get_orm_root_from_base_url(self,base_url):
        return base_url+"/orm/api"

    @property
    def root_url(self):
        return self.ROOT_URL + "/dynamic_table"

    @property
    def source_table_config_url(self):
        return self.ROOT_URL + "/source_table_config"
    @property
    def historical_update_url(self):
        return self.ROOT_URL + "/historical_update"

    @property
    def update_details_url(self):
        return self.ROOT_URL + "/update_details"

    @property
    def local_update_details_url(self):
        return self.ROOT_URL + "/local_update_details"

    @property
    def rest_token_auth_url(self):
        base=self.ROOT_URL.replace("/orm/api","")
        return f"{base}/auth/rest-token-auth/"

    def patch_update_details(self,*args,**kwargs):
        base_url = self.update_details_url
        
        data=self.serialize_for_json(kwargs)
        payload = {"json": data }
        r=self.make_request(r_type="PATCH",url=f"{base_url}/0/",payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    def patch_local_update_details(self,*args,**kwargs):
        base_url = self.local_update_details_url

        data = self.serialize_for_json(kwargs)
        payload = {"json": data}
        r = self.make_request(r_type="PATCH", url=f"{base_url}/0/", payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    def destroy(self, metadata, delete_only_table: bool):
        base_url = self.root_url

        payload = {"json": {"delete_only_table": delete_only_table},}
        r = self.s.delete(f"{base_url}/{metadata['id']}/", **payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
    def get_all_hash_id(self):
        base_url = self.root_url
        r = self.s.get(f"{base_url}/get_all_hash_id",)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.json()}")
        return r.json()
    def delete_all_data_after_date(self,after_date:str):
        base_url = self.root_url
        data = self.serialize_for_json({"after_date": after_date})
        payload = {"json": data, }
        r = self.s.patch(f"{base_url}/delete_all_data_after_date/", **payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.json()}")
        
        
    def delete_after_date(self, metadata:Union[dict,None], after_date: str):

        base_url = self.root_url
        data = self.serialize_for_json({"after_date": after_date})
        payload = {"json": data, }
        r = self.s.patch(f"{base_url}/{metadata['id']}/delete_after_date/", **payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    def search(self, key_word: str):
        base_url = self.root_url
        url=f"{base_url}/?search={key_word}"
        # r = self.s.get(url )
        r=self.make_request(r_type="GET",url=url,)

        if r.status_code != 200:
            raise Exception(f"{base_url} Error in request {r.json}")
        else:
            serialized = r.json()

            return serialized, r

    def exist(self, *args, **kwargs):


        base_url = self.root_url
        payload = {"json": kwargs,}
        # r = self.s.patch(, **payload)
        r=self.make_request(r_type="PATCH",url=f"{base_url}/exist/",payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.json()}")
        return r.json(), r



    def filter(self, *args, **kwargs):
        # base_url = self.root_url
        # instances, r = self.dts_ws.filter(*args, **kwargs)
        instances, r = self.filter_rest(*args, **kwargs)
        return instances, r

    def filter_rest(self, *args, **kwargs):
        base_url = self.root_url
        params = self._parse_parameters_filter(parameters=kwargs)
        url=f"{base_url}/"
        payload=dict(params=params)
        # r = self.s.get(url, params=params)
        r = self.make_request(r_type="GET", url=url,payload=payload )
        if r.status_code ==404:
            raise DynamicTableDoesNotExist
        elif r.status_code != 200:
            raise Exception(f"Error in request {r.url} {r.text}")
        else:
            serialized = r.json()

            return serialized, r

    def get_rest(self, *args, **kwargs):
        instance, r = self.filter_rest(*args, **kwargs)



        if len(instance) > 1:
            raise Exception(f"Get does not return only one instance {r}")
        elif len(instance) == 0:
            return {}, r
        else:
            metadata=instance[0]
            # hack to patch nodes
            if metadata["ogm_linked"]==False:
                raise Exception("OGM is not linked with metadata")

            return metadata, r

    def get(self,class_name=None, *args, **kwargs):
        instance, r = self.get_rest(*args, **kwargs)
        # try:
        #     instance, r = self.dts_ws.get(*args, **kwargs)
        # except (WebsocketMessageIdNotFound , websockets.exceptions.ConnectionClosedError)as e:
        #     instance, r = self.get_rest(*args,**kwargs)
        return instance

    def get_configuration(self, hash_id: str):

        data, _r = self.get(hash_id=hash_id)
        if len(data) == 0:
            return None, None
        build_configuration, build_meta_data = data["build_configuration"], data["build_meta_data"]

        return build_configuration, build_meta_data



    def create(self, metadata_kwargs:dict):
        """

        :return:
        :rtype:
        """

     
        metadata_kwargs = self.serialize_for_json(metadata_kwargs)
        time_serie_node, metadata = TimeSerieNode.create(metadata_kwargs=metadata_kwargs)
       
        return metadata

    def create_table_from_source_table_configuration(self,source_table_config_id:int,timeout=None):
        base_url = self.source_table_config_url


        r = self.s.post(f"{base_url}/{source_table_config_id}/create_table_from_source_table_configuration/")
        if r.status_code != 201:

            raise Exception(r.text)


    def create_source_table_configuration(self, *args, **kwargs):

        base_url = self.source_table_config_url
        data = self.serialize_for_json(kwargs)
        payload = {"json": data, }
        r = self.s.post(f"{base_url}/", **payload)
        if r.status_code != 201:
            if r.status_code == 409:
                raise AlreadyExist(r.text)
            else:
                raise Exception(r.text)
        return r.json()

    def get_latest_value(self, hash_id):
        """
        Gets latest value from Hash_id
        :param hash_id:
        :type hash_id:
        :return:
        :rtype:
        """
        r, j = self.get_rest(hash_id=hash_id,class_name=None)
        if len(r) == 0:
            return None
        if r['sourcetableconfiguration'] is None:
            return None
        if r['sourcetableconfiguration']["last_time_index_value"] is None:
            return None

        date = self.request_to_datetime(string_date=r["sourcetableconfiguration"]["last_time_index_value"])

        return date

    @classmethod
    def _break_pandas_dataframe(cls, data_frame: pd.DataFrame, time_index_name: Union[str, None] = None):
        """

        :param data_frame:
        :param time_index_name:
        :return:
        """
        if time_index_name == None:
            time_index_name = data_frame.index.names[0]
            if time_index_name is None:
                time_index_name = "time_index"
                names = [c if i != 0 else time_index_name for i, c in
                         enumerate(data_frame.index.names)]
                data_frame.index.names = names

        time_col_loc = data_frame.index.names.index(time_index_name)
        column_index_names = data_frame.columns.names
        index_names = data_frame.index.names
        data_frame = data_frame.reset_index()
        data_frame.columns = [str(c) for c in data_frame.columns]
        data_frame = data_frame.rename(columns={data_frame.columns[time_col_loc]: time_index_name})
        column_dtypes_map = {key: str(value) for key, value in data_frame.dtypes.to_dict().items()}

        return data_frame, column_index_names, index_names, column_dtypes_map, time_index_name

    def _insert_data_into_hash_id(self, serialzied_data_frame: pd.DataFrame, metadata,local_metadata:dict,
                                  overwrite: bool, time_index_name:str,index_names:list,
                                  historical_update_id:Union[None,int],
                                  batch_size_limit=10,batch_size=1000
                                  ) -> dict:#LocalMetaData
        """

        Parameters
        ----------
        serialzied_data_frame :
        metadata :
        index_stats :

        Returns local_meta_data
        -------

        """
        from joblib import Parallel, delayed
       
        
        if "asset_symbol" in serialzied_data_frame.columns:
            serialzied_data_frame['asset_symbol']=serialzied_data_frame['asset_symbol'].astype(str)

        base_url = self.root_url
        serialzied_data_frame = serialzied_data_frame.replace({np.nan: None})
        for c in serialzied_data_frame.columns:
            if any([t in c for t in JSON_COMPRESSED_PREFIX]) == True:
                assert isinstance(serialzied_data_frame[c].iloc[0], dict)




        call_end_of_execution = False
        for c in serialzied_data_frame:
            if any([t in c for t in JSON_COMPRESSED_PREFIX]) == True:
                serialzied_data_frame[c] = serialzied_data_frame[c].apply(lambda x: json.dumps(x).encode())

        #if overwrite then decompress the chunks
        recompress=False
        if overwrite==True:
            url = f"{base_url}/{metadata['id']}/decompress_chunks/"
            r = self.make_request(r_type="POST", url=url,
            payload={"json":{"start_date":serialzied_data_frame[time_index_name].min().strftime(DATE_FORMAT),
                             "end_date":serialzied_data_frame[time_index_name].max().strftime(DATE_FORMAT),
                                                                           }},timeout=60*5)
            if r.status_code not in [200,204]:
                logger.error(r.text)
                raise Exception("Error trying to decompress table")
            else:
                if r.status_code == 200:
                    recompress==True



        table_is_empty = metadata["sourcetableconfiguration"]["last_time_index_value"] is None


        data_source_configuration=DynamicTableDataSource.get_data_source_connection_details(metadata["data_source"]["id"])
        if data_source_configuration["__type__"]!=CONSTANTS.CONNECTION_TYPE_POSTGRES:
            raise Exception

        last_time_index_value,max_per_asset_symbol=direct_table_update(serialized_data_frame=serialzied_data_frame,
                            time_series_orm_db_connection=data_source_configuration["connection_details"],
                            table_name=metadata["hash_id"],
                            overwrite=overwrite,index_names=index_names,
                            time_index_name=time_index_name,table_is_empty=table_is_empty,
                            table_index_names=metadata["table_index_names"],
                            )


        r = self.TimeSerieLocalUpdate.set_last_update_index_time_from_update_stats(max_per_asset_symbol=max_per_asset_symbol,
                                                                 last_time_index_value=last_time_index_value,
                                                                                   metadata=local_metadata,

                                                                 )



        try:
            result=r.json()
        except Exception as e:
            logger.warning(insert_direct)
            logger.warning(all_records)
            raise e
        if recompress== True:
            pass

        if call_end_of_execution == True:
            #todo: fix the historical update_id
            self.TimeSerieLocalUpdate.set_end_of_execution(metadata=local_metadata,error_on_update=False,
                                      historical_update_id=historical_update_id)

        
        return result



    # def set_last_update_index_time(self,metadata,timeout=None):
    #     base_url = self.root_url
    #     url=f"{base_url}/{metadata['id']}/set_last_update_index_time/"
    #     # r = self.s.get()
    #     r = self.make_request(r_type="GET", url=url,timeout=timeout)
    #     if r.status_code != 200:
    #         raise Exception(f"{metadata['hash_id']}{r.text}")
    #     return r

    def filter_by_hash_id(self, hash_id_list: list):

        base_url = self.root_url
        url = f"{base_url}/filter_by_hash_id/"
        payload = {"json": {"hash_id__in": hash_id_list}, }
        r = self.make_request(r_type="POST", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"{r.text}")
        all_metadatas = {m["hash_id"]: m for m in r.json()}
        return all_metadatas
    def upsert_data_into_table(self, metadata:dict,local_metadata:dict,
                            historical_update_id:Union[int,None],
                               data: pd.DataFrame, overwrite: bool,
                               batch_size_limit=10, batch_size=1000
                               ):
        """
        1) Build or get metadata
        2) build table configuration relationships
        Parameters
        ----------
        build_meta_data :
        build_configuration :
        data :

        Returns
        -------

        """
        overwrite=True #ALWAYS OVERWRITE

        data, column_index_names, index_names, column_dtypes_map, time_index_name = self._break_pandas_dataframe(
            data)
        #overwrite data origina data frame to release memory
        if not data[time_index_name].is_monotonic_increasing:
            data = data.sort_values(time_index_name)

        stc=metadata['sourcetableconfiguration']
        if stc is  None:
            try:
                source_configuration = self.create_source_table_configuration(column_dtypes_map=column_dtypes_map,
                                                                              index_names=index_names,
                                                                              time_index_name=time_index_name,
                                                                              column_index_names=column_index_names,
                                                                              metadata_id=metadata["id"]
                                                                              )
                metadata["sourcetableconfiguration"]=source_configuration
            except AlreadyExist:
                source_configuration = metadata["sourcetableconfiguration"]
                if overwrite ==False:
                    # assert data_frame
                    data = data[
                        data[time_index_name] > self.request_to_datetime(source_configuration['last_time_index_value'])]
        local_metadata = self._insert_data_into_hash_id(serialzied_data_frame=data, metadata=metadata,
                                                   local_metadata=local_metadata,
                                                   overwrite=overwrite, time_index_name=time_index_name,
                                                   index_names=index_names,
                                                   historical_update_id=historical_update_id,
                                                   batch_size_limit=batch_size_limit, batch_size=batch_size,
                                                   )

        return local_metadata

    def _direct_data_from_db(self, metadata: dict, connection_config: dict,
                             start_date: Union[datetime.datetime, None] = None,
                             great_or_equal: bool = True, less_or_equal: bool = True,
                             end_date: Union[datetime.datetime, None] = None,
                             columns: Union[list, None] = None):
        """
        Connects directly to the DB without passing through the ORM to speed up calculations.

        Parameters
        ----------
        metadata : dict
            Metadata containing table and column details.
        connection_config : dict
            Connection configuration for the database.
        start_date : datetime.datetime, optional
            The start date for filtering. If None, no lower bound is applied.
        great_or_equal : bool, optional
            Whether the start_date filter is inclusive (>=). Defaults to True.
        less_or_equal : bool, optional
            Whether the end_date filter is inclusive (<=). Defaults to True.
        end_date : datetime.datetime, optional
            The end date for filtering. If None, no upper bound is applied.
        columns : list, optional
            Specific columns to select. If None, all columns are selected.

        Returns
        -------
        pd.DataFrame
            Data from the table as a pandas DataFrame, optionally filtered by date range.
        """
        import psycopg2
        import pandas as pd




        if connection_config["__type__"] != CONSTANTS.CONNECTION_TYPE_POSTGRES:
            raise NotImplementedError("Only PostgreSQL is supported.")

        def fast_table_dump(connection_config, table_name,):
            query = f"COPY {table_name} TO STDOUT WITH CSV HEADER"

            with psycopg2.connect(connection_config['connection_details']) as connection:
                with connection.cursor() as cursor:
                    import io
                    buffer = io.StringIO()
                    cursor.copy_expert(query, buffer)
                    buffer.seek(0)
                    df = pd.read_csv(buffer)
                    return df

        # Build the SELECT clause
        select_clause = ", ".join(columns) if columns else "*"

        # Build the WHERE clause dynamically
        where_clauses = []
        time_index_name = metadata['sourcetableconfiguration']['time_index_name']
        if start_date:
            operator = ">=" if great_or_equal else ">"
            where_clauses.append(f"{time_index_name} {operator} '{start_date}'")
        if end_date:
            operator = "<=" if less_or_equal else "<"
            where_clauses.append(f"{time_index_name} {operator} '{end_date}'")

        # Combine WHERE clauses
        where_clause = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        # Construct the query
        query = f"SELECT {select_clause} FROM {metadata['table_name']} {where_clause}"
        # if where_clause=="":
        #     data=fast_table_dump(connection_config, metadata['table_name'])
        #     data[metadata["sourcetableconfiguration"]['time_index_name']]=pd.to_datetime(data[metadata["sourcetableconfiguration"]['time_index_name']])
        # else:
        with psycopg2.connect(connection_config['connection_details']) as connection:
            with connection.cursor() as cursor:
                cursor.execute(query)
                column_names = [desc[0] for desc in cursor.description]
                data = cursor.fetchall()

        # Convert to DataFrame
        data = pd.DataFrame(data=data, columns=column_names)

        data=data.set_index(metadata['sourcetableconfiguration']["index_names"])

        return data

    def filter_by_assets_ranges(self,table_name:str,asset_ranges_map:dict,connection_config:dict):

        if connection_config["__type__"]!=CONSTANTS.CONNECTION_TYPE_POSTGRES:
            raise NotImplementedError

        query_base = f"""
                                SELECT * FROM {table_name}
                                WHERE
                            """
        # Initialize a list to store the query parts and another for parameters
        query_parts = []

        # Build query dynamically based on the dictionary
        for symbol, range_dict in asset_ranges_map.items():

            if range_dict['end_date'] is not None:
                tmp_query = f" (asset_symbol ='{symbol}'  AND time_index BETWEEN '{range_dict['start_date']}' AND '{range_dict['end_date']}') "

            else:
                tmp_query = f" (asset_symbol ='{symbol}'  AND time_index {range_dict['start_date_operand']} '{range_dict['start_date']}') "
            query_parts.append(tmp_query)

        full_query = query_base + " OR ".join(query_parts)

        df = read_sql_tmpfile(full_query, time_series_orm_uri_db_connection=connection_config["connection_details"])
        return df
        
    def get_data_by_time_index(self, metadata: dict,connection_config:dict,
                               start_date: Union[datetime.datetime, None] = None,
                               great_or_equal: bool = True, less_or_equal: bool = True,
                               end_date: Union[datetime.datetime, None] = None,
                               columns: Union[list, None] = None,
                               asset_symbols:Union[list,None]=None,

                               ):

        return self._direct_data_from_db(metadata=metadata,
                                             connection_config=connection_config,
                                             start_date=start_date,great_or_equal=great_or_equal,
                                             less_or_equal=less_or_equal,end_date=end_date,columns=columns)



    def time_serie_exist_in_db(self, hash_id):
        """

        Returns
        -------

        """
        metadata, _ = self.get(hash_id=hash_id)
        if len(metadata) == 0:
            return False
        else:
            if metadata["sourcetableconfiguration"] is not None:
                if metadata["sourcetableconfiguration"]["last_time_index_value"] is not None:
                    return True
        return False

    def set_compression_policy(self, metadata, interval: str):
        """

        :param hash_id:
        :type hash_id:
        :return:
        :rtype:
        """

        base_url = self.root_url

        payload = { "json": {"interval": interval,}}
        # r = self.s.patch(f"{base_url}/{metadata['id']}/set_compression_policy/", **payload)
        url=f"{base_url}/{metadata['id']}/set_compression_policy/"
        r = self.make_request(r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"{metadata['hash_id']} : {r.json()}")

    def set_retention_policy(self, metadata, interval: str):
        base_url = self.root_url

        payload = {"json": {"interval": interval, }}
        # r = self.s.patch(f"{base_url}/{metadata['id']}/set_retention_policy/", **payload)
        url=f"{base_url}/{metadata['id']}/set_retention_policy/"
        r = self.make_request(r_type="PATCH", url=url,payload=payload)
        if r.status_code != 200:
            raise Exception(f"{metadata['hash_id']} : {r.text}")
    def set_policy_for_descendants(self,hash_id,policy,pol_type,exclude_ids,extend_to_classes):

        r=TimeSerieNode.set_policy_for_descendants(hash_id,policy,pol_type,exclude_ids,extend_to_classes)




    def build_or_update_update_details(self, metadata, *args, **kwargs):

        base_url = self.root_url
        payload = { "json": kwargs}
        # r = self.s.patch(, **payload)
        url=f"{base_url}/{metadata['id']}/build_or_update_update_details/?data_source_id={metadata['data_source']['id']}"
        r=self.make_request(r_type="PATCH",url=url,payload=payload)
        if r.status_code != 202:
            raise Exception(f"Error in request {r.text}")
        return r.json()
    def patch(self,metadata,timeout=None,*args,**kwargs):
        """
        Main patch method
        :return:
        :rtype:
        """

        base_url = self.root_url

        payload = {"json": kwargs}
        # r = self.s.patch(f"{base_url}/{metadata['id']}/", **payload)
        url = f"{base_url}/{metadata['id']}/"
        r = self.make_request(r_type="PATCH", url=url, payload=payload,timeout=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r.json()

    def reset_dependencies_states(self,metadata,**kwargs):

        base_url = self.root_url

        payload = { "json": kwargs}
        # r = self.s.patch(, **payload)
        url=f"{base_url}/{metadata['id']}/reset_dependencies_states/"
        r = self.make_request(r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request ")
    def get_pending_nodes(self, metadata, **kwargs):

        base_url = self.root_url

        payload = { "json": kwargs}
        # r = self.s.patch(f"{base_url}/{metadata['id']}/get_pending_update_nodes/", **payload)
        url=f"{base_url}/{metadata['id']}/get_pending_update_nodes/"
        r = self.make_request(r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request ")
        return r.json()
    def must_update_by_hash_id(self,hash_id:str):

        base_url = self.root_url

        payload = { "json": {"use_hash_id":True}}
        # r = self.s.patch(, **payload)
        url=f"{base_url}/{hash_id}/must_update/"
        r = self.make_request(r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request ")
        r=r.json()
        return r["must_update"],r["metadata"]
    def get_metadatas_and_set_updates(self,*args,**kwargs):

        base_url = self.root_url

        payload = { "json": kwargs}
        # r = self.s.post(f"{base_url}/get_metadatas_and_set_updates/", **payload)
        url = f"{base_url}/get_metadatas_and_set_updates/"
        r = self.make_request(r_type="POST", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        r = r.json()
        return r
    def _build_table_response(self,data:pd.DataFrame,source_table_config:dict):
        infered_dtypes = {k: str(c) for k, c in data.dtypes.to_dict().items()}
        config_types = {c: source_table_config["column_dtypes_map"][c] for c in infered_dtypes.keys()}
        for c, c_type in config_types.items():
            if c_type != infered_dtypes[c]:
                if data.shape[0] > 0:
                    if c_type == 'datetime64[ns, UTC]':
                        if isinstance(data[c].iloc[0], str):
                            data[c] = pd.to_datetime(data[c])
                        else:
                            data[c] = pd.to_datetime(data[c] * 1e6, utc=True)
                    else:
                        data[c] = data[c].astype(c_type)
        data = data.set_index(source_table_config['index_names'])
        return data

    def concatenate_tables_on_time_index(self, *args, **kwargs):
        raise NotImplementedError
        base_url = self.root_url
        kwargs['target_value']=kwargs['target_value'].strftime(DATE_FORMAT)
        payload = { "json": kwargs}
        r = self.s.post(f"{base_url}/concatenate_tables_on_time_index/", **payload)
        if r.status_code != 200:
            raise Exception(f"Error in request ")
        r = r.json()
        base_table_config=kwargs['base_table_config']
        all_cols = list(base_table_config['column_dtypes_map'].keys())
        all_cols = ["key"] + all_cols
        results = pd.DataFrame(columns=all_cols, data=r)
        base_table_config["column_dtypes_map"]["key"]="object"
        results=self._build_table_response(data=results,source_table_config=base_table_config)

        return results
    # def batch_set_end_of_execution(self,update_map:dict):
    #     base_url = self.root_url
    #
    #     payload = {"json": {"update_map":update_map}}
    #     r = self.make_request(r_type="PATCH",url=f"{base_url}/batch_set_end_of_execution/", payload=payload)
    #     if r.status_code != 200:
    #         raise Exception(f"Error in request ")

    def depends_on_connect(self,source_hash_id:str, target_hash_id:str,target_class_name:str,
                           source_local_hash_id:str,
                           target_local_hash_id:str,
                           source_data_source_id:id,
                            target_data_source_id:id,
                           target_human_readable:str):


        TimeSerieNode.depends_on_connect(source_hash_id=source_hash_id, target_hash_id=target_hash_id,
                                         source_local_hash_id=source_local_hash_id,target_local_hash_id=target_local_hash_id,
                                         target_class_name=target_class_name,
                                         source_data_source_id=source_data_source_id,
                                         target_data_source_id=target_data_source_id,
                           target_human_readable=target_human_readable)


    def rename_data_table(self,source_hash_id:str,target_hash_id:str):
        from .utils import TDAG_ORM_DB_CONNECTION

        source_table_metadata, _ = self.get(hash_id=source_hash_id)
        target_metadata, _ = self.get(hash_id=target_hash_id)
        
        if len(target_metadata)>0:
            #delete table
            raise Exception("Table exist delete first")
      
        # no metadata cant coopy
        metadata_kwargs = {}

        node_kwargs = dict(hash_id=target_hash_id,
                           class_name="DeflatedPricesMCap",
                           human_readable=f"Deflated Prices Market CAP 19 Assets"
                           )
        for f in ['hash_id', 'build_configuration', 'build_meta_data', 'human_readable',
                  'retention_policy_config', 'compression_policy_config'
                  ]:
            metadata_kwargs[f] = source_table_metadata[f]
        metadata_kwargs["hash_id"]=target_hash_id
        metadata_kwargs["human_readable"]=node_kwargs["human_readable"]
        try:
            ts_created = TimeSerieNode.create(metadata_kwargs=metadata_kwargs, node_kwargs=node_kwargs)
            time.sleep(5)
            target_metadata, _ = self.get(hash_id=target_hash_id)
        except Exception as e:
            raise e
        import psycopg2
        with psycopg2.connect(TDAG_ORM_DB_CONNECTION, ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(f"ALTER TABLE {source_hash_id} RENAME TO {target_hash_id};")




    def copy_table_data_to_other_orm(self, target_database_uri: str, target_admin_user: str, target_admin_password: str,
                                     target_orm_url_api_root: str,target_gcp_credentials_path:Union[str,None],
                                     hash_id_contains: str,

                                     source_database_uri: str,
                                     source_orm_url_api_root: str,
                                     source_orm_admin_user: str,
                                     source_orm_admin_password: str,
                                     source_gcp_credentials_path: str,
                                     
                                     start_date: datetime.datetime,
                                     end_date: datetime.datetime, great_or_equal: bool, less_or_equal: bool,
                                     overwrite=False,copy_descendants=False,exclude_hash_ids:Union[list,None]=None,

                                
                                     
                                     ):
        """
         This function copies table data from an ORM to another one example:
        copy_database
        Parameters
        ----------
        target_database_uri :
        target_admin_user :
        target_admin_password :
        target_orm_url_api_root :
        hash_id_contains :
        start_date :
        end_date :
        great_or_equal :
        less_or_equal :
        overwrite :

        Returns
        -------

        """


        from tqdm import tqdm

        local_hash_prefix=f"cp_{self.ROOT_URL}".replace("/orm/api","").replace("http://","")
        updated_hids=[]
        exclude_hash_ids=[] if exclude_hash_ids is None else exclude_hash_ids
        
        authorization_headers_kwargs = dict(time_series_orm_admin_user=target_admin_user,
                                            time_series_orm_admin_password=target_admin_password,
                                            gcp_credentials_path=target_gcp_credentials_path
                                            )
        
        target_dth = DynamicTableHelpers(
            authorization_headers_kwargs=authorization_headers_kwargs,
            time_series_orm_db_connection=target_database_uri,
            time_series_orm_root=target_orm_url_api_root)

        source_dth = DynamicTableHelpers(
            authorization_headers_kwargs= dict(time_series_orm_admin_user=source_orm_admin_user,
                                            time_series_orm_admin_password=source_orm_admin_password,
                                            gcp_credentials_path=source_gcp_credentials_path
                                            ),
            time_series_orm_db_connection=source_database_uri,
            time_series_orm_root=source_orm_url_api_root)
        
        

        SKIP_EXIST=True
      

        class TargetTimeSerieNode(TimeSerieNode):
            LOADERS = target_dth.LOADERS
            ROOT_URL = get_ts_node_url(target_dth.ROOT_URL.replace("/orm/api",""))

        class TargetTimeSerieLocalUpdate(TimeSerieLocalUpdate):
            LOADERS = target_dth.LOADERS
            ROOT_URL = get_time_serie_local_update_url(target_dth.ROOT_URL.replace("/orm/api", ""))
            LOCAL_UPDATE_URL = get_time_serie_local_update_table_url(target_dth.ROOT_URL.replace("/orm/api", ""))
            LOCAL_TIME_SERIE_HISTORICAL_UPDATE = get_local_time_serie_historical_update_url(
                target_dth.ROOT_URL.replace("/orm/api", ""))
            
        class SourceTimeSerieNode(TimeSerieNode):
            LOADERS = source_dth.LOADERS
            ROOT_URL = get_ts_node_url(source_dth.ROOT_URL.replace("/orm/api", ""))

        class SourceTimeSerieLocalUpdate(TimeSerieLocalUpdate):
            LOADERS = source_dth.LOADERS
            ROOT_URL = get_time_serie_local_update_url(source_dth.ROOT_URL.replace("/orm/api", ""))
            LOCAL_UPDATE_URL = get_time_serie_local_update_table_url(source_dth.ROOT_URL.replace("/orm/api", ""))
            LOCAL_TIME_SERIE_HISTORICAL_UPDATE = get_local_time_serie_historical_update_url(
                source_dth.ROOT_URL.replace("/orm/api", ""))

        target_dth.set_TSLU(TSLU=TargetTimeSerieLocalUpdate)
        source_dth.set_TSLU(TSLU=SourceTimeSerieLocalUpdate)
        tables, _ = source_dth.search(key_word=hash_id_contains)
        if hash_id_contains == "*":
            tables, _ = source_dth.filter()

        if isinstance(tables[0], dict):
            tables = [t["hash_id"] for t in tables]
        if copy_descendants == True:
            for hash_id in tqdm(copy.deepcopy(tables)):
                desc_df=SourceTimeSerieLocalUpdate.get_all_dependencies(hash_id=hash_id)
                tables.extend(desc_df["remote_table_hash_id"].to_list())
            tables=list(set(tables))
            tables=[c for c in tables if "historicalcoinsupply" not in c]
      
        for hash_id in tqdm(tables):
            if hash_id in exclude_hash_ids:
                continue
            source_table_metadata = source_dth.get(hash_id=hash_id)
            if "wrappertimeserie" in hash_id:
                # Do not copy wrappers
                continue
            # if "historicalcoinsupply" in hash_id:
            #     continue
            if source_table_metadata["sourcetableconfiguration"] is None:
                logger.info(f"sourcetableconfiguration for {hash_id} is empty - not copying")
                continue
            

          
            target_metadata,_= target_dth.filter(hash_id=hash_id)
            if len(target_metadata)!=0 and SKIP_EXIST==True:
                logger.info(f"Skipping {hash_id} already exist")
                continue
            data = source_dth.get_data_by_time_index(metadata=source_table_metadata, start_date=start_date, end_date=end_date,
                                               great_or_equal=great_or_equal, less_or_equal=less_or_equal)
            if "testfeature2" in hash_id:
                target_metadata, _ = target_dth.filter(hash_id="testfeature2_copy")
                source_table_metadata["hash_id"]="testfeature2_copy"
            if len(target_metadata) == 0:
                #no metadata cant coopy
                metadata_kwargs={}
                node_kwargs=dict(hash_id=source_table_metadata["hash_id"],
                                 source_class_name=source_table_metadata['source_class_name'],
                                    human_readable=source_table_metadata["human_readable"]
                                 )
                for f in ['hash_id', 'build_configuration', 'build_meta_data', 'human_readable',
                          'retention_policy_config', 'compression_policy_config'
                          ]:
                    metadata_kwargs[f]=source_table_metadata[f]
                try:
                    ts_created=TargetTimeSerieNode.create(metadata_kwargs=metadata_kwargs,node_kwargs=node_kwargs)
                    time.sleep(5)
                    target_metadata = target_dth.get(hash_id=hash_id)
                except Exception as e:
                    raise e
            #build local updater
            local_hash_id = local_hash_prefix + hash_id.replace(source_table_metadata['source_class_name'].lower(),"")
            local_hash_id=local_hash_id[:63]
            target_local_metadata=TargetTimeSerieLocalUpdate.get(local_hash_id=local_hash_id)
            if len(target_local_metadata)==0:
                #create local updated

                local_metadata_kwargs = dict(local_hash_id=local_hash_id,
                                       build_configuration=target_metadata["build_configuration"],
                                             build_meta_data=target_metadata["build_meta_data"],
                                       remote_table__hash_id=target_metadata["hash_id"],
                                             human_readable=target_metadata["human_readable"]
                                             )

                local_node_kwargs = {"hash_id": local_hash_id,
                                   "source_class_name":source_table_metadata['source_class_name'],
                               "human_readable":target_metadata["human_readable"],
                               }
                target_local_metadata= TargetTimeSerieLocalUpdate.create(metadata_kwargs=local_metadata_kwargs,
                                                             node_kwargs=local_node_kwargs
                                              )

            target_last_time_index_value=None
            if  target_metadata["sourcetableconfiguration"] is not None:
                target_last_time_index_value = target_metadata["sourcetableconfiguration"]['last_time_index_value']
                index_names=target_metadata["sourcetableconfiguration"]['index_names']
                time_index_name=target_metadata["sourcetableconfiguration"]['time_index_name']
            if target_last_time_index_value is not None:
                last_time_index_value = target_dth.request_to_datetime(string_date=target_last_time_index_value)
                if overwrite == False:
                    logger.debug(f"filtering time index {hash_id}  after {last_time_index_value} ")
                    #if no overwrite filter by last value
                    if len(index_names)>1:

                        data = data[data.index.get_level_values(time_index_name).floor("us") > last_time_index_value]
                    else:
                        data = data[data.index.floor("us") > last_time_index_value]

            if data.shape[0]==0:
                logger.info(f"No data in dates for {hash_id}")
                continue

            for c in data.columns:
                if any([t in c for t in JSON_COMPRESSED_PREFIX]) == True:
                    if isinstance(data[c].iloc[0],str):
                        data[c] = data[c].apply(lambda x: json.loads(x))

            logger.debug(f"Copying {source_table_metadata['hash_id']}")

            batch_size_limit,batch_size=10,1000
            if overwrite == True:
                batch_size_limit=data.shape[0]//batch_size+1

            target_dth.upsert_data_into_table(metadata=target_metadata,local_metadata=target_local_metadata,
                                                historical_update_id=None,
                                              batch_size_limit=batch_size_limit, batch_size=batch_size,
                                              data=data, overwrite=overwrite)
            updated_hids.append(hash_id)
            
        return updated_hids