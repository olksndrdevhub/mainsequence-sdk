from .utils import (TDAG_ENDPOINT, is_process_running, get_network_ip,
                    CONSTANTS,
                    DATE_FORMAT, get_authorization_headers, AuthLoaders, make_request, get_tdag_client_logger, set_types_in_table, parse_postgres_url)
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
import os


from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from typing import Optional, List, Dict, Any
from .data_sources_interfaces.local_data_lake import DataLakeInterface
from .data_sources_interfaces import timescale as TimeScaleInterface
from functools import wraps
import math
import gzip
import base64

_default_data_source = None  # Module-level cache
BACKEND_DETACHED=lambda : os.environ.get('BACKEND_DETACHED',"false").lower()=="true"


def none_if_backend_detached(func):
    """
    Decorator that evaluates BACKEND_DETACHED before executing the function.
    If BACKEND_DETACHED() returns True, the function is skipped, and None is returned.
    Otherwise, the function is executed as normal.

    It supports regular functions, property methods, classmethods, and staticmethods.
    """
    # Handle property methods
    if isinstance(func, property):
        getter = func.fget

        @wraps(getter)
        def wrapper_getter(*args, **kwargs):
            if BACKEND_DETACHED():
                return None
            return getter(*args, **kwargs)

        return property(wrapper_getter, func.fset, func.fdel, func.__doc__)

    # Handle classmethods
    elif isinstance(func, classmethod):
        original_func = func.__func__

        @wraps(original_func)
        def wrapper(*args, **kwargs):
            if BACKEND_DETACHED():
                return None
            return original_func(*args, **kwargs)

        return classmethod(wrapper)

    # Handle staticmethods
    elif isinstance(func, staticmethod):
        original_func = func.__func__

        @wraps(original_func)
        def wrapper(*args, **kwargs):
            if BACKEND_DETACHED():
                return None
            return original_func(*args, **kwargs)

        return staticmethod(wrapper)

    # Handle regular functions
    else:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if BACKEND_DETACHED():
                return None
            return func(*args, **kwargs)

        return wrapper


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
get_local_time_serie_nodes_methods_url=lambda root_url:  root_url + "/ogm/api/local_time_serie"
get_local_time_serie_url=lambda root_url:  root_url + "/orm/api/local_time_serie"
get_local_time_serie_update_details=lambda root_url: root_url + "/orm/api/local_time_serie_update_details"
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

def build_session(loaders):
    from requests.adapters import HTTPAdapter, Retry
    s = requests.Session()
    s.headers.update(loaders.auth_headers)
    retries = Retry(total=2, backoff_factor=2, )
    s.mount('http://', HTTPAdapter(max_retries=retries))
    return s

session=build_session(loaders=loaders)

class BaseObject:
    LOADERS = loaders
    @classmethod
    def build_session(cls):
        # from requests.adapters import HTTPAdapter, Retry
        # s = requests.Session()
        # s.headers.update(cls.LOADERS.auth_headers)
        # retries = Retry(total=2, backoff_factor=2, )
        # s.mount('http://', HTTPAdapter(max_retries=retries))
        s=session
        return s

    @property
    def s(self):
        s = self.build_session()
        return s


    @none_if_backend_detached
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

    @none_if_backend_detached
    def delete(self):

        url = self.ROOT_URL + "/destroy/"
        payload = {"json":{"uid":self.uid}}
        s = self.build_session()
        r = make_request(s=s,loaders=self.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    @none_if_backend_detached
    def patch(self,*args,**kwargs):
        url = self.ROOT_URL + "/update/"
        payload = {"json": {"uid": self.uid, "patch_data":serialize_to_json(kwargs)}}
        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        return Scheduler(**r.json())

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





    @none_if_backend_detached
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
    def depends_on_connect_remote_table(cls, source_hash_id: str,
                           source_local_hash_id: str,
                           source_data_source_id: id,
                           target_data_source_id: id,
                           target_local_hash_id: str):
        """

        """
        s = cls.build_session()
        url = cls.ROOT_URL + "/depends_on_connect_remote_table/"
        payload = dict(json={"source_hash_id": source_hash_id,
                             "source_local_hash_id": source_local_hash_id,
                             "source_data_source_id": source_data_source_id,
                             "target_data_source_id":target_data_source_id,
                             "target_local_hash_id": target_local_hash_id,
                             })
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")

    @classmethod
    def depends_on_connect(cls,  target_class_name: str,
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
        payload = dict(json={ "target_class_name": target_class_name,
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
    def remove_head_from_all_schedulers(cls, hash_id):
        url = cls.ROOT_URL + f"/{hash_id}/remove_head_from_all_schedulers/"
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="PATCH", url=url, )
        if r.status_code != 200:
            raise Exception(r.text)

    @none_if_backend_detached
    @classmethod
    def patch_build_configuration(cls, remote_table_patch: Union[dict,None],
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


class SourceTableConfiguration(BaseTdagPydanticModel, BaseObject):
    related_table: Union[int,"DynamicTableMetaData"]
    time_index_name: str = Field(..., max_length=100, description="Time index name")
    column_dtypes_map: Dict[str, Any] = Field(..., description="Column data types map")
    index_names: List
    column_index_names: List
    last_time_index_value: Optional[datetime.datetime] = Field(None, description="Last time index value")
    earliest_index_value: Optional[datetime.datetime] = Field(None, description="Earliest index value")
    multi_index_stats: Optional[Dict[str, Any]] = Field(None, description="Multi-index statistics JSON field")
    table_partition: Dict[str, Any] = Field(..., description="Table partition settings")

    @classmethod
    def create(cls, *args, **kwargs):
        url = TDAG_ENDPOINT +"/orm/api" + "/source_table_config"
        data = cls.serialize_for_json(kwargs)
        payload = {"json": data, }
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            if r.status_code == 409:
                raise AlreadyExist(r.text)
            else:
                raise Exception(r.text)
        return cls(**r.json())


class DynamicTableMetaData(BaseTdagPydanticModel, BaseObject):
    id: int = Field(None, description="Primary key, auto-incremented ID")
    hash_id: str = Field(..., max_length=63, description="Max length of PostgreSQL table name")
    table_name: Optional[str] = Field(None, max_length=63, description="Max length of PostgreSQL table name")
    creation_date: datetime.datetime = Field(..., description="Creation timestamp")
    created_by_user_id: Optional[int] = Field(None, description="Foreign key reference to AUTH_USER_MODEL")
    organization_owner: int = Field(None, description="Foreign key reference to Organization")
    open_for_everyone: bool = Field(default=False, description="Whether the table is open for everyone")
    data_source_open_for_everyone: bool = Field(default=False,
                                                description="Whether the data source is open for everyone")
    build_configuration: Dict[str, Any] = Field(..., description="Configuration in JSON format")
    build_meta_data: Optional[Dict[str, Any]] = Field(None, description="Optional YAML metadata")
    human_readable: Optional[str] = Field(None, max_length=255, description="Human-readable description")
    time_serie_source_code_git_hash: Optional[str] = Field(None, max_length=255,
                                                           description="Git hash of the time series source code")
    time_serie_source_code: Optional[str] = Field(None, description="File path for time series source code")
    protect_from_deletion: bool = Field(default=False, description="Flag to protect the record from deletion")
    ogm_linked: bool = Field(default=False, description="OGM linked flag")
    data_source: Union[int,"DynamicTableDataSource"]
    source_class_name:str
    sourcetableconfiguration:Optional[SourceTableConfiguration]=None

    @property
    def ROOT_URL(self):
        return get_dynamic_table_metadata(TDAG_ENDPOINT)

    def patch(self, time_out: Union[None, int] = None, *args, **kwargs, ):
        url = self.ROOT_URL + f"/{self.id}/"
        payload = {"json": serialize_to_json(kwargs)}
        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload, time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

    @classmethod
    def get(cls, hash_id, data_source__id: int):

        result = cls.filter(payload={"hash_id": hash_id,
                                     "data_source__id": data_source__id,
                                     "detail": True})
        if len(result) != 1:
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
    def patch_by_hash(cls, hash_id: str, *args, **kwargs):
        metadata = cls.get(hash_id=hash_id)
        metadata.patch(*args, **kwargs)

    @classmethod
    def create(cls, **kwargs):
        """

        :return:
        :rtype:
        """

        kwargs = serialize_to_json(kwargs)

        url=get_ts_node_url(TDAG_ENDPOINT)+"/"

        payload = {"json": kwargs}
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            if r.status_code == 409:
                raise AlreadyExist(r.text)
            else:
                raise Exception(r.text)
        data = r.json()

        return cls(**data["metadata"])


class LocalTimeSerie(BaseTdagPydanticModel, BaseObject):

    id: Optional[int] = Field(None, description="Primary key, auto-incremented ID")
    local_hash_id: str = Field(..., max_length=63, description="Max length of PostgreSQL table name")
    remote_table: Union[int,DynamicTableMetaData]
    build_configuration: Dict[str, Any] = Field(..., description="Configuration in JSON format")
    build_meta_data: Optional[Dict[str, Any]] = Field(None, description="Optional YAML metadata")
    ogm_linked: bool = Field(default=False, description="OGM linked flag")
    ogm_dependencies_linked: bool = Field(default=False, description="OGM dependencies linked flag")
    tags: Optional[list[str]] = Field(default=[], description="List of tags")
    description: Optional[str] = Field(None, description="Optional HTML description")
    localtimeserieupdatedetails:Optional["LocalTimeSerieUpdateDetails"]=None
    run_configuration:"RunConfiguration"

    @classmethod
    def get_node_methods_url(self):
        return get_local_time_serie_nodes_methods_url(TDAG_ENDPOINT)


    @classmethod
    def get_root_url(self):
        return get_local_time_serie_url(TDAG_ENDPOINT)

    @classmethod
    def LOCAL_TIME_SERIE_HISTORICAL_UPDATE(self):
        return get_local_time_serie_historical_update_url(TDAG_ENDPOINT)

    @classmethod
    def create(cls,**kwargs):
        url=get_local_time_serie_nodes_methods_url()+"/"
        kwargs = serialize_to_json(kwargs)


        payload = {"json": kwargs}
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            if r.status_code == 409:
                raise AlreadyExist(r.text)
            else:
                raise Exception(r.text)
        data = r.json()

        return cls(**data["metadata"])

    @classmethod
    def add_tags(cls, local_metadata, tags: list, timeout=None):

        base_url = cls.get_root_url()
        s = cls.build_session()
        payload = {"json": {"tags": tags}}
        # r = self.s.get(, )
        url = f"{base_url}/{local_metadata['id']}/add_tags/"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="PATCH", url=url,
                         payload=payload,
                         time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.json()}")
        return r.json()


    def update_details_exist(self,timeout=None):

        base_url = self.get_root_url()
        s = self.build_session()

        # r = self.s.get(, )
        url = f"{base_url}/{self.id}/update_details_exist/"
        r = make_request(s=s, loaders=self.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.json()}")
        return r.json()

    @classmethod
    def filter_by_hash_id(cls, local_hash_id_list: list, timeout=None):
        s = cls.build_session()
        base_url = cls.get_root_url()
        url = f"{base_url}/filter_by_hash_id/"
        payload = {"json": {"local_hash_id__in": local_hash_id_list}, }
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"{r.text}")
        all_metadatas = {m["local_hash_id"]: m for m in r.json()}
        return all_metadatas


    def set_start_of_execution(self,**kwargs):
        s = self.build_session()
        base_url = self.get_root_url()

        payload = {"json": kwargs}
        # r = self.s.patch(, **payload)
        url = f"{base_url}/{self.id}/set_start_of_execution/"
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        result = r.json()
        if result["last_time_index_value"] is not None:
            result["last_time_index_value"] = datetime.datetime.fromtimestamp(result["last_time_index_value"]).replace(
                tzinfo=pytz.utc)

        result['update_statistics'] = {k: request_to_datetime(v) for k, v in result['update_statistics'].items()}
        result['update_statistics'] = DataUpdates(update_statistics=result['update_statistics'])
        return LocalTimeSeriesHistoricalUpdate(**result)

    def set_end_of_execution(self,
                             historical_update_id: int,
                             timeout=None, **kwargs):
        s = self.build_session()
        url = self.get_root_url() + f"/{self.id}/set_end_of_execution/"
        kwargs.update(dict(historical_update_id=historical_update_id))
        payload = {"json": kwargs}
        # r = self.s.patch(, **payload)
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request ")

    @classmethod
    def batch_set_end_of_execution(cls, update_map: dict, timeout=None):
        s = cls.build_session()
        url = f"{cls.get_root_url()}/batch_set_end_of_execution/"
        payload = {"json": {"update_map": update_map}}
        r = make_request(s=s, loaders=cls.LOADERS, r_type="PATCH", url=url, payload=payload, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request ")

    @classmethod
    def set_last_update_index_time(cls, metadata, timeout=None):
        s = cls.build_session()
        url =cls.get_root_url()+ f"/{metadata['id']}/set_last_update_index_time/"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)

        if r.status_code == 404:
            raise SourceTableConfigurationDoesNotExist

        if r.status_code != 200:
            raise Exception(f"{metadata['local_hash_id']}{r.text}")
        return r


    def set_last_update_index_time_from_update_stats(self,
                                                     last_time_index_value: float, max_per_asset_symbol,
                                                     timeout=None)->"LocalTimeSerie":
        s = self.build_session()
        url = self.get_root_url() + f"/{self.id}/set_last_update_index_time_from_update_stats/"
        payload = {
            "json": {"last_time_index_value": last_time_index_value, "max_per_asset_symbol": max_per_asset_symbol}}
        r = make_request(s=s, loaders=self.LOADERS, payload=payload, r_type="GET", url=url, time_out=timeout)

        if r.status_code == 404:
            raise SourceTableConfigurationDoesNotExist

        if r.status_code != 200:
            raise Exception(f"{self.local_hash_id}{r.text}")
        return LocalTimeSerie(**r.json())



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
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=f"{base_url}/", payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.url} {r.text}")

    @classmethod
    def get_mermaid_dependency_diagram(cls, local_hash_id, data_source_id, desc=True, timeout=None) -> dict:
        """

        :param local_hash_id:
        :return:
        """
        s = cls.build_session()
        url = cls.get_node_methods_url() + f"/{local_hash_id}/dependencies_graph_mermaid?desc={desc}&data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url,
                         time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r.json()

    # Node Updates
    @classmethod
    def get_all_dependencies(cls, hash_id, data_source_id, timeout=None):
        s = cls.build_session()
        url = cls.get_node_methods_url() + f"/{hash_id}/get_all_dependencies?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df


    def get_all_dependencies_update_priority(self,timeout=None)->pd.DataFrame:
        s = self.build_session()
        url = self.get_node_methods_url() + f"/{self.id}/get_all_dependencies_update_priority"
        r = make_request(s=s, loaders=self.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df

    @classmethod
    def get_max_depth(cls, hash_id, data_source_id, timeout=None):
        s = cls.build_session()
        url = cls.get_node_methods_url() + f"/{hash_id}/get_max_depth?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r.json()["max_depth"]

    @classmethod
    def get_upstream_nodes(cls, hash_id, data_source_id, timeout=None):
        s = cls.build_session()
        url = cls.get_node_methods_url() + f"/{hash_id}/get_upstream_nodes?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, time_out=timeout)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        depth_df = pd.DataFrame(r.json())
        return depth_df

    @classmethod
    def create(cls, timeout=None, *args, **kwargs):
        url = cls.get_node_methods_url() + "/"
        payload = {"json": serialize_to_json(kwargs)}
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload=payload, time_out=timeout)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        instance = cls(**r.json())
        return r.json()

    @classmethod
    def get(cls, *args, **kwargs):
        if "id" in kwargs:
            url = cls.get_root_url() + f"/{kwargs['id']}/"
            s = cls.build_session()
            r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url)
            return cls(**r.json())

        result = cls.filter(**kwargs)
        if len(result) > 1:
            raise Exception("More than 1 return")
        if len(result) == 0:
            return None
        return result[0]

    @classmethod
    def filter(cls, *args, **kwargs):
        url = cls.get_root_url()

        payload = {} if kwargs is None else {"params": kwargs}

        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, payload=payload)
        if r.status_code == 404:
            raise LocalTimeSeriesDoesNotExist
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        return [cls(**i) for i in r.json()]

    @classmethod
    def set_ogm_dependencies_linked(cls, hash_id, data_source_id):
        s = cls.build_session()
        url = cls.get_root_url() + f"/{hash_id}/set_ogm_dependencies_linked?data_source_id={data_source_id}"
        r = make_request(s=s, loaders=cls.LOADERS, r_type="GET", url=url, )
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")

        return r

    @classmethod
    def verify_if_direct_dependencies_are_updated(cls, id):

        s = cls.build_session()
        url = cls.LOCAL_UPDATE_URL + f"/{id}/verify_if_direct_dependencies_are_updated/"
        r = make_request(s=s, loaders=None, r_type="GET", url=url)
        if r.status_code != 200:
            raise Exception(f"Error in request: {r.text}")
        return r.json()

    @classmethod
    def get_data_between_dates_from_api(cls, local_hash_id: str, data_source_id: int, start_date: datetime.datetime,
                                        end_date: datetime.datetime, great_or_equal: bool,
                                        less_or_equal: bool,
                                        asset_symbols: list,
                                        columns: list,
                                        execution_venue_symbols: list,
                                        symbol_range_map: Union[None, dict]
                                        ):
        s = cls.build_session()
        url = cls.LOCAL_UPDATE_URL + f"/get_data_between_dates_from_remote/"

        symbol_range_map = copy.deepcopy(symbol_range_map)
        if symbol_range_map is not None:
            for symbol, date_info in symbol_range_map.items():
                # Convert start_date if present
                if 'start_date' in date_info and isinstance(date_info['start_date'], datetime.datetime):
                    date_info['start_date'] = int(date_info['start_date'].timestamp())

                # Convert end_date if present
                if 'end_date' in date_info and isinstance(date_info['end_date'], datetime.datetime):
                    date_info['end_date'] = int(date_info['end_date'].timestamp())

        payload = {"json": {
            "local_hash_id": local_hash_id,
            "data_source_id": data_source_id,
            "start_date": start_date.timestamp() if start_date else None,
            "end_date": end_date.timestamp() if end_date else None,
            "great_or_equal": great_or_equal,
            "less_or_equal": less_or_equal,
            "asset_symbols": asset_symbols,
            "columns": columns,
            "execution_venue_symbols": execution_venue_symbols,
            "offset": 0,  # Will increase in each loop
            "symbol_range_map": symbol_range_map
        }}
        all_results = []
        while True:
            # Make the POST request
            r = make_request(s=s, loaders=None, payload=payload, r_type="POST", url=url)

            if r.status_code != 200:
                raise Exception(f"Error in request: {r.text}")

            response_data = r.json()

            # Accumulate results
            chunk = response_data.get("results", [])
            all_results.extend(chunk)

            # Retrieve next offset; if None, we've got all the data
            next_offset = response_data.get("next_offset")
            if not next_offset:
                break

            # Update payload with the new offset
            payload["json"]["offset"] = next_offset

        return pd.DataFrame(all_results)

    @classmethod
    def post_data_frame_in_chunks(cls,
                                  serialized_data_frame: pd.DataFrame, logger: object,
                                  chunk_size: int = 50_000,
                                  local_metadata: dict = None,
                                  data_source: str = None,
                                  index_names: list = None,
                                  time_index_name: str = 'timestamp',
                                  overwrite: bool = False,
                                  JSON_COMPRESSED_PREFIX: str = "base64-gzip",
                                  session: requests.Session = None

                                  ):
        """
            Sends a large DataFrame to a Django backend in multiple chunks.

            :param serialized_data_frame: The DataFrame to upload.
            :param url: The endpoint URL (e.g. https://yourapi.com/upload-chunk/).
            :param chunk_size: Number of rows per chunk.
            :param local_metadata: General metadata dict you want to send with each chunk.
            :param data_source: Additional info about the source of the data.
            :param index_names: Index columns in the DataFrame.
            :param time_index_name: The column name used for time indexing.
            :param overwrite: Boolean indicating whether existing data should be overwritten.
            :param JSON_COMPRESSED_PREFIX: String indicating the compression scheme in your JSON payload.
            :param session: Optional requests.Session() for connection reuse.
            """
        s = cls.build_session()
        url = cls.LOCAL_UPDATE_URL + f"/{local_metadata['id']}/insert_data_into_table/"
        total_rows = len(serialized_data_frame)
        total_chunks = math.ceil(total_rows / chunk_size)
        logger.info(f"Starting upload of {total_rows} rows in {total_chunks} chunk(s).")
        for i in range(total_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, total_rows)

            # Slice the DataFrame for the current chunk
            chunk_df = serialized_data_frame.iloc[start_idx:end_idx]

            # Compute grouped_dates for this chunk
            chunk_stats, _ = get_chunk_stats(chunk_df=chunk_df, index_names=index_names,
                                             time_index_name=time_index_name)

            # Convert the chunk to JSON
            chunk_json_str = chunk_df.to_json(orient="records", date_format="iso")

            # (Optional) Compress JSON using gzip then base64-encode
            compressed = gzip.compress(chunk_json_str.encode('utf-8'))
            compressed_b64 = base64.b64encode(compressed).decode('utf-8')

            payload = dict(json={
                "data": compressed_b64,  # compressed JSON data
                "chunk_stats": chunk_stats,
                "overwrite": overwrite,
                "chunk_index": i,
                "total_chunks": total_chunks,
            })
            try:
                r = make_request(s=s, loaders=None, payload=payload, r_type="POST", url=url, time_out=60 * 15)
                r.raise_for_status()  # Raise if 4xx/5xx
                logger.info(f"Chunk {i + 1}/{total_chunks} uploaded successfully.")
            except requests.exceptions.RequestException as e:
                logger.exception(f"Error uploading chunk {i + 1}/{total_chunks}: {e}")
                # Optionally, you could retry or break here
                raise e
            if r.status_code not in [200, 204]:
                raise Exception(r.text)

    @classmethod
    def get_metadatas_and_set_updates(cls,local_hash_id__in,multi_index_asset_symbols_filter,
                                      update_details_kwargs,update_priority_dict):
        """
        {'local_hash_id__in': [{'local_hash_id': 'alpacaequitybarstest_97018e7280c1bad321b3f4153cc7e986', 'data_source_id': 1},
        :param local_hash_id__in:
        :param multi_index_asset_symbols_filter:
        :param update_details_kwargs:
        :param update_priority_dict:
        :return:
        """
        import ast

        base_url = cls.get_root_url()
        s = cls.build_session()
        payload = { "json": dict(local_hash_id__in=local_hash_id__in,
                                 multi_index_asset_symbols_filter=multi_index_asset_symbols_filter,
                                 update_details_kwargs=update_details_kwargs,
                                 update_priority_dict=update_priority_dict,
                                 )}
        # r = self.s.post(f"{base_url}/get_metadatas_and_set_updates/", **payload)
        url = f"{base_url}/get_metadatas_and_set_updates/"
        r = make_request(s=s, loaders=cls.LOADERS,r_type="POST", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
        r = r.json()
        r["source_table_config_map"] = {int(k): SourceTableConfiguration(**v) for k, v in r["source_table_config_map"].items()}
        r["state_data"] = {int(k): LocalTimeSerieUpdateDetails(**v) for k, v in r["state_data"].items()}
        r["all_index_stats"] = {int(k): v for k, v in r["all_index_stats"].items()}
        r["local_metadatas"]=[LocalTimeSerie(**v) for v in r["local_metadatas"]]

        return r

class Scheduler(BaseTdagPydanticModel,BaseObject):
    uid: str
    name: str
    is_running: bool
    running_process_pid: Optional[int]
    running_in_debug_mode: bool
    updates_halted:bool
    host: Optional[str]
    api_address: Optional[str]
    api_port: Optional[int]
    pre_loads_in_tree: Optional[List[str]]=None  # Assuming this is a list of strings
    in_active_tree: Optional[List[LocalTimeSerieNode]  ]=None  # Assuming this is a list of strings
    schedules_to: Optional[List[LocalTimeSerieNode]]=None
    #for heartbeat
    _stop_heart_beat:bool=False
    _executor:Optional[object]=None

    @classmethod
    @property
    def ROOT_URL(cls):
        return  get_scheduler_node_url(TDAG_ENDPOINT)

    @none_if_backend_detached
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

    @none_if_backend_detached
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

    @none_if_backend_detached
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
                                name_suffix:Union[str,None]=None,):

        if BACKEND_DETACHED() == True:
            return cls(name=f"Detached scheduler for {local_hash_id}",
                       uid="DETACHED",running_process_pid=0,host="SDf",api_address="Sdf",
                       api_port=0,
                       running_in_debug_mode=True,updates_to=local_hash_id,
                       is_running=True)
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
                               remove_from_other_schedulers=True,**kwargs):

        if BACKEND_DETACHED() == True:
            raise Exception("TDAG is detached")

        s = cls.build_session( )

        url = cls.ROOT_URL + "/build_and_assign_to_ts/"
        payload = dict(json={
            "scheduler_name": scheduler_name,
            "delink_all_ts": delink_all_ts,
            "hash_id_list": local_hash_id_list,
            "remove_from_other_schedulers": remove_from_other_schedulers,
            "scheduler_kwargs":kwargs
        })
        r = make_request(s=s,loaders=cls.LOADERS, r_type="POST", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")
        scheduler = cls(**r.json())
        return scheduler

    def in_active_tree_connect(self,hash_id_list:list):
        if BACKEND_DETACHED() == True:
            self.in_active_tree=hash_id_list
            return None
        s = self.build_session()
        url = self.ROOT_URL + f"/{self.uid}/in_active_tree_connect/"
        payload = dict(json={"hash_id_list": hash_id_list})
        r = make_request(s=s, loaders=self.LOADERS,r_type="PATCH", url=url, payload=payload)
        if r.status_code != 201:
            raise Exception(f"Error in request {r.text}")

    def assign_to_scheduler(self,hash_id_list:list):
        if BACKEND_DETACHED() == True:
            self.schedules_to=hash_id_list
            return self
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

    def _heart_beat_patch(self):
        from mainsequence.tdag_client.utils import get_network_ip
        import os
        try:
            scheduler = self.patch(is_running=True,
                                                  running_process_pid=os.getpid(),
                                                  running_in_debug_mode=self.running_in_debug_mode,
                                                  last_heart_beat=datetime.datetime.utcnow().replace(
                                                      tzinfo=pytz.utc).timestamp(),
                                                  )
            for field, value in scheduler.__dict__.items():
                setattr(self, field, value)
        except Exception as e:
            logger.error(e)
    def _heartbeat_runner(self,run_interval):
        """
        Runs forever (until the main thread ends),
        calling _scheduler_heart_beat_patch every 30 seconds.
        """
        logger.info("Heartbeat thread started with interval = %d seconds", run_interval)

        while  True:

            self._heart_beat_patch()
            # Sleep in a loop so that if we ever decide to
            # add a cancellation event, we can check it in smaller intervals
            for _ in range(run_interval):
                # could check for a stop event here if not daemon
                if self._stop_heart_beat == True:
                    return
                time.sleep(1)

    def start_heart_beat(self):
        from concurrent.futures import ThreadPoolExecutor

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        run_interval = CONSTANTS.SCHEDULER_HEART_BEAT_FREQUENCY_SECONDS

        self._heartbeat_future = self._executor.submit(self._heartbeat_runner, run_interval)

    def stop_heart_beat(self):
        """
        Stop the heartbeat gracefully.
        """
        # Signal the runner loop to exit
        self._stop_heart_beat = True

        # Optionally wait for the future to complete
        if hasattr(self, "heartbeat_future") and self._heartbeat_future:
            logger.info("Waiting for the heartbeat thread to finish...")
            self._heartbeat_future.result()  # or .cancel() if you prefer

        # Shut down the executor if no longer needed
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        logger.info("Heartbeat thread stopped.")


    def patch(self,time_out, *args, **kwargs):
        url = self.ROOT_URL + f"/{self.id}/"
        payload = {"json": serialize_to_json(kwargs)}
        s = self.build_session()
        r = make_request(s=s, loaders=self.LOADERS, r_type="PATCH", url=url, payload=payload,time_out=time_out)
        if r.status_code != 200:
            raise Exception(f"Error in request {r.text}")
    

class RunConfiguration(BaseTdagPydanticModel, BaseObject):

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

class LocalTimeSerieUpdateDetails(BaseTdagPydanticModel,BaseObject):
    related_table: Union[int,LocalTimeSerie]
    active_update: bool = Field(default=False, description="Flag to indicate if update is active")
    update_pid: int = Field(default=0, description="Process ID of the update")
    error_on_last_update: bool = Field(default=False,
                                       description="Flag to indicate if there was an error in the last update")
    last_update: Optional[datetime.datetime] = Field(None, description="Timestamp of the last update")
    next_update: Optional[datetime.datetime] = Field(None, description="Timestamp of the next update")
    update_statistics: Optional[Dict[str, Any]] = Field(None, description="JSON field for update statistics")
    active_update_status: str = Field(default="Q", max_length=20, description="Current update status")
    active_update_scheduler_uid: Optional[str] = Field(None, max_length=100,
                                                       description="Scheduler UID for active update")
    update_priority: int = Field(default=0, description="Priority level of the update")
    direct_dependencies_ids: List[int] = Field(default=[], description="List of direct upstream dependencies IDs")
    last_updated_by_user_id: Optional[int] = Field(None, description="Foreign key reference to AUTH_USER_MODEL")
    def ROOT_URL(self):
        return  get_local_time_serie_update_details(TDAG_ENDPOINT)

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


class DataUpdates(BaseTdagPydanticModel):
    """
    TODO WIP Helper function to work with the table updates
    """
    update_statistics: Optional[Dict[str, Any]]

    def get_min_latest_value(self, init_fallback_date: datetime=None):
        if not self.update_statistics:
            return init_fallback_date
        return min(self.update_statistics.values())

    def get_max_latest_value(self, init_fallback_date: datetime=None):
        if not self.update_statistics:
            return init_fallback_date
        return min(self.update_statistics.values())

    def asset_identifier(self):
        return list(self.update_statistics.keys())

    def update_assets(self, asset_list: list, init_fallback_date: datetime=None):
        new_update_statistics = {}
        for a in asset_list:
            unique_identifier = a.unique_identifier
            if self.update_statistics and unique_identifier in self.update_statistics:
                new_update_statistics[unique_identifier] = self.update_statistics[unique_identifier]
            else:
                if init_fallback_date is None: raise ValueError(f"No initial start date for {a.unique_identifier} assets defined")
                new_update_statistics[a.unique_identifier] = init_fallback_date
        return DataUpdates(update_statistics=new_update_statistics)

    def is_empty(self):
        return self.update_statistics is None or len(self.update_statistics) == 0

    def __getitem__(self, key: str) -> Any:
        if self.update_statistics is None:
            raise KeyError(f"{key} not found (update_statistics is None).")
        return self.update_statistics[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if self.update_statistics is None:
            self.update_statistics = {}
        self.update_statistics[key] = value

    def __delitem__(self, key: str) -> None:
        if not self.update_statistics or key not in self.update_statistics:
            raise KeyError(f"{key} not found in update_statistics.")
        del self.update_statistics[key]

    def __iter__(self):
        """Iterate over keys."""
        if self.update_statistics is None:
            return iter([])
        return iter(self.update_statistics)

    def __len__(self) -> int:
        if not self.update_statistics:
            return 0
        return len(self.update_statistics)

    def keys(self):
        if not self.update_statistics:
            return []
        return self.update_statistics.keys()

    def values(self):
        if not self.update_statistics:
            return []
        return self.update_statistics.values()

    def items(self):
        if not self.update_statistics:
            return []
        return self.update_statistics.items()

def get_chunk_stats(chunk_df,time_index_name,index_names):
    chunk_stats = {"_GLOBAL_": {"max": chunk_df[time_index_name].max().timestamp(),
                                "min": chunk_df[time_index_name].min().timestamp()}}

    grouped_dates = None
    if len(index_names) > 1:
        grouped_dates = chunk_df.groupby(["unique_identifier"])[
            time_index_name].agg(
            ["min", "max"])
        chunk_stats["_PER_ASSET_"] = {
            row["unique_identifier"]: {
                "max": row["max"].timestamp(),
                "min": row["min"].timestamp(),
            }
            for _, row in grouped_dates.reset_index().iterrows()
        }
    return chunk_stats, grouped_dates




class LocalTimeSeriesHistoricalUpdate(BaseTdagPydanticModel, BaseObject):
    id: Optional[int]=None
    related_table: int  # Assuming you're using the ID of the related table
    update_time_start: datetime.datetime
    update_time_end: Optional[datetime.datetime] = None
    error_on_update: bool = False
    trace_id: Optional[str] = Field(default=None, max_length=255)
    updated_by_user: Optional[int] = None  # Assuming you're using the ID of the user
    must_update:bool
    direct_dependencies_ids:List[int]
    last_time_index_value:Optional[datetime.datetime] = None
    update_statistics: DataUpdates

    @classmethod
    @property
    def ROOT_URL(cls):
        return None

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
    related_resource:Optional[Union[DataSource,int]]=None
    class Config:
        use_enum_values = True  # This ensures that enums are stored as their values (e.g., 'TEXT')
    def __str__(self):
        return f"{self.data_type}"

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
        data = r.json()
        DataClass = cls.get_class(data["data_type"])

        _default_data_source = DataClass(**r.json())
        return _default_data_source

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




    def persist_to_pickle(self, path):
        import cloudpickle
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as handle:
            cloudpickle.dump(self, handle)

class PodLocalLake(DataSource):
    id: Optional[int] = Field(None, description="The unique identifier of the Local Disk Source Lake")
    in_pod: int = Field(..., description="The ID of the related Pod Source")
    datalake_name: str = Field(..., max_length=255, description="The name of the data lake")
    datalake_end: Optional[datetime.datetime] = Field(None, description="The end time of the data lake")
    datalake_start: Optional[datetime.datetime] = Field(None, description="The start time of the data lake")
    nodes_to_get_from_db: Optional[Dict] = Field(None, description="Nodes to retrieve from the database as JSON")
    persist_logs_to_file: bool = Field(False, description="Whether to persist logs to a file")
    use_s3_if_available: bool = Field(False, description="Whether to use S3 if available")

class TimeScaleDB(DataSource):

    database_user : str
    password :str
    host : str
    database_name :str
    port :int

class LocalDiskSourceLake(DynamicTableDataSource):
    related_resource: PodLocalLake
    data_type: str=CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE

    @classmethod
    def get_or_create(cls, *args,**kwargs):
        url = cls.ROOT_URL + "/get_or_create/"
        for field in ['datalake_start', 'datalake_end']:
            if field in kwargs and kwargs[field] is not None:
                kwargs[field] = int(kwargs[field].timestamp())
        kwargs["data_type"]=CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE
        s = cls.build_session()
        r = make_request(s=s, loaders=cls.LOADERS, r_type="POST", url=url, payload={"json":kwargs})

        if r.status_code not in  [200, 201]:
            raise Exception(f"Error in request {r.text}")
        return cls(**r.json())

    def _insert_data_into_table(
            self,
            serialized_data_frame: pd.DataFrame,
            metadata,
            time_index_name: str,
            index_names: list,
            logger: object,
            *args,
            **kwargs,
    ):

        data_lake_interface = DataLakeInterface(data_lake_source=self, logger=logger)
        data_lake_interface.persist_datalake(
            serialized_data_frame,
            overwrite=True,
            time_index_name=time_index_name, index_names=index_names,
            table_name=metadata["table_name"]
        )

    def filter_by_assets_ranges(
        self,
        asset_ranges_map: dict,
        metadata: dict,
        *args,
        **kwargs
    ):
        table_name = metadata["table_name"]
        data_lake_interface = DataLakeInterface(data_lake_source=self, logger=logger)
        df = data_lake_interface.filter_by_assets_ranges(
            table_name=table_name,
            asset_ranges_map=asset_ranges_map,
        )
        return df

    def get_data_by_time_index(
        self,
        local_metadata: dict,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        great_or_equal: bool = True,
        less_or_equal: bool = True,
        columns: Optional[List[str]] = None,
        asset_symbols: Optional[List[str]] = None,
        execution_venue_symbols: Optional[List[str]] = None,
        logger: Optional[object] = None,
    ) -> pd.DataFrame:

        metadata = local_metadata["remote_table"]
        table_name = metadata["table_name"]
        data_lake_interface = DataLakeInterface(data_lake_source=self, logger=logger)

        filters = data_lake_interface.build_time_and_symbol_filter(
            start_date=start_date,
            end_date=end_date,
            great_or_equal=great_or_equal,
            less_or_equal=less_or_equal,
            asset_symbols=asset_symbols,
        )

        df = data_lake_interface.query_datalake(filters=filters, table_name=table_name)
        return df

class TimeScaleDBDataSource(DynamicTableDataSource):
    related_resource: Union[TimeScaleDB,int]
    data_type: str = CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB

    @property
    def has_direct_connection(self):
        if isinstance(self.related_resource, int):
            return False
        return True

    def get_connection_uri(self):
        if self.has_direct_connection== False:
            raise Exception("This Data source does not have direct access")
        password = self.related_resource.password  # Decrypt password if necessary
        return f"postgresql://{self.related_resource.database_user}:{password}@{self.related_resource.host}:{self.related_resource.port}/{self.related_resource.database_name}"

    def _insert_data_into_table(
            self,
            serialized_data_frame: pd.DataFrame,
            metadata,
            local_metadata: dict,
            overwrite: bool,
            time_index_name: str,
            index_names: list,
            grouped_dates: dict,
            logger: object,
            *args,
            **kwargs,
    ):
        if BACKEND_DETACHED() == True:
            return None

        if not self.has_direct_connection :
            # Do API insertion
            LocalTimeSerie.post_data_frame_in_chunks(
                serialized_data_frame=serialized_data_frame,
                logger=logger,
                local_metadata=local_metadata,
                data_source=self,
                index_names=index_names,
                time_index_name=time_index_name,
                overwrite=overwrite,
            )
        else:
            TimeScaleInterface.process_and_update_table(
                serialized_data_frame=serialized_data_frame,
                metadata=metadata,
                grouped_dates=grouped_dates,
                data_source=self,
                index_names=index_names,
                time_index_name=time_index_name,
                overwrite=overwrite,
                JSON_COMPRESSED_PREFIX=JSON_COMPRESSED_PREFIX,
                logger=logger
            )

    def filter_by_assets_ranges(
            self,
            asset_ranges_map: dict,
            metadata: dict,
            local_hash_id: str,
            *args,
            **kwargs
    ):
        table_name = metadata.table_name
        index_names = metadata.sourcetableconfiguration.index_names
        column_types = metadata.sourcetableconfiguration.column_dtypes_map
        if self.has_direct_connection:
            df = TimeScaleInterface.filter_by_assets_ranges(
                table_name=table_name,
                asset_ranges_map=asset_ranges_map,
                index_names=index_names,
                data_source=self,
                column_types=column_types
            )
        else:
            df = TimeSerieLocalUpdate.get_data_between_dates_from_api(
                local_hash_id=local_hash_id,
                data_source_id=self.id,
                start_date=None,
                end_date=None,
                great_or_equal=True,
                less_or_equal=True,
                asset_symbols=None,
                columns=None,
                execution_venue_symbols=None,
                symbol_range_map=asset_ranges_map,  # <-- key for applying ranges
            )
        return df

    def get_data_by_time_index(
        self,
        local_metadata: dict,
        start_date: Optional[datetime.datetime] = None,
        end_date: Optional[datetime.datetime] = None,
        great_or_equal: bool = True,
        less_or_equal: bool = True,
        columns: Optional[List[str]] = None,
        asset_symbols: Optional[List[str]] = None,
        execution_venue_symbols: Optional[List[str]] = None,
        logger: Optional[object] = None,
    ) -> pd.DataFrame:

        metadata = local_metadata["remote_table"]  # e.g. from your usage
        stc = metadata["sourcetableconfiguration"]

        if self.has_direct_connection:
            df = TimeScaleInterface.direct_data_from_db(
                metadata=metadata,
                connection_uri=self.get_connection_uri(),
                start_date=start_date,
                end_date=end_date,
                great_or_equal=great_or_equal,
                less_or_equal=less_or_equal,
                columns=columns,
                asset_symbols=asset_symbols,
            )
            df = set_types_in_table(df, stc["column_dtypes_map"])
            return df
        else:
            df = TimeSerieLocalUpdate.get_data_between_dates_from_api(
                local_hash_id=local_metadata["local_hash_id"],
                data_source_id=metadata["data_source"]["id"],
                start_date=start_date,
                end_date=end_date,
                great_or_equal=great_or_equal,
                less_or_equal=less_or_equal,
                asset_symbols=asset_symbols,
                columns=columns,
                symbol_range_map=None,  # pass a custom map if needed
            )
            if len(df) == 0:
                if logger:
                    logger.warning(
                        f"No data returned from remote API for {local_metadata['local_hash_id']}"
                    )
                return df

            stc = local_metadata["remote_table"]["sourcetableconfiguration"]
            df[stc["time_index_name"]] = pd.to_datetime(df[stc["time_index_name"]])
            for c, c_type in stc["column_dtypes_map"].items():
                if c != stc["time_index_name"]:
                    if c_type == "object":
                        c_type = "str"
                    df[c] = df[c].astype(c_type)
            df = df.set_index(stc["index_names"])
            return df

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
        if BACKEND_DETACHED() == True:
            return {}
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




    def get_update_statistics(self, hash_id):
        """
        Gets latest value from Hash_id
        :param hash_id:
        :type hash_id:
        :return:
        :rtype:i
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





    def filter_by_hash_id(self, hash_id_list: list):

        base_url = self.root_url
        url = f"{base_url}/filter_by_hash_id/"
        payload = {"json": {"hash_id__in": hash_id_list}, }
        r = self.make_request(r_type="POST", url=url, payload=payload)
        if r.status_code != 200:
            raise Exception(f"{r.text}")
        all_metadatas = {m["hash_id"]: m for m in r.json()}
        return all_metadatas

    @none_if_backend_detached
    @classmethod
    def _handle_source_table_configuration(cls,
            metadata:DynamicTableMetaData,
            column_dtypes_map,
            index_names,
            time_index_name,
            column_index_names,
            data,
            overwrite=False
    ):
        """
        Handles the creation or retrieval of the source table configuration.

        Parameters:
        ----------
        metadata : dict
            Metadata dictionary containing "sourcetableconfiguration" and "id".
        column_dtypes_map : dict
            Mapping of column names to their data types.
        index_names : list
            List of index names.
        time_index_name : str
            Name of the time index column.
        column_index_names : list
            List of column index names.
        data : DataFrame
            The input DataFrame.
        overwrite : bool, optional
            Whether to overwrite existing configurations (default is False).

        Returns:
        -------
        dict or None
            Updated metadata with the source table configuration, and potentially filtered data.
        """

        stc =metadata.sourcetableconfiguration

        if stc is None:
            try:
                stc = SourceTableConfiguration.create(
                    column_dtypes_map=column_dtypes_map,
                    index_names=index_names,
                    time_index_name=time_index_name,
                    column_index_names=column_index_names,
                    metadata_id=metadata["id"]
                )
                metadata.sourcetableconfiguration = stc
            except AlreadyExist:

                if not overwrite:
                    raise NotImplementedError("TODO Needs to remove values per asset")
                    # Filter the data based on time_index_name and last_time_index_value
                    data = data[
                        data[time_index_name] > self.request_to_datetime(stc.last_time_index_value)
                        ]
        return metadata, data

    @classmethod
    def upsert_data_into_table(cls,metadata:dict,
                               local_metadata:dict,
                               historical_update_id:Union[int,None],
                               data: pd.DataFrame,
                               overwrite: bool,
                               data_source:DynamicTableDataSource,
                               logger=logger
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
        overwrite = True #ALWAYS OVERWRITE

        data, column_index_names, index_names, column_dtypes_map, time_index_name = cls._break_pandas_dataframe(
            data)

        #overwrite data origina data frame to release memory
        if not data[time_index_name].is_monotonic_increasing:
            data = data.sort_values(time_index_name)


        metadata, data = (result if (result:=cls._handle_source_table_configuration(metadata=metadata, column_dtypes_map=column_dtypes_map,
                                                          index_names=index_names,
                                                          time_index_name=time_index_name,
                                                          column_index_names=column_index_names, data=data,
                                                          overwrite=overwrite
                                                          )

                          )  is not None else (metadata, data))

        duplicates_exist = data.duplicated(subset=index_names).any()
        assert not duplicates_exist, f"Duplicates found in columns: {index_names}"


        global_stats, grouped_dates = get_chunk_stats(
            chunk_df=data,
            index_names=index_names,
            time_index_name=time_index_name
        )

        data_source._insert_data_into_table(
            serialized_data_frame=data,
            metadata=metadata,
            local_metadata=local_metadata,
            overwrite=overwrite,
            time_index_name=time_index_name,
            index_names=index_names,
            historical_update_id=historical_update_id,
            logger=logger,
            global_stats=global_stats,
            grouped_dates=grouped_dates
        )

        if BACKEND_DETACHED() == True:
            return None

        min_d, last_time_index_value = global_stats["_GLOBAL_"]["min"], global_stats["_GLOBAL_"]["max"]
        max_per_asset_symbol = None
        if len(index_names) > 1:
            max_per_asset_symbol = {
                unique_identifier: stats["max"] for unique_identifier, stats in global_stats["_PER_ASSET_"].items()
            }
        local_metadata = local_metadata.set_last_update_index_time_from_update_stats(
            max_per_asset_symbol=max_per_asset_symbol,
            last_time_index_value=last_time_index_value,
        )


        return local_metadata

    def filter_by_assets_ranges(self,
                                metadata: dict,
                                asset_ranges_map:dict,
                                data_source:object,
                                local_hash_id: str
    ):
        df = data_source.filter_by_assets_ranges(
            metadata=metadata,
            asset_ranges_map=asset_ranges_map,
            data_source=data_source,
            local_hash_id=local_hash_id,
        )
        return df

    def get_data_by_time_index(
            self,
            local_metadata: Union[dict,str],
            data_source:object,
            start_date: Union[datetime.datetime, None] = None,
            great_or_equal: bool = True,
            less_or_equal: bool = True,
            end_date: Union[datetime.datetime, None] = None,
            columns: Union[list, None] = None,
            asset_symbols:Union[list,None]=None,
    ):
        return data_source.get_data_by_time_index(
            local_metadata=local_metadata,
            start_date=start_date,
            end_date=end_date,
            great_or_equal=great_or_equal,
            less_or_equal=less_or_equal,
            columns=columns,
            asset_symbols=asset_symbols,
            logger=self.logger
        )

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
        r = TimeSerieNode.set_policy_for_descendants(hash_id,policy,pol_type,exclude_ids,extend_to_classes)

    def build_or_update_update_details(self, metadata, *args, **kwargs):

        base_url = self.root_url
        payload = { "json": kwargs}
        # r = self.s.patch(, **payload)
        url=f"{base_url}/{metadata['id']}/build_or_update_update_details/?data_source_id={metadata['data_source']['id']}"
        r=self.make_request(r_type="PATCH",url=url,payload=payload)
        if r.status_code != 202:
            raise Exception(f"Error in request {r.text}")
        return r.json()

    @none_if_backend_detached
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
    def depends_on_connect_remote_table(self, source_hash_id: str,
                           source_local_hash_id: str,
                           source_data_source_id: id,
                           target_data_source_id: id,
                           target_local_hash_id: str):
        TimeSerieNode.depends_on_connect_remote_table(source_hash_id=source_hash_id,
                                         source_local_hash_id=source_local_hash_id,
                                         source_data_source_id=source_data_source_id,
                                         target_data_source_id=target_data_source_id,
                                         target_local_hash_id=target_local_hash_id)

    def depends_on_connect(self, target_class_name:str,
                           source_local_hash_id:str,
                           target_local_hash_id:str,
                           source_data_source_id:id,
                            target_data_source_id:id,
                           target_human_readable:str):


        TimeSerieNode.depends_on_connect(
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