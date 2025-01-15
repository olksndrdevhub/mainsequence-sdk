import datetime
import os
from venv import logger

import numpy as np
import pandas as pd
from typing import Dict, Any, List
import json
import time
import traceback
import pytz
import inspect
import logging
import copy
import hashlib
import importlib
import cloudpickle
from pathlib import Path
from mainsequence.tdag.instrumentation import tracer, tracer_instrumentator
from mainsequence.tdag.logconf import create_logger_in_path
from mainsequence.tdag.config import (
    ogm
)

from mainsequence.tdag.time_series.persist_managers import PersistManager, DataLakePersistManager
from mainsequence.tdag_client.models import none_if_backend_detached, DataSource
from numpy.f2py.auxfuncs import isint1

from pycares.errno import value
from pydantic import BaseModel

from abc import ABC

from typing import Union
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from mainsequence.tdag_client import TimeSerieLocalUpdate, LocalTimeSerieUpdateDetails, CONSTANTS, \
    DynamicTableDataSource
from enum import Enum
from functools import wraps
from mainsequence.tdag.config import bcolors
from mainsequence.tdag.time_series.update.models import StartUpdateDataInfo
from mainsequence.tdag.logconf import get_tdag_logger

logger = get_tdag_logger()


def serialize_model_list(value):
    value.sort(key=lambda x: x.unique_identifier, reverse=False)
    new_value = {"is_model_list": True}
    new_value['model_list'] = [v.to_serialized_dict() for v in value]
    value = new_value
    return value


def rebuild_with_type(value, rebuild_function):
    if value["__type__"] == "tuple":
        return tuple([rebuild_function(c) for c in value["items"]])
    else:
        raise NotImplementedError


from mainsequence.vam_client.models_helpers import get_model_class

build_model = lambda model_data: get_model_class(model_data["orm_class"])(**model_data)


def parse_dictionary_before_hashing(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    local_ts_dict_to_hash = {}
    for key, value in dictionary.items():
        if key != "build_meta_data":
            local_ts_dict_to_hash[key] = value
            if isinstance(value, dict):
                if "orm_class" in value.keys():

                    local_ts_dict_to_hash[key] = value['unique_identifier']

                elif "is_time_series_config" in value.keys():
                    tmp_local_ts, remote_ts = hash_signature(value["config_data"])
                    local_ts_dict_to_hash[key] = {"is_time_series_config": value["is_time_series_config"],
                                                  "config_data": tmp_local_ts}

                elif "is_model_list" in value.keys():  # do not inclue
                    local_ts_dict_to_hash[key] = {"is_model_list": value["is_model_list"],
                                                  "config_data": [v["unique_identifier"] for v in value['model_list']]}
                else:
                    # recursively apply hash signature
                    local_ts_dict_to_hash[key] = parse_dictionary_before_hashing(value)

            if isinstance(value, list):
                if len(value) == 0:
                    local_ts_dict_to_hash[key] = value
                else:
                    if isinstance(value[0], dict):
                        if "orm_class" in value[0].keys():
                            try:
                                new_list = [v["unique_identifier"] for v in value]
                            except Exception as e:
                                raise e
                            local_ts_dict_to_hash[key] = new_list
                            raise Exception("Use ModelList")

    return local_ts_dict_to_hash


def hash_signature(dictionary: Dict[str, Any]) -> str:
    """
    MD5 hash of a dictionary used to hash the local annd remote configuration of tables
    :param dictionary:
    :return:
    """
    import hashlib
    dhash_local = hashlib.md5()
    dhash_remote = hashlib.md5()

    local_ts_dict_to_hash = parse_dictionary_before_hashing(dictionary)

    remote_ts_in_db_hash = copy.deepcopy(local_ts_dict_to_hash)
    if "local_kwargs_to_ignore" in local_ts_dict_to_hash.keys():
        keys_to_ignore = local_ts_dict_to_hash['local_kwargs_to_ignore']
        keys_to_ignore.sort()
        for k in keys_to_ignore:
            remote_ts_in_db_hash.pop(k, None)
        remote_ts_in_db_hash.pop("local_kwargs_to_ignore", None)

    try:
        encoded_local = json.dumps(local_ts_dict_to_hash, sort_keys=True).encode()
        encoded_remote = json.dumps(remote_ts_in_db_hash, sort_keys=True).encode()
    except Exception as e:
        logger.error(dictionary)
        logger.error(traceback.format_exc())
        raise e
    dhash_local.update(encoded_local)
    dhash_remote.update(encoded_remote)
    return dhash_local.hexdigest(), dhash_remote.hexdigest()


class ConfigSerializer:

    @staticmethod
    def _serialize_model(model):
        columns = {"model": model.__class__.__name__, "id": model.unique_identifier}

        return columns

    @classmethod
    def rebuild_serialized_wrapper_dict(cls, time_series_dict_config: dict):
        """
        rebuilds configuration from time_series Wrapper
        :param time_series_dict_config:

        :return:
        """
        time_series_dict = {}
        for key, value in time_series_dict_config.items():
            new_ts = cls.rebuild_from_configuration(hash_id=value)
            time_series_dict[key] = new_ts

        return time_series_dict

    @classmethod
    def rebuild_pydantic_model(cls, details, state_kwargs: Union[None, dict] = None):
        """
        If there is an state rebuild the configuration then the method to rebuild related objects is from state
        Args:
            details: 
            state_kwargs: 

        Returns:

        """
        rebuild_function = lambda x, state_kwargs: cls._rebuild_configuration_argument(x, ignore_pydantic=False)
        if state_kwargs is not None:
            rebuild_function = lambda x, state_kwargs: cls.deserialize_pickle_value(x, **state_kwargs)

        module = importlib.import_module(details["pydantic_model_import_path"]["module"])
        PydanticClass = getattr(module, details["pydantic_model_import_path"]['qualname'])
        new_details = {}
        for arg, arg_value in details["serialized_model"].items():
            if isinstance(arg_value, dict):
                if "pydantic_model_import_path" in arg_value.keys():
                    arg_value = cls.rebuild_pydantic_model(arg_value,
                                                           state_kwargs=state_kwargs)
                elif "__type__" in arg_value.keys():
                    arg_value = rebuild_with_type(arg_value)
                else:
                    arg_value = {k: rebuild_function(v, state_kwargs=state_kwargs) for k, v in arg_value.items()}
            elif isinstance(arg_value, list):
                new_list = []
                for a in arg_value:

                    if isinstance(a, dict):
                        if "pydantic_model_import_path" in a.keys():
                            new_item = cls.rebuild_pydantic_model(a,
                                                                  state_kwargs=state_kwargs)
                        else:
                            new_item = rebuild_function(a, state_kwargs=state_kwargs)
                    else:
                        new_item = rebuild_function(a, state_kwargs=state_kwargs)
                    new_list.append(new_item)

                arg_value = new_list
            else:

                arg_value = rebuild_function(arg_value, state_kwargs=state_kwargs)

            new_details[arg] = arg_value
        try:

            value = PydanticClass(**new_details)
        except Exception as e:
            raise e

        return value

    @classmethod
    def rebuild_serialized_config(cls, config, time_serie_class_name):
        """
        rebulds configuration from config file, particularly Assets
        :param config:

        :param time_serie_class_name:
        :return:
        """

        config = cls.rebuild_config(config=config)
        if time_serie_class_name == "WrapperTimeSerie":
            config["time_series_dict"] = cls.rebuild_serialized_wrapper_dict(
                time_series_dict_config=config["time_series_dict"],
            )

        return config

    @classmethod
    def _rebuild_configuration_argument(cls, value, ignore_pydantic):
        """
        To be able to mix pydantic with VAM ORM models we need to add the posibility to recusevely ignore pydantic
        Args:
            value: 
            ignore_pydantic: 

        Returns:

        """

        if isinstance(value, dict):

            if "is_time_series_config" in value.keys():
                config_data = value["config_data"]
                new_config = cls.rebuild_config(config=config_data, )
                value = TimeSerieConfigKwargs(new_config)
            elif "is_model_list" in value.keys():
                value = ModelList([build_model(v) for v in value["model_list"]])
            elif "orm_class" in value.keys():
                value = build_model(value)
            elif "pydantic_model_import_path" in value.keys():
                value = cls.rebuild_config(value, ignore_pydantic=True)
                if ignore_pydantic == False:
                    value = cls.rebuild_pydantic_model(value)
            elif "__type__" in value.keys():
                value = rebuild_with_type(value, rebuild_function=lambda x: cls._rebuild_configuration_argument(x,
                                                                                                                ignore_pydantic=ignore_pydantic))
            else:
                value = cls.rebuild_config(config=value)


        elif isinstance(value, list):
            if len(value) == 0:
                return value
            value = [cls._rebuild_configuration_argument(v, ignore_pydantic=ignore_pydantic) for v in value]

        return value

    @classmethod
    def rebuild_config(cls, config, ignore_pydantic=False):
        """

        :param config:
        :return:
        """

        for key, value in config.items():
            config[key] = cls._rebuild_configuration_argument(value, ignore_pydantic)

        return config

    def _serialize_signature_argument(self, value, pickle_ts: bool):
        """
        Serializes signature argument
        """
        from types import SimpleNamespace
        from mainsequence.vam_client import BaseObjectOrm

        if issubclass(value.__class__, TimeSerie):
            if pickle_ts == True:
                new_value = {"is_time_serie_pickled": True}
                try:
                    full_path, relative_path = value.persist_to_pickle()
                    new_value["local_hash_id"] = value.local_hash_id
                    new_value['data_source_id'] = value.data_source_id
                except Exception as e:
                    raise e
                value = new_value
        elif issubclass(value.__class__, APITimeSerie):
            if pickle_ts == True:
                new_value = {"is_api_time_serie_pickled": True}
                try:
                    full_path, relative_path = value.persist_to_pickle()
                    new_value["local_hash_id"] = value.local_hash_id
                    new_value['data_source_id'] = value.data_source_id
                except Exception as e:
                    raise e
                value = new_value
        elif isinstance(value, SimpleNamespace):
            value = vars(value)
        elif isinstance(value, Enum):
            value = value.value
        elif isinstance(value, BaseModel) == True and issubclass(value.__class__, BaseObjectOrm) == False:

            import_path = {"module": value.__class__.__module__,
                           "qualname": value.__class__.__qualname__}

            serialized_model = {}
            for key, model_tmp_value in value.model_dump().items():  # model_dict.items():
                serialized_model[key] = self._serialize_signature_argument(model_tmp_value, pickle_ts)
            try:

                json.dumps(serialized_model)
            except Exception as e:
                raise e
            value = {"pydantic_model_import_path": import_path, "serialized_model": serialized_model}
        elif issubclass(value.__class__, BaseObjectOrm):
            value = value.to_serialized_dict()
        elif isinstance(value, tuple):
            tuple_list = []
            for v in value:
                new_v = v
                if isinstance(v, BaseModel):
                    new_v = self._serialize_signature_argument(v, pickle_ts=pickle_ts)
                else:
                    try:
                        new_v = self._serialize_signature_argument(v, pickle_ts=pickle_ts)
                    except Exception as e:
                        raise e
                tuple_list.append(new_v)
            new_value = {"__type__": "tuple", "items": tuple_list}
            value = new_value
        elif isinstance(value, list):
            if len(value) == 0:
                return []
            if isinstance(value, ModelList):
                value = serialize_model_list(value)
            else:
                if all(isinstance(member, BaseObjectOrm) for member in value):
                    value = serialize_model_list(value)
                elif issubclass(value[0].__class__, BaseObjectOrm):

                    raise Exception("Use Mixed Object Model ")
                elif issubclass(value[0].__class__, dict):
                    value = [self._serialize_configuration_dict(v) for v in value]
                elif isinstance(value[0], BaseModel):
                    value = [self._serialize_signature_argument(v, pickle_ts=pickle_ts) for v in value]
                else:
                    new_value = []
                    sort = True
                    for v in value:
                        if issubclass(v.__class__, dict):
                            if "__type__" in v.keys():
                                new_value.append(v)
                            else:
                                new_value.append(self._serialize_configuration_dict(v))
                            sort = False
                        elif isinstance(value[0], BaseModel):
                            new_value.append(self._serialize_signature_argument(v, pickle_ts=pickle_ts))
                            sort = False
                        else:
                            new_value.append(v)
                    if sort == True:
                        new_value.sort()
                    value = new_value

        elif isinstance(value, dict):
            for value_key, value_value in value.items():
                if isinstance(value_value, dict):
                    value[value_key] = self._serialize_configuration_dict(value_value, pickle_ts=pickle_ts)
                else:
                    value[value_key] = self._serialize_signature_argument(value_value, pickle_ts=pickle_ts)
        elif isinstance(value, TimeSerieConfigKwargs):
            self.logger.warning("TimeSeriesConfigKwargs will be depreciated")
            new_value = {"is_time_series_config": True}
            if pickle_ts == False:
                value = self._serialize_configuration_dict(value, pickle_ts=pickle_ts)
            else:
                value = self._serialize_configuration_dict(value, pickle_ts=pickle_ts)
            new_value["config_data"] = value
            value = new_value
        return value

    def _serialize_configuration_dict(self, kwargs, pickle_ts=False, ordered_dict=True):

        import collections
        new_kwargs = {}

        for key, value in kwargs.items():

            if key in ["model_list"]:
                ##already serialized arguments
                new_kwargs[key] = value
                continue
            if isinstance(value, dict):
                if "is_time_serie_pickled" in value.keys() or "is_api_time_serie_pickled"  in value.keys():
                    new_kwargs[key] = value
                    continue

            value = self._serialize_signature_argument(value, pickle_ts=pickle_ts)

            new_kwargs[key] = value
        if ordered_dict == True:
            ordered_kwargs = collections.OrderedDict(sorted(new_kwargs.items()))
        else:
            ordered_kwargs = new_kwargs
        return ordered_kwargs

    def serialize_to_pickle(self, properties):
        serialized_properties = self._serialize_configuration_dict(kwargs=properties, pickle_ts=True)
        return serialized_properties

    def serialize_init_kwargs(self, kwargs):
        """
          serializes  TimeSeries init_kwargs to be able to  persist in local configuration
          :param kwargs:
          :return:
          """

        import collections
        if kwargs["time_series_class_import_path"]["qualname"] == "WrapperTimeSerie":
            ts_kwargs = {}
            ts = collections.OrderedDict(sorted(kwargs["time_series_dict"].items()))
            for key, ts in ts.items():

                ts_kwargs[key] = f"{ts.local_hash_id}_{ts.data_source_id}"
            kwargs["time_series_dict"] = ts_kwargs

        ordered_kwargs = self._serialize_configuration_dict(kwargs=kwargs)

        return ordered_kwargs

    @classmethod
    def deserialize_pickle_value(cls, value, include_vam_client_objects: bool,
                                 graph_depth_limit: int,
                                 graph_depth: int, local_metadatas: Union[dict, None],
                                 ignore_pydantic=False, data_source_id: Union[DynamicTableDataSource, None] = None, ):
        from mainsequence.vam_client.models_helpers import get_model_class
        import cloudpickle

        new_value = value

        state_kwargs = dict(local_metadatas=local_metadatas,
                            graph_depth_limit=graph_depth_limit,
                            data_source_id=data_source_id,
                            graph_depth=copy.deepcopy(graph_depth),
                            include_vam_client_objects=include_vam_client_objects)
        if isinstance(value, dict):

            if "is_time_series_config" in value.keys():
                new_value = TimeSerieConfigKwargs(**value["config_data"])
            elif "__type__" in value.keys():
                rebuild_function = lambda x: cls.deserialize_pickle_state(x, **state_kwargs)
                new_value = rebuild_with_type(value, rebuild_function=rebuild_function)
            elif "is_model_list" in value.keys():
                new_value = ModelList([build_model(v) for v in value['model_list']])
            elif "is_time_serie_pickled" in value.keys():
                try:
                    full_path = TimeSerie.get_pickle_path(local_hash_id=value['local_hash_id'],
                                                          data_source_id=value['data_source_id']
                                                          )
                except Exception as e:
                    raise e
                with open(full_path, 'rb') as handle:
                    ts = cloudpickle.load(handle)
                    ts.set_data_source_from_pickle_path(full_path)
                    if graph_depth - 1 <= graph_depth_limit:
                        ts.set_state_with_sessions(
                            graph_depth_limit=graph_depth_limit,
                            graph_depth=graph_depth,
                            local_metadatas=local_metadatas,
                            include_vam_client_objects=include_vam_client_objects)
                    new_value = ts
            elif "is_api_time_serie_pickled" in value.keys():
                full_path = APITimeSerie.get_pickle_path(local_hash_id=value['local_hash_id'],
                                                      data_source_id=value['data_source_id']
                                                      )
                with open(full_path, 'rb') as handle:
                    ts = cloudpickle.load(handle)
                new_value=ts
            elif "orm_class" in value.keys():
                new_value = build_model(value)
            elif "pydantic_model_import_path" in value.keys():

                new_value = cls.deserialize_pickle_state(value,

                                                         ignore_pydantic=True, **state_kwargs
                                                         )
                if ignore_pydantic == False:
                    new_value = cls.rebuild_pydantic_model(new_value,
                                                           state_kwargs=state_kwargs)
            else:
                new_value = cls.deserialize_pickle_state(value,
                                                         **state_kwargs)
        if isinstance(value, tuple):
            new_value = [
                cls.deserialize_pickle_value(v, ignore_pydantic=ignore_pydantic, **state_kwargs) if isinstance(v,
                                                                                                               dict) else v
                for v in value]
            new_value = tuple(new_value)
        if isinstance(value, list):
            if len(value) == 0:
                return new_value
            # if isinstance(value[0], dict):
            #     if "orm_class" in value[0].keys():
            #         new_value = [build_model(v) for v in value]
            #     elif "pydantic_model_import_path" in value[0].keys():
            #             new_value = [cls.rebuild_pydantic_model(v) for v in value]

            new_value = [
                cls.deserialize_pickle_value(v, ignore_pydantic=ignore_pydantic, **state_kwargs) if isinstance(v,
                                                                                                               dict) else v
                for v in value]

        return new_value

    @classmethod
    def deserialize_pickle_state(cls, state, include_vam_client_objects: bool, data_source_id: int,
                                 graph_depth_limit: int,
                                 graph_depth: int, local_metadatas: Union[dict, None],
                                 ignore_pydantic=False,
                                 ):
        """
        
        Parameters
        ----------
        state


        deserialize_pickle_state

        Returns
        -------

        """
        if isinstance(state, dict):
            for key, value in state.items():
                state[key] = cls.deserialize_pickle_value(value, include_vam_client_objects=include_vam_client_objects,
                                                          graph_depth_limit=graph_depth_limit, graph_depth=graph_depth,
                                                          local_metadatas=local_metadatas,
                                                          data_source_id=data_source_id,
                                                          )
        elif isinstance(state, tuple):
            state = tuple([cls.deserialize_pickle_value(v, include_vam_client_objects=include_vam_client_objects,
                                                        graph_depth_limit=graph_depth_limit, graph_depth=graph_depth,
                                                        local_metadatas=local_metadatas, data_source_id=data_source_id,
                                                        ) for v in state])
        elif isinstance(state, str) or isinstance(state, float) or isinstance(state, int) or isinstance(state, bool):
            pass
        else:
            raise NotImplementedError

        return state


def get_time_serie_relation_tree(time_serie, as_object=False):
    import inspect
    members = inspect.getmembers(time_serie, lambda a: not (inspect.isroutine(a)))
    # get dicto of members
    members = [a[1] for a in members if a[0] == "__dict__"][0]

    def ts_data(ts):
        if as_object is True:
            return time_serie
        else:
            return (time_serie.data_folder, time_serie.__class__.__name__)

    tree = {"father": ts_data(time_serie), "children": []}
    for key, value in members.items():
        if "related_time_series" in key:
            # add to tree
            if isinstance(value, list):
                raise NotImplementedError
            elif isinstance(value, dict):
                for tm_ts in value.values():
                    children = get_time_serie_relation_tree(tm_ts, as_object=as_object)

                    tree["children"].append(children)
        if isinstance(value, TimeSerie):
            children = get_time_serie_relation_tree(value, as_object=as_object)

            tree["children"].append(children)

    return tree


class DependencyUpdateError(Exception):
    pass


class GraphNodeMethods(ABC):

    def get_mermaid_dependency_diagram(self):
        from IPython.display import display, HTML

        mermaid_diagram = self.local_persist_manager.display_mermaid_dependency_diagram()

        # Mermaid.js initialization script (only run once)
        if not hasattr(display, "_mermaid_initialized"):
            mermaid_initialize = """
                   <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
                   <script>
                       function initializeMermaid() {
                           if (typeof mermaid !== 'undefined') {
                               console.log('Initializing Mermaid.js...');
                               const mermaidDivs = document.querySelectorAll('.mermaid');
                               mermaidDivs.forEach(mermaidDiv => {
                                   mermaid.init(undefined, mermaidDiv);
                               });
                           } else {
                               console.error('Mermaid.js is not loaded.');
                           }
                       }
                   </script>
                   """
            display(HTML(mermaid_initialize))
            display._mermaid_initialized = True

        # HTML template for rendering the Mermaid diagram
        html_template = f"""
               <div class="mermaid">
               {mermaid_diagram}
               </div>
               <script>
                   initializeMermaid();
               </script>
               """

        # Display the Mermaid diagram in the notebook
        display(HTML(html_template))

        # Optionally return the raw diagram code for further use
        return mermaid_diagram

    def get_all_local_dependencies(self, ):
        """
        get relation tree by ids in the graph
        :return:
        """

        dependencies_df = self.local_persist_manager.get_all_local_dependencies()
        return dependencies_df

    @property
    def is_local_relation_tree_set(self):
        return self.local_persist_manager.local_metadata["ogm_dependencies_linked"]


    def get_update_map(self,dependecy_map:Union[dict,None]=None):
        """
        Obtain all local time_series in the dependency graph by introspecting the code
        :return:
        """
        members = self.__dict__
        dependecy_map={} if dependecy_map is None else dependecy_map
        for key, value in members.items():
            try:
                if "related_time_series" in key:  # this is necessary for WrapperTimeSeries
                    # add to tree
                    assert isinstance(value, dict)

                    for tm_ts in value.values():
                        if isinstance(tm_ts, dict):
                            pickle_path = self.get_pickle_path(local_hash_id=tm_ts["local_hash_id"])
                            dependecy_map[(tm_ts["local_hash_id"],tm_ts["local_hash_id"])]={"is_pickle":True,"ts":pickle_path}

                        else:
                            is_api = isinstance(tm_ts, APITimeSerie)
                            dependecy_map[(tm_ts.local_hash_id,tm_ts.data_source_id)]={"is_pickle":False,"ts":tm_ts}
                            tm_ts.get_update_map(dependecy_map)

                if isinstance(value, TimeSerie):
                    value.local_persist_manager  # before connection call local persist manager to garantee ts is created
                    dependecy_map[(value.local_hash_id,value.data_source_id)]={"is_pickle":False,"ts":value}
                    value.get_update_map(dependecy_map)
                if isinstance(value, APITimeSerie):
                    value.local_persist_manager  # before conne
                    dependecy_map[(value.local_hash_id,value.data_source_id)] = {"is_pickle": False, "ts": value}


                if isinstance(value, dict):
                    if "is_time_serie_pickled" in value.keys():
                        pickle_path = self.get_pickle_path(local_hash_id=value["local_hash_id"])
                        dependecy_map[(value.local_hash_id,value.data_source_id)] = {"is_pickle": True, "ts": pickle_path}

                    if "is_api_time_serie_pickled" in value.keys():
                        is_api = isinstance(tm_ts, APITimeSerie)
                        dependecy_map[(value.local_hash_id,value.data_source_id)] = {"is_pickle": False, "ts": tm_ts}
                        tm_ts.get_update_map(dependecy_map)
            except Exception as e:
                raise e
        return dependecy_map
    def set_relation_tree(self):
        """
        Sets relationhsip in the DB
        :return:
        """

        members = self.__dict__

        if self.is_local_relation_tree_set == False:

            for key, value in members.items():
                try:
                    if "related_time_series" in key: #this is necessary for WrapperTimeSeries
                        # add to tree
                        if isinstance(value, list):
                            raise NotImplementedError
                        elif isinstance(value, dict):
                            for tm_ts in value.values():
                                if isinstance(tm_ts, dict):
                                    pickle_path = self.get_pickle_path(local_hash_id=tm_ts["local_hash_id"])
                                    new_ts = TimeSerie.load_and_set_from_pickle(pickle_path=pickle_path)
                                    new_ts.local_persist_manager  # before connection call local persist manager to garantee ts is created
                                    self.local_persist_manager.depends_on_connect(new_ts)
                                    new_ts.set_relation_tree()
                                else:
                                    is_api = isinstance(tm_ts, APITimeSerie)
                                    tm_ts.local_persist_manager  # before connection call local persist manager to garantee ts is created
                                    self.local_persist_manager.depends_on_connect(tm_ts,is_api=is_api)
                                    tm_ts.set_relation_tree()

                    if isinstance(value, TimeSerie):
                        value.local_persist_manager  # before connection call local persist manager to garantee ts is created
                        self.local_persist_manager.depends_on_connect(value, is_api=False)
                        value.set_relation_tree()
                    if isinstance(value, APITimeSerie):
                        value.local_persist_manager  # before conne
                        self.local_persist_manager.depends_on_connect(value, is_api=True)

                    if isinstance(value, dict):
                        if "is_time_serie_pickled" in value.keys():
                            pickle_path = self.get_pickle_path(local_hash_id=value["local_hash_id"],data_source_id=value["data_source_id"])
                            new_ts = TimeSerie.load_and_set_from_pickle(pickle_path=pickle_path)
                            new_ts.local_persist_manager  # before connection call local persist manager to garantee ts is created
                            self.local_persist_manager.depends_on_connect(new_ts)
                            new_ts.set_relation_tree()
                        if "is_api_time_serie_pickled" in value.keys():

                            new_ts = APITimeSerie.load_and_set_from_pickle(local_hash_id=value["local_hash_id"],
                                                                           data_source_id=value["data_source_id"]
                                                                           )
                            self.local_persist_manager.depends_on_connect(new_ts)
                            new_ts.set_relation_tree()
                except Exception as e:
                    raise e
            self.local_persist_manager.set_ogm_dependencies_linked()




class TimeSerieRebuildMethods(ABC):

    @none_if_backend_detached
    def verify_backend_git_hash_with_pickle(self):
        if self.local_persist_manager.metadata is not None:
            load_git_hash = self.get_time_serie_source_code_git_hash(self.__class__)

            persisted_pickle_hash = self.local_persist_manager.metadata["time_serie_source_code_git_hash"]
            if load_git_hash != persisted_pickle_hash:
                self.logger.warning(
                    f"{bcolors.WARNING}Source code does not match with pickle rebuilding{bcolors.ENDC}")
                self.flush_pickle()

                rebuild_time_serie = TimeSerie.rebuild_from_configuration(local_hash_id=self.local_hash_id,
                                                                          data_source=self.data_source,
                )
                rebuild_time_serie.persist_to_pickle()
            else:
                # if no need to rebuild, just sync the metadata
                self.local_persist_manager.synchronize_metadata(meta_data=None, local_metadata=None)

    @classmethod
    @tracer.start_as_current_span("TS: load_from_pickle")
    def load_from_pickle(cls, pickle_path):
        time_serie = load_from_pickle(pickle_path)

        return time_serie

    @classmethod
    def load_data_source_from_pickle(self, pickle_path):
        data_path = Path(pickle_path).parent / "data_source.pickle"
        with open(data_path, 'rb') as handle:
            data_source = cloudpickle.load(handle)
        return data_source

    def set_data_source_from_pickle_path(self, pikle_path):
        data_source = self.load_data_source_from_pickle(pikle_path)
        self.set_data_source(data_source=data_source)

    @classmethod
    def load_and_set_from_pickle(cls, pickle_path, graph_depth_limit=1, ):
        ts = cls.load_from_pickle(pickle_path)
        ts.set_state_with_sessions(
            graph_depth=0,
            graph_depth_limit=graph_depth_limit,
            include_vam_client_objects=False)
        return ts


    @classmethod
    def rebuild_from_pickle_or_configuration(self,local_hash_id,data_source_id,
                                             graph_depth_limit=1
                                             ):
        """
        :param local_hash_id:
        :param data_source_id:
        :return:
        """
        pickle_path = self.get_pickle_path(local_hash_id,
                                           data_source_id=data_source_id)
        data_source = self.load_data_source_from_pickle(pickle_path=pickle_path)
        if os.path.isfile(pickle_path) == False:
            ts = TimeSerie.rebuild_from_configuration(local_hash_id=local_hash_id,
                                                      data_source=data_source
                                                      )
            ts.persist_to_pickle()

        ts = TimeSerie.load_and_set_from_pickle(pickle_path=pickle_path,
                                                graph_depth_limit=graph_depth_limit
                                                )
        return ts

    @classmethod
    @tracer.start_as_current_span("TS: Rebuild From Configuration")
    def rebuild_from_configuration(cls, local_hash_id,
                                   data_source: Union[int, object],

                                   ):
        """

        :param serie_data_folder:

        :return: TimeSerie
        """

        import importlib
        from mainsequence.tdag.time_series.persist_managers import PersistManager

        tracer_instrumentator.append_attribute_to_current_span("time_serie_hash_id", local_hash_id)

        if isinstance(data_source, int):
            pickle_path = cls.get_pickle_path(local_hash_id,
                                              data_source_id=data_source)
            if os.path.isfile(pickle_path) == False:
                data_source = DynamicTableDataSource.get(id=data_source)
                data_source.persist_to_pickle(data_source_pickle_path(data_source.id))

            data_source = cls.load_data_source_from_pickle(pickle_path=pickle_path)

        persist_manager = PersistManager.get_from_data_type(local_hash_id=local_hash_id,
                                                            data_source=data_source,
                                                            )
        try:
            time_serie_config = persist_manager.local_build_configuration
        except Exception as e:
            raise e

        try:
            mod = importlib.import_module(time_serie_config["time_series_class_import_path"]["module"])
            TimeSerieClass = getattr(mod, time_serie_config["time_series_class_import_path"]["qualname"])
        except Exception as e:
            raise e

        time_serie_class_name = time_serie_config["time_series_class_import_path"]["qualname"]

        time_serie_config.pop("time_series_class_import_path")
        time_serie_config = ConfigSerializer.rebuild_serialized_config(time_serie_config,
                                                                       time_serie_class_name=time_serie_class_name)
        time_serie_config["init_meta"] = {}

        re_build_ts = TimeSerieClass(**time_serie_config)

        return re_build_ts

    def __setstate__(self, state):
        # Restore instance attributes (i.e., filename and lineno).
        self.__dict__.update(state)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self._prepare_state_for_pickle(state=self.__dict__)

        # Remove the unpicklable entries.
        return state

    @classmethod
    def get_pickle_path(cls, local_hash_id, data_source_id: int):
        return f"{ogm.pickle_storage_path}/{data_source_id}/{local_hash_id}.pickle"

    @classmethod
    def load_and_set_from_hash_id(cls, local_hash_id, data_source_id: int):
        path = cls.get_pickle_path(local_hash_id=local_hash_id, data_source_id=data_source_id)
        ts = cls.load_and_set_from_pickle(pickle_path=path)
        return ts

    @property
    def pickle_path(self):
        pp = data_source_dir_path(self.data_source.id)
        path = f"{pp}/{self.local_hash_id}.pickle"
        return path

    @none_if_backend_detached
    def _update_git_and_code_in_backend(self):
        self.local_persist_manager.update_source_informmation(
            git_hash_id=self.get_time_serie_source_code_git_hash(self.__class__),
            source_code=self.get_time_serie_source_code(self.__class__),
        )

    def persist_to_pickle(self, overwrite=False):
        """

        :return:
        :rtype:
        """
        import cloudpickle
        path = self.pickle_path
        # after persisting pickle , build_hash and source code need to be patched
        self.logger.info(f"Persisting pickle and patching source code and git hash for {self.hash_id}")
        self._update_git_and_code_in_backend()

        pp = data_source_pickle_path(self.data_source.id)
        if os.path.isfile(pp) == False or overwrite == True:
            self.data_source.persist_to_pickle(pp)

        if os.path.isfile(path) == False or overwrite == True:
            if overwrite == True:
                self.logger.warning("overwriting pickle")

            with open(path, 'wb') as handle:
                cloudpickle.dump(self, handle)

        return path, path.replace(ogm.pickle_storage_path + "/", "")

    @tracer.start_as_current_span("TS: set_state_with_sessions")
    def set_state_with_sessions(self, include_vam_client_objects=True,
                                graph_depth_limit=1000, local_metadatas: Union[dict, None] = None,
                                graph_depth=0):
        """
         Method to set state after it was loaded from pickle.
        Parameters
        ----------
        include_vam_client_objects :
        graph_depth_limit :
        metadatas : pre-requestd dictionary of metadatas to speed calculation of rebuild of state
        graph_depth :

        Returns
        -------

        """
        if graph_depth_limit==-1:
            graph_depth_limit=1e6
        if local_metadatas is not None:
            local_metadatas = None if len(local_metadatas) == 0 else local_metadatas

        minimum_required_depth_for_update = self.get_minimum_required_depth_for_update()

        state = self.__dict__

        state, init_meta = self._sanitize_init_meta(kwargs=state)
        self._set_logger(local_hash_id=self.hashed_name)
        if graph_depth_limit < minimum_required_depth_for_update and graph_depth == 0:
            graph_depth_limit = minimum_required_depth_for_update
            self.logger.warning(f"Graph depht limit overrided to {minimum_required_depth_for_update}")

        #if the data source is not local then the de-serialization needs to happend after setting the local persist manager
        #to guranteed a proper patch in the back-end
        if graph_depth <= graph_depth_limit and self.data_source.data_type:
            local_metadata = local_metadatas[
                (self.local_hash_id, self.data_source.id)] if local_metadatas is not None else None
            self._set_local_persist_manager(local_hash_id=self.local_hash_id,
                                            remote_table_hashed_name=self.remote_table_hashed_name,
                                            local_metadata=local_metadata,verify_local_run=False,
                                            )

        serializer = ConfigSerializer()
        state = serializer.deserialize_pickle_state(state=state, data_source_id=self.data_source.id,
                                                    include_vam_client_objects=include_vam_client_objects,
                                                    graph_depth_limit=graph_depth_limit,
                                                    local_metadatas=local_metadatas,
                                                    graph_depth=graph_depth + 1)

        self.__dict__.update(state)

        self.local_persist_manager.synchronize_metadata(meta_data=None, local_metadata=None)

    def get_minimum_required_depth_for_update(self):
        """
        Controls the minimum depth that needs to be rebuil
        Returns
        -------

        """
        return 0

    def _prepare_state_for_pickle(self, state):
        """
        Method to run before ttime series is pikledl
        :return:
        :rtype:
        """
        import cloudpickle
        properties = state
        serializer = ConfigSerializer()
        properties = serializer.serialize_to_pickle(properties)
        names_to_remove = []
        for name, attr in properties.items():
            if name in ["local_persist_manager", "logger", "init_meta"]:
                names_to_remove.append(name)
                continue
            try:
                cloudpickle.dumps(attr)
            except Exception as e:
                self.logger.exception(f"Cant Pickle property {name}")
                raise e

        properties = {key: value for key, value in properties.items() if key not in names_to_remove}
        return properties


    def run_in_debug_scheduler(self, break_after_one_update=True, run_head_in_main_process=True,
                               wait_for_update=True, force_update=True, debug=True, update_tree=True,
                               raise_exception_on_error=True
                               ):
        """

        Args:
            break_after_one_update:
            run_head_in_main_process:
            wait_for_update:
            force_update:
            debug:
            update_tree:

        Returns:

        """


        from .update.scheduler import SchedulerUpdater
        SchedulerUpdater.debug_schedule_ts(
            time_serie_hash_id=self.local_hash_id,
            data_source_id=self.data_source.id,
            break_after_one_update=break_after_one_update,
            run_head_in_main_process=True,
            wait_for_update=False,
            force_update=True,
            debug=True,
            update_tree=True,
            raise_exception_on_error=raise_exception_on_error
        )


    @tracer.start_as_current_span("TS: Update")
    def update(self, update_tracker: object, debug_mode: bool,
               raise_exceptions=True, update_tree=False, start_update_data: Union[StartUpdateDataInfo, None] = None,
               metadatas: Union[dict, None] = None, update_only_tree=False, force_update=False,use_state_for_update=False
               ):
        """
        Main update method for time series that interacts with Graph node. Time series should be updated through this
        method only
        :param update_tree_kwargs:
        :param raise_exceptions:
        :param update_tree:
        :param scheduler: models.Scheduler
        :param metadatas: pre-requested metadatas to speed initiation of ts
        :return:
        """

        if start_update_data is None:
            start_update_data = update_tracker.set_start_of_execution(local_hash_id=self.local_hash_id,
                                                                      data_source_id=self.data_source.id)

        latest_value, must_update = start_update_data.last_time_index_value, start_update_data.must_update

        error_on_last_update = False

        if force_update == True:
            must_update = True

        if must_update == True:
            try:

                max_update_time_days = os.getenv("TDAG_MAX_UPDATE_TIME_DAYS", None)
                update_on_batches = False
                if update_on_batches is not None:
                    max_update_time_days = datetime.timedelta(days=update_on_batches)
                    update_on_batches = True

                self.update_local(update_tree=update_tree, debug_mode=debug_mode,
                                  overwrite_latest_value=latest_value, metadatas=metadatas,
                                  update_tracker=update_tracker, update_only_tree=update_only_tree,
                                  use_state_for_update=use_state_for_update,
                                  )

                update_tracker.set_end_of_execution(local_hash_id=self.local_hash_id,data_source_id=self.data_source.id,
                                                    error_on_update=error_on_last_update)

            except Exception as e:

                error_on_last_update = True
                self.logger.exception(f"Error updating")

                logging.shutdown()
                if raise_exceptions is True:
                    update_tracker.set_end_of_execution(local_hash_id=self.local_hash_id,data_source_id=self.data_source.id,
                                                        error_on_update=error_on_last_update)






        else:

            self.logger.info("Already updated, waiting until next update time")
            update_tracker.set_end_of_execution(local_hash_id=self.local_hash_id,data_source_id=self.data_source.id,
                                                error_on_update=error_on_last_update)
        self._run_post_update_routines(error_on_last_update=error_on_last_update)
        # close all logging handlers
        logging.shutdown()
        return error_on_last_update


class DataPersistanceMethods(ABC):

    # sets
    def get_metadatas_and_set_updates(self, *args, **kwargs):
        from mainsequence.tdag_client import DynamicTableHelpers
        dth = DynamicTableHelpers()
        return dth.get_metadatas_and_set_updates(*args, **kwargs)

    def patch_update_details(self, local_hash_id=None, *args, **kwargs):
        return self.local_persist_manager.patch_update_details(local_hash_id=local_hash_id, **kwargs)

    def reset_dependencies_states(self, hash_id_list: list):
        return self.local_persist_manager.reset_dependencies_states(hash_id_list=hash_id_list)

    def update_details_in_dependecy_tree(self, set_relation_tree=True, include_head=False, *args, **kwargs):
        """
        updates schedule from all tree related time series
        :param schedule:
        :return:
        """

        if set_relation_tree == True:
            self.set_relation_tree()
        dependants_df = self.get_all_local_dependencies()

        dependants_records = []
        if len(dependants_df) > 0:
            dependants_records = dependants_df[["local_hash_id", "data_source_id"]].to_dict("records")

        if include_head:
            dependants_records.append({"local_hash_id": self.local_hash_id, "data_source_id": self.data_source.id})

        all_metadatas = self.get_metadatas_and_set_updates(local_hash_id__in=dependants_records,
                                                           multi_index_asset_symbols_filter=self.multi_index_asset_symbols_filter,
                                                           update_priority_dict=None,
                                                           update_details_kwargs=kwargs)

    # direct contact
    @property
    def update_details(self):
        return self.local_persist_manager.update_details

    @none_if_backend_detached
    @property
    def metadata(self):
        return self.local_persist_manager.metadata

    @none_if_backend_detached
    @property
    def local_metadata(self):
        return self.local_persist_manager.local_metadata

    @property
    def source_table_configuration(self):
        return self.local_persist_manager.source_table_configuration

    def set_policy(self, interval: str, comp_type: str, overwrite=False, ):

        self.local_persist_manager.set_policy(interval, overwrite=overwrite, comp_type=comp_type)

    def verify_tree_compression_policy_is_set(self):

        deps = self.dependencies_df
        for _, ts_row in deps.iterrows():
            try:
                ts = TimeSerie.rebuild_from_configuration(hash_id=ts_row["hash_id"],
                                                          )
                ts.set_compression_policy()
            except Exception as e:
                self.logger.exception(f"{ts_row['hash_id']} compression policy not set")

    def upsert_data(self, data_df: pd.DataFrame):
        """
        Updates and Insert data into DB
        :param data_df:
        :return:
        """
        self.local_persist_manager.upsert_data(data_df=data_df)

    @property
    def persist_size(self):

        return self.local_persist_manager.persist_size

    def get_update_statistics(self, asset_symbols: Union[list, None] = None,
                         ) -> datetime.datetime:
        """
        getts latest value directly from querying the DB,
        args and kwargs are nedeed for datalake
        Parameters
        ----------
        args :
        kwargs :

        Returns
        -------

        """

        last_update_in_table, last_update_per_asset = self.local_persist_manager.get_update_statistics(
            remote_table_hash_id=self.remote_table_hashed_name,
            asset_symbols=asset_symbols,time_serie=self)
        return last_update_in_table, last_update_per_asset

    def get_earliest_value(self) -> datetime.datetime:
        earliest_value = self.local_persist_manager.get_earliest_value()
        return earliest_value

    @property
    def local_parquet_file(self):
        try:
            return self.data_folder + "/time_series_data.parquet"
        except Exception as e:
            raise

    def is_persisted(self, session: Union[None, object] = None):

        ip = self.local_persist_manager.time_serie_exist()
        return ip


    def filter_by_assets_ranges(self, asset_ranges_map: dict):
        """

        Parameters
        ----------
        asset_ranges

        Returns
        -------

        """
        df = self.local_persist_manager.filter_by_assets_ranges(asset_ranges_map)
        return df



    def get_df_between_dates(self, start_date: Union[datetime.datetime, None] = None,
                             end_date: Union[datetime.datetime, None] = None,
                             asset_symbols: Union[None, list] = None,
                             data_lake_force_db_look=False, great_or_equal=True, less_or_equal=True,
                             ):
        func = self.local_persist_manager.get_df_between_dates
        sig = inspect.signature(func)
        kwargs=dict(start_date=start_date,
                                                                        end_date=end_date,
                                                                        asset_symbols=asset_symbols,
                                                                        great_or_equal=great_or_equal,
                                                                        less_or_equal=less_or_equal,
                    )
        if 'time_serie' in sig.parameters:
            kwargs["time_serie"] = self
        filtered_data = self.local_persist_manager.get_df_between_dates(**kwargs)


        return filtered_data

    def get_persisted_ts(self):
        try:
            persisted_df = self.local_persist_manager.get_persisted_ts()
        except Exception as e:
            # if local file got corrupted flush local time serie withouth deleting configuration
            # self.logger.error(f"{self} local file got corrupted flush local time serie withouth deleting configuration")
            # self.flush_local_persisted()
            raise e

        return persisted_df

    def load_ts_df(self, session: Union[None, object] = None, *args, **kwargs):
        if self.is_persisted:
            pandas_df = self.get_persisted_ts(session=session)
        else:
            self.update_local(update_tree=False, *args, **kwargs)
            pandas_df = self.get_persisted_ts(session=session)

        return pandas_df


    def get_latest_update_by_assets_filter(self,asset_symbols:Union[list,None], last_update_per_asset:dict):
        """
        Gets the latest update from a symbol list
        :param asset_symbols:
        :return:
        """
        if asset_symbols is not None:
            last_update_in_table = np.max([np.max(list(i.values()))
                                           for asset_symbol, i in last_update_per_asset.items()
                                           if asset_symbol in asset_symbols
                                           ])
        else:
            last_update_in_table = np.max([np.max(list(i.values()))
                                           for i in list(last_update_per_asset.values())
                                           ])
        return last_update_in_table

    def get_earliest_updated_asset_filter(self, asset_symbols:Union[list],
                                          last_update_per_asset:dict):
        if asset_symbols is not None:
            last_update_in_table = min(
                [t for a in last_update_per_asset.values() for t in a.values() if a in asset_symbols])
        else:
            last_update_in_table = min([t for a in last_update_per_asset.values() for t in a.values()])
        return last_update_in_table

    def get_last_observation(self, asset_symbols: Union[None, list] = None,

                             ):
        """

        (1) Requests last observatiion from local persist manager
        (3) evaluates if last observation is consistent
        Parameters
        ----------
        asset_symbols :

        Returns
        -------

        """

        last_update_in_table, last_update_per_asset= self.get_update_statistics(asset_symbols=asset_symbols,

                                                         )
        if last_update_in_table is None and last_update_per_asset is None:
            return None
        if asset_symbols is not None and last_update_per_asset is not None:
            if len(last_update_per_asset) > 0:
                last_update_in_table = self.get_latest_update_by_assets_filter(asset_symbols=asset_symbols,
                                                                              last_update_per_asset=last_update_per_asset
                                                                              )


        last_observation = self.get_df_between_dates(
            start_date=last_update_in_table, great_or_equal=True,
            asset_symbols=asset_symbols
        )

        return last_observation

    @property
    def last_observation(self):
        last_observation = self.get_last_observation()
        return last_observation

    def delete_time_series(self, delete_only_time_series: bool = False, ):

        self.local_persist_manager.delete_time_series(delete_only_time_series=delete_only_time_series)

    def flush_local_persisted(self, flush_only_time_series=True, session: Union[object, None] = None):
        """
        deletes  persisted data
        :param flush_sub_folders:
        :return:
        """
        self.local_persist_manager.flush_local_persisted(flush_only_time_series=flush_only_time_series)

    @tracer.start_as_current_span("TS: Persist Data")
    def persist_updated_data(self, temp_df, update_tracker,
                             latest_value: Union[None, datetime.datetime],
                             overwrite=False) -> bool:
        persisted = False
        if temp_df.shape[0] > 0:
            if overwrite == True:
                self.logger.warning(f"Values will be overwritten assuming latest value of  {latest_value}")
            self.local_persist_manager.persist_updated_data(temp_df=temp_df,
                                                            update_tracker=update_tracker,
                                                            historical_update_id=None,
                                                            overwrite=overwrite)
            persisted = True
        return persisted




class TimeSerieConfigKwargs(dict):
    """
    Necessary class for configuration
    """
    pass


class ModelList(list):
    """
    Necessary for configuration
    """
    pass


def data_source_dir_path(data_source_id):
    path = ogm.pickle_storage_path
    return f"{path}/{data_source_id}"


def data_source_pickle_path(data_source_id):

    return f"{data_source_dir_path(data_source_id)}/data_source.pickle"

@tracer.start_as_current_span("TS: load_from_pickle")
def load_from_pickle(pickle_path):
    import cloudpickle
    from pathlib import Path

    directory = os.path.dirname(pickle_path)
    filename = os.path.basename(pickle_path)
    prefixed_path = os.path.join(directory, f"{APITimeSerie.PICKLE_PREFIFX }{filename}")
    if os.path.isfile(prefixed_path) and os.path.isfile(pickle_path):
        raise FileExistsError("Both default and API timeseries pickle exist - cannot decide which to load")

    if os.path.isfile(prefixed_path):
        pickle_path = prefixed_path


    try:
        with open(pickle_path, 'rb') as handle:
            time_serie = cloudpickle.load(handle)
    except Exception as e:
        raise e

    if isinstance(time_serie, APITimeSerie):
        return time_serie

    data_source = time_serie.load_data_source_from_pickle(pickle_path=pickle_path)
    time_serie.set_data_source(data_source=data_source)
    # verify pickle
    time_serie.verify_backend_git_hash_with_pickle()
    return time_serie

class APITimeSerie:

    PICKLE_PREFIFX = "api-"
    @classmethod
    def build_from_local_time_serie(cls, local_time_serie: "TimeSerieLocalUpdate"):
        return cls(data_source_id=local_time_serie.remote_table["data_source"]["id"],
                   local_hash_id=local_time_serie.local_hash_id
                   )

    def __init__(self, data_source_id:int, local_hash_id:str, data_source_local_lake: Union[None,DataSource]=None):
        """
        A time serie is uniquely identified in tdag by  data_source_id and table_name
        :param data_source_id:
        :param table_name:
        """
        if data_source_local_lake is not None:
            assert data_source_local_lake.data_type==CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE, "data_source_local_lake should be of type CONSTANTS.DATA_SOURCE_TYPE_LOCAL_DISK_LAKE"
        self.data_source_id = data_source_id
        self.local_hash_id = local_hash_id
        self.data_source = data_source_local_lake
        self.local_persist_manager


    @property
    def local_persist_manager(self):
        if hasattr(self, "logger") == False:
            self._set_logger(local_hash_id=self.local_hash_id)
        if hasattr(self, "_local_persist_manager") == False:
            self.logger.info(f"Setting local persist manager for {self.local_hash_id}")
            self._set_local_persist_manager()
        return self._local_persist_manager
    def set_relation_tree(self):
        pass #do nothing  for API Time Series
    def _set_logger(self, local_hash_id):
        if hasattr(self, "logger") == False:

            self.logger = create_logger_in_path(logger_name=f"{self.data_source_id}_{self.local_hash_id}", application_name="tdag",
                                                logger_file=f'{ogm.get_logging_path()}/{self.data_source_id}/{self.local_hash_id}.log',
                                                local_hash_id=self.local_hash_id, data_source_id=self.data_source_id,
                                                api_time_series=True,
                                                )
    def _verify_local_data_source(self):
        pod_source = os.environ.get("POD_DEFAULT_DATA_SOURCE", None)
        if pod_source != None:
            from mainsequence.tdag_client import models as models
            pod_source = json.loads(pod_source)
            ModelClass = pod_source["tdag_orm_class"]
            pod_source.pop("tdag_orm_class", None)
            ModelClass = getattr(models, ModelClass)
            pod_source = ModelClass(**pod_source)
            self.data_source = pod_source
    def build_data_source_from_configuration(self,data_config):
        from mainsequence.tdag_client import models as models
        ModelClass =DynamicTableDataSource.get_class(data_config['data_type'])
        pod_source = ModelClass.get(data_config["id"])
        return pod_source

    def _set_local_persist_manager(self):
        from mainsequence.tdag.time_series.persist_managers import APIPersistManager


        self._verify_local_data_source()

        self._local_persist_manager = APIPersistManager(data_source_id=self.data_source_id,
                                                        local_hash_id=self.local_hash_id,
                                                        logger=self.logger)
        local_metadata = TimeSerieLocalUpdate.get(local_hash_id=self.local_hash_id)
        self.remote_table_hashed_name = local_metadata["remote_table"]["hash_id"]
        if self.data_source is None:

            if isinstance(local_metadata["remote_table"]["data_source"], dict):
                #data source is open use direct connection

                self._local_persist_manager = PersistManager.get_from_data_type(local_hash_id=self.local_hash_id,
                                                                                class_name=local_metadata["build_configuration"]["time_series_class_import_path"]["qualname"],
                                                                                human_readable=local_metadata["remote_table"]["human_readable"],
                                                                                logger=self.logger,
                                                                                local_metadata=local_metadata,
                                                                                description="",
                                                                                data_source= self.build_data_source_from_configuration(local_metadata["remote_table"]["data_source"])
                                                                                )
        else:


            local_persist_manager_lake = DataLakePersistManager(local_hash_id=self.local_hash_id,
                                                                        class_name=self.__class__.__name__,
                                                                        human_readable=f"Local API Lake for {self.local_hash_id}",
                                                                        logger=self.logger,
                                                                        local_metadata=local_metadata,
                                                                        description=f"Local API Lake for {self.local_hash_id}",
                                                                        data_source=self.data_source)
            table_exist_locally = local_persist_manager_lake.table_exist(local_metadata["remote_table"]["hash_id"])
            if table_exist_locally:
                self.data_source_id=local_persist_manager_lake.data_source.id
                self._local_persist_manager=local_persist_manager_lake
                self._set_logger(local_hash_id=self.local_hash_id)


    def get_df_between_dates(self, start_date: Union[datetime.datetime, None] = None,
                             end_date: Union[datetime.datetime, None] = None,
                             asset_symbols: Union[None, list] = None,
                             data_lake_force_db_look=False, great_or_equal=True, less_or_equal=True,
                             ):

        filtered_data = self.local_persist_manager.get_df_between_dates(start_date=start_date,
                                                                        end_date=end_date,
                                                                        asset_symbols=asset_symbols,
                                                                        great_or_equal=great_or_equal,
                                                                        less_or_equal=less_or_equal,
                                                                        )

        return filtered_data

    def filter_by_assets_ranges(self, asset_ranges_map: dict):
        """

        Parameters
        ----------
        asset_ranges

        Returns
        -------

        """
        df = self.local_persist_manager.filter_by_assets_ranges(asset_ranges_map,time_serie=self)
        return df


    @property
    def pickle_path(self):
        pp = data_source_dir_path(self.data_source_id)
        path = f"{pp}/{self.PICKLE_PREFIFX}{self.local_hash_id}.pickle"
        return path

    @classmethod
    def get_pickle_path(cls, local_hash_id, data_source_id: int):
        return f"{ogm.pickle_storage_path}/{data_source_id}/{cls.PICKLE_PREFIFX}{local_hash_id}.pickle"

    def persist_to_pickle(self,overwrite=False):
        path = self.pickle_path
        # after persisting pickle , build_hash and source code need to be patched
        self.logger.info(f"Persisting pickle")

        pp = data_source_pickle_path(self.data_source_id)
        if os.path.isfile(pp) == False or overwrite == True:

            self.data_source.persist_to_pickle(pp)

        if os.path.isfile(path) == False or overwrite == True:
            with open(path, 'wb') as handle:
                cloudpickle.dump(self, handle)
        return path, path.replace(ogm.pickle_storage_path + "/", "")

    def get_update_statistics(self, asset_symbols: Union[list, None] = None,
                              ) -> datetime.datetime:
        """
        getts latest value directly from querying the DB,
        args and kwargs are nedeed for datalake
        Parameters
        ----------
        args :
        kwargs :

        Returns
        -------

        """

        last_update_in_table, last_update_per_asset = self.local_persist_manager.get_update_statistics(
            remote_table_hash_id=self.remote_table_hashed_name,
            asset_symbols=asset_symbols, time_serie=self)
        return last_update_in_table, last_update_per_asset
    def get_earliest_updated_asset_filter(self, asset_symbols:Union[list],
                                          last_update_per_asset:dict):
        if asset_symbols is not None:
            last_update_in_table = min(
                [t for a in last_update_per_asset.values() for t in a.values() if a in asset_symbols])
        else:
            last_update_in_table = min([t for a in last_update_per_asset.values() for t in a.values()])
        return last_update_in_table

    def update_series_from_source(self, latest_value: Union[None, datetime.datetime], *args, **kwargs) -> pd.DataFrame:
        self.logger.info("Not updating series")
        pass
    def persist_data_to_local_lake(self, temp_df, update_tracker,
                             latest_value: Union[None, datetime.datetime],
                             overwrite=False) -> bool:
        """
        Helper series to  persist data to a local lake for reading purposes
        :param temp_df:
        :param update_tracker:
        :param latest_value:
        :param overwrite:
        :return:
        """
        persisted = False
        if temp_df.shape[0] > 0:
            if overwrite == True:
                self.logger.warning(f"Values will be overwritten assuming latest value of  {latest_value}")

            if isinstance(self.local_persist_manager, DataLakePersistManager)==False:
                local_metadata = TimeSerieLocalUpdate.get(local_hash_id=self.local_hash_id)
                local_persist_manager = DataLakePersistManager(local_hash_id=self.local_hash_id,

                                                                    class_name=local_metadata["build_configuration"][
                                                                        "time_series_class_import_path"]["qualname"],
                                                                    human_readable=f"Local API Lake for {self.local_hash_id}",
                                                                    logger=self.logger,
                                                                    local_metadata=local_metadata,
                                                                    description=f"Local API Lake for {self.local_hash_id}",
                                                                    data_source=self.data_source)
                local_persist_manager.persist_updated_data(temp_df=temp_df,
                                                                update_tracker=update_tracker,
                                                                historical_update_id=None,
                                                                overwrite=overwrite)
            else:
                self.local_persist_manager.persist_updated_data(temp_df=temp_df,
                                                                update_tracker=update_tracker,
                                                                historical_update_id=None,
                                                                overwrite=overwrite)
            persisted = True
        return persisted


class TimeSerie(DataPersistanceMethods, GraphNodeMethods, TimeSerieRebuildMethods):
    """
    Pipeline


        -__init__
        - _create_config

        - _init_db_properties_config
        - set_graph node

    """

    @staticmethod
    def get_time_serie_source_code(TimeSerieClass: "TimeSerie"):
        return inspect.getsource(TimeSerieClass)

    @staticmethod
    def get_time_serie_source_code_git_hash(TimeSerieClass: "TimeSerie"):
        """
        Hashes a time serie source code
        Returns
        -------

        """
        time_serie_class_source_code = TimeSerieClass.get_time_serie_source_code(TimeSerieClass)
        # Prepare the content for Git-style hashing
        # Git hashing format: "blob <size_of_content>\0<content>"
        content = f"blob {len(time_serie_class_source_code)}\0{time_serie_class_source_code}"
        # Compute the SHA-1 hash (Git hash)
        hash_object = hashlib.sha1(content.encode('utf-8'))
        git_hash = hash_object.hexdigest()
        return git_hash

    def _post_init_routines():
        def wrap(init_method):
            @wraps(init_method)
            def run_init(*args, **kwargs):
                import inspect
                expected_arguments = [i for i in inspect.getfullargspec(init_method).args if i != "self"]
                signature = inspect.signature(init_method)
                default_arguments = {k: v.default
                                     for k, v in signature.parameters.items()
                                     if v.default is not inspect.Parameter.empty}
                post_init_log_messages = []
                for a in expected_arguments:
                    if a not in kwargs.keys():
                        if a in default_arguments.keys():
                            kwargs[a] = default_arguments[a]
                            post_init_log_messages.append(
                                f' In {args[0].__class__.__name__} Used default value for {a} in __init__({a}={default_arguments[a]} ,*args,**kwargs)')
                        else:
                            raise Exception(
                                f' In {args[0].__class__.__name__} explicitly declare argument {a} in __init__({a}= ,*args,**kwargs')
                init_method(*args, **kwargs)

                self = args[0]

                kwargs["time_series_class_import_path"] = {"module": self.__class__.__module__,
                                                           "qualname": self.__class__.__qualname__}



                self._create_config(kwargs=kwargs, post_init_log_messages=post_init_log_messages)
                # create logger
                self.run_after_post_init_routines()

                # patch if necesary build configuration
                self.patch_build_configuration()

            return run_init

        return wrap

    @staticmethod
    def sanitize_default_build_metadata(build_meta_data: Union[dict, None] = None):
        if build_meta_data is None:
            build_meta_data = {"initialize_with_default_partitions": False}

        if "initialize_with_default_partitions" not in build_meta_data.keys():
            build_meta_data["initialize_with_default_partitions"] = False
        return build_meta_data

    def __init__(self, init_meta=None,
                 build_meta_data: Union[dict, None] = None, local_kwargs_to_ignore: Union[List[str], None] = None,

                 *args, **kwargs):
        """
        Initializes the TimeSerie object with the provided metadata and configurations.

        This method sets up the time series object, loading the necessary configurations
        and metadata. If `is_local_relation_tree_set` is True, it avoids recalculating the
        relationship tree in schedulers, optimizing the process if the tree is already
        calculated during initialization.

        Parameters
        ----------
        init_meta : dict, optional
            Metadata for initializing the time series instance.
        build_meta_data : dict, optional
            Metadata related to the building process of the time series.
        local_kwargs_to_ignore : list, optional
            List of keyword arguments to ignore during configuration.
        *args : tuple
            Additional arguments.
        **kwargs : dict
            Additional keyword arguments.
        """

        self.init_meta = init_meta
        self.build_meta_data = self.sanitize_default_build_metadata(build_meta_data)
        self.local_kwargs_to_ignore = local_kwargs_to_ignore

        self.pre_load_routines_run = False

        # asser that method is decorated
        if not len(self.__init__.__closure__) == 1:
            logger.error("init method is not decorated with @TimeSerie._post_init_routines()")
            raise Exception
        # create logger

    def get_html_description(self) -> Union[str, None]:
        """
        must return a descript on html tags so it can be readable and rendedered
        Returns:

        """
        description = f"""<p>Time Serie Instance of {self.__class__.__name__} updating table {self.remote_table_hashed_name}</p>"""
        return description

    @property
    def multi_index_asset_symbols_filter(self):
        if hasattr(self, "asset_symbols_filter"):
            return self.asset_symbols_filter
        return None

    @property
    def hash_id(self):
        """

        Returns:

        """
        return self.remote_table_hashed_name
    @property
    def data_source_id(self):
        return self.data_source.id
    @property
    def local_hash_id(self):
        return self.hashed_name

    def _run_pre_load_routines(self):
        """
        Override this method to execute after load and before dependencies are updated
        Returns
        -------

        """
        self.pre_load_routines_run = True

    def get_data_source_from_orm(self):
        from mainsequence.tdag_client import POD_DEFAULT_DATA_SOURCE

        pod_source = os.environ.get("POD_DEFAULT_DATA_SOURCE", None)

        if pod_source == None:
            if POD_DEFAULT_DATA_SOURCE.related_resource is None:
                raise Exception("This Pod does not have a default data source")
            return POD_DEFAULT_DATA_SOURCE
        # returnn to pod to override with context manager
        from mainsequence.tdag_client import models as models
        pod_source = json.loads(pod_source)
        ModelClass = pod_source["tdag_orm_class"]
        pod_source.pop("tdag_orm_class", None)
        ModelClass = getattr(models, ModelClass)
        pod_source = ModelClass(**pod_source)
        return pod_source

    @property
    def data_source(self):
        if hasattr(self, "_data_source"):
            return self._data_source
        else:
            raise Exception("Data source has not been set")

    def set_data_source(self,data_source:Union[object,None]=None):
        """

        :return:
        """
        if data_source is None:
            self._data_source = self.get_data_source_from_orm()
        else:
            self._data_source = data_source

    def run_after_post_init_routines(self):
        pass

    def __repr__(self):
        try:
            local_id=self.local_metadata["id"]
        except:
            local_id = 0
        repr = self.__class__.__name__ + f" {os.environ['TDAG_ENDPOINT']}/local-time-series/details/?local_time_serie_id={local_id}"
        return repr

    def _get_target_time_index(self, idx):
        raise NotImplementedError

    @property
    def human_readable(self):
        return None

    @property
    def update_uses_parallel(self):
        meta_data = self.meta_data
        if "update_uses_parallel" in meta_data:
            return meta_data["update_uses_parallel"]
        else:
            return False

    def plot_relation_tree(self):
        # from fdaccess.time_seriesV2.drawing import build_data_for_tree
        # tree = self.get_relation_tree(as_object=True)
        # data = build_data_for_tree(tree=tree)
        # return data
        raise NotImplementedError

    def _sanitize_init_meta(self, kwargs) -> dict:
        """
        Handles Initial Configuration in preconfig process
        :return:
        """

        if "init_meta" in kwargs.keys():
            init_meta = kwargs["init_meta"] if kwargs["init_meta"] is not None else {}
            kwargs.pop("init_meta", None)
        else:
            init_meta = {}
        return kwargs, init_meta

    def _set_logger(self, local_hash_id):
        if hasattr(self, "logger") == False:
            self.logger = create_logger_in_path(logger_name=f"{self.data_source.id}_{local_hash_id}", application_name="tdag",
                                                logger_file=f'{ogm.get_logging_path()}/{self.data_source.id}/{local_hash_id}.log',
                                                local_hash_id=local_hash_id, data_source_id=self.data_source.id
                                                )
    @staticmethod
    def get_logger_from_context(logger_context:dict):
        local_hash_id = logger_context['local_hash_id']
        data_source_id =logger_context['data_source_id']
        logger=create_logger_in_path(logger_name=f"{local_hash_id}_{data_source_id}", application_name="tdag",
                              logger_file=f'{ogm.get_logging_path()}/{data_source_id}/{local_hash_id}.log',
                              local_hash_id=local_hash_id, data_source_id=data_source_id
                              )
        return logger

    def run_local_update(self, tdag_detached=True):
        self.local_persist_manager
        return self.get_df_between_dates()

    def run(self, debug_mode:bool, *, update_tree:bool=True, force_update:bool=False,
            update_only_tree:bool=False, remote_scheduler:Union[object,None]=None):
        """

        Args:
            debug_mode:
            update_tree:
            force_update:
            update_only_tree:
            remote_scheduler:

        Returns:

        """
        from mainsequence.tdag.instrumentation import TracerInstrumentator
        from mainsequence.tdag.config import configuration
        from mainsequence.tdag.time_series.update.utils import UpdateInterface, wait_for_update_time
        from mainsequence.tdag.time_series.update.ray_manager import RayUpdateManager
        from mainsequence.tdag_client import Scheduler
        import gc
        if update_tree:
            update_only_tree = False

        # set tracing
        tracer_instrumentator = TracerInstrumentator(
            configuration.configuration["instrumentation_config"]["grafana_agent_host"])
        tracer = tracer_instrumentator.build_tracer("tdag_head_distributed", __name__)
        error_on_update= None
        # 1 Create Scheduler for this time serie

        if remote_scheduler == None:
            name_prefix = "DEBUG_" if debug_mode else ""
            scheduler = Scheduler.build_and_assign_to_ts(
                scheduler_name=f"{name_prefix}{self.local_hash_id}_{self.data_source.id}",
                local_hash_id_list=[(self.local_hash_id, self.data_source.id)],
                remove_from_other_schedulers=True,
                running_in_debug_mode=debug_mode

            )
            scheduler.start_heart_beat()
        else:
            scheduler = remote_scheduler

        error_to_raise = None
        with tracer.start_as_current_span(f"Scheduler TS Head Update ") as span:
            span.set_attribute("time_serie_local_hash_id", self.local_hash_id)
            span.set_attribute("remote_table_hashed_name", self.remote_table_hashed_name)
            span.set_attribute("head_scheduler", scheduler.name)

            # 2 add actor manager for distributed
            distributed_actor_manager = RayUpdateManager(scheduler_uid=scheduler.uid,
                                                         skip_health_check=True
                                                         )
            try:
                local_metadatas, state_data = self.pre_update_setting_routines(scheduler=scheduler,
                                                                               set_time_serie_queue_status=False,
                                                                               update_tree=update_tree)
                self.set_actor_manager(actor_manager=distributed_actor_manager)
                self.logger.info("state set with dependencies metadatas")

                # 3 build update tracker
                update_tracker = UpdateInterface(head_hash=self.local_hash_id, trace_id=None,
                                                 logger=self.logger,
                                                 state_data=state_data, debug=debug_mode,
                                                 )
                self.update_tracker = update_tracker
                if force_update == False:
                    wait_for_update_time(local_hash_id=self.local_hash_id,
                                                          data_source_id=self.data_source.id,
                                                          logger=self.logger,
                                                          force_next_start_of_minute=False)
                error_on_update = self.update(debug_mode=debug_mode,
                                              raise_exceptions=True,
                                              force_update=force_update,
                                              update_tree=update_tree,
                                              update_only_tree=update_only_tree,
                                              metadatas=local_metadatas,
                                              update_tracker=self.update_tracker,
                                              use_state_for_update=True
                )
                del self.update_tracker
                gc.collect()
            except TimeoutError as te:
                self.logger.error("TimeoutError Error on update")
                error_to_raise = te
            except DependencyUpdateError as de:
                self.logger.error("DependecyError on update")
                error_to_raise = de
            except Exception as e:
                self.logger.exception(e)
                error_to_raise = e

        if remote_scheduler == None:
            scheduler.stop_heart_beat()
        if error_to_raise != None:
            raise error_to_raise






    
    
    @property
    def local_persist_manager(self):
        if hasattr(self, "logger") == False:
            self._set_logger(local_hash_id=self.hashed_name)
        if hasattr(self, "_local_persist_manager") == False:
            self.logger.info(f"Setting local persist manager for {self.hash_id}")
            self._set_local_persist_manager(local_hash_id=self.local_hash_id,
                                            remote_table_hashed_name=self.remote_table_hashed_name
                                            )
        return self._local_persist_manager

    def _set_local_persist_manager(self, local_hash_id: str, remote_table_hashed_name: str,
                                   local_metadata: Union[None, dict] = None,
                                   time_serie_meta_build_configuration: Union[None, dict] = None,
                                   verify_local_run=True,
                                   ) -> None:
        """
        Initializes the local persistence manager for the time series. It sets up
        the necessary configurations and checks for existing metadata. If the metadata doesn't
        exist or is incomplete, it sets up the initial configuration and builds the update details.

        Args:
           hashed_name : str
               The local hash ID for the time series.
           remote_table_hashed_name : str
               The remote table hash name for the time series.
           local_metadata : Union[None, dict], optional
               Local metadata for the time series, if available.
        """
        from .persist_managers import DataLakePersistManager
        try:
            human_readable = self.human_readable
        except:
            human_readable = None

        self._local_persist_manager = PersistManager.get_from_data_type(local_hash_id=local_hash_id,

                                                                        class_name=self.__class__.__name__,
                                                                        human_readable=human_readable,
                                                                        logger=self.logger,
                                                                        local_metadata=local_metadata,
                                                                        description=self.get_html_description(),
                                                                        data_source=self.data_source,

                                                                        )
        if isinstance(self._local_persist_manager, DataLakePersistManager) and verify_local_run:
            self._local_persist_manager.verify_if_already_run(self)
        self._verify_and_build_remote_objects()

    @none_if_backend_detached
    def _verify_and_build_remote_objects(self):

        time_serie_source_code_git_hash = self.get_time_serie_source_code_git_hash(self.__class__)
        time_serie_source_code = self.get_time_serie_source_code(self.__class__)
        remote_meta_exist, local_meta_exist = self._local_persist_manager.local_persist_exist_set_config(
            remote_table_hashed_name=self.remote_table_hashed_name,
            local_configuration=self.local_initial_configuration,
            remote_configuration=self.remote_initial_configuration,
            remote_build_metadata=self.remote_build_metadata,
            time_serie_source_code_git_hash=time_serie_source_code_git_hash,
            time_serie_source_code=time_serie_source_code,
            data_source=self.data_source,

        )
        if remote_meta_exist == False or local_meta_exist == False:

            # should be on creation
            update_details = self.local_persist_manager.update_details_exist() if local_meta_exist == True else {
                "remote_exist": remote_meta_exist, "local_exist": local_meta_exist}
            if update_details["remote_exist"] == False or update_details["local_exist"] == False:
                self.set_relation_tree()
                self.local_persist_manager.build_update_details(source_class_name=self.__class__.__name__)

        if self.local_persist_manager.metadata['human_readable'] is None and self.human_readable is not None:
            self.local_persist_manager.patch_table(human_readable=self.human_readable)

        if self.local_persist_manager.update_details is not None:
            source_class_name = self.local_persist_manager.metadata["source_class_name"]
            if source_class_name is None or source_class_name == "":
                # patch class name
                self.local_persist_manager.patch_update_details(source_class_name=self.__class__.__name__)

    def flush_pickle(self):
        if os.path.isfile(self.pickle_path):
            os.remove(self.pickle_path)

    @none_if_backend_detached
    def patch_build_configuration(self):
        """
        This method comes in handy when there is a change in VAM models extra configuration. This method will properly
        update the models on all the tree
        Returns
        -------

        """
        patch_build = os.environ.get("PATCH_BUILD_CONFIGURATION", False) in ["true", "True", 1]
        if patch_build == True:
            self.local_persist_manager  # just call it before to initilaize dts
            self.logger.warning(f"Patching build configuration for {self.hash_id}")
            self.flush_pickle()

            self.local_persist_manager.patch_build_configuration(local_configuration=self.local_initial_configuration,
                                                                 remote_configuration=self.remote_initial_configuration,
                                                                 remote_build_metadata=self.remote_build_metadata
                                                                 )

        if self.local_persist_manager.metadata['sourcetableconfiguration'] is not None:
            if self.remote_build_metadata["initialize_with_default_partitions"] == False:
                if self.data_source.data_type == CONSTANTS.DATA_SOURCE_TYPE_TIMESCALEDB:
                    self.logger.warning("Default Partitions will be initialized ")

    def _create_config(self, kwargs, post_init_log_messages: list):
        """
        This methods executes  after serialization
        :param kwargs:
        :return:
        """

        kwargs, init_meta = self._sanitize_init_meta(kwargs=kwargs)
        config_serializer = ConfigSerializer()

        kwargs = config_serializer.serialize_init_kwargs(kwargs)

        # hash config dict
        local_ts_hash, remote_table_hash = hash_signature(kwargs)
        local_hashed_name = self.__class__.__name__ + "_" + local_ts_hash
        local_hashed_name = local_hashed_name.lower()

        remote_hashed_name = self.__class__.__name__ + "_" + remote_table_hash
        remote_hashed_name = remote_hashed_name.lower()

        self.set_data_source()
        self._set_logger(local_hash_id=local_hashed_name)

        for m in post_init_log_messages:
            self.logger.warning(m)

        if len(remote_hashed_name) > 60:
            self.logger.error(f"hashed name {remote_hashed_name} is to long {len(remote_hashed_name)} limit i 60 ")
            raise AssertionError
        self.hashed_name = local_hashed_name
        self.remote_table_hashed_name = remote_hashed_name
        self.local_initial_configuration = kwargs
        remote_initial_configuration = copy.deepcopy(kwargs)
        if "local_kwargs_to_ignore" in remote_initial_configuration:
            for k in remote_initial_configuration["local_kwargs_to_ignore"]:
                remote_initial_configuration.pop(k, None)
            remote_initial_configuration.pop("local_kwargs_to_ignore", None)
        self.remote_initial_configuration = remote_initial_configuration
        self.remote_build_metadata = self.sanitize_default_build_metadata(kwargs.get("build_meta_data", None))
        self.init_meta = init_meta

        self.logger.debug(f"local/remote {self.hashed_name}/{self.remote_table_hashed_name}")

    def set_update_uses_parallel(self, value):
        self.local_persist_manager.set_meta("update_uses_parallel", value)

    def set_dependencies_df(self):
        """

        :return:
        """

        self.logger.info("Initializing update priority ... ")
        depth_df = self.local_persist_manager.get_all_dependencies_update_priority()
        self.depth_df = depth_df

        self.dependencies_df = depth_df.copy()

        if depth_df.shape[0] > 0:
            self.dependencies_df = self.depth_df[self.depth_df["local_hash_id"] != self.hashed_name].copy()

    def pre_update_setting_routines(self, scheduler, set_time_serie_queue_status: bool, update_tree: True,
                                    metadata: Union[dict, None] = None, local_metadata: Union[dict, None] = None
                                    ):
        """
        Routines to execute previous to an update
        Returns
        -------

        """

        # set scheduler
        update_details = {}
        # reset metadata
        self.local_persist_manager.synchronize_metadata(meta_data=metadata, local_metadata=local_metadata)
        self.set_relation_tree()
        if hasattr(self, "logger") == False:
            self._set_logger(self.hashed_name)

        update_priority_dict = None
        # build priority update

        if hasattr(self, "depth_df") == False:
            self.set_dependencies_df()

            self.logger.info("Setting dependencies in scheduler active_tree")
            # only set once
            all_hash_id_in_tree = []

            if self.depth_df.shape[0] > 0:
                all_hash_id_in_tree = self.depth_df["local_hash_id"].to_list()
                if update_tree == True:
                    scheduler.in_active_tree_connect(hash_id_list=all_hash_id_in_tree + [self.local_hash_id])

        depth_df = self.depth_df.copy()
        # set active tree connections
        all_hash_id_in_tree = []

        if self.depth_df.shape[0] > 0:
            all_hash_id_in_tree = depth_df[["local_hash_id","data_source_id"]].to_dict("records")
            assert depth_df.groupby("local_hash_id").count().max().max() < 2
        all_hash_id_in_tree.append({"local_hash_id":self.local_hash_id,"data_source_id":self.data_source.id})

        update_details_batch = dict(error_on_last_update=False,
                                    active_update_scheduler_uid=scheduler.uid)
        if set_time_serie_queue_status == True:
            update_details_batch['active_update_status'] = "Q"
        all_metadatas = self.get_metadatas_and_set_updates(local_hash_id__in=all_hash_id_in_tree,
                                                           multi_index_asset_symbols_filter=self.multi_index_asset_symbols_filter,
                                                           update_details_kwargs=update_details_batch,
                                                           update_priority_dict=update_priority_dict,
                                                           )
        state_data, local_metadatas, source_table_config_map = all_metadatas['state_data'], all_metadatas[
            "local_metadatas"], all_metadatas["source_table_config_map"]
        local_metadatas = {(m["local_hash_id"],m["remote_table"]["data_source"]["id"]): m for m in local_metadatas}

        self.scheduler = scheduler
        self.update_details_tree = {key: v["localtimeserieupdatedetails"] for key, v in local_metadatas.items()}
        return local_metadatas, state_data

    @tracer.start_as_current_span("Verify time series tree update")
    def _verify_tree_is_updated(self, metadatas: dict, debug_mode:bool,
                                use_state_for_update:bool=False
                                ):
        """

        Args:
            metadatas:
            debug_mode:
            use_state_for_update: timeseries will be collected from code introspection

        Returns:

        """
        # build tree
        if self.is_local_relation_tree_set == False:
            start_tree_relationship_update_time = time.time()
            self.set_relation_tree()
            self.logger.info(
                f"relationship tree updated took {time.time() - start_tree_relationship_update_time} seconds ")

        else:
            self.logger.info("Tree is not updated as is_local_relation_tree_set== True")

        update_map = {}
        if use_state_for_update == True:
            update_map = self.get_update_map()

        if debug_mode == False:
            tmp_ts = self.dependencies_df.copy()
            if tmp_ts.shape[0] == 0:
                return None
            tmp_ts = tmp_ts[tmp_ts["source_class_name"] != "WrapperTimeSerie"]

            if tmp_ts.shape[0] > 0:
                self._execute_parallel_distributed_update(tmp_ts=tmp_ts,
                                                          metadatas=metadatas)
        else:
            updated_uids = []
            if self.dependencies_df.shape[0] > 0:
                unique_priorities = self.dependencies_df["update_priority"].unique().tolist()
                unique_priorities.sort()

                local_time_series_list = self.dependencies_df[
                    self.dependencies_df["source_class_name"] != "WrapperTimeSerie"
                    ][["local_hash_id", "data_source_id"]].values.tolist()
                all_start_data = self.update_tracker.set_start_of_execution_batch(local_time_series_list=local_time_series_list)
                for prioriity in unique_priorities:
                    # get hierarchies ids
                    tmp_ts = self.dependencies_df[self.dependencies_df["update_priority"] == prioriity].sort_values(
                        "number_of_upstreams", ascending=False).copy()

                    tmp_ts = tmp_ts[tmp_ts["source_class_name"] != "WrapperTimeSerie"]
                    tmp_ts = tmp_ts[~tmp_ts.index.isin(updated_uids)]

                    # update on the same process
                    for row, ts_row in tmp_ts.iterrows():

                        if (ts_row["local_hash_id"], ts_row["data_source_id"]) in update_map.keys():
                            ts = update_map[(ts_row["local_hash_id"], ts_row["data_source_id"])]["ts"]
                        else:
                            try:
                                pickle_path = self.get_pickle_path(ts_row["local_hash_id"],
                                                                   data_source_id=ts_row["data_source_id"])
                                data_source = self.load_data_source_from_pickle(pickle_path=pickle_path)
                                if os.path.isfile(pickle_path) == False:
                                    ts = TimeSerie.rebuild_from_configuration(local_hash_id=ts_row["local_hash_id"],
                                                                              data_source=data_source
                                                                              )
                                    ts.persist_to_pickle()

                                ts = TimeSerie.load_and_set_from_pickle(pickle_path=pickle_path)
                            except Exception as e:
                                self.logger.exception(f"Error updating dependencie {ts.local_hash_id} when loading pickle")
                                raise e

                        try:
                            self.update_tracker.set_start_of_execution(local_hash_id=ts_row["local_hash_id"],
                                                                       data_source_id=ts_row["data_source_id"]
                                                                       )
                            error_on_last_update = ts.update(debug_mode=debug_mode,
                                                             raise_exceptions=True,
                                                             update_tree=False,
                                                             start_update_data=all_start_data[
                                                                 (ts_row["local_hash_id"], ts_row["data_source_id"])],
                                                             update_tracker=self.update_tracker
                                                             )

                            if error_on_last_update == True:
                                raise Exception(f"Error updating dependencie {ts.local_hash_id} pipeline stopped")
                        except Exception as e:
                            self.logger.exception(f"Error updating dependencie {ts.local_hash_id}")
                            raise e
                    updated_uids.extend(tmp_ts.index.to_list())
        self.logger.info(f'Dependency Tree evaluated for  {self}')

    def set_actor_manager(self, actor_manager: object):
        self.update_actor_manager = actor_manager

    @tracer.start_as_current_span("Execute distributed parallel update")
    def _execute_parallel_distributed_update(self, tmp_ts: pd.DataFrame,
                                             metadatas: Union[dict, None],
                                             ):

        telemetry_carrier = tracer_instrumentator.get_telemetry_carrier()

        pre_loaded_ts = [t.hash_id for t in self.scheduler.pre_loads_in_tree]
        tmp_ts = tmp_ts.sort_values(["update_priority", "number_of_upstreams"], ascending=[False, False])
        pre_load_df = tmp_ts[tmp_ts["local_hash_id"].isin(pre_loaded_ts)].copy()
        tmp_ts = tmp_ts[~tmp_ts["local_hash_id"].isin(pre_loaded_ts)].copy()
        tmp_ts = pd.concat([pre_load_df, tmp_ts], axis=0)

        start_update_date = datetime.datetime.now(pytz.utc)

        futures_ = []

        local_time_series_list = self.dependencies_df[
            self.dependencies_df["source_class_name"] != "WrapperTimeSerie"
            ][["local_hash_id", "data_source_id"]].values.tolist()
        all_start_data = self.update_tracker.set_start_of_execution_batch(local_time_series_list=local_time_series_list)

        for counter, (uid, data) in enumerate(tmp_ts.iterrows()):

            local_hash_id = data['local_hash_id']
            data_source_id = data['data_source_id']
            start_update_data = all_start_data[(local_hash_id, data_source_id)]
            if start_update_data.must_update == False:
                continue
            kwargs_update = dict(local_hash_id=local_hash_id,
                                 execution_start=start_update_date,
                                 telemetry_carrier=telemetry_carrier,
                                 local_metadatas=metadatas,
                                 start_update_data=start_update_data,

                                 )

            update_details = self.update_details_tree[(local_hash_id, data_source_id)]
            num_cpus = update_details['distributed_num_cpus']

            p = self.update_actor_manager.launch_update_task(task_options={"num_cpus": num_cpus,
                                                                           "name": local_hash_id,
                                                                           "max_retries": 0},
                                                             kwargs_update=kwargs_update
                                                             )
            # p = self.update_actor_manager.launch_update_task_in_process(task_options={"num_cpus": num_cpus,
            #                                                                "name": local_hash_id,
            #
            #                                                                "max_retries": 0},
            #                                                  kwargs_update=kwargs_update
            #                                                  )
            futures_.append(p)

            # are_dependencies_updated, all_dependencies_nodes, pending_nodes, error_on_dependencies = self.update_tracker.get_pending_update_nodes(
            #     hash_id_list=list(all_start_data.keys()))
            # self.are_dependencies_updated( target_nodes=all_dependencies_nodes)
            # raise Exception

        # block update until termination
        try:
            self._run_pre_load_routines()
        except Exception as e:
            self.logger.exception("Error running pre_load_routines")
            error_on_dependencies = True

        tasks_with_errors = self.update_actor_manager.get_results_from_futures_list(futures=futures_)
        if len(tasks_with_errors) > 0:
            raise DependencyUpdateError(f"Update Stop from error in Ray in tasks {tasks_with_errors}")
        # verify there is no error in hierarchy. this prevents to updating next level if dependencies fails

        dependencies_update_details = LocalTimeSerieUpdateDetails.filter(
            related_table__local_hash_id__in=tmp_ts["local_hash_id"].to_list())
        ts_with_errors = []
        for local_ts_details in dependencies_update_details:
            if local_ts_details["error_on_last_update"] == True:
                ts_with_errors.append(local_ts_details["related_table__local_hash_id__in"])
        # Verify there are no errors after finishing hierarchy
        if len(ts_with_errors) > 0:
            raise DependencyUpdateError(f"Update Stop from error in children \n {ts_with_errors}")



    @tracer.start_as_current_span("TimeSerie.update_local")
    def update_local(self, update_tree, update_tracker: object, debug_mode: bool,
                     metadatas: Union[None, dict] = None,
                     overwrite_latest_value: Union[datetime.datetime, None] = None, update_only_tree: bool = False,
                     use_state_for_update:bool=False,
                     *args, **kwargs) -> bool:

        from mainsequence.tdag.instrumentation.utils import Status, StatusCode
        persisted = False
        if update_tree == True:

            self._verify_tree_is_updated(debug_mode=debug_mode,
                                         metadatas=metadatas,
                                         use_state_for_update=use_state_for_update,
                                         )
            update_only_tree == True
            if update_only_tree == True:
                self.logger.info(f'Local Time Series  {self}   only tree updated')
                return None

        # hardcore check to fix missing values
        # from mainsequence.tdag_client.utils import read_one_value_from_table
        #
        # if self.local_persist_manager.metadata["sourcetableconfiguration"] is None:
        #     r=read_one_value_from_table(self.hashed_name)
        #     if len(r)>0:
        #         #there is data but not source table configuration overwrite
        #         overwrite_latest_value=datetime.datetime(2023,6,23).replace(tzinfo=pytz.utc)

        with tracer.start_as_current_span("Update Calculation") as update_span:

            if overwrite_latest_value is not None:  # overwrite latest values is passed form def_update method to reduce calls to api
                latest_value = overwrite_latest_value

                self.logger.info(f'Updating Local Time Series for  {self}  since {latest_value}')
                temp_df = self.update_series_from_source(latest_value=latest_value,

                                                         **kwargs)

                if temp_df.shape[0] == 0:
                    # concatenate empty

                    self.logger.info(f'Local Time Series Nothing to update  {self}  updated')
                    return False

                for col, ddtype in temp_df.dtypes.items():
                    if "datetime64" in str(ddtype):
                        self.logger.info(f"WARNING DATETIME TYPE IN {self}")
                        raise Exception(f"""Datetime in {col}
                                            {temp_df}""")
                self.logger.info(f'Persisting Time Series for  {self}  since {latest_value} ')




            else:
                latest_value, last_multiindex = self.get_update_statistics()
                if latest_value is None:
                    self.logger.info(f'Updating Local Time Series for  {self}  for first time')

                temp_df = self.update_series_from_source(latest_value=latest_value,
                                                         **kwargs)
                for col, ddtype in temp_df.dtypes.items():
                    if "datetime64" in str(ddtype):
                        self.logger.info(f"WARNING DATETIME TYPE IN {self}")
                        raise Exception

            try:

                # verify index order is correct
                overwrite = True if overwrite_latest_value is not None else False
                persisted = self.persist_updated_data(temp_df,
                                                      update_tracker=update_tracker,
                                                      latest_value=latest_value, overwrite=overwrite)

                update_span.set_status(Status(StatusCode.OK))
            except Exception as e:
                self.logger.exception("Error updating time serie")
                update_span.set_status(Status(StatusCode.ERROR))
                raise e
            self.logger.info(f'Local Time Series  {self}  updated')

            return persisted

    def _run_post_update_routines(self, error_on_last_update: bool):
        pass

    def update_series_from_source(self, latest_value: Union[None, datetime.datetime], *args, **kwargs) -> pd.DataFrame:
        """
        This method performs all the necessary logic to update our time series. The method should always return a DataFrame with the following characteristics:

        1) A unidimensional index where the index is of the type `DatetimeIndex` and the dates are in `pytz.utc`.
        2) A multidimensional index that should always have 3 dimensions: `time_index` (with the same characteristics as before), `asset_symbol`, and `execution_venue_symbol`.

        Parameters
        ----------
        latest_value
        args
        kwargs

        Returns
        -------

        """

        raise NotImplementedError


class WrapperTimeSerie(TimeSerie):
    """A wrapper class for managing multiple TimeSerie objects."""

    @TimeSerie._post_init_routines()
    def __init__(self, time_series_dict: Dict[str, TimeSerie], *args, **kwargs):
        """
        Initialize the WrapperTimeSerie.

        Args:
            time_series_dict: Dictionary of TimeSerie objects.
        """
        super().__init__(*args, **kwargs)
        for key, value in time_series_dict.items():
            if isinstance(value, TimeSerie) == False and "tdag.time_series.time_series.TimeSerie" not in [
                ".".join([o.__module__, o.__name__]) for o in inspect.getmro(value.__class__)]:
                if isinstance(value, APITimeSerie) ==False and "tdag.time_series.time_series.APITimeSerie" not in [
                ".".join([o.__module__, o.__name__]) for o in inspect.getmro(value.__class__)]:
                    logger.error("value is not of class TimeSerie nor APITimeSerie")
                    logger.error(self)
                    raise Exception

        self.related_time_series = time_series_dict

        # todo set minimum update date.

    @property
    def wrapped_latest_index_value(self) -> Dict[str, Any]:
        """
        Get the latest values of all wrapped TimeSeries.

        Returns:
            A dictionary with keys corresponding to TimeSerie keys and values being their latest values.
        """
        updates = {}
        for key, ts in self.related_time_series.items():
            updates[key] = ts.get_update_statistics()

        return updates

    @property
    def wrapper_keys(self) -> List[str]:
        """
        Get the keys of all wrapped TimeSeries.

        Returns:
            A list of keys for all wrapped TimeSeries.
        """
        return list(self.related_time_series.keys())

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """ Restore instance attributes from a pickled state. """

        # Restore instance attributes (i.e., filename and lineno).
        for key, value in state["related_time_series"].items():
            if isinstance(value, dict) == True:
                local_hash_id = value["local_hash_id"]

                module = importlib.import_module(state["_data_source"]["pydantic_model_import_path"]["module"])
                PydanticClass = getattr(module, state["_data_source"]["pydantic_model_import_path"]['qualname'])
                data_source = PydanticClass(**state["_data_source"]['serialized_model'])
                pickle_path = TimeSerie.get_pickle_path(local_hash_id=local_hash_id,
                                                        data_source_id=data_source.id
                                                        )

                state["related_time_series"][key] = load_from_pickle(pickle_path=pickle_path)

        self.__dict__.update(state)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.

        state = self.__dict__
        for key, value in state["related_time_series"].items():
            new_value = {"is_time_serie_pickled": True}
            if isinstance(value, dict):
                assert value["is_time_serie_pickled"] == True
                new_value = value
            else:
                value.persist_to_pickle()
                new_value["local_hash_id"] = value.local_hash_id
            state["related_time_series"][key] = new_value
        state = self._prepare_state_for_pickle(state=state)

        # Remove the unpicklable entries.
        return state

    def set_data_source_from_pickle_path(self, pikle_path):
        data_source = self.load_data_source_from_pickle(pikle_path)
        self.set_data_source(data_source=data_source)

    def set_local_persist_manager_if_not_set(self) -> None:
        """
        Set local persist manager for all wrapped TimeSeries.
        """
        raise NotImplementedError

        def update_ts(related_time_series, key):
            related_time_series[key].set_local_persist_manager_if_not_set()

        with ThreadPoolExecutor(max_workers=20) as executor:
            future_list = []
            for key, value in self.related_time_series.items():
                future = executor.submit(update_ts, self.related_time_series, key)
                future_list.append(future)

            for future in as_completed(future_list):
                # You can optionally handle exceptions here if any
                try:
                    result = future.result()  # This will block until the future is done
                except Exception as e:
                    self.logger.exception("Error in thread")
                    raise e

    def set_state_with_sessions(self, include_vam_client_objects: bool,
                                graph_depth_limit: int,
                                graph_depth: int,
                                local_metadatas: Union[dict, None] = None
                                ) -> None:
        """
        Set state with sessions for all wrapped TimeSeries.

        Args:
            include_vam_client_objects: Whether to include asset ORM objects.
            graph_depth_limit: The maximum depth of the graph to traverse.
            graph_depth: The current depth in the graph.
            local_metadatas: Optional metadata dictionary.
        """

        USE_THREADS = True

        super(TimeSerie, self).set_state_with_sessions(
            include_vam_client_objects=include_vam_client_objects,
            graph_depth_limit=graph_depth_limit,
            local_metadatas=local_metadatas,
            graph_depth=graph_depth)
        errors = {}

        def update_ts(related_time_series, ts_key, include_vam_client_objects,
                      graph_depth,
                      graph_depth_limit, error_list, local_metadatas, rel_ts,raise_exceptions=USE_THREADS):
            if isinstance(rel_ts, dict):
                pickle_path = TimeSerie.get_pickle_path(local_hash_id=rel_ts['local_hash_id'])
                related_time_series[ts_key] = load_from_pickle(pickle_path=pickle_path)
            try:
                if isinstance(related_time_series[ts_key],APITimeSerie):
                    return None
                related_time_series[ts_key].set_state_with_sessions(
                    graph_depth=graph_depth, graph_depth_limit=graph_depth_limit,
                    include_vam_client_objects=include_vam_client_objects,
                    local_metadatas=local_metadatas
                )
            except Exception as e:
                if raise_exceptions ==True:
                    raise e
                tb_str = traceback.format_exc()
                # Store the exception along with its traceback
                error_list[ts_key] = {'error': str(e), 'traceback': tb_str}

        if USE_THREADS == True:
            thread_list = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                for ts_key, rel_ts in self.related_time_series.items():
                    future = executor.submit(update_ts, self.related_time_series, ts_key,
                                             include_vam_client_objects, graph_depth,
                                             graph_depth_limit, errors, local_metadatas,
                                             rel_ts)

                    thread_list.append(future)
                for future in as_completed(thread_list):
                    # You can optionally handle exceptions here if any
                    try:
                        result = future.result()  # This will block until the future is done
                    except Exception as e:
                        self.logger.exception("Error in thread")
                        raise e
        else:
            self.logger.warning("NOT using threads for  loading state")
            for ts_key, rel_ts in self.related_time_series.items():
                t = update_ts(self.related_time_series, ts_key,
                              include_vam_client_objects, graph_depth,
                              graph_depth_limit, errors, local_metadatas, rel_ts,True)

        if len(errors.keys()) > 0:
            raise Exception(f"Error setting state for {errors}")

        # for t in thread_list:
        #     t.join()

    @tracer.start_as_current_span("Wrapper.concat_between_dates")
    def pandas_df_concat_on_rows_by_key_between_dates(self, start_date: Union[datetime.datetime, dict],
                                                      great_or_equal: bool,
                                                      end_date: datetime.datetime, less_or_equal: bool,
                                                      thread: bool = False,
                                                      asset_symbols: Union[None, list] = None,
                                                      return_as_list=False, key_date_filter: Union[dict, None] = None,

                                                      ) -> pd.DataFrame:
        """
         Concatenate DataFrames from all wrapped TimeSeries between given dates.

         Args:
             start_date: The start date for the data range.
             great_or_equal: Whether to include the start date (True) or not (False).
             end_date: The end date for the data range.
             less_or_equal: Whether to include the end date (True) or not (False).
             thread: Whether to use threading for parallel processing.
             asset_symbols: asset_symbol filter
             return_as_list: If True, return a dictionary of DataFrames instead of concatenating.
            key_date_filter: Concatenate DataFrames only for key date filter.
         Returns:
             A concatenated DataFrame or a dictionary of DataFrames if return_as_list is True.
         """
        all_dfs = []
        all_dfs_thread = {}
        thread = True

        def add_ts(ts, key, thread, asset_symbols, key_date_filter):
            data_start_date = start_date
            if isinstance(start_date, dict):
                data_start_date = start_date[key]
            if key_date_filter is not None:
                data_start_date = key_date_filter.get(key, data_start_date)

            tmp_df = ts.get_df_between_dates(start_date=data_start_date, great_or_equal=great_or_equal,
                                             end_date=end_date, less_or_equal=less_or_equal,
                                             asset_symbols=asset_symbols,
            )
            tmp_df["key"] = key
            all_dfs_thread[key] = tmp_df

        if thread == False:
            for key, ts in self.related_time_series.items():
                add_ts(ts, key, thread, asset_symbols, key_date_filter)
        else:

            with ThreadPoolExecutor(max_workers=10) as executor:
                future_list = []
                for key, ts in self.related_time_series.items():
                    future = executor.submit(add_ts, ts, key, thread, asset_symbols, key_date_filter)
                    future_list.append(future)

                for future in as_completed(future_list):
                    # You can optionally handle exceptions here if any
                    try:
                        result = future.result()  # This will block until the future is done
                    except Exception as e:
                        self.logger.exception("Error in thread")
                        raise e

        if return_as_list == False:
            all_dfs = pd.concat(all_dfs_thread.values(), axis=0)
        else:
            all_dfs = all_dfs_thread

        return all_dfs

    @tracer.start_as_current_span("Wrapper.concat_greater_than")
    def pandas_df_concat_on_rows_by_key_greater_than(self, target_value: datetime.datetime, great_or_equal: bool,
                                                     thread: bool = False, return_as_list=False,
                                                     columns: Union[None, list] = None, *args,
                                                     **kwargs) -> pd.DataFrame:
        """
         Concatenate DataFrames from all wrapped TimeSeries greater than a target value.

         Args:
             target_value: The latest datetime value to compare against.
             great_or_equal: Whether to include the target value (True) or not (False).
             thread: Whether to use threading for parallel processing.
             return_as_list: If True, return a dictionary of DataFrames instead of concatenating.
             columns: Optional list of columns to include.

         Returns:
             A concatenated DataFrame or a dictionary of DataFrames if return_as_list is True.

         """
        all_dfs = []
        all_dfs_thread = {}

        def add_ts(ts, key, thread, columns):
            tmp_df = ts.get_df_between_dates(start_date=target_value,
                                                     great_or_equal=great_or_equal,
                                                     columns=columns,
                                                     *args, **kwargs
                                                     )
            tmp_df["key"] = key
            if thread == False:
                return tmp_df
            all_dfs_thread[key] = tmp_df
            if return_as_list == False:
                all_dfs = pd.concat(all_dfs, axis=0)

        if thread == False:

            for key, ts in self.related_time_series.items():
                tmp_df = add_ts(ts, key, thread, columns)
                all_dfs.append(tmp_df)

            all_dfs = pd.concat(all_dfs, axis=0)
        else:
            thread_list = []
            for key, ts in self.related_time_series.items():
                t = threading.Thread(target=add_ts, args=(ts, key, thread, columns))
                t.start()
                thread_list.append(t)
            for t in thread_list:
                t.join()
            all_dfs = pd.concat(all_dfs_thread.values(), axis=0)

        return all_dfs

    def get_pandas_df_list_data_greater_than(self, target_value: datetime.datetime, great_or_equal: bool,
                                             thread=True) -> list:
        """
        Get DataFrames from all wrapped TimeSeries greater than a target value.

        Args:
            target_value: The target datetime value to compare against.
            great_or_equal: Whether to include the target value (True) or not (False).
            thread: Whether to use threading for parallel processing.

        Returns:
            A dictionary with TimeSerie keys and their corresponding DataFrames or error messages.
        """
        thread_list = []

        def get_df(all_dfs, key, ts, target_value, great_or_equal):
            try:
                tmp_df = ts.get_df_between_dates(start_date=target_value, great_or_equal=great_or_equal)
                all_dfs[key] = tmp_df
            except Exception as e:
                all_dfs[key] = "Error"

        all_dfs = {}
        for key, ts in self.related_time_series.items():

            if thread == False:
                get_df(all_dfs, key, ts, target_value, great_or_equal, )
            else:
                t = threading.Thread(target=get_df, args=(all_dfs, key, ts, target_value, great_or_equal,))
                t.start()
                thread_list.append(t)

        for t in thread_list:
            t.join()

        return all_dfs

    def update_series_from_source(self, latest_value, *args, **kwargs):
        """ Implemented in the wrapped nodes"""
        pass

    def __getitem__(self, item):
        return self.related_time_series[item]

    def children_is_updating(self) -> bool:
        """ Check if any wrapped TimeSerie is currently updating. """

        return any([i.active_update for i in self.get_wrapped()])

    def items(self):
        """Get items of wrapped TimeSeries. """
        return self.related_time_series.items()

    def values(self):
        """ Get values of wrapped TimeSeries. """
        return self.related_time_series.values()

    def get_ts_as_pandas(self) -> List[pd.DataFrame]:
        """
         Get all wrapped TimeSeries as a list of pandas DataFrames.

         Returns:
             A list of pandas DataFrames, one for each wrapped TimeSerie.
         """
        pandas_list = []
        for ts in self.related_time_series.values():
            pandas_list.append(ts.pandas_df)
        return pandas_list

    def get_wrapped(self) -> List[TimeSerie]:
        """
        Get all wrapped TimeSeries, including nested ones.

        Returns:
            A list of all wrapped TimeSerie objects, including those nested in other WrapperTimeSeries.
        """
        wrapped = []
        for ts in self.related_time_series.values():
            if isinstance(ts, WrapperTimeSerie):
                tmp_w = ts.get_wrapped()
                wrapped.extend(tmp_w)
            else:
                wrapped.append(ts)
        return wrapped
