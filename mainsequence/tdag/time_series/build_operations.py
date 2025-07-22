
import inspect
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import copy
from pydantic import BaseModel
import json
from mainsequence.client import POD_PROJECT
import os
import importlib
from mainsequence.client.models_helpers import get_model_class
from enum import Enum
from types import SimpleNamespace
from mainsequence.client import BaseObjectOrm
import collections
from functools import singledispatch
from mainsequence.tdag.config import bcolors
import cloudpickle
from pathlib import Path
from mainsequence.instrumentation import tracer, tracer_instrumentator
from mainsequence.tdag.config import API_TS_PICKLE_PREFIFX
import mainsequence.client as ms_client
from .persist_managers import PersistManager
from mainsequence.tdag.config import (
    ogm
)
from dataclasses import dataclass
from mainsequence.logconf import logger


build_model = lambda model_data: get_model_class(model_data["orm_class"])(**model_data)


# 1. Create a "registry" function using the decorator
@singledispatch
def serialize_argument(value: Any, pickle_ts: bool) -> Any:
    """
    Default implementation for any type not specifically registered.
    It can either return the value as is or raise a TypeError.
    """
    # For types we don't explicitly handle, we can check if they are serializable
    # or just return them. For simplicity, we return as is.
    return value


def _serialize_timeserie(value: "TimeSerie", pickle_ts: bool = False) -> Dict[str, Any]:
    """Serialization logic for TimeSerie objects."""
    print(f"Serializing TimeSerie: {value.local_hash_id}")
    # This logic can be expanded, for example, to handle pickling.
    if pickle_ts:
        # Placeholder for actual pickling logic
        return {"is_time_serie_pickled": True, "local_hash_id": value.local_hash_id, "data": "pickled_data_placeholder"}
    return {"is_time_serie_instance": True, "local_hash_id": value.local_hash_id}

def _serialize_api_timeserie(value, pickle_ts: bool):
    if pickle_ts:
        new_value = {"is_api_time_serie_pickled": True}
        value.persist_to_pickle() # Assumes this method exists
        new_value["local_hash_id"] = value.local_hash_id
        new_value['data_source_id'] = value.data_source_id
        return new_value
    return value

@serialize_argument.register(BaseModel)
def _(value: BaseModel, pickle_ts: bool = False) -> Dict[str, Any]:
    """Serialization logic for any Pydantic BaseModel."""
    import_path = {"module": value.__class__.__module__, "qualname": value.__class__.__qualname__}
    # Recursively call serialize_argument on each value in the model's dictionary.
    serialized_model = {k: serialize_argument(v, pickle_ts) for k, v in value.model_dump().items()}
    return {"pydantic_model_import_path": import_path, "serialized_model": serialized_model}

@serialize_argument.register(BaseObjectOrm)
def _(value, pickle_ts: bool):
    return value.to_serialized_dict()


@serialize_argument.register(list)
def _(value: list, pickle_ts: bool):
    if not value:
        return []

    # 1. DETECT if it's a list of ORM models
    if isinstance(value[0], BaseObjectOrm):
        # 2. SORT the list to ensure a stable hash
        sorted_value = sorted(value, key=lambda x: x.unique_identifier)

        # 3. SERIALIZE each item in the now-sorted list
        serialized_items = [serialize_argument(item, pickle_ts) for item in sorted_value]

        # 4. WRAP the result in an identifiable structure for deserialization
        return {"__type__": "orm_model_list", "items": serialized_items}

    # Fallback for all other list types
    return [serialize_argument(item, pickle_ts) for item in value]

@serialize_argument.register(tuple)
def _(value, pickle_ts: bool):
    items = [serialize_argument(item, pickle_ts) for item in value]
    return {"__type__": "tuple", "items": items}


@serialize_argument.register(dict)
def _(value: dict, pickle_ts: bool):
    # Check for the special marker key.
    if value.get("is_time_series_config") is True:
        # If it's a special config dict, preserve its unique structure.
        # Serialize its contents recursively.
        config_data = {k: serialize_argument(v, pickle_ts) for k, v in value.items()}

        return {"is_time_series_config": True, "config_data": config_data}

    # Otherwise, handle it as a regular dictionary.
    return {k: serialize_argument(v, pickle_ts) for k, v in value.items()}


@serialize_argument.register(SimpleNamespace)
def _(value, pickle_ts: bool):
    return serialize_argument.dispatch(dict)(vars(value), pickle_ts)

@serialize_argument.register(Enum)
def _(value, pickle_ts: bool):
    return value.value


class TimeSerieInitMeta(BaseModel):
    ...

def data_source_dir_path(data_source_id: int) -> str:
    path = ogm.pickle_storage_path
    return f"{path}/{data_source_id}"

def data_source_pickle_path(data_source_id: int) -> str:
    return f"{data_source_dir_path(data_source_id)}/data_source.pickle"


def parse_dictionary_before_hashing(dictionary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parses a dictionary before hashing, handling nested structures and special types.

    Args:
        dictionary: The dictionary to parse.

    Returns:
        A new dictionary ready for hashing.
    """
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


                elif isinstance(value, dict) and value.get("__type__") == "orm_model_list":

                    # The value["items"] are already serialized dicts

                    local_ts_dict_to_hash[key] = [v["unique_identifier"] for v in value["items"]]
                else:
                    # recursively apply hash signature
                    local_ts_dict_to_hash[key] = parse_dictionary_before_hashing(value)


    return local_ts_dict_to_hash

def prepare_config_kwargs(kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Separates all meta-arguments from the core configuration arguments and applies defaults.
    This replaces _separate_meta_kwargs and sanitize_default_build_metadata.

    Returns:
        A tuple of (core_kwargs, meta_kwargs).
    """
    meta_keys = ["init_meta", "build_meta_data", "local_kwargs_to_ignore"]
    meta_kwargs = {}

    for key in meta_keys:
        if key in kwargs:
            # Move the argument from the main dict to the meta dict
            meta_kwargs[key] = kwargs.pop(key)

    # --- Apply Defaults (replaces sanitize_default_build_metadata) ---
    if meta_kwargs.get("init_meta") is None:
        meta_kwargs["init_meta"] = TimeSerieInitMeta()

    if meta_kwargs.get("build_meta_data") is None:
        meta_kwargs["build_meta_data"] = {"initialize_with_default_partitions": True}

    return kwargs, meta_kwargs  # Returns (core_kwargs, meta_kwargs)


def get_time_serie_source_code(TimeSerieClass: "TimeSerie") -> str:
    """
    Gets the source code of a TimeSerie class.

    Args:
        TimeSerieClass: The class to get the source code for.

    Returns:
        The source code as a string.
    """
    global logger
    try:
        # First try the standard approach.
        source = inspect.getsource(TimeSerieClass)
        if source.strip():
            return source
    except Exception:
        logger.warning \
            ("Your TimeSeries is not in a python module this will likely bring exceptions when running in a pipeline")
    from IPython import get_ipython
    # Fallback: Scan IPython's input history.
    ip = get_ipython()  # Get the current IPython instance.
    if ip is not None:
        # Retrieve the full history as a single string.
        history = "\n".join(code for _, _, code in ip.history_manager.get_range())
        marker = f"class {TimeSerieClass.__name__}"
        idx = history.find(marker)
        if idx != -1:
            return history[idx:]
    return "Source code unavailable."

def get_time_serie_source_code_git_hash(TimeSerieClass: "TimeSerie") -> str:
    """
    Hashes the source code of a TimeSerie class using SHA-1 (Git style).

    Args:
        TimeSerieClass: The class to hash.

    Returns:
        The Git-style hash of the source code.
    """
    time_serie_class_source_code = get_time_serie_source_code(TimeSerieClass)
    # Prepare the content for Git-style hashing
    # Git hashing format: "blob <size_of_content>\0<content>"
    content = f"blob {len(time_serie_class_source_code)}\0{time_serie_class_source_code}"
    # Compute the SHA-1 hash (Git hash)
    hash_object = hashlib.sha1(content.encode('utf-8'))
    git_hash = hash_object.hexdigest()
    return git_hash




def hash_signature(dictionary: Dict[str, Any]) -> Tuple[str, str]:
    """
    Computes MD5 hashes for local and remote configurations from a single dictionary.
    """
    dhash_local = hashlib.md5()
    dhash_remote = hashlib.md5()

    # The function expects to receive the full dictionary, including meta-args
    local_ts_dict_to_hash = parse_dictionary_before_hashing(dictionary)
    remote_ts_in_db_hash = copy.deepcopy(local_ts_dict_to_hash)

    # Add project_id for local hash
    local_ts_dict_to_hash["project_id"] = POD_PROJECT.id

    # Handle remote hash filtering internally
    if "local_kwargs_to_ignore" in local_ts_dict_to_hash:
        keys_to_ignore = sorted(local_ts_dict_to_hash['local_kwargs_to_ignore'])
        for k in keys_to_ignore:
            remote_ts_in_db_hash.pop(k, None)
        remote_ts_in_db_hash.pop("local_kwargs_to_ignore", None)

    # Encode and hash both versions
    encoded_local = json.dumps(local_ts_dict_to_hash, sort_keys=True).encode()
    encoded_remote = json.dumps(remote_ts_in_db_hash, sort_keys=True).encode()

    dhash_local.update(encoded_local)
    dhash_remote.update(encoded_remote)

    return dhash_local.hexdigest(), dhash_remote.hexdigest()



def rebuild_with_type(value: Dict[str, Any], rebuild_function: Callable) -> Union[tuple, Any]:
    """
    Rebuilds a tuple from a serialized dictionary representation.

    Args:
        value: A dictionary with a '__type__' key.
        rebuild_function: A function to apply to each item in the tuple.

    Returns:
        A rebuilt tuple.

    Raises:
        NotImplementedError: If the type is not 'tuple'.
    """
    type_marker = value.get("__type__")

    if type_marker == "tuple":
        return tuple([rebuild_function(c) for c in value["items"]])
        # Add this block to handle the ORM model list
    elif type_marker == "orm_model_list":
        return [rebuild_function(c) for c in value["items"]]
    else:
        raise NotImplementedError


class Serializer:
    """Encapsulates the logic for converting a configuration dict into a serializable format."""

    def serialize_init_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serializes __init__ keyword arguments for a TimeSerie.
        This maps to your original `serialize_init_kwargs`.
        """
        return self._serialize_dict(kwargs=kwargs, pickle_ts=False)

    def serialize_for_pickle(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """
        Serializes properties to a pickle-friendly dictionary.
        """
        return self._serialize_dict(kwargs=properties, pickle_ts=True)

    def _serialize_dict(self, kwargs: Dict[str, Any], pickle_ts: bool) -> Dict[str, Any]:
        """
        Internal worker that serializes a dictionary by calling the dispatcher.
        This maps to your original `_serialize_configuration_dict`.
        """
        new_kwargs = {key: serialize_argument(value, pickle_ts) for key, value in kwargs.items()}
        return collections.OrderedDict(sorted(new_kwargs.items()))

from abc import ABC, abstractmethod

class BaseRebuilder(ABC):
    """
    Abstract base class for deserialization specialists.
    Defines a common structure with a registry and a dispatch method.
    """

    @property
    @abstractmethod
    def registry(self) -> Dict[str, callable]:
        """The registry mapping keys to handler methods."""
        pass

    def rebuild(self, value: Any, **kwargs) -> Any:
        """
        Main dispatch method. Recursively rebuilds a value using the registry.
        """
        # Base cases for recursion
        if not isinstance(value, (dict, list, tuple)):
            return value
        if isinstance(value, list):
            return [self.rebuild(item, **kwargs) for item in value]
        if isinstance(value, tuple):
            return tuple(self.rebuild(item, **kwargs) for item in value)

        # For dictionaries, use the specialized registry
        if isinstance(value, dict):
            # Find a handler in the registry and use it
            for key, handler in self.registry.items():
                if key in value:
                    return handler(self, value, **kwargs)

            # If no handler, it's a generic dict; rebuild its contents
            return {k: self.rebuild(v, **kwargs) for k, v in value.items()}

        return value  # Fallback


class PickleRebuilder(BaseRebuilder):
    """Specialist for deserializing objects from a pickled state."""

    @classmethod
    def _rebuild_pickled_timeserie(cls, value: Dict, **state_kwargs) -> "TimeSerie":
        """Handles 'is_time_serie_pickled' markers."""
        import cloudpickle
        # Note: You need to make TimeSerie available here
        full_path = TimeSerie.get_pickle_path(
            local_hash_id=value['local_hash_id'],
            data_source_id=value['data_source_id']
        )
        with open(full_path, 'rb') as handle:
            ts = cloudpickle.load(handle)
        ts.set_data_source_from_pickle_path(full_path)
        if state_kwargs.get('graph_depth', 0) - 1 <= state_kwargs.get('graph_depth_limit', 0):
            ts.set_state_with_sessions(**state_kwargs)
        return ts
    @classmethod
    def _rebuild_api_timeserie(cls, value: Dict, **state_kwargs) -> "APITimeSerie":
        """Handles 'is_api_time_serie_pickled' markers."""
        import cloudpickle
        # Note: You need to make APITimeSerie available here
        full_path = APITimeSerie.get_pickle_path(
            local_hash_id=value['local_hash_id'],
            data_source_id=value['data_source_id']
        )
        with open(full_path, 'rb') as handle:
            ts = cloudpickle.load(handle)
        return ts
    @classmethod
    def _rebuild_timeseries_config(cls, value: Dict, **state_kwargs) -> Dict:
        """Handles 'is_time_series_config' markers."""
        return cls.deserialize_pickle_state(value["config_data"], **state_kwargs)

    @classmethod
    def _rebuild_orm_model(cls, value: Dict, **state_kwargs) -> Any:
        """Handles 'orm_class' markers for single models."""
        return build_model(value)
    @classmethod
    def _rebuild_orm_model_list(cls, value: Dict, **state_kwargs) -> list:
        """Handles '__type__: orm_model_list' markers."""
        # Using build_model directly as items are already serialized model dicts
        return [build_model(item) for item in value["items"]]

    @classmethod
    def _rebuild_complex_type(cls, value: Dict, **state_kwargs) -> Any:
        """Handles generic '__type__' markers (like tuples)."""
        rebuild_function = lambda x: cls.deserialize_value(x, **state_kwargs)
        # Assumes rebuild_with_type handles different __type__ values
        return rebuild_with_type(value, rebuild_function=rebuild_function)
    @property
    def registry(self) -> Dict[str, callable]:
        return {
            "pydantic_model_import_path": self._rebuild_pydantic_model,
            "is_time_serie_pickled": self._rebuild_pickled_timeserie,
            "is_api_time_serie_pickled": self._rebuild_api_timeserie,
            "is_time_series_config": self._rebuild_timeseries_config,
            "orm_class": self._rebuild_orm_model,
            "__type__": self._rebuild_complex_type,
        }

class ConfigRebuilder(BaseRebuilder):

    @property
    def registry(self) -> Dict[str, Callable]:
        return {
            "pydantic_model_import_path": self._handle_pydantic_model,
            "is_time_series_config": self._handle_timeseries_config,
            "orm_class": self._handle_orm_model,
            "__type__": self._handle_complex_type,
        }

    def _handle_pydantic_model(self, value: Dict, **kwargs) -> Any:
        path_info = value["pydantic_model_import_path"]
        module = importlib.import_module(path_info["module"])
        PydanticClass = getattr(module, path_info['qualname'])

        rebuilt_value = self.rebuild(value["serialized_model"], **kwargs)
        return PydanticClass(**rebuilt_value)

    def _handle_timeseries_config(self, value: Dict, **kwargs) -> Dict:
        return self.rebuild(value["config_data"], **kwargs)

    def _handle_orm_model(self, value: Dict, **kwargs) -> Any:
        return build_model(value)

    def _handle_complex_type(self, value: Dict, **kwargs) -> Any:
        # Special case for ORM lists within the generic complex type handler
        if value.get("__type__") == "orm_model_list":
            return [build_model(item) for item in value["items"]]
        # Fallback to the generic rebuild_with_type for other types (like tuples)
        return rebuild_with_type(value, rebuild_function=lambda x: self.rebuild(x, **kwargs))


class DeserializerManager:
    """Handles serialization and deserialization of configurations."""

    def __init__(self):
        self.pickle_rebuilder = PickleRebuilder()
        self.config_rebuilder = ConfigRebuilder()

    def rebuild_config(self, config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Rebuilds an entire configuration dictionary."""
        return self.config_rebuilder.rebuild(config, **kwargs)
    def rebuild_serialized_config(self, config: Dict[str, Any], time_serie_class_name: str) -> Dict[str, Any]:
        """
        Rebuilds a configuration dictionary from a serialized config.

        Args:
            config: The configuration dictionary.
            time_serie_class_name: The name of the TimeSerie class.

        Returns:
            The rebuilt configuration dictionary.
        """
        config = self.rebuild_config(config=config)
        if time_serie_class_name == "WrapperTimeSerie":
            config["time_series_dict"] = self.rebuild_serialized_wrapper_dict(
                time_series_dict_config=config["time_series_dict"],
            )

        return config

    def rebuild_serialized_wrapper_dict(self, time_series_dict_config: dict) -> Dict[str, Any]:
        """
        Rebuilds a dictionary of TimeSerie objects from a serialized wrapper configuration.

        Args:
            time_series_dict_config: The serialized wrapper dictionary.

        Returns:
            A dictionary of TimeSerie objects.
        """
        time_series_dict = {}
        for key, value in time_series_dict_config.items():
            new_ts = self.rebuild_from_configuration(hash_id=value)
            time_series_dict[key] = new_ts

        return time_series_dict


    def deserialize_pickle_state(self, state: Any, **kwargs) -> Any:
        """Deserializes an entire pickled state object."""
        return self.pickle_rebuilder.rebuild(state, **kwargs)




@dataclass
class TimeSerieConfig:
    """A container for all computed configuration attributes."""
    init_meta: Any
    remote_build_metadata: Any
    local_kwargs_to_ignore: Optional[List[str]]
    hashed_name: str
    remote_table_hashed_name: str
    local_initial_configuration: Dict[str, Any]
    remote_initial_configuration: Dict[str, Any]

def create_config(ts_class_name: str,  kwargs: Dict[str, Any]):
    """
    Creates the configuration and hashes using the original hash_signature logic.
    """
    global logger

    # 1. Use the helper to separate meta args from core args.
    core_kwargs, meta_kwargs = prepare_config_kwargs(kwargs)

    # 2. Serialize the core arguments
    serialized_core_kwargs = Serializer().serialize_init_kwargs(core_kwargs)

    # 3. Prepare the dictionary for hashing
    dict_to_hash = copy.deepcopy(serialized_core_kwargs)
    local_kwargs_to_ignore = meta_kwargs.get("local_kwargs_to_ignore")
    if local_kwargs_to_ignore:
        dict_to_hash['local_kwargs_to_ignore'] = local_kwargs_to_ignore

    # 4. Generate the hashes
    local_ts_hash, remote_table_hash = hash_signature(dict_to_hash)

    # 5. Create the remote configuration by removing ignored keys
    remote_config = copy.deepcopy(dict_to_hash)
    if 'local_kwargs_to_ignore' in remote_config:
        for k in remote_config['local_kwargs_to_ignore']:
            remote_config.pop(k, None)
        remote_config.pop('local_kwargs_to_ignore', None)



    # 6. Return all computed values in the structured dataclass
    return TimeSerieConfig(
        init_meta=meta_kwargs["init_meta"],
        remote_build_metadata=meta_kwargs["build_meta_data"],
        local_kwargs_to_ignore=local_kwargs_to_ignore,
        hashed_name=f"{ts_class_name}_{local_ts_hash}".lower(),
        remote_table_hashed_name=f"{ts_class_name}_{remote_table_hash}".lower(),
        local_initial_configuration=dict_to_hash,
        remote_initial_configuration=remote_config,
    )


def flush_pickle(self) -> None:
    """Deletes the pickle file for this time series."""
    if os.path.isfile(self.pickle_path):
        os.remove(self.pickle_path)

# In class BuildManager:


@tracer.start_as_current_span("TS: load_from_pickle")
def load_from_pickle(pickle_path: str) -> "TimeSerie":
    """
    Loads a TimeSerie object from a pickle file, handling both standard and API types.

    Args:
        pickle_path: The path to the pickle file.

    Returns:
        The loaded TimeSerie object.
    """
    import cloudpickle
    from pathlib import Path

    directory = os.path.dirname(pickle_path)
    filename = os.path.basename(pickle_path)
    prefixed_path = os.path.join(directory, f"{API_TS_PICKLE_PREFIFX}{filename}")
    if os.path.isfile(prefixed_path) and os.path.isfile(pickle_path):
        raise FileExistsError("Both default and API timeseries pickle exist - cannot decide which to load")

    if os.path.isfile(prefixed_path):
        pickle_path = prefixed_path

    try:
        with open(pickle_path, 'rb') as handle:
            time_serie = cloudpickle.load(handle)
    except Exception as e:
        raise e

    if time_serie.is_api:
        return time_serie

    data_source = load_data_source_from_pickle(pickle_path=pickle_path)
    time_serie.set_data_source(data_source=data_source)
    # verify pickle
    verify_backend_git_hash_with_pickle(time_serie_instance=time_serie)
    return time_serie


def patch_build_configuration(time_serie_instance :"TimeSerie") -> None:
    """
    Patches the build configuration for the time series and its dependencies.
    """
    patch_build = os.environ.get("PATCH_BUILD_CONFIGURATION", False) in ["true", "True", 1]
    if patch_build == True:
        time_serie_instance.local_persist_manager # ensure lpm exists
        time_serie_instance.verify_and_build_remote_objects()  # just call it before to initilaize dts
        time_serie_instance.logger.warning(f"Patching build configuration for {time_serie_instance.hash_id}")
        flush_pickle()

        time_serie_instance.local_persist_manager.patch_build_configuration \
            (local_configuration=time_serie_instance.local_initial_configuration,
                                                                            remote_configuration=time_serie_instance.remote_initial_configuration,
                                                                            remote_build_metadata=time_serie_instance.remote_build_metadata,
                                                                            )
def verify_backend_git_hash_with_pickle(time_serie_instance:"TimeSerie") -> None:
    """Verifies if the git hash in the backend matches the one from the pickled object."""
    if time_serie_instance.local_persist_manager.metadata is not None:
        load_git_hash =  get_time_serie_source_code_git_hash(time_serie_instance.__class__)

        persisted_pickle_hash = time_serie_instance.local_persist_manager.metadata.time_serie_source_code_git_hash
        if load_git_hash != persisted_pickle_hash:
            time_serie_instance.logger.warning(
                f"{bcolors.WARNING}Source code does not match with pickle rebuilding{bcolors.ENDC}")
            flush_pickle()

            rebuild_time_serie = rebuild_from_configuration(local_hash_id= time_serie_instance.local_hash_id,
                                                                      data_source= time_serie_instance.data_source,
                                                                      )
            rebuild_time_serie.persist_to_pickle()
        else:
            # if no need to rebuild, just sync the metadata
            time_serie_instance.local_persist_manager.synchronize_metadata(local_metadata=None)



def load_data_source_from_pickle( pickle_path: str) -> Any:
    data_path = Path(pickle_path).parent / "data_source.pickle"
    with open(data_path, 'rb') as handle:
        data_source = cloudpickle.load(handle)
    return data_source

def rebuild_and_set_from_local_hash_id( local_hash_id: int, data_source_id: int, set_dependencies_df: bool = False,
                                       graph_depth_limit: int = 1) -> Tuple["TimeSerie", str]:
    """
    Rebuilds a TimeSerie from its local hash ID and pickles it if it doesn't exist.

    Args:
        local_hash_id: The local hash ID of the TimeSerie.
        data_source_id: The data source ID.
        set_dependencies_df: Whether to set the dependencies DataFrame.
        graph_depth_limit: The depth limit for graph traversal.

    Returns:
        A tuple containing the TimeSerie object and the path to its pickle file.
    """
    pickle_path = get_pickle_path(local_hash_id=local_hash_id,
                                      data_source_id=data_source_id,
                                      )
    if os.path.isfile(pickle_path) == False or os.stat(pickle_path).st_size == 0:
        # rebuild time serie and pickle
        ts = rebuild_from_configuration(
            local_hash_id=local_hash_id,
            data_source=data_source_id,
        )
        if set_dependencies_df == True:
            ts.set_relation_tree()

        ts.persist_to_pickle()
        ts.logger.info(f"ts {local_hash_id} pickled ")

    ts = load_and_set_from_pickle(
        pickle_path=pickle_path,
        graph_depth_limit=graph_depth_limit,
    )
    ts.logger.debug(f"ts {local_hash_id} loaded from pickle ")
    return ts, pickle_path

def set_data_source_from_pickle_path(self, pikle_path: str) -> None:
    """
    Sets the data source for the TimeSerie from a pickle path.

    Args:
        pikle_path: The path to the pickle file.
    """
    data_source =  self.owner.load_data_source_from_pickle(pikle_path)
    self.owner.set_data_source(data_source=data_source)

def load_and_set_from_pickle( pickle_path: str, graph_depth_limit: int = 1) -> "TimeSerie":
    """
    Loads a TimeSerie from a pickle file and sets its state.

    Args:
        pickle_path: The path to the pickle file.
        graph_depth_limit: The depth limit for setting the state.

    Returns:
        The loaded and configured TimeSerie object.
    """
    ts = load_from_pickle(pickle_path)
    set_state_with_sessions(
        time_serie_instance=ts,
        graph_depth=0,
        graph_depth_limit=graph_depth_limit,
        include_vam_client_objects=False)
    return ts

@tracer.start_as_current_span("TS: Rebuild From Configuration")
def rebuild_from_configuration( local_hash_id: str,
                               data_source: Union[int, object]) -> "TimeSerie":
    """
    Rebuilds a TimeSerie instance from its configuration.

    Args:
        local_hash_id: The local hash ID of the TimeSerie.
        data_source: The data source ID or object.

    Returns:
        The rebuilt TimeSerie instance.
    """
    import importlib

    tracer_instrumentator.append_attribute_to_current_span("local_hash_id", local_hash_id)

    if isinstance(data_source, int):
        pickle_path = get_pickle_path(data_source_id=data_source,
                                          local_hash_id=local_hash_id)
        if os.path.isfile(pickle_path) == False:
            data_source = ms_client.DynamicTableDataSource.get(pk=data_source)
            data_source.persist_to_pickle(data_source_pickle_path(data_source.id))

        data_source = load_data_source_from_pickle(pickle_path=pickle_path)

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
    time_serie_config = Deserializer().rebuild_serialized_config(time_serie_config,
                                                                   time_serie_class_name=time_serie_class_name)
    time_serie_config["init_meta"] = {}

    re_build_ts = TimeSerieClass(**time_serie_config)

    return re_build_ts

def __setstate__(self, state: Dict[str, Any]) -> None:
    # Restore instance attributes (i.e., filename and lineno).
    self.owner.__dict__.update(state)

def __getstate__(self) -> Dict[str, Any]:
    # Copy the object's state from self.__dict__ which contains
    # all our instance attributes. Always use the dict.copy()
    # method to avoid modifying the original state.
    state = self._prepare_state_for_pickle(state=self.__dict__)

    # Remove the unpicklable entries.
    return state

def get_pickle_path( local_hash_id: str, data_source_id: int) -> str:
    return os.path.join(ogm.pickle_storage_path, str(data_source_id), f"{local_hash_id}.pickle")

def load_and_set_from_hash_id( local_hash_id: int, data_source_id: int) -> "TimeSerie":
    path = get_pickle_path(local_hash_id=local_hash_id ,data_source_id=data_source_id)
    ts = load_and_set_from_pickle(pickle_path=path)
    return ts


def get_pickle_path_from_time_serie(time_serie_instance) -> str:
    pp = data_source_dir_path(time_serie_instance.data_source.id)
    path = f"{pp}/{time_serie_instance.local_hash_id}.pickle"
    return path
def update_git_and_code_in_backend(time_serie_instance :PersistManager) -> None:
    """Updates the source code and git hash information in the backend."""
    time_serie_instance.local_persist_manager.update_source_informmation(
        git_hash_id=get_time_serie_source_code_git_hash(time_serie_instance.__class__),
        source_code=get_time_serie_source_code(time_serie_instance.__class__),
    )


@tracer.start_as_current_span("TS: set_state_with_sessions")
def set_state_with_sessions(time_serie_instance :"TimeSerie", include_vam_client_objects: bool = True,
                            graph_depth_limit: int = 1000,
                            graph_depth: int = 0) -> None:
    """
    Sets the state of the TimeSerie after loading from pickle, including sessions.

    Args:
        include_vam_client_objects: Whether to include VAM client objects.
        graph_depth_limit: The depth limit for graph traversal.
        graph_depth: The current depth in the graph.
    """
    if graph_depth_limit == -1:
        graph_depth_limit = 1e6

    minimum_required_depth_for_update = time_serie_instance.get_minimum_required_depth_for_update()

    state = time_serie_instance.__dict__

    if graph_depth_limit < minimum_required_depth_for_update and graph_depth == 0:
        graph_depth_limit = minimum_required_depth_for_update
        time_serie_instance.logger.warning(f"Graph depht limit overrided to {minimum_required_depth_for_update}")

    # if the data source is not local then the de-serialization needs to happend after setting the local persist manager
    # to guranteed a proper patch in the back-end
    if graph_depth <= graph_depth_limit and time_serie_instance.data_source.related_resource_class_type:
        time_serie_instance._set_local_persist_manager(
            local_hash_id=time_serie_instance.local_hash_id,
            local_metadata=None,
        )

    serializer = Deserializer()
    state = serializer.deserialize_pickle_state(
        state=state,
        data_source_id=time_serie_instance.data_source.id,
        include_vam_client_objects=include_vam_client_objects,
        graph_depth_limit=graph_depth_limit,
        graph_depth=graph_depth + 1
    )

    time_serie_instance.__dict__.update(state)

    time_serie_instance.local_persist_manager.synchronize_metadata(local_metadata=None)


def _prepare_state_for_pickle(logger, state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepares the object's state for pickling by serializing and removing unpicklable entries.

    Args:
        state: The object's __dict__.

    Returns:
        A pickle-safe dictionary representing the object's state.
    """
    properties = state
    serializer = Serializer()
    properties = serializer.serialize_for_pickle(properties)
    names_to_remove = []
    for name, attr in properties.items():
        if name in [
            "local_persist_manager",
            "logger",
            "init_meta",
            "_local_metadata_future",
            "_local_metadata_lock",
            "_local_persist_manager",
            "update_tracker",
        ]:
            names_to_remove.append(name)
            continue

        try:
            cloudpickle.dumps(attr)
        except Exception as e:
            logger.exception(f"Cant Pickle property {name}")
            raise e

    for n in names_to_remove:
        properties.pop(n, None)

    return properties
def run_in_debug_scheduler(time_serie_instance :"TimeSerie", break_after_one_update: bool = True, run_head_in_main_process: bool = True,
                           wait_for_update: bool = True, force_update: bool = True, debug: bool = True,
                           update_tree: bool = True,
                           raise_exception_on_error: bool = True) -> None:
    """
    Runs the TimeSerie update in a debug scheduler.

    Args:
        break_after_one_update: If True, stops after one update cycle.
        run_head_in_main_process: If True, runs the head node in the main process.
        wait_for_update: If True, waits for the update to complete.
        force_update: If True, forces an update.
        debug: If True, runs in debug mode.
        update_tree: If True, updates the entire dependency tree.
        raise_exception_on_error: If True, raises exceptions on errors.
    """
    from .update.scheduler import SchedulerUpdater
    SchedulerUpdater.debug_schedule_ts(
        time_serie_hash_id= time_serie_instance.local_hash_id,
        data_source_id= time_serie_instance.data_source.id,
        break_after_one_update=break_after_one_update,
        run_head_in_main_process=True,
        wait_for_update=False,
        force_update=True,
        debug=True,
        update_tree=True,
        raise_exception_on_error=raise_exception_on_error
    )