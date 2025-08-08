import copy
import inspect
import json
import logging
from typing import Optional, Union, get_origin, get_args

from pydantic.fields import PydanticUndefined, FieldInfo

import pandas as pd
from mainsequence.client import CONSTANTS, Asset
from mainsequence.tdag.data_nodes import  DataNode
import numpy as np
from tqdm import tqdm
from numpy.linalg import LinAlgError
from numpy.typing import NDArray
from joblib import delayed, Parallel
from typing import Dict, Any, List, Tuple, Union, Optional
from datetime import datetime
import docstring_parser
from typing import Any, Dict, List, Union, get_type_hints
from enum import Enum
from pydantic import BaseModel
import yaml
from mainsequence.logconf import logger
import inspect
from typing import Any, Dict, List, Optional, Type, Union
from pydantic import BaseModel, Field, create_model

def get_vfb_logger():
    global logger

    # If the logger doesn't have any handlers, create it using the custom function
    logger.bind(sub_application="virtualfundbuilder")
    return logger

logger = get_vfb_logger()

# Symbol mapping for CoinGecko API
GECKO_SYMBOL_MAPPING = {
    "ADA": "cardano",
    "AVAX": "avalanche-2",
    "BCH": "bitcoin-cash",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "MATIC": "matic-network",
    "SOL": "solana",
    "ATOM": "cosmos",
    "BTC": "bitcoin",
    "ETH": "ethereum"
}

# Small time delta for precision operations
TIMEDELTA = pd.Timedelta("5ms")

def runs_in_main_process() -> bool:
    import multiprocessing
    return multiprocessing.current_process().name == "MainProcess"

def reindex_df(df: pd.DataFrame, start_time: datetime, end_time: datetime, freq: str) -> pd.DataFrame:
    """
    Aligns two DataFrames on a new index based on a specified frequency, filling missing entries with the last known values.

    Args:
        df (pd.DataFrame): Reference DataFrame used to determine the new index range.
        start_time (datetime): start of index
        end_time (datetime): end of index
        freq (str): Frequency string (e.g., '1T' for one minute) to define the interval of the new index.

    Returns:
        pd.DataFrame: The df_to_align DataFrame reindexed to match the new timeline and filled with forward filled values.
    """
    new_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    return df.reindex(new_index).ffill()

def convert_to_binance_frequency(freq: str) -> str:
    """
    Converts a generic frequency format to a format compatible with Binance API requirements.

    Args:
        freq (str): The generic frequency format (e.g., '1m', '1h').

    Returns:
        str: A frequency string adapted for Binance API (e.g., '1m', '1h').
    """
    frequency_mappings = {'min': 'm', 'h': 'h', 'd': 'd', 'w': 'w'}  # TODO extend
    for unit, binance_unit in frequency_mappings.items():
        if freq.endswith(unit):
            return freq[:-len(unit)] + binance_unit
    raise NotImplementedError(f"Frequency of {freq} not supported")



def get_last_query_times_per_asset(
        latest_value: datetime,
        metadata: dict,
        asset_list: List[Asset],
        max_lookback_time: datetime,
        current_time: datetime,
        query_frequency: str
) -> Dict[str, Optional[float]]:
    """
    Determines the last query times for each asset based on metadata, a specified lookback limit, and a query frequency.

    Args:
        latest_value (datetime|None): Timestamp of the last value in the database for each asset.
        metadata (dict): Metadata containing previous query information for each coin.
        asset_list (List[Asset]): List of asset objects to process.
        max_lookback_time (datetime): Maximum historical lookback time allowed for the node.
        current_time (datetime): Current time to consider for the calculations.
        query_frequency (str): Query frequency as a pandas-parseable string to determine if new data needs fetching.

    Returns:
        Dict[str, Optional[float]]: A dictionary mapping asset IDs to their respective last query times expressed in UNIX timestamp.
    """
    if latest_value:
        last_query_times_per_asset = metadata["sourcetableconfiguration"]["multi_index_stats"]["max_per_asset_symbol"]
    else:
        last_query_times_per_asset = {}

    for asset in asset_list:
        asset_id = asset.unique_identifier
        if asset_id in last_query_times_per_asset:
            asset_start_time = pd.to_datetime(last_query_times_per_asset[asset_id])
        else:
            asset_start_time = max_lookback_time

        if asset_start_time >= (current_time - pd.Timedelta(query_frequency)):
            logger.info(f"no new data for asset {asset.name} from {asset_start_time} to {current_time}")
            last_query_times_per_asset[asset_id] = None
        else:
            last_query_times_per_asset[asset_id] = (asset_start_time + TIMEDELTA).timestamp()

    return last_query_times_per_asset

def do_single_regression(xx: NDArray, XTX_inv_list: list, rolling_window: int, col_name: str, tmp_y: NDArray, XTX_inv_diag: list) -> pd.DataFrame:
    """
    Performs a single regression analysis on a sliding window of data points for a specific column.

    Args:
        xx (NDArray): An array of independent variable data with a sliding window applied.
        XTX_inv_list (list): A list of precomputed inverse matrices of X.T @ X for each window.
        rolling_window (int): The number of observations per window.
        col_name (str): The name of the column being analyzed, used for labeling the output.
        tmp_y (NDArray): The dependent variable data.
        XTX_inv_diag (list): Diagonals of the precomputed inverse matrices, used for standard error calculation.

    Returns:
        pd.DataFrame: A DataFrame containing the regression results with coefficients, R-squared, and t-statistics.
    """
    mean_y = tmp_y.mean(axis=1).reshape(-1, 1)
    SST = np.sum((tmp_y - mean_y) ** 2, axis=1)
    precompute_y_ = {"SST": SST, "mean_y": mean_y}

    results = []
    for i in tqdm(range(xx.shape[0]), desc=f"building regression {col_name}"):
        xxx = xx[i].reshape(rolling_window, xx[i].shape[-1])
        tmpy_ = tmp_y[i]
        x_mult = XTX_inv_list[i] @ (xxx.T)
        coefs = (x_mult @ tmpy_.T)
        y_estimates = (xxx @ coefs.reshape(-1, 1)).ravel()
        residuals = tmpy_ - y_estimates
        SSR = np.sum((y_estimates - precompute_y_["mean_y"][i]) ** 2)
        rsquared = SSR / precompute_y_["SST"][i]
        residuals_var = np.sum(residuals ** 2) / (rolling_window - coefs.shape[0] + 1)
        standard_errors = np.sqrt(XTX_inv_diag[i] * residuals_var)
        ts = coefs / standard_errors
        results.append(dict(beta=coefs[0], intercept=coefs[1],
                            rsquared=rsquared, t_intercept=ts[1], t_beta=ts[0]
                            ))
    results = pd.concat([pd.DataFrame(results)], keys=[col_name], axis=1)

    return results

def build_rolling_regression_from_df(x: NDArray, y: NDArray, rolling_window: int, column_names: list, threads: int=5) -> pd.DataFrame:
    """
    Builds rolling regressions for multiple variables in parallel using a specified rolling window.

    Args:
        x (NDArray): An array of independent variables.
        y (NDArray): An array of dependent variables.
        rolling_window (int): The size of the rolling window for each regression.
        column_names (list): Names of the dependent variables, used for labeling the output.
        threads (int): Number of threads to use for parallel processing.

    Returns:
        pd.DataFrame: A DataFrame containing the regression results for all variables.
    """
    XX = np.concatenate([x.reshape(-1, 1), np.ones((x.shape[0], 1))], axis=1)
    xx = np.lib.stride_tricks.sliding_window_view(XX, (rolling_window, XX.shape[1]))

    XTX_inv_list, XTX_inv_diag = [], []  # pre multiplication of x before y and diagonal for standard errros

    # precompute for x
    for i in tqdm(range(xx.shape[0]), desc="building x precomputes"):
        xxx = xx[i].reshape(rolling_window, xx[i].shape[-1])
        try:
            XTX_inv = np.linalg.inv(xxx.T @ xxx)
            XTX_inv_list.append(XTX_inv)
            XTX_inv_diag.append(np.diag(XTX_inv))
        except LinAlgError as le:
            XTX_inv_list.append(XTX_inv_list[-1] * np.nan)
            XTX_inv_diag.append(XTX_inv_diag[-1] * np.nan)

    y_views = {i: np.lib.stride_tricks.sliding_window_view(y[:, i], (rolling_window,)) for i in range(y.shape[1])}

    work_details = dict(n_jobs=threads, prefer="threads")
    reg_results = Parallel(**work_details)(
        delayed(do_single_regression)(xx=xx, tmp_y=tmp_y, XTX_inv_list=XTX_inv_list,
                                      rolling_window=rolling_window,
                                      XTX_inv_diag=XTX_inv_diag, col_name=column_names[y_col]
                                      ) for y_col, tmp_y in y_views.items())

    reg_results = pd.concat(reg_results, axis=1)
    reg_results.columns = reg_results.columns.swaplevel()
    return reg_results


def parse_google_docstring(docstring):
    parsed = docstring_parser.parse(docstring)
    return {
        "description": parsed.description,
        "args_descriptions": {param.arg_name: param.description for param in parsed.params},
        "returns": parsed.returns.description if parsed.returns else None,
        "example": "\n".join([param.description for param in parsed.examples]),
        "raises": {exc.type_name: exc.description for exc in parsed.raises}
    }


def get_basic_default_value(elem):
    if elem == str:
        return ""
    elif elem == int:
        return 0
    elif elem == float:
        return 0.0
    elif elem == bool:
        return False
    elif elem == Any:
        return None
    else:
        raise ValueError(f"Unsupported type: {elem}")

def parse_object_signature_raw(elem, parent_description=None, attr_name=None, attr_default=None):
    if isinstance(attr_default, Enum):
        attr_default = attr_default.value

    potential_field_info = attr_default
    if isinstance(potential_field_info, tuple) and len(potential_field_info) > 0 and isinstance(potential_field_info[0], FieldInfo):
        potential_field_info = potential_field_info[0]

    if isinstance(potential_field_info, FieldInfo):
        # Handle both Pydantic v1 ('extra') and v2 ('json_schema_extra') to be robust
        extra_data = getattr(potential_field_info, 'json_schema_extra', None) or getattr(potential_field_info, 'extra', {})
        example = extra_data.get('example')

        if example is not None:
            attr_default = example
        elif potential_field_info.default is not PydanticUndefined:
            attr_default = potential_field_info.default
        else:
            attr_default = None

    element_info = {
        "name": attr_name,
        "type": elem,
        "parent_description": parent_description,
        "description": None,
        "default": attr_default,
        "allowed_values": None,
        "elements": [],
        "example": None
    }

    if get_origin(elem) == Union:
        for arg in get_args(elem):
            sub_elem = parse_object_signature_raw(arg, parent_description, attr_name, attr_default)
            if sub_elem["allowed_values"]:
                return sub_elem

    elif isinstance(elem, type) and issubclass(elem, Enum):
        element_info["allowed_values"] = [e.value for e in elem]
        if element_info["default"] is None and element_info["allowed_values"]:
            element_info["default"] = element_info["allowed_values"][0]

    elif hasattr(elem, '__origin__') and elem.__origin__ is list:
        element_info["name"] = None
        if hasattr(elem, '__args__'):
            element_info["elements"] = [
                parse_object_signature_raw(elem.__args__[0], parent_description, attr_name, attr_default)
            ]
            for e in element_info["elements"]:
                e["type"] = element_info["type"]

    elif hasattr(elem, '__origin__') and elem.__origin__ is dict:
        if hasattr(elem, '__args__'):
            element_info["elements"] = [
                parse_object_signature_raw(arg, parent_description, attr_name, attr_default)
                for arg in elem.__args__
            ]

    elif elem in (str, int, float, bool, Any):
        if element_info["default"] is None:
            element_info["default"] = get_basic_default_value(elem)

    elif isinstance(elem, type) and issubclass(elem, BaseModel):
        doc_parsed = parse_google_docstring(elem.__doc__)
        element_info.update({
            "example": doc_parsed["example"],
            "description": doc_parsed["description"]
        })

        for field_name, field_info in elem.__fields__.items():
            element_info["elements"].append(
                parse_object_signature_raw(
                    field_info.annotation,
                    doc_parsed["args_descriptions"].get(field_name),
                    field_name,
                    field_info.default if field_info.default is not PydanticUndefined else None
                )
            )

    elif isinstance(elem, type):
        doc_parsed = parse_google_docstring(elem.__doc__)
        element_info.update({
            "example": doc_parsed["example"],
            "description": doc_parsed["description"]
        })

        for field_name, field_type in get_type_hints(elem).items():
            element_info["elements"].append(
                parse_object_signature_raw(
                    field_type,
                    doc_parsed["args_descriptions"].get(field_name),
                    field_name,
                    getattr(elem, field_name, None)
                )
            )

    elif callable(elem):
        doc_parsed = parse_google_docstring(elem.__doc__)
        element_info.update({
            "example": doc_parsed["example"],
            "description": doc_parsed["description"]
        })

        default_values = {k: v.default for k, v in inspect.signature(elem).parameters.items() if v.default is not inspect.Parameter.empty}
        for field_name, field_type in get_type_hints(elem).items():
            element_info["elements"].append(
                parse_object_signature_raw(
                    field_type,
                    doc_parsed["args_descriptions"].get(field_name),
                    field_name,
                    default_values.get(field_name)
                )
            )

    elif isinstance(elem, FieldInfo):
        configuration_elem = elem.json_schema_extra.get("portfolio_configuration_overwrite")
        element_info["elements"] = parse_object_signature_raw(configuration_elem)["elements"]
        element_info["type"] = elem.default

    return element_info

def parse_object_signature(base_object: Any, use_examples_for_default: Optional[list]=None, exclude_attr: Optional[list]=None ) -> dict:

    signature_raw = parse_object_signature_raw(base_object)
    documentation_dict = parse_raw_object_signature(
        object_signature=signature_raw,
        use_examples_for_default=use_examples_for_default,
        exclude_attr=exclude_attr
    )
    return documentation_dict


def object_signature_to_markdown(root_dict, level=1, elements_to_exclude=None, children_to_exclude=None):
    """
    Convert a nested dictionary structure into a markdown formatted string.

    Args:
    - root_dict (dict): The nested dictionary to convert.
    - level (int): The current markdown header level.

    Returns:
    - str: The markdown formatted string.
    """
    elements_to_exclude = elements_to_exclude or []
    children_to_exclude = children_to_exclude or []

    def nested_dict_to_markdown(nested_dict, level):
        if nested_dict['name'] in elements_to_exclude:
            return ""

        md_str = ""
        indent = "#" * level + " "
        name = nested_dict.get('name')

        if name:
            md_str += f"{indent}{name}\n\n"
            description = nested_dict.get('description') or nested_dict.get('parent_description')
            if description:
                md_str += f"- **Description:** {description}\n"
            if nested_dict.get('type'):
                md_str += f"- **Type:** `{nested_dict['type']}`\n"
            if nested_dict.get('example'):
                md_str += f"- **Example:** \n```yaml\n{nested_dict['example']}\n```\n"
            if nested_dict.get('default') is not None:
                md_str += f"- **Default:** {nested_dict['default']}\n"
            if nested_dict.get('allowed_values'):
                md_str += f"- **Allowed Values:** {', '.join(map(str, nested_dict['allowed_values']))}\n"
            md_str += "\n"

        if 'elements' in nested_dict and nested_dict['elements'] and name not in children_to_exclude:
            for element in nested_dict['elements']:
                md_str += nested_dict_to_markdown(element, level + 1)

        return md_str

    return nested_dict_to_markdown(root_dict, level=level)

def parse_raw_object_signature(object_signature, use_examples_for_default=None, exclude_attr=None):
    use_examples_for_default = use_examples_for_default or []
    exclude_attr = exclude_attr or []

    def parse_elements(elements):
        if not elements:
            return None

        # filter out elements without a name (Dict or List objects)
        filtered_elements = []
        for e in elements:
            if e["name"] is not None:
                filtered_elements.append(e)
            else:
                filtered_elements += e["elements"]

        parsed = {}
        for element in filtered_elements:
            # some attributes are not supposed to be in the configuration
            name = element['name']
            if element['name'] in exclude_attr:
                continue

            info_dict = {
                "info": {
                    "description": element['description'] or element.get('parent_description'),
                    "allowed_values": element['allowed_values'],
                    "default": element['default'],
                }
            }

            # special cases where we need to use examples (like asset_list)
            if name in use_examples_for_default:
                default = yaml.safe_load(element['example'])
                if not default:
                    info_dict["info"]["default"] = ""
                else:
                    info_dict.update(default)
                parsed[name] = info_dict
                continue
            elif element['elements']:
                elements = parse_elements(element['elements'])

                if str(element["type"]).startswith("typing.List"):
                    info_dict["typing.List"] = {}
                    for key, item in elements.items():
                        info_dict["typing.List"][key] = item
                else:
                    for key, item in elements.items():
                        info_dict[key] = item

            parsed[name] = info_dict
        return parsed

    parsed = parse_elements(object_signature['elements'])
    if parsed is None:
        parsed = {}
    return parsed


def default_config_to_dict(default_config):
    """Convert the default configuration into a Python dictionary.

    Args:
        default_config (dict): Default configuration from the VFB tool.

    Returns:
        dict: Processed configuration dictionary.
    """
    def process_dict(d):
        if not isinstance(d, dict):
            return d

        new_dict = {}
        for key, value in d.items():
            if key == "info" and isinstance(value, dict) and "default" in value:
                if value["default"] is not None:
                    return value["default"]
            elif key == "typing.List":
                return [process_dict(value)]
            elif isinstance(value, dict):
                new_dict[key] = process_dict(value)
            else:
                new_dict[key] = value

        return new_dict

    default_config = process_dict(default_config)
    if not default_config:
        return {}

    return default_config

def object_signature_to_yaml(default_config):
    """Convert the default configuration dictionary to a YAML string.

    Args:
        default_config (dict): Default configuration from the VFB tool.

    Returns:
        str: YAML formatted string of the configuration.
    """
    yaml_string = yaml.dump(default_config_to_dict(default_config), default_flow_style=False)
    return f"```yaml\n{yaml_string}\n```"


def build_markdown(root_class, persist: bool = True, elements_to_exclude=None, children_to_exclude=None):
    """
    Builds standards portfolio configuration documentation
    Returns:
    """
    parsed_documentation = parse_object_signature_raw(root_class)

    # Iteratively go through the structure to set the signal weights names
    for element in parsed_documentation.get('elements', []):
        if element['name'] == 'portfolio_build_configuration':
            for sub_element in element.get('elements', []):
                if sub_element['name'] == 'backtesting_weights_configuration':
                    for weight_config in sub_element.get('elements', []):
                        if weight_config['name'] == "signal_weights_name":
                            weight_config['default'] = "MarketCap"
                            break

    portfolio_config = object_signature_to_markdown(
        parsed_documentation,
        children_to_exclude=children_to_exclude,
        elements_to_exclude=elements_to_exclude,
    )

    return portfolio_config


def get_default_documentation(exclude_arguments=None):
    if exclude_arguments is None:
        exclude_arguments = [
                "is_live"
        ]
    from mainsequence.virtualfundbuilder.models import PortfolioConfiguration
    object_signature = parse_object_signature(
        base_object=PortfolioConfiguration,
        use_examples_for_default=[],
        exclude_attr=exclude_arguments
    )

    default_yaml = object_signature_to_yaml(object_signature)

    markdown_documentation = build_markdown(
        elements_to_exclude=exclude_arguments,
        root_class=PortfolioConfiguration
    )

    return {"default_config": default_yaml, "markdown": markdown_documentation, "documentation_dict": object_signature}

def extract_code(output_string):
    import re
    # Use regex to find content between triple backticks
    match = re.search(r'```[^\n]*\n(.*?)```', output_string, re.DOTALL)
    if match:
        code = match.group(1)
        return code
    else:
        return ''


def _convert_unknown_to_string(obj):
    """Converts unsupported/unknown types to strings."""
    try:
        return str(obj)
    except Exception:
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def is_jupyter_environment():
    try:
        from IPython import get_ipython
        return "ipykernel" in str(get_ipython())
    except ImportError:
        return False




def type_to_json_schema(py_type: Type, definitions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively converts a Python type annotation to a JSON schema dictionary.
    Handles Pydantic models, Enums, Lists, Unions, and basic types.

    Args:
        py_type: The Python type to convert.
        definitions: A dict to store schemas of nested models, used for $defs.

    Returns:
        A dictionary representing the JSON schema for the given type.
    """
    origin = get_origin(py_type)
    args = get_args(py_type)

    # Handle Optional[T] by making the inner type nullable
    if origin is Union and len(args) == 2 and type(None) in args:
        non_none_type = args[0] if args[1] is type(None) else args[1]
        schema = type_to_json_schema(non_none_type, definitions)
        # Add null type to anyOf or create a new anyOf
        if "anyOf" in schema:
            if not any(sub.get("type") == "null" for sub in schema["anyOf"]):
                schema["anyOf"].append({"type": "null"})
        else:
            schema = {"anyOf": [schema, {"type": "null"}]}
        return schema

    if origin is Union:
        return {"anyOf": [type_to_json_schema(arg, definitions) for arg in args]}
    if origin in (list, List):
        item_schema = type_to_json_schema(args[0], definitions) if args else {}
        return {"type": "array", "items": item_schema}
    if origin in (dict, Dict):
        value_schema = type_to_json_schema(args[1], definitions) if len(args) > 1 else {}
        return {"type": "object", "additionalProperties": value_schema}

    # Handle Pydantic Models by creating a reference
    if inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        model_name = py_type.__name__
        if model_name not in definitions:
            definitions[model_name] = {}  # Placeholder to break recursion
            model_schema = py_type.model_json_schema(ref_template="#/$defs/{model}")
            if "$defs" in model_schema:
                for def_name, def_schema in model_schema.pop("$defs").items():
                    if def_name not in definitions:
                        definitions[def_name] = def_schema
            definitions[model_name] = model_schema
        return {"$ref": f"#/$defs/{model_name}"}

    # Handle Enums
    if inspect.isclass(py_type) and issubclass(py_type, Enum):
        return {"type": "string", "enum": [e.value for e in py_type]}

    # Handle basic types
    type_map = {str: "string", int: "integer", float: "number", bool: "boolean"}
    if py_type in type_map:
        return {"type": type_map[py_type]}
    if py_type is Any:
        return {}  # Any type, no constraint

    # Fallback for unknown types
    return {"type": "string", "description": f"Unrecognized type: {getattr(py_type, '__name__', str(py_type))}"}


def create_schema_from_signature(func: callable) -> Dict[str, Any]:
    """
    Parses a function's signature (like __init__) and creates a JSON schema.

    Args:
        func: The function or method to parse.

    Returns:
        A dictionary representing the JSON schema of the function's signature.
    """
    try:
        signature = inspect.signature(func)
        type_hints = get_type_hints(func)
    except (TypeError, NameError): # Handles cases where hints can't be resolved
        return {}

    parsed_doc = docstring_parser.parse(func.__doc__ or "")
    arg_descriptions = {p.arg_name: p.description for p in parsed_doc.params}

    properties = {}
    required = []
    definitions = {}  # For nested models

    for name, param in signature.parameters.items():
        if name in ('self', 'cls', 'args', 'kwargs'):
            continue

        param_type = type_hints.get(name, Any)
        prop_schema = type_to_json_schema(param_type, definitions)
        prop_schema['title'] = name.replace('_', ' ').title()

        if name in arg_descriptions:
            prop_schema['description'] = arg_descriptions[name]

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            default_value = param.default
            try:
                # Ensure default is JSON serializable
                json.dumps(default_value)
                prop_schema['default'] = default_value
            except TypeError:
                 if isinstance(default_value, Enum):
                     prop_schema['default'] = default_value.value
                 else:
                     # Fallback for non-serializable defaults
                     prop_schema['default'] = str(default_value)

        properties[name] = prop_schema

    schema = {
        "title": getattr(func, '__name__', 'Schema'),
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    if definitions:
        schema["$defs"] = definitions

    return schema

# Maps JSON schema types to Python types
JSON_TYPE_TO_PYTHON_TYPE: Dict[str, Type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "object": dict,
    "array": list,
    "null": type(None),
}

# Maps JSON schema constraints to Pydantic Field arguments
CONSTRAINT_MAPPING: Dict[str, str] = {
    "minLength": "min_length",
    "maxLength": "max_length",
    "pattern": "pattern",
    "minimum": "ge",
    "maximum": "le",
    "exclusiveMinimum": "gt",
    "exclusiveMaximum": "lt",
    "multipleOf": "multiple_of",
    "minItems": "min_length",
    "maxItems": "max_length",
}


def create_model_from_schema(
        schema: Dict[str, Any], model_name: Optional[str] = None
) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model from a JSON schema dictionary.
    This version handles dependencies in any order within the '$defs' section.
    """
    created_models: Dict[str, Type[BaseModel]] = {}

    def _get_field_info(field_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts Pydantic Field parameters from a JSON schema property."""
        field_params = {}
        if "default" in field_schema:
            field_params["default"] = field_schema["default"]
        if "description" in field_schema:
            field_params["description"] = field_schema["description"]
        for json_key, pydantic_key in CONSTRAINT_MAPPING.items():
            if json_key in field_schema:
                field_params[pydantic_key] = field_schema[json_key]
        return field_params

    def _resolve_type_from_schema(field_schema: Dict[str, Any]) -> Any:
        """Recursively resolves the Python type for a given schema definition."""
        if "$ref" in field_schema:
            model_ref_name = field_schema["$ref"].split('/')[-1]
            # This will raise a KeyError if the dependency is not yet created,
            # which is caught by the multi-pass logic below.
            return created_models[model_ref_name]

        if "anyOf" in field_schema:
            types = [_resolve_type_from_schema(s) for s in field_schema["anyOf"]]
            non_none_types = [t for t in types if t is not type(None)]
            if not non_none_types:
                return type(None)
            final_type = Union[tuple(non_none_types)] if len(non_none_types) > 1 else non_none_types[0]
            return Optional[final_type] if type(None) in types else final_type

        json_type = field_schema.get("type")
        if isinstance(json_type, list):
            return _resolve_type_from_schema({"anyOf": [{"type": t} for t in json_type]})

        if json_type == "object":
            properties = field_schema.get("properties")
            if properties:
                nested_model_name = field_schema.get("title", f"NestedModel{len(created_models)}")
                # Recursively call the main creation logic for inline nested objects
                return _create_model_from_schema(field_schema, nested_model_name)
            add_props = field_schema.get("additionalProperties")
            return Dict[str, _resolve_type_from_schema(add_props)] if isinstance(add_props, dict) else Dict[str, Any]

        if json_type == "array":
            items = field_schema.get("items", {})
            return List[_resolve_type_from_schema(items)] if items else List[Any]

        return JSON_TYPE_TO_PYTHON_TYPE.get(json_type, Any)

    def _create_model_from_schema(sub_schema: Dict[str, Any], name: str) -> Type[BaseModel]:
        """The core recursive model creation function for a single model."""
        if name in created_models:
            return created_models[name]

        fields = {}
        required_fields = set(sub_schema.get("required", []))
        for field_name, field_schema in sub_schema.get("properties", {}).items():
            field_type = _resolve_type_from_schema(field_schema)
            field_params = _get_field_info(field_schema)

            field_definition: Any
            if "default" in field_params:
                default_value = field_params.pop("default")
                field_definition = Field(default=default_value, **field_params)
            elif field_name in required_fields:
                field_definition = Field(**field_params)
            else:
                field_definition = Field(default=None, **field_params)
            fields[field_name] = (field_type, field_definition)

        model = create_model(name, **fields, __base__=BaseModel, __doc__=sub_schema.get("description"))
        created_models[name] = model
        return model

    definitions = schema.get("$defs", {})
    processing_queue = list(definitions.keys())
    max_passes = len(processing_queue) + 1  # Failsafe for circular dependencies
    passes = 0

    while processing_queue and passes < max_passes:
        passes += 1
        deferred = []
        for def_name in processing_queue:
            try:
                _create_model_from_schema(definitions[def_name], def_name)
            except KeyError:
                # This model depends on another one not yet created, defer it.
                deferred.append(def_name)

        if len(deferred) == len(processing_queue):
            raise RuntimeError(f"Could not resolve model dependencies in $defs. Unresolved: {deferred}")

        processing_queue = deferred

    # After the loop, all definitions are created. Now create the main model.
    final_model_name = model_name or schema.get("title", "DynamicModel")
    return _create_model_from_schema(schema, final_model_name)