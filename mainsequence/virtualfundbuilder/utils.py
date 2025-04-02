import copy
import inspect
import json
import logging
from typing import Optional, Union, get_origin, get_args

from pydantic.fields import PydanticUndefined, FieldInfo

import pandas as pd
from mainsequence.client import CONSTANTS, Asset
from mainsequence.tdag.time_series import ModelList, TimeSerie
from mainsequence.client import CONSTANTS as TDAG_CONSTANTS
from mainsequence.client.models_tdag import register_strategy
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
import os
from pathlib import Path
import os
import tempfile
from mainsequence.logconf import logger

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

def filter_assets(df: pd.DataFrame, asset_list: ModelList) -> pd.DataFrame:
    """ Filters a DataFrame to include only rows that have asset symbols contained in a given asset list. """
    asset_ids = [a.unique_identifier for a in asset_list]
    return df[df.index.get_level_values("asset_symbol").isin(asset_ids)]


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
        if element_info["default"] is None:
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


def find_ts_recursively(root_ts, ts_names):
    def _find_ts_recursively(parent_ts):
        results = []
        for attr_name, attr_value in parent_ts.__dict__.items():

            # wrapped ts for prices
            if attr_name == "related_time_series" and isinstance(attr_value, dict):
                for value in attr_value.values():
                    if isinstance(value, TimeSerie) and value.hash_id.split("_")[0] in ts_names:
                        results.append(value)

            if not isinstance(attr_value, TimeSerie):
                continue

            if attr_value.hash_id.split("_")[0] in ts_names:
                results.append(attr_value)
                continue

            results += _find_ts_recursively(attr_value)

        return results
    return list(set(_find_ts_recursively(root_ts)))


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
                "tracking_funds_expected_exposure_from_latest_holdings",
                "portfolio_tdag_update_configuration",
                "builds_from_target_positions",
                "is_live"
        ]
    from mainsequence.virtualfundbuilder.models import PortfolioConfiguration
    object_signature = parse_object_signature(
        base_object=PortfolioConfiguration,
        use_examples_for_default=["asset_universe"],
        exclude_attr=exclude_arguments
    )

    default_yaml = object_signature_to_yaml(object_signature)

    markdown_documentation = build_markdown(
        elements_to_exclude=exclude_arguments,
        root_class=PortfolioConfiguration
    )

    return {"default_yaml": default_yaml, "markdown_documentation": markdown_documentation, "object_signature": object_signature}

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