import json
import os
import inspect
import importlib.util
from threading import Thread

from mainsequence.client.models_tdag import DynamicResource
from mainsequence.virtualfundbuilder.utils import get_vfb_logger, parse_object_signature, build_markdown, object_signature_to_yaml
from typing import get_type_hints, List, Optional
from pydantic import BaseModel
from enum import Enum
from pathlib import Path
import sys
import ast

logger = get_vfb_logger()
from mainsequence.virtualfundbuilder.enums import ResourceType

class BaseResource():
    @classmethod
    def get_source_notebook(cls):
        """Retrieve the exact source code of the class from notebook cells."""
        from IPython import get_ipython
        ipython_shell = get_ipython()
        history = ipython_shell.history_manager.get_range()

        for _, _, cell_content in history:
            try:
                # Parse the cell content as Python code
                parsed = ast.parse(cell_content)

                # Look for the class definition in the AST (Abstract Syntax Tree)
                for node in ast.walk(parsed):
                    if isinstance(node, ast.ClassDef) and node.name == cls.__name__:
                        # Extract the start and end lines of the class
                        start_line = node.lineno - 1
                        end_line = max(
                            [child.lineno for child in ast.walk(node) if hasattr(child, "lineno")]
                        )
                        lines = cell_content.splitlines()
                        return "\n".join(lines[start_line:end_line])
            except Exception as e:
                print(e)
                continue

        return "Class definition not found in notebook history."

    @classmethod
    def build_and_parse_from_configuration(cls, **kwargs) -> 'WeightsBase':
        # Get the __init__ method's type hints and signature
        type_hints = get_type_hints(cls.__init__)

        # Loop through each argument in kwargs
        for arg, value in kwargs.items():
            # Check if the argument has a type hint in __init__
            if arg in type_hints:
                hint = type_hints[arg]

                # Check if the hint is a single Pydantic model
                if inspect.isclass(hint) and issubclass(hint, BaseModel):
                    # Convert to the Pydantic model if the value is not already an instance
                    if not isinstance(value, hint):
                        kwargs[arg] = hint(**value)

                if inspect.isclass(hint) and issubclass(hint, Enum):
                    # Convert to the Pydantic model if the value is not already an instance
                    if not isinstance(value, hint):
                        kwargs[arg] = hint(value)
                # Check if the hint is a List of Pydantic models
                elif getattr(hint, '__origin__', None) is list:
                    inner_type = hint.__args__[0]
                    if inspect.isclass(inner_type) and issubclass(inner_type, BaseModel):
                        # Convert each item in the list to the Pydantic model if not already an instance
                        if not all(isinstance(item, inner_type) for item in value):
                            kwargs[arg] = [inner_type(**item) if not isinstance(item, inner_type) else item for item in value]

        return cls( **kwargs)


def insert_in_registry(registry, cls, register_in_agent, name=None, attributes: Optional[dict]=None):
    """ helper for strategy decorators """
    key = name or cls.__name__  # Use the given name or the class name as the key

    if key in registry and register_in_agent:
        logger.debug(f"{cls.TYPE} '{key}' is already registered.")
        return cls

    registry[key] = cls
    logger.debug(f"Registered {cls.TYPE} class '{key}': {cls}")

    if register_in_agent:
        # send_resource_to_backend(cls, attributes)
        Thread(
            target=send_resource_to_backend,
            args=(cls, attributes),
        ).start()

    return cls


class BaseFactory:
    @staticmethod
    def import_module(strategy_name):
        VFB_PROJECT_PATH = os.environ.get("VFB_PROJECT_PATH", None)
        assert VFB_PROJECT_PATH, "There is no signals folder variable specified"

        project_path = Path(VFB_PROJECT_PATH)

        strategy_folder_path = project_path / strategy_name
        logger.debug(f"Registering signals from {strategy_folder_path}")
        package_name = f"{project_path.name}.{strategy_name}"

        project_root_path = project_path.parent.parent
        if project_root_path not in sys.path:
            sys.path.insert(0, project_root_path)

        for filename in os.listdir(strategy_folder_path):
            try:
                if filename.endswith(".py"):
                    # Build the full module name
                    module_name = f"{package_name}.{filename[:-3]}"

                    # Dynamically import the module
                    module = importlib.import_module(module_name)
            except Exception as e:
                logger.warning(f"Error reading code in strategy {filename}: {e}")

def send_default_configuration():
    # TODO should be a tool
    from mainsequence.virtualfundbuilder.utils import _convert_unknown_to_string, get_default_documentation
    from mainsequence.client.models_tdag import register_default_configuration

    default_config_dict = get_default_documentation()
    payload = {
        "default_config_dict": default_config_dict,
    }

    logger.debug(f"Send default documentation to Backend")
    payload = json.loads(json.dumps(payload, default=_convert_unknown_to_string))
    headers = {"Content-Type": "application/json"}
    try:
        response = register_default_configuration(json_payload=payload)
        if response.status_code not in [200, 201]:
            print(response.text)
    except Exception as e:
        logger.warning("Could register strategy to TSORM", e)

def send_resource_to_backend(resource_class, attributes: Optional[dict]=None):
    """
    Helper function to send the strategy payload to the registry.
    Parses the arguments of the classes __init__ function and the __init__ functions of the parent classes
    """
    # TODO exclude arguments and example arguments need to come from subclass (or not at all)
    def _get_wrapped_or_init(resource_class):
        """Returns the wrapped __init__ method if it exists, otherwise returns the normal __init__."""
        init_method = resource_class.__init__
        return getattr(init_method, '__wrapped__', init_method)

    init_method = _get_wrapped_or_init(resource_class)

    object_signature = parse_object_signature(init_method, use_examples_for_default=[])
    markdown_documentation = build_markdown(
        children_to_exclude=["front_end_details"],
        root_class=init_method
    )

    # get the init signature form class and parent class, might need to be generalized to all parents
    exclude_args = ["is_live", "build_meta_data", "local_kwargs_to_ignore"]
    for parent_class in resource_class.__mro__[1:]:
        init_method_parent = _get_wrapped_or_init(parent_class)

        parent_object_signature = parse_object_signature(
            init_method_parent,
            use_examples_for_default=["is_live"],
            exclude_attr=exclude_args
        )
        parent_markdown_documentation = build_markdown(
            children_to_exclude=["front_end_details"],
            root_class=init_method_parent,
            elements_to_exclude=exclude_args
        )
        markdown_documentation += parent_markdown_documentation
        object_signature.update(parent_object_signature)

    default_yaml = object_signature_to_yaml(object_signature)

    resource_config = DynamicResource.create(
        name=resource_class.__name__,
        type=resource_class.TYPE.value,
        object_signature=object_signature,
        markdown_documentation=markdown_documentation,
        default_yaml=default_yaml,
        attributes=attributes,
    )

