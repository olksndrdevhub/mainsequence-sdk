import os
from typing import Optional

from pydantic import BaseModel

from mainsequence.virtualfundbuilder.utils import parse_object_signature, object_signature_to_yaml, object_signature_to_yaml, build_markdown

from mainsequence.virtualfundbuilder.enums import ResourceType

from mainsequence.virtualfundbuilder.resource_factory.base_factory import insert_in_registry, BaseResource


class ToolConfig(BaseModel):
    configuration : BaseModel
    markdown : Optional[str]=None

class BaseTool(BaseResource):
    TYPE = ResourceType.TOOL

def send_tool_to_registry(strategy_type, strategy_class, is_jupyter=False, is_production=False):
    """Helper function to send the tool payload to the registry."""
    assert os.environ.get("TDAG_ENDPOINT", None), "TDAG_ENDPOINT is not set"

    config_class = strategy_class.configuration
    report_signature = parse_object_signature(config_class)
    documentation_dict = object_signature_to_yaml(report_signature)
    example_yaml = object_signature_to_yaml(documentation_dict)
    markdown = build_markdown(
        root_class=config_class
    )
    # where to register?

    tool_config = ToolConfig(
        tool_configuration=config_class,
        markdown=markdown,
    )
    tool_config = {"default_config": report_signature, "markdown": markdown, "documentation_dict": documentation_dict}
    print(f"Register with tool_config {tool_config}")

TOOL_REGISTRY = TOOL_REGISTRY if 'TOOL_REGISTRY' in globals() else {}
def register_tool(name=None, register_in_agent=True):
    """
    Decorator to register a model class in the factory.
    If `name` is not provided, the class's name is used as the key.
    """
    def decorator(cls):
        return insert_in_registry(TOOL_REGISTRY, cls, register_in_agent, name, custom_registry_function=send_tool_to_registry)
    return decorator
