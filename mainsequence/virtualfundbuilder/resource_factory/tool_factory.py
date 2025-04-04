import os
from mainsequence.virtualfundbuilder.enums import ResourceType
from mainsequence.virtualfundbuilder.utils import parse_object_signature, object_signature_to_yaml, object_signature_to_yaml, build_markdown


from mainsequence.virtualfundbuilder.resource_factory.base_factory import insert_in_registry, BaseResource


class BaseTool(BaseResource):
    TYPE = ResourceType.TOOL

TOOL_REGISTRY = TOOL_REGISTRY if 'TOOL_REGISTRY' in globals() else {}
def register_tool(name=None, register_in_agent=True):
    """
    Decorator to register a model class in the factory.
    If `name` is not provided, the class's name is used as the key.
    """
    def decorator(cls):
        return insert_in_registry(TOOL_REGISTRY, cls, register_in_agent, name)
    return decorator
