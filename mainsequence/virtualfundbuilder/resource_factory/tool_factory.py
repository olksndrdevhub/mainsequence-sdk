import os
from mainsequence.virtualfundbuilder.enums import ResourceType
from mainsequence.virtualfundbuilder.utils import parse_object_signature, object_signature_to_yaml, object_signature_to_yaml, build_markdown


from mainsequence.virtualfundbuilder.resource_factory.base_factory import insert_in_registry, BaseResource, \
    ResourceConfig


class BaseTool(BaseResource):
    TYPE = ResourceType.TOOL

def send_tool_to_registry(strategy_type, strategy_class, is_jupyter=False, is_production=False):
    """Helper function to send the tool payload to the registry."""
    assert os.environ.get("TDAG_ENDPOINT", None), "TDAG_ENDPOINT is not set"

    def _get_wrapped_or_init(strategy_class):
        """Returns the wrapped __init__ method if it exists, otherwise returns the normal __init__."""
        init_method = strategy_class.__init__
        return getattr(init_method, '__wrapped__', init_method)


    init_method = _get_wrapped_or_init(strategy_class)

    object_signature = parse_object_signature(init_method, use_examples_for_default=["asset_universe"])
    markdown_documentation = build_markdown(
        children_to_exclude=["front_end_details"],
        root_class=init_method
    )
    default_yaml = object_signature_to_yaml(object_signature)


    ResourceConfig(
        name=strategy_class.__name__,
        type=strategy_type,
        object_signature=report_signature,
        markdown_documentation=markdown_documentation,
        default_yaml=example_yaml,
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
