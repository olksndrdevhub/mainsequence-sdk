import os
from mainsequence.virtualfundbuilder.enums import ResourceType
from mainsequence.virtualfundbuilder.resource_factory.base_factory import insert_in_registry, BaseResource


class BaseApp(BaseResource):
    TYPE = ResourceType.APP

APP_REGISTRY = APP_REGISTRY if 'APP_REGISTRY' in globals() else {}
def register_app(name=None, register_in_agent=True):
    """
    Decorator to register a model class in the factory.
    If `name` is not provided, the class's name is used as the key.
    """
    def decorator(cls):
        return insert_in_registry(APP_REGISTRY, cls, register_in_agent, name)
    return decorator
