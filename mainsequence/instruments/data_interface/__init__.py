from .data_interface import DateInfo, MockDataInterface, MSInterface
from mainsequence.instruments import settings

def _make_backend():
    if getattr(settings, "data", None) and getattr(settings.data, "backend", "mock") == "mainsequence":
        return MSInterface()
    return MockDataInterface()

# export a single, uniform instance
data_interface = _make_backend()