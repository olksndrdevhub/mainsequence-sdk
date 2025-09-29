from .data_interface import DateInfo, MockDataInterface, MSInterface
import os

def _make_backend():
    backend = os.getenv("MSI_DATA_BACKEND", "mock").lower()
    return MSInterface() if backend == "mainsequence" else MockDataInterface()

# export a single, uniform instance
data_interface = _make_backend()