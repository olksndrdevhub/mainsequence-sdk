from .utils import *



if 'tracer' not in locals():
    from mainsequence.tdag.config import configuration
    instrumentation_config = configuration.configuration["instrumentation_config"]
    tracer_instrumentator = TracerInstrumentator(instrumentation_config["grafana_agent_host"])
    tracer = tracer_instrumentator.build_tracer("tdag", __name__,
                                              export_trace_to_console=instrumentation_config["export_trace_to_console"])
