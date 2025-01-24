import logging
from opentelemetry.trace import (
    INVALID_SPAN,
    INVALID_SPAN_CONTEXT,
    get_current_span,
    get_tracer_provider,
    set_tracer_provider,
    get_tracer,
    Status,
    SpanKind,
    StatusCode,

)
from opentelemetry.sdk.trace import (
    TracerProvider,

)
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
ConsoleSpanExporter
)
from opentelemetry.trace.propagation.tracecontext import \
    TraceContextTextMapPropagator
from typing import Union
from mainsequence.tdag.config import bcolors
from mainsequence.logconf import logger

def is_port_in_use(port: int,agent_host:str) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((agent_host, port)) == 0

class TracerInstrumentator():
    __doc__ = f"""
    Main instrumentator class controlls building and exporting of traces 
    
    
       """
    def __init__(self,agent_host:Union[str,None]):
        
        self.logger = logger
        self.agent_host = agent_host
        if agent_host is not None:
            self.logger.debug(f"{bcolors.WARNING}*****SETTING TRACES, TO REMOVE tracing set grafan agent {agent_host} to None {bcolors.ENDC}")

            self.otlp_endpoint=f"http://{agent_host}:4317"
        

    def build_tracer(self,service_name:str,origin:str,export_trace_to_console=False) -> TraceContextTextMapPropagator:
        """
        buidl_tracer("Time Series",__name__)
        :return:
        """

        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        resource = Resource(attributes={SERVICE_NAME: service_name})
        set_tracer_provider(TracerProvider(resource=resource))
        if self.agent_host is not None:
    
            otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
            if  is_port_in_use(4317,agent_host=self.agent_host)== True:
                get_tracer_provider().add_span_processor(BatchSpanProcessor(otlp_exporter))
            if export_trace_to_console ==True:
                get_tracer_provider().add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
        tracer = get_tracer(__name__)
        return tracer

    def get_current_trace_id(self):
        current_span = get_current_span()
        return format(current_span.context.trace_id, "032x")

    def get_telemetry_carrier(self):
        prop = TraceContextTextMapPropagator()
        telemetry_carrier = {}
        prop.inject(carrier=telemetry_carrier)
        return telemetry_carrier
    def append_attribute_to_current_span(self,attribute_key,attribute_value):
        current_span = get_current_span()
        current_span.set_attribute(attribute_key, attribute_value)

class LoggingInstrumentor():  # pylint: disable=empty-docstring
    __doc__ = f"""Logging Instrumentator to fit Loki detection
    """


    def build_formatter(self, **kwargs):

        provider = kwargs.get("tracer_provider", None) or get_tracer_provider()
        old_factory = logging.getLogRecordFactory()
        LoggingInstrumentor._old_factory = old_factory
        LoggingInstrumentor._log_hook = kwargs.get("log_hook", None)

        service_name = None

        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)

            record.otelSpanID = "0"
            record.otelTraceID = "0"

            nonlocal service_name
            if service_name is None:
                resource = getattr(provider, "resource", None)
                if resource:
                    service_name = (
                        resource.attributes.get("service.name") or ""
                    )
                else:
                    service_name = ""
            record.otelServiceName = service_name
            span = get_current_span()
            if span != INVALID_SPAN:
                ctx = span.get_span_context()
                if ctx != INVALID_SPAN_CONTEXT:
                    record.otelSpanID = format(ctx.span_id, "016x")
                    record.otelTraceID = format(ctx.trace_id, "032x")
                    if callable(LoggingInstrumentor._log_hook):
                        try:
                            LoggingInstrumentor._log_hook(  # pylint: disable=E1102
                                span, record
                            )
                        except Exception:  # pylint: disable=W0703
                            pass

            return record
        logging.setLogRecordFactory(record_factory)
        f="%(asctime)s %(levelname)s [%(name)s] [%(filename)s:%(lineno)d] [ traceID=%(otelTraceID)s SpanID=%(otelSpanID)s service.name=%(otelServiceName)s] - %(message)s"
        f=logging.Formatter(f)
        return f

