import logging
import logging.config
import os
from pathlib import Path
import structlog
from typing import Union
from structlog.dev import ConsoleRenderer

def console_logger(logger_name,application_name,**metadata):
    """
    Mockup logger
    """
    return create_logger_in_path(logger_name,application_name=application_name,logger_file=None,**metadata)
    return logger

def extract_from_record(_, __, event_dict):
    """
    Extract thread and process names and add them to the event dict.
    """
    record = event_dict.get("_record")
    if record is not None:
        event_dict["thread_name"] = record.threadName
        event_dict["process_name"] = record.processName
    return event_dict

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    Path(directory).mkdir(parents=True, exist_ok=True)

def add_structlog_event_to_record(logger, method_name, event_dict):
    record = event_dict.get('_record')
    if record is not None:
        # Remove '_record' to prevent circular reference
        event_dict.pop('_record', None)
        record.structlog_event = event_dict.copy()
    return event_dict


class CustomConsoleRenderer(ConsoleRenderer):
    def __call__(self, logger, name, event_dict):
        # Extract call site parameters
        lineno = event_dict.pop('lineno', None)
        filename = event_dict.pop('filename', None)
        func_name = event_dict.pop('func_name', None)
        # application_name = event_dict.pop('application_name', None)
        # local_hash_id=event_dict.pop('local_hash_id', "")
        # Call the parent renderer
        rendered = super().__call__(logger, name, event_dict)
        # Append the call site information to the rendered output
        if filename and lineno and func_name:
            rendered += f" (at {filename}:{lineno} in {func_name}())"
        elif filename and lineno:
            rendered += f" (at {filename}:{lineno})"



        return rendered



def create_logger_in_path(logger_name, logger_file:Union[str,None],application_name:str="tdag", **metadata):
    """
    Create a logger that logs to console and file in JSON format.
    """
    # Ensure the directory for the log file exists
    logger = logging.getLogger(logger_name)
    # if logger.hasHandlers():
    #     logger = structlog.get_logger(logger_name)
    #     return logger

    # Define the timestamper and pre_chain processors
    timestamper = structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S")
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        timestamper,

    ]

    handlers= {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "level":os.getenv("LOG_LEVEL", "DEBUG")
        },

    }
    if logger_file is not None:
        ensure_dir(logger_file)

        handlers.update({"file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "plain",
            "level": os.getenv("LOG_LEVEL_FILE", "DEBUG"),
            "filename": logger_file,
            "mode": "a",
            "maxBytes": 5 * 1024 * 1024,  # Rotate after 5 MB
            "backupCount": 5,  # Keep up to 5 backup files
        }})

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),  # Outputs logs as JSON
                "foreign_pre_chain": pre_chain,
            },
            "colored": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": CustomConsoleRenderer(colors=True),
                "foreign_pre_chain": pre_chain,
            },
        },
        "handlers":handlers,
        "loggers": {
            logger_name: {
                "handlers":list(handlers.keys()),
                "level": os.getenv("LOG_LEVEL_STDOUT", "INFO"),
                "propagate": False,
            },
        },
    }
    try:
    # Configure logging with dictConfig
        logging.config.dictConfig(logging_config)
    except Exception as e:
        raise e
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.CallsiteParameterAdder(
                {
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                    structlog.processors.CallsiteParameter.LINENO,
                }
            ),
            timestamper,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info, #suggested to remove for pretty exceptions
            add_structlog_event_to_record,  # Add this processor before wrap_for_formatter
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        cache_logger_on_first_use=True,
    )

    # Create the structlog logger and bind metadata
    logger = structlog.get_logger(logger_name)
    logger = logger.bind(application_name=application_name,**metadata)

    return logger


def get_tdag_logger():
    # Check if the logger with the name 'virtualfundbuilder' already exists
    logger = logging.getLogger('tdag')

    # If the logger doesn't have any handlers, create it using the custom function
    if not logger.hasHandlers():
        logger_file = os.environ.get('VFB_LOGS_PATH', os.path.join(os.path.expanduser("~"), "tdag/logs"))
        logger = create_logger_in_path(logger_name="tdag", logger_file=logger_file, application_name="tdag")

    return logger