import logging
import logging.config
import os
from pathlib import Path

import requests
import structlog
from typing import Union

from requests.structures import CaseInsensitiveDict
from structlog.dev import ConsoleRenderer
from .instrumentation import OTelJSONRenderer
logger = None

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

def build_application_logger(
        application_name:str="ms-sdk",
        **metadata
):
    """
    Create a logger that logs to console and file in JSON format.
    """

    # do initial request when on logger initialization
    headers = CaseInsensitiveDict()
    headers["Content-Type"] = "application/json"
    headers["Authorization"] = "Token " + os.getenv("MAINSEQUENCE_TOKEN")

    project_info_endpoint = f'{os.getenv("TDAG_ENDPOINT")}/orm/api/pods/job/get_job_startup_state/'

    command_id = os.getenv("COMMAND_ID")
    params = {}
    if command_id:
        params['command_id'] = command_id

    response = requests.get(project_info_endpoint, headers=headers, params=params)

    if response.status_code != 200:
        print(f"Got Status Code {response.status_code} with response {response.text}")

    json_response = response.json()
    if "project_id" not in json_response:
        raise ValueError(f"Project ID not found, server response {json_response}")

    # set additional args from backend
    if "additional_environment" in json_response:
        for key, value in json_response["additional_environment"].items():
            os.environ[key] = value


    logger_file = os.getenv("LOGGER_FILE_PATH", "/var/log/tdag/tdag.log")

    logger_name = "tdag"

    # Define the timestamper and pre_chain processors
    timestamper = structlog.processors.TimeStamper( fmt="iso",
    utc=True,)
    pre_chain = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        timestamper,
    ]

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colored",
            "level":os.getenv("LOG_LEVEL", "DEBUG")
        },
    }
    if logger_file is not None:
        ensure_dir(logger_file) # Ensure the directory for the log file exists

        handlers.update(
            {
                "file": {
                        "class": "concurrent_log_handler.ConcurrentRotatingFileHandler",
                        "formatter": "plain",
                        "level": os.getenv("LOG_LEVEL_FILE", "DEBUG"),
                        "filename": logger_file,
                        "mode": "a",
                        "delay": True,
                        "maxBytes": 5 * 1024 * 1024,  # Rotate after 5 MB
                        "backupCount": 5,  # Keep up to 5 backup files
                }
            }
        )

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor":OTelJSONRenderer(),#structlog.processors.JSONRenderer(),
                "foreign_pre_chain": pre_chain,
            },
            "colored": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": CustomConsoleRenderer(colors=True),
                "foreign_pre_chain":  pre_chain,
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
            structlog.contextvars.merge_contextvars,  # context that always appears in the logs
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

    try:
        logger = logger.bind(project_id=json_response["project_id"], **metadata)
        logger = logger.bind(data_source_id=json_response["data_source_id"], **metadata)
        logger = logger.bind(job_run_id=json_response["job_run_id"], **metadata)
        logger = logger.bind(command_id=command_id, **metadata)

    except Exception as e:
        logger.exception(f"Could not retrive pod project {e}")
        raise e

    logger = logger.bind()
    return logger

logger = build_application_logger()
