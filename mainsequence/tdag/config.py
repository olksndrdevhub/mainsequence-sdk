import os
import logging
from pathlib import Path
from enum import Enum
from .utils import read_key_from_yaml, write_yaml, read_yaml
from mainsequence.tdag.logconf import get_tdag_logger

logger = get_tdag_logger()


DEFAULT_RETENTION_POLICY = dict(scheduler_name="default", retention_policy_time="90 days")

TIME_SERIES_SOURCE_TIMESCALE = "timescale"
TIME_SERIES_SOURCE_PARQUET = "parquet"

TDAG_PATH = os.environ.get("TDAG_ROOT_PATH", f"{str(Path.home())}/tdag")
TDAG_CONFIG_PATH = os.environ.get("TDAG_CONFIG_PATH", f"{TDAG_PATH}/config.yml")

TDAG_DATA_PATH = f"{TDAG_PATH}/data"
GT_TEMP_PATH = f"{TDAG_PATH}/temp"
GT_RAY_FOLDER = f"{TDAG_PATH}/ray"





TIME_SERIES_FOLDER = f"{TDAG_DATA_PATH}/time_series_data"
os.makedirs(TIME_SERIES_FOLDER, exist_ok=True)
Path(GT_TEMP_PATH).mkdir(parents=True, exist_ok=True)
Path(GT_RAY_FOLDER).mkdir(parents=True, exist_ok=True)

dir_path = os.path.dirname(os.path.realpath(__file__))

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    IMPORTANT = '\033[45m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

class RunningMode(Enum):
    TRAINING = "train"
    LIVE = "live"

class Configuration:
    OBLIGATORY_ENV_VARIABLES = [
        "VAM_ENDPOINT", 
        "TDAG_ENDPOINT",
        "TDAG_RAY_CLUSTER_ADDRESS",
        "MAINSEQUENCE_TOKEN",
    ]

    def __init__(self):
        self.set_gt_configuration()
        self._assert_env_variables()

    @classmethod
    def add_env_variables_to_registry(cls,env_vars:list):
        cls.OBLIGATORY_ENV_VARIABLES.extend(env_vars)

    def set_gt_configuration(self):
        if not os.path.isfile(TDAG_CONFIG_PATH):
            self._build_template_yaml()

        self.configuration = read_yaml(TDAG_CONFIG_PATH)

    def _assert_env_variables(self):
        for ob_var in self.OBLIGATORY_ENV_VARIABLES:
            assert ob_var in os.environ, f"{ob_var} not in environment variables"

    def _build_template_yaml(self):
        config = {
            "time_series_config": {
                "ignore_update_timeout": False,
                "logs_destination":"logstash",
                "logs_destination_configuration":{"host":"localhost","port":5005}

            },
            "instrumentation_config": {
                "grafana_agent_host": "localhost",
                "export_trace_to_console": False
            }
        }
        write_yaml(path=TDAG_CONFIG_PATH, dict_file=config)

configuration = Configuration()

LOG_LEVELS = {
    'CRITICAL': logging.CRITICAL,
    'ERROR': logging.ERROR,
    'WARNING': logging.WARNING,
    'INFO': logging.INFO,
    'DEBUG': logging.DEBUG,
    'NOTSET': logging.NOTSET,
}

if TDAG_CONFIG_PATH is not None:
    logging_folder = TDAG_DATA_PATH + "/logs"


    LOG_LEVEL = LOG_LEVELS[os.getenv("LOG_LEVEL", "DEBUG")]

class TimeSeriesOGM:
    def __init__(self):
        os.makedirs(self.time_series_config["LOCAL_DATA_PATH"], exist_ok=True)

    @property
    def time_series_config(self):
        ts_config = read_key_from_yaml("time_series_config", path=TDAG_CONFIG_PATH)
        ts_config["LOCAL_DATA_PATH"] = TIME_SERIES_FOLDER
        return ts_config

    def verify_exist(self, target_path):
        os.makedirs(target_path, exist_ok=True)

    @property
    def time_series_folder(self):
        target_path = self.time_series_config["LOCAL_DATA_PATH"]
        self.verify_exist(target_path=target_path)
        return target_path

    @staticmethod
    def get_logging_path():
        return f"{logging_folder}/time_series"

    @staticmethod
    def logging_folder():
        return logging_folder

    @property
    def temp_folder(self):
        target_path = f"{self.time_series_folder}/temp"
        self.verify_exist(target_path=target_path)
        return target_path

    @property
    def local_metadata_path(self):
        target_path = f"{self.time_series_folder}/metadata"
        self.verify_exist(target_path=target_path)
        return target_path

    @property
    def pickle_storage_path(self):
        target_path = f"{self.time_series_folder}/pickled_ts"
        self.verify_exist(target_path=target_path)
        return target_path

    def get_ts_pickle_path(self, local_hash_id: str):
        return f"{self.pickle_storage_path}/{local_hash_id}.pickle"

ogm = TimeSeriesOGM()