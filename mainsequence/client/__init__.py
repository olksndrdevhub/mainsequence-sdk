
from .utils import AuthLoaders
from .models_tdag import (request_to_datetime, LocalTimeSeriesDoesNotExist, DynamicTableDoesNotExist,
                          SourceTableConfigurationDoesNotExist, LocalTimeSerieUpdateDetails,
                          JSON_COMPRESSED_PREFIX, Scheduler, SchedulerDoesNotExist, LocalTimeSerie,
                          DynamicTableMetaData, DynamicTableDataSource, LocalTimeSerieNode, PodLocalLake,
                          Project, UniqueIdentifierRangeMap, LocalTimeSeriesHistoricalUpdate,
                          DataUpdates)

from .utils import TDAG_CONSTANTS, MARKETS_CONSTANTS
from mainsequence.logconf import logger

from .models_helpers import *
from .models_vam import *

try:
    POD_PROJECT = Project.get_user_default_project()
    POD_DEFAULT_DATA_SOURCE = POD_PROJECT.data_source
except Exception as e:
    POD_PROJECT = None
    logger.exception(f"Could not retrive pod project {e}")
    raise e