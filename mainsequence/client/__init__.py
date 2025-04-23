
from .utils import AuthLoaders
from .models_tdag import (request_to_datetime, LocalTimeSeriesDoesNotExist, DynamicTableDoesNotExist,
                          SourceTableConfigurationDoesNotExist, LocalTimeSerieUpdateDetails,
                          JSON_COMPRESSED_PREFIX, Scheduler, SchedulerDoesNotExist, LocalTimeSerie,
                          DynamicTableMetaData, DynamicTableDataSource, LocalTimeSerieNode,
                          Project, UniqueIdentifierRangeMap, LocalTimeSeriesHistoricalUpdate,
                          DataUpdates)

from .utils import TDAG_CONSTANTS, MARKETS_CONSTANTS
from mainsequence.logconf import logger

from .models_helpers import *
from .models_vam import *

class PodDataSource:
    def __init__(self, data_source):
        self.data_source = data_source

try:
    POD_PROJECT = Project.get_user_default_project()
    POD_DEFAULT_DATA_SOURCE = PodDataSource(data_source=POD_PROJECT.data_source)
except Exception as e:
    POD_PROJECT = None
    logger.exception(f"Could not retrive pod project {e}")
    raise e