
from .models import  ( request_to_datetime,
LocalTimeSeriesDoesNotExist,DynamicTableDoesNotExist,SourceTableConfigurationDoesNotExist,ChatYamls,SignalYamls,LocalTimeSerieUpdateDetails,
    JSON_COMPRESSED_PREFIX,Scheduler ,SchedulerDoesNotExist,LocalTimeSerie,DynamicTableMetaData,
DynamicTableDataSource,LocalTimeSerieNode,PodLocalLake,Project,
BACKEND_DETACHED,DataUpdates,logger,
                       ChatObject)

from .utils import CONSTANTS, get_tdag_client_logger
from mainsequence.logconf import logger




try:
    POD_PROJECT=Project.get_user_default_project()
    POD_DEFAULT_DATA_SOURCE=POD_PROJECT.data_source



except Exception as e:
    POD_PROJECT = None
    logger.exception(f"Could not retrive pod project {e}")
    raise e