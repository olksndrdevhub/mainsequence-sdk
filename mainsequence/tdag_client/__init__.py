from build.lib.mainsequence.tdag_client import POD_DEFAULT_DATA_SOURCE

from .models import  ( request_to_datetime,
LocalTimeSeriesDoesNotExist,DynamicTableDoesNotExist,SourceTableConfigurationDoesNotExist,ChatYamls,SignalYamls,LocalTimeSerieUpdateDetails,
    JSON_COMPRESSED_PREFIX,Scheduler ,SchedulerDoesNotExist,LocalTimeSerie,DynamicTableMetaData,
DynamicTableDataSource,LocalTimeSerieNode,PodLocalLake,Project,
BACKEND_DETACHED,DataUpdates,
                       ChatObject)

from .utils import CONSTANTS, get_tdag_client_logger

logger = get_tdag_client_logger()

try:
    POD_PROJECT=Project.get_user_default_project()
    POD_DEFAULT_DATA_SOURCE=POD_PROJECT.data_source

    logger.bind(project_id=POD_PROJECT.id)
except Exception as e:
    POD_PROJECT = None
    logger.exception(f"Could not retrive pod project {e}")
    raise e