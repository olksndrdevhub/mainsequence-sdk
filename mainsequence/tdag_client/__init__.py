from .models import  ( request_to_datetime,
LocalTimeSeriesDoesNotExist,DynamicTableDoesNotExist,SourceTableConfigurationDoesNotExist,ChatYamls,SignalYamls,LocalTimeSerieUpdateDetails,
    JSON_COMPRESSED_PREFIX,Scheduler ,SchedulerDoesNotExist,LocalTimeSerie,DynamicTableMetaData,
DynamicTableDataSource,LocalDiskSourceLake,LocalTimeSerieNode,PodLocalLake,TimeScaleDBDataSource,
BACKEND_DETACHED,
                       ChatObject)

from .utils import CONSTANTS, get_tdag_client_logger

logger = get_tdag_client_logger()

try:
    POD_DEFAULT_DATA_SOURCE = DynamicTableDataSource.get_default_data_source_for_token()
except Exception as e:
    POD_DEFAULT_DATA_SOURCE = None
    logger.exception(f"Could not set default data source {e}")