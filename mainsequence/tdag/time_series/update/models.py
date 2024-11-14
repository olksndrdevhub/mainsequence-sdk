from datetime import datetime

from pydantic import BaseModel
from typing import List, Optional
import datetime

class StartUpdateDataInfo(BaseModel):
    update_time_start:datetime.datetime
    update_time_end:Optional[datetime.datetime]=None
    update_completed:bool
    error_on_update:bool
    last_time_index_value:Optional[datetime.datetime]=None
    must_update:bool
    direct_dependencies_hash_id:List[str]
