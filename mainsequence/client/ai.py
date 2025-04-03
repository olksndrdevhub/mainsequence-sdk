
from .base import BaseObjectOrm,BasePydanticModel
from typing import Optional


class AgentTool(BaseObjectOrm, BasePydanticModel):
    id: Optional[int] = None
    project_id:int
    name: str

