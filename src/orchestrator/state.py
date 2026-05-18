from typing import List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
import operator

class SubTask(BaseModel):
    id: str = Field(description="Unique identifier for the subtask")
    description: str = Field(description="Clear instruction of what needs to be done")
    assigned_model: Optional[str] = Field(default=None, description="The model assigned to this subtask")
    result: Optional[str] = Field(default=None, description="The output of the subtask execution")
    status: str = Field(default="pending", description="Status: pending, assigned, completed, failed")

class State(TypedDict):
    user_input: str
    subtasks: Annotated[List[SubTask], operator.add]
    final_output: Optional[str]
    metadata: dict
