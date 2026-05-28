from typing import List, Optional, TypedDict, Annotated, Dict
import operator
from pydantic import BaseModel, Field

class SubTask(BaseModel):
    id: str = Field(description="Unique identifier for the subtask")
    description: str = Field(description="Clear instruction of what needs to be done")
    expected_output: Optional[str] = Field(default=None, description="Description of what the expected result should look like")
    required_capabilities: List[str] = Field(default_factory=list, description="Capabilities needed (e.g. 'research', 'coding', 'summarization')")
    priority: int = Field(default=1, description="Priority of the task (lower is higher priority)")
    dependencies: List[str] = Field(default_factory=list, description="IDs of tasks that must be completed before this one")
    assigned_model: Optional[str] = Field(default=None, description="The model assigned to this subtask")
    result: Optional[str] = Field(default=None, description="The output of the subtask execution")
    status: str = Field(default="pending", description="Status: pending, assigned, completed, failed")

class State(TypedDict):
    user_input: str
    # Use a dictionary with operator.ior (dictionary merge) to allow parallel updates by ID
    subtasks: Annotated[Dict[str, SubTask], operator.ior]
    final_output: Optional[str]
    metadata: Annotated[dict, operator.ior]
    retry_count: int
