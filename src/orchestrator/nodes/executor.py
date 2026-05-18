from ..state import State
from litellm import completion
import os

def executor_node(state: State):
    """
    Executes subtasks using LiteLLM.
    """
    subtasks = state["subtasks"]
    print("--- Executing Subtasks ---")
    
    updated_subtasks = []
    for task in subtasks:
        if task.status == "assigned" and task.assigned_model:
            print(f"Executing task: {task.description} with model: {task.assigned_model}")
            
            try:
                response = completion(
                    model=task.assigned_model,
                    messages=[{"role": "user", "content": task.description}]
                )
                task.result = response.choices[0].message.content
                task.status = "completed"
            except Exception as e:
                print(f"Error executing task {task.id}: {e}")
                task.status = "failed"
            
        updated_subtasks.append(task)
        
    return {"subtasks": updated_subtasks}
