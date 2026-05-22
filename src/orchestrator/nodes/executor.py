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
            print(f"Executing task: {task.id} - {task.description[:50]}... with model: {task.assigned_model}")
            
            system_prompt = (
                "You are a specialized worker in an AI swarm. "
                "Your task is to execute the specific instruction provided. "
                "Do NOT ask clarifying questions. If you need more information, make reasonable assumptions to fulfill the request. "
                f"Your output should match this expected format: {task.expected_output or 'Provide the content requested.'}"
            )
            
            try:
                response = completion(
                    model=task.assigned_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task.description}
                    ]
                )
                task.result = response.choices[0].message.content
                task.status = "completed"
                print(f"Completed task {task.id}")
            except Exception as e:
                print(f"Error executing task {task.id}: {e}")
                task.status = "failed"
            
        updated_subtasks.append(task)
        
    return {"subtasks": updated_subtasks}
