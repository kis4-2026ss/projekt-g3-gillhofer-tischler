from ..state import State
from ..utils import call_with_retry
from .registry import registry
import os

def executor_node(state: State):
    """
    Executes subtasks using LiteLLM with model rotation on rate limits.
    """
    subtasks = state["subtasks"]
    metadata = state.get("metadata", {})
    rate_limited_models = metadata.get("rate_limited_models", [])
    
    print("--- Executing Subtasks ---")
    
    updated_subtasks = []
    for task in subtasks:
        if task.status == "assigned" and task.assigned_model:
            success = False
            current_model = task.assigned_model
            tried_models = [current_model]
            
            while not success and len(tried_models) <= 3: # Try up to 3 different models
                # Ensure openrouter prefix
                if not current_model.startswith("openrouter/") and "gpt" not in current_model and "claude" not in current_model:
                     current_model = f"openrouter/{current_model}"

                print(f"Executing task: {task.id} with model: {current_model}")
                
                system_prompt = (
                    "You are a specialized worker in an AI swarm. "
                    "Your task is to execute the specific instruction provided. "
                    "Do NOT ask clarifying questions. "
                    f"Expected format: {task.expected_output or 'Provide the content requested.'}"
                )
                
                is_image_task = any(cap in task.required_capabilities for cap in ["image", "image_generation"])
                
                try:
                    kwargs = {
                        "model": current_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": task.description}
                        ]
                    }
                    if is_image_task:
                        kwargs["modalities"] = ["image"]
                    
                    response = call_with_retry(kwargs)
                    task.result = response.choices[0].message.content
                    task.status = "completed"
                    task.assigned_model = current_model # Update in case we rotated
                    success = True
                    print(f"  [SUCCESS] Completed task {task.id}")
                    
                except Exception as e:
                    err_str = str(e).lower()
                    if ("429" in err_str or "rate limit" in err_str) and len(tried_models) < 3:
                        print(f"  [ROTATION] Model {current_model} rate limited. Trying another...")
                        rate_limited_models.append(current_model)
                        
                        # Get next best model excluding already tried ones
                        current_model = registry.get_best_model(task.required_capabilities, exclude_models=tried_models)
                        tried_models.append(current_model)
                    else:
                        print(f"  [ERROR] Task {task.id} failed: {e}")
                        task.status = "failed"
                        break
            
        updated_subtasks.append(task)
    
    metadata["rate_limited_models"] = list(set(rate_limited_models))
    return {"subtasks": updated_subtasks, "metadata": metadata}
