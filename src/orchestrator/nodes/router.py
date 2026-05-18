from ..state import State
import os

def router_node(state: State):
    """
    Assigns models to subtasks (Initially manual).
    """
    subtasks = state["subtasks"]
    print("--- Routing Subtasks ---")
    
    # Use a free model by default to avoid credit issues
    default_free_model = os.getenv("MAIN_LLM_MODEL", "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free")
    
    updated_subtasks = []
    for task in subtasks:
        # For now, assign the same free model to all tasks
        task.assigned_model = default_free_model
        task.status = "assigned"
        updated_subtasks.append(task)
        
    return {"subtasks": updated_subtasks}
