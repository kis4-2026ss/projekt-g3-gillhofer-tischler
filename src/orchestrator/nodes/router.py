from ..state import State
from .registry import registry
import os

def router_node(state: State):
    """
    Dynamically assigns models to subtasks based on required capabilities.
    """
    subtasks = state["subtasks"]
    metadata = state.get("metadata", {})
    routing_log = []
    
    print("--- Routing Subtasks Dynamically ---")
    
    updated_subtasks = []
    for task in subtasks:
        if task.status == "pending" or not task.assigned_model:
            # Use the registry to find the best model
            best_model = registry.get_best_model(task.required_capabilities)
            
            task.assigned_model = best_model
            task.status = "assigned"
            
            log_entry = {
                "task_id": task.id,
                "required": task.required_capabilities,
                "assigned": best_model,
                "rationale": f"Matched capabilities: {', '.join(task.required_capabilities)}"
            }
            routing_log.append(log_entry)
            print(f"  [ROUTER] Task {task.id} -> {best_model}")
            
        updated_subtasks.append(task)
        
    metadata["routing_decisions"] = routing_log
    
    return {"subtasks": updated_subtasks, "metadata": metadata}
