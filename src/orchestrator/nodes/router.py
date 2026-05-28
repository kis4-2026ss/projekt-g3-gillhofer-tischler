from ..state import State
from .registry import registry, normalize_capabilities

def router_node(state: State):
    """
    Dynamically assigns models to subtasks based on required capabilities.
    """
    subtasks = state["subtasks"]
    metadata = state.get("metadata", {})
    dead_models = metadata.get("dead_models", [])
    routing_log = []

    print("--- Routing Subtasks Dynamically ---")

    updated_subtasks = {}
    for task_id, task in subtasks.items():
        if task.status == "pending" or not task.assigned_model:
            # Normalize so the executor's downstream checks see canonical values too
            normalized_caps = normalize_capabilities(task.required_capabilities)
            # Use the registry to find the best model, skipping any known-dead models
            best_model = registry.get_best_model(normalized_caps, exclude_models=dead_models)

            # Create updated task
            updated_task = task.copy(update={
                "required_capabilities": normalized_caps,
                "assigned_model": best_model,
                "status": "assigned"
            })
            
            log_entry = {
                "task_id": task_id,
                "required": normalized_caps,
                "assigned": best_model,
                "rationale": f"Matched capabilities: {', '.join(normalized_caps)}"
            }
            routing_log.append(log_entry)
            print(f"  [ROUTER] Task {task_id} -> {best_model}")
            updated_subtasks[task_id] = updated_task
        else:
            updated_subtasks[task_id] = task
        
    metadata["routing_decisions"] = routing_log
    
    return {"subtasks": updated_subtasks, "metadata": metadata}
