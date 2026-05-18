from ..state import State

def aggregator_node(state: State):
    """
    Aggregates results from subtasks.
    """
    subtasks = state["subtasks"]
    print("--- Aggregating Results ---")
    
    results = [task.result for task in subtasks if task.result]
    final_output = "\n\n".join(results)
    
    return {"final_output": final_output}
