from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from .state import State
from .nodes.analyzer import analyzer_node
from .nodes.router import router_node
from .nodes.executor import subtask_worker
from .nodes.aggregator import aggregator_node
from .nodes.validator import validator_node

def dispatch_subtasks(state: State):
    """
    Dispatcher function that fans out to parallel subtask workers.
    Returns a list of Send objects for all subtasks that are ready for execution.
    """
    subtasks = state["subtasks"]
    metadata = state.get("metadata", {})
    
    # In this simple implementation, we fan out all assigned tasks at once.
    # More advanced logic could handle dependencies (e.g., only Send tasks whose deps are 'completed').
    
    sends = []
    for task_id, task in subtasks.items():
        if task.status == "assigned":
            sends.append(Send("executor", {"task": task, "metadata": metadata}))
    
    if not sends:
        # If no tasks are ready (shouldn't happen with router before us), go straight to aggregator
        return "aggregator"
        
    return sends

def should_continue(state: State):
    """
    Determines whether to retry or end based on validation results.
    """
    metadata = state.get("metadata", {})
    validation = metadata.get("validation", {})
    is_valid = validation.get("is_valid", True)
    retry_count = state.get("retry_count", 0)
    
    if is_valid or retry_count >= 2:
        return END
    else:
        print(f"--- Quality Check Failed (Score: {validation.get('score')}). Retrying... (Attempt {retry_count + 1}) ---")
        return "analyzer"

def create_orchestrator():
    # Initialize the graph
    workflow = StateGraph(State)

    # Add nodes
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("router", router_node)
    workflow.add_node("executor", subtask_worker)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("validator", validator_node)

    # Set up edges
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "router")
    
    # Use conditional edge for parallel fan-out
    workflow.add_conditional_edges(
        "router",
        dispatch_subtasks,
        {
            "executor": "executor",
            "aggregator": "aggregator"
        }
    )
    
    # All executor instances fan in to the aggregator
    workflow.add_edge("executor", "aggregator")
    
    workflow.add_edge("aggregator", "validator")
    
    # Conditional edge for validation and retry loop
    workflow.add_conditional_edges(
        "validator",
        should_continue,
        {
            "analyzer": "analyzer",
            END: END
        }
    )

    # Compile the graph
    return workflow.compile()
