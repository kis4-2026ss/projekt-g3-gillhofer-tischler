from langgraph.graph import StateGraph, START, END
from .state import State
from .nodes.analyzer import analyzer_node
from .nodes.router import router_node
from .nodes.executor import executor_node
from .nodes.aggregator import aggregator_node
from .nodes.validator import validator_node

def should_continue(state: State):
    """
    Determines whether to retry or end based on validation results.
    """
    metadata = state.get("metadata", {})
    validation = metadata.get("validation", {})
    is_valid = validation.get("is_valid", False)
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
    workflow.add_node("executor", executor_node)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("validator", validator_node)

    # Set up edges
    workflow.add_edge(START, "analyzer")
    workflow.add_edge("analyzer", "router")
    workflow.add_edge("router", "executor")
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
