from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from .state import State
from .nodes.analyzer import analyzer_node
from .nodes.router import router_node
from .nodes.executor import subtask_worker, ready_subtasks, prereq_context
from .nodes.verifier import verifier_node
from .nodes.aggregator import aggregator_node
from .nodes.validator import validator_node

def dispatch_subtasks(state: State):
    """
    Dispatcher that fans out one parallel worker per subtask that is ready to run.

    A subtask is ready when all of its declared dependencies have reached a
    terminal state; each Send carries the prerequisites' results as context so
    the dependent task can build on them. This runs in waves: after a wave fans
    in to the verifier, this is called again for the next wave, and once nothing
    remains it routes to the aggregator.
    """
    subtasks = state["subtasks"]
    metadata = state.get("metadata", {})

    ready = ready_subtasks(subtasks)
    if not ready:
        return "aggregator"

    return [
        Send("executor", {
            "task": task,
            "metadata": metadata,
            "context": prereq_context(task, subtasks),
        })
        for task in ready
    ]

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
    workflow.add_node("verifier", verifier_node)
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
    
    # All executor instances in a wave fan in to the verifier, which then loops
    # back to the dispatcher for the next dependency wave (or on to the aggregator
    # once every subtask has finished).
    workflow.add_edge("executor", "verifier")
    workflow.add_conditional_edges(
        "verifier",
        dispatch_subtasks,
        {
            "executor": "executor",
            "aggregator": "aggregator"
        }
    )

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
