from langgraph.graph import StateGraph, START, END
from .state import State
from .nodes.analyzer import analyzer_node
from .nodes.router import router_node
from .nodes.executor import executor_node
from .nodes.aggregator import aggregator_node
from .nodes.validator import validator_node

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
    workflow.add_edge("validator", END)

    # Compile the graph
    return workflow.compile()
