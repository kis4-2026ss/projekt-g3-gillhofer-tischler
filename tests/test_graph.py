import pytest
from src.orchestrator.graph import create_orchestrator

def test_graph_structure():
    app = create_orchestrator()
    assert app is not None
    # Check if we can run it with a mock state
    initial_state = {
        "user_input": "Test task",
        "subtasks": [],
        "final_output": None,
        "metadata": {}
    }
    result = app.invoke(initial_state)
    assert "final_output" in result
    assert result["final_output"] is not None
