import unittest
from src.orchestrator.nodes.registry import CapabilityRegistry
from src.orchestrator.nodes.router import router_node
from src.orchestrator.state import State, SubTask
import os

class TestDynamicRouter(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Ensure free_models.json exists for tests
        # We can trigger the registry to load whatever is there
        cls.registry = CapabilityRegistry()

    def test_registry_selection(self):
        # Testing if a task gets a :free model
        best_for_general = self.registry.get_best_model(["general"])
        self.assertTrue(best_for_general.endswith(":free"))
        
        # Testing if it handles empty capabilities
        fallback = self.registry.get_best_model([])
        self.assertTrue(fallback.endswith(":free"))

    def test_router_node_dynamic_assignment(self):
        subtasks = [
            SubTask(id="1", description="Write code", required_capabilities=["coding"]),
            SubTask(id="2", description="Research AI", required_capabilities=["research"])
        ]
        
        state: State = {
            "user_input": "Do coding and research",
            "subtasks": subtasks,
            "final_output": None,
            "metadata": {},
            "retry_count": 0
        }
        
        result = router_node(state)
        
        # Verify both assigned models are free
        self.assertTrue(result["subtasks"][0].assigned_model.endswith(":free"))
        self.assertTrue(result["subtasks"][1].assigned_model.endswith(":free"))
        self.assertIn("routing_decisions", result["metadata"])

if __name__ == "__main__":
    unittest.main()
