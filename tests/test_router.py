import unittest
from src.orchestrator.nodes.registry import CapabilityRegistry, normalize_capabilities
from src.orchestrator.nodes.router import router_node
from src.orchestrator.state import State, SubTask

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

    def test_normalize_capabilities(self):
        self.assertEqual(normalize_capabilities(["creativity", "vision"]), ["creative", "image"])
        self.assertEqual(normalize_capabilities(["CODE", "  Programming "]), ["coding"])
        self.assertEqual(normalize_capabilities(["bogus"]), [])

    def test_creativity_routes_to_creative_model(self):
        # Regression: "creativity" used to match nothing and fall back to models[0]
        # (a dead model). It must now normalize to "creative" and pick a creative model.
        model_id = self.registry.get_best_model(["creativity"])
        model = next(m for m in self.registry.models if m.model_id == model_id)
        self.assertIn("creative", model.capabilities)

    def test_fallback_to_general_when_no_capability_match(self):
        # Exclude every image-capable model so "image" matches nothing -> must fall
        # back to a general-capable model, never the raw first JSON entry.
        image_models = [m.model_id for m in self.registry.models if "image" in m.capabilities]
        chosen = self.registry.get_best_model(["image"], exclude_models=image_models)
        self.assertTrue(chosen.endswith(":free"))
        self.assertNotIn(chosen, image_models)
        model = next(m for m in self.registry.models if m.model_id == chosen)
        self.assertIn("general", model.capabilities)

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
