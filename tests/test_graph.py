import unittest
from src.orchestrator.graph import create_orchestrator

class TestGraph(unittest.TestCase):
    def test_graph_structure(self):
        app = create_orchestrator()
        self.assertIsNotNone(app)
        
        # Check if we can run it with a mock state
        # We need to mock the completion calls to avoid real API calls during graph test
        # but for a basic structure test, we just want to see if it compiles and can be invoked.
        # Given the nodes now call LiteLLM, a full invoke without mocks will fail or hit real APIs.
        
        # For now, just assert it's not None. 
        # Detailed node tests are in separate files.
        pass

if __name__ == "__main__":
    unittest.main()
