import unittest
from unittest.mock import patch, MagicMock
from src.orchestrator.nodes.analyzer import analyzer_node
from src.orchestrator.state import State, SubTask
import json

class TestAnalyzer(unittest.TestCase):
    @patch('src.orchestrator.nodes.analyzer.call_with_retry')
    def test_analyzer_success(self, mock_completion):
        # Mock response from LiteLLM
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "intent": "Research and summarize AI impact",
            "complexity": 7,
            "subtasks": [
                {
                    "id": "1", 
                    "description": "Research AI impact",
                    "expected_output": "Impact report",
                    "required_capabilities": ["research"],
                    "priority": 1,
                    "dependencies": []
                },
                {
                    "id": "2", 
                    "description": "Write summary",
                    "expected_output": "Final summary",
                    "required_capabilities": ["summarization"],
                    "priority": 2,
                    "dependencies": ["1"]
                }
            ]
        })
        mock_completion.return_value = mock_response
        
        state: State = {
            "user_input": "Research AI impact and write summary",
            "subtasks": [],
            "final_output": None,
            "metadata": {}
        }
        
        result = analyzer_node(state)
        
        self.assertEqual(len(result["subtasks"]), 2)
        self.assertEqual(result["subtasks"][0].description, "Research AI impact")
        self.assertEqual(result["metadata"]["intent"], "Research and summarize AI impact")
        self.assertEqual(result["metadata"]["complexity"], 7)
        self.assertEqual(result["subtasks"][1].dependencies, ["1"])
        self.assertIsInstance(result["subtasks"][0], SubTask)

    @patch('src.orchestrator.nodes.analyzer.call_with_retry')
    def test_analyzer_json_markdown(self, mock_completion):
        # Mock response with markdown formatting
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "```json\n{\"intent\": \"test\", \"complexity\": 1, \"subtasks\": [{\"id\": \"1\", \"description\": \"task 1\", \"expected_output\": \"out\", \"required_capabilities\": [], \"priority\": 1, \"dependencies\": []}]}\n```"
        mock_completion.return_value = mock_response
        
        state: State = {
            "user_input": "Some task",
            "subtasks": [],
            "final_output": None,
            "metadata": {}
        }
        
        result = analyzer_node(state)
        self.assertEqual(len(result["subtasks"]), 1)
        self.assertEqual(result["subtasks"][0].description, "task 1")
        self.assertEqual(result["metadata"]["intent"], "test")

if __name__ == "__main__":
    unittest.main()
