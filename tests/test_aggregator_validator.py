import unittest
from unittest.mock import patch, MagicMock
from src.orchestrator.nodes.aggregator import aggregator_node
from src.orchestrator.nodes.validator import validator_node
from src.orchestrator.state import State, SubTask
import json

class TestAggregatorValidator(unittest.TestCase):
    
    @patch('src.orchestrator.nodes.aggregator.completion')
    def test_aggregator_success(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is the synthesized final response."
        mock_completion.return_value = mock_response
        
        subtasks = [
            SubTask(id="1", description="Task 1", result="Result 1", status="completed"),
            SubTask(id="2", description="Task 2", result="Result 2", status="completed")
        ]
        
        state: State = {
            "user_input": "Combine task 1 and task 2",
            "subtasks": subtasks,
            "final_output": None,
            "metadata": {}
        }
        
        result = aggregator_node(state)
        
        self.assertEqual(result["final_output"], "This is the synthesized final response.")
        mock_completion.assert_called_once()

    @patch('src.orchestrator.nodes.validator.completion')
    def test_validator_success(self, mock_completion):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "score": 85,
            "consistency_check": "Consistent",
            "correctness_check": "Correct",
            "completeness_check": "Complete",
            "is_valid": True,
            "feedback": "Good job"
        })
        mock_completion.return_value = mock_response
        
        state: State = {
            "user_input": "Test request",
            "subtasks": [],
            "final_output": "The final answer.",
            "metadata": {}
        }
        
        result = validator_node(state)
        
        validation = result["metadata"]["validation"]
        self.assertEqual(validation["score"], 85)
        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["feedback"], "Good job")

if __name__ == "__main__":
    unittest.main()
