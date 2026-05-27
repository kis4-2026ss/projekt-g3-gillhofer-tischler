import unittest
from unittest.mock import patch, MagicMock
from src.orchestrator.nodes.aggregator import aggregator_node
from src.orchestrator.nodes.validator import validator_node
from src.orchestrator.state import State, SubTask
import json

class TestAggregatorValidator(unittest.TestCase):

    @patch('src.orchestrator.nodes.aggregator.call_with_retry')
    def test_aggregator_success(self, mock_call):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is the synthesized final response."
        mock_call.return_value = mock_response

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
        mock_call.assert_called_once()

    @patch('src.orchestrator.nodes.aggregator.call_with_retry')
    def test_aggregator_all_failed(self, mock_call):
        # No completed subtasks -> no synthesis, failure flagged, LLM not called.
        subtasks = [
            SubTask(id="1", description="Task 1", status="failed"),
            SubTask(id="2", description="Task 2", status="failed"),
        ]
        state: State = {
            "user_input": "Do something",
            "subtasks": subtasks,
            "final_output": None,
            "metadata": {},
        }

        result = aggregator_node(state)

        mock_call.assert_not_called()
        self.assertTrue(result["metadata"]["execution_failed"])
        self.assertIn("No results were generated", result["final_output"])

    @patch('src.orchestrator.nodes.validator.call_with_retry')
    def test_validator_success(self, mock_call):
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "score": 85,
            "consistency_check": "Consistent",
            "correctness_check": "Correct",
            "completeness_check": "Complete",
            "is_valid": True,
            "feedback": "Good job"
        })
        mock_call.return_value = mock_response

        state: State = {
            "user_input": "Test request",
            "subtasks": [SubTask(id="1", description="t", result="r", status="completed")],
            "final_output": "The final answer.",
            "metadata": {}
        }

        result = validator_node(state)

        validation = result["metadata"]["validation"]
        self.assertEqual(validation["score"], 85)
        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["feedback"], "Good job")

    @patch('src.orchestrator.nodes.validator.call_with_retry')
    def test_validator_score_below_threshold_is_invalid(self, mock_call):
        # LLM claims is_valid True but score is 50 -> code must override to False.
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "score": 50,
            "is_valid": True,
            "feedback": "Mediocre"
        })
        mock_call.return_value = mock_response

        state: State = {
            "user_input": "Test request",
            "subtasks": [SubTask(id="1", description="t", result="r", status="completed")],
            "final_output": "The final answer.",
            "metadata": {}
        }

        result = validator_node(state)
        validation = result["metadata"]["validation"]
        self.assertEqual(validation["score"], 50)
        self.assertFalse(validation["is_valid"])

    @patch('src.orchestrator.nodes.validator.call_with_retry')
    def test_validator_no_completed_subtasks_short_circuits(self, mock_call):
        # All subtasks failed -> invalid without calling the LLM.
        state: State = {
            "user_input": "Test request",
            "subtasks": [SubTask(id="1", description="t", status="failed")],
            "final_output": "No results were generated. All subtasks failed to execute (1).",
            "metadata": {"execution_failed": True}
        }

        result = validator_node(state)

        mock_call.assert_not_called()
        validation = result["metadata"]["validation"]
        self.assertEqual(validation["score"], 0)
        self.assertFalse(validation["is_valid"])

if __name__ == "__main__":
    unittest.main()
