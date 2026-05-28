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

        subtasks = {
            "1": SubTask(id="1", description="Task 1", result="Result 1", status="completed"),
            "2": SubTask(id="2", description="Task 2", result="Result 2", status="completed")
        }

        state: State = {
            "user_input": "Combine task 1 and task 2",
            "subtasks": subtasks,
            "final_output": None,
            "metadata": {}
        }

        result = aggregator_node(state)

        # The aggregator appends an "ORCHESTRATOR INSIGHTS" footer, so check the
        # synthesized body is present rather than asserting exact equality.
        self.assertIn("This is the synthesized final response.", result["final_output"])
        mock_call.assert_called_once()

    @patch('src.orchestrator.nodes.aggregator.call_with_retry')
    def test_aggregator_restores_dropped_result(self, mock_call):
        # The synthesis model includes t1 and t2 but drops t3 entirely; the backstop
        # must re-append t3's content so nothing is lost from the main answer.
        mock_response = MagicMock()
        mock_response.choices[0].message.content = (
            "France's capital Paris hosts the Eiffel Tower. "
            "Photosynthesis converts sunlight into chemical energy in plants."
        )
        mock_call.return_value = mock_response

        subtasks = {
            "t1": SubTask(id="t1", description="capital of France",
                          result="The capital of France is Paris and the Eiffel Tower stands there.",
                          status="completed"),
            "t2": SubTask(id="t2", description="explain photosynthesis",
                          result="Photosynthesis converts sunlight into chemical energy in plants.",
                          status="completed"),
            "t3": SubTask(id="t3", description="explain quicksort",
                          result="Quicksort is a divide-and-conquer sorting algorithm with average nlogn complexity.",
                          status="completed"),
        }
        state: State = {
            "user_input": "Answer three distinct questions",
            "subtasks": subtasks,
            "final_output": None,
            "metadata": {},
        }

        result = aggregator_node(state)
        out = result["final_output"]
        # Dropped t3 was restored; t1/t2 already present and not duplicated.
        self.assertIn("Quicksort", out)
        self.assertIn("divide-and-conquer", out)
        self.assertEqual(out.count("divide-and-conquer"), 1)

    @patch('src.orchestrator.nodes.aggregator.call_with_retry')
    def test_aggregator_all_failed(self, mock_call):
        # No completed subtasks -> no synthesis, failure flagged, LLM not called.
        subtasks = {
            "1": SubTask(id="1", description="Task 1", status="failed"),
            "2": SubTask(id="2", description="Task 2", status="failed"),
        }
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
            "subtasks": {"1": SubTask(id="1", description="t", result="r", status="completed")},
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
            "subtasks": {"1": SubTask(id="1", description="t", result="r", status="completed")},
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
            "subtasks": {"1": SubTask(id="1", description="t", status="failed")},
            "final_output": "No results were generated. All subtasks failed to execute (1).",
            "metadata": {"execution_failed": True}
        }

        result = validator_node(state)

        mock_call.assert_not_called()
        validation = result["metadata"]["validation"]
        self.assertEqual(validation["score"], 0)
        self.assertFalse(validation["is_valid"])

    @patch('src.orchestrator.nodes.validator.call_with_retry')
    def test_validator_weighted_subscores(self, mock_call):
        # Overall is a weighted blend (corr 0.4, cons/comp 0.3) of the three sub-scores.
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "consistency_score": 80,
            "correctness_score": 60,
            "completeness_score": 100,
            "feedback": "ok",
        })
        mock_call.return_value = mock_response

        state: State = {
            "user_input": "Test request",
            "subtasks": {"1": SubTask(id="1", description="t", result="r", status="completed")},
            "final_output": "The final answer.",
            "metadata": {},
        }

        result = validator_node(state)
        validation = result["metadata"]["validation"]
        self.assertEqual(validation["score"], 78)  # round(.3*80 + .4*60 + .3*100)
        self.assertTrue(validation["is_valid"])

    @patch('src.orchestrator.nodes.validator.call_with_retry')
    def test_validator_partial_failure_lowers_score(self, mock_call):
        # One of two subtasks failed -> overall is scaled by the 0.5 success ratio.
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "consistency_score": 90,
            "correctness_score": 90,
            "completeness_score": 90,
            "feedback": "ok",
        })
        mock_call.return_value = mock_response

        state: State = {
            "user_input": "Two-part request",
            "subtasks": {
                "1": SubTask(id="1", description="a", result="done", status="completed"),
                "2": SubTask(id="2", description="b", status="failed"),
            },
            "final_output": "Partial answer.",
            "metadata": {},
        }

        result = validator_node(state)
        validation = result["metadata"]["validation"]
        self.assertEqual(validation["score"], 45)  # round(90 * 1/2)
        self.assertFalse(validation["is_valid"])

if __name__ == "__main__":
    unittest.main()
