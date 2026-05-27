from ..state import State
from ..utils import call_with_retry, get_main_model
import json

def validator_node(state: State):
    """
    Validates the final output for consistency, correctness, and completeness.
    """
    final_output = state.get("final_output")
    user_input = state.get("user_input")
    subtasks = state.get("subtasks", [])
    metadata = state.get("metadata", {})

    if not final_output:
        print("--- Validator: No output to validate ---")
        return state

    # If no subtask actually produced output, the run failed — don't ask the LLM to
    # grade a failure message, just mark it invalid.
    completed = [t for t in subtasks if t.status == "completed" and t.result]
    if not completed or metadata.get("execution_failed"):
        print("--- Validator: No successful subtasks; marking invalid ---")
        metadata["validation"] = {
            "score": 0,
            "is_valid": False,
            "feedback": "No subtasks produced output.",
        }
        print("[VALIDATOR] Quality Score: 0/100")
        print("[VALIDATOR] Valid: False")
        return {"metadata": metadata}

    print("--- Validating Output ---")

    model = get_main_model()

    # Prepare a summary of subtasks for context
    subtask_summary = "\n".join([f"- {t.id}: {t.description} (Status: {t.status})" for t in subtasks])

    prompt = f"""
    You are a Quality Validator for an AI Orchestrator. Your goal is to review the final output of the system and ensure it meets high standards.

    Original User Request: {user_input}

    Subtasks Performed:
    {subtask_summary}

    Final Output to Validate:
    {final_output}

    Evaluate the final output based on these criteria:
    1. Consistency: Is the output logical and free of internal contradictions?
    2. Correctness: Does the output accurately address the facts and requirements of the request?
    3. Completeness: Are all parts of the user's original request and the subtasks fully addressed?

    Provide your evaluation in JSON format:
    {{
      "score": (integer 0-100),
      "consistency_check": "Brief feedback on consistency",
      "correctness_check": "Brief feedback on correctness",
      "completeness_check": "Brief feedback on completeness",
      "is_valid": (boolean, true if score >= 70),
      "feedback": "Overall feedback and suggestions for improvement"
    }}

    Return ONLY the JSON object.
    """

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise quality assurance agent that only outputs JSON."},
                {"role": "user", "content": prompt}
            ]
        }
        response = call_with_retry(kwargs)

        content = response.choices[0].message.content
        # Clean up JSON if needed
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        validation_data = json.loads(content)

        # Decide validity in code rather than trusting the LLM's self-reported boolean.
        try:
            score = int(validation_data.get("score", 0))
        except (TypeError, ValueError):
            score = 0
        validation_data["score"] = score
        validation_data["is_valid"] = score >= 70

        print(f"[VALIDATOR] Quality Score: {score}/100")
        print(f"[VALIDATOR] Valid: {validation_data['is_valid']}")

        # Add validation results to metadata
        metadata["validation"] = validation_data

        return {"metadata": metadata}

    except Exception as e:
        print(f"Error during validation: {e}")
        # Default fallback metadata
        metadata["validation"] = {
            "score": 0,
            "error": str(e),
            "is_valid": False
        }
        return {"metadata": metadata}
