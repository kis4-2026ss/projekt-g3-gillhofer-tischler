from ..state import State
from ..utils import call_with_retry, get_main_model
import json

def validator_node(state: State):
    """
    Validates the final output for consistency, correctness, and completeness.
    """
    final_output = state.get("final_output")
    user_input = state.get("user_input")
    subtasks = state.get("subtasks", {})
    metadata = state.get("metadata", {})

    subtask_list = list(subtasks.values()) if isinstance(subtasks, dict) else subtasks

    if not final_output:
        print("--- Validator: No output to validate ---")
        return state

    # If no subtask actually produced output, the run failed — don't ask the LLM to
    # grade a failure message, just mark it invalid.
    completed = [t for t in subtask_list if t.status == "completed" and t.result]
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

    # Include a snippet of each subtask's actual result so completeness can be judged
    # against what was produced — not just the task descriptions.
    subtask_summary = "\n".join(
        f"- {t.id} [{t.status}]: {t.description}"
        + (f"\n      produced: {t.result.strip()[:300]}" if t.result else "")
        for t in subtask_list
    )

    prompt = f"""
    You are a STRICT Quality Validator for an AI Orchestrator. Score the final output critically and honestly.

    Original User Request: {user_input}

    Subtasks Performed:
    {subtask_summary}

    Final Output to Validate:
    {final_output}

    Rate the output on three INDEPENDENT dimensions, each from 0 to 100:
    - consistency: logical, coherent, free of internal contradictions.
    - correctness: factually accurate and actually does what was asked.
    - completeness: every part of the request is addressed AND every subtask's produced result is reflected in the final output. If a subtask produced a deliverable that is missing from the final output, completeness MUST be low.

    Scoring guide (apply honestly — most real outputs are NOT perfect):
      90-100 = flawless, fully meets the bar
      75-89  = good, only minor gaps or rough edges
      60-74  = acceptable but with clear gaps or inaccuracies
      40-59  = weak, major parts missing or wrong
      0-39   = poor, largely fails the request
    Judge each dimension separately and do NOT default to round numbers like 90 or 100;
    reserve 95+ for genuinely excellent work.

    Return ONLY this JSON object:
    {{
      "consistency_score": <integer 0-100>,
      "correctness_score": <integer 0-100>,
      "completeness_score": <integer 0-100>,
      "consistency_check": "one-sentence justification",
      "correctness_check": "one-sentence justification",
      "completeness_check": "one-sentence justification",
      "feedback": "specific, actionable suggestions to improve the output"
    }}
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

        # Build the score in code from the three sub-scores rather than trusting a
        # single gestalt number (which the LLM tends to peg at ~95). Correctness is
        # weighted highest. Falls back to a legacy single "score" field if present.
        def _clamp(value):
            try:
                return max(0, min(100, int(round(float(value)))))
            except (TypeError, ValueError):
                return None

        cons = _clamp(validation_data.get("consistency_score"))
        corr = _clamp(validation_data.get("correctness_score"))
        comp = _clamp(validation_data.get("completeness_score"))
        present = [s for s in (cons, corr, comp) if s is not None]

        if cons is not None and corr is not None and comp is not None:
            base = round(0.30 * cons + 0.40 * corr + 0.30 * comp)
        elif present:
            base = round(sum(present) / len(present))
        else:
            base = _clamp(validation_data.get("score")) or 0

        # Tie quality to execution success: a run that only completed part of its
        # subtasks cannot score as if everything had worked.
        total = len(subtask_list) or 1
        success_ratio = len(completed) / total
        score = round(base * success_ratio)

        validation_data["consistency_score"] = cons
        validation_data["correctness_score"] = corr
        validation_data["completeness_score"] = comp
        validation_data["score"] = score
        validation_data["is_valid"] = score >= 70

        print(f"[VALIDATOR] Sub-scores — consistency:{cons} correctness:{corr} completeness:{comp}")
        if success_ratio < 1:
            print(f"[VALIDATOR] Execution penalty: {len(completed)}/{total} subtasks completed (x{success_ratio:.2f})")
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
