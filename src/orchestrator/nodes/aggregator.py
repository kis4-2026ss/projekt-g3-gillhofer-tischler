from ..state import State
from ..utils import call_with_retry, get_main_model
import re


def _signature_words(text: str) -> set:
    """Distinctive (5+ char) tokens, used to check whether a result survived synthesis."""
    return {w for w in re.findall(r"[a-z0-9]{5,}", (text or "").lower())}


def _coverage(result: str, output: str) -> float:
    """Fraction of a result's distinctive words that appear in the aggregated output."""
    sig = _signature_words(result)
    if not sig:
        return 1.0
    return len(sig & _signature_words(output)) / len(sig)


def aggregator_node(state: State):
    """
    Aggregates results from subtasks and synthesizes a final response.
    """
    subtasks = state["subtasks"]
    user_input = state["user_input"]
    metadata = state.get("metadata", {})
    print("--- Aggregating Results ---")

    subtask_list = list(subtasks.values())
    completed = [task for task in subtask_list if task.status == "completed" and task.result]

    if not completed:
        failed_ids = ", ".join(task.id for task in subtask_list) or "none"
        metadata["execution_failed"] = True
        print(f"  [AGGREGATOR] No subtasks produced output (failed: {failed_ids}).")
        return {
            "final_output": f"No results were generated. All subtasks failed to execute ({failed_ids}).",
            "metadata": metadata,
        }

    results = [f"Subtask ({task.id}): {task.result}" for task in completed]
    results_text = "\n\n".join(results)
    
    # The available free models cannot emit binary image files, so a pure image
    # request is answered with a text description. Only disclaim when the WHOLE
    # request was image generation — not when one subtask of a larger task happens
    # to carry an "image" capability.
    had_image_task = bool(completed) and all("image" in task.required_capabilities for task in completed)
    image_note = (
        "\n\n_Note: the available free models cannot generate actual image files, "
        "so the above is a detailed visual description you can feed to an image generator._"
    )

    model = get_main_model()
    
    prompt = f"""
    You are a Result Aggregator. Your task is to take the outputs from several specialized subtasks and synthesize them into a single, coherent, and comprehensive response that directly addresses the original user request.
    
    Original User Request: {user_input}
    
    Subtask Results:
    {results_text}
    
    Instructions:
    1.  INCLUDE THE FULL CONTENT of EVERY subtask result above. Never omit, drop, or summarize-away any subtask's deliverable — the user must receive ALL of them. This is the most important rule.
    2.  Organize distinct deliverables under clear markdown section headings (## Heading). Closely related results may be merged into one coherent section.
    3.  Preserve concrete artifacts VERBATIM: code inside fenced ```code``` blocks, tables, data, formulas, and lists.
    4.  If a result contains a local file path (e.g. [LOCAL IMAGE SAVED TO: ...]), highlight it clearly.
    5.  If a result contains an image URL or a Markdown image (e.g. ![image](url)), preserve it EXACTLY.
    6.  Only remove EXACT duplicate sentences shared across subtasks. Never drop unique content just to shorten the text.
    7.  Write the answer itself in a professional tone — not a description of the tasks.

    Final Response:
    """
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert synthesizer. Your goal is to provide a final, polished response based on multiple input sources."},
                {"role": "user", "content": prompt}
            ]
        }
        response = call_with_retry(kwargs)
        final_output = response.choices[0].message.content or ""

        # Safety net: a weak synthesis model sometimes silently drops whole subtask
        # deliverables. Detect any completed result largely absent from the synthesis
        # and append it verbatim so nothing is lost from the main answer.
        missing = [t for t in completed if _coverage(t.result, final_output) < 0.25]
        if missing:
            restored = ["\n\n---\n\n## Additional task results"]
            for t in missing:
                heading = (t.description or t.id).strip().rstrip(".")
                restored.append(f"### {heading}\n\n{t.result.strip()}")
            final_output += "\n\n" + "\n\n".join(restored)
            print(f"  [AGGREGATOR] Restored {len(missing)} subtask result(s) dropped during synthesis.")

        # Add Orchestrator Insights
        insights = ["\n\n--- 🤖 ORCHESTRATOR INSIGHTS ---"]
        for task in subtask_list:
            model_name = task.assigned_model.split("/")[-1] if task.assigned_model else "Unknown"
            status_emoji = "✅" if task.status == "completed" else "❌"
            insights.append(f"{status_emoji} **{task.id}**: `{model_name}` was selected for its high proficiency in: {', '.join(task.required_capabilities)}")
        
        final_output += "\n".join(insights)

        if had_image_task:
            final_output += image_note
        print(f"Aggregation complete. Length: {len(final_output)} characters.")

        return {"final_output": final_output}
    except Exception as e:
        print(f"Error during aggregation: {e}")
        # Fallback to simple concatenation if LLM fails
        final_output = "\n\n".join(task.result for task in completed)
        if had_image_task:
            final_output += image_note
        return {"final_output": final_output}
