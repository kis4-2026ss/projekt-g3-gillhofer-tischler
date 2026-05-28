from ..state import State
from ..utils import call_with_retry, get_main_model

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
    1.  Synthesize the information into a logical flow.
    2.  Ensure all parts of the user's request are addressed.
    3.  IMPORTANT: If any subtask result contains a local file path (e.g., [LOCAL IMAGE SAVED TO: ...]), YOU MUST HIGHLIGHT THIS clearly at the beginning or end of your response so the user knows where their file is.
    4.  IMPORTANT: If a subtask provided an 'image generation prompt' instead of an actual image, present this clearly in a separate block titled 'IMAGE GENERATION PROMPT' for the user to copy-copy.
    5.  IMPORTANT: If any subtask result contains an image URL or a Markdown image (e.g., ![image](url)), YOU MUST PRESERVE IT EXACTLY in the final output.
    6.  Remove any redundant information or conflicting statements.
    7.  Maintain a professional and helpful tone.
    8.  The final output should be the complete answer to the user, not a summary of the tasks.
    
    Final Synthesized Response:
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
        final_output = response.choices[0].message.content
        
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
