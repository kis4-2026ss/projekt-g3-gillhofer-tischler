from ..state import State, SubTask
from ..utils import call_with_retry
from .registry import registry
import os

# Errors that mean "this specific model is unusable" — rotate to a different model.
_DEAD_MODEL_MARKERS = ("404", "not found", "no endpoints", "not a valid model", "does not exist")
_RATE_LIMIT_MARKERS = ("429", "rate limit", "too many requests")

MAX_MODELS_PER_TASK = 4


def _ensure_prefix(model: str) -> str:
    if not model.startswith("openrouter/") and "gpt" not in model and "claude" not in model:
        return f"openrouter/{model}"
    return model


# A dependency is "satisfied" once its task reaches a terminal state. We treat a
# failed prerequisite as satisfied too, so a dependent task still runs (degraded)
# instead of being stranded forever — and so a cycle can never deadlock the graph.
_TERMINAL_STATUSES = ("completed", "failed")


def ready_subtasks(subtasks: dict):
    """
    Return the assigned subtasks whose dependencies are all in a terminal state,
    i.e. the next wave that is safe to run in parallel.

    If nothing is ready but assigned tasks remain (a dependency cycle or a dep on
    a task that will never run), every remaining assigned task is returned so the
    graph makes progress instead of hanging.
    """
    assigned = [t for t in subtasks.values() if t.status == "assigned"]
    if not assigned:
        return []

    ready = []
    for task in assigned:
        deps = [subtasks[d] for d in task.dependencies if d in subtasks]
        if all(d.status in _TERMINAL_STATUSES for d in deps):
            ready.append(task)

    if not ready:
        print("[DISPATCH] No subtask is ready but some remain (dependency cycle?); "
              "dispatching all remaining to avoid deadlock.")
        return assigned
    return ready


def prereq_context(task, subtasks: dict):
    """Build the context block from a task's completed prerequisites (empty if none)."""
    deps = [subtasks[d] for d in task.dependencies if d in subtasks]
    if not deps:
        return ""
    parts = []
    for dep in deps:
        if dep.status == "completed" and dep.result:
            parts.append(f"[{dep.id}] {dep.result}")
        else:
            parts.append(f"[{dep.id}] (prerequisite produced no result)")
    return "Context from prerequisite subtasks:\n" + "\n\n".join(parts)


def subtask_worker(input_data: dict):
    """
    Executes a single subtask. This is the worker node for parallel execution.
    Expected input_data: {"task": SubTask, "metadata": dict}
    """
    task: SubTask = input_data["task"]
    metadata: dict = input_data.get("metadata", {})
    rate_limited_models = metadata.get("rate_limited_models", [])
    dead_models = metadata.get("dead_models", [])
    context = input_data.get("context", "")

    current_model = _ensure_prefix(task.assigned_model)
    tried_models = [current_model]

    # Feed completed prerequisite results forward so a dependent task can use them.
    if context:
        user_content = f"{context}\n\nNow perform your task:\n{task.description}"
    else:
        user_content = task.description

    while True:
        current_model = _ensure_prefix(current_model)
        print(f"Executing task: {task.id} with model: {current_model}")

        system_prompt = (
            "You are a specialized worker in an AI swarm. "
            "Your task is to execute the specific instruction provided. "
            "Do NOT ask clarifying questions. "
            f"Expected format: {task.expected_output or 'Provide the content requested.'}"
        )

        is_image_task = any(cap in task.required_capabilities for cap in ["image_generation"])
        
        try:
            kwargs = {
                "model": current_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            }
            
            # Only request image output if the model is supposed to support it
            # and it's an image generation task
            model_caps = []
            for m in registry.models:
                if m.model_id == current_model:
                    model_caps = m.capabilities
                    break
            
            if is_image_task and "image_generation" in model_caps:
                kwargs["modalities"] = ["image"]
            
            try:
                response = call_with_retry(kwargs)
            except Exception as e:
                # If image generation failed, try falling back to text if it was a multimodal request
                if is_image_task and "modalities" in kwargs:
                    print(f"  [FALLBACK] Image generation failed for {current_model}. Requesting a detailed prompt instead...")
                    del kwargs["modalities"]
                    kwargs["messages"].append({
                        "role": "user", 
                        "content": "The image generation failed. Instead, provide a highly detailed, professional-grade prompt that I can use in Midjourney or DALL-E to generate this image. Include lighting, style, and composition details."
                    })
                    response = call_with_retry(kwargs)
                else:
                    raise e

            # Handle multimodal response (images)
            content = response.choices[0].message.content
            
            # Check for image URLs in the response if it was an image task
            if is_image_task:
                import re
                import requests
                from pathlib import Path
                
                # Find URLs
                urls = re.findall(r'(https?://\S+\.(?:png|jpg|jpeg|gif|webp))', content)
                if not urls and hasattr(response, 'data') and response.data:
                    # Some models return data in a different format
                    for item in response.data:
                        if hasattr(item, 'url'):
                            urls.append(item.url)
                
                for url in urls:
                    try:
                        img_data = requests.get(url).content
                        filename = f"task_{task.id}_{os.path.basename(url.split('?')[0])}"
                        if not any(ext in filename for ext in ['.png', '.jpg', '.jpeg']):
                            filename += ".png"
                        output_path = Path("outputs") / filename
                        with open(output_path, "wb") as f:
                            f.write(img_data)
                        print(f"    [IMAGE SAVED] Saved image to {output_path}")
                        content += f"\n\n[LOCAL IMAGE SAVED TO: {output_path}]"
                    except Exception as img_err:
                        print(f"    [IMAGE ERROR] Failed to save image from {url}: {img_err}")

            task.result = content
            task.status = "completed"
            task.assigned_model = current_model
            print(f"  [SUCCESS] Completed task {task.id}")
            
            # Return updated subtask in dictionary format for merging
            return {
                "subtasks": {task.id: task},
                "metadata": {
                    "rate_limited_models": list(set(rate_limited_models)),
                    "dead_models": list(set(dead_models))
                }
            }
            
        except Exception as e:
            err_str = str(e).lower()

            if any(m in err_str for m in _RATE_LIMIT_MARKERS):
                rate_limited_models.append(current_model)
                reason = "rate limited"
            elif any(m in err_str for m in _DEAD_MODEL_MARKERS):
                dead_models.append(current_model)
                reason = "unavailable (no endpoints)"
            else:
                reason = f"error ({e})"

            if len(tried_models) >= MAX_MODELS_PER_TASK:
                print(f"  [ERROR] Task {task.id} failed after {len(tried_models)} models: {e}")
                task.status = "failed"
                return {
                    "subtasks": {task.id: task},
                    "metadata": {
                        "rate_limited_models": list(set(rate_limited_models)),
                        "dead_models": list(set(dead_models))
                    }
                }

            print(f"  [ROTATION] Model {current_model} {reason}. Trying another...")
            excluded = list(set(tried_models + rate_limited_models + dead_models))
            current_model = registry.get_best_model(task.required_capabilities, exclude_models=excluded)
            tried_models.append(_ensure_prefix(current_model))
