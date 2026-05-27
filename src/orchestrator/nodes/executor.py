from ..state import State
from ..utils import call_with_retry
from .registry import registry

# Errors that mean "this specific model is unusable" — rotate to a different model.
_DEAD_MODEL_MARKERS = ("404", "not found", "no endpoints", "not a valid model", "does not exist")
_RATE_LIMIT_MARKERS = ("429", "rate limit", "too many requests")

MAX_MODELS_PER_TASK = 4


def _ensure_prefix(model: str) -> str:
    if not model.startswith("openrouter/") and "gpt" not in model and "claude" not in model:
        return f"openrouter/{model}"
    return model


def executor_node(state: State):
    """
    Executes subtasks using LiteLLM, rotating to a different model when the
    assigned one is rate-limited, dead (404/no endpoints), or otherwise failing.
    """
    subtasks = state["subtasks"]
    metadata = state.get("metadata", {})
    rate_limited_models = metadata.get("rate_limited_models", [])
    dead_models = metadata.get("dead_models", [])

    print("--- Executing Subtasks ---")

    updated_subtasks = []
    for task in subtasks:
        if task.status == "assigned" and task.assigned_model:
            current_model = _ensure_prefix(task.assigned_model)
            tried_models = [current_model]

            is_image_task = any(cap in task.required_capabilities for cap in ["image", "image_generation"])

            if is_image_task:
                # No free model here can emit binary images, so ask for a vivid,
                # generation-ready textual description instead of attempting bytes.
                system_prompt = (
                    "You are a specialized worker in an AI swarm. The system cannot output "
                    "binary image files, so produce a vivid, richly detailed visual description "
                    "(a generation-ready prompt) for the requested image. Do NOT ask clarifying "
                    "questions and do NOT attempt to output image data."
                )
            else:
                system_prompt = (
                    "You are a specialized worker in an AI swarm. "
                    "Your task is to execute the specific instruction provided. "
                    "Do NOT ask clarifying questions. "
                    f"Expected format: {task.expected_output or 'Provide the content requested.'}"
                )

            while True:
                current_model = _ensure_prefix(current_model)
                print(f"Executing task: {task.id} with model: {current_model}")

                try:
                    kwargs = {
                        "model": current_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": task.description},
                        ],
                    }
                    response = call_with_retry(kwargs)
                    task.result = response.choices[0].message.content
                    task.status = "completed"
                    task.assigned_model = current_model  # Record the model that actually worked
                    print(f"  [SUCCESS] Completed task {task.id}")
                    break

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
                        break

                    print(f"  [ROTATION] Model {current_model} {reason}. Trying another...")
                    excluded = list(set(tried_models + rate_limited_models + dead_models))
                    current_model = registry.get_best_model(task.required_capabilities, exclude_models=excluded)
                    tried_models.append(_ensure_prefix(current_model))

        updated_subtasks.append(task)

    metadata["rate_limited_models"] = list(set(rate_limited_models))
    metadata["dead_models"] = list(set(dead_models))
    return {"subtasks": updated_subtasks, "metadata": metadata}
