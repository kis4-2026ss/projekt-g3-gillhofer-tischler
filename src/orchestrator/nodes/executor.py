from ..state import State
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
                            {"role": "user", "content": task.description},
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
                    print(f"    Result (first 100 chars): {content[:100]}...")
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
