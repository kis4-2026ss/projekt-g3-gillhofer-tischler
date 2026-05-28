from ..state import State, SubTask
import os
from ..utils import call_with_retry, get_main_model
from .registry import registry
from typing import List
import json

def analyzer_node(state: State):
    """
    Analyzes the user input and decomposes it into subtasks.
    """
    user_input = state["user_input"]
    retry_count = state.get("retry_count", 0)
    
    if retry_count > 0:
        print(f"--- Retrying Task Analysis (Attempt {retry_count + 1}) ---")

    print(f"--- Analyzing Task: {user_input} ---")
    
    model = get_main_model()
    
    # Get available capabilities from registry
    all_caps = set()
    for m in registry.models:
        all_caps.update(m.capabilities)
    available_capabilities = ", ".join(sorted(list(all_caps)))

    # On a retry, feed the validator's feedback back in so the re-decomposition
    # fixes what failed instead of repeating the same (poor) split.
    feedback_note = ""
    if retry_count > 0:
        prior_feedback = (state.get("metadata", {}) or {}).get("validation", {}).get("feedback")
        if prior_feedback:
            feedback_note = (
                f"\n    A PREVIOUS attempt was judged insufficient. Reviewer feedback: "
                f"\"{prior_feedback}\".\n"
                f"    Re-decompose to fix this directly — add a missing step, tighten the scope, "
                f"or merge redundant subtasks.\n"
            )

    prompt = f"""
    You are an expert task analyzer and decomposer. Interpret the user's request, identify the core intent, and break it into clean, non-overlapping subtasks that together fully cover it.

    User Request: {user_input}

    Available model capabilities in this system: {available_capabilities}
    {feedback_note}
    Decomposition rules — follow strictly:
    1. ATOMIC: each subtask is ONE self-contained unit of work with a single concrete deliverable. Never bundle two unrelated jobs into one subtask.
    2. NON-OVERLAPPING & COMPLETE: no two subtasks may produce the same output, and together they must cover the entire request — nothing missing, nothing duplicated.
    3. MINIMAL: use the FEWEST subtasks that cleanly cover the request. A simple, single-intent request = exactly ONE subtask. Split only when the request has genuinely distinct steps or needs different capabilities. Do not fragment trivially.
    4. ONE CAPABILITY EACH: give each subtask a single primary capability from the list above so it can be routed to a specialist model (e.g. 'coding' for code, 'research' for fact-finding, 'creative' for prose/imagery, 'summarization' for condensing).
    5. DEPENDENCIES DRIVE PARALLELISM: put an id in "dependencies" ONLY when the subtask genuinely needs another subtask's output before it can start. Independent subtasks MUST have empty dependencies so they execute in parallel. A subtask that combines or builds on earlier results must list those ids.
    6. SPECIFIC: each description states exactly what to produce, not just a topic.

    Useful patterns:
    - Deep research: t1) collect facts/data, t2) analyze & synthesize (depends on t1), t3) verify / fact-check (depends on t2).
    - Image request: exactly ONE subtask using the image capability.
    - Code request: use the 'coding' capability.
    - If the available capabilities cannot fully satisfy the request, do your best with 'reasoning' / 'general'.

    EXAMPLE — request: "Research the best note-taking apps, then write a short blog post recommending one, and suggest a logo concept":
    {{
      "intent": "Recommend a note-taking app via a short blog post plus a logo concept",
      "complexity": 6,
      "subtasks": [
        {{"id": "t1", "description": "Research the top note-taking apps and list each one's key strengths and weaknesses.", "expected_output": "A comparison of 3-5 apps with pros and cons.", "required_capabilities": ["research"], "priority": 1, "dependencies": []}},
        {{"id": "t2", "description": "Write a ~200-word blog post recommending one app, justified by the research.", "expected_output": "A short, polished blog post.", "required_capabilities": ["creative"], "priority": 2, "dependencies": ["t1"]}},
        {{"id": "t3", "description": "Describe a logo concept for the recommended app.", "expected_output": "A vivid visual description suitable for an image generator.", "required_capabilities": ["image"], "priority": 2, "dependencies": ["t1"]}}
      ]
    }}
    Here t2 and t3 both depend only on t1, so they run in parallel once t1 is done.

    For each subtask provide: id, description, expected_output, required_capabilities (from the list above), priority (1 = highest), dependencies (ids that must finish first).

    Return ONLY a JSON object with this structure:
    {{
      "intent": "The core intent of the user",
      "complexity": 5,
      "subtasks": [
        {{
          "id": "t1",
          "description": "...",
          "expected_output": "...",
          "required_capabilities": ["..."],
          "priority": 1,
          "dependencies": []
        }}
      ]
    }}

    Do not include any other text, explanations, or markdown formatting.
    """
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise task analysis engine that only outputs JSON."},
                {"role": "user", "content": prompt}
            ]
        }
        response = call_with_retry(kwargs)
        
        content = response.choices[0].message.content
        print(f"Analyzer Response: {content}")
        
        # Clean up common LLM formatting issues
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Remove any leading/trailing non-JSON characters
        content = content.strip()
        if not content.startswith("{"):
            start_idx = content.find("{")
            end_idx = content.rfind("}")
            if start_idx != -1 and end_idx != -1:
                content = content[start_idx:end_idx+1]
        
        data = json.loads(content)
        
        tasks_data = data.get("subtasks", [])
        if not tasks_data and isinstance(data, list):
            tasks_data = data
            
        subtasks = {task["id"]: SubTask(**task) for task in tasks_data}
        
        # Log metadata
        metadata = state.get("metadata", {})
        metadata["analyzer_model"] = model
        metadata["intent"] = data.get("intent", "Unknown")
        metadata["complexity"] = data.get("complexity", 0)
        metadata["task_count"] = len(subtasks)
        
        # Verbose Logging of Decomposition
        print(f"\n[ANALYZER] Intent: {metadata['intent']}")
        print(f"[ANALYZER] Complexity: {metadata['complexity']}/10")
        print(f"[ANALYZER] Subtasks Created ({len(subtasks)}):")
        for task_id, task in subtasks.items():
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            caps = f" [Capabilities: {', '.join(task.required_capabilities)}]" if task.required_capabilities else ""
            print(f"  - [{task_id}] {task.description}{deps}{caps}")
        print("-" * 50 + "\n")
        
        return {"subtasks": subtasks, "metadata": metadata, "retry_count": retry_count + 1}
        
    except Exception as e:
        print(f"Error parsing decomposition: {e}")
        # Fallback to a single subtask if parsing fails
        fallback_id = "fallback_1"
        fallback_task = SubTask(
            id=fallback_id, 
            description=f"Process the user request: {user_input}",
            expected_output="Final result for the user request.",
            required_capabilities=["reasoning"],
            priority=1,
            dependencies=[]
        )
        return {"subtasks": {fallback_id: fallback_task}, "retry_count": retry_count + 1}
