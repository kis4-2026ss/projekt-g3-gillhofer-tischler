from ..state import State, SubTask
import os
from litellm import completion
from typing import List
import json

def analyzer_node(state: State):
    """
    Analyzes the user input and decomposes it into subtasks.
    """
    user_input = state["user_input"]
    print(f"--- Analyzing Task: {user_input} ---")
    
    model = os.getenv("MAIN_LLM_MODEL", "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free")
    
    prompt = f"""
    You are an expert task analyzer and decomposer. Your goal is to interpret a complex user request, identify the core intent, and break it down into smaller, manageable subtasks.
    
    User Request: {user_input}
    
    Step 1: Analyze the overall intent and complexity (1-10).
    Step 2: Decompose the task into a logical sequence of subtasks.
    
    For each subtask, provide:
    - id: A unique string identifier.
    - description: A clear, actionable instruction.
    - expected_output: What the result of this subtask should look like.
    - required_capabilities: A list of capabilities needed (e.g., 'research', 'coding', 'summarization', 'reasoning', 'creativity').
    - priority: An integer (1 for highest priority, larger for lower).
    - dependencies: A list of 'id's of subtasks that must be completed BEFORE this one.
    
    Return ONLY a JSON object with the following structure:
    {{
      "intent": "The identified core intent of the user",
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
        response = completion(
            model=model, 
            messages=[{"role": "system", "content": "You are a precise task analysis engine that only outputs JSON."},
                      {"role": "user", "content": prompt}]
        )
        
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
            
        subtasks = [SubTask(**task) for task in tasks_data]
        
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
        for i, task in enumerate(subtasks):
            deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
            caps = f" [Capabilities: {', '.join(task.required_capabilities)}]" if task.required_capabilities else ""
            print(f"  {i+1}. [{task.id}] {task.description}{deps}{caps}")
        print("-" * 50 + "\n")
        
        return {"subtasks": subtasks, "metadata": metadata}
        
    except Exception as e:
        print(f"Error parsing decomposition: {e}")
        # Fallback to a single subtask if parsing fails
        fallback_task = SubTask(
            id="fallback_1", 
            description=f"Process the user request: {user_input}",
            expected_output="Final result for the user request.",
            required_capabilities=["reasoning"],
            priority=1,
            dependencies=[]
        )
        return {"subtasks": [fallback_task]}
