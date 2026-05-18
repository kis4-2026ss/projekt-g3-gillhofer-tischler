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
    Decompose the following complex task into a list of smaller, executable subtasks.
    Return ONLY a JSON list of objects, each with 'id' (string) and 'description' (string).
    Do not include any other text or markdown formatting.
    
    Task: {user_input}
    
    Example Output:
    [
      {{"id": "1", "description": "task 1"}},
      {{"id": "2", "description": "task 2"}}
    ]
    """
    
    try:
        response = completion(
            model=model, 
            messages=[{"role": "user", "content": prompt}]
        )
        
        content = response.choices[0].message.content
        print(f"Analyzer Response: {content}")
        
        # Clean up common LLM formatting issues
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        data = json.loads(content)
        
        if isinstance(data, dict) and "subtasks" in data:
            tasks_data = data["subtasks"]
        elif isinstance(data, list):
            tasks_data = data
        else:
            tasks_data = [data]
            
        subtasks = [SubTask(**task) for task in tasks_data]
    except Exception as e:
        print(f"Error parsing decomposition: {e}")
        # Fallback to a single subtask if parsing fails
        subtasks = [SubTask(id="1", description=f"Execute the task: {user_input}")]
    
    return {"subtasks": subtasks}
