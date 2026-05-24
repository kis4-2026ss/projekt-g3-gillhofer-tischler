from ..state import State
from litellm import completion
import os
import json

def validator_node(state: State):
    """
    Validates the final output for consistency, correctness, and completeness.
    """
    final_output = state.get("final_output")
    user_input = state.get("user_input")
    subtasks = state.get("subtasks", [])
    
    if not final_output:
        print("--- Validator: No output to validate ---")
        return state

    print("--- Validating Output ---")
    
    model = os.getenv("MAIN_LLM_MODEL", "openrouter/google/gemini-2.0-flash-lite-preview-02-05:free")
    
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
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise quality assurance agent that only outputs JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        
        content = response.choices[0].message.content
        # Clean up JSON if needed
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        validation_data = json.loads(content)
        
        print(f"[VALIDATOR] Quality Score: {validation_data.get('score')}/100")
        print(f"[VALIDATOR] Valid: {validation_data.get('is_valid')}")
        
        # Add validation results to metadata
        metadata = state.get("metadata", {})
        metadata["validation"] = validation_data
        
        return {"metadata": metadata}
        
    except Exception as e:
        print(f"Error during validation: {e}")
        # Default fallback metadata
        metadata = state.get("metadata", {})
        metadata["validation"] = {
            "score": 0,
            "error": str(e),
            "is_valid": False
        }
        return {"metadata": metadata}
