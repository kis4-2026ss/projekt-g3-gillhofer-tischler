from ..state import State
from ..utils import call_with_retry, get_main_model
import os

def verifier_node(state: State):
    """
    Scans completed subtasks for code and performs a quick quality check.
    """
    subtasks = state["subtasks"]
    metadata = state.get("metadata", {})
    
    print("--- Verifying Results (Code Quality Check) ---")
    
    updated_subtasks = {}
    verification_count = 0
    
    for task_id, task in subtasks.items():
        if task.status == "completed" and task.result:
            is_coding = any(cap in task.required_capabilities for cap in ["coding", "programming"])
            has_code_blocks = "```" in task.result
            
            if is_coding or has_code_blocks:
                print(f"  [VERIFIER] Performing code audit for task {task_id}...")
                verification_count += 1
                
                model = get_main_model()
                prompt = f"""
                You are a Senior Software Engineer performing a quick code review.
                
                Task Description: {task.description}
                Model Output: {task.result}
                
                Analyze the code for:
                1. Syntax errors or clear logical flaws.
                2. Security vulnerabilities (e.g., hardcoded secrets).
                3. Adherence to the task description.
                
                If the code is correct, simply state: "✅ Code Verified: No issues found."
                If there are issues, provide a brief, actionable correction.
                
                Return ONLY the verification result.
                """
                
                try:
                    kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": "You are a precise technical reviewer."},
                            {"role": "user", "content": prompt}
                        ]
                    }
                    response = call_with_retry(kwargs)
                    verification_note = response.choices[0].message.content
                    
                    # Append verification note to the result
                    updated_task = task.copy(update={
                        "result": f"{task.result}\n\n---\n🛡️ **Code Verification**: {verification_note}"
                    })
                    updated_subtasks[task_id] = updated_task
                    print(f"  [VERIFIER] Audit complete for {task_id}.")
                    
                except Exception as e:
                    print(f"  [VERIFIER] Audit failed for {task_id}: {e}")
                    updated_subtasks[task_id] = task
            else:
                updated_subtasks[task_id] = task
        else:
            updated_subtasks[task_id] = task
            
    if verification_count == 0:
        print("  [VERIFIER] No code blocks identified for audit.")
        
    return {"subtasks": updated_subtasks}
