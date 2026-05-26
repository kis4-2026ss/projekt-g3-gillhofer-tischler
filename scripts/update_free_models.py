import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

def update_free_models():
    print("Fetching free models from OpenRouter...")
    url = "https://openrouter.ai/api/v1/models"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        all_models = response.json().get("data", [])
        
        free_models = []
        for model in all_models:
            model_id = model.get("id", "")
            if model_id.endswith(":free"):
                # Extract capabilities based on name and description
                description = model.get("description", "").lower()
                name = model.get("name", "").lower()
                
                capabilities = ["general"]
                
                # Coding
                if any(k in description or k in name for k in ["code", "coder", "programming", "python", "javascript", "developer"]):
                    capabilities.append("coding")
                
                # Reasoning / Logic
                if any(k in description or k in name for k in ["instruct", "chat", "reasoning", "logic", "math", "thinking", "complex"]):
                    capabilities.append("reasoning")
                
                # Research / Knowledge
                if any(k in description or k in name for k in ["research", "search", "knowledge", "qa", "information", "impact"]):
                    capabilities.append("research")
                
                # Summarization / Extraction
                if any(k in description or k in name for k in ["summary", "summarize", "extraction", "extract", "rag"]):
                    capabilities.append("summarization")

                # Creative / Roleplay
                if any(k in description or k in name for k in ["creative", "writing", "story", "roleplay", "fiction", "uncensored", "dolphin"]):
                    capabilities.append("creative")
                
                # Complex Logic
                if any(k in description or k in name for k in ["complex-logic", "mathematical", "technical"]):
                    capabilities.append("complex-logic")
                
                # Simple latency score based on context length (proxy for model size/speed)
                context_length = model.get("context_length", 0)
                if context_length > 100000:
                    latency_score = 6 # Likely slower
                elif context_length > 32000:
                    latency_score = 4
                else:
                    latency_score = 2 # Fast
                
                free_models.append({
                    "model_id": model_id,
                    "capabilities": list(set(capabilities)),
                    "cost_per_1k_tokens": 0.0,
                    "latency_score": latency_score,
                    "description": model.get("description", "No description available.")
                })
        
        output_path = "src/orchestrator/nodes/free_models.json"
        with open(output_path, "w") as f:
            json.dump(free_models, f, indent=4)
            
        print(f"Successfully updated {len(free_models)} free models in {output_path}")
        
    except Exception as e:
        print(f"Error updating free models: {e}")

if __name__ == "__main__":
    update_free_models()
