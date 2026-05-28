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
            pricing = model.get("pricing", {})
            
            # Consider it free if it has the :free suffix OR if prompt/completion costs are zero
            is_free_price = pricing.get("prompt") == "0" and pricing.get("completion") == "0"
            
            if model_id.endswith(":free") or is_free_price:
                # Ensure the model ID starts with the provider prefix for LiteLLM
                # OpenRouter models often need 'openrouter/' prefix explicitly
                full_model_id = f"openrouter/{model_id}" if not model_id.startswith("openrouter/") else model_id

                # Capability Detection
                capabilities = ["general"]
                description = model.get("description", "").lower()
                name = model.get("name", "").lower()
                
                arch = model.get("architecture", {})
                input_mods = arch.get("input_modalities", [])
                output_mods = arch.get("output_modalities", [])
                modality = arch.get("modality", "").lower()

                # Image Generation (Text to Image)
                if "image" in output_mods or "->image" in modality or "image generation" in description or "generate image" in description:
                    capabilities.append("image_generation")
                
                # Image Analysis (Image to Text)
                if "image" in input_mods or "image->" in modality or "vision" in description or "multimodal" in description or "visual" in description:
                    capabilities.append("image_analysis")
                    capabilities.append("image") # Backward compatibility
                
                # Coding
                if any(k in description or k in name for k in ["code", "coder", "programming", "python", "javascript", "developer"]):
                    capabilities.append("coding")
                
                # Reasoning / Logic
                if any(k in description or k in name for k in ["instruct", "chat", "reasoning", "logic", "math", "thinking", "complex", "reasoner"]):
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
                    "model_id": full_model_id,
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
