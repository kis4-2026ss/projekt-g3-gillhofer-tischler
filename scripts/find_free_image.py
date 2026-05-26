import requests
import json

def find_image_models():
    url = "https://openrouter.ai/api/v1/models?output_modalities=image"
    try:
        response = requests.get(url)
        response.raise_for_status()
        models = response.json().get("data", [])
        
        print(f"{'ID':<50} | {'Price'}")
        print("-" * 60)
        for m in models:
            pricing = m.get("pricing", {})
            prompt_price = pricing.get("prompt", "N/A")
            print(f"{m['id']:<50} | {prompt_price}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    find_image_models()
