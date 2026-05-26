import json

def analyze_free_models():
    with open("src/orchestrator/nodes/free_models.json", "r") as f:
        models = json.load(f)
        
    for m in models:
        desc = m["description"].lower()
        if "generate" in desc or "synthesis" in desc or "create" in desc or "output" in desc:
            print(f"ID: {m['model_id']}")
            print(f"Description: {m['description'][:200]}...")
            print("-" * 40)

if __name__ == "__main__":
    analyze_free_models()
