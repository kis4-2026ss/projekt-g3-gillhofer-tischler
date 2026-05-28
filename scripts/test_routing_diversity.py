from src.orchestrator.nodes.registry import registry
from src.orchestrator.state import SubTask

def test_diverse_routing():
    test_scenarios = [
        {
            "name": "Coding Task",
            "capabilities": ["coding"]
        },
        {
            "name": "Research Task",
            "capabilities": ["research"]
        },
        {
            "name": "Reasoning & Logic",
            "capabilities": ["reasoning", "complex-logic"]
        },
        {
            "name": "Summarization",
            "capabilities": ["summarization"]
        },
        {
            "name": "Creative Writing",
            "capabilities": ["creative"]
        },
        {
            "name": "General Assistant",
            "capabilities": ["general"]
        },
        {
            "name": "Image Generation",
            "capabilities": ["image_generation"]
        },
        {
            "name": "Image Analysis",
            "capabilities": ["image_analysis"]
        }
    ]

    print(f"{'Scenario':<25} | {'Assigned Model':<50} | {'Score'}")
    print("-" * 95)

    for scenario in test_scenarios:
        best_model = registry.get_best_model(scenario["capabilities"])
        
        # We can find the score manually for reporting
        scored_models = []
        for model in registry.models:
            match_count = sum(1 for cap in scenario["capabilities"] if cap in model.capabilities)
            if match_count > 0 or not scenario["capabilities"]:
                missing_count = len(scenario["capabilities"]) - match_count
                score = model.latency_score + (missing_count * 10)
                
                quality_bonus = 0
                for cap in scenario["capabilities"]:
                    if cap == "reasoning": quality_bonus += model.reasoning_score
                    if cap == "coding": quality_bonus += model.coding_score
                    if cap == "creative": quality_bonus += model.creativity_score
                
                score -= (quality_bonus * 0.5)
                scored_models.append((model.model_id, score))
        
        scored_models.sort(key=lambda x: x[1])
        score = scored_models[0][1] if scored_models else "N/A"
        
        print(f"{scenario['name']:<25} | {best_model:<50} | {score}")

if __name__ == "__main__":
    print(f"--- Testing Model Routing Logic with {len(registry.models)} Free Models ---\n")
    test_diverse_routing()
    print("\nNote: 'Match Score' is (Latency Score + 5 per missing capability). Lower is better.")
