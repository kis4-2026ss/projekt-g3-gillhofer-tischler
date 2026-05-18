from src.orchestrator.graph import create_orchestrator
from dotenv import load_dotenv

load_dotenv()

def main():
    app = create_orchestrator()
    
    initial_state = {
        "user_input": "Research the impact of AI on job markets and write a summary.",
        "subtasks": [],
        "final_output": None,
        "metadata": {}
    }
    
    print("Starting AI Meta-Orchestrator...")
    result = app.invoke(initial_state)
    
    print("\n--- FINAL OUTPUT ---")
    print(result.get("final_output"))

if __name__ == "__main__":
    main()
