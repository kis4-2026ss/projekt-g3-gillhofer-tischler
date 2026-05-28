import sys
from src.orchestrator.graph import create_orchestrator
from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROBLEM = "Research the top 2 free AI models for coding, summarize their main features, and generate a creative prompt to test them."

def get_user_input(argv):
    """Problem text from CLI args (quoted or not); falls back to the default."""
    if len(argv) > 1:
        return " ".join(argv[1:]).strip()
    return DEFAULT_PROBLEM

def main():
    user_input = get_user_input(sys.argv)
    app = create_orchestrator()

    initial_state = {
        "user_input": user_input,
        "subtasks": {},
        "final_output": None,
        "metadata": {},
        "retry_count": 0
    }

    print("Starting AI Meta-Orchestrator...")
    result = app.invoke(initial_state)

    print("\n--- FINAL OUTPUT ---")
    print(result.get("final_output"))

if __name__ == "__main__":
    main()
