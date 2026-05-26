import requests
import os
from dotenv import load_dotenv

load_dotenv()

def check_quota():
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY not found in .env")
        return

    print("Checking OpenRouter Key Status...")
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json().get("data", {})
        
        limit = data.get("limit", 0)
        usage = data.get("usage", 0)
        rate_limit = data.get("rate_limit", {})
        
        print(f"\n--- OpenRouter Quota Status ---")
        print(f"Usage: {usage} units")
        print(f"Daily Limit: {limit} units")
        print(f"Rate Limit (Requests): {rate_limit.get('requests', 'N/A')} per {rate_limit.get('interval', 'N/A')}")
        
        if limit > 0:
            remaining = limit - usage
            print(f"Remaining Today: {remaining} requests/units")
        else:
            print("Note: This key may have an unlimited or credit-based limit.")
            
    except Exception as e:
        print(f"Error checking quota: {e}")

if __name__ == "__main__":
    check_quota()
