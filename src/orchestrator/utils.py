import os
import time
import random
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

# List of API keys to rotate through
def get_api_keys():
    keys = []
    # Primary key
    primary = os.getenv("OPENROUTER_API_KEY")
    if primary:
        keys.append(primary)
    
    # Check for additional keys like OPENROUTER_API_KEY_2, _3, etc.
    i = 2
    while True:
        key = os.getenv(f"OPENROUTER_API_KEY_{i}")
        if not key:
            break
        keys.append(key)
        i += 1
    
    # If no keys found, check for any key starting with OPENROUTER_API_KEY_
    for k, v in os.environ.items():
        if k.startswith("OPENROUTER_API_KEY_") and v not in keys:
            keys.append(v)
            
    return keys

API_KEYS = get_api_keys()
CURRENT_KEY_INDEX = 0

def call_with_retry(kwargs, max_retries=None):
    """
    Calls completion with exponential backoff and API key rotation for rate limits.
    """
    global CURRENT_KEY_INDEX
    
    # If max_retries not specified, use 2 * number of keys
    if max_retries is None:
        max_retries = len(API_KEYS) * 2 if API_KEYS else 3
        
    for i in range(max_retries):
        # Set the current key
        if API_KEYS:
            key = API_KEYS[CURRENT_KEY_INDEX]
            kwargs["api_key"] = key
            # Also set it in environment just in case LiteLLM checks there
            os.environ["OPENROUTER_API_KEY"] = key

        try:
            return completion(**kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                # Rotate key if we have more than one
                if len(API_KEYS) > 1:
                    CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
                    print(f"  [ROTATION] Rate limit hit. Switching to API key {CURRENT_KEY_INDEX + 1}...")
                
                wait_time = (2 ** i) + random.random()
                print(f"  [RETRY] Waiting {wait_time:.2f}s before retry {i+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            raise e
            
    return completion(**kwargs)

def get_main_model():
    """Returns the default general purpose model."""
    # Use the first free general model from registry if available, else fallback
    from .nodes.registry import registry
    return registry.get_best_model(["general", "reasoning"])
