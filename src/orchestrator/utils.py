import os
import time
import random
import threading
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

# Subtasks fan out into parallel worker threads (LangGraph Send), so the shared
# key index and usage counters must be guarded by a lock.
_KEY_LOCK = threading.Lock()


def _current_key():
    """Snapshot the current API key under the lock (None if no keys configured)."""
    if not API_KEYS:
        return None
    with _KEY_LOCK:
        return API_KEYS[CURRENT_KEY_INDEX]


def _rotate_key():
    """Advance to the next API key under the lock and return the new index."""
    global CURRENT_KEY_INDEX
    with _KEY_LOCK:
        if len(API_KEYS) > 1:
            CURRENT_KEY_INDEX = (CURRENT_KEY_INDEX + 1) % len(API_KEYS)
        return CURRENT_KEY_INDEX


# --- Token/cost usage accumulator (thread-safe) -----------------------------
# The benchmark resets this before an orchestrator run and reads it after to
# total tokens across all of the run's internal (and parallel) model calls.
_USAGE_LOCK = threading.Lock()
_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "calls": 0}


def reset_usage():
    with _USAGE_LOCK:
        for k in _USAGE:
            _USAGE[k] = 0


def get_usage():
    with _USAGE_LOCK:
        return dict(_USAGE)


def _record_usage(response):
    """Accumulate token usage from a litellm response; never raises."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return
    with _USAGE_LOCK:
        _USAGE["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _USAGE["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _USAGE["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        _USAGE["calls"] += 1


def call_with_retry(kwargs, max_retries=None):
    """
    Calls completion with exponential backoff and API key rotation for rate limits.
    """
    # If max_retries not specified, use 2 * number of keys
    if max_retries is None:
        max_retries = len(API_KEYS) * 2 if API_KEYS else 3

    for i in range(max_retries):
        # Set the current key per-call (passed directly to litellm — no global
        # env mutation, which would race across parallel task threads).
        key = _current_key()
        if key:
            kwargs["api_key"] = key

        try:
            response = completion(**kwargs)
            _record_usage(response)
            return response
        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "too many requests" in err_str:
                # Rotate key if we have more than one
                new_index = _rotate_key()
                if len(API_KEYS) > 1:
                    print(f"  [ROTATION] Rate limit hit. Switching to API key {new_index + 1}...")

                wait_time = (2 ** i) + random.random()
                print(f"  [RETRY] Waiting {wait_time:.2f}s before retry {i+1}/{max_retries}...")
                time.sleep(wait_time)
                continue
            raise e

    response = completion(**kwargs)
    _record_usage(response)
    return response

def get_main_model():
    """Returns the default general purpose model."""
    # Use the first free general model from registry if available, else fallback
    from .nodes.registry import registry
    return registry.get_best_model(["general", "reasoning"])
