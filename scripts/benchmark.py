"""
Benchmark the AI Meta-Orchestrator against a single baseline model (e.g. GPT-4).

For each prompt in benchmark/prompts.json this runs:
  1. the full orchestrator pipeline, and
  2. a single direct call to BASELINE_MODEL,
then scores BOTH answers blind (0-100) with the same JUDGE_MODEL using the
orchestrator's own quality rubric (consistency / correctness / completeness).
Latency and token usage are recorded; a comparison table is printed and the raw
results are written to benchmark/results/<timestamp>.json.

Config (env / .env):
  BASELINE_MODEL   the single model to beat, e.g. gpt-4o  (needs OPENAI_API_KEY)
  JUDGE_MODEL      the model that scores answers
  MAX_PARALLEL_TASKS   passed through to the orchestrator executor

NOTE ON JUDGE BIAS: if JUDGE_MODEL is the same family as BASELINE_MODEL (e.g.
both GPT-4), the judge tends to favour the baseline. Prefer a THIRD, independent
model as the judge so the comparison is fair.

Usage:
  python scripts/benchmark.py            # all prompts
  python scripts/benchmark.py 2          # first 2 prompts (quick/cheap smoke test)
"""

import os
import sys
import json
import time
import datetime

# Make the repo root importable whether run as `python scripts/benchmark.py` or `-m`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import litellm
from litellm import completion
from dotenv import load_dotenv

from src.orchestrator.graph import create_orchestrator
from src.orchestrator.utils import reset_usage, get_usage

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_PATH = os.path.join(ROOT, "benchmark", "prompts.json")
RESULTS_DIR = os.path.join(ROOT, "benchmark", "results")

BASELINE_MODEL = os.getenv("BASELINE_MODEL", "gpt-4o")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "openrouter/deepseek/deepseek-chat:free")


def _simple_completion(model, system, user, max_retries=3):
    """Direct litellm call that lets litellm resolve provider keys from env.

    Used for the baseline and judge so we don't inject an OpenRouter key into an
    OpenAI model (which call_with_retry would do)."""
    last_err = None
    for i in range(max_retries):
        try:
            return completion(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
        except Exception as e:
            last_err = e
            if "429" in str(e).lower() or "rate limit" in str(e).lower():
                time.sleep(2 ** i)
                continue
            raise
    raise last_err


def _usage_total(response):
    usage = getattr(response, "usage", None)
    return getattr(usage, "total_tokens", 0) or 0 if usage else 0


def _cost(response):
    try:
        return litellm.completion_cost(completion_response=response) or 0.0
    except Exception:
        return 0.0


def run_orchestrator(app, prompt):
    initial_state = {
        "user_input": prompt,
        "subtasks": {},  # dict channel with an operator.ior merge reducer
        "final_output": None,
        "metadata": {},
        "retry_count": 0,
    }
    reset_usage()
    start = time.perf_counter()
    result = app.invoke(initial_state)
    latency = time.perf_counter() - start
    usage = get_usage()
    return {
        "answer": result.get("final_output") or "",
        "latency_s": round(latency, 2),
        "total_tokens": usage.get("total_tokens", 0),
        "model_calls": usage.get("calls", 0),
        "cost_usd": 0.0,  # free OpenRouter models
        "validation": result.get("metadata", {}).get("validation", {}),
    }


def run_baseline(prompt):
    system = "You are a helpful assistant. Answer the user's request fully and directly."
    start = time.perf_counter()
    response = _simple_completion(BASELINE_MODEL, system, prompt)
    latency = time.perf_counter() - start
    return {
        "answer": response.choices[0].message.content or "",
        "latency_s": round(latency, 2),
        "total_tokens": _usage_total(response),
        "model_calls": 1,
        "cost_usd": round(_cost(response), 5),
    }


JUDGE_SYSTEM = "You are a precise quality assurance agent that only outputs JSON."

JUDGE_TEMPLATE = """\
You are a Quality Judge. Score how well the answer addresses the request.

Request:
{prompt}

Answer:
{answer}

Evaluate on:
1. Consistency: logical and free of internal contradictions?
2. Correctness: accurately addresses the facts and requirements?
3. Completeness: are ALL parts of the request addressed?

Return ONLY this JSON:
{{
  "score": (integer 0-100),
  "consistency_check": "brief",
  "correctness_check": "brief",
  "completeness_check": "brief",
  "feedback": "brief overall feedback"
}}
"""


def judge(prompt, answer):
    """Blind 0-100 score for a single answer (judge never sees which system made it)."""
    if not answer.strip():
        return {"score": 0, "feedback": "Empty answer."}
    user = JUDGE_TEMPLATE.format(prompt=prompt, answer=answer)
    try:
        response = _simple_completion(JUDGE_MODEL, JUDGE_SYSTEM, user)
        content = response.choices[0].message.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        content = content.strip()
        if not content.startswith("{"):
            s, e = content.find("{"), content.rfind("}")
            if s != -1 and e != -1:
                content = content[s:e + 1]
        data = json.loads(content)
        try:
            data["score"] = int(data.get("score", 0))
        except (TypeError, ValueError):
            data["score"] = 0
        return data
    except Exception as e:
        return {"score": 0, "feedback": f"Judge error: {e}"}


def main():
    limit = None
    if len(sys.argv) > 1:
        try:
            limit = int(sys.argv[1])
        except ValueError:
            pass

    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    if limit:
        prompts = prompts[:limit]

    print(f"Benchmark: orchestrator vs baseline '{BASELINE_MODEL}', judged by '{JUDGE_MODEL}'")
    print(f"Running {len(prompts)} prompt(s).\n")

    app = create_orchestrator()
    rows = []

    for i, item in enumerate(prompts, 1):
        pid, prompt = item["id"], item["prompt"]
        print(f"\n========== [{i}/{len(prompts)}] {pid} ==========")

        print("--- Running orchestrator ---")
        orch = run_orchestrator(app, prompt)
        print("--- Running baseline ---")
        base = run_baseline(prompt)

        print("--- Judging (blind) ---")
        orch["judge"] = judge(prompt, orch["answer"])
        base["judge"] = judge(prompt, base["answer"])
        orch_score = orch["judge"].get("score", 0)
        base_score = base["judge"].get("score", 0)

        winner = "orchestrator" if orch_score > base_score else ("baseline" if base_score > orch_score else "tie")
        rows.append({
            "id": pid,
            "prompt": prompt,
            "orchestrator": orch,
            "baseline": base,
            "winner": winner,
        })
        print(f"  score  orch={orch_score}  base={base_score}  -> {winner}")

    # ---- Summary ----
    n = len(rows) or 1
    avg = lambda key, side: round(sum(r[side][key] for r in rows) / n, 2)
    avg_score = lambda side: round(sum(r[side]["judge"].get("score", 0) for r in rows) / n, 1)
    wins = lambda who: sum(1 for r in rows if r["winner"] == who)

    print("\n\n================== SUMMARY ==================")
    header = f"{'metric':<22}{'orchestrator':>16}{'baseline':>16}"
    print(header)
    print("-" * len(header))
    print(f"{'avg quality (0-100)':<22}{avg_score('orchestrator'):>16}{avg_score('baseline'):>16}")
    print(f"{'avg latency (s)':<22}{avg('latency_s', 'orchestrator'):>16}{avg('latency_s', 'baseline'):>16}")
    print(f"{'avg tokens':<22}{avg('total_tokens', 'orchestrator'):>16}{avg('total_tokens', 'baseline'):>16}")
    print(f"{'avg model calls':<22}{avg('model_calls', 'orchestrator'):>16}{avg('model_calls', 'baseline'):>16}")
    print(f"{'total cost (USD)':<22}{0.0:>16}{round(sum(r['baseline']['cost_usd'] for r in rows), 4):>16}")
    print("-" * len(header))
    print(f"wins: orchestrator={wins('orchestrator')}  baseline={wins('baseline')}  tie={wins('tie')}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_model": BASELINE_MODEL,
            "judge_model": JUDGE_MODEL,
            "results": rows,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    main()
