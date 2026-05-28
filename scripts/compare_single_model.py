"""
Compare the AI Meta-Orchestrator against a SINGLE general free OpenRouter model
on the same input(s) — to validate whether the orchestrator is more *successful*
and how much *slower* it is than just asking one free model directly.

For every prompt it measures, for BOTH systems:
  - success: produced a non-empty answer (the orchestrator also reports its own
             internal validation score / is_valid)
  - speed:   wall-clock latency in seconds
  - quality: 0-100 from an independent free "judge" model (scored blind)

This complements scripts/benchmark.py (which compares against a paid frontier model
like GPT-4). Here the baseline is a free model, so the comparison isolates the value
of the orchestration itself — same free-tier resources on both sides.

Usage:
  python scripts/compare_single_model.py                 # runs benchmark/prompts.json
  python scripts/compare_single_model.py "your prompt"   # runs a single prompt

Config (optional, in .env):
  SINGLE_MODEL   override the single baseline model (default: best free 'general' model)
  JUDGE_MODEL    model that scores answers      (default: best free general/reasoning model)

Requires OPENROUTER_API_KEY in .env (the orchestrator and both calls use free models).
"""

import os
import sys
import json
import time
import datetime

# Make the repo root importable whether run as `python scripts/...` or `-m`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

from src.orchestrator.graph import create_orchestrator
from src.orchestrator.nodes.registry import registry
from src.orchestrator.utils import call_with_retry, get_main_model, reset_usage, get_usage

load_dotenv()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS_PATH = os.path.join(ROOT, "benchmark", "prompts.json")
RESULTS_DIR = os.path.join(ROOT, "benchmark", "results")

SINGLE_SYSTEM = "You are a helpful assistant. Answer the user's request as fully and directly as possible."

JUDGE_TEMPLATE = """\
You are a strict evaluator. Score how well the ANSWER fulfills the REQUEST, from 0 to 100.
Rubric: 90-100 excellent and complete; 75-89 good, minor gaps; 60-74 acceptable with clear
gaps; 40-59 weak, major parts missing; 0-39 poor. Be critical and reserve 90+ for genuinely
complete, correct answers.

REQUEST:
{prompt}

ANSWER:
{answer}

Return ONLY this JSON: {{"score": <integer 0-100>, "reason": "one sentence"}}
"""


def _parse_json(content: str) -> dict:
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    content = content.strip()
    if not content.startswith("{"):
        s, e = content.find("{"), content.rfind("}")
        if s != -1 and e != -1:
            content = content[s:e + 1]
    return json.loads(content)


def run_single(model: str, prompt: str) -> dict:
    """One direct call to the single free model."""
    start = time.perf_counter()
    try:
        resp = call_with_retry({
            "model": model,
            "messages": [
                {"role": "system", "content": SINGLE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
        })
        answer = resp.choices[0].message.content or ""
        error = None
    except Exception as e:
        answer, error = "", str(e)
    latency = time.perf_counter() - start
    return {"answer": answer, "latency_s": round(latency, 2), "ok": bool(answer.strip()), "error": error}


def run_orchestrator(app, prompt: str) -> dict:
    """Full orchestrator run (decompose -> route -> parallel execute -> aggregate -> validate)."""
    reset_usage()
    start = time.perf_counter()
    try:
        result = app.invoke(
            {"user_input": prompt, "subtasks": {}, "final_output": None, "metadata": {}, "retry_count": 0},
            config={"recursion_limit": 50},
        )
        answer = result.get("final_output") or ""
        validation = (result.get("metadata") or {}).get("validation", {}) or {}
        error = None
    except Exception as e:
        answer, validation, error = "", {}, str(e)
    latency = time.perf_counter() - start
    usage = get_usage()
    return {
        "answer": answer,
        "latency_s": round(latency, 2),
        "ok": bool(answer.strip()),
        "validation_score": validation.get("score"),
        "is_valid": validation.get("is_valid"),
        "model_calls": usage.get("calls", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "error": error,
    }


def judge_quality(model: str, prompt: str, answer: str) -> int:
    """Blind 0-100 quality score for a single answer (judge never sees which system made it)."""
    if not answer.strip():
        return 0
    try:
        resp = call_with_retry({
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a precise evaluator that only outputs JSON."},
                {"role": "user", "content": JUDGE_TEMPLATE.format(prompt=prompt, answer=answer)},
            ],
        })
        data = _parse_json(resp.choices[0].message.content)
        return max(0, min(100, int(round(float(data.get("score", 0))))))
    except Exception:
        return 0


def compare(prompts, app, single_model: str, judge_model: str) -> list:
    rows = []
    for i, item in enumerate(prompts, 1):
        pid, prompt = item["id"], item["prompt"]
        print(f"\n========== [{i}/{len(prompts)}] {pid} ==========")

        print("--- single free model ---")
        single = run_single(single_model, prompt)
        print(f"    ok={single['ok']} latency={single['latency_s']}s")

        print("--- orchestrator ---")
        orch = run_orchestrator(app, prompt)
        print(f"    ok={orch['ok']} latency={orch['latency_s']}s calls={orch['model_calls']} "
              f"validation={orch['validation_score']}")

        print("--- judging (blind) ---")
        single["quality"] = judge_quality(judge_model, prompt, single["answer"])
        orch["quality"] = judge_quality(judge_model, prompt, orch["answer"])
        print(f"    quality  single={single['quality']}  orchestrator={orch['quality']}")

        rows.append({"id": pid, "prompt": prompt, "single": single, "orchestrator": orch})
    return rows


def print_report(rows, single_model: str, judge_model: str):
    n = len(rows) or 1
    avg = lambda side, key: round(sum((r[side].get(key) or 0) for r in rows) / n, 2)
    wins = lambda: sum(1 for r in rows if r["orchestrator"]["quality"] > r["single"]["quality"])
    ties = lambda: sum(1 for r in rows if r["orchestrator"]["quality"] == r["single"]["quality"])

    print("\n\n==================== SUMMARY ====================")
    print(f"single model : {single_model}")
    print(f"judge model  : {judge_model}\n")

    print(f"{'prompt':<18}{'single q':>10}{'orch q':>9}{'single s':>11}{'orch s':>9}")
    print("-" * 57)
    for r in rows:
        s, o = r["single"], r["orchestrator"]
        print(f"{r['id'][:17]:<18}{s['quality']:>10}{o['quality']:>9}{s['latency_s']:>11}{o['latency_s']:>9}")
    print("-" * 57)

    so, oo = avg("single", "quality"), avg("orchestrator", "quality")
    sl, ol = avg("single", "latency_s"), avg("orchestrator", "latency_s")
    print(f"{'AVG':<18}{so:>10}{oo:>9}{sl:>11}{ol:>9}")

    s_ok = sum(1 for r in rows if r["single"]["ok"])
    o_ok = sum(1 for r in rows if r["orchestrator"]["ok"])
    print(f"\nsuccess (non-empty answer): single {s_ok}/{len(rows)}   orchestrator {o_ok}/{len(rows)}")
    print(f"quality wins (orchestrator > single): {wins()}/{len(rows)}  (ties: {ties()})")
    speed = (ol / sl) if sl else 0
    print(f"avg quality:  single {so}  vs  orchestrator {oo}  (Δ {round(oo - so, 1)})")
    print(f"avg latency:  single {sl}s vs orchestrator {ol}s  (orchestrator ~{round(speed, 1)}x slower — it makes multiple model calls)")


def main():
    if len(sys.argv) > 1:
        prompts = [{"id": "cli", "prompt": " ".join(sys.argv[1:]).strip()}]
    else:
        with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
            prompts = json.load(f)

    single_model = os.getenv("SINGLE_MODEL") or registry.get_best_model(["general"])
    judge_model = os.getenv("JUDGE_MODEL") or get_main_model()

    print(f"Comparing orchestrator vs single free model '{single_model}' on {len(prompts)} prompt(s).")
    app = create_orchestrator()

    rows = compare(prompts, app, single_model, judge_model)
    print_report(rows, single_model, judge_model)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RESULTS_DIR, f"compare_single_{stamp}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"single_model": single_model, "judge_model": judge_model, "results": rows}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nRaw results written to {out_path}")


if __name__ == "__main__":
    main()
