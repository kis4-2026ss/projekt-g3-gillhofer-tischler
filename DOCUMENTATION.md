# AI Meta-Orchestrator — Project Documentation

*FH Hagenberg · KIS course project · Group 3 (Gillhofer / Tischler)*

---

## 1. What it is

The **AI Meta-Orchestrator** turns a single complex request into a coordinated team of AI models.
Instead of sending everything to one model, it:

1. **Decomposes** the request into smaller subtasks,
2. **Routes** each subtask to the best-suited *free* model on OpenRouter,
3. **Executes** independent subtasks **in parallel** (respecting dependencies),
4. **Verifies** generated code,
5. **Aggregates** the subtask outputs into one coherent answer, and
6. **Validates** the answer's quality, retrying if it falls short.

It is built on **LangGraph** (a state-machine framework for LLM workflows) and **LiteLLM** (a
unified client for many model providers). It ships with two interfaces: a **Streamlit GUI** and a
**command-line** entry point.

---

## 2. Why build it (motivation)

> "No single free model is good at everything. A team of specialists, coordinated well, can be."

- **Specialization** — a coding model writes code, a research model gathers facts, a creative model
  writes prose, all within one request.
- **Cost** — it uses only OpenRouter **free-tier** models, so a full run costs ~€0.
- **Speed** — independent subtasks run concurrently, so wall-clock time is the *critical path*, not
  the sum of all tasks.
- **Resilience** — if a model is rate-limited or unavailable, it automatically rotates to another
  model (and another API key).
- **Transparency** — every routing decision, subtask result, and quality score is exposed in the UI.

---

## 3. Architecture at a glance

```
                ┌─────────────┐
  user request →│  ANALYZER   │  decompose into subtasks (with dependencies)
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │   ROUTER    │  pick the best free model per subtask
                └──────┬──────┘
                       ▼
              ┌──────────────────┐   parallel fan-out (LangGraph Send),
              │  EXECUTOR (×N)   │   one worker per ready subtask
              └──────┬───────────┘
                     ▼
                ┌─────────────┐
                │  VERIFIER   │  code audit
                └──────┬──────┘
                       │  more subtasks ready? ── yes ──► back to EXECUTOR (next wave)
                       │  no
                       ▼
                ┌─────────────┐
                │ AGGREGATOR  │  synthesize one answer (never drops a deliverable)
                └──────┬──────┘
                       ▼
                ┌─────────────┐
                │  VALIDATOR  │  score 0–100; if < 70 and retries left → back to ANALYZER
                └──────┬──────┘
                       ▼
                  final answer
```

State flows through the graph as a single dictionary; each node reads it and returns updates.

---

## 4. The pipeline stages

| Stage | File | What it does |
|-------|------|--------------|
| **Analyzer** | `src/orchestrator/nodes/analyzer.py` | Reads the request, decides complexity, and decomposes it into atomic, non-overlapping subtasks with explicit dependencies. On a retry it incorporates the validator's feedback. |
| **Router** | `src/orchestrator/nodes/router.py` + `registry.py` | Normalizes each subtask's required capabilities and asks the **Capability Registry** for the best free model for that capability. |
| **Executor** | `src/orchestrator/nodes/executor.py` | Runs the subtasks. Independent ones run **in parallel**; dependent ones receive their prerequisites' results as context. Rotates models on rate-limit/404. Image requests produce a textual description. |
| **Verifier** | `src/orchestrator/nodes/verifier.py` | For subtasks that produced code, runs a quick "senior engineer" audit for obvious bugs/security issues. |
| **Aggregator** | `src/orchestrator/nodes/aggregator.py` | Synthesizes all subtask results into one answer. A safety net re-appends any deliverable the synthesis model accidentally dropped. |
| **Validator** | `src/orchestrator/nodes/validator.py` | Scores the answer 0–100 on consistency, correctness, and completeness. If the score is below 70 (and retries remain), the run loops back to the analyzer. |

---

## 5. Key features

- **Capability-based dynamic routing** across ~27 free OpenRouter models (catalog in
  `src/orchestrator/nodes/free_models.json`).
- **True parallel execution** via LangGraph's `Send` API, scheduled in **dependency waves**: each
  wave runs all currently-runnable subtasks at once; a dependent subtask only starts once its
  prerequisites finish, and receives their output as context.
- **Robust API handling** — multiple OpenRouter keys with **thread-safe rotation** and exponential
  backoff on rate limits (HTTP 429); automatic model rotation when a model is dead (404) or limited.
- **Honest quality scoring** — the validator blends three independent sub-scores and scales the
  result by the fraction of subtasks that actually succeeded.
- **No silent data loss** — the aggregator guarantees every completed subtask's result appears in
  the final answer.
- **Image requests** — free models can't emit image files, so an image request is answered with a
  vivid, generation-ready *description* (directly, without asking the user first).
- **Two interfaces** — a polished Streamlit GUI with a live parallel-execution view, and a CLI.
- **Validation tooling** — scripts to benchmark the orchestrator against a single model.

---

## 6. Setup

**Requirements:** Python 3.10+, and the dependencies in `pyproject.toml`
(`langgraph`, `langchain`, `langchain-openai`, `litellm`, `pydantic`, `python-dotenv`, `streamlit`).

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate           # Windows
# source venv/bin/activate      # macOS/Linux

# 2. install the project
pip install -e .

# 3. add your OpenRouter key(s) to a .env file (see below)
```

`.env` (never commit this — it is git-ignored):

```ini
OPENROUTER_API_KEY=sk-or-v1-...your-first-key...
OPENROUTER_API_KEY_2=sk-or-v1-...optional-second-key...   # used automatically when the first is rate-limited
```

Get free keys at <https://openrouter.ai>. You can add as many fallback keys as you like
(`OPENROUTER_API_KEY_3`, `_4`, …); the orchestrator rotates through them on rate limits.

---

## 7. Usage

### Graphical interface (recommended)

```bash
python -m streamlit run app.py
```

A browser tab opens. Type a problem in the chat box; watch the subtasks light up and run in
parallel; read the synthesized answer with its quality score. You can keep refining or asking new
problems in the same session.

### Command line

```bash
python main.py "Research Rust vs Go for backends, give a recommendation, and a tagline."
```

### Keep the model catalog fresh

```bash
python scripts/update_free_models.py
```

### Validate that the orchestrator helps (see §10)

```bash
python scripts/compare_single_model.py            # vs a single free model
python scripts/benchmark.py                        # vs a paid baseline (e.g. GPT-4)
```

---

## 8. Configuration

Only these environment variables are read by the code:

| Variable | Used by | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | core (`utils.py`) | Primary OpenRouter key. **Required.** |
| `OPENROUTER_API_KEY_2`, `_3`, … | core (`utils.py`) | Fallback keys; rotation switches to the next on a 429. |
| `BASELINE_MODEL` | `scripts/benchmark.py` | The single model to compare against (default `gpt-4o`). |
| `JUDGE_MODEL` | `scripts/benchmark.py`, `compare_single_model.py` | The model that scores answers. |
| `SINGLE_MODEL` | `scripts/compare_single_model.py` | Override the free baseline model (default: best free `general` model). |

The model used for the analyzer/aggregator/validator/verifier is **auto-selected** from the registry
(`get_main_model()` → best free general+reasoning model), so no model env var is needed for normal runs.

---

## 9. Project structure

```
app.py                         # Streamlit GUI
main.py                        # CLI entry point
pyproject.toml                 # dependencies & metadata
.env / .env.example            # API keys & config (.env is git-ignored)

src/orchestrator/
  graph.py                     # the LangGraph workflow: nodes, edges, parallel fan-out, retry loop
  state.py                     # State (shared dict) and SubTask data model
  utils.py                     # API-key rotation, retry/backoff, token-usage tracking
  nodes/
    analyzer.py                # decomposition
    router.py                  # model assignment
    registry.py                # capability registry + model selection
    executor.py                # parallel subtask worker + dependency waves
    verifier.py                # code audit
    aggregator.py              # result synthesis (+ no-loss backstop)
    validator.py               # quality scoring
    free_models.json           # catalog of free models + capabilities

scripts/
  benchmark.py                 # orchestrator vs a paid baseline model
  compare_single_model.py      # orchestrator vs a single free model (success / speed / quality)
  update_free_models.py        # regenerate free_models.json
  test_routing_diversity.py    # sanity-check routing spread
  check_quota.py               # check key quota

benchmark/
  prompts.json                 # evaluation prompts
  results/                     # saved comparison runs

tests/                         # unit tests (analyzer, router, aggregator/validator, executor waves, graph)
```

---

## 10. Validating "does it work, and is it fast?"

Two scripts answer this objectively (both write a JSON record to `benchmark/results/`):

- **`scripts/compare_single_model.py`** — runs the same input through the orchestrator **and** a
  single general free model, then reports, for each: success (non-empty answer + the orchestrator's
  own validation score), **latency** (speed), and a blind **quality score (0–100)** from an
  independent judge model. Summary shows average quality, average latency, and how often the
  orchestrator wins. *Apples-to-apples: both sides use free models, so it isolates the value of
  orchestration.* The orchestrator is expected to be slower (it makes several calls) but to win on
  quality for multi-part requests.

- **`scripts/benchmark.py`** — the same idea, but against a **paid frontier model** (e.g. GPT-4) to
  tell the "quality-per-cost" story.

```bash
python scripts/compare_single_model.py "Research X, then summarize, then suggest a logo idea."
```

---

## 11. Testing

```bash
python -m unittest discover -s tests
```

Covers capability normalization & routing, dependency-wave ordering (`ready_subtasks` /
`prereq_context`), aggregation (including the no-loss backstop), validator scoring (weighted
sub-scores and partial-failure penalty), and the graph structure. The tests mock all model calls,
so they run offline in well under a second.

---

## 12. Limitations & honest notes

- **Free models only.** Quality is bounded by the free tier; a single frontier model will often beat
  it on raw quality. The orchestrator's edge is **cost (≈€0), parallel speed, resilience, and
  structured multi-part answers** — measure those (see §10), not raw quality alone.
- **No real image generation.** Image requests yield a text description, not an image file.
- **Quality scores are LLM-judged**, so they are indicative, not absolute. The score is now grounded
  (three sub-scores × execution-success ratio) rather than a single gestalt number.
- **Rate limits.** On the free tier you may hit 429s; the orchestrator rotates keys/models and backs
  off, but heavy use still needs multiple keys.
```
