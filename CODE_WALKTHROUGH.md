# Code Walkthrough — for Presentation

A guided tour of the codebase, in the order it makes sense to *present* it. Each section explains
**what the code does**, **why it's built that way**, and gives a **talking point** you can say out
loud. File paths are exact so you can open them live during the talk.

---

## 0. The 30-second pitch

> "We built a LangGraph state machine that breaks a complex request into subtasks, sends each to the
> best free AI model, runs the independent ones in parallel, then merges and quality-checks the
> result. It's like a project manager coordinating a team of specialist models."

**Tech stack:** Python · **LangGraph** (workflow state machine) · **LiteLLM** (one client for every
model) · **Pydantic** (typed data) · **Streamlit** (GUI).

---

## 1. The shared data model — `src/orchestrator/state.py`

Everything in the graph passes through one shared object, the **State**:

```python
class State(TypedDict):
    user_input: str
    subtasks: Annotated[Dict[str, SubTask], operator.ior]   # keyed by id, merged with |=
    final_output: Optional[str]
    metadata: Annotated[dict, operator.ior]
    retry_count: int
```

A `SubTask` is a Pydantic model: `id`, `description`, `expected_output`, `required_capabilities`,
`priority`, `dependencies`, `assigned_model`, `result`, `status`.

**Why a `Dict` with `Annotated[..., operator.ior]`?** This is the key to safe parallelism. When
several subtask workers finish *at the same time*, each returns `{"subtasks": {its_id: its_task}}`.
LangGraph merges those concurrent updates using `operator.ior` (dictionary `|=`), so updates combine
**by id** instead of overwriting each other. A plain list would race.

> **Talking point:** "The reducer on `subtasks` is what lets multiple workers write to the state
> concurrently without clobbering one another — each owns its own key."

---

## 2. The workflow graph — `src/orchestrator/graph.py`

This wires the nodes together into a LangGraph `StateGraph`:

```
START → analyzer → router → [dispatch] → executor(×N) → verifier → [dispatch] ─┐
                                  ▲                                            │
                                  └──────────── next wave ─────────────────────┘
                                                                               │
                                                                  (no more) → aggregator → validator → [should_continue]
                                                                                                         ├─ valid / out of retries → END
                                                                                                         └─ else → analyzer (retry)
```

Two conditional edges do the clever parts:

**(a) `dispatch_subtasks` — parallel fan-out in dependency waves.**

```python
def dispatch_subtasks(state):
    ready = ready_subtasks(state["subtasks"])      # see executor.py
    if not ready:
        return "aggregator"                         # nothing left → move on
    return [Send("executor", {"task": t,
                              "metadata": state.get("metadata", {}),
                              "context": prereq_context(t, state["subtasks"])})
            for t in ready]
```

`Send(...)` is LangGraph's map-reduce primitive: returning a **list of `Send`s** spawns one
`executor` invocation per ready subtask, and they run **concurrently**. After the wave fans back in
to the verifier, `dispatch_subtasks` runs again for the next wave — so a chain `t1 → t2 → t3` runs
as three sequential waves, while independent tasks share a wave.

**(b) `should_continue` — the quality retry loop.**

```python
def should_continue(state):
    v = state.get("metadata", {}).get("validation", {})
    if v.get("is_valid", True) or state.get("retry_count", 0) >= 2:
        return END
    return "analyzer"          # re-decompose and try again (max 2 retries)
```

> **Talking point:** "Parallelism and dependency-ordering both live in one function,
> `dispatch_subtasks`. It only ever dispatches subtasks whose prerequisites are done, and it hands
> each one its prerequisites' output as context."

---

## 3. Analyzer — `src/orchestrator/nodes/analyzer.py`

Turns the request into subtasks. The prompt enforces strict decomposition rules: **atomic,
non-overlapping, minimal, one capability each, dependency-driven** (independent tasks get empty
`dependencies` so they parallelize), plus a worked example. It returns a JSON object that becomes a
`Dict[str, SubTask]`.

Two details worth pointing out:
- **Retry feedback:** on a retry (`retry_count > 0`) it injects the validator's previous feedback so
  the second decomposition fixes what failed, instead of repeating it.
- **Robust parsing:** it strips ```` ```json ```` fences and trims stray text before `json.loads`,
  with a single-subtask fallback if parsing fails.

> **Talking point:** "The decomposition quality *is* the prompt — we encode the rules of a good
> split (atomic, non-overlapping, dependency-aware) directly into it."

---

## 4. Router + Registry — `nodes/router.py`, `nodes/registry.py`

The router normalizes each subtask's capabilities and asks the registry for the best model:

```python
normalized = normalize_capabilities(task.required_capabilities)   # 'image_generation' → 'image', etc.
best = registry.get_best_model(normalized, exclude_models=dead_models)
```

The **Capability Registry** loads ~27 free models from `free_models.json`, each tagged with
canonical capabilities (`general, coding, reasoning, research, summarization, creative, image`) and a
latency score. `get_best_model` scores candidates by capability match + latency, prefers
specialists on ties, and skips known-dead models.

> **Talking point:** "Routing is just scoring: capability fit first, then speed. Swapping models is a
> JSON edit, not a code change."

---

## 5. Executor — `src/orchestrator/nodes/executor.py`

The heart of the parallel engine. Three pieces:

**(a) `ready_subtasks(subtasks)`** — which subtasks can run *now*:

```python
def ready_subtasks(subtasks):
    assigned = [t for t in subtasks.values() if t.status == "assigned"]
    ready = [t for t in assigned
             if all(subtasks[d].status in ("completed", "failed")
                    for d in t.dependencies if d in subtasks)]
    return ready or assigned     # if nothing is "ready", a cycle exists → run all (never deadlock)
```

A failed prerequisite still counts as "terminal", so a dependent task isn't stranded; and the
`or assigned` fallback guarantees the graph can't hang on a dependency cycle.

**(b) `prereq_context(task, subtasks)`** — gathers completed prerequisites' results so the dependent
task can build on them (this is what makes the decomposition *real* rather than cosmetic).

**(c) `subtask_worker(input_data)`** — runs one subtask:
- builds the prompt (prepending prerequisite context),
- **image tasks** get a dedicated prompt that *writes a vivid description directly* and never asks
  the user or claims it "can't make images",
- calls the model, and on **rate-limit (429)** or **dead model (404)** rotates to the next best
  model (up to `MAX_MODELS_PER_TASK`),
- emits **live `task_start` / `task_end` events** via `get_stream_writer()` so the GUI can show tasks
  running in parallel,
- returns `{"subtasks": {id: task}}` for the dict reducer to merge.

> **Talking point:** "Because each worker only writes its own subtask id and rotation is per-task,
> the workers are fully independent — that's why we can run them on parallel threads safely."

---

## 6. Verifier — `src/orchestrator/nodes/verifier.py`

A lightweight quality gate: for any completed subtask that produced code (capability `coding` or a
```` ``` ```` block), it runs a "senior engineer" review for syntax/logic/security issues and
appends the note to the result. Non-code subtasks pass straight through.

---

## 7. Aggregator — `src/orchestrator/nodes/aggregator.py`

Merges all completed results into one answer. The prompt's #1 rule is **"include the full content of
every subtask — never drop a deliverable"**, organized under headings.

Because weak free models still sometimes drop content, there's a **deterministic safety net**:

```python
def _coverage(result, output):                    # fraction of a result's distinctive words present
    sig = _signature_words(result)
    return len(sig & _signature_words(output)) / len(sig) if sig else 1.0

missing = [t for t in completed if _coverage(t.result, final_output) < 0.25]
# any dropped deliverable is re-appended verbatim under "Additional task results"
```

> **Talking point:** "We don't *trust* the synthesis model to keep everything — we *verify* it. If a
> deliverable's words are largely absent from the merged answer, we re-attach it. Nothing is lost."

---

## 8. Validator — `src/orchestrator/nodes/validator.py`

Scores the final answer. Rather than asking for one gut-feel number (which clustered at ~95), it asks
for **three independent 0–100 sub-scores** and combines them in code, then scales by how many
subtasks actually succeeded:

```python
base = round(0.30*consistency + 0.40*correctness + 0.30*completeness)   # correctness weighted highest
score = round(base * (len(completed) / total))                          # partial runs score lower
is_valid = score >= 70
```

The validator also sees **snippets of each subtask's result**, and its completeness rule says: *if a
produced deliverable is missing from the final answer, completeness must be low* — so the score
reflects reality, not just fluency.

> **Talking point:** "The score is grounded in three dimensions and in execution success — a run that
> only finished two of three tasks can't score as if everything worked."

---

## 9. Infrastructure — `src/orchestrator/utils.py`

- **Multi-key rotation:** `get_api_keys()` collects `OPENROUTER_API_KEY`, `OPENROUTER_API_KEY_2`, …
- **`call_with_retry(kwargs)`** wraps every model call with exponential backoff and, on a 429,
  rotates to the next API key — all guarded by a `threading.Lock` because workers run on parallel
  threads. The key is passed per-call (`kwargs["api_key"]`), never via a shared env var, to stay
  thread-safe.
- **Usage tracking:** `reset_usage()` / `get_usage()` accumulate token counts across a run (used by
  the comparison scripts).

> **Talking point:** "This is the one place real concurrency bites — a shared key index — so it's the
> one place we take a lock."

---

## 10. The GUI — `app.py`

Streamlit app with three ideas:

1. **Continuous conversation:** a `st.chat_input` plus session history — you can refine or ask new
   problems without restarting.
2. **Live parallel view:** it streams the run with `stream_mode=["values", "custom"]`. The `values`
   stream gives full-state snapshots; the `custom` stream carries the `task_start`/`task_end` events
   from the workers. A live "task board" shows each subtask's role (e.g. `t1 · research books`) and
   flips multiple tasks to **🔄 running** at once — *visible* proof of parallelism.
3. **Structured result:** the final answer (markdown) on the left, and a side panel with the quality
   score, validation badge, and an expander per subtask (what each one did + its output).

> **Talking point:** "The board isn't a fake animation — those `running` badges come from real events
> emitted by the worker threads as they start, so you're watching genuine parallel execution."

---

## 11. Validation tooling — `scripts/`

- **`compare_single_model.py`** — orchestrator vs **one free model**, same input, reporting success,
  **latency**, and a blind quality score for each. This is the "does our approach actually help, and
  what does it cost in time?" experiment (apples-to-apples on free models).
- **`benchmark.py`** — same, but vs a **paid baseline** (e.g. GPT-4) for the quality-per-cost story.

Both judge answers **blind** (the judge never knows which system produced an answer) and save a JSON
record to `benchmark/results/`.

---

## 12. Suggested live demo (≈3 minutes)

1. `python -m streamlit run app.py`
2. Enter a deliberately multi-part prompt, e.g.
   *"Research the pros and cons of Rust vs Go for a backend, write a short recommendation, and suggest a project tagline."*
3. Point at the **task board**: independent subtasks turn **🔄 running** simultaneously; a dependent
   one waits, then runs — *that's the parallelism + dependency handling*.
4. Show the **final answer** containing every deliverable, and the **quality score** + per-subtask
   panel.
5. (Optional) In a terminal: `python scripts/compare_single_model.py "<same prompt>"` to show the
   orchestrator vs a single free model on quality and speed.

---

## 13. Likely questions (prep)

- **"Is it really parallel, or just async-looking?"** Yes — LangGraph runs the `Send` fan-out on a
  thread pool; we verified independent subtasks start before any finishes, and the key-rotation lock
  exists precisely because of that concurrency.
- **"What if a model fails or is rate-limited?"** Per-task model rotation (404/429) plus multi-key
  rotation with backoff; a failed subtask doesn't block its dependents.
- **"How do you handle dependencies?"** `dispatch_subtasks` + `ready_subtasks` schedule waves; a
  dependent task only starts once prerequisites are terminal and receives their output as context.
- **"Why is the score not always 95 anymore?"** It's three sub-scores combined and scaled by
  execution success, and the validator sees the actual subtask outputs — so incomplete answers score
  lower.
- **"Why free models?"** Cost ≈ €0; the project's thesis is that *coordination* of cheap specialists
  can rival a single bigger model on multi-part tasks — which the comparison scripts let us measure.
- **"Biggest limitation?"** Free-tier quality and rate limits; no real image generation (we return a
  description).
```
