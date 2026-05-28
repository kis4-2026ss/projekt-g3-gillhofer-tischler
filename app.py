"""
Streamlit GUI for the AI Meta-Orchestrator.

Run with:
    streamlit run app.py

Type a problem in the chat box at the bottom; the orchestrator decomposes it,
routes subtasks to free models, executes them (in dependency-ordered parallel
waves), and synthesizes a verified answer. You can keep refining or asking new
problems in the same session.
"""

import os
import re

import streamlit as st
from dotenv import load_dotenv

from src.orchestrator.graph import create_orchestrator
from src.orchestrator.nodes.registry import registry

load_dotenv()

st.set_page_config(page_title="AI Meta-Orchestrator", page_icon="🤖", layout="wide")

_IMG_RE = re.compile(r"\[LOCAL IMAGE SAVED TO: (.+?)\]")

_STATUS_META = {
    "queued":    ("⏳", "queued"),
    "running":   ("🔄", "running"),
    "completed": ("✅", "done"),
    "failed":    ("❌", "failed"),
}


@st.cache_resource(show_spinner=False)
def get_orchestrator():
    """Compile the LangGraph app once and reuse it across reruns."""
    return create_orchestrator()


def status_icon(status: str) -> str:
    return {"completed": "✅", "failed": "❌", "assigned": "⏳", "pending": "◻️"}.get(status, "•")


def show_images_in(text: str):
    """Render any locally-saved image files referenced in the text."""
    for path in _IMG_RE.findall(text or ""):
        path = path.strip()
        if os.path.exists(path):
            st.image(path, caption=os.path.basename(path))


# --- Live task board (shows parallel execution) ------------------------------

def _board_from_state(state: dict) -> dict:
    """Reconstruct a task board from a (final) state — used to replay history."""
    board = {}
    for tid, task in (state.get("subtasks") or {}).items():
        board[tid] = {
            "description": task.description,
            "model": (task.assigned_model or "").split("/")[-1] or "…",
            "capabilities": list(task.required_capabilities),
            "dependencies": list(task.dependencies),
            "status": task.status if task.status in ("completed", "failed") else "queued",
        }
    return board


def _merge_state(board: dict, state: dict):
    """Fold a full-state snapshot into the board without downgrading a live status."""
    for tid, task in (state.get("subtasks") or {}).items():
        b = board.setdefault(tid, {})
        b["description"] = task.description
        b["capabilities"] = list(task.required_capabilities)
        b["dependencies"] = list(task.dependencies)
        model = (task.assigned_model or "").split("/")[-1]
        if model:
            b["model"] = model
        if task.status in ("completed", "failed"):
            b["status"] = task.status
        elif b.get("status") not in ("running", "completed", "failed"):
            b["status"] = "queued"


def _apply_event(board: dict, event: dict):
    """Apply a live task_start / task_end event emitted by a worker."""
    tid = event.get("id")
    if not tid:
        return
    b = board.setdefault(tid, {})
    if event["type"] == "task_start":
        if event.get("description"):
            b["description"] = event["description"]
        if event.get("capabilities"):
            b["capabilities"] = event["capabilities"]
        if event.get("dependencies"):
            b["dependencies"] = event["dependencies"]
        model = (event.get("model") or "").split("/")[-1]
        if model:
            b["model"] = model
        b["status"] = "running"
    elif event["type"] == "task_end":
        b["status"] = event.get("status", "completed")


def _board_markdown(board: dict, live: bool) -> str:
    if not board:
        return "_Decomposing the request…_"
    lines = []
    running = sum(1 for b in board.values() if b.get("status") == "running")
    if live and running:
        lines.append(f"**⚡ {running} task(s) running in parallel**")
    for tid in sorted(board):
        b = board[tid]
        icon, label = _STATUS_META.get(b.get("status", "queued"), ("•", ""))
        deps = b.get("dependencies") or []
        dep_txt = f" · depends on {', '.join(deps)}" if deps else ""
        lines.append(
            f"{icon} **`{tid}`** — {b.get('description', '…')}  \n"
            f"&nbsp;&nbsp;&nbsp;↳ `{b.get('model', '…')}`{dep_txt} · _{label}_"
        )
    return "\n\n".join(lines)


def render_board(state: dict):
    """Static board for replaying a finished run from history."""
    st.markdown("**🧩 Task plan & status**")
    st.markdown(_board_markdown(_board_from_state(state), live=False))


def run_with_progress(prompt: str) -> dict:
    """Stream the run, showing a live board of subtasks executing in parallel."""
    initial_state = {
        "user_input": prompt,
        "subtasks": {},
        "final_output": None,
        "metadata": {},
        "retry_count": 0,
    }
    final_state = initial_state
    board: dict = {}

    with st.status("Orchestrating… (tasks run in parallel)", expanded=True) as status:
        board_ph = st.empty()
        try:
            for mode, data in get_orchestrator().stream(
                initial_state, stream_mode=["values", "custom"], config={"recursion_limit": 50}
            ):
                if mode == "values":
                    final_state = data
                    _merge_state(board, data)
                else:  # custom progress event from a worker thread
                    _apply_event(board, data)
                board_ph.markdown(_board_markdown(board, live=True))
            board_ph.markdown(_board_markdown(board, live=False))
            status.update(label="Done ✓", state="complete", expanded=True)
        except Exception as e:  # surface API/key/runtime errors instead of a blank page
            status.update(label="Failed", state="error")
            st.error(f"Orchestration failed: {e}")
    return final_state


# --- Result rendering --------------------------------------------------------

def render_result(state: dict):
    metadata = state.get("metadata") or {}
    subtasks = state.get("subtasks") or {}
    final_output = state.get("final_output")

    answer_col, side_col = st.columns([2, 1], gap="large")

    with answer_col:
        st.markdown("#### Final answer")
        if final_output:
            st.markdown(final_output)
            show_images_in(final_output)
        else:
            st.warning("No final answer was produced.")

    with side_col:
        validation = metadata.get("validation") or {}
        score = validation.get("score")
        m1, m2 = st.columns(2)
        m1.metric("Quality", f"{score}/100" if score is not None else "—")
        m2.metric("Complexity", f"{metadata.get('complexity', '—')}/10")

        if validation:
            if validation.get("is_valid"):
                st.success("Passed quality validation")
            else:
                st.warning("Did not pass validation")
            if validation.get("feedback"):
                st.caption(validation["feedback"])

        st.markdown(f"**Subtasks ({len(subtasks)}) — what each one does**")
        for task in subtasks.values():
            model = (task.assigned_model or "unassigned").split("/")[-1]
            short = task.description if len(task.description) <= 45 else task.description[:45] + "…"
            with st.expander(f"{status_icon(task.status)} `{task.id}` · {short}"):
                st.caption(f"Model: `{model}`")
                if task.required_capabilities:
                    st.caption("Capabilities: " + ", ".join(task.required_capabilities))
                if task.dependencies:
                    st.caption("Depends on: " + ", ".join(task.dependencies))
                if task.result:
                    st.markdown(task.result)

        routing = metadata.get("routing_decisions")
        if routing:
            with st.expander("Routing decisions"):
                for r in routing:
                    st.caption(f"`{r.get('task_id')}` → `{r.get('assigned')}`  ({', '.join(r.get('required', []))})")


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.title("🤖 AI Meta-Orchestrator")
    st.caption(
        "Decomposes your request, routes each subtask to the best free model, "
        "runs them in dependency-ordered parallel waves, then synthesizes and "
        "validates a final answer."
    )
    st.metric("Free models loaded", len(registry.models))
    st.divider()
    if st.button("🗑️ New conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()
    st.caption("Pipeline: Analyze → Route → Execute → Verify → Aggregate → Validate")


# --- Main: history replay + chat input --------------------------------------

st.session_state.setdefault("history", [])

if not st.session_state.history:
    st.info(
        "👋 Describe a problem below. Try something multi-part, e.g.\n\n"
        "- *Research the pros and cons of Rust vs Go for backends, then give a recommendation and a tagline.*\n"
        "- *Explain Dijkstra's algorithm, write a commented Python implementation, and analyze its complexity.*"
    )

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["problem"])
    with st.chat_message("assistant"):
        with st.expander("🧩 Task plan & parallel execution"):
            st.markdown(_board_markdown(_board_from_state(turn["state"]), live=False))
        render_result(turn["state"])

if prompt := st.chat_input("Describe your problem, or refine the previous one…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        final_state = run_with_progress(prompt)
        render_result(final_state)
    st.session_state.history.append({"problem": prompt, "state": final_state})
