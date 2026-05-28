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
_STAGE_ORDER = ["Analyze", "Route", "Execute", "Aggregate", "Validate"]


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


# --- Live progress -----------------------------------------------------------

def _new_progress_lines(state: dict, seen: set):
    """Yield human-readable lines for milestones reached since the last snapshot."""
    subtasks = state.get("subtasks") or {}
    metadata = state.get("metadata") or {}

    if subtasks and "analyzed" not in seen:
        seen.add("analyzed")
        intent = metadata.get("intent", "")
        yield f"🧩 Decomposed into **{len(subtasks)}** subtask(s)" + (f" — _{intent}_" if intent else "")

    if subtasks and "routed" not in seen and all(t.assigned_model for t in subtasks.values()):
        seen.add("routed")
        yield "🧭 Routed each subtask to its best-fit model"

    for task in subtasks.values():
        if task.status in ("completed", "failed"):
            key = f"task:{task.id}:{task.status}"
            if key not in seen:
                seen.add(key)
                model = (task.assigned_model or "?").split("/")[-1]
                yield f"{status_icon(task.status)} `{task.id}` → `{model}`"

    if state.get("final_output") and "aggregated" not in seen:
        seen.add("aggregated")
        yield "🧵 Synthesized the final answer"

    validation = metadata.get("validation")
    if validation and "validated" not in seen:
        seen.add("validated")
        yield f"🏁 Quality score **{validation.get('score', 0)}/100**"


def run_with_progress(prompt: str) -> dict:
    """Stream the orchestrator run, showing progress live, and return the final state."""
    initial_state = {
        "user_input": prompt,
        "subtasks": {},
        "final_output": None,
        "metadata": {},
        "retry_count": 0,
    }
    final_state = initial_state
    seen: set = set()

    with st.status("Orchestrating…", expanded=True) as status:
        try:
            for snapshot in get_orchestrator().stream(
                initial_state, stream_mode="values", config={"recursion_limit": 50}
            ):
                final_state = snapshot
                for line in _new_progress_lines(snapshot, seen):
                    st.write(line)
            status.update(label="Done ✓", state="complete", expanded=False)
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

        st.markdown(f"**Subtasks ({len(subtasks)})**")
        for task in subtasks.values():
            model = (task.assigned_model or "unassigned").split("/")[-1]
            with st.expander(f"{status_icon(task.status)} `{task.id}` · {model}"):
                st.caption(task.description)
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
        render_result(turn["state"])

if prompt := st.chat_input("Describe your problem, or refine the previous one…"):
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        final_state = run_with_progress(prompt)
        render_result(final_state)
    st.session_state.history.append({"problem": prompt, "state": final_state})
