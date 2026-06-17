from __future__ import annotations

from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import GraphState
from backend.tools.retrieval import retrieve

_SYSTEM = load("care")


def run_care(state: GraphState) -> GraphState:
    record = state["patient_record"]
    ctx = retrieve(state["query"], source_filter=["alz", "aan"])

    cdr = record.screening_scores.cdr
    history = _format_history(state)

    prompt = f"""{_SYSTEM}

Patient history:
{history}

Disease type: {record.dementia_type.value if record.dementia_type else "undetermined"}
CDR (disease severity): {cdr if cdr is not None else "not assessed"}
Risk flags: {record.risk_flags or "none"}
Current medications: {record.current_medications or "none"}

Retrieved evidence:
{ctx["text"] or "(knowledge base not yet populated)"}

Physician query: {state["query"]}

Provide practical care recommendations addressing both patient and caregiver needs."""

    response = get_llm().invoke(prompt)
    return {**state, "specialist_response": response.content, "citations": ctx["citations"]}


def _format_history(state: GraphState) -> str:
    visits = state["patient_record"].visits
    if not visits:
        return "No prior visits."
    return "\n".join(
        f"[{v.timestamp.date()} | {v.stage.value}] {v.query[:120]}"
        for v in visits[-5:]
    )
