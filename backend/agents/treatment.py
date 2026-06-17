from __future__ import annotations

from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import GraphState
from backend.tools.calculators import donepezil_dose, memantine_dose
from backend.tools.retrieval import retrieve

_SYSTEM = load("treatment")


def run_treatment(state: GraphState) -> GraphState:
    record = state["patient_record"]
    ctx = retrieve(state["query"], source_filter=["pubmed", "clinicaltrials", "awmf"])

    disease = record.dementia_type.value if record.dementia_type else "undetermined"
    history = _format_history(state)

    # Pre-compute standard dosing guidance to inject into the prompt
    dose_context = ""
    if record.dementia_type and "alzheimer" in disease:
        dose_context = (
            f"Donepezil initiation: {donepezil_dose('initiation')}\n"
            f"Donepezil maintenance: {donepezil_dose('maintenance')}\n"
            f"Memantine initiation: {memantine_dose('initiation')}\n"
            f"Memantine maintenance: {memantine_dose('maintenance')}"
        )

    prompt = f"""{_SYSTEM}

Patient history:
{history}

Disease type: {disease}
Risk flags: {record.risk_flags or "none"}
Current medications: {record.current_medications or "none"}
Pending workups: {record.pending_workups or "none"}

{f"Standard dosing reference:{chr(10)}{dose_context}" if dose_context else ""}

Retrieved evidence:
{ctx["text"] or "(knowledge base not yet populated)"}

Physician query: {state["query"]}

Provide specific treatment recommendations with dosing where applicable."""

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
