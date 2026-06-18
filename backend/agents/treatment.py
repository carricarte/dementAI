from __future__ import annotations

from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import Citation, GraphState
from backend.tools.calculators import donepezil_dose, memantine_dose
from backend.tools.retrieval import enrich_query, retrieve

_SYSTEM = load("treatment")


def prepare(state: GraphState) -> tuple[str, list[Citation]]:
    """Returns (prompt, citations) without invoking the LLM."""
    report = state.get("patient_status_report")
    record = state.get("patient_record")

    if report:
        rq = enrich_query(
            state["query"],
            {
                "diagnosis": report.primary_diagnosis,
                "stage": report.diagnosis_stage,
                "mmse": report.mmse,
                "cdr": report.cdr,
                "apoe": report.apoe_status,
                "medications": report.current_medications,
            },
        )
        disease = report.primary_diagnosis or "undetermined"
        patient_section = f"""Patient clinical summary:
{report.clinical_summary}

Disease type: {disease}
Diagnosis stage: {report.diagnosis_stage or "unknown"}
Treatment priorities: {", ".join(report.treatment_priorities) or "none"}
Current medications: {", ".join(report.current_medications) or "none"}
Risk factors: {", ".join(report.risk_factors) or "none"}
Pending workups: {", ".join(report.pending_workups) or "none"}"""
        is_alzheimers = "alzheimer" in disease.lower()
    elif record:
        rq = enrich_query(
            state["query"],
            {
                "diagnosis": record.dementia_type.value if record.dementia_type else None,
                "mmse": record.screening_scores.mmse,
                "cdr": record.screening_scores.cdr,
                "apoe": record.risk_flags[0] if record.risk_flags else None,
                "medications": record.current_medications,
            },
        )
        disease = record.dementia_type.value if record.dementia_type else "undetermined"
        patient_section = f"""Patient history:
{_format_history(state)}

Disease type: {disease}
Risk flags: {record.risk_flags or "none"}
Current medications: {record.current_medications or "none"}
Pending workups: {record.pending_workups or "none"}"""
        is_alzheimers = record.dementia_type is not None and "alzheimer" in disease
    else:
        rq = state["query"]
        patient_section = ""
        is_alzheimers = False

    dose_context = ""
    if is_alzheimers:
        dose_context = (
            f"Donepezil initiation: {donepezil_dose('initiation')}\n"
            f"Donepezil maintenance: {donepezil_dose('maintenance')}\n"
            f"Memantine initiation: {memantine_dose('initiation')}\n"
            f"Memantine maintenance: {memantine_dose('maintenance')}"
        )

    ctx = retrieve(rq, source_filter=["pubmed", "clinicaltrials", "awmf"])

    prompt = f"""{_SYSTEM}

{patient_section}

{f"Standard dosing reference:{chr(10)}{dose_context}" if dose_context else ""}

Retrieved evidence:
{ctx["text"] or "(knowledge base not yet populated)"}

Physician query: {state["query"]}

Provide specific treatment recommendations with dosing where applicable."""

    return prompt, ctx["citations"]


def run_treatment(state: GraphState) -> GraphState:
    prompt, citations = prepare(state)
    response = get_llm().invoke(prompt)
    return {**state, "specialist_response": response.content, "citations": citations}


def _format_history(state: GraphState) -> str:
    record = state.get("patient_record")
    if not record or not record.visits:
        return "No prior visits."
    return "\n".join(
        f"[{v.timestamp.date()} | {v.stage.value}] {v.query[:120]}" for v in record.visits[-5:]
    )
