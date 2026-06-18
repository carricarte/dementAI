from __future__ import annotations

from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import Citation, GraphState
from backend.tools.retrieval import enrich_query, retrieve

_SYSTEM = load("screening")


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
                "medications": report.current_medications,
            },
        )
        patient_section = f"""Patient clinical summary:
{report.clinical_summary}

Cognitive scores — MMSE: {report.mmse}, MoCA: {report.moca}, CDR: {report.cdr}
Risk factors: {", ".join(report.risk_factors) or "none"}
Pending workups: {", ".join(report.pending_workups) or "none"}"""
    elif record:
        rq = enrich_query(
            state["query"],
            {
                "diagnosis": record.dementia_type.value if record.dementia_type else None,
                "mmse": record.screening_scores.mmse,
                "cdr": record.screening_scores.cdr,
                "medications": record.current_medications,
            },
        )
        scores = record.screening_scores.model_dump(exclude_none=True)
        patient_section = f"""Patient history:
{_format_history(state)}

Current screening scores: {scores or "none recorded"}
Risk flags: {record.risk_flags or "none"}
Pending workups: {record.pending_workups or "none"}"""
    else:
        rq = state["query"]
        patient_section = ""

    ctx = retrieve(rq, source_filter=["aan", "awmf", "alz"])

    prompt = f"""{_SYSTEM}

{patient_section}

Retrieved evidence:
{ctx["text"] or "(knowledge base not yet populated)"}

Physician query: {state["query"]}

Provide a structured clinical assessment with recommendations."""

    return prompt, ctx["citations"]


def run_screening(state: GraphState) -> GraphState:
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
