from __future__ import annotations

from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import Citation, GraphState
from backend.tools.retrieval import enrich_query, retrieve

_SYSTEM = load("care")


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

Disease type: {report.primary_diagnosis or "undetermined"}
Diagnosis stage: {report.diagnosis_stage or "unknown"}
CDR: {report.cdr if report.cdr is not None else "not assessed"}
Current medications: {", ".join(report.current_medications) or "none"}
Key findings: {", ".join(report.key_findings) or "none"}"""
    elif record:
        rq = enrich_query(
            state["query"],
            {
                "diagnosis": record.dementia_type.value if record.dementia_type else None,
                "stage": record.screening_scores.cdr,
                "mmse": record.screening_scores.mmse,
                "cdr": record.screening_scores.cdr,
                "medications": record.current_medications,
            },
        )
        cdr = record.screening_scores.cdr
        dementia_type = record.dementia_type.value if record.dementia_type else "undetermined"
        patient_section = f"""Patient history:
{_format_history(state)}

Disease type: {dementia_type}
CDR (disease severity): {cdr if cdr is not None else "not assessed"}
Risk flags: {record.risk_flags or "none"}
Current medications: {record.current_medications or "none"}"""
    else:
        rq = state["query"]
        patient_section = ""

    ctx = retrieve(rq, source_filter=["alz", "aan", "awmf", "pubmed"])

    prompt = f"""{_SYSTEM}

{patient_section}

Retrieved evidence:
{ctx["text"] or "No relevant sources found in the knowledge base."}

Physician query: {state["query"]}

Answer the physician's query above. Only include what directly addresses what was asked."""

    return prompt, ctx["citations"]


def run_care(state: GraphState) -> GraphState:
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
