from __future__ import annotations

from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import Citation, GraphState
from backend.tools.retrieval import enrich_query, retrieve

_SYSTEM = load("diagnosis")


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
        patient_section = f"""Patient clinical summary:
{report.clinical_summary}

Primary diagnosis: {report.primary_diagnosis or "undetermined"}
Diagnosis stage: {report.diagnosis_stage or "unknown"}
Key findings: {", ".join(report.key_findings) or "none"}
MRI findings: {report.mri_findings or "none"}
APOE status: {report.apoe_status or "not tested"}
Completed workups: {", ".join(report.completed_workups) or "none"}
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
        differentials = [d.value for d in record.differential_diagnoses]
        patient_section = f"""Patient history:
{_format_history(state)}

Current diagnosis: {record.dementia_type.value if record.dementia_type else "undetermined"}
Differential diagnoses under consideration: {differentials or "none established"}
Screening scores: {record.screening_scores.model_dump(exclude_none=True) or "not available"}
Completed workups: {record.completed_workups or "none"}
Pending workups: {record.pending_workups or "none"}"""
    else:
        rq = state["query"]
        patient_section = ""

    ctx = retrieve(rq, source_filter=["pubmed", "neurology", "awmf"])

    prompt = f"""{_SYSTEM}

{patient_section}

Retrieved evidence:
{ctx["text"] or "No relevant sources found in the knowledge base."}

Physician query: {state["query"]}

Answer the physician's query above. Only include what directly addresses what was asked."""

    return prompt, ctx["citations"]


def run_diagnosis(state: GraphState) -> GraphState:
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
