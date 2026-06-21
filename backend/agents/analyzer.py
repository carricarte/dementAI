"""Patient Status Analysis Agent.

Reads structured patient data directly from the PatientStore and the NACC UDS
research dataset, then uses an LLM to produce a structured PatientStatusReport.

Invoked only on the patient_specific path; its output is never written into or
persisted within the knowledge base.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from backend.llm import get_llm
from backend.state.schema import GraphState, PatientStatusReport
from backend.state.store import PatientStore

_store = PatientStore()

# ---------------------------------------------------------------------------
# Data readers
# ---------------------------------------------------------------------------


def _read_clinical_record(patient_id: str) -> dict:
    record = _store.load(patient_id)
    if record is None:
        return {"error": f"No clinical record for patient_id={patient_id!r}"}
    return json.loads(record.model_dump_json())


def _read_research_data(patient_id: str) -> dict:
    data: dict[str, dict] = {}
    for sheet, rel_path in [
        ("nacc_uds", "data/synthetic/nacc_uds.csv"),
        ("scan_mri", "data/synthetic/scan_mri.csv"),
        ("genetics", "data/synthetic/genetics.csv"),
    ]:
        path = Path(rel_path)
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("patient_uuid") == patient_id:
                    data[sheet] = dict(row)
                    break
    return data


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are the Patient Status Analysis Agent for a clinical decision support system.

You receive structured patient data (clinical record + NACC UDS research dataset)
and must produce a structured clinical status report as a single JSON object —
no markdown, no explanation, pure JSON.

Required fields:
{
  "patient_id": string,
  "primary_diagnosis": string | null,
  "diagnosis_stage": string | null,        // "normal" | "MCI" | "mild" | "moderate" | "severe"
  "mmse": integer | null,
  "moca": integer | null,
  "cdr": number | null,
  "apoe_status": string | null,            // e.g. "ε3/ε4 (1 ε4 allele)"
  "risk_factors": [string],
  "current_medications": [string],
  "completed_workups": [string],
  "pending_workups": [string],
  "mri_findings": string | null,           // key imaging findings (atrophy, WMH, MTA scores)
  "clinical_summary": string,             // 3–5 sentence narrative of current status
  "key_findings": [string],               // concise bullet-point findings
  "treatment_priorities": [string]        // ranked clinical priorities for the plan
}

Rules:
- Extract values directly from the provided data; do not invent or assume missing fields.
- If a field is absent from the data, set it to null or an empty list as appropriate.
- clinical_summary must integrate cognitive, functional, imaging, and genetic findings.
- treatment_priorities should reflect the most actionable clinical needs.
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:])
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


# ---------------------------------------------------------------------------
# Agent node
# ---------------------------------------------------------------------------


def run_analyzer_agent(state: GraphState) -> GraphState:
    """Analyzer Agent node.

    Reads structured patient data, then uses an LLM to produce a
    PatientStatusReport stored in ``state["patient_status_report"]``.
    """
    patient_id = state["patient_id"]

    patient_data = json.dumps(
        {
            "clinical_record": _read_clinical_record(patient_id),
            "research_data": _read_research_data(patient_id),
        },
        indent=2,
    )

    llm = get_llm()
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM),
            HumanMessage(content=f"Patient ID: {patient_id}\n\nData:\n\n{patient_data}"),
        ]
    )

    text = _strip_fences(response.content)
    try:
        data = json.loads(text)
        data["patient_id"] = patient_id
        report = PatientStatusReport(**data)
    except Exception:
        report = PatientStatusReport(
            patient_id=patient_id,
            clinical_summary=text[:1000],
        )

    return {**state, "patient_status_report": report}
