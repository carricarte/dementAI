from __future__ import annotations

import json
import re
from typing import Iterator

from langgraph.graph import END, StateGraph

from backend.agents.analyzer import run_analyzer_agent
from backend.audit.logger import audit_logger
from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import Citation, ClinicalStage, GraphState, PatientRecord, VisitRecord
from backend.state.store import patient_store

from . import care as _care
from . import diagnosis as _diagnosis
from . import prevention as _prevention
from . import screening as _screening
from . import treatment as _treatment
from .care import run_care
from .diagnosis import run_diagnosis
from .prevention import run_prevention
from .screening import run_screening
from .treatment import run_treatment

_CLASSIFY_PROMPT = load("coordinator_classify")

_OFF_TOPIC_RESPONSE = (
    "## Outside Scope\n"
    "This system answers questions about dementia, cognitive decline, and neurodegeneration only. "
    "Please consult a general clinical reference for this query."
)


def _renumber_citations(response: str, citations: list[Citation]) -> tuple[str, list[Citation]]:
    """Renumber [N] markers so they follow order of first appearance in the text."""
    order: list[int] = []
    seen: set[int] = set()
    for m in re.finditer(r"\[(\d+)\]", response):
        n = int(m.group(1))
        if n not in seen and 1 <= n <= len(citations):
            seen.add(n)
            order.append(n)
    remap = {old: new for new, old in enumerate(order, 1)}
    new_response = re.sub(
        r"\[(\d+)\]", lambda m: f"[{remap.get(int(m.group(1)), m.group(1))}]", response
    )
    new_citations = [citations[n - 1] for n in order]
    return new_response, new_citations


# ---------------------------------------------------------------------------
# Shared graph nodes
# ---------------------------------------------------------------------------


def refuse_off_topic(state: GraphState) -> GraphState:
    return {
        **state,
        "stage": ClinicalStage.SCREENING,  # sentinel for audit log
        "specialist_response": _OFF_TOPIC_RESPONSE,
        "citations": [],
        "final_response": _OFF_TOPIC_RESPONSE,
    }


def classify_stage(state: GraphState) -> GraphState:
    """Classify query into a clinical stage, or mark off_topic if out of scope."""
    response = get_llm().invoke(_CLASSIFY_PROMPT.format(query=state["query"]))
    raw = response.content.strip().lower().split()[0]
    if raw == "off_topic":
        return {**state, "is_on_topic": False, "stage": None}
    try:
        stage = ClinicalStage(raw)
    except ValueError:
        stage = ClinicalStage.SCREENING
    return {**state, "is_on_topic": True, "stage": stage}


def route_after_classify(state: GraphState) -> str:
    return "classify_intent" if state.get("is_on_topic", True) else "refuse_off_topic"


def classify_intent(state: GraphState) -> GraphState:
    """Route to patient-specific path when patient_id is provided, general otherwise."""
    intent = "patient_specific" if state.get("patient_id") else "general"
    return {**state, "query_intent": intent}


def route_pipeline(state: GraphState) -> str:
    if state.get("query_intent") == "patient_specific":
        return "analyzer_agent"
    return "load_patient"


def load_patient(state: GraphState) -> GraphState:
    if not state.get("patient_id"):
        return state
    record = patient_store.load_or_create(state["patient_id"])
    return {**state, "patient_record": record}


def route_to_specialist(state: GraphState) -> str:
    return state["stage"].value


def merge_output(state: GraphState) -> GraphState:
    response, citations = _renumber_citations(
        state["specialist_response"] or "", state["citations"]
    )
    return {
        **state,
        "specialist_response": response,
        "citations": citations,
        "final_response": response,
    }


def save_state(state: GraphState) -> GraphState:
    if not state.get("patient_id") or state.get("patient_record") is None:
        return state
    record: PatientRecord = state["patient_record"]
    record.visits.append(
        VisitRecord(
            stage=state["stage"],
            query=state["query"],
            specialist_response=state["specialist_response"] or "",
            citations=state["citations"],
        )
    )
    patient_store.save(record)
    return state


def audit_log(state: GraphState) -> GraphState:
    audit_logger.log(state)
    return state


# ---------------------------------------------------------------------------
# Streaming helpers
# ---------------------------------------------------------------------------

_PREPARE_MAP = {
    ClinicalStage.SCREENING: lambda s: _screening.prepare(s),
    ClinicalStage.DIAGNOSIS: lambda s: _diagnosis.prepare(s),
    ClinicalStage.PREVENTION: lambda s: _prevention.prepare(s),
    ClinicalStage.TREATMENT: lambda s: _treatment.prepare(s),
    ClinicalStage.CARE: lambda s: _care.prepare(s),
}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _stream_response(state: GraphState, personalized: bool) -> Iterator[str]:
    """Shared streaming logic for both paths after state is fully prepared."""
    try:
        prompt, citations = _PREPARE_MAP[state["stage"]](state)
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    full_response = ""
    try:
        for chunk in get_llm().stream(prompt):
            text = chunk.content
            if text:
                full_response += text
                yield _sse({"type": "chunk", "text": text})
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    renumbered_response, renumbered_citations = _renumber_citations(full_response, citations)

    state = {
        **state,
        "specialist_response": renumbered_response,
        "citations": renumbered_citations,
        "final_response": renumbered_response,
    }
    try:
        save_state(state)
        audit_log(state)
    except Exception:
        pass

    yield _sse(
        {
            "type": "done",
            "response": renumbered_response,
            "citations": [c.model_dump() for c in renumbered_citations],
            "personalized": personalized,
        }
    )


# ---------------------------------------------------------------------------
# Public streaming entry point
# ---------------------------------------------------------------------------


def stream_query(patient_id: str, query: str) -> Iterator[str]:
    """Yield SSE-formatted lines for a streaming query response."""
    state: GraphState = {
        "patient_id": patient_id,
        "query": query,
        "is_on_topic": True,
        "stage": None,
        "query_intent": None,
        "patient_record": None,
        "patient_status_report": None,
        "specialist_response": None,
        "citations": [],
        "final_response": None,
    }

    try:
        state = classify_stage(state)
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})
        return

    if not state.get("is_on_topic", True):
        yield _sse({"type": "stage", "stage": "off_topic"})
        yield _sse(
            {
                "type": "done",
                "response": _OFF_TOPIC_RESPONSE,
                "citations": [],
                "personalized": False,
            }
        )
        return

    yield _sse({"type": "stage", "stage": state["stage"].value})

    state = classify_intent(state)

    if state["query_intent"] == "patient_specific":
        try:
            state = run_analyzer_agent(state)
        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
            return
        if state.get("patient_status_report") is None:
            yield _sse({"type": "error", "message": "Analyzer produced no report."})
            return
        state = load_patient(state)
        yield from _stream_response(state, personalized=True)
    else:
        state = load_patient(state)
        yield from _stream_response(state, personalized=False)


# ---------------------------------------------------------------------------
# LangGraph (non-streaming endpoint)
# ---------------------------------------------------------------------------


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("classify_stage", classify_stage)
    g.add_node("classify_intent", classify_intent)
    g.add_node("refuse_off_topic", refuse_off_topic)

    # Both paths share load_patient → specialist routing
    g.add_node("load_patient", load_patient)
    g.add_node(ClinicalStage.SCREENING.value, run_screening)
    g.add_node(ClinicalStage.DIAGNOSIS.value, run_diagnosis)
    g.add_node(ClinicalStage.PREVENTION.value, run_prevention)
    g.add_node(ClinicalStage.TREATMENT.value, run_treatment)
    g.add_node(ClinicalStage.CARE.value, run_care)

    # Patient-specific path: analyzer feeds into load_patient → specialist
    g.add_node("analyzer_agent", run_analyzer_agent)

    g.add_node("merge_output", merge_output)
    g.add_node("save_state", save_state)
    g.add_node("audit_log", audit_log)

    g.set_entry_point("classify_stage")
    g.add_conditional_edges(
        "classify_stage",
        route_after_classify,
        {"classify_intent": "classify_intent", "refuse_off_topic": "refuse_off_topic"},
    )
    # Off-topic path skips specialist and merge; goes straight to audit
    g.add_edge("refuse_off_topic", "audit_log")
    g.add_conditional_edges(
        "classify_intent",
        route_pipeline,
        {"load_patient": "load_patient", "analyzer_agent": "analyzer_agent"},
    )

    # analyzer_agent rejoins the general path at load_patient
    g.add_edge("analyzer_agent", "load_patient")

    # load_patient routes to the stage specialist
    g.add_conditional_edges(
        "load_patient",
        route_to_specialist,
        {s.value: s.value for s in ClinicalStage},
    )
    for stage in ClinicalStage:
        g.add_edge(stage.value, "merge_output")

    g.add_edge("merge_output", "save_state")
    g.add_edge("save_state", "audit_log")
    g.add_edge("audit_log", END)

    return g.compile()


coordinator = build_graph()
