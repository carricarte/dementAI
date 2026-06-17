from __future__ import annotations

from langgraph.graph import END, StateGraph

from backend.audit.logger import audit_logger
from backend.llm import get_llm
from backend.prompts import load
from backend.state.schema import ClinicalStage, GraphState, PatientRecord, VisitRecord
from backend.state.store import patient_store

from .care import run_care
from .diagnosis import run_diagnosis
from .prevention import run_prevention
from .screening import run_screening
from .treatment import run_treatment

_CLASSIFY_PROMPT = load("coordinator_classify")


def classify_stage(state: GraphState) -> GraphState:
    response = get_llm().invoke(_CLASSIFY_PROMPT.format(query=state["query"]))
    raw = response.content.strip().lower().split()[0]
    try:
        stage = ClinicalStage(raw)
    except ValueError:
        stage = ClinicalStage.SCREENING
    return {**state, "stage": stage}


def load_patient(state: GraphState) -> GraphState:
    record = patient_store.load_or_create(state["patient_id"])
    return {**state, "patient_record": record}


def route_to_specialist(state: GraphState) -> str:
    return state["stage"].value


def merge_output(state: GraphState) -> GraphState:
    # Single-specialist path: final_response == specialist_response.
    # Extend here to merge multiple specialists if needed.
    return {**state, "final_response": state["specialist_response"]}


def save_state(state: GraphState) -> GraphState:
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


def build_graph():
    g = StateGraph(GraphState)

    g.add_node("classify_stage", classify_stage)
    g.add_node("load_patient", load_patient)
    g.add_node(ClinicalStage.SCREENING.value, run_screening)
    g.add_node(ClinicalStage.DIAGNOSIS.value, run_diagnosis)
    g.add_node(ClinicalStage.PREVENTION.value, run_prevention)
    g.add_node(ClinicalStage.TREATMENT.value, run_treatment)
    g.add_node(ClinicalStage.CARE.value, run_care)
    g.add_node("merge_output", merge_output)
    g.add_node("save_state", save_state)
    g.add_node("audit_log", audit_log)

    g.set_entry_point("classify_stage")
    g.add_edge("classify_stage", "load_patient")
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
