from unittest.mock import MagicMock, patch

import pytest

from backend.state.schema import ClinicalStage, GraphState


@pytest.fixture
def base_state() -> GraphState:
    return {
        "patient_id": "test-001",
        "query": "",
        "stage": None,
        "patient_record": None,
        "specialist_response": None,
        "citations": [],
        "final_response": None,
    }


def test_classify_stage_screening(base_state):
    from backend.agents.coordinator import classify_stage

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="screening")
    with patch("backend.agents.coordinator.get_llm", return_value=mock_llm):
        result = classify_stage({**base_state, "query": "MoCA score is 21, what does this mean?"})
    assert result["stage"] == ClinicalStage.SCREENING


def test_classify_stage_diagnosis(base_state):
    from backend.agents.coordinator import classify_stage

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="diagnosis")
    with patch("backend.agents.coordinator.get_llm", return_value=mock_llm):
        result = classify_stage({**base_state, "query": "Asymmetric cortical atrophy, tau PET positive"})
    assert result["stage"] == ClinicalStage.DIAGNOSIS


def test_classify_stage_falls_back_on_unknown(base_state):
    from backend.agents.coordinator import classify_stage

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="gibberish")
    with patch("backend.agents.coordinator.get_llm", return_value=mock_llm):
        result = classify_stage({**base_state, "query": "something unclear"})
    assert result["stage"] == ClinicalStage.SCREENING


def test_patient_state_persists_across_visits(tmp_path):
    from backend.state.store import PatientStore

    store = PatientStore(path=str(tmp_path))
    record = store.create("p-001")
    assert record.visits == []

    record.pending_workups.append("MRI brain")
    store.save(record)

    reloaded = store.load("p-001")
    assert "MRI brain" in reloaded.pending_workups
