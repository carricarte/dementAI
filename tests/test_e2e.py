"""End-to-end tests for the DCA API.

Tiers (controlled by markers):
  (no marker)   — structural only; no LLM, no knowledge base; always fast
  requires_llm  — calls the real Anthropic API; skipped if ANTHROPIC_API_KEY unset
  requires_kb   — needs a populated LanceDB table; skipped if empty

Run all:
    pytest tests/test_e2e.py -v
Skip slow tests:
    pytest tests/test_e2e.py -v -m "not requires_llm"
"""

from __future__ import annotations

import os
import uuid

import pytest
from fastapi.testclient import TestClient

# ── markers ──────────────────────────────────────────────────────────────────

_has_api_key = bool(os.getenv("ANTHROPIC_API_KEY"))
_has_kb = False
try:
    import lancedb

    from backend.config import settings as _s

    _tbl = lancedb.connect(_s.lancedb_path).open_table("knowledge")
    _has_kb = _tbl.count_rows() > 0
    del _tbl  # don't hold a stale handle for the whole session
except Exception:
    pass

requires_llm = pytest.mark.skipif(not _has_api_key, reason="ANTHROPIC_API_KEY not set")
requires_kb = pytest.mark.skipif(not _has_kb, reason="Knowledge base not populated")


# ── fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """TestClient with patient store redirected to a temp directory."""
    import backend.state.store as _store_mod
    from backend.main import app
    from backend.state.store import PatientStore

    tmp = tmp_path_factory.mktemp("patients")
    _store_mod.patient_store = PatientStore(path=str(tmp))
    with TestClient(app) as c:
        yield c
    # restore (harmless; module is re-imported per test session anyway)
    _store_mod.patient_store = PatientStore(path=_s.patient_store_path)


@pytest.fixture
def patient_id():
    """A unique patient ID that is isolated per test."""
    return f"e2e-{uuid.uuid4().hex[:8]}"


# ── structural tests (no LLM, no KB) ─────────────────────────────────────────


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_patient_not_found(client):
    resp = client.get("/patient/no-such-patient-xyz")
    assert resp.status_code == 404


def test_query_request_requires_body(client):
    resp = client.post("/query/", json={})
    assert resp.status_code == 422  # missing required fields


def test_query_response_schema(client, patient_id):
    """Response always has the required keys, even when LLM/KB are unavailable."""
    from unittest.mock import MagicMock, patch

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="screening")

    with (
        patch("backend.agents.coordinator.get_llm", return_value=mock_llm),
        patch("backend.agents.screening.get_llm", return_value=mock_llm),
        patch_retrieval(),
    ):
        resp = client.post(
            "/query/",
            json={"patient_id": patient_id, "query": "MoCA score is 21"},
        )

    assert resp.status_code == 200
    body = resp.json()
    assert "patient_id" in body
    assert "stage" in body
    assert "response" in body
    assert "citations" in body
    assert isinstance(body["citations"], list)


# ── LLM-dependent tests ───────────────────────────────────────────────────────


@requires_llm
@pytest.mark.parametrize(
    "query,expected_stage",
    [
        ("Patient's MoCA score is 21/30. What does this indicate?", "screening"),
        (
            "MRI shows hippocampal atrophy, CSF amyloid/tau ratio abnormal. Likely diagnosis?",
            "diagnosis",
        ),
        (
            "How can we reduce this patient's dementia risk given hypertension and obesity?",
            "prevention",
        ),
        ("Should we start donepezil for this mild Alzheimer's patient?", "treatment"),
        ("Patient needs 24-hour care. When is nursing home placement appropriate?", "care"),
    ],
)
def test_query_stage_routing(client, patient_id, query, expected_stage):
    """Each query must be routed to the correct clinical stage."""
    with patch_retrieval():
        resp = client.post("/query/", json={"patient_id": patient_id, "query": query})
    assert resp.status_code == 200
    assert resp.json()["stage"] == expected_stage


@requires_llm
def test_query_visit_persisted(client, patient_id):
    """A completed query must appear as a visit in the patient record."""
    with patch_retrieval():
        resp = client.post(
            "/query/",
            json={"patient_id": patient_id, "query": "MoCA is 24, what does this mean?"},
        )
    assert resp.status_code == 200

    patient_resp = client.get(f"/patient/{patient_id}")
    assert patient_resp.status_code == 200
    visits = patient_resp.json()["record"]["visits"]
    assert len(visits) == 1
    assert visits[0]["stage"] == "screening"


@requires_llm
def test_query_response_not_empty(client, patient_id):
    """Specialist response must be non-empty even without a knowledge base."""
    with patch_retrieval():
        resp = client.post(
            "/query/",
            json={"patient_id": patient_id, "query": "MoCA is 18, memory concerns"},
        )
    assert resp.status_code == 200
    assert len(resp.json()["response"]) > 50


# ── Full pipeline tests (LLM + knowledge base) ────────────────────────────────


@requires_llm
@requires_kb
def test_full_pipeline_returns_citations(client, patient_id):
    """With a real KB, every query must return at least one citation."""
    resp = client.post(
        "/query/",
        json={
            "patient_id": patient_id,
            "query": "75-year-old, MoCA 22, personality changes. Likely diagnosis and workup?",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stage"] == "diagnosis"
    assert len(body["citations"]) >= 1
    for c in body["citations"]:
        assert c["source"] in ("pubmed", "awmf", "clinicaltrials", "neurology", "alz", "aan")
        assert c["title"]


@requires_llm
@requires_kb
def test_full_pipeline_response_mentions_condition(client, patient_id):
    """Response to a bvFTD presentation must mention frontotemporal dementia."""
    resp = client.post(
        "/query/",
        json={
            "patient_id": patient_id,
            "query": (
                "65-year-old with disinhibition, apathy, executive dysfunction"
                " for 2 years. Diagnosis?"
            ),
        },
    )
    assert resp.status_code == 200
    response_text = resp.json()["response"].lower()
    assert any(term in response_text for term in ("frontotemporal", "bvftd", "ftd", "frontal"))


@requires_llm
@requires_kb
def test_full_pipeline_screening_recommends_workup(client, patient_id):
    """A screening response must include next-step recommendations."""
    resp = client.post(
        "/query/",
        json={"patient_id": patient_id, "query": "Patient scored 23 on MoCA. What next?"},
    )
    assert resp.status_code == 200
    response_text = resp.json()["response"].lower()
    assert any(
        term in response_text
        for term in ("workup", "evaluation", "referral", "follow", "assessment", "test")
    )


# ── helpers ───────────────────────────────────────────────────────────────────


def patch_retrieval():
    """Stub retrieval at each specialist's import site so LLM tests don't need a real KB.

    Patching `backend.tools.retrieval.retrieve` alone would miss already-bound
    local references in each agent module. We patch every use-site instead.
    """
    from contextlib import ExitStack
    from unittest.mock import patch

    def _noop(*_a, **_kw):
        return {"text": "", "citations": []}

    stack = ExitStack()
    for agent in ("screening", "diagnosis", "prevention", "treatment", "care"):
        stack.enter_context(patch(f"backend.agents.{agent}.retrieve", side_effect=_noop))
    return stack
