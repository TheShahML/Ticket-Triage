from __future__ import annotations

"""API-level integration tests for FastAPI triage routes."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main_module
from app.index import build_index

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a test client with deterministic local embeddings."""

    def _build_index(_data_dir: Path) -> object:
        return build_index(ROOT / "data", force_local=True, local_backend="tfidf")

    monkeypatch.setattr(main_module, "build_index", _build_index)
    monkeypatch.setattr(main_module, "llm_enabled", lambda: False)
    main_module.INDEX = None
    main_module.DEMO_EXAMPLE_TICKETS = []

    with TestClient(main_module.app) as test_client:
        yield test_client


def test_health_endpoint(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert isinstance(payload["example_tickets"], list)
    assert payload["example_tickets"]


def test_triage_endpoint_valid_request(client: TestClient) -> None:
    response = client.post("/triage", json={"text": "I was charged twice and need a refund.", "top_k": 3})
    assert response.status_code == 200
    payload = response.json()
    assert payload["category"] == "Billing"
    assert payload["match_quality"] == "sufficient_signal"
    assert payload["priority"] in {"Medium", "High"}
    assert len(payload["similar_examples"]) == 3
    assert len(payload["kb_context"]) == 3


def test_triage_endpoint_rejects_invalid_top_k(client: TestClient) -> None:
    response = client.post("/triage", json={"text": "Password reset emails are not arriving.", "top_k": 0})
    assert response.status_code == 422


def test_draft_reply_requires_llm_key(client: TestClient) -> None:
    response = client.post("/draft_reply", json={"text": "Can you help with login issue?", "kb_top_k": 2})
    assert response.status_code == 400
    assert "OPENAI_API_KEY not set" in response.json()["detail"]
