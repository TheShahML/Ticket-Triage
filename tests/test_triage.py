from __future__ import annotations

"""Fast, local tests for triage output behavior."""

from pathlib import Path

from app.index import build_index
from app.triage import triage_ticket

ROOT = Path(__file__).resolve().parents[1]


def test_billing_ticket_category_and_priority() -> None:
    """Billing-like text should map to Billing with elevated priority."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("I was charged twice and need an urgent refund.", index, top_k=5)
    assert result.category == "Billing"
    assert result.priority in {"Medium", "High"}
    assert result.similar_examples
    assert result.kb_context


def test_account_ticket_category() -> None:
    """Login/access language should classify as Account."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Cannot login after password reset, account seems locked.", index, top_k=5)
    assert result.category == "Account"


def test_feature_request_priority_low() -> None:
    """Feature-request phrasing should result in low urgency by default."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Would like dark mode and weekly scheduled exports.", index, top_k=5)
    assert result.category == "Feature"
    assert result.priority == "Low"
