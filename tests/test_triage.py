from __future__ import annotations

"""Fast, local tests for triage output behavior."""

from pathlib import Path

import pytest

from app.index import build_index
from app.triage import triage_ticket

ROOT = Path(__file__).resolve().parents[1]


def test_billing_ticket_category_and_priority() -> None:
    """Billing-like text should map to Billing with elevated priority."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("I was charged twice and need an urgent refund.", index, top_k=5)
    assert result.category == "Billing"
    assert result.match_quality == "sufficient_signal"
    assert result.priority in {"Medium", "High"}
    assert result.similar_examples
    assert result.kb_context


def test_account_ticket_category() -> None:
    """Login/access language should classify as Account."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Cannot login after password reset, account seems locked.", index, top_k=5)
    assert result.category == "Account"
    assert result.match_quality == "sufficient_signal"


def test_feature_request_priority_low() -> None:
    """Feature-request phrasing should result in low urgency by default."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Would like dark mode and weekly scheduled exports.", index, top_k=5)
    assert result.category == "Feature"
    assert result.match_quality == "sufficient_signal"
    assert result.priority == "Low"


def test_top_k_must_be_positive() -> None:
    """Invalid neighbor counts should fail fast with a clear error."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    with pytest.raises(ValueError, match="top_k must be >= 1"):
        triage_ticket("Cannot login to my account.", index, top_k=0)


def test_download_word_does_not_trigger_outage_urgency() -> None:
    """`down` urgency cue should not match unrelated words like `download`."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Feature request: add bulk CSV download in settings.", index, top_k=5)
    assert result.category == "Feature"
    assert result.priority == "Low"


def test_top_k_larger_than_dataset_is_bounded() -> None:
    """Asking for more neighbors than available should gracefully cap results."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Cannot login after enabling MFA.", index, top_k=999)
    assert len(result.similar_examples) == len(index.labeled_tickets)
    assert len(result.kb_context) == len(index.kb_snippets)


def test_urgent_broad_impact_ticket_maps_high_priority() -> None:
    """Strong urgency + broad-impact language should map to high priority."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Urgent outage: all users cannot sign in to the app.", index, top_k=5)
    assert result.priority == "High"


def test_low_signal_text_routes_to_other_with_flag() -> None:
    """Very low-information text should explicitly mark insufficient matching signal."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("zxqv qwrty plmokn", index, top_k=5)
    assert result.category == "Other"
    assert result.match_quality == "insufficient_signal"
    assert result.confidence <= 0.35
    assert "Insufficient matching signal" in result.reasoning


def test_greeting_text_marks_insufficient_signal() -> None:
    """Common conversational text should not be treated as confident triage input."""
    index = build_index(ROOT / "data", force_local=True, local_backend="tfidf")
    result = triage_ticket("Hello, how are you?", index, top_k=5)
    assert result.category == "Other"
    assert result.match_quality == "insufficient_signal"
    assert "Insufficient matching signal" in result.reasoning
