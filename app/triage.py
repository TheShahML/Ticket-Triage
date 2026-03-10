from __future__ import annotations

"""Core triage logic: classify, prioritize, and retrieve context."""

import re
from typing import Dict, List, Tuple

import numpy as np

from app.embed import cosine_similarity
from app.index import TicketIndex
from app.schema import KBContext, SimilarExample, TriageResponse


CATEGORY_STEPS: Dict[str, List[str]] = {
    "Billing": [
        "Confirm impacted invoice or charge ID.",
        "Check payment history and any failed transactions.",
        "Apply refund/credit policy or escalate to finance if needed.",
    ],
    "Account": [
        "Verify account ownership using standard identity checks.",
        "Review login/security events for suspicious activity.",
        "Reset access path and confirm user can sign in.",
    ],
    "Bug": [
        "Capture reproduction steps, browser/app version, and timestamps.",
        "Check status page and recent deployments for related incidents.",
        "Create engineering bug ticket with severity and customer impact.",
    ],
    "Feature": [
        "Confirm the desired workflow and business goal.",
        "Map request to existing roadmap or similar asks.",
        "Share workaround and log feature request for product review.",
    ],
    "Other": [
        "Clarify the request scope and expected outcome.",
        "Route to the correct queue or team owner.",
        "Set follow-up expectation with the customer.",
    ],
}

LOW_SIGNAL_TOP_SIMILARITY = 0.18
LOW_SIGNAL_POSITIVE_SUM = 0.35
LOW_SIGNAL_CONFIDENCE = 0.30


def _top_k(scores: np.ndarray, k: int) -> List[Tuple[int, float]]:
    """Return top-k `(index, score)` pairs sorted by descending score."""
    k = min(k, len(scores))
    idx = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in idx]


def _choose_category(index: TicketIndex, ranked: List[Tuple[int, float]]) -> Tuple[str, float]:
    """Pick category via weighted vote across nearest labeled examples.

    Returns:
        `(category, confidence)` where confidence is winner vote share.
    """
    # Weighted voting keeps the logic explainable while still using similarity strength.
    votes: Dict[str, float] = {}
    for i, score in ranked:
        label = index.labeled_tickets[i].category
        votes[label] = votes.get(label, 0.0) + max(score, 0.0)

    winner = max(votes, key=votes.get)
    total = sum(votes.values()) or 1.0
    vote_share = votes[winner] / total

    sorted_votes = sorted(votes.values(), reverse=True)
    top1 = sorted_votes[0]
    top2 = sorted_votes[1] if len(sorted_votes) > 1 else 0.0
    margin = (top1 - top2) / max(top1 + top2, 1e-8)

    # Similarity-strength term reduces confidence for weak nearest matches.
    top_similarity = ranked[0][1] if ranked else 0.0
    sim_strength = min(max(top_similarity, 0.0), 1.0)

    # Diversity penalty avoids saturating confidence when vote concentration is artificial.
    unique_labels = len(votes)
    diversity_penalty = 0.9 if unique_labels == 1 else 1.0

    confidence = (0.45 * vote_share + 0.35 * margin + 0.20 * sim_strength) * diversity_penalty
    confidence = min(max(confidence, 0.0), 0.95)
    return winner, confidence


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return true when a phrase appears as a standalone term sequence."""
    pattern = r"\b" + r"\s+".join(re.escape(part) for part in phrase.split()) + r"\b"
    return bool(re.search(pattern, text))


def _priority_score_from_text(text: str) -> Tuple[float, List[str]]:
    """Produce a deterministic urgency score from explicit keyword cues."""
    lower = text.lower()
    score = 0.45
    reasons: List[str] = []

    high_terms = ["urgent", "asap", "outage", "down", "cannot", "can't", "error 500", "payment failed"]
    medium_terms = ["blocked", "issue", "failed", "problem", "unable", "refund"]
    low_terms = ["feature", "enhancement", "would like", "request", "nice to have"]

    if any(_contains_phrase(lower, term) for term in high_terms):
        score += 0.35
        reasons.append("contains urgent/impact language")
    elif any(_contains_phrase(lower, term) for term in medium_terms):
        score += 0.15
        reasons.append("contains issue-oriented language")

    if any(_contains_phrase(lower, term) for term in low_terms):
        score -= 0.25
        reasons.append("looks like a non-urgent request")

    if _contains_phrase(lower, "all users") or _contains_phrase(lower, "entire team"):
        score += 0.2
        reasons.append("potentially broad customer impact")

    return max(0.0, min(score, 1.0)), reasons


def _priority_from_neighbors(index: TicketIndex, ranked: List[Tuple[int, float]]) -> float:
    """Estimate priority from neighbor priorities using similarity as weight."""
    mapping = {"Low": 0.2, "Medium": 0.55, "High": 0.9}
    weighted = 0.0
    total_weight = 0.0
    for i, sim in ranked:
        weight = max(sim, 0.0)
        weighted += weight * mapping.get(index.labeled_tickets[i].priority, 0.55)
        total_weight += weight
    if total_weight == 0:
        return 0.55
    return weighted / total_weight


def _score_to_priority(score: float) -> str:
    """Map normalized priority score to label buckets."""
    if score < 0.38:
        return "Low"
    if score < 0.72:
        return "Medium"
    return "High"


def _reason_text(category: str, priority: str, confidence: float, cues: List[str]) -> str:
    """Generate a short human-readable explanation for the triage result."""
    cue_text = "; ".join(cues) if cues else "nearest historical tickets"
    return (
        f"Predicted {category} from nearest labeled tickets; priority set to {priority} "
        f"using keyword cues + neighbor signal ({cue_text}). Confidence {confidence:.2f}."
    )


def _build_template_reply(ticket_text: str, category: str, kb_context: List[KBContext]) -> str:
    """Create deterministic fallback reply grounded in retrieved KB titles."""
    cited_titles = ", ".join(item.title for item in kb_context[:3]) or "our help center"
    return (
        "Hi there,\n\n"
        "Thanks for reporting this. I reviewed your message and routed it as "
        f"{category}. Based on your note: \"{ticket_text[:140]}\"\n\n"
        "Here is what we recommend next:\n"
        "1) Follow the steps in the cited guidance.\n"
        "2) Reply with any missing details (time, screenshots, account ID).\n"
        "3) We will confirm resolution or next update shortly.\n\n"
        f"Helpful references: {cited_titles}.\n\n"
        "Best,\nSupport Team"
    )


def _has_insufficient_signal(ranked: List[Tuple[int, float]], confidence: float) -> bool:
    """Detect weak-neighbor scenarios where categorization should be conservative."""
    if not ranked:
        return True
    top_similarity = max(ranked[0][1], 0.0)
    positive_sum = sum(max(score, 0.0) for _, score in ranked)
    if confidence < LOW_SIGNAL_CONFIDENCE:
        return True
    return top_similarity < LOW_SIGNAL_TOP_SIMILARITY and positive_sum < LOW_SIGNAL_POSITIVE_SUM


def triage_ticket(ticket_text: str, index: TicketIndex, top_k: int = 5) -> TriageResponse:
    """Run the full triage pipeline for one ticket.

    Steps:
        1) Embed ticket text.
        2) Retrieve nearest labeled examples + KB snippets.
        3) Predict category and priority with explainable logic.
        4) Build next steps and a deterministic draft reply.
    """
    if top_k < 1:
        raise ValueError("top_k must be >= 1")
    if not index.labeled_tickets or not index.kb_snippets:
        raise ValueError("index must include labeled tickets and KB snippets")

    q = index.embedder.embed_query(ticket_text)
    sim_labels = cosine_similarity(q, index.labeled_vectors)
    sim_kb = cosine_similarity(q, index.kb_vectors)

    ranked_label = _top_k(sim_labels, top_k)
    ranked_kb = _top_k(sim_kb, top_k)

    category, confidence = _choose_category(index, ranked_label)
    match_quality = "sufficient_signal"
    low_signal = _has_insufficient_signal(ranked_label, confidence)
    if low_signal:
        category = "Other"
        confidence = min(confidence, 0.35)
        match_quality = "insufficient_signal"

    heuristic_score, cues = _priority_score_from_text(ticket_text)
    neighbor_score = _priority_from_neighbors(index, ranked_label)
    blended_score = 0.7 * heuristic_score + 0.3 * neighbor_score
    priority = _score_to_priority(blended_score)

    similar_examples = [
        SimilarExample(
            text=index.labeled_tickets[i].text,
            label=index.labeled_tickets[i].category,
            score=round(score, 3),
        )
        for i, score in ranked_label
    ]

    kb_context = [
        KBContext(
            title=index.kb_snippets[i].title,
            snippet=index.kb_snippets[i].snippet,
            score=round(score, 3),
        )
        for i, score in ranked_kb
    ]

    next_steps = CATEGORY_STEPS.get(category, CATEGORY_STEPS["Other"]).copy()
    if priority == "High":
        next_steps.insert(0, "Acknowledge customer quickly and set a 1-hour update window.")

    drafted_reply = _build_template_reply(ticket_text, category, kb_context)

    return TriageResponse(
        category=category,
        match_quality=match_quality,
        priority=priority,
        confidence=round(confidence, 3),
        reasoning=(
            "Insufficient matching signal for confident categorization. "
            + _reason_text(category, priority, confidence, cues)
            if low_signal
            else _reason_text(category, priority, confidence, cues)
        ),
        similar_examples=similar_examples,
        kb_context=kb_context,
        next_steps=next_steps,
        drafted_reply=drafted_reply,
    )
