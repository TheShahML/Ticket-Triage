from __future__ import annotations

"""Pydantic request/response schemas used by the API layer."""

from typing import List

from pydantic import BaseModel, Field


class TriageRequest(BaseModel):
    """Payload for `/triage` endpoint."""

    text: str = Field(min_length=5, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=10)


class SimilarExample(BaseModel):
    """A nearest labeled ticket returned for transparency."""

    text: str
    label: str
    score: float


class KBContext(BaseModel):
    """A retrieved knowledge-base snippet relevant to the ticket."""

    title: str
    snippet: str
    score: float


class TriageResponse(BaseModel):
    """Complete triage output shown in the UI and API clients."""

    category: str
    match_quality: str
    priority: str
    confidence: float
    reasoning: str
    similar_examples: List[SimilarExample]
    kb_context: List[KBContext]
    next_steps: List[str]
    drafted_reply: str


class DraftReplyRequest(BaseModel):
    """Payload for `/draft_reply` endpoint."""

    text: str = Field(min_length=5, max_length=4000)
    kb_top_k: int = Field(default=5, ge=1, le=10)


class DraftReplyResponse(BaseModel):
    """LLM or deterministic reply output with citations."""

    drafted_reply: str
    tone: str
    citations: List[str]


class HealthResponse(BaseModel):
    """Basic service health payload used by frontend bootstrap."""

    status: str
    llm_enabled: bool
    example_tickets: List[str]
