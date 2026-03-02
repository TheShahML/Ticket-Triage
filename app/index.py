from __future__ import annotations

"""Load source data and build the runtime search index."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from app.embed import EmbeddingEngine


@dataclass
class LabeledTicket:
    """One historical support ticket with human labels.

    These are the examples the classifier learns from at inference time.
    """

    text: str
    category: str
    priority: str


@dataclass
class KBSnippet:
    """One retrieval chunk from support documentation."""

    title: str
    snippet: str
    category: str


@dataclass
class TicketIndex:
    """In-memory bundle of records and vectors used for live triage."""

    labeled_tickets: List[LabeledTicket]
    kb_snippets: List[KBSnippet]
    labeled_vectors: np.ndarray
    kb_vectors: np.ndarray
    embedder: EmbeddingEngine


def _read_jsonl(path: Path) -> List[dict]:
    """Read newline-delimited JSON records from disk.

    Args:
        path: JSONL file path.

    Returns:
        Parsed records in file order for deterministic loading.
    """
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_index(
    data_dir: Path,
    force_local: bool = False,
    local_backend: str = "sentence-transformers",
) -> TicketIndex:
    """Build the in-memory triage index from fixture files.

    Args:
        data_dir: Directory containing labeled ticket and KB JSONL fixtures.
        force_local: If true, disable OpenAI embeddings even if an API key exists.
        local_backend: Local backend name passed to `EmbeddingEngine`.

    Returns:
        Fully materialized `TicketIndex` ready for similarity search at request time.
    """
    labeled_rows = _read_jsonl(data_dir / "labeled_tickets.jsonl")
    kb_rows = _read_jsonl(data_dir / "kb_snippets.jsonl")

    labeled_tickets = [
        LabeledTicket(text=row["text"], category=row["category"], priority=row["priority"])
        for row in labeled_rows
    ]
    kb_snippets = [
        KBSnippet(title=row["title"], snippet=row["snippet"], category=row["category"])
        for row in kb_rows
    ]

    corpus = [ticket.text for ticket in labeled_tickets] + [f"{kb.title} {kb.snippet}" for kb in kb_snippets]
    embedder = EmbeddingEngine(corpus=corpus, force_local=force_local, local_backend=local_backend)

    labeled_vectors = embedder.embed_documents([ticket.text for ticket in labeled_tickets])
    kb_vectors = embedder.embed_documents([f"{kb.title} {kb.snippet}" for kb in kb_snippets])

    return TicketIndex(
        labeled_tickets=labeled_tickets,
        kb_snippets=kb_snippets,
        labeled_vectors=labeled_vectors,
        kb_vectors=kb_vectors,
        embedder=embedder,
    )
