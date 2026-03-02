from __future__ import annotations

"""Reply generation utilities for deterministic and optional LLM draft paths."""

import json
import os
from typing import List

from app.schema import DraftReplyResponse, KBContext


def llm_enabled() -> bool:
    """Return true when an OpenAI API key is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


def deterministic_reply(ticket_text: str, kb_context: List[KBContext]) -> DraftReplyResponse:
    """Build deterministic support reply using retrieved KB snippet titles."""
    titles = [item.title for item in kb_context[:3]]
    title_text = ", ".join(titles) if titles else "General Support Policy"
    drafted = (
        "Hi,\n\n"
        "Thanks for contacting support. I reviewed your request and collected the most relevant guidance. "
        "Please try the listed steps and reply with any missing details if the issue continues.\n\n"
        f"Summary of your request: {ticket_text[:160]}\n"
        f"References used: {title_text}.\n\n"
        "Best,\nSupport Team"
    )
    return DraftReplyResponse(drafted_reply=drafted, tone="professional", citations=titles)


def llm_reply(ticket_text: str, kb_context: List[KBContext]) -> DraftReplyResponse:
    """Generate a grounded support reply from one LLM call.

    The prompt includes only the ticket text and retrieved KB snippets and asks
    for strict JSON output (`drafted_reply`, `tone`, `citations`).
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context_block = "\n".join(f"- {k.title}: {k.snippet}" for k in kb_context)
    prompt = (
        "You draft support replies grounded only in provided snippets. "
        "Return JSON with keys: drafted_reply (string), tone (string), citations (array of snippet titles). "
        "Do not invent citations.\n\n"
        f"Ticket:\n{ticket_text}\n\n"
        f"KB snippets:\n{context_block}\n"
    )

    response = client.chat.completions.create(
        model=os.getenv("OPENAI_REPLY_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        messages=[
            {"role": "system", "content": "Return strict JSON only."},
            {"role": "user", "content": prompt},
        ],
    )

    raw = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(raw)
        return DraftReplyResponse(
            drafted_reply=payload.get("drafted_reply", ""),
            tone=payload.get("tone", "professional"),
            citations=payload.get("citations", []),
        )
    except json.JSONDecodeError:
        # Keep API behavior stable even if the model returns malformed JSON.
        return deterministic_reply(ticket_text, kb_context)
