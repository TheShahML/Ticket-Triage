from __future__ import annotations

"""FastAPI entrypoint for triage APIs and demo UI."""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from app.index import TicketIndex, build_index
from app.reply import deterministic_reply, llm_enabled, llm_reply
from app.schema import (
    DraftReplyRequest,
    DraftReplyResponse,
    HealthResponse,
    TriageRequest,
    TriageResponse,
)
from app.triage import triage_ticket

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
STATIC_DIR = ROOT / "static"
DEMO_TICKETS_FILE = DATA_DIR / "demo_tickets.json"
load_dotenv(ROOT / ".env")

DEFAULT_DEMO_TICKETS = [
    "We were charged twice and need the duplicate payment reversed.",
    "Password reset emails are not arriving, so the user is locked out.",
    "Our app setup keeps failing with HTTP 500 during startup.",
    "Feature request: please add dark mode and OS theme sync.",
]

app = FastAPI(title="Ticket Triage Embeddings", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

INDEX: TicketIndex | None = None
DEMO_EXAMPLE_TICKETS: list[str] = []


def _load_demo_tickets(path: Path) -> list[str]:
    """Load out-of-sample demo tickets used by the frontend example dropdown."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            tickets = [item.strip() for item in payload if isinstance(item, str) and item.strip()]
            if tickets:
                return tickets
    except FileNotFoundError:
        pass
    except json.JSONDecodeError:
        pass
    return DEFAULT_DEMO_TICKETS


@app.on_event("startup")
def startup_event() -> None:
    """Build and cache the in-memory index at startup."""
    global INDEX, DEMO_EXAMPLE_TICKETS
    INDEX = build_index(DATA_DIR)
    DEMO_EXAMPLE_TICKETS = _load_demo_tickets(DEMO_TICKETS_FILE)


@app.get("/", include_in_schema=False)
def read_home() -> FileResponse:
    """Serve the single-page frontend used by support users."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status and capability flags for UI bootstrapping."""
    assert INDEX is not None
    return HealthResponse(
        status="ok",
        llm_enabled=llm_enabled(),
        example_tickets=DEMO_EXAMPLE_TICKETS,
    )


@app.post("/triage", response_model=TriageResponse)
def triage(payload: TriageRequest) -> TriageResponse:
    """Run end-to-end triage for one ticket."""
    assert INDEX is not None
    return triage_ticket(payload.text, INDEX, top_k=payload.top_k)


@app.post("/draft_reply", response_model=DraftReplyResponse)
def draft_reply(payload: DraftReplyRequest) -> DraftReplyResponse:
    """Generate an optional LLM-backed reply using retrieved context."""
    assert INDEX is not None
    triage_result = triage_ticket(payload.text, INDEX, top_k=payload.kb_top_k)

    if not llm_enabled():
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY not set. LLM drafting is unavailable.")

    try:
        return llm_reply(payload.text, triage_result.kb_context)
    except Exception as exc:  # pragma: no cover - network path
        raise HTTPException(status_code=502, detail=f"LLM call failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
