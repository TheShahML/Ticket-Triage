# ticket-triage

FastAPI service for explainable support-ticket triage.

Given a free-form support ticket, the API returns:

- predicted category: `Billing`, `Account`, `Bug`, `Feature`, or `Other`
- match quality: `sufficient_signal` or `insufficient_signal`
- predicted priority: `Low`, `Medium`, or `High`
- short priority reasoning
- nearest labeled tickets used for prediction
- relevant knowledge-base snippets
- recommended next-step checklist
- drafted customer reply

The system is designed to stay interpretable: predictions are based on nearest-neighbor retrieval over labeled examples, and the response includes supporting examples and KB context.

## Features

- explainable category prediction using embeddings + kNN
- deterministic priority scoring with auditable cues
- retrieval of similar historical tickets
- retrieval of relevant knowledge-base snippets
- deterministic reply generation by default
- optional LLM-based reply drafting when `OPENAI_API_KEY` is set
- no database or vector store required
- local JSONL fixtures and in-memory index

## How it works

1. Ticket text is embedded into a vector.
2. The vector is compared against labeled support tickets using cosine similarity.
3. The top-k nearest tickets are used for weighted category voting.
4. Priority is assigned using deterministic urgency cues plus nearby ticket signals.
5. Relevant knowledge-base snippets are retrieved using the same similarity pipeline.
6. A reply is drafted from retrieved context, either with a deterministic template or a single grounded LLM call.

## Embedding backends

The app supports three embedding paths:

1. **OpenAI embeddings** when `OPENAI_API_KEY` is available
2. **Local sentence-transformer** (`sentence-transformers/all-MiniLM-L6-v2`) when no API key is present
3. **TF-IDF fallback** if the local model is unavailable

Tests force TF-IDF mode so they remain fast and deterministic.

## Project structure

```text
app/
  main.py       FastAPI routes and app startup
  index.py      JSONL loading and in-memory vector index construction
  embed.py      Embedding abstraction and cosine similarity
  triage.py     Category prediction, priority logic, and retrieval
  reply.py      Deterministic reply generation and optional LLM draft

data/
  labeled_tickets.jsonl
  kb_snippets.jsonl
  demo_tickets.json

scripts/
  evaluate.py   Evaluation script for baseline vs embedding-kNN

static/
  index.html    Single-page UI

tests/
  test_triage.py
```

## API

### `GET /health`

Health check.

### `POST /triage`

Request:

```json
{"text":"I was charged twice today and need a refund","top_k":5}
```

Response includes:

- `category`
- `match_quality`
- `priority`
- `confidence`
- `reasoning`
- `similar_examples`
- `kb_context`
- `next_steps`
- `drafted_reply`

### `POST /draft_reply`

Request:

```json
{"text":"...","kb_top_k":5}
```

Response:

```json
{"drafted_reply":"...","tone":"...","citations":[...]}
```

Uses the LLM path only when `OPENAI_API_KEY` is set.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## Tests

```bash
pytest -q
```

## Evaluation

```bash
python scripts/evaluate.py
```

The evaluation script compares embedding-kNN against a simple keyword baseline and reports:

- train/test sizes
- overall accuracy
- per-category accuracy

## Example tickets

- `I was charged twice today and our finance team needs an urgent refund.`
- `Mobile app crashes every time we upload an image from iOS 18.`

## Design goals

- keep the retrieval and decision path easy to inspect
- avoid hidden classification logic
- make predictions traceable to similar historical examples
- keep the project lightweight enough to run locally without infrastructure dependencies

## Notes

- The application works without any LLM dependency.
- LLM usage is limited to optional reply drafting.
- Low-signal tickets are explicitly marked with `match_quality="insufficient_signal"` and routed conservatively.
- Retrieved examples and knowledge-base snippets are included to make outputs easier to inspect and debug.

## Future work

See `docs/SCALING.md` for notes on scaling this prototype toward a production deployment.
