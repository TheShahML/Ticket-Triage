# ticket-triage-embeddings

A small, explainable FastAPI app for customer-support triage.

It takes free text and returns:
1. predicted category (`Billing`, `Account`, `Bug`, `Feature`, `Other`)
2. predicted priority (`Low`, `Medium`, `High`) with short reasoning
3. top-k similar labeled tickets (transparency)
4. top-k relevant KB snippets (context retrieval)
5. recommended next-step checklist
6. drafted reply
   - deterministic template by default
   - optional single-call LLM draft if `OPENAI_API_KEY` exists

## Why this project is intentionally simple

- No database
- No vector DB
- Small JSONL fixtures committed to repo
- In-memory embedding index built at startup
- Clear functions with explicit names and short control flow

## Design Decisions
- **Embeddings + kNN:** chosen for explainability/understandability and fast iteration; predictions can be traced to nearest labeled tickets.
- **Optional LLM Embeddings** enabled only when key is present for higher-quality embeddings; otherwise uses local transformer processing for embeddings with TF-IDF 
- **Weighted neighbor voting:** closer matches have more influence than weaker matches.
- **Shared similarity pipeline:** same vector search powers both category prediction and KB retrieval.
- **Priority logic:** deterministic cues + neighbor signal for auditable urgency decisions.
- **Optional LLM drafting:** enabled only when key is present for higher-quality replies; core flow works without it to control cost and keep reliability high.
- **Scope choice:** avoided heavier trained classifiers to keep the project small, readable, and easy to demo end-to-end.

## Data Provenance

- `data/labeled_tickets.jsonl` and `data/kb_snippets.jsonl` use examples adapted from public support threads and issue trackers.
- Each record includes a `source_url` field for traceability.
- Text is normalized/paraphrased for consistency and to avoid copying long verbatim content.

## Architecture

- `app/main.py`: FastAPI routes + static file serving
- `app/index.py`: loads JSONL files and builds in-memory vectors
- `app/embed.py`: embedding abstraction + cosine similarity
- `app/triage.py`: kNN category voting, priority heuristics, retrieval
- `app/reply.py`: deterministic reply and optional LLM grounded reply
- `static/index.html`: single-page Tailwind UI (no React)
- `tests/test_triage.py`: fast golden tests
- `scripts/evaluate.py`: tiny benchmark (keyword baseline vs embedding-kNN)
- `data/demo_tickets.json`: out-of-sample examples used by the UI `Load Example` dropdown

## Architecture Diagram

### Executive View (Non-Technical)

```text
+----------------------------+    +---------------------------+    +----------------------------------+
| Support agent enters ticket| -> | API triages the request   | -> | Finds similar tickets + KB context|
+----------------------------+    +---------------------------+    +----------------------------------+
                                                                  |
                                                                  v
                                     +-----------------------------------------+
                                     | Returns category, priority, next steps, | 
                                     | drafted reply                           | 
                                     +-----------------------------------------+
   
```

### Technical View

```text
+-------------------------------------+
| Browser UI (static/index.html)      |
+-------------------------------------+
                  |
                  | GET /, GET /health, POST /triage, POST /draft_reply
                  v
+-------------------------------------+
| FastAPI (app/main.py)               |
+-------------------------------------+
                  |
                  | startup
                  v
+-------------------------------------+        +-------------------------------------+
| build_index (app/index.py)          | -----> | labeled_tickets.jsonl              |
+-------------------------------------+        +-------------------------------------+
                  |
                  +---------------------------> +-------------------------------------+
                  |                             | kb_snippets.jsonl                   |
                  |                             +-------------------------------------+
                  v
+-------------------------------------+
| EmbeddingEngine (app/embed.py)      |
+-------------------------------------+
          |                                         |
          v                                         v
+---------------------------+             +---------------------------+
| Labeled vectors           |             | KB vectors                |
+---------------------------+             +---------------------------+
          \                                         /
           \                                       /
            v                                     v
          +-----------------------------------------------+
          | triage_ticket (app/triage.py)                |
          +-----------------------------------------------+
                              |
                              v
          +-----------------------------------------------+
          | Triage response JSON                           |
          | category, priority, confidence, reasoning      |
          | similar_examples, kb_context, next_steps,      |
          | drafted_reply                                  |
          +-----------------------------------------------+

Optional draft reply path:
FastAPI -> app/reply.py -> (with OPENAI_API_KEY) OpenAI chat completion
                      -> (without key) deterministic fallback
```

Quick read of the flow:
1. Frontend calls FastAPI endpoints.
2. On startup, backend loads JSONL data and builds vectors.
3. `/triage` runs kNN + retrieval and returns explainable JSON.
4. `/draft_reply` uses LLM only when key exists, otherwise deterministic fallback.

## How embeddings + kNN works

1. Turn ticket text into a vector (embedding).
2. Compare that vector with vectors of labeled tickets using cosine similarity.
3. Pick top-k nearest tickets.
4. Category is chosen by weighted voting (nearer neighbors get more vote).
5. Confidence = winning vote share of total positive vote.
6. Priority comes from deterministic keyword cues, then lightly adjusted by neighbor priorities.
7. KB snippets are retrieved with the same nearest-neighbor similarity process.

This is easy to explain because every prediction can show its nearest examples and supporting KB context.

Embedding backend behavior:
- If `OPENAI_API_KEY` is set, the app uses OpenAI embeddings.
- If no key is set, the app uses a local sentence-transformer model (`sentence-transformers/all-MiniLM-L6-v2`).
- If the local model is unavailable, it falls back to TF-IDF as a safety net.

## Run locally

```bash
cd ticket-triage-embeddings
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API

- `GET /health`
- `POST /triage`
  - input: `{"text":"...","top_k":5}`
  - output: category, priority, confidence, reasoning, similar examples, kb context, next steps, drafted reply
- `POST /draft_reply`
  - input: `{"text":"...","kb_top_k":5}`
  - output: `{"drafted_reply":"...","tone":"...","citations":[...]}`
  - requires `OPENAI_API_KEY`

## Test

```bash
pytest -q
```

Tests force TF-IDF mode so they stay fast and do not require downloading a local model.

## Evaluate

```bash
python scripts/evaluate.py
```

Output includes:
- train/test sizes
- overall accuracy for embedding-kNN
- overall accuracy for keyword baseline
- per-category accuracy for embedding-kNN

## DEMO.md (quick demo)

Use these exact demo steps:
1. Start app: `uvicorn app.main:app --reload`
2. Open UI and click `Load Example` or paste ticket.
3. Click `Triage` to show category, priority, similar examples, KB snippets, next steps.
4. If `OPENAI_API_KEY` is set, click `Draft Reply (LLM)`.

Try these two demo tickets:
- `I was charged twice today and our finance team needs an urgent refund.`
- `Mobile app crashes every time we upload an image from iOS 18.`

## Notes on LLM usage

- App works fully without LLM.
- With key set, `/draft_reply` makes one grounded call using ticket + retrieved snippets.
- If key is missing, triage still returns deterministic drafted reply based on snippet titles.

## Production Scaling Notes

For a step-by-step plan to evolve this prototype into a large-scale production system, see:
- `docs/SCALING.md`
