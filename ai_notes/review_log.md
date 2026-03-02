# Review Log

## 2026-02-23 - Project scaffold and API contract check
- Reviewed:
  - Required repo structure and file presence.
  - Endpoint contract for `GET /health`, `POST /triage`, `POST /draft_reply`.
- Result:
  - All required directories/files created.
  - Endpoint shapes match the requested schema and response fields.
- Notes:
  - Static UI served from FastAPI root to keep demo flow simple.

## 2026-02-23 - Evaluation script startup fix
- Command:
  - `python3 scripts/evaluate.py`
- Initial issue:
  - Script failed with `ModuleNotFoundError: No module named 'app'` when run directly.
- Fix applied:
  - Added repo root to `sys.path` in `scripts/evaluate.py`.
- Re-check:
  - Script runs and prints train/test metrics.

## 2026-02-23 - Dependency compatibility pass
- Command:
  - `pip3 install -r requirements.txt`
- Initial issue:
  - Strict version pin on `numpy` failed in local environment.
- Fix applied:
  - Replaced strict pins with compatible ranges in `requirements.txt`.
- Re-check:
  - Dependency installation completed successfully.

## 2026-02-23 - Unit test validation
- Command:
  - `python3 -m pytest -q`
- Result:
  - Pass (`3 passed`).
- Scope:
  - Category and priority behavior for core triage flow.
  - Tests intentionally use TF-IDF backend for deterministic runtime.

## 2026-02-23 - Runtime smoke checks
- Checked:
  - Direct invocation path (`build_index` + `triage_ticket`) and response composition.
- Result:
  - Output includes category, priority, confidence, similar examples, KB context, next steps, drafted reply.
- Environment note:
  - Full Uvicorn server bind/reload was constrained in this sandbox.

## 2026-02-23 - Local embedding backend upgrade
- Change reviewed:
  - Local default moved from TF-IDF to sentence-transformers (`all-MiniLM-L6-v2`).
  - TF-IDF retained as fallback if local model load fails.
- Files touched:
  - `app/embed.py`, `app/index.py`, `requirements.txt`, `.env.example`, `README.md`.
- Re-check:
  - Core tests still pass in forced TF-IDF mode.

## 2026-02-23 - Eval backend flag support
- Change reviewed:
  - Added backend selection flag in eval script.
- Command examples:
  - `python3 scripts/evaluate.py --backend tfidf`
  - `python3 scripts/evaluate.py --backend sentence-transformers`
  - `python3 scripts/evaluate.py --backend both`
- Validation:
  - `--backend tfidf` runs and prints expected metrics.
  - sentence-transformers mode depends on model availability/network on first run.

## 2026-02-23 - Documentation quality pass
- Change reviewed:
  - Added production-style docstrings across app modules, test module, and evaluation script.
- Scope:
  - Module-level docstrings.
  - Public classes/functions/endpoints.
  - Key helper functions in triage/eval flow.
- Validation:
  - `python3 -m pytest -q` still passes after documentation-only edits.

## Current Risks / Follow-up
- Metric realism:
  - On the tiny fixture set and fixed split, keyword baseline can still outperform kNN in TF-IDF mode.
- Environment coupling:
  - sentence-transformers evaluation may require first-run model download.
- Next recommended pass:
  - Compare `tfidf` vs `sentence-transformers` vs keyword baseline on the same split in a network-enabled run.
  - Expand labeled fixture diversity before attempting architecture changes.
  - Add one additional split/seed to reduce variance in reported metrics.
