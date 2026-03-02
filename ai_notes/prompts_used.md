# Prompts Used

## 2026-02-22 - Initial project brief
- Prompt intent: Build a very small, explainable ticket-triage app with embeddings + kNN, no DB/vector DB, FastAPI backend, single static UI.
- What I asked for:
  - Full repo scaffold with strict file layout.
  - Local deterministic fallback when `OPENAI_API_KEY` is missing.
  - Transparent output fields (similar examples + KB context + reasoning).
  - Tests + evaluation script + recruiter-friendly README.
  - Production-style docstrings on core modules/functions so the codebase is explainable in review.
- What I kept:
  - Core architecture and endpoint shape.
  - JSONL fixture approach for labeled tickets and KB snippets.
  - Deterministic reply path and optional LLM draft path.
- What I edited manually afterward:
  - Dependency pins were relaxed to avoid install failures on older local Python/pip setups.
  - Added a safer JSON parse fallback in LLM reply generation.
  - Fixed script import path so `python scripts/evaluate.py` works from repo root.

## 2026-02-22 - UI cleanup pass
- Prompt intent: Keep frontend modern but simple enough for a 10-15 minute walkthrough.
- What I asked for:
  - Tailwind-only static page (no React).
  - Clear input/output layout, expandable sections, load-example dropdown.
  - Conditional LLM button based on backend health response.
- What I kept:
  - Card layout and neutral styling.
  - Basic client-side fetch flow and status messages.
- What I adjusted:
  - Tightened labels and spacing for readability.
  - Kept JS logic inline for transparency instead of splitting into extra files.

## 2026-02-23 - Test/eval pass
- Prompt intent: Ensure project can be demonstrated without network/LLM.
- What I asked for:
  - Fast golden tests for category/priority behavior.
  - Tiny evaluation script with train/test split and keyword baseline.
- What I kept:
  - Three local tests using TF-IDF fallback.
  - Evaluation output with overall accuracy + per-category accuracy.
- Notes:
  - On the small fixture set and fixed split seed, keyword baseline currently beats kNN fallback. I left this as-is and documented it instead of overfitting to make numbers look better.

## 2026-02-23 - Local embedding model switch
- Prompt intent: Replace TF-IDF as the default local embedding path because quality was too weak.
- What I asked for:
  - Use a real local embedding model (`sentence-transformers/all-MiniLM-L6-v2`) when no OpenAI key is present.
  - Keep a non-breaking fallback path so the app still starts if local model load fails.
  - Update docs and notes so behavior is explicit for reviewers.
- What I kept:
  - OpenAI embeddings remain the top-priority path when API key exists.
  - TF-IDF still used in tests/eval to keep CI and offline runs deterministic/fast.
- What changed in practice:
  - `app/embed.py` now tries sentence-transformers first in local mode.
  - Added `LOCAL_EMBED_MODEL` to `.env.example`.
  - Added `sentence-transformers` dependency and README notes.

## 2026-02-23 - Evaluation script parity request
- Prompt intent: Compare both local backends in one script without rewriting code each time.
- What I asked for:
  - Add CLI flags so eval can run TF-IDF only, sentence-transformers only, or both.
  - Keep keyword baseline in output for quick sanity comparison.
  - Keep the output readable for recruiter demo notes.
- What changed:
  - `scripts/evaluate.py` now accepts `--backend tfidf|sentence-transformers|both`.
  - Default is `both` so side-by-side backend comparison is one command.
  - Eval prints a separate kNN accuracy block per selected backend.
- Practical note:
  - sentence-transformers mode may require model download on first run.

## 2026-02-24 - Production-grade scaling strategy discussion
- Prompt intent: Document a realistic path from this prototype to a production-ready system.
- What I asked for:
  - A phased scaling plan covering data, retrieval, model serving, reliability, and governance.
  - Practical migration order instead of a “big bang” redesign.
  - Clear operational concerns (latency, observability, security, fallback behavior).
- What changed:
  - Added `docs/SCALING.md` with end-to-end productionization strategy.
  - Linked scaling guidance from `README.md` under a dedicated section.
- Why this matters:
  - Helps reviewers see system-thinking beyond prototype implementation.
  - Shows explicit tradeoffs and rollout sequencing for real-world adoption.
