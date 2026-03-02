# Decisions

## 1) Keep data in-repo with JSONL fixtures
- Decision: Store both labeled tickets and KB snippets in committed JSONL files.
- Why: This project is for explainability and walkthrough speed, not scale. Anyone reviewing can open the data directly.
- Tradeoff: No concurrency controls, no update workflows, and no efficient querying at larger sizes.
- Revisit trigger: If fixture size grows enough to slow startup or reviews become hard to track, add a lightweight persistence layer.

## 2) Embedding strategy: OpenAI when available, local model by default otherwise
- Decision: Use OpenAI embeddings when `OPENAI_API_KEY` exists; otherwise use local `sentence-transformers/all-MiniLM-L6-v2`.
- Why: Better semantic quality than TF-IDF while still supporting fully local/no-key execution.
- Tradeoff: First local run may download model weights and startup can be slower.
- Safety fallback: If local model load fails, fallback to TF-IDF so the app still runs.

## 3) One embedding interface for all backends
- Decision: Keep backend selection inside a single `EmbeddingEngine` abstraction.
- Why: The triage pipeline should not care whether vectors came from OpenAI, sentence-transformers, or TF-IDF.
- Tradeoff: Slightly more branching in one class, but easier maintenance overall than scattering backend checks.

## 4) Category prediction via weighted kNN
- Decision: Predict category from top-k nearest labeled tickets with similarity-weighted voting.
- Why: Easy to explain to non-technical reviewers and transparent in UI because nearest examples are shown.
- Tradeoff: Quality is sensitive to fixture wording/distribution.
- Revisit trigger: If this becomes unstable on larger eval sets, consider class prototypes or a small supervised model.

## 5) Priority scoring stays deterministic-first
- Decision: Priority is computed from explicit text heuristics, then blended with neighbor priority signal.
- Why: Priority should be defensible in customer operations; deterministic cues are easier to audit than opaque model-only outputs.
- Tradeoff: Heuristic thresholds are hand-tuned and can be brittle at edges.
- Revisit trigger: If precision/recall targets become strict, calibrate thresholds from held-out ticket sets.

## 6) Retrieval transparency is a first-class output
- Decision: Always return similar examples and top KB snippets as part of triage response.
- Why: Builds trust and gives support agents immediate context for action.
- Tradeoff: Response payload is larger, but still acceptable for this demo scope.

## 7) LLM reply generation is optional and constrained
- Decision: `/draft_reply` only uses LLM when key exists; otherwise deterministic template reply is returned by triage flow.
- Why: Core functionality must work without paid APIs.
- Guardrails:
  - Single LLM call.
  - Prompt grounded in ticket text + retrieved KB snippets only.
  - Strict JSON output contract.
- Tradeoff: LLM quality depends on retrieval quality and model compliance.

## 8) Evaluation script supports backend comparison by flag
- Decision: `scripts/evaluate.py` supports `--backend tfidf|sentence-transformers|both`.
- Why: Makes comparison explicit and avoids editing code between experiments.
- Tradeoff: `sentence-transformers` eval may require network once for model download if cache is missing.
- Current usage note: Keyword baseline remains in output for quick sanity checks.

## 9) Keep tests fast and deterministic
- Decision: Unit tests force TF-IDF backend to avoid model-download/network coupling.
- Why: CI/local test runs should be quick and predictable.
- Tradeoff: Test environment is not identical to production-preferred local backend.
- Mitigation: Evaluation script covers backend comparisons outside unit tests.

## 10) Documentation quality standard: production-style docstrings
- Decision: Add module/class/function docstrings across app and scripts with concise intent and behavior.
- Why: This repo is meant for walkthroughs; documentation quality is part of engineering quality.
- Tradeoff: Minor maintenance overhead when code changes.
- Rule: Keep docstrings specific and operational; avoid generic filler language.
