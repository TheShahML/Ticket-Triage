# Scaling Guide: From Prototype to Production

This project is intentionally small. This guide describes how to scale it for larger workloads while preserving reliability, transparency, and maintainability.

## Current State (Prototype)
- Data stored in local JSONL files.
- In-memory embedding index built at startup.
- Single FastAPI process serving API + static UI.
- No durable storage, no async queueing, no rate limiting.

This is ideal for demo, testing, and small internal pilots, not high-volume production.

## Target Production Outcomes
1. Handle high ticket volume with predictable latency.
2. Keep model outputs observable and auditable.
3. Maintain high uptime and safe deploy/rollback paths.
4. Support continuous data refresh and model quality monitoring.

## Phase 1: Productionize Data and Storage
### 1) Move source data to managed storage
- Replace JSONL fixtures with a relational source of truth (e.g., Postgres).
- Use tables for:
  - tickets (raw text + metadata)
  - labels (category, priority, reviewer)
  - KB snippets (content, version, status)
- Keep immutable snapshots for reproducible model evaluation.

### 2) Add durable event ingestion
- Ingest new tickets via queue or event stream (e.g., SQS/Kafka/PubSub).
- Persist raw ticket events before inference.
- Add idempotency keys so retries do not duplicate work.

### 3) Introduce schema/version controls
- Version label schema and KB schema.
- Add migration process and backward compatibility checks.

## Phase 2: Retrieval and Embedding Scale
### 1) Move from in-memory vectors to a vector index
- Use a managed vector store or self-hosted ANN index.
- Store two indexes:
  - labeled-ticket embeddings (classification neighbors)
  - KB embeddings (context retrieval)
- Support metadata filters (team, product area, language, recency).

### 2) Build embedding pipelines
- Add batch + incremental embedding jobs.
- Trigger re-embedding on:
  - new/updated tickets
  - KB edits
  - model upgrades
- Keep embedding model version and index version alongside each vector.

### 3) Latency and cost controls
- Cache query embeddings where possible.
- Batch embedding requests to reduce API calls.
- Add retrieval timeouts and fallback behavior.

## Phase 3: Inference Service Hardening
### 1) Split responsibilities into services
- `triage-api`: request validation + orchestration.
- `retrieval-service`: nearest-neighbor + KB retrieval.
- `reply-service`: optional LLM draft generation.
- `feature-store` or metadata service for operational context.

### 2) Add asynchronous execution where needed
- Keep `/triage` synchronous for fast response path.
- Run heavier workflows async (bulk backfills, offline scoring, reporting).
- Add job status endpoints for long-running tasks.

### 3) Add resiliency patterns
- Timeouts, retries with backoff, and circuit breakers.
- Graceful degradation:
  - if LLM unavailable -> deterministic reply
  - if vector search degraded -> fallback strategy + clear status

## Phase 4: Quality, Evaluation, and Governance
### 1) Expand evaluation framework
- Build offline benchmark datasets from real labeled tickets.
- Track:
  - overall accuracy
  - per-category precision/recall
  - priority calibration quality
  - retrieval relevance quality
- Evaluate by segment (region, product line, customer tier).

### 2) Human-in-the-loop feedback loop
- Allow agents to correct predictions.
- Store corrections as training/evaluation signals.
- Route low-confidence cases to manual triage.

### 3) Model governance
- Version models, prompts, thresholds, and retrieval config.
- Record full inference trace for audit:
  - input hash
  - model/index version
  - retrieved neighbors/snippets
  - final outputs

## Phase 5: Security and Compliance
### 1) Data protection
- Encrypt data at rest and in transit.
- Redact or tokenize PII before storage/indexing where possible.

### 2) Access control
- Service-to-service auth (mTLS/JWT).
- RBAC for admin actions (reindex, config updates, eval reports).

### 3) Compliance readiness
- Define retention/deletion policies.
- Add audit logs for admin and model configuration changes.

## Phase 6: Observability and Operations
### 1) Metrics
Track at minimum:
- request volume, latency (P50/P95/P99), error rate
- retrieval latency and hit quality metrics
- LLM call success/failure/cost
- confidence distribution and manual override rates

### 2) Logging and tracing
- Structured logs with request IDs.
- Distributed traces across API, retrieval, and LLM services.
- Alerting on SLO breaches and drift signals.

### 3) Runbooks and SLOs
- Define SLOs for triage response and draft reply endpoints.
- Create incident runbooks for degraded retrieval/LLM outages.

## Suggested Target Architecture
1. API Gateway -> FastAPI Triage API
2. Triage API -> Vector DB (tickets + KB)
3. Triage API -> Feature/Metadata store
4. Triage API -> Optional Reply Service (LLM)
5. Async workers -> embedding/reindex pipelines
6. Postgres + object storage for source-of-truth and snapshots
7. Monitoring stack for metrics, logs, traces, and alerting

## Migration Plan (Practical Sequence)
1. Add Postgres + migration of JSONL fixtures.
2. Add vector index while keeping current in-memory path as fallback.
3. Add offline eval dashboard and confidence-based routing.
4. Add async ingestion/reindex workers.
5. Split reply generation into separate service.
6. Introduce SLOs, runbooks, and production alerts.
7. Decommission prototype-only paths after stabilization.

## What to Keep from This Prototype
Even at scale, keep these principles:
- Explainability: return nearest examples and KB evidence.
- Fallbacks: deterministic behavior when external dependencies fail.
- Simplicity first: scale complexity only where bottlenecks are measured.
