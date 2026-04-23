# PRBE Roadmap

Living doc. Source of truth for "what's built vs what's next." Keep in sync with reality as work lands.

Full design context: [`phase0-design.md`](phase0-design.md).

---

## Phase 0 — Knowledge Layer Infrastructure

Infra-only. Zero standalone customer value. Foundation for Phase 1+. Built against thesis confidence, not a named pilot.

### Tier 1 — Shared plumbing (foundation)

Blocks everything else. Strictly sequential.

- [ ] `shared/config.py` — Pydantic Settings loading `.env.local`
- [ ] `shared/db.py` — async asyncpg pool + per-request `SET app.current_customer_id` for RLS
- [ ] `shared/storage.py` — R2 client (boto3 `<1.36.0`), put/get/create-bucket helpers
- [ ] `shared/embeddings.py` — OpenAI embedder (batched, native retries, recursive half-split on partial fail, writes `failed_chunks`)
- [ ] `shared/exceptions.py` — named exception registry (25+ classes from Error & Rescue Registry)
- [ ] `shared/encryption.py` — Fernet wrap for OAuth tokens at rest in `integration_tokens`

### Tier 2 — Ingestion core

Handler contract + async worker loop. Blocks per-source handlers.

- [ ] `services/ingestion/chunker.py` — naive token-based chunker (`tiktoken`)
- [ ] `services/ingestion/graph_writer.py` — upsert `graph_nodes` + `graph_edges`
- [ ] `services/ingestion/normalizer.py` — dispatch to correct handler by `source_system`
- [ ] `services/ingestion/handlers/base.py` — `abc.ABC` handler contract (6 methods)
- [ ] `services/ingestion/worker.py` — `SELECT ... FOR UPDATE SKIP LOCKED` drain loop, heartbeat, DLQ
- [ ] `services/ingestion/main.py` — FastAPI app, `/webhooks/{source}`, `/health`, OAuth callback routes

### Tier 3 — First end-to-end slice

Prove the pattern before parallelizing. Pick one source (Slack — simplest), ship it all the way through.

- [ ] `services/ingestion/handlers/slack.py` — full handler contract
- [ ] `services/retrieval/main.py` (minimal) — `/query` with vector-only retrieval (no BM25, graph, or Haiku router yet)
- [ ] End-to-end smoke test — post signed fixture webhook → worker processes → `/query` returns chunk

**Gate:** smoke passes → architectural pattern proven → parallelize Tier 4.

### Tier 4 — Parallel handlers (4 more sources)

Four Claude subagents in parallel, one per source. Integration review by main agent as each lands.

- [ ] `services/ingestion/handlers/linear.py`
- [ ] `services/ingestion/handlers/github.py` — CODEOWNERS parsing is the hard part
- [ ] `services/ingestion/handlers/notion.py` — ACL extraction is the hard part
- [ ] `services/ingestion/handlers/sentry.py`

Each handler passes its `tests/handlers/test_{source}.py` before merge.

### Tier 5 — Retrieval service (full shape)

- [ ] `services/retrieval/router.py` — Haiku entity extraction + query expansion + 1h cache in `query_cache`
- [ ] `services/retrieval/retrievers/vector.py` — pgvector HNSW, top_k = 50
- [ ] `services/retrieval/retrievers/bm25.py` — Postgres `ts_rank_cd` + GIN, top_k = 50
- [ ] `services/retrieval/retrievers/graph.py` — SQL 1-hop traversal, top_k = 20
- [ ] `services/retrieval/fusion.py` — RRF `k=60`, doc-level collapse, deterministic tie-break
- [ ] `services/retrieval/dedup.py` — per-doc + cross-doc cosine > 0.95
- [ ] `services/retrieval/acl.py` — pass-through stub in Phase 0 (real filter lives in Phase 1 deferred list)

### Tier 6 — Operational scripts

- [ ] `scripts/bootstrap_customer.py` — insert customer row, create R2 bucket, issue `/query` API key, generate 5 OAuth install URLs
- [ ] `scripts/seed_synthetic.py` — 100+ synthetic documents for integration testing
- [ ] `scripts/backfill.py` — historical paginated backfill per (customer, source)

### Tier 7 — OAuth framework

Required before any real customer can connect sources. Webhooks work without it (source pushes to you); authenticated API calls (`fetch_supplementary`, backfill) need tokens.

- [ ] `services/ingestion/oauth/base.py` — generic install-URL generation + callback handler
- [ ] `services/ingestion/oauth/slack.py`
- [ ] `services/ingestion/oauth/linear.py`
- [ ] `services/ingestion/oauth/github.py`
- [ ] `services/ingestion/oauth/notion.py`
- [ ] `services/ingestion/oauth/sentry.py`
- [ ] Token refresh cron (every 15 min, refreshes anything expiring within the hour)
- [ ] `integration_tokens.last_refresh_error` populated on refresh failure

### Tier 8 — Observability

- [ ] `structlog` configured everywhere, trace IDs propagated from `X-Trace-Id`
- [ ] OpenTelemetry → Grafana Cloud Free (10K series, 14d retention)
- [ ] Metrics: ingestion lag, chunks/min, `/query` p95 per stage, API cost/customer
- [ ] Alert: `/query` p95 > 5s (retrieval broken)
- [ ] Alert: webhook 5xx rate > 5% over 5min window (handler crashing)

### Tier 9 — Cron jobs

- [ ] Token refresh (every 15 min) — see Tier 7
- [ ] Stuck-queue reclaim (every 2 min) — resets `ingestion_queue` rows with `heartbeat_at > 5 min ago`
- [ ] `query_cache` expiry sweep (hourly) — delete expired router cache entries

### Tier 10 — Tests + LLM evals

- [ ] Handler unit tests against committed fixtures (5 files, one per source)
- [ ] Integration test: 20 sample queries with expected ranking quality
- [ ] Idempotency test: replay same signed webhook 10x → 1 doc version
- [ ] Multi-tenant isolation test: customer A cannot see customer B's chunks (RLS + explicit filter, defense in depth)
- [ ] LLM eval: entity extractor, 20 hand-labeled docs, ≥80% match on entity_type + canonical_id
- [ ] LLM eval: query expansion, 10 hand-labeled queries, cosine drift check
- [ ] CI wires eval jobs to run weekly against staging, alert on regression

### Tier 11 — Deploy configs + CI/CD

- [ ] `services/ingestion/Dockerfile`
- [ ] `services/retrieval/Dockerfile`
- [ ] `fly.ingestion.toml`
- [ ] `fly.retrieval.toml`
- [ ] `.github/workflows/tests.yml` (on PR)
- [ ] `.github/workflows/deploy.yml` (on main merge)
- [ ] `.github/workflows/evals.yml` (weekly)
- [ ] Fly secrets set: `DATABASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `R2_*`, OAuth client secrets, `TOKEN_ENCRYPTION_KEY`
- [ ] Migration ordering: `release_command = "alembic upgrade head"` in both fly.tomls

### Tier 12 — Dogfood

- [ ] Connect founder Slack + GitHub to staging
- [ ] Validate real webhook payloads match committed fixtures
- [ ] Run 20 real queries against own data, verify ranking quality
- [ ] Measure `/query` p95 on warm + cold Haiku cache with real corpus
- [ ] Document any schema drift → fix in-place

### Tier 13 — Onboarding completion (close Phase 0 gap vs design)

Onboarding scaffolding landed in Tiers 6-7, but the historical-fetch step is a stub on every connector and the OAuth callback does not trigger backfill as the design specifies (`docs/phase0-design.md:1162`). Close these gaps before any Phase 1 pilot lands.

**Current state today:**
- `scripts/bootstrap_customer.py` provisions customer row + R2 bucket + prints 5 OAuth install URLs. Works.
- `services/ingestion/oauth/routes.py` handles `/oauth/{source}/install` and `/oauth/{source}/callback`, persists encrypted tokens. Works, but does NOT trigger backfill — just returns an HTML "Connected" page.
- `scripts/backfill.py` drains a connector's `backfill()` generator into R2 + `ingestion_queue`, then the normal worker loop picks up the queue rows through the same normalize → chunk → embed → upsert path as live webhooks. Works — but every handler still inherits `base.py`'s default `NotSupportedByConnector` raise, so the script short-circuits on every source.

**Gap-closing work:**

- [ ] **OAuth callback auto-triggers backfill.** `oauth/routes.py:90` after `save_token(token)`: enqueue a `backfill_state` row with `status=PENDING` on successful exchange; a worker drains it. Do NOT fire-and-forget via `asyncio.create_task` — Fly restarts drop in-flight tasks.
- [ ] **Per-source `backfill()` implementations.** `handlers/base.py:164-179` defines the generator contract; every handler currently inherits the default raise. Implement in priority order: Slack (`conversations.history`, cursor = `oldest` ts) → GitHub (REST pagination, `since=` param) → Linear (GraphQL cursor) → Notion (search + pagination) → Sentry (issues list, cursor).
- [ ] **Rate-limit handling in backfill generators.** Per-source 429 backoff — not in shared plumbing today. Add `shared/http.py` retry wrapper with exponential backoff respecting `Retry-After`, reused by all handlers' backfill paths.
- [ ] **HNSW tuning during bulk ingest.** `SET LOCAL hnsw.ef_construction = 200` in the worker's per-transaction session when the queue row is tagged `event_type='backfill'`. Defaults degrade index quality at scale (design doc line 763).
- [ ] **Sign OAuth `state` param.** `oauth/routes.py:13` acknowledges plain `customer_id` is a Phase 0 shortcut. HMAC-sign with a new `OAUTH_STATE_SECRET` Fly secret before Phase 1 — prevents CSRF on the callback.
- [ ] **Multi-source parallel driver.** `scripts/backfill_all.py --customer <id>` wrapping 5 parallel subprocesses over `scripts/backfill.py`. Defer per-customer parallelism until ≥10 tenants.
- [ ] **Operator progress surface.** `GET /admin/backfill_state?customer_id=X` returning `[{source, status, started_at, last_progress_at, events_processed}, ...]`. Unblocks "is the onboarding stuck?" without grepping structlog.
- [ ] **Fix `bootstrap_customer.py` API-key footgun.** Current `ON CONFLICT DO UPDATE` silently rotates the API key on re-run. Either error out with "customer exists — use `--rotate-key` to overwrite" or only upsert `display_name`.

**Gate:** all 5 handlers implement `backfill()`, OAuth callback wires into the queue, and one end-to-end test (bootstrap → OAuth click-through stub → ≥100 historical events processed → `/query` returns chunks from backfilled data) passes. Without this, Phase 1 pilot onboarding regresses to "operator babysits the CLI for an hour."

### Phase 0 Success Criteria

Copy of the design doc's criteria. Phase 0 shippable when all hold:

- [ ] Direct webhooks for all 5 sources accepted + persisted (raw + normalized) with signature verification
- [ ] Idempotency: replay same webhook 10x → exactly one document version
- [ ] `/query` returns ranked + deduplicated chunks (ACL pass-through in Phase 0)
- [ ] `/query` p95 < 2s warm / < 3s cold against 10K-chunk corpus
- [ ] ACL ingestion test passes (capture-but-not-enforce)
- [ ] `/health` checks all external deps
- [ ] `scripts/bootstrap_customer.py` provisions new tenant in <30 min operator time
- [ ] `scripts/seed_synthetic.py` loads 100+ doc corpus
- [ ] Per-stage retrieval timing emitted
- [ ] Per-customer API cost attribution exported
- [ ] Ingestion lag p95 <30s under normal load
- [ ] README covers `/query` API + bootstrap flow

---

## Phase 0.5 — Selective Wiki Compilation

Build on Phase 0 schema with a separate `prbe-knowledge-compile` Fly app. One wiki type only: `wiki.service_card`.

Karpathy-style LLM wiki pattern applied to **stable entities only** (services, repos, features) — not to high-volume streaming content (Slack messages, Sentry events) where the compilation cost doesn't pay back.

### Scope

- [ ] Daily compile worker (cron at 00:00 UTC, also manual-admin trigger)
- [ ] Stage 1 — collect candidates from `ingestion_events` → group by wiki page doc_id
- [ ] Stage 2 — Haiku triage (parallel, ~$0.0001/page) → filter to pages needing update
- [ ] Stage 3+4 — Sonnet full regeneration (sequential per customer with Postgres advisory lock), full-regen not incremental (avoids drift)
- [ ] ACL propagation: compiled page = intersection of source ACLs
- [ ] `COMPILED_FROM` edges in graph
- [ ] `compile_trigger` field populated (scheduled | manual | normalizer_reprocess | source_update)
- [ ] Cost ceiling: < $100/month per customer (Sonnet + Haiku combined), alert on breach
- [ ] Freshness SLO: <24h lag between source update and wiki update

### Source expansion (parallel workstream)

- [ ] **Langfuse** handler — LLM observability (first priority: Reevo's GTM co-pilot produces its real bug signal in LLM traces, not stack traces)

---

## Phase 1 — Ticket Enrichment (First Revenue)

The smallest thing a customer actually pays for. Wire the Sentry→Linear→agent chain that already exists at target customers.

### Scope

- [ ] Webhook handler on Linear issue-create
- [ ] For each auto-created ticket: pull related context
  - Services involved (from `FIRES_IN` edges on the error group)
  - Owners + team conventions (from `OWNS` + Notion wiki pages)
  - Related recent PRs (from graph traversal)
  - Linked design docs (Notion)
  - Recent decisions relevant to the symptom (Slack + Notion)
- [ ] Compose enrichment block in agent-friendly format
- [ ] Write block back to Linear ticket description before agent picks up
- [ ] **Flip ACL enforcement ON** in retrieval `acl.py` (data is already captured from Phase 0)
- [ ] MCP server for coding agents: `get_ticket_context(ticket_id)` tool
- [ ] Design partner: Reevo (or equivalent AI-product company with Devin + Sentry + Linear)

### Source expansion

- [ ] **Datadog** handler — APM/infra context for "did this fix change memory/CPU/latency?"
- [ ] **PostHog** handler — user-impact context for prioritization

---

## Phase 2 — Verification Pipeline (The Moat)

The defensible product. Glean/Resolve commoditize retrieval. "This fix doesn't conflict with PR #1234, doesn't regress bug-456, doesn't break the in-flight auth rewrite" is what nobody else has.

### Scope

- [ ] MCP server expanded: `verify_fix(diff)`, `explain_verdict(verification_id)` tools
- [ ] Conflict detection: diff intersects with any `status=open` PR touching same paths
- [ ] Regression detection: diff touches code paths with recent `FIXES` edges (the bug you're about to re-introduce)
- [ ] Roadmap-break detection: diff touches planned feature areas flagged in Notion
- [ ] Convention violation: diff violates team style/architecture notes in wiki
- [ ] Structured verdict output: `{merge_safe: bool, reasons: [{category, evidence_doc_ids, explanation}]}`
- [ ] GitHub PR check integration — verdict as PR status check
- [ ] **Pylon** handler — customer-support context for user-facing regressions
- [ ] **SOC 2 Type II** certification (blocks enterprise customers, 6-9mo calendar clock)
- [ ] Audit log populated for all MCP tool calls (from `audit_log` table, added in Phase 0)

### Source expansion

- [ ] **Pylon** handler — customer support

---

## Phase 3 — Closed Loop Autonomous Debugging

Verification becomes gate, not advisory. The product shifts from "human-reviewed agent" to "auto-merged agent, human reviews summary in morning."

### Scope

- [ ] GitHub Actions gate: if `verify_fix` passes → auto-approve, else block
- [ ] Auto-merge behind feature flag on verification pass
- [ ] Auto-route back to agent with specific conflict/regression context on verification fail → agent iterates
- [ ] Post-merge Sentry watch: did the error fingerprint actually resolve?
- [ ] Auto-rollback if post-merge Sentry spikes
- [ ] Per-customer "overnight report": N PRs auto-merged, N rolled back, N still pending human review
- [ ] `FixArtifact` + `VerificationResult` graph nodes populated (reserved in Phase 0 schema)

---

## Phase 4+ — Full Stack Vision

Expands PRBE from "orchestration layer on top of existing tools" to "replace the stack." Directionally ambitious, calendar-distant.

### Replace observability

- [ ] AI-native error tracking (alternative to Sentry for greenfield AI companies, not drop-in replacement for existing Sentry customers)
- [ ] LLM-specific observability built in (prompt regressions, eval drift, tool-call hallucinations, context overflow)

### On-device agent integration

The pre-pivot Probe differentiator. Carries forward as a Phase 4 capability unique to PRBE.

- [ ] Electron + Swift SDK clients maintained (they already exist in `prbe-electron-sdk`, `prbe-swift-sdk`)
- [ ] Customer-side context flows into investigation pipeline
- [ ] Privacy / PII redaction at the client boundary
- [ ] ReAct loop in `prbe-middleware` orchestrates server reasoning + local tool execution

### 15-minute onboarding

- [ ] One-click source connection for Slack + Linear + GitHub + Notion + Sentry
- [ ] Auto-provisioned R2 bucket + OAuth flow chaining
- [ ] Pre-seeded tribal knowledge surfacing within first 10 min post-connect

---

## Cross-Cutting Workstreams

Not phase-bound. Run in parallel throughout.

### Compliance

- [ ] SOC 2 Type II (required before enterprise pricing unlocks, plan for Phase 2 window)
- [ ] GDPR DPA template
- [ ] Customer data export / deletion runbook
- [ ] HIPAA eligibility assessment (if healthcare-adjacent customer shows up)

### Security

- [ ] Secrets rotation automation (webhook signing secrets, OAuth tokens, customer API keys)
- [ ] Log redaction helper (`redact_for_logs()`) — LLM prompts + source content must not land in third-party log storage
- [ ] Prompt injection defense on Haiku entity extractor

### Commercial

- [ ] Design-partner MSA template with grandfather clause + case-study rights
- [ ] Commercial pricing tiers live (see roadmap notes in private strategy docs — Phase 1 mid-market: $4-8K/mo, enterprise: scales up)
- [ ] Reference architecture doc for security reviews

---

## Deferred from Phase 0 → Phase 1+

Items explicitly punted on in the Phase 0 design. Pre-seeded as TODOs when repo scaffolded. Ordered by priority.

### P1 — must resolve before Phase 1 ships

- [ ] Enable real ACL enforcement in retrieval (`acl.py` filter uses data already captured)

### P2 — resolve early

- [ ] Log redaction helper (`redact_for_logs()`)
- [ ] Secrets rotation machinery
- [ ] Prompt injection defense on entity extractor
- [ ] ACL drift documentation for enterprise security reviews

### P3 — can wait

- [ ] Nightly ACL reconciliation sweep (complement to webhook-reactive updates)
- [ ] Webhook-reactive ACL updates (member_left_channel, user_deactivated, etc.)
- [ ] Rollback procedure runbook (Neon branch + Fly prior-version redeploy)
- [ ] Source-vendor payload drift detection (weekly CI job posts real events, compares shape)
- [ ] OAuth public distribution (Slack app review, GitHub App publication, Notion public integration)
- [ ] Phase 2 verification feasibility spike (validate LLM judgment + context + diff → correct verdict against own PRBE git history)
