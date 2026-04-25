# TODOs

Living list of known work items. Ordered by priority. Each item states what it is,
why it matters, and roughly what it takes.

---

## P0 — do before first real webhook lands in prod

### Notion signature bypass
**Where:** `services/ingestion/handlers/notion.py:253-254`

In production, any request with an `X-Prbe-Customer` header passes signature check
(no Notion signature required). The code comment claims the ingestion service
authenticates internal callers, but `main.py` does not. Anyone on the internet
can inject arbitrary Notion-shaped payloads into any tenant.

**Fix:** remove the synthetic-poll fallback branch until we actually build a poll
worker, OR require a separate `NOTION_INTERNAL_POLL_SECRET` env var and HMAC-verify
the synthetic path. ~5 lines.

---

## P1 — fix before onboarding a second tenant

### `/query` is unauthenticated
**Where:** `services/retrieval/main.py`

Client sends `customer_id` in the request body, no validation. A curl with
`customer_id: any-tenant` reads that tenant's data. Fine for solo dogfood, fatal
with real customers.

**Fix:** Bearer API key auth. Hash, look up `customers.api_key_hash`, derive
`customer_id` from the row. Remove `customer_id` from `QueryRequest`. ~30 lines.

### OAuth `state` parameter unsigned
**Where:** `services/ingestion/oauth/routes.py:67, 89`

`state` carries `customer_id` as plaintext. Attacker with their own Slack workspace
can craft a callback with `state=<victim_customer>` and attach their token to
the victim's tenant.

**Fix:** HMAC-sign `state` with `TOKEN_ENCRYPTION_KEY` at install, verify at
callback. ~20 lines.

### `_upsert_document` check-then-act race
**Where:** `services/ingestion/normalizer.py:216-264`

Two concurrent workers processing different events that normalize to the same
`doc_id` can both read `version=N`, both compute `N+1`, one wins the INSERT,
the other silently no-ops via `ON CONFLICT DO NOTHING`. The losing worker's new
content_hash is lost.

**Fix:** Postgres advisory lock per `(customer_id, doc_id)` inside `with_tenant`,
or retry loop that increments version on conflict. ~10 lines.

---

## P2 — operational hygiene

### R2 bucket lifecycle rule
**Where:** Cloudflare R2 dashboard (per bucket) OR `scripts/bootstrap_customer.py`

Raw webhook payloads in R2 currently accumulate forever. Intended as a bounded
hand-off buffer (webhook fast path → worker), not an archive.

**Fix:** add lifecycle rule "delete objects with prefix `raw/` older than 30 days"
to every per-tenant bucket. Either:
- Set manually in Cloudflare UI (30 seconds per bucket)
- Extend `ObjectStore.ensure_bucket` to PUT the lifecycle rule via
  `put_bucket_lifecycle_configuration` so new tenants get it automatically. ~20 lines.

### Retention sweep for `ingestion_events`
**Where:** new cron in `scripts/`

Complements R2 lifecycle — delete `ingestion_events` rows older than the R2
retention window so the table stays bounded. Write `scripts/cron_events_retention.py`
(pattern matches the other crons). Run hourly via Fly cron.

---

## P3 — connector completeness

### GitHub `identify_workspaces`
**Where:** `services/ingestion/handlers/github.py`

Currently returns `[]`. GitHub Apps deliver `installation_id` as a query param on
the post-install redirect, separately from the OAuth `code`. The OAuth callback
route doesn't pass this through to `identify_workspaces` today.

**Fix:** extend the callback to capture `installation_id` from the query string
and pass it to a GitHub-specific identify method, OR have `identify_workspaces`
call `GET /user/installations` with the user token to list installs.

**Workaround today:** workaround documented — see `scripts/github_seed_token.py`.
Operator grabs `installation_id` from the install redirect URL and seeds the
token row manually. `customer_source_mapping` is written at the same time so
live webhooks route correctly without relying on `single_customer_fallback`.
The real fix (auto-capturing `installation_id` in the OAuth callback) is
still unstarted.

### Sentry `identify_workspaces`
**Where:** `services/ingestion/handlers/sentry.py`

Currently returns `[]`. Sentry internal integrations don't go through standard
OAuth — the `installation.created` webhook carries organization info.

**Fix:** treat `installation.created` as a special case in `parse_webhook_event`
that writes the mapping directly (bypassing the normal doc-producing path).

**Workaround today:** same as GitHub — `single_customer_fallback` on first webhook.

---

## P4 — phase 1 gates

Items the design doc explicitly defers to Phase 1:

- Enable ACL enforcement in `services/retrieval/acl.py` (flip `ENFORCE_ACL`
  and implement `_filter_with_acl` against `acl_snapshots`)
- Log redaction helper (`redact_for_logs`) — strip prompt + source content
  before logs land in third-party storage
- Secrets rotation machinery (Fernet key, OAuth tokens, webhook signing)
- Prompt injection defense on the Haiku entity extractor
- Webhook-reactive ACL updates (member_left_channel, user_deactivated)
- Nightly ACL reconciliation sweep

---

## P5 — Phase 1 retrieval

### Event-anchor index

Agents asking "since the auth refactor" or "after we shipped v2" hit a wall:
the Haiku temporal extractor returns `unresolvable_anchor` and we fall back
to LATEST with `applied_temporal.source = "extraction_failed"`. The agent
sees the error and can decide what to do. No automatic resolution today.

**Fix:** define first-class "event" entities ingested into `graph_nodes`:
- GitHub releases (`/releases`) → `Release` entity with `published_at`
- Linear milestone-tagged issues → `Release` entity with `completed_at`
- Notion "Decision" DB pages → `Decision` entity with frontmatter date
- Slack `#releases` posts → `Release` entity

Each gets a `graph_nodes` row with `label IN ('Release','Decision','Migration')`
and `properties.date`. Anchor resolution becomes a single SQL hit on
`graph_nodes` keyed by canonical_id or property text match. Resolution
plugs into `services/retrieval/temporal.py:resolve_temporal()` as a new
branch when `unresolvable_anchor` is set.

**Scope:** ~400 LOC across all 5 connectors + dashboard tagging UX.
Pays back in: temporal resolution accuracy, Phase 2 verification ("did
this PR conflict with anything since release v2"), and "show me the
decision history" queries.

Discussed: 2026-04-24. Skipped from current PR because event extraction
needs per-connector ontology work + customer-specific tagging conventions.

### In-memory router cache (LRU)

We dropped the Postgres-backed `query_cache` in migration 0006 because at
single-tenant scale the hit rate didn't pay back the schema + cron sweep
overhead. Add a per-process LRU when query volume justifies it:
- Scope: `functools.lru_cache(maxsize=512)` or a TTL dict in
  `services/retrieval/router.py` keyed by `(customer_id, query, prompt_version)`.
- Triggers when `/query` p95 starts pressing the 2s SLO with Haiku as the
  dominant chunk OR monthly Anthropic spend on the router crosses ~$50/customer.
- ~30 LOC. No DB schema work.

---

## Done recently (clear periodically)

- Tier 3 end-to-end smoke test passes against local Postgres + MinIO
- Five connectors implemented with the shared `Connector` contract
- Full retrieval pipeline: router + vector + BM25 + graph + fusion + dedup
- OAuth install/callback routes wired
- `customer_source_mapping` + `identify_workspaces` + `extract_external_id_from_payload`
  so webhooks route without `X-Prbe-Customer` headers
- `ProxyHeadersMiddleware` so OAuth redirect_uri resolves to `https://` behind Fly
- Fly + CI/CD configs ready (three Dockerfiles, three fly.tomls, three GH Actions)
- `/review` ran; dead imports + worker subquery + test-fixture import caching fixed
