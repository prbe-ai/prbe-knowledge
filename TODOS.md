# TODOs

Living list of known work items. Ordered by priority. Each item states what it is,
why it matters, and roughly what it takes.

---

## Transcript extraction — from the 2026-09-23 spend review (docs/plans/extraction-spend-plan.md)

### Per-segment extraction cache (Phase 1 of the spend plan)
**Where:** `engine/shared/claude_code_extraction.py` (`extract_units_from_session`, `_extract_one`).

**What:** cache each segment's grounded `UnitBundle` in R2 keyed on
`(customer, session, sha256(rendered segment text), agent, prompt+schema version)`; skip the
model call on a hit. **Why:** a re-completed session re-reads every unchanged segment.
**Context:** measured 2026-09-23, re-completions are 73 of 5,251 v2 sessions (30 d) and 461 of
5,712 passes, so this is worth under $20/month once the sweep loop is fixed. The `(part i of n)`
prompt text must stay out of the key; the cap keeps the LAST 16 segments so a capped session's
earliest segments change identity (misses, not corruption); never cache a non-authoritative
result. `claude_code_extraction.pass` logs per-segment hashes, which gives the hit rate before
anything is built. **Effort:** M. **Priority:** P3.
**Depends on:** the sweep-loop fix shipped and 1 week of pass logs showing repeated segment
hashes >= 25 % of segment calls.

### Jev tail screen (Phase 2 of the spend plan), shadow first
**Where:** `engine/shared/claude_code_extraction.py`, `engine/retrieval/agent/jev.py`.

**What:** before mining a re-completed session's newest segment, ask Jev (yes/no + probability)
whether the tail holds a decision, directive, Q&A or code change; shadow-only until the miss
rate is measured. **Why:** the residual call after the cache is the changed tail, and many tails
hold nothing. **Context:** ~32k-token request cap (`docs/jev-contract.md`), so screen the tail
truncated to its last 100k chars and log the truncation; run extraction anyway in shadow and log
Jev's probability next to `units`; nothing is skipped until >= 500 tails show the miss rate; a
skip is its own outcome, never "mined, found nothing" (the
`claude_code.extraction_disabled` branch of `normalize` is the model). The key is hand-patched into `engine-secrets`; `sync-secrets` full-replaces a
Secret. **Effort:** M. **Priority:** P3. **Depends on:** the per-segment cache, and its
measurement showing tails that produce nothing dominate the residual.

### Finalize `reason` on the wire
**Where:** prbe-backend gateway (`SessionFinalizeRequest`, two-field model), research-os tap
`build_finalize_body`, `kb/session_receipts.py:validate_payload`.

**What:** carry why a session ended (`session_end_hook`, `process_exit`, `daemon_reconcile`)
on the finalize. **Why:** today the server cannot tell a clean `/exit` from an orphan-detected
exit, so the reliability of each client signal is unmeasurable. **Context:** the gateway
validates a two-field model and rebuilds the forwarded body, so this is three repos.
**Effort:** S per repo. **Priority:** P3. **Depends on:** PR 2 of the spend plan (`completed_by`).

### pi daemon: verify process-death finalize
**Where:** research-os `agent/plugins/probe-research-pi/src/daemon.ts`.

**What:** confirm the pi daemon enqueues a finalize when the pi process exits without a hook,
as the Claude Code tap does (`tap/main.py:629-650`). **Why:** otherwise pi sessions rely on the
server idle sweep alone. **Effort:** S. **Priority:** P3. **Depends on:** None.

### Legacy partial passes: done v1 rows whose last pass kept its end signal
**Where:** `ingestion_queue` (research kb), `scripts/backfill_finalize_markers.py`.

**What:** find out why the old worker's last pass on these sessions was partial (a segment
failed, the cap hit, the model declined the tool -- which logged nothing before the
`tool_declined` warning -- or extraction was off), and re-mine them. **Why:** the old
worker kept an end signal only after a partial pass, so these sessions hold fewer units
than they have. **Context:** after the one-rule change, end signals are never removed, so
"done with a marker on top" no longer identifies them; the population is the backfill's
`--report` `already_ended_queue_ids`, captured before the new worker mined anything
(~240 rows on research on 2026-09-23). The follow-up change records every pass's outcome
and retries partial ones. **Effort:** S. **Priority:** P3. **Depends on:** that change.

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

### 18 documents still route to their mention through a stray copy
**Where:** `entity_aliases` (managed plane, probe-founders).

On 2026-09-23 the 660 documents that past auto-merges had folded into their
mention were repaired: each was unmerged, then its mention folded INTO it (642
`repair: rule:document-twin` audit rows). 18 were skipped because a stray node
with the document's id already existed next to the alias row, so unmerge
cannot re-create it. Those documents stay reachable through the stray node,
but new edges written for them still route to the mention. **Fix:** move the
alias-lane edges onto the existing node and drop the routing row (unmerge's
steps 3-7 against an existing node), or delete the stray and unmerge.

### A `Co-authored-by` trailer can rename a real person
**Where:** `kb/handlers/github.py` (co-author Person nodes), `engine/ingest/graph_writer.py`
(`properties || EXCLUDED.properties`).

Checked 2026-09-23: an email IS sound identity for auto-merge -- GitHub fills a
commit's `author.username` only when the email belongs to that account, so a
login node's email is GitHub-linked, and a co-author node is keyed BY the email.
What a stranger can still do is write any NAME in a trailer: the co-author node
(and, once merged, the real person's node) takes that name on every upsert.
**Fix:** don't let a trailer's name overwrite a name that came from an account.

### neon_auth person enrichment is unwired on managed, not just unpermitted
**Where:** `neon_auth."user"` on managed-shared; the query lives in prbe-backend
(see `kb/handlers/claude_code.py:717` — "Gateway injects this from
neon_auth.user.name").

Migration 0112 grants the app role SELECT, which stops 1,514 errors/day. It does
not make the lookup return anything. Measured 2026-08-20 on managed:

  * `neon_auth."user"` is the 4-column shim `scripts/migrate.py` creates so
    `customers.organization_id`'s FK has a target. `relpages = 0` — it has never
    held a row. It is not Neon Auth's real data.
  * All 5 active customers have `organization_id IS NULL`, so nothing links a
    tenant to an organisation either.

So person name/email enrichment cannot work on managed regardless of grants.
**Decide:** wire the Neon Auth sync (populate the tables and the org links), or
delete the lookup in prbe-backend. Doing neither leaves a query that is
permitted, silent, and permanently empty.


### Wire `tenant_virtual_key_context` at LLM entrypoints (Phase 0c)
**Where:** `services/retrieval/router.py` (`_call_haiku`),
`services/retrieval/synthesis.py` (`synthesize_stream` + `_call_*` non-
stream), `services/ingestion/inferred_edges/extractor.py`,
`services/ingestion/code_graph/cross_repo_deps.py`.

This PR added `shared/litellm_key.py` — a per-tenant LiteLLM virtual-key
fetcher (control plane `GET /routing/customer/{id}/litellm-key`) plus a
`tenant_virtual_key_context(customer_id)` async-context-manager that
`shared.llm._maybe_inject_gateway` consults via ContextVar. Until each
LLM entrypoint wraps its call window in that context manager, the
shared-managed data plane attributes every call to the master
`LLM_GATEWAY_KEY` and per-tenant cost accounting silently degrades.

**Order, in descending impact:**

1. Retrieval router + synthesize (every user query — wrap at the FastAPI
   request handler so the whole pipeline inherits the binding).
2. Cross-repo deps + inferred-edges extractors (ingest-side).

**Dependency:** control-plane endpoint
`GET /routing/customer/{customer_id}/litellm-key` must be live (PR #206
in the control-plane repo). `get_tenant_virtual_key` raises
`LiteLLMKeyUnavailable` until then — wrap calls in try/except to
soft-fall-back to the env-var key (which is the path we have today).

**Trigger:** shared-managed cutover, or any prod logs showing the
LiteLLM master key dominating spend with no per-customer breakdown.

### Slack display-name cache: 429 / Retry-After retry
**Where:** `services/ingestion/handlers/slack.py` (`_SlackUserCache.resolve` /
`.prime`)

`_SlackUserCache` currently returns None on transient failure (no cache
poisoning, but no retry either). On a 429 storm during prime, the workspace
falls back to per-message users.info indefinitely until the Retry-After
window passes.

**Fix:** read the `Retry-After` header, sleep, retry once or twice with
exponential backoff. ~25 lines.

**Trigger:** logs show repeated `slack.users_info_non_200` (status=429)
followed by no recovery, OR cache hit rate visibly degrades after a known
rate-limit burst.

Discussed: 2026-05-01 in `/review` — flagged by performance specialist,
deferred as slow-burn after the JSONB-backed LRU shipped (which already
caps memory and survives worker restarts so the cost of a transient prime
failure is much lower than it was with the in-memory-only cache).

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

### usage_events + query_traces retention
**Where:** new cron `scripts/cron_usage_retention.py` or pg_partman config

**Why:** search needs history but both tables grow unbounded. Search
latency on usage_events degrades past ~10M rows; query_traces rows are
~50x fatter (full request/response JSONB) so storage pressure hits
sooner.

**Trigger:** when a tenant crosses 1M rows on usage_events, OR
query_traces exceeds ~10GB total, OR /usage/search latency exceeds 500ms.

**Fix:** ~30 lines for cron sweep deleting >180d on BOTH tables, OR
pg_partman monthly partitions on both. Pair them — one cron, two DELETEs.
The two tables share `request_id` and a 1:1 row relationship, so the
retention window must match to avoid orphan traces.

### Hot-node write strategy (per-customer drain ceiling)
**Where:** `services/ingestion/graph_writer.py:upsert_nodes` (and the
provenance UPSERT in the same file)

PRs #40, #41, and #44 closed the contention-driven DLQ failures (batched
writes, sorted lock order, 5min timeout, 50 retries, per-customer cap=30).
Per-customer drain rate is now bounded at ~150-600/min by hot-row
serialization: every Slack/GitHub/Linear batch UPSERTs a small set of
super-hot nodes (`team`, `channel`, `repo`, recurring `user`/`author`
canonical_ids), and `ON CONFLICT DO UPDATE` row locks are held until COMMIT,
so concurrent txs serialize on each other's full Phase B duration.

**Trigger:** a customer's queue stays >1000 pending sustained for >30 min
without DLQ, OR onboarding/replay drain takes >12 hours, OR Phase B p99
duration climbs past ~1s.

**Fix options** (do not bump knobs — adding workers does not help, they all
queue on the same row lock):
1. **Dedupe touches at the queue level:** if N pending rows for the same
   customer all touch `team:X`, collapse the upsert into one statement at
   claim time. Saves N-1 lock acquisitions on the hottest nodes.
2. **Async hot-node provenance writes:** defer "bump last_seen_at" on
   `graph_node_provenance` to a separate batched queue that flushes every
   few seconds. Removes the second per-batch UPSERT entirely from the
   critical path.
3. **Skip no-op provenance updates:** check before writing — most touches
   don't change anything material.

These are real refactors, not config changes. Worth doing only when a real
drain-rate incident motivates the complexity; for any reasonable workspace
size today the queue absorbs and drains overnight at worst.

---

### Envelope `priority_hint` so bulk imports stop riding the top tier

**Where:** `engine/shared/custom_ingest.py` (envelope), `engine/ingest/handlers/custom_ingest.py`,
and research-os `app/indexing/relay.py` + `index_outbox`.

`custom_ingest` now sits at `PRIORITY_RESEARCH_CONTENT` (100) because a note
someone deliberately wrote should not queue behind every tenant's automatic
transcripts. But the envelope carries nothing that distinguishes a typed note
from a 21,000-row bulk import or from the reconciler re-pushing drift, and all
three arrive as `custom_ingest`. Measured 2026-09-16: bucket-robotics pushed
21,153 rows in three days, one per document, no churn -- a mirror import
wearing a researcher's badge.

Today the per-(customer, tier) in-flight cap bounds the damage to half the
loops in that lane. The real fix is for research-os to say which kind a push
is: a `priority_hint` on the envelope (`interactive` | `mirror` | `reconcile`),
mapped to 100 / 60 / 50 by the connector, with anything unhinted staying where
it is now.

**Depends on:** the tier table (shipped). **Cost:** two repos, an origin column
on `index_outbox`, two deploys. **Trigger:** a tenant's import measurably
delaying another tenant's writes, which the queue-age signal will now show.

---

### Per-session parse cache for transcript re-normalization

**Where:** `kb/handlers/claude_code.py` (`fetch_supplementary`), `engine/ingest/worker.py`.

Every run of a transcript row re-fetches and re-parses EVERY batch in
`payload_s3_keys`, so the fixed cost of a run grows with session length: one
row on 2026-09-16 carried 2,485 keys. Fitted over 327 runs that day, a run cost
`12.0s x new_documents + 8.8s` -- the 8.8s is this, and it is not constant, it
is proportional to how long the session has been going.

The claim-protocol fix stopped two slots doing this at once, and the CPU raise
made each pass faster, but the work itself is still O(session length) per
batch. A per-session parse cache keyed on the R2 key set, invalidated on
reclaim, would make it O(new batches).

**Depends on:** re-measuring after the worker is off half a core.
**Cost:** a day, mostly cache-invalidation care.
**Re-prioritized 2026-09-23:** the session-completer sweep was re-mining every idle v1 session
daily (docs/plans/extraction-spend-plan.md); once that fix lands, transcript runs drop ~98 %,
so re-measure the parse cost before building this.

---

### Reconciler churn: six documents per tenant re-index every ~3 hours

**Where:** research-os `app/indexing/reconcile.py` (30-minute CronJob).

Measured on 2026-09-16: for tenant `probe`, six paper documents re-enqueued
about every three hours with a changed `source_content_hash`, and `anthrogen`
showed the same shape at 148 rows for 7 documents. Something in the projection
is not stable across recomputation, so the reconciler sees drift where there
is none and pays a full re-index -- including re-embedding -- for it.

Small in absolute terms; worth one afternoon because it is pure waste and it
pollutes every queue measurement taken from now on.

**Cost:** an afternoon on the research-os side, diffing one document's
projected payload across two reconcile passes.

---

### `attempts` on a hot transcript row is ~3.6x the runs it actually made

**Where:** `engine/ingest/worker.py`.

Row 41234 finished with `attempts=1510` and `version=1840`, while the worker
log showed 218 `normalizer.start` lines for it. Every increment path is
accounted for in code, so the gap is unexplained rather than wrong -- but
`worker_max_attempts` is 50, and a row whose `attempts` climbs 3.6x faster
than its real work reaches the dead-letter ceiling 3.6x sooner. Today's
effective-retry-forever setting hides it.

One log line at claim time carrying `attempts` would settle it in a day of
traffic.

**Cost:** an hour to instrument, a day to observe.

---

### Second worker replica

**Where:** research-os `charts/research-os/values.yaml` (`engine.worker.replicas`).

One replica is a single point of failure: a crash stops ALL ingestion for
every tenant until Kubernetes restarts it, and nothing alerts on the gap
except the drain-stall beacon.

Not yet, deliberately. Two replicas multiply claim contention, and the
per-(customer, tier) cap and the claim-then-release protocol should be watched
under real traffic on one replica first. The queue-age signal is what will say
whether one replica is actually the constraint.

**Depends on:** the per-tier cap and claim-protocol changes running in prod.
**Cost:** a chart value, plus a week of watching before trusting it.

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
  (Minimal `<query>...</query>` wrapping shipped in feature/router-list-mode.
   Residual: input length cap, structured-input validation across all
   extractors, detection-pattern logging, response-shape sanity checks.)
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

### Multi-hop graph retrieval — measure + tune

The graph retriever (`services/retrieval/retrievers/graph.py`) does
single-hop traversal today: feed `(entity_type, canonical_id)`, get docs
that have edges to that entity. Multi-hop reasoning ("did the PR that
closes ABC-123 ship to prod yet?") relies on the graph retriever surfacing
the related docs and the synthesis LLM connecting them. We've never
measured precision/recall on this path.

**Fix:** build a small eval set of ~10 multi-hop queries with hand-labeled
expected docs. Measure precision@5 + recall@5 with the current fusion
weights. If the graph retriever underweights when entity confidence is
high, bump its RRF contribution conditionally (e.g., when any extracted
entity has confidence ≥ 0.85, treat graph hits with a multiplied score in
fusion). ~50 LOC + the eval set.

Discussed: 2026-04-28 in plan-eng-review for feature/router-list-mode.

### Cross-source author identity resolution

`documents.author_id` and the `Person` graph node are populated with
whatever raw identifier each connector saw, with no resolution between
forms. The same human appears under multiple values:

- GitHub PR/issue/review → GitHub login (`mahit`)
- GitHub commit where the email didn't resolve to a login → email (`mahit@prbe.ai`)
- GitHub commit `Co-authored-by:` trailers → email (always)
- Slack → user id (`U07ABC123`)
- Linear → user id (`user_a3f9`)
- Notion / Granola / Sentry → varies

Today, `search_knowledge` exposes `author_id` (doc-level, raw form, detail="full" only since the detail parameter landed). The
`Co-authored-by:` work landed alongside this comment surfaces co-authors
under their email — usually a *different* string from the same person's
primary `author_id` on PRs. Agents filtering by `author_ids=["mahit"]`
silently miss the email-form rows, and "who's been most active across
sources" is impossible to answer without per-tenant manual mapping.

**Fix:** introduce a canonical `person_id` per customer with a many-to-one
mapping from raw forms. Two layers:

1. **Storage:** add `person_aliases` table — `(customer_id, raw_id,
   source_system, person_id)` — and a `persons` table — `(customer_id,
   person_id, display_name, primary_email)`. Pre-populate via heuristic
   (email match across sources, GitHub login → noreply email pattern,
   Linear/Slack profile email if available via OAuth scope).
2. **Surface:** retrieval returns `author_id` (raw, kept for back-compat)
   AND `author` (resolved canonical `{person_id, display_name}`) on each
   chunk. Filter API gains `person_ids=[...]` that joins through aliases.

**Scope:** ~600 LOC across schema migration, alias-resolver in retrieval
helpers, connector hooks for alias hints (Slack profile email, Linear
user export), and dashboard UX for manual merges. Material work, deserves
its own design doc.

Discussed: 2026-04-28 in `/plan-eng-review` for the surface-author-id
chunk shape change. Skipped in that PR — the visibility fix could ship
without resolution; trying to do both would have ballooned scope.

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
- OAuth `state` HMAC-signed (moved to prbe-backend's gateway in
  `app/dependencies/oauth_state.py` — closes the unsigned-state P1)
- Notion connector OAuth: `exchange_oauth_code` + `identify_workspaces` reading
  from `IntegrationToken.install_metadata` so `/api/oauth/notion/exchange` works
  end-to-end

---

## P4 — follow-ups from the Notion connector OAuth work

### Tuple return from `Connector.exchange_oauth_code`
**Where:** `services/ingestion/handlers/base.py:193` and all six connector subclasses

Today connectors that capture workspace info during exchange (Notion, and
any future provider that gives back the workspace id directly) plumb it
through `IntegrationToken.install_metadata` — a Pydantic transient field
that exists only in memory between `exchange_oauth_code` and
`identify_workspaces`. Works, but pollutes the shared model with one inert
attribute and creates a request-scoped lifetime that's awkward to reason
about.

**Fix:** change `exchange_oauth_code` to return
`tuple[IntegrationToken, list[ExternalWorkspaceRef]]`. Drop the
`install_metadata` field. Slack/GitHub/Linear/Granola/Sentry return
`(token, [])` and keep their `identify_workspaces` methods; Notion drops
`identify_workspaces` entirely. ~50 lines across all connectors.

Defer until the next connector lands that needs install-time metadata.

### Drop the legacy `payload_s3_key` column

**Where:** `db/migrations/versions/` + `shared/models.py` + ingestion paths

Migration 0026 added `payload_s3_keys text[]` and backfilled every existing
row with `ARRAY[payload_s3_key]`, but did NOT drop the legacy column —
dropping in the same deploy as the code change would race with mid-deploy
old workers reading a column the migration just dropped. After this
deploy stabilizes, a follow-up PR can drop the column cleanly:

1. New migration: `ALTER TABLE ingestion_queue DROP COLUMN payload_s3_key`.
2. Remove the legacy field from `WebhookEvent` (shared/models.py:142),
   the connectors that synthesize `payload_s3_key=""` placeholders, and
   the legacy fallback branches in worker.py:_process and
   claude_code.py:fetch_supplementary.
3. ~50 LOC, fully mechanical once no caller is reading the legacy field.

**Trigger:** any time after migration 0026 has been live ~1 deploy cycle
(say a week) with no rollback signals.

### Notion refresh-token rotation
**Where:** `services/ingestion/handlers/notion.py:exchange_oauth_code` + new helper

Notion is rolling out refresh-token rotation. Today's access tokens don't
expire; older public integrations get long-lived static tokens. Newer ones
may get tokens with a finite lifetime + a refresh_token. We persist the
refresh_token already (column was always there); we just don't refresh.

**Fix:** when a Notion API call returns 401 with a token-expired error,
exchange the refresh_token for a new access_token via
`POST /v1/oauth/token` with `grant_type=refresh_token`. Update both columns
on `integration_tokens` and retry the original call once.

Trigger to do this work: first observed 401 from Notion in production logs.

---

## related_entities follow-ups (from /review)

### ACL-aware IDF denominator for `score`
**Where:** `services/retrieval/retrievers/related_entities.py:172` (neighbor_global_freq CTE)

`score = doc_count / ln(1 + global_doc_count)` divides by the count of all
tenant docs the neighbor is attached to, regardless of `requesting_user_id`.
Today ACL is a no-op for nearly all retrievals so this is latent. When ACL
enforcement turns on, a low score becomes a side channel hinting "this entity
is mentioned in private docs you can't see" for any caller with read access
to even one related doc.

**Fix:** thread `requesting_user_id` into `walk_result_doc_neighbors` and join
`acl_snapshots` (or whatever ACL filter the rest of the pipeline uses) inside
`neighbor_global_freq`. Cost: ~15 LOC + extra join cost on the hot path.

**Trigger:** before flipping ACL enforcement on for any tenant.

### Distinguish missing-graph_node from legitimate-empty
**Where:** `services/retrieval/retrievers/related_entities.py:109` (doc_anchors CTE)

`walk_result_doc_neighbors` anchors only through `graph_nodes(label='Document',
canonical_id=doc_id)`. If a doc was ingested but graph_node creation failed (or
a connector emitted no graph_node row), the response is `[]` -- which the
three-state contract defines as "walked, no neighbors". That masks partial
ingestion / data corruption as a clean empty result.

**Fix:** count how many `ranked_result_docs` failed to resolve to a graph_node
row and surface as a debug field (e.g. `unanchored_doc_count`) on the response,
or emit a structlog warning at info level when the count is non-zero.

**Trigger:** if related_entities ever appears suspiciously empty in production
when the underlying docs clearly have graph relationships.
