# Plan A — Post-Approval Knowledge Surface Summary

This branch (`post-approval`, 20 commits over `incidents-writeback`) lands the
**prbe-knowledge** side of the post-approval downstream-actions pipeline
described in `2026-05-17-post-approval-knowledge-plan.md` (Plan A).

Plan A is **persistence + APIs only** inside prbe-knowledge. The
orchestrator side that consumes the dispatch + authors the wiki
artifacts (Plan B) and the dashboard review UI (Plan C) ship in
follow-up branches and are not part of this diff.

This doc gives a reviewer the shape of the diff without scrolling
through 20 commits.

## Base + downstream deps

- **Base:** `incidents-writeback` — Plan 4's investigation review surface
  (`POST /api/incident-investigations`, `/approve`, `/reject`).
- **Depends on:**
  - **Plan B** (prbe-orchestrator): will provide the
    `POST /internal/post-approval-actions` endpoint that
    `services/post_approval/dispatch.py` POSTs to on the
    (approved ∧ resolved) edge, plus the Pass 1 / Pass 2 agents that
    POST evidence packs + wiki artifacts back here. The dispatch seam
    is robust to a missing orchestrator (`orchestrator_base_url`
    unset, connection refused, 4xx, 5xx-exhaustion all surface as
    `metadata.post_approval_dispatch_failed=true` for dashboard
    recovery).
  - **Plan C** (dashboard BFF): will surface the review queue
    (`GET /api/wiki-artifacts`), the postmortem template editor
    (`/api/customer-postmortem-templates/*`), and the
    "Re-trigger post-approval" recovery button
    (`fire_post_approval_dispatch`).

## Schema additions

Five new alembic migrations (`0082`–`0086`):

| Revision | Purpose |
|---|---|
| `0082_visibility_columns` | `documents.visibility` + `chunks.visibility` TEXT NOT NULL CHECK IN ('draft','approved'), default `'approved'`. Existing rows backfill to `'approved'`. Backed by partial indexes scoped to `visibility='approved'` so default retrieval gets the same plan it does today. |
| `0083_incident_investigations_post_approval_cols` | Adds `incident_investigations.approved_at`, `resolved_at`, `post_approval_dispatched_at`, `evidence_pack jsonb` columns. On the resolution-first race (resolution arrives before the investigation row exists), `on_resolution_event` INSERTs a partial row with `state='pending_review'` (no constraint drop needed). |
| `0084_wiki_review_queue` | New table `wiki_review_queue` — per-artifact review lifecycle (state in `pending_writeback`, `pending_review`, `approved`, `rejected`, `failed_pending_review`), with `parent_artifact_doc_id` self-referencing FK for the reject-and-revise version lineage. RLS enabled with `tenant_isolation` policy. |
| `0085_customer_postmortem_templates` | New table `customer_postmortem_templates` — per-customer template override (inline body or doc_ref). RLS enabled. |
| `0086_incident_investigations_metadata` | `incident_investigations.metadata jsonb` for the dispatcher's `post_approval_dispatch_failed` recovery flag and future per-incident dashboard hints. |

Net schema delta: **2 new columns + 4 new tables + 1 dispatch-state column tuple**.

## Behavior changes

- **Default retrieval defaults to `visibility='approved'`.** All seven
  retrievers (`vector`, `bm25`, `graph`, `directed`, `inferred_edges`,
  `sql`, `id_lookup`) take a new `include_drafts: bool = False`
  argument. The Plan C reviewer surface will opt in via
  `include_drafts=True`; everything else (API-key surface, MCP,
  agentic search) stays approved-only. The `/sources/{doc_id}`
  endpoint hard-codes `visibility='approved'` (no opt-in).
- **Template resolver fetches `doc_ref` bodies with `visibility='approved'`**
  — `template_resolver.py` mirrors the new retrieval default so a
  postmortem template can't accidentally resolve to a draft body.

## New endpoints (all on `services.ingestion.main:app`)

All gated by `X-Internal-Knowledge-Key` (service-to-service trust
boundary; same header existing internal routes use).

| Method + path | Purpose |
|---|---|
| `POST /api/incident-evidence-packs` | Orchestrator Pass 1 caches an `EvidencePack` on the per-incident row. Idempotent on `(customer_id, incident_doc_id)`. |
| `GET /api/incident-evidence-packs?customer_id=&incident_doc_id=` | Read back the cached pack (Pass 2 input). |
| `POST /api/wiki-artifacts` | Orchestrator Pass 2 writeback for `postmortem` / `knowledge_page` / `correction` artifacts. Persists as `Document(visibility=DRAFT)` via `Normalizer.persist_single_document` and upserts a `wiki_review_queue` row at `pending_review` (or `failed_pending_review` for stub-mode artifacts). |
| `GET /api/wiki-artifacts` | Reviewer list (dashboard). Filters by state, kind, incident. |
| `GET /api/wiki-artifacts/{artifact_doc_id}` | Reviewer detail + version-lineage chain. |
| `POST /api/wiki-artifacts/{artifact_doc_id}/approve` | Atomic flip: `documents.visibility`, `chunks.visibility`, and `wiki_review_queue.state` all transition to `approved` in one tenant-scoped transaction. |
| `POST /api/wiki-artifacts/{artifact_doc_id}/reject` | State flip to `rejected` is durable independent of the orchestrator re-dispatch — failure stamps `metadata.re_dispatch_failed=true` for ops recovery. |
| `GET /api/customer-postmortem-templates/{customer_id}` | Read the override row (or null). |
| `GET /api/customer-postmortem-templates/{customer_id}/effective` | Resolved template body (override → doc_ref → bundled default). |
| `PUT /api/customer-postmortem-templates/{customer_id}` | Upsert override (inline body or doc_ref). |

## New internal dispatch (caller side)

`services/post_approval/dispatch.py` exposes:

- `on_resolution_event(customer_id, incident_doc_id, resolved_at=None)` — connectors call this when PD's `incident.resolved` or incident.io's resolution webhook lands. Idempotent (COALESCE preserves first observed `resolved_at`); creates a partial row if no investigation exists yet.
- `on_approval(customer_id, incident_doc_id, approved_at=None)` — `mark_approved` calls this. Idempotent.
- `fire_post_approval_dispatch(customer_id, incident_doc_id)` — dashboard "Re-trigger" entrypoint; clears the guard + the `post_approval_dispatch_failed` flag and re-runs `_check_and_dispatch`.
- `_check_and_dispatch` — exactly-once HTTP semantics via `SELECT ... FOR UPDATE` pessimistic claim. Stamps `post_approval_dispatched_at` inside the lock BEFORE the HTTP, releases the lock, POSTs to `${ORCHESTRATOR_BASE_URL}/internal/post-approval-actions` with `X-Internal-Backend-Key`. On failure clears the guard back to NULL with a CAS predicate (guards against the race with a concurrent recovery dispatch) and stamps `metadata.post_approval_dispatch_failed=true`.

## Commit history (18 feature commits + 2 from this component)

| SHA | Subject |
|---|---|
| `7d5213b` | `feat(knowledge): visibility flag on documents + chunks for draft-gated wiki artifacts` |
| `d48ed19` | `feat(knowledge): incident_investigations post-approval columns` |
| `97d901b` | `feat(knowledge): wiki_review_queue table + RLS` |
| `416815b` | `feat(knowledge): customer_postmortem_templates table + RLS` |
| `301d115` | `feat(knowledge): wiki doc types, Visibility enum, pydantic schemas, default postmortem template` |
| `d3050f0` | `feat(knowledge): wiki_review_queue CRUD layer with version-lineage support` |
| `271`...`8` | `feat(knowledge): postmortem template resolver (override → doc_ref → default)` |
| `527f8ca` | `feat(knowledge): connectors detect PD + incident.io resolution events` |
| `e545e32` | `feat(knowledge): post-approval dispatch seam with exactly-once semantics` |
| `9cf3d2d` | `feat(knowledge): wire mark_approved → post_approval.on_approval` |
| `4ce88e3` | `feat(knowledge): ingest worker fires on_resolution_event after normalize` |
| `8cf2889` | `feat(knowledge): thread visibility through Normalizer.persist_single_document` |
| `64c2fd8` | `feat(knowledge): POST/GET /api/incident-evidence-packs` |
| `d0cbb37` | `feat(knowledge): POST /api/wiki-artifacts writeback` |
| `c888645` | `feat(knowledge): wiki-artifacts review routes (list/detail/approve/reject)` |
| `dced19c` | `feat(knowledge): customer postmortem template routes (GET / GET-effective / PUT)` |
| `0c70880` | `feat(knowledge): default retrieval to visibility='approved'; include_drafts opt-in for reviewers` |
| `c8b0807` | `fix(knowledge): filter chunks by visibility in template_resolver doc_ref fetch` |
| _Component 7_ | `chore(knowledge): live smoke script for post-approval flow` |
| _Component 7_ | `docs(knowledge): Plan A summary for post-approval downstream actions` |

## "Done" gate (Plan A Task 19)

Per the team's `feedback_container_smoke_tests` memory, pytest is not
sufficient for prbe-knowledge features. The live smoke script
`scripts/smoke_post_approval_knowledge.sh` is the actual "done" gate.
It brings up:

- The Docker Postgres at the alembic head (asserts `0086_inv_metadata`).
- A `python -m http.server`-style mock orchestrator on `:9099` that
  accepts `POST /internal/post-approval-actions` with 202.
- `services.ingestion.main:app` on `:8090`.
- `services.retrieval.main:app` on `:8091`.

And walks 11 named steps covering: investigation writeback → approve →
`on_resolution_event` → dispatch fires (HTTP captured) → evidence pack
round-trip → wiki postmortem writeback at `visibility=draft` →
`/sources` 404s for the draft → artifact approve flips doc+chunk
visibility → `/sources` now returns the doc → reject path's state
flip is durable even when the mock orchestrator is killed → the
resolution-first ordering path (resolved before approved still
fires dispatch on the second timestamp landing) → postmortem template
PUT + GET + effective round-trip.

To run:

```bash
docker compose up -d postgres
./scripts/neon-migrate.sh local   # if migrations not yet applied
./scripts/smoke_post_approval_knowledge.sh
```

Expected last line: `=== ALL SMOKE CHECKS PASSED ===`.

## Known asymmetry: incident-investigation vs wiki-artifact approve semantics

Plan 4's `services/ingestion/investigation_state.py::mark_approved`
silently transitions any state (including `rejected`) to `approved`
without raising. Plan A's `services/post_approval/wiki_review_state.py::mark_approved`
filters `WHERE state IN ('pending_review','failed_pending_review')` and
raises `ValueError` on terminal-`rejected`, which the wiki review route
maps to a 409.

The two surfaces therefore respond differently to "approve a rejected row":

- `POST /api/incident-investigations/{id}/approve` on a rejected
  investigation: **200**, state flips to `approved`.
- `POST /api/wiki-artifacts/{id}/approve` on a rejected artifact: **409**.

Plan A's stricter semantic is the desired model. Plan 4's permissive
semantic predates Plan A and was preserved here to avoid scope creep.
A focused Plan-D (or investigation-state cleanup) should tighten the
Plan 4 surface to match.

Plan C's dashboard should be aware of this — its UI will see different
responses from the two endpoints and needs to handle the 409 path for
wiki-artifact approve while currently never seeing it from the
investigation approve endpoint.
