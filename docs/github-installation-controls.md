# GitHub integration control protocol 2

This is the Knowledge data-source lane, not a research workspace or a code
archive. The existing ingestion process runs both legacy and v2 workers. No HTTP
handler starts a background import task.

## Internal HTTP contract

Every endpoint below requires `X-Internal-Knowledge-Key`. Installation operations
take the tenant from `X-Prbe-Customer`; the research-os gateway must enforce live
membership and owner/admin management permissions before proxying.

- `GET /api/github/capabilities`: protocol_version=2, installation_controls,
  durable_backfills, worker_ready. Readiness requires a v2 worker heartbeat within
  90 seconds, not merely a new API binary.
- `POST /api/github/connect`: existing customer_id and installation_id, plus
  optional protocol_version=2. Version 2 connects without implicit backfill and
  starts with live disabled and empty scope. Omitted version retains legacy
  behavior for unadopted installations.
- `/api/github/installations/{installation_id}/sync`: GET settings/health;
  PATCH `{scope?,sync_enabled?,expected_revision?}`. Scope entries contain
  `{external_id:"owner/repository",label?}`. Stale settings return 409. Enabling
  requires nonempty accessible scope and schedules an independent catch-up job.
- `GET .../scope-options`: `{options:[{external_id,label}],complete:true}`.
  Enumeration failures never replace the existing selection with an empty list.
- `POST .../backfills`: `{scope,idempotency_key}` creates a durable job. The same
  key/selection returns the same job; key reuse for a different selection is 409.
- `POST .../backfills/lookup`: the same request body recovers an accepted job
  without provider calls or worker readiness. Returns 404 when absent and 409
  on a different selection; repository name case and ordering are immaterial.
  The gateway checks this receipt before gating new work on capabilities.
- `GET .../backfills?limit=30&cursor={job_uuid}`: `{jobs,next_cursor}`.
- `GET .../backfills/{job_uuid}`, `POST .../{job_uuid}/cancel`,
  `POST .../{job_uuid}/retry`: operate on exactly the selected installation/job.
- `POST .../backfills/{job_uuid}/retry/lookup`: recovers a previously accepted
  retry in queued/running/completed state, or 404. A durable `retry_count`
  increments only when a user retry is accepted, independently of worker
  attempts. A subsequent failed/canceled attempt can be retried again.
- `GET .../purge-preview` and `POST .../purge`: counts for documents, chunks and
  active_jobs; warnings describe retained shared graph entities/unbound legacy
  data. Purge only deletes documents owned by connector-issued bindings, and a
  document also bound to another installation is retained.

Jobs return id, connection_id (installation string), source=github, state, scope,
workspace_id=null, created_at, started_at, finished_at, processed_count, counts,
warnings, last_error, attempts, retry_count. States are queued, running, cancel_requested,
canceled, completed, failed. Cancellation takes the installation/job write fence
and can immediately acknowledge canceled once accepted writes have committed.

The token mint contract adds optional installation_id to
`POST /internal/github/installation_token`. The minting service MUST verify that
exact installation belongs to the supplied tenant. Omitting it remains supported
for legacy consumers. V2 never resolves "the latest token for this customer."

## Storage, execution and safety

Migration 0127 is additive. Installation settings, job snapshots, document
bindings and worker heartbeats are separate from legacy singleton token and
backfill_state rows. Tenant tables use FORCE RLS. The worker discovers tenant IDs
then enters the tenant transaction before reading jobs/settings.

New ingestion rows use v2_pending/v2_processing/terminal status values, so old
worker binaries cannot claim them. Bounded transient envelopes (at most 1 MiB,
no binary files) are kept on the existing queue, never archived to R2, and erased
at success, terminal failure, cancellation or purge. Enumeration applies a
200-outstanding-event backpressure limit per installation. Provider walkers are
serial per history cursor, have a bounded buffer, and are canceled when a slice
closes. Only remote provider requests/embeddings are stubbed in the new DB tests.

Write lock order is adoption (when needed) → installation → history job → queue
→ sorted document IDs. Purge locks shared document candidates before deciding
ownership, through binding removal; a projection whose reused chunks disappeared
during planning retries against the new base.
Provider and embedding calls happen before the short projection transaction.
Incomplete v2 embedding batches commit no projection, so a healthy retry can
restore every chunk before the queue or job receives a successful receipt.
Live generations do not invalidate explicit history jobs. Source updated_at and
live-versus-history precedence prevent late older history overwriting live data.
Per-queue leases fence reclaimed writers. Jobs remain running after enumeration
until every queued item has a terminal indexing receipt. Unsupported/failed
provider pages cannot be reported as completed.

Worker crashes retain checkpoints and pending envelopes. Retrying a failed or
canceled job re-enumerates its immutable roots because its erased pending payloads
can predate the last fetch checkpoint; successful receipts deduplicate the replay.
This does not reset another job or the live generation/catch-up state.

## Compatibility and release gates

Deploy migration, API and every ingestion worker before enabling the research-os
page capability. Verify completed rollout as well as the heartbeat: one new worker
does not prove all old pods have stopped. A rollback preserves new queued rows
but old workers will not drain them; disable the UI gate and restore a v2 worker.

Existing mappings become unmanaged control rows. Existing accepted legacy work
must drain before the installation explicitly adopts v2; otherwise settings or
an installation-scoped disconnect return 409. The adoption advisory lock closes
the legacy-enqueue race. Legacy source-wide APIs retain source-wide meaning;
source-wide purge also revokes every v2 installation first.

Scope/content limits are explicit: repository enumeration is bounded to 5,000,
selection to 500, queue envelopes to 1 MiB, and native GitHub history traverses
currently available objects rather than every historical revision. The native
GraphQL queries retain their nested comment/review/file limits. No source bytes
are deleted. Existing unbound documents and shared graph-entity properties are
not silently claimed for deletion.

Optional code-graph extraction and post-commit inferred-edge generation are not
enabled for v2 events: those separate derived workers have no installation fence.
Native documents, body chunks and deterministic relationships remain supported;
legacy derived-lane behavior is unchanged. Adding those optional v2 derivatives
requires their own installation provenance and cancellation/purge fencing.

CI explicitly runs `tests/test_github_installation_controls.py`, GitHub connector,
token-client and code-graph compatibility tests. Actual-provider smoke is separate
from these hermetic tests and must report provider access limitations honestly.
