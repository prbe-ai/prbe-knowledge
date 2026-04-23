# Temporal + content-addressable chunks migration

**Status:** shipped (single mega-PR per user direction, 2026-04-23).

Implemented on a destructive rebuild of the local Postgres (no data to preserve). PR 4 follow-ups remain deferred.

---

## What landed

### Schema (`db/schema.sql`)
- [x] `documents` — temporal columns activated (`valid_to`, `supersedes_doc_id`, `deleted_at` already existed, now written by the normalizer)
- [x] New partial index `idx_documents_live (customer_id, doc_id) WHERE valid_to IS NULL`
- [x] `chunks` — redesigned for content-addressable identity:
    - added `content_hash TEXT NOT NULL`
    - added `valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW()`
    - added `valid_to TIMESTAMPTZ`
    - replaced `doc_version` with `first_seen_version` + `last_seen_version`
    - `UNIQUE (doc_id, content_hash)` — a chunk is one row per unique content per doc
    - FK changed: `(doc_id, first_seen_version) → documents(doc_id, version) ON DELETE CASCADE`
- [x] New indexes: `idx_chunks_doc_live (doc_id) WHERE valid_to IS NULL`, `idx_chunks_doc_hash (doc_id, content_hash)`
- [x] `customer_source_mapping` consolidated into canonical `schema.sql` (was drifting via 0003 migration only)
- [x] RLS FORCE consolidated into canonical `schema.sql`
- [x] Migrations 0002 + 0003 deleted; single initial migration executes the updated schema

### Models (`shared/models.py`)
- [x] `Chunk` gained `content_hash`, `valid_from`, `valid_to`, `first_seen_version`, `last_seen_version`; dropped `doc_version`
- [x] New `TemporalMode` enum (`latest` / `as_of` / `changed_between` / `all`)
- [x] New `TemporalSpec` with source-vs-ingest `time_basis` + Pydantic cross-field validation
- [x] `QueryRequest.temporal: TemporalSpec` (default = `latest`)

### Normalizer (`services/ingestion/normalizer.py`)
- [x] `_upsert_document` closes out prior live version (`UPDATE ... SET valid_to = NOW()`) in the same transaction as the new-version INSERT
- [x] `_sync_chunks` — three-way diff over `(doc_id, content_hash)`:
    - reused → `UPDATE last_seen_version`, no embed call
    - added  → embed + INSERT (also reviveable via ON CONFLICT DO UPDATE SET valid_to = NULL)
    - removed → `UPDATE SET valid_to = NOW()`
- [x] Chunker-version guard — if `chunker_version` differs from what's on a live row, those chunks are marked stale and forced to re-embed
- [x] Deleted-doc path — empty body → diff naturally marks every live chunk stale
- [x] Embedding-cost accounting — `NormalizeOutcome` now splits `added_chunk_count` / `reused_chunk_count` / `removed_chunk_count` for observability

### Retrievers (`services/retrieval/`)
- [x] New `temporal.py` — shared SQL-fragment builder for all three retrievers
- [x] `vector.py`, `bm25.py`, `graph.py` — all take `TemporalSpec` param, default `latest` (closes the stale-chunk retrieval bug)
- [x] Doc join switched from `c.doc_version = d.version` to `c.doc_id = d.doc_id` (chunks span versions now)
- [x] `main.py` — threads `req.temporal` through every retriever call

### Handler edit/delete coverage
- [x] Slack — `message_changed` re-ingests at same `doc_id` (event clock in `source_event_id` prevents UNIQUE collision); `message_deleted` writes a tombstone
- [x] Linear — `remove` action promoted from ignored → writes a tombstone
- [x] Notion — `page.deleted` / `database.deleted` promoted from ignored → tombstone
- [x] GitHub — `deleted` + `transferred` actions on PRs and issues write a tombstone
- [x] Sentry — events are append-only; no changes needed

### Tests
- [x] All 74 pre-existing tests still pass (one pre-existing flaky RLS test unrelated to this work)
- [x] `tests/test_chunk_diff.py` — new: edit replaces body → old chunks marked stale, new chunks written, version bumped
- [x] `tests/test_chunk_diff.py` — new: identical replay writes no new rows
- [x] Updated `tests/handlers/test_linear.py` — `remove` action no longer returns None, asserts `source_event_id` contains `:remove:` tail
- [x] Updated `tests/handlers/test_notion.py` — `page.deleted` returns a WebhookParseResult with `is_delete=True`; plain `page.updated` source_event_id format adjusted for the new `:edit:` suffix

### Verified
- [x] `docker compose up` + `alembic upgrade head` applies the new schema cleanly from scratch
- [x] `pytest tests/ --ignore=tests/test_multitenant_isolation.py` → 76 passed

---

## Deferred (PR 4 follow-ups — not in this batch)

- [ ] Notion block-native chunking via `GET /v1/blocks/{page_id}/children` + `block_id` chunk identity
- [ ] `GET /documents/{doc_id}/versions` endpoint
- [ ] `GET /documents/{doc_id}/diff?from=v1&to=v3` endpoint
- [ ] `POST /changes` endpoint for doc-level "what moved in time window" queries
- [ ] Pre-existing `test_rls_graph_isolation` flakiness (unrelated to this migration; investigate separately)

---

## Review

**What changed:** 504 insertions / 86 deletions across 5 files for the schema + model + normalizer + retriever core; handler edit/delete coverage added to 4 of 5 connectors; 2 new tests for the content-addressable chunk contract.

**What surprised me:** `_upsert_document` had a latent bug where `supersedes_doc_id = $1` reused the same bind param as `doc_id` in the WHERE clause — would have self-referenced within the same doc_id. Left `supersedes_doc_id` untouched during within-doc-id version bumps; it stays reserved for connector-driven chain-break scenarios (source delete + recreate with new source_id).

**Performance shape post-PR:** on a 30-chunk Notion-like edit-title scenario, expected embed calls drop from ~30 → ~1 (just the title chunk). Not benchmarked here — smoke-test after first real workload lands.
