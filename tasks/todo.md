# fix: normalizer transaction lifetime + same-session claim serialization

**Status:** in progress (started 2026-04-29).

## Problem

Claude Code ingestion is failing under load. Confirmed root cause:
`Normalizer._persist` opens a Postgres transaction via `with_tenant(...)` and
calls `embedder.embed_many(...)` while that transaction is still open. For long
sessions the transaction stays open 60-120s holding row locks; concurrent
workers hit `db_statement_timeout_ms=30000` (30s) on `graph_nodes` upserts and
DLQ. 9 claude_code rows already DLQ'd in prod (2026-04-29 19:31-19:41 UTC).

Compounding: 1GB worker memory triggers OOM kills on large sessions during the
embed loop, which orphans `processing` rows that then have to be reclaimed.

## Plan

- [ ] Bump `fly.worker.toml` memory `1gb -> 2gb`
- [ ] `worker._claim_one`: add `NOT EXISTS` clause on `split_part(source_event_id, ':', 1)` to serialize same-session claims
- [ ] Refactor `normalizer._persist`:
  - Phase A (no write txn): per-doc, short read txn for live-chunks SELECT, then `embed_many` outside any txn
  - Phase B (one short write txn): upsert nodes/edges/ACL/docs + apply pre-computed chunk plans
  - Replace `_sync_chunks` with `_plan_chunks` (read+diff, returns `ChunkPlan`) and `_apply_chunk_plan` (writes only)
- [ ] Tests: txn-free embed assertion, same-session claim skip, full pytest
- [ ] Deploy worker + ingestion, verify drain, resurrect 9 DLQ'd rows

## Affected files

- `fly.worker.toml`
- `services/ingestion/worker.py` (`_claim_one`)
- `services/ingestion/normalizer.py` (`_persist`, `_sync_chunks`)
- `tests/...` (new test files)

## Out of scope (deferred)

- Cross-doc embedding fan-in (one `embed_many` call across all docs in a result). Pure perf, not stability.
- Streaming chunker for huge sessions. Only matters if memory bump 2GB still OOMs.
- Coalescing same-session batches at enqueue time. Replaces O(N²) re-merge with one ingest per session. Big win, separate change.
