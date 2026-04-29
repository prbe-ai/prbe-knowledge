# fix: launch-readiness — capacity bump + same-session coalescing + CC deprioritize

**Status:** in progress (started 2026-04-29).

## What ships in this PR

### A. Worker capacity bump (fly.worker.toml)

- count: 9 → 18
- WORKER_MAX_CONCURRENT: 2 → 4
- memory: 2gb → 3gb
- Net: 72 in-flight slots, ~360 batches/min ceiling, ~300+ concurrent CC sessions handled.

### B. Same-session enqueue coalescing for claude_code

Today's bug discovered during eng review: live CC batches land at date-partitioned R2 keys
(`raw/claude_code/<cust>/YYYY/MM/DD/<sess>:<batch>.json`) but `fetch_supplementary` only
merges from per-session prefix (`raw/claude_code/<cust>/<sess>/`) — different paths.
Result: each batch's processing only sees ITS OWN events. Session document gets
overwritten per batch with that 30s window. Chunk diff expires prior batch's chunks.
Net effect: only the latest batch is searchable per session. Silent data loss.

Fix:

1. **Migration** — ADD `payload_s3_keys text[] NOT NULL DEFAULT '{}'`,
   ADD `version int NOT NULL DEFAULT 0`, backfill existing rows with
   `payload_s3_keys = ARRAY[payload_s3_key]`. Do NOT drop `payload_s3_key`
   in this migration (avoids rolling-deploy race) — defer to follow-up.
2. **Ingestion enqueue** — claude_code uses UPSERT on
   (customer_id, source_system, session_id), appending key to array,
   bumping version. source_event_id becomes bare session_id (no :batch_seq).
   Other connectors INSERT with payload_s3_keys=ARRAY[$key],
   ON CONFLICT DO NOTHING. All connectors use the array column going forward.
3. **Worker** — `_claim_one` returns `version`. After Phase B commits,
   atomic UPDATE … SET status='done' WHERE queue_id=$1 AND version=$captured.
   Mismatch → row stays 'pending' for re-claim with extended array.
   Same-session NOT EXISTS clause from PR #33 deleted (dead code post-coalescing).
4. **CC connector** — `fetch_supplementary` reads payload_s3_keys, parallel-fetches
   via asyncio.Semaphore(16). Detects session_complete via either session_end event
   OR finalize.marker key in array.
5. **Session-completer cron** — UPSERTs into the live session row (appending
   finalize.marker to array, bumping version). Removes the legacy
   :finalize source_event_id flow for new triggers; legacy in-flight rows still drain.

### C. Per-source priority deprioritization

SOURCE_PRIORITY map (shared/constants.py):
- claude_code: 75
- other live (github/slack/notion/linear/granola/sentry): 100
- backfill: 50 (unchanged)

Tier order: live(100) > CC(75) > backfill(50).
Worker._claim_one already orders by priority DESC — no claim-side change.

## Out of scope (deferred)

- Drop payload_s3_key column (follow-up after this deploy stabilizes)
- Phase A intra-process version-check (handle burst thrash)
- Backfill fairness / max-wait priority bump
- Multi-region redundancy
- Killswitch (user implementing in separate session)

## Tests

- Coalescing happy path (3 batches → 1 row, array length 3, version 3)
- Phase B + CAS commit (version unchanged → done)
- CAS race (mid-process batch → version advances → CAS misses → row stays pending)
- Session resurrection (done → new batch → reopens to pending)
- Other connector regression (slack still works, single-element array)
- Priority claim ordering (github 100 beats CC 75 when both pending)
- Data-loss regression (full session body has events from ALL batches)
- Finalize via marker (cron upserts marker → worker detects → unit docs extracted)
- Migration backfill safety (existing rows get correct array)
- Worker crash between Phase B commit and CAS (reclaim recovers)

## Affected files

- alembic migration (new)
- fly.worker.toml
- shared/constants.py (SOURCE_PRIORITY)
- services/ingestion/main.py (_enqueue)
- services/ingestion/handlers/claude_code.py (parse_webhook_event, fetch_supplementary)
- services/ingestion/worker.py (_claim_one, _process)
- services/ingestion/normalizer.py (process_queue_row signature)
- services/ingestion/session_completer.py (rework finalize)
- tests/ (new test files)

## Eng-review findings (resolved)

- A1 migration backfill — auto-applied
- A2 watermark monotonicity — chose integer counter over timestamp
- A3 schema shape — chose unify-on-array over keep-both-columns
- B1 dead NOT EXISTS clause — removed in this PR
- D1 R2 fan-out memory — semaphore(16) cap
- Outside-voice T1 migration drop-column race — defer drop to follow-up
- Outside-voice T2 finalize clobber — UPSERT into live row
