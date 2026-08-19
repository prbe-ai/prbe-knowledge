# /knowledge stats latency — fixes 1-5

Baseline measured on do-sfo3-probe-research, tenant `probe` (19.5k docs / 101.5k chunks):

| query    | cold    | warm   | buffer accesses |
|----------|---------|--------|-----------------|
| docs     | 5187 ms |  64 ms |          10,533 |
| chunks   | 2936 ms | 450 ms |         148,808 |
| queue    |   49 ms |        |                 |
| backfill |    9 ms |        |                 |
| TOTAL    | 8181 ms | ~550ms |                 |

Relay timeout is 10s (research-os `app/integrations/ingestion_router.py:47`) —
8.2s cold leaves no margin, and the dashboard swallows the resulting 502.

## Discoveries that change the plan

1. **The doc-liveness join filter is load-bearing.** ~3% of live chunks belong to
   documents that are superseded or soft-deleted (probe 3090, anthrogen 1083,
   new-workspace 377). Dropping the join to `documents` would overcount.
   `(customer_id, doc_id)` is unique among live non-deleted docs, so it is 1:1.

2. **autovacuum has NEVER run on `chunks` or `documents`.** `autovacuum_count = 0`,
   `last_autovacuum = NULL`, while every neighbouring table vacuums normally.
   chunks: 397k dead / 575k live (41% bloat). documents: 10.3k dead / 9.9k live.
   This GATES fixes 1+2 — index-only scans need the visibility map, which only
   VACUUM populates. Without a vacuum the new indexes fall back to heap fetches.

3. **Migration 0100's backfill silently did nothing.** The migrate job runs as
   `app` (rolsuper=f, rolbypassrls=f) and `chunks`/`documents` are FORCE RLS, so an
   unscoped `UPDATE` matches zero rows and still reports success. Evidence:
   chunks created after 0100 -> 313,230 rows, 0 empty titles;
   chunks created before  -> 79,688 rows, 78,192 empty (98.1%).
   BM25 title matching has been dead for every pre-2026-08-05 document.
   => Any backfill here MUST loop per customer and bind app.current_customer_id.

## Result (measured on the research plane, tenant `probe`)

| query  | before cold | before warm | after  | buffers before | buffers after |
|--------|-------------|-------------|--------|----------------|---------------|
| docs   |     5187 ms |       64 ms |  28 ms |         10,533 |         2,359 |
| chunks |     2936 ms |      450 ms |  92 ms |        148,808 |        72,182 |

Both aggregates are now Index Only Scans. The docs query keeps 15 heap fetches,
the chunks query 10,058 -- the chunks residue is the missing visibility map
(discovery 2), and it disappears once `chunks` can be vacuumed. The chunks plan
also flipped direction: it now drives from the 19.7k live documents into the
chunk index rather than scanning every live chunk and looking each document up.

Index sizes: idx_chunks_stats_live 3.4 MB, idx_documents_stats_live 2.5 MB.
Build times on the live DB: 16.8s and 4.9s, plain (not CONCURRENTLY).

## Tasks

- [x] Fix 2: covering indexes migration (chunks + documents)
- [x] Fix 4: run the four stats queries concurrently
- [x] Fix 3: TTL cache on the stats endpoint (+ `?refresh=true` bypass, relayed)
- [ ] Fix 1: denormalize source_system onto chunks — NOT LANDED, see below
- [x] Fix 5: dashboard stops gating the header + stops swallowing the error
- [x] VACUUM documents (25.6s, 10,345 dead tuples reclaimed, visibility map built)
- [ ] VACUUM chunks — BLOCKED, see discovery 4

## Discovery 4: `idx_chunks_bm25_v2` is corrupt

`VACUUM (ANALYZE) chunks` fails outright:

    could not open file "base/16643/3204790.1" (target block 6815897):
    previous segment is only 107882 blocks

relfilenode 3204790 is `idx_chunks_bm25_v2`, the pg_search BM25 index from
migration 0102 (843 MB). This is the whole explanation for discovery 2:
autovacuum must process every index on a table, so it dies here on every run and
`chunks` has never been vacuumed. The index also reports idx_scan = 0, so BM25
keyword retrieval over chunks is probably degraded as well.

Repair is `REINDEX INDEX CONCURRENTLY idx_chunks_bm25_v2`, which is an 843 MB
rebuild against the retrieval path — not attempted here, and recorded in
~/feedback.md for a human. Until it happens the chunks aggregate keeps its
10,058 heap fetches.

## Fix 1 was built and deliberately not landed

The plan was to denormalize `source_system` onto `chunks` so the count needs no
join, mirroring what 0100 did for `title`. The measurement says do not:

* The join is now the CHEAP side. Of the chunks query's 72,182 buffers, the
  documents index-only scan is 2,448 — about 3%. Fix 1 optimises that 3%.
* What actually remains are the 10,058 heap fetches on the chunks index-only
  scan, and Fix 1 does not touch them. Only a vacuum does, which is blocked on
  discovery 4.
* The cost is a ~575k-row UPDATE on a table already carrying 397k dead tuples
  that provably CANNOT be vacuumed. It would roughly double the bloat on the
  one table in the database with no way to reclaim it.
* It adds a permanent drift surface (two triggers) for a number nothing else
  reads.

Revisit after the BM25 index is repaired and `chunks` vacuums cleanly — at which
point the remaining cost is the join, and Fix 1 becomes the right next step
rather than the wrong one.
