# Entity Clusters — Manual Identity Merging via Dashboard

**Status:** design approved 2026-05-14, ready for implementation planning
**Repos touched:** `prbe-knowledge` (schema, `graph_writer`, retrieval, admin endpoints), `prbe-backend` (BFF thin wrappers), `prbe-dashboard` (UX — Phase 3)
**Author / approver:** mahit
**Date:** 2026-05-13 inception, 2026-05-14 pivot to physical merge
**Supersedes:** the 2026-05-13 logical-merge ("Approach A") draft of this same file. The pivot reasoning + the discarded approaches are in §"Design pivots during brainstorming" and §"Discarded approaches" at the bottom.

---

## Problem

`prbe-knowledge` represents real-world entities as `graph_nodes` rows keyed on `(customer_id, label, canonical_id)`. The same human appears under multiple raw IDs across source systems:

- GitHub login (`richardwei6`)
- GitHub commit email (`richard@prbe.ai`)
- `Co-authored-by:` trailer email (varies)
- Slack user ID (`U07ABC123`)
- Linear user ID (`user_a3f9`)
- Notion / Granola / Sentry — varies

Each lands as a separate `Person` node. Today's entity-matching surfaces (graph retriever anchor lookup, related-entities walker, list-mode author filter, entity post-filter, `id_lookup`) do exact string equality or lowercased substring containment — no record linkage. Filtering by `richardwei6` silently misses Richard's Slack-authored docs; the `related_entities` response lists Richard 4 times, eating the agent's top-K budget; cross-source identity is impossible to query.

The team has a designed-but-unbuilt `person_aliases` table in `TODOS.md` (P5, ~600 LOC), but nothing has shipped.

**This spec covers manual identity merging through the dashboard** — a user notices duplicates, clicks "these are the same person," and from that moment onward the graph physically reflects the cluster as one canonical node. Same-label only (Person↔Person, Repo↔Repo). Not for generic relationship-edge creation (deferred).

## Design pivots during brainstorming

This spec went through several serious design considerations across two sessions.

1. **2026-05-13: Logical merge (Approach A).** Sparse `entity_aliases` routing table; underlying graph untouched; read-time JOINs fuse cluster members at three retrieval surfaces. Reversible, ingest-untouched. **Initially approved as "final form" and a Phase 1 plan was written against it.**

2. **2026-05-13: Physical merge B-full (considered, then rejected).** Edge rewrites with property merge + edge-collision resolution. Rejected after Example 2 (TOUCHES collision losing one alias's `commit_count`) — property-merge complexity, audit complexity, and ingest-path changes outweighed read-time savings.

3. **2026-05-14: Pivot back to physical merge — B-promote with composite UNIQUE.** mahit revisited the design and rejected Approach A. The pivot's key insight: by extending `graph_edges` UNIQUE to include the alias provenance columns (`aliased_from_canonical_id`, `aliased_to_canonical_id`), the collision problem that killed B-full in the first round disappears entirely. Alias-rewritten edges coexist with the canonical's pre-existing edges as distinct rows in separate "alias lanes," each upsertable from its own webhook origin. No property data is lost on merge. The pivot also gave us simpler retrieval code paths (one canonical node, not N to fuse), accurate `degree` / surprise score per cluster, and a snapshot table that collapses to "self-loops only" because rewrites no longer need replay.

The discarded approaches (A, B-full, B-mint, B-soft, cluster-as-new-node-label) are documented at the bottom for posterity.

## Approach (B-promote with composite UNIQUE)

**Physical merge.** On merge, the operator picks one of the existing alias canonical_ids to be the cluster's **primary**. The merge transaction:

1. Snapshots every alias `graph_nodes` row (full pre-state including properties + provenance).
2. Rewrites every edge touching an alias node so its endpoint(s) point at the primary's `node_id`, **and stamps the original alias canonical_id into a new `aliased_from_canonical_id` / `aliased_to_canonical_id` column on that edge row**.
3. Drops self-loops created by the rewrite (matching Lane B Rule 5).
4. Merges alias provenance into the canonical's `graph_node_provenance` (`ON CONFLICT DO UPDATE` with min/max timestamps).
5. Hard-deletes the alias `graph_nodes` rows (CASCADE cleans their provenance).
6. Recomputes `degree` on the canonical (it just gained edges).
7. Inserts routing rows into `entity_aliases` so post-merge webhook ingest resolves the alias to the primary.
8. Inserts an `entity_merge_audit` row with actor + timestamp + reason.

**The crucial schema change** is on `graph_edges`: the existing `UNIQUE (customer_id, edge_type, from_node_id, to_node_id)` is replaced with:

```sql
UNIQUE (customer_id, edge_type, from_node_id, to_node_id,
        COALESCE(aliased_from_canonical_id, ''),
        COALESCE(aliased_to_canonical_id, ''))
```

After merge, multiple edges can connect the same `(canonical, doc)` pair if they came from different alias origins:

```
Person:richardwei6 → Doc-X [TOUCHES, {commit_count: 47, first_seen_sha: "abc"}]   aliased_from=NULL
Person:richardwei6 → Doc-X [TOUCHES, {commit_count: 23, first_seen_sha: "def"}]   aliased_from='mahit@prbe.ai'
Person:richardwei6 → Doc-X [TOUCHES, {commit_count: 12, first_seen_sha: "ghi"}]   aliased_from='U07ABC123'
```

Three rows. Each is upsertable from its own origin's webhooks. **No property data is lost on merge.** Reads that need "the cluster's edges" naturally find all three (all `from_node_id = richardwei6_node_id`). Reads that want a single aggregated view (e.g., a dashboard "total commit count") dedupe at the read layer (Phase 2) or rely on Lane B to maintain materialized aggregates (future work).

**Ingest is alias-aware** post-merge: `graph_writer.upsert_nodes` / `upsert_edges` consult `entity_aliases` and rewrite incoming aliased canonical_ids to the primary before the existing dedup logic runs. Each rewrite populates `aliased_from_canonical_id` / `aliased_to_canonical_id` on the INSERT row. Self-loops produced by the resolution (e.g., a webhook event linking two aliases of the same cluster) are dropped at the same point as Lane B Rule 5.

**Unmerge** is the inverse: re-INSERT the alias node from snapshot, UPDATE all edges with `aliased_from/to_canonical_id = <unmerged alias>` to point back at the restored alias node (clearing the provenance column), re-INSERT any snapshotted self-loops, recompute degree on both nodes, drop the `entity_aliases` routing row, and flip the audit row's status to `'reversed'` if no aliases remain under the merge_id.

## Trade-offs accepted

- **`graph_edges` row count grows post-merge.** Each alias contributes its own lane of edges; the table is N× the pre-merge edge count where N is the cluster size. Most clusters are 2-4 aliases; not a real storage concern. Indexes are unchanged in shape.
- **Read-time aggregation falls on Phase 2 retrievers** if they need "one cluster, one summed count" semantics. Default behavior (one row per alias lane) is correct for most retrieval but cluttered for dashboards. Phase 2 picks the aggregation strategy per-surface.
- **`graph_writer.py` ingest path now consults `entity_aliases`.** ~30 LOC at the top of `upsert_nodes`/`upsert_edges` for a single bulk lookup per batch. Performance impact: one extra SELECT per webhook batch (table is small, typically O(100s) rows per tenant). Acceptable.
- **Merge is irreversible IF the snapshot tables are pruned.** We don't prune them today; design assumes infinite retention. If retention becomes a concern, expose a "permanently abandon merge_id" cleanup that deletes the snapshot rows after some TTL.
- **`query_traces` replay produces different scores after a merge.** Eval tooling should note that scores drift after manual merges. Same caveat as Approach A.

## Schema

### New tables (five)

```sql
-- One row per merge action (a single user click). Lives forever.
CREATE TABLE entity_merge_audit (
    merge_id                    UUID PRIMARY KEY,
    customer_id                 TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                       TEXT NOT NULL,
    primary_canonical_id        TEXT NOT NULL,
    merged_alias_canonical_ids  TEXT[] NOT NULL,
    performed_by_user_id        UUID NOT NULL,
    performed_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reason                      TEXT NULL,
    status                      TEXT NOT NULL DEFAULT 'active'
                                CHECK (status IN ('active', 'reversed'))
);
CREATE INDEX idx_entity_merge_audit_primary
    ON entity_merge_audit (customer_id, label, primary_canonical_id);

-- Full pre-state of each alias graph_nodes row, captured at merge time.
-- Drives the re-INSERT step of unmerge. Provenance inlined as JSONB to
-- avoid a fourth table.
CREATE TABLE entity_merge_node_snapshot (
    merge_id      UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label         TEXT NOT NULL,
    canonical_id  TEXT NOT NULL,
    properties    JSONB NOT NULL,
    degree        INT  NOT NULL,
    community_id  INT  NULL,
    created_at    TIMESTAMPTZ NOT NULL,
    -- [{"source_system": "slack", "first_seen_at": "...", "last_seen_at": "..."}, ...]
    provenance    JSONB NOT NULL,
    PRIMARY KEY (merge_id, label, canonical_id)
);

-- Full pre-state of each edge that was deleted as a self-loop during merge.
-- Edges that were cleanly rewritten don't need a snapshot — unmerge reads
-- the live row's aliased_from/to columns and UPDATEs from_node_id back.
-- Edges that would have been collision-deleted in B-full don't exist under
-- the composite UNIQUE — they coexist as separate alias-lane rows.
CREATE TABLE entity_merge_edge_snapshot (
    merge_id                       UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
    snapshot_seq                   INT  NOT NULL,
    customer_id                    TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    operation                      TEXT NOT NULL
                                   CHECK (operation IN ('deleted_self_loop')),
    pre_edge_type                  TEXT NOT NULL,
    pre_from_canonical_id          TEXT NOT NULL,
    pre_from_label                 TEXT NOT NULL,
    pre_to_canonical_id            TEXT NOT NULL,
    pre_to_label                   TEXT NOT NULL,
    pre_properties                 JSONB NOT NULL,
    pre_confidence                 TEXT NOT NULL,
    pre_valid_from                 TIMESTAMPTZ NOT NULL,
    pre_valid_to                   TIMESTAMPTZ NULL,
    pre_source_system              TEXT NULL,
    pre_extractor_id               TEXT NULL,
    pre_extracted_at               TIMESTAMPTZ NULL,
    pre_aliased_from_canonical_id  TEXT NULL,
    pre_aliased_to_canonical_id    TEXT NULL,
    PRIMARY KEY (merge_id, snapshot_seq)
);
CREATE INDEX idx_entity_merge_edge_snapshot_merge
    ON entity_merge_edge_snapshot (merge_id);

-- Forward routing: which aliases route to which primary. Consulted by
-- graph_writer at ingest, by retrieval at anchor lookup (Phase 2), and
-- by the merge/unmerge endpoints for conflict detection.
CREATE TABLE entity_aliases (
    customer_id           TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                 TEXT NOT NULL,
    alias_canonical_id    TEXT NOT NULL,
    primary_canonical_id  TEXT NOT NULL,
    merge_id              UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
    added_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (customer_id, label, alias_canonical_id),
    CONSTRAINT entity_aliases_not_self CHECK (alias_canonical_id <> primary_canonical_id)
);
CREATE INDEX idx_entity_aliases_primary
    ON entity_aliases (customer_id, label, primary_canonical_id);
CREATE INDEX idx_entity_aliases_merge
    ON entity_aliases (merge_id);

-- Sparse display-name override for clusters and singletons. Phase 1 creates
-- the table but does not read or write it. Phase 3 dashboard adds the
-- PATCH endpoint that populates it.
CREATE TABLE entity_cluster_metadata (
    customer_id                  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                        TEXT NOT NULL,
    primary_canonical_id         TEXT NOT NULL,
    display_name                 TEXT NOT NULL,
    display_name_last_edited_by  UUID NULL,
    display_name_last_edited_at  TIMESTAMPTZ NULL,
    PRIMARY KEY (customer_id, label, primary_canonical_id)
);
```

**RLS:** every table is `ENABLE` + `FORCE ROW LEVEL SECURITY` with a `USING + WITH CHECK` `tenant_isolation` policy on `customer_id = current_setting('app.current_customer_id', true)` — same pattern as `graph_nodes` / `graph_edges` (migrations 0067, 0070).

### Changes to `graph_edges`

```sql
-- Add provenance columns. NULL for never-rewritten edges (the vast majority
-- pre-migration); set on alias-aware ingest or merge-time rewrite.
ALTER TABLE graph_edges
    ADD COLUMN aliased_from_canonical_id TEXT NULL,
    ADD COLUMN aliased_to_canonical_id   TEXT NULL;

-- Replace the existing UNIQUE constraint so different alias lanes coexist.
-- Constraint name is checked at migration-write time via `\d graph_edges`.
ALTER TABLE graph_edges
    DROP CONSTRAINT <existing_unique_constraint_name>;

ALTER TABLE graph_edges
    ADD CONSTRAINT graph_edges_unique_lane UNIQUE (
        customer_id, edge_type, from_node_id, to_node_id,
        COALESCE(aliased_from_canonical_id, ''),
        COALESCE(aliased_to_canonical_id, '')
    );
```

The COALESCE form works on any PG version; if the cluster is PG 15+ we can use `UNIQUE NULLS NOT DISTINCT` for the same effect with cleaner syntax. Migration picks one based on the target PG version.

`graph_writer.upsert_edges`'s existing `ON CONFLICT` clause needs to be updated to reference the new constraint name. 2-3 line change.

## Merge action

```
POST /knowledge/entity-clusters/merge        ← BFF thin wrapper (prbe-backend)
POST /api/entity-clusters/merge              ← actual handler (prbe-knowledge,
                                                gated by X-Internal-Knowledge-Key)

Body (BFF → knowledge):
{
  "customer_id":            "<from JWT session>",
  "performed_by_user_id":   "<from JWT session>",
  "label":                  "Person",
  "primary_canonical_id":   "richardwei6",
  "alias_canonical_ids":    ["mahit@prbe.ai", "U07ABC123", "user_a3f9"],
  "reason":                 "Confirmed same human via signed commit email + Slack profile"
}
```

### Merge transaction (prbe-knowledge, ~250 LOC)

```sql
BEGIN;
SELECT set_config('app.current_customer_id', $1, true);  -- with_tenant

-- 1. Validate: all canonical_ids exist as graph_nodes for this (customer, label).
--    404 with the missing list if any are absent.
SELECT canonical_id FROM graph_nodes
 WHERE customer_id = $1 AND label = $2 AND canonical_id = ANY($3::text[]);

-- 2. Validate: none of the aliases are already in another cluster.
--    409 with conflicting_aliases map if any are.
SELECT alias_canonical_id, primary_canonical_id FROM entity_aliases
 WHERE customer_id = $1 AND label = $2
   AND alias_canonical_id = ANY($3::text[]);

-- 3. Validate: primary is not itself an alias of another cluster.
--    409 with actual_primary if so.
SELECT primary_canonical_id FROM entity_aliases
 WHERE customer_id = $1 AND label = $2 AND alias_canonical_id = $3
 LIMIT 1;

-- 4. Look up node_ids for primary + every alias.
SELECT canonical_id, node_id, properties, degree, community_id, created_at
  FROM graph_nodes
 WHERE customer_id = $1 AND label = $2 AND canonical_id = ANY($3::text[]);
-- ($alias_node_ids, $primary_node_id derived in Python from this fetch.)

-- 5. Lock every edge touching any alias node. Held until COMMIT.
SELECT edge_id FROM graph_edges
 WHERE customer_id = $1
   AND (from_node_id = ANY($alias_node_ids) OR to_node_id = ANY($alias_node_ids))
 FOR UPDATE;

-- 6. Mint merge_id; INSERT audit row.
INSERT INTO entity_merge_audit
    (merge_id, customer_id, label, primary_canonical_id,
     merged_alias_canonical_ids, performed_by_user_id, reason)
VALUES (...);

-- 7. Snapshot every alias node (incl. inlined provenance JSONB).
INSERT INTO entity_merge_node_snapshot
    (merge_id, customer_id, label, canonical_id, properties, degree,
     community_id, created_at, provenance)
SELECT $merge_id, gn.customer_id, gn.label, gn.canonical_id,
       gn.properties, gn.degree, gn.community_id, gn.created_at,
       COALESCE(
         (SELECT jsonb_agg(jsonb_build_object(
            'source_system', p.source_system,
            'first_seen_at', p.first_seen_at,
            'last_seen_at',  p.last_seen_at))
          FROM graph_node_provenance p
          WHERE p.node_id = gn.node_id),
         '[]'::jsonb
       ) AS provenance
  FROM graph_nodes gn
 WHERE gn.customer_id = $1 AND gn.node_id = ANY($alias_node_ids);

-- 8. Merge alias provenance into the canonical's provenance.
INSERT INTO graph_node_provenance (node_id, customer_id, source_system, first_seen_at, last_seen_at)
SELECT $primary_node_id, p.customer_id, p.source_system,
       MIN(p.first_seen_at), MAX(p.last_seen_at)
  FROM graph_node_provenance p
 WHERE p.node_id = ANY($alias_node_ids)
 GROUP BY p.customer_id, p.source_system
ON CONFLICT (node_id, source_system) DO UPDATE
   SET first_seen_at = LEAST(graph_node_provenance.first_seen_at, EXCLUDED.first_seen_at),
       last_seen_at  = GREATEST(graph_node_provenance.last_seen_at, EXCLUDED.last_seen_at);

-- 9. For each edge touching an alias node, compute the post-rewrite endpoints
--    and classify in Python: "clean rewrite" vs. "self-loop after rewrite."
--    No "collision" case — composite UNIQUE allows different alias-lane
--    rows to coexist.
--
--    Clean rewrite (UPDATE the existing row):
UPDATE graph_edges
   SET from_node_id = $primary_node_id,
       aliased_from_canonical_id = $original_alias_canonical_id
 WHERE edge_id = $edge_id;
-- (Symmetric variants for to-side rewrite and both-sides rewrite.)
--
--    Self-loop (both endpoints resolve to the same canonical):
INSERT INTO entity_merge_edge_snapshot (merge_id, snapshot_seq, ..., operation, pre_*)
VALUES (...);
DELETE FROM graph_edges WHERE edge_id = $edge_id;

-- 10. Hard-delete alias graph_nodes. CASCADE drops their remaining
--     graph_node_provenance rows (already merged into canonical at step 8).
DELETE FROM graph_nodes
 WHERE customer_id = $1 AND node_id = ANY($alias_node_ids);

-- 11. Recompute degree on the canonical (it just gained edges from
--     the rewrites). For accuracy after merge.
UPDATE graph_nodes
   SET degree = (
       SELECT COUNT(*) FROM graph_edges
        WHERE customer_id = $1
          AND (from_node_id = $primary_node_id OR to_node_id = $primary_node_id)
   )
 WHERE customer_id = $1 AND node_id = $primary_node_id;

-- 12. INSERT one entity_aliases row per alias.
INSERT INTO entity_aliases
    (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
SELECT $1, $2, unnest($alias_canonical_ids), $primary_canonical_id, $merge_id;

COMMIT;
```

### Race conditions

- **Two concurrent merges on the same alias.** Second one fails on the `entity_aliases` PK; re-validation surfaces a clean 409 (the alias is already in another cluster — return the existing primary so the dashboard can refresh).
- **Merge concurrent with ingest writing to alias nodes.** The merge's step 5 `FOR UPDATE` on `graph_edges` blocks any concurrent edge writes touching the alias nodes. Once the merge commits, subsequent ingest reads the new `entity_aliases` rows and resolves to the primary naturally.
- **Merge concurrent with disconnect-integration.** `disconnect_integration` in `prbe-backend/apps/data_plane/routers/dashboard/knowledge.py` operates under `with_tenant`, so it doesn't take row-level locks on `graph_edges` upfront. If disconnect runs while merge is in flight, one of them will fail with a serialization error from conflicting deletes / updates. Acceptable — surfaces a clean error to the operator who can retry.

## Unmerge action

```
DELETE /knowledge/entity-clusters/{label}/{primary}/aliases/{alias}    ← BFF
DELETE /api/entity-clusters/{label}/{primary}/aliases/{alias}          ← prbe-knowledge
```

### Unmerge transaction

```sql
BEGIN;
SELECT set_config('app.current_customer_id', $1, true);  -- with_tenant

-- 1. Look up the merge_id from entity_aliases.
SELECT merge_id FROM entity_aliases
 WHERE customer_id = $1 AND label = $2 AND alias_canonical_id = $3;
-- 404 if no row.

-- 2. Re-INSERT the alias node from snapshot. Gets a fresh node_id (BIGSERIAL).
INSERT INTO graph_nodes
    (customer_id, label, canonical_id, properties, degree, community_id, created_at, updated_at)
SELECT customer_id, label, canonical_id, properties, degree, community_id, created_at, NOW()
  FROM entity_merge_node_snapshot
 WHERE merge_id = $merge_id AND label = $2 AND canonical_id = $3
 RETURNING node_id AS new_alias_node_id;

-- 3. Restore alias's provenance rows from the snapshot's JSONB column.
INSERT INTO graph_node_provenance (node_id, customer_id, source_system, first_seen_at, last_seen_at)
SELECT $new_alias_node_id, customer_id,
       p->>'source_system',
       (p->>'first_seen_at')::timestamptz,
       (p->>'last_seen_at')::timestamptz
  FROM entity_merge_node_snapshot, LATERAL jsonb_array_elements(provenance) AS p
 WHERE merge_id = $merge_id AND label = $2 AND canonical_id = $3;

-- 4. Rewrite edges back. The composite UNIQUE means restoring
--    (from_node_id = <alias>, aliased_from_canonical_id = NULL) returns
--    the row to its original "NULL alias lane." Safe because the NULL
--    lane for the alias's canonical_id is empty: post-merge ingest from
--    the alias resolved to the primary (aliased_from set), leaving
--    nothing in the alias's NULL lane during the merge window.
UPDATE graph_edges
   SET from_node_id = $new_alias_node_id,
       aliased_from_canonical_id = NULL
 WHERE customer_id = $1 AND aliased_from_canonical_id = $3;
UPDATE graph_edges
   SET to_node_id = $new_alias_node_id,
       aliased_to_canonical_id = NULL
 WHERE customer_id = $1 AND aliased_to_canonical_id = $3;

-- 5. Re-INSERT self-loop edges from snapshot (rare).
INSERT INTO graph_edges (...)
SELECT ... FROM entity_merge_edge_snapshot
 WHERE merge_id = $merge_id
   AND operation = 'deleted_self_loop'
   AND (pre_from_canonical_id = $3 OR pre_to_canonical_id = $3);

-- 6. Provenance handling on unmerge: do NOT subtract from canonical.
--    A merge that brought 'slack' provenance into the canonical via an
--    alias does NOT remove 'slack' from the canonical when the alias
--    is unmerged. Conservative — slight over-attribution preferred to
--    under-attribution that would break disconnect-integration cleanup.

-- 7. Recompute degree on the canonical (it just lost edges) and the
--    restored alias.
UPDATE graph_nodes
   SET degree = (
       SELECT COUNT(*) FROM graph_edges
        WHERE customer_id = $1
          AND (from_node_id = graph_nodes.node_id OR to_node_id = graph_nodes.node_id)
   )
 WHERE customer_id = $1 AND node_id IN ($primary_node_id, $new_alias_node_id);

-- 8. Drop the routing row + flip audit status if last alias.
DELETE FROM entity_aliases
 WHERE customer_id = $1 AND label = $2 AND alias_canonical_id = $3;
UPDATE entity_merge_audit
   SET status = 'reversed'
 WHERE merge_id = $merge_id
   AND NOT EXISTS (SELECT 1 FROM entity_aliases WHERE merge_id = $merge_id);

COMMIT;
```

### Unmerge edge cases

- **Whole-cluster unmerge:** call the per-alias endpoint repeatedly, or add a `DELETE /api/entity-clusters/{label}/{primary_canonical_id}` that loops in one transaction. Phase 1 ships the per-alias version; whole-cluster is a Phase 3 convenience.
- **Primary canonical_id has been soft-deleted by some other mechanism** (future SCD2): not relevant in Phase 1 — `graph_nodes` doesn't soft-delete today. If/when it does, unmerge would need a primary-election fallback. Out of scope.

## Read-side behavior (Phase 2 preview)

The pivot to B-promote dramatically simplifies retrieval-side code compared to Approach A:

- **Graph anchor lookup** finds ONE node per cluster (the primary). No anchor expansion needed. But it must consult `entity_aliases` to translate a user-typed alias canonical_id to the primary's canonical_id before the lookup.
- **`related_entities` walker** sees one node per cluster naturally. No IDF dedup logic, no exclude-key expansion.
- **`MatchProvenance` graph channel** has one entry per cluster naturally (no collapse).
- **`graph_evidence.via_entity`** points at the primary already.
- **Author filter (list mode)** must expand `author_ids` through `entity_aliases` because `documents.author_id` is the historical raw text and is never rewritten.
- **Surprise score / hub-anti-bonus / degree** are cluster-correct because `degree` is recomputed at merge time and maintained by graph_writer for all incoming edges (resolved or not).

`RelatedEntity` still gains `member_count` and `member_sources` fields (count of aliases + distinct source_systems across the cluster) for agent-facing metadata, even though retrieval doesn't need them internally. Computed from `entity_aliases` + `graph_node_provenance` of the primary.

**Phase 2 scope is therefore smaller than Approach A would have been:**
- Anchor-lookup translation (one bulk lookup at the router boundary)
- Author filter expansion (one bulk lookup at the list pipeline boundary)
- `RelatedEntity` field population (one query per result row, batchable)
- `entity_cluster_metadata` consulted for display name override

Three touchpoints vs. Approach A's three, but each is shorter and simpler.

## graph_writer.py ingest-path changes (Phase 1)

```python
# At top of upsert_nodes (BEFORE existing dedup/INSERT)
alias_map = await _fetch_aliases(
    conn, customer_id,
    keys=[(n.label.value, n.canonical_id) for n in nodes],
)
for n in nodes:
    primary = alias_map.get((n.label.value, n.canonical_id))
    if primary is not None:
        n.canonical_id = primary  # rewrite to canonical
# Existing dedup + INSERT logic runs on canonicalized nodes.

# At top of upsert_edges (BEFORE existing dedup/INSERT)
alias_map = await _fetch_aliases(conn, customer_id, _endpoint_keys(edges))
for e in edges:
    from_primary = alias_map.get((e.from_label, e.from_canonical_id))
    to_primary   = alias_map.get((e.to_label,   e.to_canonical_id))
    if from_primary is not None:
        e.aliased_from_canonical_id = e.from_canonical_id
        e.from_canonical_id = from_primary
    if to_primary is not None:
        e.aliased_to_canonical_id = e.to_canonical_id
        e.to_canonical_id = to_primary
    # Self-loop check after resolution (matches Lane B Rule 5).
    if (e.from_canonical_id == e.to_canonical_id and
        e.from_label == e.to_label):
        _inc(dropped, "self_edge_post_alias")
        continue  # skip — don't INSERT the self-loop.
# Existing ON CONFLICT upsert runs against the new composite UNIQUE.
# ON CONFLICT clause is updated to reference the new constraint name.
```

`_fetch_aliases` is one bulk query per batch:

```sql
SELECT label, alias_canonical_id, primary_canonical_id
FROM entity_aliases
WHERE customer_id = $1
  AND (label, alias_canonical_id) IN (
        SELECT * FROM UNNEST($2::text[], $3::text[])
      );
```

Returned rows → in-memory `dict[(label, alias_canonical_id), primary_canonical_id]`.

**Behavior pre-merge:** `entity_aliases` is empty. `alias_map` is empty. No rewrites. `aliased_from/to_canonical_id` stay NULL on every INSERT. Composite UNIQUE behaves identically to the old single-column UNIQUE (both NULLs coalesce to `''`). **No regression in ingest behavior until at least one merge happens.**

## Locked design decisions

| Area | Decision |
|---|---|
| Approach | **B-promote** (physical merge, alias edges rewritten to primary, alias nodes hard-deleted). |
| Cross-label | Deferred — same-label merging only in Phase 1. |
| Cluster external face | One of the existing alias canonical_ids (the primary). No synthetic id. |
| Edge collision on rewrite | **Composite UNIQUE includes `aliased_from/to_canonical_id`** — collisions don't happen; different alias lanes coexist as distinct rows. No property merge logic. |
| Property data preservation | Full preservation. Each lane's properties survive merge. |
| Edge dedup / aggregation | Phase 2+ concern — read-side aggregation or Lane B materialized rollups. Out of scope for Phase 1. |
| Self-loops created by merge | Deleted, snapshotted to `entity_merge_edge_snapshot` for unmerge. |
| Self-loops created by post-merge ingest | Dropped at `graph_writer` with `dropped.self_edge_post_alias` counter (matches Lane B Rule 5). |
| Alias `graph_nodes` after merge | Hard-deleted. Full pre-state (incl. inlined provenance JSONB) in `entity_merge_node_snapshot`. |
| Alias provenance handling | Merged into canonical's `graph_node_provenance` at merge time (`ON CONFLICT … DO UPDATE` min/max timestamps). |
| Degree maintenance | Recomputed in-transaction on merge AND unmerge for all affected nodes. |
| graph_writer alias resolution | Added in Phase 1 alongside the schema. Bulk lookup per batch. |
| Reversibility | Full. Unmerge re-INSERTs node from snapshot + UPDATEs edges via `aliased_from/to` columns. |
| Reversibility limitation | Provenance is NOT subtracted from canonical on unmerge (conservative — over-attribution preferred to disconnect-integration data loss). Documented. |
| `entity_aliases` schema | Sparse (only non-primary rows), PK on `(customer_id, label, alias_canonical_id)`, FK to `entity_merge_audit(merge_id)`. |
| `entity_cluster_metadata` | Shipped in Phase 1 migration, empty until Phase 3. Sparse display-name override; primary-keyed. |
| Endpoint location | **prbe-knowledge** owns the merge/unmerge transaction logic. **prbe-backend** ships a thin BFF wrapper that JWT-auths + forwards via `X-Internal-Knowledge-Key`. |
| `performed_by_user_id` | Injected by BFF from validated JWT session. UUID. |
| Customer scoping | `customer_id` injected by BFF from session; prbe-knowledge sets `app.current_customer_id` GUC; RLS enforces. |
| Permissioning | `admin` role minimum (`require_role("admin")` on BFF). |
| Strict validation on merge | Yes — 404 on missing canonical_id, 409 on already-aliased, 409 on primary-is-alias. |
| Multi-level merging | Forbidden at the API layer (409 if primary is already an alias of something else). |
| All five new tables RLS | `ENABLE` + `FORCE` + `USING/WITH CHECK` on customer_id. |

## Open items (for Phase 2/3)

1. **Phase 2 retrieval-side cluster awareness.** Anchor-lookup translation, author filter expansion, `RelatedEntity.member_count` / `member_sources` population, `entity_cluster_metadata` display-name resolution. Smaller scope than the Approach A version of Phase 2.
2. **Dashboard UX (Phase 3).** Duplicate-detection picker, merge preview, confirmation, conflict resolution (alias already in cluster X), audit log view, unmerge button, display-name override editor.
3. **Conflict resolution UX on re-merge.** When a user attempts to merge an alias already in cluster X, surface "this is already in cluster X — merge into X instead?" Phase 3.
4. **Phase 2 edge dedup strategy.** Cluster-aware aggregate views for the dashboard (e.g., total commit_count across all alias lanes). Options: read-side aggregation in the response builder, materialized rollup table maintained by Lane B, or "show per-source" UX that surfaces the lanes without summing. Decide during Phase 3 UX design.
5. **`entity_merge_edge_snapshot` and `entity_merge_node_snapshot` table pruning policy.** Today: kept forever. If retention becomes a concern, expose an "abandon merge_id" cleanup after some TTL. No urgency.
6. **MCP tool description updates.** `search_knowledge`'s description should note that merged entities show their primary canonical_id with `member_count > 1`. Phase 4.

## Implementation phasing

Three stacked PRs across two repos, each self-shippable:

| PR | Repo | Scope |
|---|---|---|
| **1a** | prbe-knowledge | Migration: 5 new tables (`entity_merge_audit`, `entity_merge_node_snapshot`, `entity_merge_edge_snapshot`, `entity_aliases`, `entity_cluster_metadata`) + 2 new columns on `graph_edges` + composite UNIQUE swap + RLS + indexes + `db/schema.sql` sync. `graph_writer.upsert_nodes`/`upsert_edges` alias-resolution helper. Migration tests + graph_writer alias-rewrite tests. **No API surface change** — ingest is alias-aware but `entity_aliases` is empty so it no-ops until merges happen. |
| **1b** | prbe-knowledge | `POST /api/entity-clusters/merge` + `DELETE /api/entity-clusters/{label}/{primary}/aliases/{alias}` behind `X-Internal-Knowledge-Key`. Live-DB integration tests covering full transaction (validate → snapshot → rewrite → delete → provenance-merge → degree-recompute → routing → audit). Endpoints exist but unreachable from the dashboard until 1c. |
| **1c** | prbe-backend | `POST /knowledge/entity-clusters/merge` + `DELETE /knowledge/entity-clusters/{label}/{primary}/aliases/{alias}` thin wrappers under `apps/data_plane/routers/dashboard/`. JWT + `require_role("admin")`. httpx-mocked unit tests. Dashboard becomes able to merge / unmerge once 1c deploys. |

Future phases:
- **Phase 2 (prbe-knowledge):** retrieval-side cluster awareness (smaller than the Approach A version). Anchor-lookup translation, author filter expansion, `RelatedEntity` field population, `entity_cluster_metadata` consultation for display name.
- **Phase 3 (prbe-dashboard + prbe-backend):** dashboard UX. Picker, preview, audit view, unmerge button, display-name editor + corresponding BFF endpoints.
- **Phase 4 (cross-cutting):** MCP tool description update, eval scripts noting query_traces score drift, optional Lane B materialized aggregates if Phase 3 UX surfaces a need.

## Honest concerns to revisit during implementation

- **`graph_edges` UNIQUE constraint swap is destructive at the DDL level.** The migration drops the existing constraint and adds the composite version. Mitigation: name the new constraint explicitly (`graph_edges_unique_lane`); audit known callers (only `graph_writer.upsert_edges` references the old `ON CONFLICT`-targeted columns by my read); consider doing this in a maintenance window even though the operation itself is fast.
- **`graph_writer` alias-resolution lookup is a new SELECT in the hottest write path.** Profile before shipping 1a. For an empty `entity_aliases`, the query is index-scanned and returns immediately — cost should be sub-millisecond per batch. Verify with EXPLAIN ANALYZE on probe-founders before deploying.
- **Drift risk for new retrieval surfaces.** Every retrieval surface added going forward must consult `entity_aliases` for alias→primary translation (at anchor lookup AND at author-filter expansion). Mitigation: a shared `resolve_aliases()` helper in `services/retrieval/helpers.py` that future surfaces can call without re-implementing.
- **Unmerge restores degree from snapshot, but the canonical's degree may have drifted between merge and unmerge** due to subsequent ingest. We recompute degree on unmerge for both the restored alias and the canonical, so the canonical's degree is correct post-unmerge. The restored alias's degree is correct relative to its restored edges (computed by COUNT). ✓
- **Provenance is NOT subtracted from canonical on unmerge.** A merge that brought Slack provenance into the canonical via an alias does NOT remove that provenance when the alias is unmerged. The canonical retains the Slack source_system entry even if no Slack-authored edges remain. Conservative — prevents disconnect-integration from missing data. Documented limitation.
- **Lane B's bundle quality** for merged identities. Lane B's bundle reads 1-hop neighbors of an anchor. After merge, the bundle sees the primary's full edge set (including rewritten alias lanes). Should "just work" but verify on a real cluster before declaring victory.

## Discarded approaches (for posterity)

**Approach A (logical merge with `entity_aliases` routing, edges untouched).** Originally chosen as "final form" in the 2026-05-13 session. Rejected in 2026-05-14 in favor of physical merge. The simpler-merge / simpler-unmerge story was real, but the cost of threading "fuse cluster members" through every retrieval surface and the inability to get accurate per-cluster `degree` / surprise scores at read time were the decisive cons.

**Approach B-full (physical merge with property merge on collision).** Considered in the 2026-05-13 session. Cleaner schema and faster reads than A, but the property-merge data loss on Example 2 (TOUCHES collision losing one alias's `commit_count`) was the deciding factor against. The 2026-05-14 pivot resolved this by using composite UNIQUE so collisions don't happen — property data lives in separate lanes.

**Approach B-mint (mint a new synthetic canonical_id for the cluster).** Considered briefly. Avoids the asymmetry of one alias being "promoted" but the synthetic canonical_id doesn't map to any source-system id, so dashboards can't show a recognizable name without an `entity_cluster_metadata` JOIN. B-promote keeps a recognizable canonical_id naturally.

**Approach B-soft (aliases TEXT[] on `graph_nodes`).** Considered in the original 2026-05-13 session. The `WHERE canonical_id = $1 OR $1 = ANY(aliases)` pattern breaks B-tree index utilization on the hottest entity-anchored search path, causing measurable regression. ALTER on `graph_nodes` (a hot, large table) was an additional concern. Rejected.

**Cluster as a new graph_node label.** Considered briefly. Adds a node namespace ("Cluster") that retrievers would need to follow, but doesn't actually simplify any read path vs. the alias-table approach. Rejected.

---

*Spec authored during the 2026-05-13 brainstorming session as Approach A; revised in the 2026-05-14 session as Approach B-promote after mahit reconsidered the physical-merge tradeoffs. See conversation history for full rationale and the discarded alternatives.*
