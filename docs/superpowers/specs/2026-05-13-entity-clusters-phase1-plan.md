# Entity Clusters — Phase 1 Implementation Plan (B-promote)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the schema + write-path + endpoints needed for manual entity merging via the B-promote design. Three stacked PRs across two repos. After Phase 1, an operator with the `admin` role can call the BFF to merge/unmerge entity clusters; the graph physically reflects the cluster as one canonical node with alias-lane provenance preserved on every rewritten edge. Phase 2 (retrieval) and Phase 3 (dashboard UX) are out of scope.

**Architecture:** Physical merge with composite UNIQUE on `graph_edges` keyed by `(customer_id, edge_type, from_node_id, to_node_id, COALESCE(aliased_from_canonical_id, ''), COALESCE(aliased_to_canonical_id, ''))`. Alias edges are rewritten to point at the primary; the original alias canonical_id is stamped into `aliased_from/to_canonical_id` columns, letting multiple alias-lane rows coexist. Alias `graph_nodes` are hard-deleted with full pre-state in `entity_merge_node_snapshot`; only self-loops require `entity_merge_edge_snapshot` entries. `graph_writer` consults `entity_aliases` at ingest so post-merge webhooks resolve to the primary.

**Design doc:** `docs/superpowers/specs/2026-05-13-entity-clusters-design.md` (in this same dir).

**Tech Stack:** Python 3.12, FastAPI, asyncpg, Pydantic v2, Alembic, Postgres (target 16; design works on 14+), pytest, pytest-asyncio.

**Worktrees:**
- `prbe-knowledge`: `/Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1` (branch `entity-clusters-phase1`)
- `prbe-backend`: `/Users/mahitnamburu/Desktop/prbe/prbe-backend-worktrees/entity-clusters-phase1` (branch `entity-clusters-phase1`)

Both worktrees already exist from earlier in this session.

---

## File Structure

### PR 1a — prbe-knowledge (schema + graph_writer)

| File | Action | Responsibility |
|---|---|---|
| `db/migrations/versions/20260514_0071_entity_clusters.py` | Create | Alembic migration: 5 new tables, 2 new columns on `graph_edges`, composite UNIQUE swap, RLS policies, indexes. |
| `db/schema.sql` | Modify | Append new tables + RLS + insert new `graph_edges` columns and UNIQUE constraint. |
| `shared/models.py` | Modify | Add `aliased_from_canonical_id` / `aliased_to_canonical_id` to `GraphEdgeSpec`. |
| `services/ingestion/graph_writer.py` | Modify | Add `_fetch_aliases` helper. Wire into `upsert_nodes` (rewrites `n.canonical_id`) and `upsert_edges` (rewrites endpoints + stamps `aliased_from/to_canonical_id` + drops self-loops). Update `ON CONFLICT` to use the new constraint name. |
| `tests/test_entity_clusters_migration.py` | Create | Pin schema invariants: PK/CHECK/RLS on all 5 new tables, composite UNIQUE behavior on `graph_edges`, cross-tenant denial. |
| `tests/test_graph_writer_alias_resolution.py` | Create | Pin alias-resolution behavior: empty `entity_aliases` is no-op, node alias rewrites canonical_id, edge alias rewrites endpoints + populates `aliased_from/to`, post-resolution self-loops are dropped. |

### PR 1b — prbe-knowledge (merge + unmerge endpoints)

| File | Action | Responsibility |
|---|---|---|
| `services/ingestion/entity_clusters_routes.py` | Create | FastAPI router under `/api/entity-clusters/*` gated by `X-Internal-Knowledge-Key`. `POST /merge` and `DELETE /{label}/{primary}/aliases/{alias}`. Pydantic request/response models. The full merge + unmerge transactions live here. |
| `services/ingestion/main.py` | Modify | Import + `include_router(entity_clusters_router)`. |
| `tests/test_entity_clusters_routes.py` | Create | Live-DB integration tests covering: merge happy-path, merge validation errors (404 / 409 alias-already / 409 primary-is-alias / 422), unmerge happy-path, unmerge 404, audit-status flip on last-alias-removed, post-merge `graph_writer` ingest correctly resolves to primary. |

### PR 1c — prbe-backend (BFF thin wrappers)

| File | Action | Responsibility |
|---|---|---|
| `apps/data_plane/routers/dashboard/entity_clusters.py` | Create | Thin BFF wrappers under `/knowledge/entity-clusters/*`. JWT-validated session + `require_role("admin")`. Forwards to prbe-knowledge with `X-Internal-Knowledge-Key`, injecting `customer_id` + `performed_by_user_id` from session. |
| `apps/data_plane/routers/dashboard/__init__.py` | Modify | Import + `include_router(entity_clusters_router)`. |
| `tests/test_dashboard_entity_clusters.py` | Create | TestClient + httpx mock tests covering happy paths, 403 for member role, 4xx bubble-up. |

---

## PR 1a — prbe-knowledge: schema migration + graph_writer alias resolution

**Worktree:** `cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1`

### Task A1: Inspect existing schema + record findings

We need to know two things before writing the migration:
- The auto-generated name of the existing `UNIQUE` constraint on `graph_edges`. Postgres auto-names UNIQUE constraints; the exact string is needed for the DROP CONSTRAINT statement.
- The Postgres version (PG 15+ enables `UNIQUE NULLS NOT DISTINCT`; otherwise we use the COALESCE form).

- [ ] **Step 1: Start the local DB if not running**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1
docker compose up -d
scripts/neon-migrate.sh local
```

Expected: Migrations apply cleanly through `0070_gnp_rls`.

- [ ] **Step 2: Capture the existing constraint name**

```bash
psql postgresql://prbe:prbe@localhost:5432/prbe_knowledge -c "
SELECT conname
FROM pg_constraint
WHERE conrelid = 'graph_edges'::regclass
  AND contype = 'u';
"
```

Expected output: a single row with the constraint name. The default for Postgres is:

```
graph_edges_customer_id_edge_type_from_node_id_to_node_id_key
```

Record the exact value — it goes into the migration's `DROP CONSTRAINT`. If your local matches the default, use it verbatim in Task A3.

- [ ] **Step 3: Capture the Postgres version**

```bash
psql postgresql://prbe:prbe@localhost:5432/prbe_knowledge -c "SHOW server_version;"
```

Record the major version. If >= 15, you can use `UNIQUE NULLS NOT DISTINCT` in the migration (cleaner syntax). If < 15, use the `COALESCE(...,'')` form. **The plan below uses the COALESCE form** for portability — adapt if you'd rather use NULLS NOT DISTINCT and you've confirmed PG 15+.

### Task A2: Write the failing migration test

**Files:**
- Create: `tests/test_entity_clusters_migration.py`

- [ ] **Step 1: Write the test file**

```python
"""Migration assertions for entity_clusters (0071).

Pins schema-level invariants for the manual-entity-merge tables:

  * Five new tables exist with the right columns / PKs / CHECKs.
  * entity_aliases PK uniqueness + entity_aliases_not_self CHECK fire.
  * entity_merge_audit.status CHECK rejects unknown values.
  * entity_merge_edge_snapshot.operation CHECK rejects unknown ops.
  * graph_edges gains aliased_from_canonical_id + aliased_to_canonical_id.
  * graph_edges UNIQUE now includes the alias provenance columns, so two
    rows with the same (edge_type, from, to) coexist if they differ in
    aliased_from_canonical_id.
  * graph_edges UNIQUE still dedups the common-case "both aliased_from
    cols NULL" inserts (the existing graph_writer upsert path).
  * RLS is ENABLE + FORCE + USING + WITH CHECK on all 5 new tables;
    cross-tenant SELECT under tenant A's GUC returns zero rows for B's
    data; cross-tenant INSERT is rejected by WITH CHECK.
"""

from __future__ import annotations

import uuid

import asyncpg
import pytest

from shared.db import raw_conn, with_tenant


async def _seed_customer(conn: asyncpg.Connection, customer_id: str) -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'mig', 'mig-hash') ON CONFLICT DO NOTHING",
        customer_id,
    )


async def _seed_person_node(
    conn: asyncpg.Connection, customer_id: str, canonical_id: str
) -> int:
    """Insert one Person graph_nodes row; return its node_id."""
    row = await conn.fetchrow(
        """
        INSERT INTO graph_nodes(customer_id, label, canonical_id)
        VALUES ($1, 'Person', $2)
        ON CONFLICT (customer_id, label, canonical_id) DO UPDATE
            SET updated_at = NOW()
        RETURNING node_id
        """,
        customer_id,
        canonical_id,
    )
    return row["node_id"]


async def _seed_doc_node(
    conn: asyncpg.Connection, customer_id: str, canonical_id: str
) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO graph_nodes(customer_id, label, canonical_id)
        VALUES ($1, 'Document', $2)
        ON CONFLICT (customer_id, label, canonical_id) DO UPDATE
            SET updated_at = NOW()
        RETURNING node_id
        """,
        customer_id,
        canonical_id,
    )
    return row["node_id"]


async def _insert_audit(
    conn: asyncpg.Connection,
    *,
    customer_id: str,
    label: str,
    primary: str,
    aliases: list[str],
    user_id: uuid.UUID,
) -> uuid.UUID:
    merge_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entity_merge_audit
          (merge_id, customer_id, label, primary_canonical_id,
           merged_alias_canonical_ids, performed_by_user_id)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        merge_id, customer_id, label, primary, aliases, user_id,
    )
    return merge_id


# ---------------------------------------------------------------------------
# Table existence + PK / CHECK behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_aliases_pk_rejects_duplicate_alias(live_db) -> None:
    cust = "mig-ec-pk"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await _seed_person_node(conn, cust, "p1")
        await _seed_person_node(conn, cust, "a1")
        await _seed_person_node(conn, cust, "p2")
        merge1 = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="p1", aliases=["a1"], user_id=user_id,
        )
        merge2 = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="p2", aliases=["a1"], user_id=user_id,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            VALUES ($1, 'Person', 'a1', 'p1', $2)
            """,
            cust, merge1,
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO entity_aliases
                  (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
                VALUES ($1, 'Person', 'a1', 'p2', $2)
                """,
                cust, merge2,
            )


@pytest.mark.asyncio
async def test_entity_aliases_check_rejects_self_alias(live_db) -> None:
    cust = "mig-ec-self"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await _seed_person_node(conn, cust, "richard")
        merge_id = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="richard", aliases=[], user_id=user_id,
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_aliases
                  (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
                VALUES ($1, 'Person', 'richard', 'richard', $2)
                """,
                cust, merge_id,
            )


@pytest.mark.asyncio
async def test_entity_merge_audit_status_check(live_db) -> None:
    cust = "mig-ec-status"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_merge_audit
                  (merge_id, customer_id, label, primary_canonical_id,
                   merged_alias_canonical_ids, performed_by_user_id, status)
                VALUES ($1, $2, 'Person', 'x', ARRAY['y']::text[], $3, 'bogus')
                """,
                uuid.uuid4(), cust, user_id,
            )


@pytest.mark.asyncio
async def test_entity_merge_edge_snapshot_operation_check(live_db) -> None:
    cust = "mig-ec-op"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        merge_id = await _insert_audit(
            conn, customer_id=cust, label="Person",
            primary="x", aliases=["y"], user_id=user_id,
        )
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_merge_edge_snapshot
                  (merge_id, snapshot_seq, customer_id, operation,
                   pre_edge_type, pre_from_canonical_id, pre_from_label,
                   pre_to_canonical_id, pre_to_label, pre_properties,
                   pre_confidence, pre_valid_from)
                VALUES ($1, 1, $2, 'bogus_op',
                        'AUTHORED', 'a', 'Person', 'b', 'Document',
                        '{}'::jsonb, 'EXTRACTED', NOW())
                """,
                merge_id, cust,
            )


# ---------------------------------------------------------------------------
# graph_edges schema changes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graph_edges_has_alias_columns(live_db) -> None:
    """The two new nullable TEXT columns exist on graph_edges."""
    async with raw_conn() as conn:
        cols = await conn.fetch(
            """
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'graph_edges'
              AND column_name IN ('aliased_from_canonical_id', 'aliased_to_canonical_id')
            ORDER BY column_name
            """
        )
    assert [c["column_name"] for c in cols] == [
        "aliased_from_canonical_id",
        "aliased_to_canonical_id",
    ]


@pytest.mark.asyncio
async def test_graph_edges_composite_unique_allows_alias_lanes(live_db) -> None:
    """Same (edge_type, from, to) with different aliased_from coexist."""
    cust = "mig-ec-lanes"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        p_node = await _seed_person_node(conn, cust, "richard")
        d_node = await _seed_doc_node(conn, cust, "doc-1")
        # Lane 1: NULL aliased_from
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               source_system, confidence)
            VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED')
            """,
            cust, p_node, d_node,
        )
        # Lane 2: aliased_from='mahit@prbe.ai'
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               source_system, confidence, aliased_from_canonical_id)
            VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED', 'mahit@prbe.ai')
            """,
            cust, p_node, d_node,
        )
        # Both rows persist
        rows = await conn.fetch(
            """
            SELECT aliased_from_canonical_id FROM graph_edges
            WHERE customer_id = $1 AND from_node_id = $2 AND to_node_id = $3
            ORDER BY edge_id
            """,
            cust, p_node, d_node,
        )
    assert [r["aliased_from_canonical_id"] for r in rows] == [None, "mahit@prbe.ai"]


@pytest.mark.asyncio
async def test_graph_edges_composite_unique_still_dedups_null_lane(live_db) -> None:
    """The common case (both aliased_from cols NULL) still dedups via upsert."""
    cust = "mig-ec-nulldedup"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        p_node = await _seed_person_node(conn, cust, "richard")
        d_node = await _seed_doc_node(conn, cust, "doc-1")
        # Two inserts in the NULL lane should collide on the composite UNIQUE.
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               source_system, confidence)
            VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED')
            """,
            cust, p_node, d_node,
        )
        with pytest.raises(asyncpg.UniqueViolationError):
            await conn.execute(
                """
                INSERT INTO graph_edges
                  (customer_id, edge_type, from_node_id, to_node_id,
                   source_system, confidence)
                VALUES ($1, 'AUTHORED', $2, $3, 'github', 'EXTRACTED')
                """,
                cust, p_node, d_node,
            )


# ---------------------------------------------------------------------------
# RLS cross-tenant denial
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_entity_aliases_rls_cross_tenant_denied(live_db) -> None:
    cust_a = "mig-ec-rls-a"
    cust_b = "mig-ec-rls-b"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust_a)
        await _seed_customer(conn, cust_b)

    async with with_tenant(cust_a) as conn:
        await _seed_person_node(conn, cust_a, "pa")
        await _seed_person_node(conn, cust_a, "aa")
        merge_id_a = await _insert_audit(
            conn, customer_id=cust_a, label="Person",
            primary="pa", aliases=["aa"], user_id=user_id,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            VALUES ($1, 'Person', 'aa', 'pa', $2)
            """,
            cust_a, merge_id_a,
        )

    # Read under B sees zero of A's rows.
    async with with_tenant(cust_b) as conn:
        rows = await conn.fetch(
            "SELECT alias_canonical_id FROM entity_aliases WHERE label = 'Person'"
        )
        assert rows == []

    # Write under A with customer_id = B → WITH CHECK rejects.
    async with with_tenant(cust_b) as conn:
        await _seed_person_node(conn, cust_b, "pb")
        await _seed_person_node(conn, cust_b, "ab")
        merge_id_b = await _insert_audit(
            conn, customer_id=cust_b, label="Person",
            primary="pb", aliases=["ab"], user_id=user_id,
        )
    async with with_tenant(cust_a) as conn:
        with pytest.raises(asyncpg.CheckViolationError):
            await conn.execute(
                """
                INSERT INTO entity_aliases
                  (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
                VALUES ($1, 'Person', 'ab', 'pb', $2)
                """,
                cust_b, merge_id_b,
            )


@pytest.mark.asyncio
async def test_entity_merge_audit_rls_cross_tenant_denied(live_db) -> None:
    cust_a = "mig-ec-audit-rls-a"
    cust_b = "mig-ec-audit-rls-b"
    user_id = uuid.uuid4()
    async with raw_conn() as conn:
        await _seed_customer(conn, cust_a)
        await _seed_customer(conn, cust_b)
    async with with_tenant(cust_a) as conn:
        await _insert_audit(
            conn, customer_id=cust_a, label="Person",
            primary="pa", aliases=["aa"], user_id=user_id,
        )
    async with with_tenant(cust_b) as conn:
        rows = await conn.fetch(
            "SELECT merge_id FROM entity_merge_audit WHERE label = 'Person'"
        )
        assert rows == []
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
pytest tests/test_entity_clusters_migration.py -v
```

Expected: All tests fail with `UndefinedTableError: relation "entity_aliases" does not exist` (or the migration assertions fail).

### Task A3: Write the migration

**Files:**
- Create: `db/migrations/versions/20260514_0071_entity_clusters.py`

- [ ] **Step 1: Write the migration file**

Replace `<EXISTING_UNIQUE_NAME>` with the value captured in Task A1, Step 2 (likely `graph_edges_customer_id_edge_type_from_node_id_to_node_id_key`).

```python
"""Entity clusters: tables + graph_edges changes for manual identity merging (B-promote).

Revision ID: 0071_entity_clusters
Revises: 0070_gnp_rls
Create Date: 2026-05-14

Five new RLS-isolated tables backing manual entity merging (design doc:
``docs/superpowers/specs/2026-05-13-entity-clusters-design.md``):

  * ``entity_merge_audit`` — one row per merge action.
  * ``entity_merge_node_snapshot`` — full pre-state of each hard-deleted
    alias node (incl. inlined provenance JSONB).
  * ``entity_merge_edge_snapshot`` — pre-state of each edge deleted as a
    self-loop during merge (the only deletion type under composite UNIQUE).
  * ``entity_aliases`` — forward routing (alias → primary), consulted by
    ``graph_writer`` at ingest and by retrieval at anchor lookup (Phase 2).
  * ``entity_cluster_metadata`` — sparse display-name override; created
    empty, populated only by the Phase 3 dashboard.

Plus two ``graph_edges`` schema changes:

  * New nullable columns ``aliased_from_canonical_id``,
    ``aliased_to_canonical_id`` — provenance for alias-resolved or
    merge-rewritten edges. NULL for all existing rows.
  * The existing single-column UNIQUE
    ``(customer_id, edge_type, from_node_id, to_node_id)`` is replaced by
    a composite UNIQUE that COALESCEs the two new alias columns to ''.
    This lets different alias lanes coexist as distinct rows (post-merge),
    while still dedupping the common-case "both aliased_from cols NULL"
    inserts (the existing graph_writer upsert path).

Migration is purely additive: all new tables are empty at creation, and
the column ADDs on graph_edges are instant (ADD COLUMN on a nullable TEXT
is O(1) in PG 11+).

Verification
------------
::

    \\d entity_merge_audit
    \\d entity_aliases
    \\d entity_cluster_metadata

    SELECT relname, relrowsecurity AS rls, relforcerowsecurity AS forced
    FROM   pg_class
    WHERE  relname IN ('entity_merge_audit', 'entity_aliases',
                       'entity_cluster_metadata',
                       'entity_merge_node_snapshot',
                       'entity_merge_edge_snapshot');

    \\d graph_edges  -- verify the new constraint and columns
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0071_entity_clusters"
down_revision: str | Sequence[str] | None = "0070_gnp_rls"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


_POLICY_EXPR = "customer_id = current_setting('app.current_customer_id', true)"


def upgrade() -> None:
    # ----- entity_merge_audit (created first; entity_aliases FKs it) -----
    op.execute(
        """
        CREATE TABLE entity_merge_audit (
            merge_id                    UUID PRIMARY KEY,
            customer_id                 TEXT NOT NULL
                                        REFERENCES customers(customer_id)
                                        ON DELETE CASCADE,
            label                       TEXT NOT NULL,
            primary_canonical_id        TEXT NOT NULL,
            merged_alias_canonical_ids  TEXT[] NOT NULL,
            performed_by_user_id        UUID NOT NULL,
            performed_at                TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            reason                      TEXT NULL,
            status                      TEXT NOT NULL DEFAULT 'active'
                                        CHECK (status IN ('active', 'reversed'))
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_entity_merge_audit_primary "
        "ON entity_merge_audit (customer_id, label, primary_canonical_id)"
    )

    # ----- entity_merge_node_snapshot -----
    op.execute(
        """
        CREATE TABLE entity_merge_node_snapshot (
            merge_id      UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
            customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            label         TEXT NOT NULL,
            canonical_id  TEXT NOT NULL,
            properties    JSONB NOT NULL,
            degree        INT  NOT NULL,
            community_id  INT  NULL,
            created_at    TIMESTAMPTZ NOT NULL,
            provenance    JSONB NOT NULL,
            PRIMARY KEY (merge_id, label, canonical_id)
        )
        """
    )

    # ----- entity_merge_edge_snapshot -----
    op.execute(
        """
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
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_entity_merge_edge_snapshot_merge "
        "ON entity_merge_edge_snapshot (merge_id)"
    )

    # ----- entity_aliases -----
    op.execute(
        """
        CREATE TABLE entity_aliases (
            customer_id           TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            label                 TEXT NOT NULL,
            alias_canonical_id    TEXT NOT NULL,
            primary_canonical_id  TEXT NOT NULL,
            merge_id              UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
            added_at              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (customer_id, label, alias_canonical_id),
            CONSTRAINT entity_aliases_not_self
                CHECK (alias_canonical_id <> primary_canonical_id)
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_entity_aliases_primary "
        "ON entity_aliases (customer_id, label, primary_canonical_id)"
    )
    op.execute(
        "CREATE INDEX idx_entity_aliases_merge "
        "ON entity_aliases (merge_id)"
    )

    # ----- entity_cluster_metadata -----
    op.execute(
        """
        CREATE TABLE entity_cluster_metadata (
            customer_id                  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
            label                        TEXT NOT NULL,
            primary_canonical_id         TEXT NOT NULL,
            display_name                 TEXT NOT NULL,
            display_name_last_edited_by  UUID NULL,
            display_name_last_edited_at  TIMESTAMPTZ NULL,
            PRIMARY KEY (customer_id, label, primary_canonical_id)
        )
        """
    )

    # ----- RLS: ENABLE + FORCE + USING/WITH CHECK on all 5 new tables -----
    for table in (
        "entity_merge_audit",
        "entity_merge_node_snapshot",
        "entity_merge_edge_snapshot",
        "entity_aliases",
        "entity_cluster_metadata",
    ):
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"""
            CREATE POLICY tenant_isolation ON {table}
                USING ({_POLICY_EXPR})
                WITH CHECK ({_POLICY_EXPR})
            """
        )

    # ----- graph_edges: add columns, swap UNIQUE constraint -----
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD COLUMN aliased_from_canonical_id TEXT NULL,
            ADD COLUMN aliased_to_canonical_id   TEXT NULL
        """
    )
    op.execute(
        # Replace <EXISTING_UNIQUE_NAME> with the actual auto-generated name
        # captured in Task A1.
        "ALTER TABLE graph_edges "
        "DROP CONSTRAINT <EXISTING_UNIQUE_NAME>"
    )
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD CONSTRAINT graph_edges_unique_lane UNIQUE (
                customer_id, edge_type, from_node_id, to_node_id,
                COALESCE(aliased_from_canonical_id, ''),
                COALESCE(aliased_to_canonical_id, '')
            )
        """
    )


def downgrade() -> None:
    # graph_edges first — restore original UNIQUE.
    op.execute("ALTER TABLE graph_edges DROP CONSTRAINT graph_edges_unique_lane")
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD CONSTRAINT graph_edges_customer_id_edge_type_from_node_id_to_node_id_key
                UNIQUE (customer_id, edge_type, from_node_id, to_node_id)
        """
    )
    op.execute("ALTER TABLE graph_edges DROP COLUMN aliased_to_canonical_id")
    op.execute("ALTER TABLE graph_edges DROP COLUMN aliased_from_canonical_id")

    # Drop tables in reverse FK order (aliases / snapshots → audit).
    for table in (
        "entity_cluster_metadata",
        "entity_aliases",
        "entity_merge_edge_snapshot",
        "entity_merge_node_snapshot",
        "entity_merge_audit",
    ):
        op.execute(f"DROP POLICY IF EXISTS tenant_isolation ON {table}")
        op.execute(f"DROP TABLE IF EXISTS {table}")
```

**Important:** Postgres doesn't accept a `COALESCE(...)` expression directly as a column reference in `UNIQUE (...)`. You must use an **expression index** + `EXCLUDE` or wrap via `UNIQUE` on a generated column, OR use the PG 15+ `UNIQUE NULLS NOT DISTINCT` syntax.

The portable form across PG 14/15/16 is to use a `CREATE UNIQUE INDEX` with the COALESCE expressions (UNIQUE INDEX, unlike UNIQUE CONSTRAINT, accepts expressions):

```sql
-- Replace the constraint syntax with an explicit unique INDEX.
CREATE UNIQUE INDEX graph_edges_unique_lane ON graph_edges (
    customer_id, edge_type, from_node_id, to_node_id,
    COALESCE(aliased_from_canonical_id, ''),
    COALESCE(aliased_to_canonical_id, '')
);
```

A unique INDEX is equivalent to a UNIQUE constraint for `ON CONFLICT` purposes (you reference it by index name in `ON CONFLICT ON CONSTRAINT <index_name>` or by the index column list). Update the migration's UNIQUE section accordingly:

```python
    # Replace the previous "ADD CONSTRAINT graph_edges_unique_lane UNIQUE (...)" with:
    op.execute(
        """
        CREATE UNIQUE INDEX graph_edges_unique_lane ON graph_edges (
            customer_id, edge_type, from_node_id, to_node_id,
            COALESCE(aliased_from_canonical_id, ''),
            COALESCE(aliased_to_canonical_id, '')
        )
        """
    )
```

And in `downgrade`:

```python
    op.execute("DROP INDEX IF EXISTS graph_edges_unique_lane")
    # Restore the original UNIQUE constraint name.
    op.execute(
        """
        ALTER TABLE graph_edges
            ADD CONSTRAINT graph_edges_customer_id_edge_type_from_node_id_to_node_id_key
                UNIQUE (customer_id, edge_type, from_node_id, to_node_id)
        """
    )
```

(If you've verified PG 15+ on the target cluster and prefer the constraint form, you can use `UNIQUE NULLS NOT DISTINCT (...)` instead. Either way, the `ON CONFLICT` clause in `graph_writer.upsert_edges` will reference the unique-index/constraint name `graph_edges_unique_lane` — see Task A8.)

- [ ] **Step 2: Apply the migration**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1
scripts/neon-migrate.sh local
```

Expected: `Running upgrade 0070_gnp_rls -> 0071_entity_clusters, Entity clusters: tables + graph_edges changes for manual identity merging (B-promote).`

- [ ] **Step 3: Verify in psql**

```bash
psql postgresql://prbe:prbe@localhost:5432/prbe_knowledge -c "\d entity_aliases" \
                                                          -c "\d entity_merge_audit" \
                                                          -c "\d graph_edges"
```

Expected: All five new tables present. `graph_edges` shows the new columns + the `graph_edges_unique_lane` index.

- [ ] **Step 4: Run the migration tests, confirm they pass**

```bash
pytest tests/test_entity_clusters_migration.py -v
```

Expected: All 9 tests PASS.

### Task A4: Sync db/schema.sql to match the migration

`db/schema.sql` is the canonical SQL-only reference. Anything in a migration must also appear here.

**Files:**
- Modify: `db/schema.sql` (insert new tables after line ~636; modify `graph_edges` section at line ~565)

- [ ] **Step 1: Update the `graph_edges` table definition**

In `db/schema.sql`, find the `CREATE TABLE graph_edges` block (around line 565). After the existing column list and before the closing `)`, the existing `UNIQUE (customer_id, edge_type, from_node_id, to_node_id)` line must be **removed**. Then **add** two new columns to the column list (just after `extracted_at TIMESTAMPTZ`) so the block looks like:

```sql
CREATE TABLE graph_edges (
    edge_id       BIGSERIAL PRIMARY KEY,
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    edge_type     TEXT NOT NULL,
    from_node_id  BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    to_node_id    BIGINT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
    properties    JSONB NOT NULL DEFAULT '{}',
    valid_from    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to      TIMESTAMPTZ,
    source_system TEXT,
    confidence    TEXT NOT NULL DEFAULT 'EXTRACTED'
        CONSTRAINT graph_edges_confidence_check
        CHECK (confidence IN ('EXTRACTED', 'INFERRED', 'AMBIGUOUS')),
    extractor_id  TEXT,
    extracted_at  TIMESTAMPTZ,
    -- Provenance for alias-resolved or merge-rewritten edges. NULL when
    -- the edge has never been touched by alias resolution. Populated by
    -- graph_writer at ingest (when the inbound canonical_id was an alias)
    -- and by the merge transaction (when an alias node's edge was rewritten
    -- to point at the primary).
    aliased_from_canonical_id TEXT,
    aliased_to_canonical_id   TEXT
);
```

Then, right after the existing `CREATE INDEX idx_graph_edges_customer_extractor` (around line 599), add the new unique INDEX:

```sql
-- Composite UNIQUE keyed by (edge_type, from, to, alias_from, alias_to).
-- Different alias lanes coexist as distinct rows; common-case "both
-- aliased_from cols NULL" inserts still dedup (COALESCE-to-empty-string
-- collides). graph_writer.upsert_edges' ON CONFLICT references this
-- index by name.
CREATE UNIQUE INDEX graph_edges_unique_lane ON graph_edges (
    customer_id, edge_type, from_node_id, to_node_id,
    COALESCE(aliased_from_canonical_id, ''),
    COALESCE(aliased_to_canonical_id, '')
);
```

- [ ] **Step 2: Append the new tables + RLS**

Find the section ending with `CREATE POLICY tenant_isolation ON graph_node_provenance` (around line 636). Immediately after that block, insert:

```sql
-- ---------------------------------------------------------------------------
-- Entity clusters: manual identity merging via dashboard (migration 0071).
-- Physical merge (B-promote) — alias edges rewritten, alias nodes
-- hard-deleted. See docs/superpowers/specs/2026-05-13-entity-clusters-design.md.
-- ---------------------------------------------------------------------------
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

CREATE TABLE entity_merge_node_snapshot (
    merge_id      UUID NOT NULL REFERENCES entity_merge_audit(merge_id),
    customer_id   TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label         TEXT NOT NULL,
    canonical_id  TEXT NOT NULL,
    properties    JSONB NOT NULL,
    degree        INT  NOT NULL,
    community_id  INT  NULL,
    created_at    TIMESTAMPTZ NOT NULL,
    provenance    JSONB NOT NULL,
    PRIMARY KEY (merge_id, label, canonical_id)
);

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

CREATE TABLE entity_cluster_metadata (
    customer_id                  TEXT NOT NULL REFERENCES customers(customer_id) ON DELETE CASCADE,
    label                        TEXT NOT NULL,
    primary_canonical_id         TEXT NOT NULL,
    display_name                 TEXT NOT NULL,
    display_name_last_edited_by  UUID NULL,
    display_name_last_edited_at  TIMESTAMPTZ NULL,
    PRIMARY KEY (customer_id, label, primary_canonical_id)
);

ALTER TABLE entity_merge_audit         ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_audit         FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_merge_node_snapshot ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_node_snapshot FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_merge_edge_snapshot ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_merge_edge_snapshot FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_aliases             ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_aliases             FORCE  ROW LEVEL SECURITY;
ALTER TABLE entity_cluster_metadata    ENABLE ROW LEVEL SECURITY;
ALTER TABLE entity_cluster_metadata    FORCE  ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON entity_merge_audit
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_merge_node_snapshot
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_merge_edge_snapshot
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_aliases
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
CREATE POLICY tenant_isolation ON entity_cluster_metadata
    USING       (customer_id = current_setting('app.current_customer_id', true))
    WITH CHECK  (customer_id = current_setting('app.current_customer_id', true));
```

- [ ] **Step 3: Sanity-check schema.sql applies to a fresh DB**

```bash
psql postgresql://prbe:prbe@localhost:5432/postgres -c "DROP DATABASE IF EXISTS prbe_knowledge_schemacheck; CREATE DATABASE prbe_knowledge_schemacheck"
psql postgresql://prbe:prbe@localhost:5432/prbe_knowledge_schemacheck -f db/schema.sql
psql postgresql://prbe:prbe@localhost:5432/prbe_knowledge_schemacheck -c "\d entity_aliases" -c "\d graph_edges"
psql postgresql://prbe:prbe@localhost:5432/postgres -c "DROP DATABASE prbe_knowledge_schemacheck"
```

Expected: no SQL errors. Both tables resolve, and `graph_edges` shows the new columns + `graph_edges_unique_lane` index.

### Task A5: Add alias columns to GraphEdgeSpec

**Files:**
- Modify: `shared/models.py` (the `GraphEdgeSpec` class)

- [ ] **Step 1: Find `GraphEdgeSpec`**

```bash
grep -n "class GraphEdgeSpec" shared/models.py
```

Expected: one line number.

- [ ] **Step 2: Add the two new fields**

Within `GraphEdgeSpec`, add two `Optional[str]` fields (Pydantic) defaulting to None. After the existing field declarations (after `extracted_at` and similar), insert:

```python
    aliased_from_canonical_id: str | None = None
    aliased_to_canonical_id:   str | None = None
```

These fields are populated by `graph_writer` alias resolution (Task A7) and otherwise stay `None`.

### Task A6: Write the failing graph_writer alias-resolution test

**Files:**
- Create: `tests/test_graph_writer_alias_resolution.py`

- [ ] **Step 1: Write the test file**

```python
"""Graph_writer alias-resolution invariants.

Pins the behaviour we wire into upsert_nodes + upsert_edges so post-merge
webhook ingest correctly routes aliased canonical_ids to the primary.

Invariants:
  * Empty entity_aliases → no rewrite. Pre-merge ingest is a no-op for
    the new code path.
  * upsert_nodes with an aliased canonical_id → INSERT lands on the
    primary's (label, canonical_id) row.
  * upsert_edges with an aliased endpoint → INSERT row's
    from_node_id (or to_node_id) is the primary's; aliased_from_canonical_id
    (or aliased_to) is set to the original alias.
  * upsert_edges where both endpoints resolve to the same canonical
    (post-merge self-loop) → row is dropped, dropped['self_edge_post_alias']
    counter is incremented.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import pytest

from shared.constants import EdgeType, NodeLabel
from shared.db import raw_conn, with_tenant
from shared.models import GraphEdgeSpec, GraphNodeSpec
from services.ingestion.graph_writer import upsert_edges, upsert_nodes


async def _seed_customer(conn, customer_id: str) -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'mig', 'mig-hash') ON CONFLICT DO NOTHING",
        customer_id,
    )


async def _insert_audit(conn, *, customer_id, primary, aliases) -> uuid.UUID:
    merge_id = uuid.uuid4()
    await conn.execute(
        """
        INSERT INTO entity_merge_audit
          (merge_id, customer_id, label, primary_canonical_id,
           merged_alias_canonical_ids, performed_by_user_id)
        VALUES ($1, $2, 'Person', $3, $4, $5)
        """,
        merge_id, customer_id, primary, aliases, uuid.uuid4(),
    )
    return merge_id


async def _insert_alias_row(conn, *, customer_id, alias, primary, merge_id) -> None:
    await conn.execute(
        """
        INSERT INTO entity_aliases
          (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
        VALUES ($1, 'Person', $2, $3, $4)
        """,
        customer_id, alias, primary, merge_id,
    )


# ---------------------------------------------------------------------------
# upsert_nodes alias resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_nodes_no_aliases_no_rewrite(live_db) -> None:
    cust = "alres-noop"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="richardwei6")],
            source_system="github",
        )
        assert ("Person", "richardwei6") in node_ids
        row = await conn.fetchrow(
            "SELECT canonical_id FROM graph_nodes WHERE node_id = $1",
            node_ids[("Person", "richardwei6")],
        )
        assert row["canonical_id"] == "richardwei6"


@pytest.mark.asyncio
async def test_upsert_nodes_rewrites_aliased_canonical_id(live_db) -> None:
    cust = "alres-node-rewrite"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        # Pre-seed the primary so the alias row's FK pre-check is happy.
        await conn.execute(
            "INSERT INTO graph_nodes(customer_id, label, canonical_id) "
            "VALUES ($1, 'Person', 'richardwei6')",
            cust,
        )
        merge_id = await _insert_audit(
            conn, customer_id=cust, primary="richardwei6",
            aliases=["mahit@prbe.ai"],
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="mahit@prbe.ai",
            primary="richardwei6", merge_id=merge_id,
        )
        # Incoming webhook: ingest "mahit@prbe.ai" — should resolve.
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="mahit@prbe.ai")],
            source_system="github",
        )
        # The returned map keys on the ORIGINAL (label, canonical_id) for callers,
        # but the underlying graph_nodes row is the primary's.
        # Implementations may choose either convention; assert the underlying row.
        assert len(node_ids) == 1
        node_id = list(node_ids.values())[0]
        row = await conn.fetchrow(
            "SELECT canonical_id FROM graph_nodes WHERE node_id = $1",
            node_id,
        )
        assert row["canonical_id"] == "richardwei6"
        # No row was created with canonical_id='mahit@prbe.ai'.
        leaked = await conn.fetchrow(
            "SELECT 1 FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Person' AND canonical_id = 'mahit@prbe.ai'",
            cust,
        )
        assert leaked is None


# ---------------------------------------------------------------------------
# upsert_edges alias resolution
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upsert_edges_rewrites_aliased_endpoint(live_db) -> None:
    cust = "alres-edge-rewrite"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await conn.execute(
            "INSERT INTO graph_nodes(customer_id, label, canonical_id) "
            "VALUES ($1, 'Person', 'richardwei6'), ($1, 'Document', 'doc-1')",
            cust,
        )
        merge_id = await _insert_audit(
            conn, customer_id=cust, primary="richardwei6",
            aliases=["mahit@prbe.ai"],
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="mahit@prbe.ai",
            primary="richardwei6", merge_id=merge_id,
        )
        # Upsert nodes first (graph_writer's normal call sequence).
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[
                GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="mahit@prbe.ai"),
                GraphNodeSpec(label=NodeLabel.DOCUMENT, canonical_id="doc-1"),
            ],
            source_system="github",
        )
        # Inbound edge from the (aliased) Slack user → doc-1.
        await upsert_edges(
            conn,
            customer_id=cust,
            edges=[
                GraphEdgeSpec(
                    edge_type=EdgeType.AUTHORED,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id="mahit@prbe.ai",
                    to_label=NodeLabel.DOCUMENT,
                    to_canonical_id="doc-1",
                ),
            ],
            node_ids=node_ids,
            source_system="github",
        )
        # Assert the edge landed on the primary's node and stamped aliased_from.
        rows = await conn.fetch(
            """
            SELECT ge.aliased_from_canonical_id, ge.aliased_to_canonical_id,
                   gn_from.canonical_id AS from_canonical
            FROM graph_edges ge
            JOIN graph_nodes gn_from ON gn_from.node_id = ge.from_node_id
            WHERE ge.customer_id = $1 AND ge.edge_type = 'AUTHORED'
            """,
            cust,
        )
        assert len(rows) == 1
        assert rows[0]["from_canonical"] == "richardwei6"
        assert rows[0]["aliased_from_canonical_id"] == "mahit@prbe.ai"
        assert rows[0]["aliased_to_canonical_id"] is None


@pytest.mark.asyncio
async def test_upsert_edges_drops_self_loop_after_resolution(live_db) -> None:
    """Both endpoints resolve to the same canonical → row dropped."""
    cust = "alres-selfloop"
    async with raw_conn() as conn:
        await _seed_customer(conn, cust)
    async with with_tenant(cust) as conn:
        await conn.execute(
            "INSERT INTO graph_nodes(customer_id, label, canonical_id) "
            "VALUES ($1, 'Person', 'richardwei6')",
            cust,
        )
        merge_id = await _insert_audit(
            conn, customer_id=cust, primary="richardwei6",
            aliases=["mahit@prbe.ai", "U07ABC123"],
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="mahit@prbe.ai",
            primary="richardwei6", merge_id=merge_id,
        )
        await _insert_alias_row(
            conn, customer_id=cust, alias="U07ABC123",
            primary="richardwei6", merge_id=merge_id,
        )
        # Upsert nodes for both aliases — both resolve to richardwei6.
        node_ids = await upsert_nodes(
            conn,
            customer_id=cust,
            nodes=[
                GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="mahit@prbe.ai"),
                GraphNodeSpec(label=NodeLabel.PERSON, canonical_id="U07ABC123"),
            ],
            source_system="github",
        )
        # Try to write Person↔Person edge between two aliases of the same
        # cluster — post-resolution it's a self-loop.
        count = await upsert_edges(
            conn,
            customer_id=cust,
            edges=[
                GraphEdgeSpec(
                    edge_type=EdgeType.MENTIONS,
                    from_label=NodeLabel.PERSON,
                    from_canonical_id="mahit@prbe.ai",
                    to_label=NodeLabel.PERSON,
                    to_canonical_id="U07ABC123",
                ),
            ],
            node_ids=node_ids,
            source_system="github",
        )
        # Either:
        #   - upsert_edges returns 0 and no row was written, OR
        #   - the dropped counter records the drop (implementation choice).
        # The DB is the authoritative check.
        rows = await conn.fetch(
            "SELECT 1 FROM graph_edges WHERE customer_id = $1 AND edge_type = 'MENTIONS'",
            cust,
        )
        assert rows == [], "self-loop edge should not be persisted"
```

- [ ] **Step 2: Run the test, confirm it fails**

```bash
pytest tests/test_graph_writer_alias_resolution.py -v
```

Expected: All tests fail (graph_writer doesn't consult `entity_aliases` yet).

### Task A7: Implement `_fetch_aliases` helper

**Files:**
- Modify: `services/ingestion/graph_writer.py` (add helper near top)

- [ ] **Step 1: Add the helper**

At the top of `services/ingestion/graph_writer.py` (after the existing imports), add:

```python
async def _fetch_aliases(
    conn: asyncpg.Connection,
    customer_id: str,
    keys: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Bulk-resolve `(label, canonical_id) → primary_canonical_id` for aliased keys.

    Returned dict only contains entries for keys that ARE aliases. Non-aliased
    keys are absent; callers should treat absence as "no rewrite needed."

    One bulk query per call regardless of input size — entity_aliases is
    typically O(100s) of rows per tenant; the (customer_id, label,
    alias_canonical_id) PK answers this with an index-only scan.
    """
    if not keys:
        return {}
    labels = [k[0] for k in keys]
    aliases = [k[1] for k in keys]
    rows = await conn.fetch(
        """
        SELECT label, alias_canonical_id, primary_canonical_id
        FROM entity_aliases
        WHERE customer_id = $1
          AND (label, alias_canonical_id) IN (
                SELECT * FROM UNNEST($2::text[], $3::text[])
              )
        """,
        customer_id, labels, aliases,
    )
    return {(r["label"], r["alias_canonical_id"]): r["primary_canonical_id"] for r in rows}
```

### Task A8: Wire alias resolution into `upsert_nodes`

**Files:**
- Modify: `services/ingestion/graph_writer.py:upsert_nodes` (top of function body)

- [ ] **Step 1: Add the rewrite block**

In `upsert_nodes`, immediately after the early-return `if not nodes: return {}`, add:

```python
    # Alias resolution: if any inbound (label, canonical_id) is an alias of
    # a merged cluster, rewrite to the primary BEFORE the existing dedup
    # logic. Empty entity_aliases → no-op.
    alias_map = await _fetch_aliases(
        conn,
        customer_id,
        keys=[(n.label.value, n.canonical_id) for n in nodes],
    )
    if alias_map:
        for n in nodes:
            primary = alias_map.get((n.label.value, n.canonical_id))
            if primary is not None:
                n.canonical_id = primary
```

(GraphNodeSpec is a Pydantic model; mutating `canonical_id` works because Pydantic v2 dataclasses allow assignment unless `frozen=True` is set. Verify with `python -c "from shared.models import GraphNodeSpec; n=GraphNodeSpec(label='Person', canonical_id='x'); n.canonical_id='y'"` — if it raises, switch to constructing new instances via `.model_copy(update={'canonical_id': primary})`.)

### Task A9: Wire alias resolution into `upsert_edges` + update `ON CONFLICT` clause

**Files:**
- Modify: `services/ingestion/graph_writer.py:upsert_edges`

- [ ] **Step 1: Resolve aliases at the top of `upsert_edges`**

Immediately after `if not edges: return 0`, add:

```python
    # Alias resolution for both endpoints. Populate aliased_from/to on
    # the spec object so the INSERT row carries the original alias.
    endpoint_keys = []
    for e in edges:
        endpoint_keys.append((e.from_label.value, e.from_canonical_id))
        endpoint_keys.append((e.to_label.value,   e.to_canonical_id))
    alias_map = await _fetch_aliases(conn, customer_id, endpoint_keys)

    dropped_self_loops = 0
    if alias_map:
        kept: list[GraphEdgeSpec] = []
        for e in edges:
            from_primary = alias_map.get((e.from_label.value, e.from_canonical_id))
            to_primary   = alias_map.get((e.to_label.value,   e.to_canonical_id))
            if from_primary is not None:
                e.aliased_from_canonical_id = e.from_canonical_id
                e.from_canonical_id = from_primary
            if to_primary is not None:
                e.aliased_to_canonical_id = e.to_canonical_id
                e.to_canonical_id = to_primary
            # Self-loop after resolution → drop (matches Lane B Rule 5 +
            # the system-wide "no self-edges" convention).
            if (e.from_label == e.to_label
                    and e.from_canonical_id == e.to_canonical_id):
                dropped_self_loops += 1
                continue
            kept.append(e)
        edges = kept
    # (dropped_self_loops can be logged or exported via a metric; for
    # Phase 1 we just suppress the row.)
```

- [ ] **Step 2: Update the dedup key to include alias provenance**

The existing dedup keys on `(edge_type, from_node_id, to_node_id)`. After alias resolution, two webhook events from *different* alias origins (say `mahit@prbe.ai` and `U07ABC123`, both merged into `richardwei6`) would have the same `(edge_type, primary_node_id, doc_node_id)` and would collapse into ONE deduped entry — losing the alias-lane distinction we need to preserve.

Find the existing dedup block (the `deduped: dict[tuple[str, int, int], dict] = {}` line + the for-loop that follows, ~line 130-160) and:

1. Widen the dict's key type to a 5-tuple `(edge_type, from_node_id, to_node_id, aliased_from, aliased_to)` where the alias columns are coalesced to `""` for the NULL case (matching what the UNIQUE index does).
2. Carry `aliased_from_canonical_id` / `aliased_to_canonical_id` in each deduped entry's value dict.

```python
    # Resolve endpoints + dedupe. Composite UNIQUE means different alias
    # lanes (same edge_type/from/to but different aliased_from_canonical_id)
    # coexist; the dedup key must reflect that so two webhook events from
    # different alias origins don't collapse into one entry.
    deduped: dict[tuple[str, int, int, str, str], dict] = {}
    for edge in edges:
        from_id = node_ids.get((edge.from_label.value, edge.from_canonical_id))
        to_id   = node_ids.get((edge.to_label.value,   edge.to_canonical_id))
        if from_id is None or to_id is None:
            continue
        aliased_from = edge.aliased_from_canonical_id or ""
        aliased_to   = edge.aliased_to_canonical_id   or ""
        key = (edge.edge_type.value, from_id, to_id, aliased_from, aliased_to)
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = {
                "properties":   dict(edge.properties),
                "valid_from":   edge.valid_from,
                "valid_to":     edge.valid_to,
                "confidence":   edge.confidence,
                "aliased_from": edge.aliased_from_canonical_id,
                "aliased_to":   edge.aliased_to_canonical_id,
            }
        else:
            existing["properties"] = {**existing["properties"], **edge.properties}
            if edge.valid_from is not None and (
                existing["valid_from"] is None
                or edge.valid_from < existing["valid_from"]
            ):
                existing["valid_from"] = edge.valid_from
            existing["valid_to"]   = edge.valid_to
            existing["confidence"] = _stronger_confidence(
                existing["confidence"], edge.confidence
            )
```

- [ ] **Step 3: Thread `aliased_from/to_canonical_id` through the INSERT**

Find the parallel-array build (the `sorted_keys = sorted(deduped.keys())` block around line 165-175). Adjust the index unpacking to read 5-tuples and add two new parallel lists. The blocks become:

```python
    sorted_keys = sorted(deduped.keys())
    edge_types     = [k[0]                                for k in sorted_keys]
    from_ids       = [k[1]                                for k in sorted_keys]
    to_ids         = [k[2]                                for k in sorted_keys]
    properties_json = [
        orjson.dumps(deduped[k]["properties"]).decode("utf-8") for k in sorted_keys
    ]
    valid_from_list = [deduped[k]["valid_from"]   for k in sorted_keys]
    valid_to_list   = [deduped[k]["valid_to"]     for k in sorted_keys]
    confidences     = [deduped[k]["confidence"]   for k in sorted_keys]
    aliased_from_list = [deduped[k]["aliased_from"] for k in sorted_keys]
    aliased_to_list   = [deduped[k]["aliased_to"]   for k in sorted_keys]
```

Then the existing INSERT statement (~line 191) needs:

1. **Two new columns** in the INSERT column list, the SELECT, and the `unnest(...)`:

```python
    inserted_rows = await conn.fetch(
        """
        INSERT INTO graph_edges (
            customer_id, edge_type, from_node_id, to_node_id,
            properties, valid_from, valid_to, source_system, confidence,
            extractor_id, extracted_at,
            aliased_from_canonical_id, aliased_to_canonical_id
        )
        SELECT $1, edge_type, from_node_id, to_node_id,
               properties::jsonb, COALESCE(valid_from, NOW()), valid_to, $2, confidence,
               $10, $11,
               aliased_from, aliased_to
        FROM unnest(
            $3::text[], $4::bigint[], $5::bigint[],
            $6::text[], $7::timestamptz[], $8::timestamptz[], $9::text[],
            $12::text[], $13::text[]
        ) AS t(edge_type, from_node_id, to_node_id,
               properties, valid_from, valid_to, confidence,
               aliased_from, aliased_to)
        ON CONFLICT ON CONSTRAINT graph_edges_unique_lane
        DO UPDATE
           SET properties = graph_edges.properties || EXCLUDED.properties,
               valid_from = LEAST(graph_edges.valid_from, EXCLUDED.valid_from),
               valid_to   = EXCLUDED.valid_to,
               confidence = (CASE
                   WHEN graph_edges.confidence = 'EXTRACTED' THEN graph_edges.confidence
                   WHEN EXCLUDED.confidence    = 'EXTRACTED' THEN EXCLUDED.confidence
                   WHEN graph_edges.confidence = 'INFERRED'  THEN graph_edges.confidence
                   WHEN EXCLUDED.confidence    = 'INFERRED'  THEN EXCLUDED.confidence
                   ELSE EXCLUDED.confidence
               END)
        RETURNING from_node_id, to_node_id, (xmax = 0) AS inserted
        """,
        customer_id,
        source_system,
        edge_types, from_ids, to_ids,
        properties_json, valid_from_list, valid_to_list, confidences,
        extractor_id, extracted_at,
        aliased_from_list, aliased_to_list,
    )
```

(The exact `$N` numbering depends on the existing call's parameter order; verify against the surrounding code. The point is: pass the two new arrays as the final parameters and reference them in the SELECT.)

2. **Change `ON CONFLICT`** to reference the new unique INDEX by name (`graph_edges_unique_lane`). Postgres treats a unique INDEX as a usable conflict target via `ON CONSTRAINT <index_name>`. Functionally equivalent to the column-list `ON CONFLICT (...)` form but doesn't require restating all 6 columns + 2 COALESCE expressions.

- [ ] **Step 3: Run the test, confirm it passes**

```bash
pytest tests/test_graph_writer_alias_resolution.py -v
```

Expected: All 4 tests PASS.

### Task A10: Run full prbe-knowledge test suite to confirm no regression

- [ ] **Step 1: Run the suite**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1
pytest -q
```

Expected: All tests pass. The new tables are empty so existing tests don't exercise alias resolution; graph_writer's behavior is unchanged in the no-alias case.

If any existing graph_writer test fails (likely candidates: `tests/test_backfill.py`, `tests/test_chunker.py`, anything that calls `upsert_edges`), the failure is most likely:
- An ON CONFLICT reference to the old constraint name that you missed elsewhere in the codebase.
- A test that hardcoded `aliased_from_canonical_id IS NULL` and got `None` from Python instead.

Fix and rerun until green.

### Task A11: Commit PR 1a

- [ ] **Step 1: Stage + commit**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1
git add db/migrations/versions/20260514_0071_entity_clusters.py \
        db/schema.sql \
        shared/models.py \
        services/ingestion/graph_writer.py \
        tests/test_entity_clusters_migration.py \
        tests/test_graph_writer_alias_resolution.py \
        docs/superpowers/specs/2026-05-13-entity-clusters-design.md \
        docs/superpowers/specs/2026-05-13-entity-clusters-phase1-plan.md
git commit -m "$(cat <<'EOF'
feat(db,graph_writer): entity-clusters schema + ingest alias resolution (Phase 1a)

Adds the schema + ingest-side scaffolding for manual entity merging
(B-promote design). Five new RLS-isolated tables:

  * entity_merge_audit            — one row per merge action.
  * entity_merge_node_snapshot    — pre-state of each hard-deleted alias
                                    node (incl. inlined provenance JSONB).
  * entity_merge_edge_snapshot    — pre-state of edges deleted as
                                    self-loops during merge (the only
                                    deletion type under composite UNIQUE).
  * entity_aliases                — forward routing (alias → primary).
  * entity_cluster_metadata       — sparse display-name override; empty
                                    until Phase 3.

graph_edges schema changes:

  * New nullable TEXT columns aliased_from_canonical_id /
    aliased_to_canonical_id capture pre-rewrite endpoint provenance.
  * The existing single-column UNIQUE is replaced by a composite UNIQUE
    INDEX that COALESCEs the alias columns to ''. Different alias lanes
    coexist as distinct rows; common-case "both alias cols NULL" inserts
    still dedup via graph_writer.upsert_edges' ON CONFLICT path.

graph_writer.upsert_nodes / upsert_edges now consult entity_aliases:

  * Inbound aliased (label, canonical_id) is rewritten to the primary
    before the existing dedup/INSERT logic.
  * aliased_from_canonical_id / aliased_to_canonical_id are stamped on
    the INSERT row whenever resolution kicked in.
  * Edges whose both endpoints resolve to the same canonical are dropped
    (Lane B Rule 5 — never emit self-edges).
  * ON CONFLICT clause now references the new constraint by name.

Pre-merge behavior is unchanged: entity_aliases is empty, _fetch_aliases
returns an empty map, no rewrites happen, no aliased_from/to columns are
populated. Composite UNIQUE acts identically to the old single-column
UNIQUE for the all-NULLs case.

Phase 1b will land the merge / unmerge endpoints that USE these tables.

Design + plan docs included.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 2: Verify the commit**

```bash
git log --oneline -1
git status
```

Expected: One new commit; clean working tree.

---

## PR 1b — prbe-knowledge: merge + unmerge endpoints

**Worktree:** `cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1` (continue from PR 1a's branch)

### Task B1: Scaffold the entity_clusters_routes module

**Files:**
- Create: `services/ingestion/entity_clusters_routes.py`

- [ ] **Step 1: Write the scaffold**

```python
"""Manual entity-cluster merging — internal API.

Mounted into the ingestion service alongside admin_routes.py. Gated by
X-Internal-Knowledge-Key (caller is prbe-backend BFF after JWT validation).

Endpoints:

  POST   /api/entity-clusters/merge
         Run the full merge transaction: validate → lock → snapshot →
         rewrite edges → drop self-loops → merge provenance → delete
         alias nodes → recompute degree → INSERT routing + audit.

  DELETE /api/entity-clusters/{label}/{primary}/aliases/{alias}
         Run the unmerge transaction: re-INSERT alias node from snapshot,
         UPDATE edges via aliased_from/to back to alias, re-INSERT
         snapshotted self-loops, recompute degree, drop routing row,
         flip audit status if last alias under merge_id.

Design + plan:
  docs/superpowers/specs/2026-05-13-entity-clusters-design.md
  docs/superpowers/specs/2026-05-13-entity-clusters-phase1-plan.md
"""

from __future__ import annotations

import hmac
import logging
import uuid
from typing import Any

import asyncpg
import orjson
from fastapi import APIRouter, Depends, Header, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field, field_validator

from shared.config import get_settings
from shared.db import with_tenant

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/entity-clusters", tags=["internal-api"])


# ---------------------------------------------------------------------------
# Auth: X-Internal-Knowledge-Key (same gate as admin_routes.py)
# ---------------------------------------------------------------------------


def _require_internal_key(
    x_internal_knowledge_key: str | None = Header(default=None),
) -> None:
    expected = get_settings().internal_knowledge_api_key
    if not x_internal_knowledge_key or not hmac.compare_digest(
        x_internal_knowledge_key, expected
    ):
        raise HTTPException(status_code=401, detail="invalid internal key")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class MergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id:          str            = Field(..., min_length=1, max_length=128)
    performed_by_user_id: uuid.UUID
    label:                str            = Field(..., min_length=1, max_length=64)
    primary_canonical_id: str            = Field(..., min_length=1, max_length=512)
    alias_canonical_ids:  list[str]      = Field(..., min_length=1, max_length=64)
    reason:               str | None     = Field(default=None, max_length=2000)

    @field_validator("alias_canonical_ids")
    @classmethod
    def _unique_non_blank(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("alias_canonical_ids must be unique")
        for a in v:
            if not a or not a.strip():
                raise ValueError("alias_canonical_ids may not contain blanks")
        return v


class MergeResponse(BaseModel):
    merge_id:                   uuid.UUID
    label:                      str
    primary_canonical_id:       str
    merged_alias_canonical_ids: list[str]
```

### Task B2: Mount the router in `services/ingestion/main.py`

**Files:**
- Modify: `services/ingestion/main.py`

- [ ] **Step 1: Import + include**

Find the existing import block (around `from services.ingestion.admin_routes import router as admin_router`). After it, add:

```python
from services.ingestion.entity_clusters_routes import (
    router as entity_clusters_router,
)
```

Find the `app.include_router(...)` block. After the existing `app.include_router(admin_router)`, add:

```python
app.include_router(entity_clusters_router)
```

- [ ] **Step 2: Verify the app boots**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1
python -c "from services.ingestion.main import app; print([r.path for r in app.routes if 'entity-clusters' in str(r.path)])"
```

Expected: `[]` (no endpoints decorated yet) and no import errors.

### Task B3: Write the failing merge happy-path test

**Files:**
- Create: `tests/test_entity_clusters_routes.py`

- [ ] **Step 1: Write the test file with the happy-path test**

```python
"""Integration tests for /api/entity-clusters/merge and /unmerge endpoints.

These exercise the full DB transaction against live Postgres (via the
live_db fixture). The endpoints are gated by X-Internal-Knowledge-Key;
tests pass the configured key in the header.

Covers:
  * Merge happy-path: 3 INSERTs in the routing/audit tables, edges
    rewritten to primary with aliased_from set, alias nodes deleted,
    provenance merged into canonical, degree recomputed.
  * Merge validation errors: 404 / 409 alias-already / 409 primary-is-alias
    / 422 duplicate-aliases.
  * Unmerge happy-path: alias node restored, edges UPDATEd back via
    aliased_from columns, audit row flipped to 'reversed' when last
    alias removed.
  * Unmerge 404 when alias not in any cluster.
"""

from __future__ import annotations

import uuid
from typing import Any

import asyncpg
import pytest
from fastapi.testclient import TestClient

from shared.config import get_settings
from shared.db import raw_conn, with_tenant
from services.ingestion.main import app


CUSTOMER_ID = "ec-routes-cust"
USER_ID = "11111111-1111-1111-1111-111111111111"


def _headers() -> dict[str, str]:
    return {"X-Internal-Knowledge-Key": get_settings().internal_knowledge_api_key}


async def _seed_customer(conn: asyncpg.Connection) -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'mig', 'mig-hash') ON CONFLICT DO NOTHING",
        CUSTOMER_ID,
    )


async def _seed_person(conn: asyncpg.Connection, canonical_id: str, *, props: dict[str, Any] | None = None) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO graph_nodes(customer_id, label, canonical_id, properties)
        VALUES ($1, 'Person', $2, $3::jsonb)
        ON CONFLICT (customer_id, label, canonical_id) DO UPDATE
            SET properties = EXCLUDED.properties, updated_at = NOW()
        RETURNING node_id
        """,
        CUSTOMER_ID, canonical_id,
        '{"display_name": "' + canonical_id + '"}' if props is None else orjson_dumps(props),
    )
    return row["node_id"]


def orjson_dumps(o: Any) -> str:  # local helper to avoid import wrangling
    import orjson
    return orjson.dumps(o).decode("utf-8")


async def _seed_doc(conn: asyncpg.Connection, canonical_id: str) -> int:
    row = await conn.fetchrow(
        """
        INSERT INTO graph_nodes(customer_id, label, canonical_id)
        VALUES ($1, 'Document', $2)
        ON CONFLICT (customer_id, label, canonical_id) DO UPDATE
            SET updated_at = NOW()
        RETURNING node_id
        """,
        CUSTOMER_ID, canonical_id,
    )
    return row["node_id"]


async def _seed_edge(
    conn: asyncpg.Connection,
    *,
    edge_type: str,
    from_node_id: int,
    to_node_id: int,
    properties: dict[str, Any],
    source_system: str = "github",
) -> None:
    await conn.execute(
        """
        INSERT INTO graph_edges
          (customer_id, edge_type, from_node_id, to_node_id, properties, source_system, confidence)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6, 'EXTRACTED')
        """,
        CUSTOMER_ID, edge_type, from_node_id, to_node_id,
        orjson_dumps(properties), source_system,
    )


# ---------------------------------------------------------------------------
# Merge happy-path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_happy_path(live_db) -> None:
    """Two aliases merged into a primary. Edges rewritten with aliased_from set.
    Alias nodes deleted. Provenance merged. Audit + routing rows inserted.
    """
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        p_node = await _seed_person(conn, "richardwei6")
        a1_node = await _seed_person(conn, "mahit@prbe.ai")
        a2_node = await _seed_person(conn, "U07ABC123")
        d_node  = await _seed_doc(conn, "doc-1")
        # Each alias has its own AUTHORED edge to doc-1 with distinct properties.
        await _seed_edge(conn, edge_type="AUTHORED",
                         from_node_id=p_node, to_node_id=d_node,
                         properties={"commit_count": 47, "sha": "abc"})
        await _seed_edge(conn, edge_type="AUTHORED",
                         from_node_id=a1_node, to_node_id=d_node,
                         properties={"commit_count": 23, "sha": "def"},
                         source_system="slack")
        await _seed_edge(conn, edge_type="AUTHORED",
                         from_node_id=a2_node, to_node_id=d_node,
                         properties={"commit_count": 12, "sha": "ghi"},
                         source_system="linear")
        # Provenance: github on p, slack on a1, linear on a2.
        # (graph_node_provenance is INSERTed by graph_writer in real ingest;
        # we mirror it here for the test.)
        for nid, source in (
            (p_node, "github"), (a1_node, "slack"), (a2_node, "linear")
        ):
            await conn.execute(
                """
                INSERT INTO graph_node_provenance
                  (node_id, customer_id, source_system)
                VALUES ($1, $2, $3)
                ON CONFLICT DO NOTHING
                """,
                nid, CUSTOMER_ID, source,
            )

    # POST merge.
    client = TestClient(app)
    resp = client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "richardwei6",
            "alias_canonical_ids":  ["mahit@prbe.ai", "U07ABC123"],
            "reason":               "test merge",
        },
    )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["label"] == "Person"
    assert body["primary_canonical_id"] == "richardwei6"
    assert sorted(body["merged_alias_canonical_ids"]) == ["U07ABC123", "mahit@prbe.ai"]
    uuid.UUID(body["merge_id"])

    # Verify DB state.
    async with with_tenant(CUSTOMER_ID) as conn:
        # Alias nodes gone.
        gone = await conn.fetch(
            "SELECT canonical_id FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Person' "
            "  AND canonical_id IN ('mahit@prbe.ai', 'U07ABC123')",
            CUSTOMER_ID,
        )
        assert gone == []
        # Primary still there.
        p_row = await conn.fetchrow(
            "SELECT node_id, degree FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Person' AND canonical_id = 'richardwei6'",
            CUSTOMER_ID,
        )
        assert p_row is not None
        # Degree recomputed: 3 edges (one per lane).
        assert p_row["degree"] == 3
        # Three AUTHORED edges, each in its own alias lane.
        rows = await conn.fetch(
            """
            SELECT properties, aliased_from_canonical_id
            FROM graph_edges
            WHERE customer_id = $1 AND edge_type = 'AUTHORED'
            ORDER BY (aliased_from_canonical_id IS NULL) DESC, aliased_from_canonical_id
            """,
            CUSTOMER_ID,
        )
        assert len(rows) == 3
        alias_lanes = [r["aliased_from_canonical_id"] for r in rows]
        assert alias_lanes == [None, "U07ABC123", "mahit@prbe.ai"]
        # Provenance merged onto the primary.
        prov = await conn.fetch(
            "SELECT source_system FROM graph_node_provenance "
            "WHERE node_id = $1 ORDER BY source_system",
            p_row["node_id"],
        )
        assert [p["source_system"] for p in prov] == ["github", "linear", "slack"]
        # Routing rows present.
        routing = await conn.fetch(
            "SELECT alias_canonical_id, primary_canonical_id FROM entity_aliases "
            "WHERE customer_id = $1 ORDER BY alias_canonical_id",
            CUSTOMER_ID,
        )
        assert [(r["alias_canonical_id"], r["primary_canonical_id"]) for r in routing] == [
            ("U07ABC123", "richardwei6"),
            ("mahit@prbe.ai", "richardwei6"),
        ]
        # Audit row.
        audit = await conn.fetchrow(
            "SELECT status, merged_alias_canonical_ids FROM entity_merge_audit "
            "WHERE customer_id = $1",
            CUSTOMER_ID,
        )
        assert audit["status"] == "active"
        assert sorted(audit["merged_alias_canonical_ids"]) == ["U07ABC123", "mahit@prbe.ai"]
        # Node snapshots captured.
        snaps = await conn.fetch(
            "SELECT canonical_id FROM entity_merge_node_snapshot "
            "WHERE customer_id = $1 ORDER BY canonical_id",
            CUSTOMER_ID,
        )
        assert [s["canonical_id"] for s in snaps] == ["U07ABC123", "mahit@prbe.ai"]
```

- [ ] **Step 2: Run, confirm 404**

```bash
pytest tests/test_entity_clusters_routes.py::test_merge_happy_path -v
```

Expected: FAIL with 404 — no endpoint exists yet.

### Task B4: Implement the merge endpoint

**Files:**
- Modify: `services/ingestion/entity_clusters_routes.py` (append the handler)

- [ ] **Step 1: Append the merge handler**

At the end of `services/ingestion/entity_clusters_routes.py`, append:

```python
# ---------------------------------------------------------------------------
# POST /api/entity-clusters/merge
# ---------------------------------------------------------------------------


@router.post(
    "/merge",
    response_model=MergeResponse,
    dependencies=[Depends(_require_internal_key)],
)
async def merge_cluster(body: MergeRequest) -> MergeResponse:
    """Run the merge transaction described in the design doc."""
    customer_id = body.customer_id

    if body.primary_canonical_id in body.alias_canonical_ids:
        raise HTTPException(
            status_code=400,
            detail="primary_canonical_id must not appear in alias_canonical_ids",
        )

    all_ids = [body.primary_canonical_id, *body.alias_canonical_ids]
    merge_id = uuid.uuid4()

    async with with_tenant(customer_id) as conn:
        # 1. Existence check.
        existing_rows = await conn.fetch(
            """
            SELECT canonical_id, node_id FROM graph_nodes
            WHERE customer_id = $1 AND label = $2 AND canonical_id = ANY($3::text[])
            """,
            customer_id, body.label, all_ids,
        )
        existing = {r["canonical_id"]: r["node_id"] for r in existing_rows}
        missing = [c for c in all_ids if c not in existing]
        if missing:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "unknown canonical_ids for label",
                    "label": body.label,
                    "missing": missing,
                },
            )
        primary_node_id = existing[body.primary_canonical_id]
        alias_node_ids = [existing[a] for a in body.alias_canonical_ids]

        # 2. None of the aliases are already in another cluster.
        already = await conn.fetch(
            """
            SELECT alias_canonical_id, primary_canonical_id FROM entity_aliases
            WHERE customer_id = $1 AND label = $2
              AND alias_canonical_id = ANY($3::text[])
            """,
            customer_id, body.label, body.alias_canonical_ids,
        )
        if already:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "one or more aliases already belong to a cluster",
                    "conflicting_aliases": {
                        r["alias_canonical_id"]: r["primary_canonical_id"]
                        for r in already
                    },
                },
            )

        # 3. Primary not itself an alias.
        primary_as_alias = await conn.fetchrow(
            """
            SELECT primary_canonical_id FROM entity_aliases
            WHERE customer_id = $1 AND label = $2 AND alias_canonical_id = $3
            LIMIT 1
            """,
            customer_id, body.label, body.primary_canonical_id,
        )
        if primary_as_alias is not None:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "primary_canonical_id is already an alias of another cluster",
                    "actual_primary": primary_as_alias["primary_canonical_id"],
                },
            )

        # 4. Lock every edge touching any alias node.
        await conn.execute(
            """
            SELECT edge_id FROM graph_edges
            WHERE customer_id = $1
              AND (from_node_id = ANY($2::bigint[]) OR to_node_id = ANY($2::bigint[]))
            FOR UPDATE
            """,
            customer_id, alias_node_ids,
        )

        # 5. Insert audit row.
        await conn.execute(
            """
            INSERT INTO entity_merge_audit
              (merge_id, customer_id, label, primary_canonical_id,
               merged_alias_canonical_ids, performed_by_user_id, reason)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            merge_id, customer_id, body.label, body.primary_canonical_id,
            body.alias_canonical_ids, body.performed_by_user_id, body.reason,
        )

        # 6. Snapshot alias nodes (incl. inlined provenance).
        await conn.execute(
            """
            INSERT INTO entity_merge_node_snapshot
              (merge_id, customer_id, label, canonical_id, properties,
               degree, community_id, created_at, provenance)
            SELECT $1, gn.customer_id, gn.label, gn.canonical_id,
                   gn.properties, gn.degree, gn.community_id, gn.created_at,
                   COALESCE(
                     (SELECT jsonb_agg(jsonb_build_object(
                        'source_system', p.source_system,
                        'first_seen_at', p.first_seen_at,
                        'last_seen_at',  p.last_seen_at))
                      FROM graph_node_provenance p
                      WHERE p.node_id = gn.node_id),
                     '[]'::jsonb
                   )
              FROM graph_nodes gn
             WHERE gn.customer_id = $2 AND gn.node_id = ANY($3::bigint[])
            """,
            merge_id, customer_id, alias_node_ids,
        )

        # 7. Merge alias provenance into canonical.
        await conn.execute(
            """
            INSERT INTO graph_node_provenance
              (node_id, customer_id, source_system, first_seen_at, last_seen_at)
            SELECT $1, $2, p.source_system,
                   MIN(p.first_seen_at), MAX(p.last_seen_at)
              FROM graph_node_provenance p
             WHERE p.node_id = ANY($3::bigint[]) AND p.customer_id = $2
             GROUP BY p.source_system
            ON CONFLICT (node_id, source_system) DO UPDATE
              SET first_seen_at = LEAST(graph_node_provenance.first_seen_at, EXCLUDED.first_seen_at),
                  last_seen_at  = GREATEST(graph_node_provenance.last_seen_at, EXCLUDED.last_seen_at)
            """,
            primary_node_id, customer_id, alias_node_ids,
        )

        # 8. Classify each touched edge and apply.
        #    First fetch them so we can classify in Python.
        edges_to_rewrite = await conn.fetch(
            """
            SELECT edge_id, edge_type, from_node_id, to_node_id,
                   properties, confidence, valid_from, valid_to,
                   source_system, extractor_id, extracted_at,
                   aliased_from_canonical_id, aliased_to_canonical_id
            FROM graph_edges
            WHERE customer_id = $1
              AND (from_node_id = ANY($2::bigint[]) OR to_node_id = ANY($2::bigint[]))
            """,
            customer_id, alias_node_ids,
        )
        node_id_to_canonical = {existing[c]: c for c in body.alias_canonical_ids}
        snapshot_seq = 0
        for e in edges_to_rewrite:
            from_aliased = e["from_node_id"] in alias_node_ids
            to_aliased   = e["to_node_id"]   in alias_node_ids
            new_from = primary_node_id if from_aliased else e["from_node_id"]
            new_to   = primary_node_id if to_aliased   else e["to_node_id"]
            new_aliased_from = (
                node_id_to_canonical[e["from_node_id"]]
                if from_aliased else e["aliased_from_canonical_id"]
            )
            new_aliased_to = (
                node_id_to_canonical[e["to_node_id"]]
                if to_aliased else e["aliased_to_canonical_id"]
            )
            # Self-loop after rewrite → snapshot + DELETE.
            if new_from == new_to:
                snapshot_seq += 1
                await conn.execute(
                    """
                    INSERT INTO entity_merge_edge_snapshot
                      (merge_id, snapshot_seq, customer_id, operation,
                       pre_edge_type,
                       pre_from_canonical_id, pre_from_label,
                       pre_to_canonical_id,   pre_to_label,
                       pre_properties, pre_confidence,
                       pre_valid_from, pre_valid_to,
                       pre_source_system, pre_extractor_id, pre_extracted_at,
                       pre_aliased_from_canonical_id, pre_aliased_to_canonical_id)
                    SELECT $1, $2, $3, 'deleted_self_loop',
                           $4,
                           gn_from.canonical_id, gn_from.label,
                           gn_to.canonical_id,   gn_to.label,
                           $5::jsonb, $6, $7, $8, $9, $10, $11, $12, $13
                      FROM graph_nodes gn_from, graph_nodes gn_to
                     WHERE gn_from.node_id = $14 AND gn_to.node_id = $15
                    """,
                    merge_id, snapshot_seq, customer_id,
                    e["edge_type"],
                    e["properties"], e["confidence"],
                    e["valid_from"], e["valid_to"],
                    e["source_system"], e["extractor_id"], e["extracted_at"],
                    e["aliased_from_canonical_id"], e["aliased_to_canonical_id"],
                    e["from_node_id"], e["to_node_id"],
                )
                await conn.execute(
                    "DELETE FROM graph_edges WHERE edge_id = $1", e["edge_id"]
                )
                continue
            # Clean rewrite: UPDATE endpoints + stamp aliased_from/to.
            await conn.execute(
                """
                UPDATE graph_edges
                   SET from_node_id = $1,
                       to_node_id   = $2,
                       aliased_from_canonical_id = $3,
                       aliased_to_canonical_id   = $4
                 WHERE edge_id = $5
                """,
                new_from, new_to, new_aliased_from, new_aliased_to,
                e["edge_id"],
            )

        # 9. Hard-delete alias graph_nodes (CASCADE drops their remaining
        #    provenance rows — already merged into canonical at step 7).
        await conn.execute(
            "DELETE FROM graph_nodes "
            "WHERE customer_id = $1 AND node_id = ANY($2::bigint[])",
            customer_id, alias_node_ids,
        )

        # 10. Recompute degree on canonical.
        await conn.execute(
            """
            UPDATE graph_nodes
               SET degree = (
                   SELECT COUNT(*) FROM graph_edges
                    WHERE customer_id = $1
                      AND (from_node_id = $2 OR to_node_id = $2)
               )
             WHERE customer_id = $1 AND node_id = $2
            """,
            customer_id, primary_node_id,
        )

        # 11. INSERT routing rows.
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            SELECT $1, $2, alias, $3, $4
              FROM UNNEST($5::text[]) AS alias
            """,
            customer_id, body.label, body.primary_canonical_id, merge_id,
            body.alias_canonical_ids,
        )

    log.info(
        "entity_clusters.merge: customer=%s label=%s primary=%s aliases=%d merge_id=%s",
        customer_id, body.label, body.primary_canonical_id,
        len(body.alias_canonical_ids), merge_id,
    )
    return MergeResponse(
        merge_id=merge_id,
        label=body.label,
        primary_canonical_id=body.primary_canonical_id,
        merged_alias_canonical_ids=list(body.alias_canonical_ids),
    )
```

- [ ] **Step 2: Run the happy-path test, confirm it passes**

```bash
pytest tests/test_entity_clusters_routes.py::test_merge_happy_path -v
```

Expected: PASS.

### Task B5: Write the failing edge-case tests for merge

**Files:**
- Modify: `tests/test_entity_clusters_routes.py` (append)

- [ ] **Step 1: Append the validation tests**

```python
# ---------------------------------------------------------------------------
# Merge validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_404_on_missing_canonical_id(live_db) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "richardwei6")
        # 'unknown-id' does not exist.
    client = TestClient(app)
    resp = client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "richardwei6",
            "alias_canonical_ids":  ["unknown-id"],
        },
    )
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert detail["error"] == "unknown canonical_ids for label"
    assert detail["missing"] == ["unknown-id"]


@pytest.mark.asyncio
async def test_merge_409_when_alias_already_in_cluster(live_db) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "richardwei6")
        await _seed_person(conn, "second-primary")
        await _seed_person(conn, "U07ABC123")
        # Pre-existing cluster: U07ABC123 belongs to second-primary.
        merge_id_existing = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO entity_merge_audit
              (merge_id, customer_id, label, primary_canonical_id,
               merged_alias_canonical_ids, performed_by_user_id)
            VALUES ($1, $2, 'Person', 'second-primary',
                    ARRAY['U07ABC123']::text[], $3)
            """,
            merge_id_existing, CUSTOMER_ID, uuid.UUID(USER_ID),
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            VALUES ($1, 'Person', 'U07ABC123', 'second-primary', $2)
            """,
            CUSTOMER_ID, merge_id_existing,
        )
    client = TestClient(app)
    resp = client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "richardwei6",
            "alias_canonical_ids":  ["U07ABC123"],
        },
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["conflicting_aliases"] == {"U07ABC123": "second-primary"}


@pytest.mark.asyncio
async def test_merge_409_when_primary_is_already_alias(live_db) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "actual-primary")
        await _seed_person(conn, "richardwei6")
        await _seed_person(conn, "extra-alias")
        merge_id_existing = uuid.uuid4()
        await conn.execute(
            """
            INSERT INTO entity_merge_audit
              (merge_id, customer_id, label, primary_canonical_id,
               merged_alias_canonical_ids, performed_by_user_id)
            VALUES ($1, $2, 'Person', 'actual-primary',
                    ARRAY['richardwei6']::text[], $3)
            """,
            merge_id_existing, CUSTOMER_ID, uuid.UUID(USER_ID),
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases
              (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id)
            VALUES ($1, 'Person', 'richardwei6', 'actual-primary', $2)
            """,
            CUSTOMER_ID, merge_id_existing,
        )
    client = TestClient(app)
    resp = client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "richardwei6",
            "alias_canonical_ids":  ["extra-alias"],
        },
    )
    assert resp.status_code == 409
    detail = resp.json()["detail"]
    assert detail["actual_primary"] == "actual-primary"


def test_merge_400_when_primary_in_aliases():
    """No DB setup needed — request fails at the early body check."""
    client = TestClient(app)
    resp = client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "x",
            "alias_canonical_ids":  ["x"],
        },
    )
    assert resp.status_code == 400


def test_merge_422_on_duplicate_aliases():
    client = TestClient(app)
    resp = client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "richardwei6",
            "alias_canonical_ids":  ["a", "a"],
        },
    )
    assert resp.status_code == 422


def test_merge_401_without_internal_key():
    client = TestClient(app)
    resp = client.post(
        "/api/entity-clusters/merge",
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "x",
            "alias_canonical_ids":  ["y"],
        },
    )
    assert resp.status_code == 401
```

- [ ] **Step 2: Run them, confirm all pass**

```bash
pytest tests/test_entity_clusters_routes.py -v -k "merge"
```

Expected: All `test_merge_*` PASS. The implementation in Task B4 already covers them — these tests pin the behavior.

### Task B6: Write the failing unmerge tests

**Files:**
- Modify: `tests/test_entity_clusters_routes.py` (append)

- [ ] **Step 1: Append unmerge tests**

```python
# ---------------------------------------------------------------------------
# Unmerge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unmerge_happy_path_restores_alias_and_edges(live_db) -> None:
    """Unmerge one alias: node restored from snapshot, edges UPDATEd back."""
    async with raw_conn() as conn:
        await _seed_customer(conn)
    # Same setup as merge happy-path, then perform merge via the API so the
    # state we unmerge against is exactly what the merge endpoint produces.
    async with with_tenant(CUSTOMER_ID) as conn:
        p_node  = await _seed_person(conn, "richardwei6")
        _a1node = await _seed_person(conn, "mahit@prbe.ai")
        d_node  = await _seed_doc(conn, "doc-1")
        await _seed_edge(conn, edge_type="AUTHORED",
                         from_node_id=p_node, to_node_id=d_node,
                         properties={"commit_count": 47})
        await _seed_edge(conn, edge_type="AUTHORED",
                         from_node_id=_a1node, to_node_id=d_node,
                         properties={"commit_count": 23})
    client = TestClient(app)
    merge_resp = client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": "richardwei6",
            "alias_canonical_ids":  ["mahit@prbe.ai"],
        },
    )
    assert merge_resp.status_code == 200
    merge_id = merge_resp.json()["merge_id"]

    # Unmerge.
    unmerge_resp = client.delete(
        "/api/entity-clusters/Person/richardwei6/aliases/mahit@prbe.ai",
        headers=_headers(),
    )
    assert unmerge_resp.status_code == 204

    # Verify: alias node back; edges rewritten to alias; audit reversed.
    async with with_tenant(CUSTOMER_ID) as conn:
        alias = await conn.fetchrow(
            "SELECT node_id FROM graph_nodes "
            "WHERE customer_id = $1 AND label = 'Person' AND canonical_id = 'mahit@prbe.ai'",
            CUSTOMER_ID,
        )
        assert alias is not None
        # AUTHORED edges: one on primary (NULL lane), one on the restored alias (NULL lane).
        rows = await conn.fetch(
            """
            SELECT ge.aliased_from_canonical_id, gn.canonical_id AS from_canonical
            FROM graph_edges ge
            JOIN graph_nodes gn ON gn.node_id = ge.from_node_id
            WHERE ge.customer_id = $1 AND ge.edge_type = 'AUTHORED'
            ORDER BY gn.canonical_id
            """,
            CUSTOMER_ID,
        )
        assert [(r["from_canonical"], r["aliased_from_canonical_id"]) for r in rows] == [
            ("mahit@prbe.ai", None),
            ("richardwei6",   None),
        ]
        # Audit: this was the only alias, so status flips to 'reversed'.
        audit = await conn.fetchrow(
            "SELECT status FROM entity_merge_audit WHERE merge_id = $1",
            uuid.UUID(merge_id),
        )
        assert audit["status"] == "reversed"
        # Routing row gone.
        routing = await conn.fetch(
            "SELECT 1 FROM entity_aliases WHERE customer_id = $1",
            CUSTOMER_ID,
        )
        assert routing == []


def test_unmerge_404_when_alias_not_in_cluster():
    client = TestClient(app)
    resp = client.delete(
        "/api/entity-clusters/Person/whatever/aliases/nothing",
        headers=_headers(),
    )
    assert resp.status_code == 404


def test_unmerge_401_without_internal_key():
    client = TestClient(app)
    resp = client.delete(
        "/api/entity-clusters/Person/x/aliases/y",
    )
    assert resp.status_code == 401
```

- [ ] **Step 2: Run, confirm failures**

```bash
pytest tests/test_entity_clusters_routes.py -v -k "unmerge"
```

Expected: FAIL — unmerge endpoint not registered yet.

### Task B7: Implement the unmerge endpoint

**Files:**
- Modify: `services/ingestion/entity_clusters_routes.py` (append)

- [ ] **Step 1: Append the handler**

```python
# ---------------------------------------------------------------------------
# DELETE /api/entity-clusters/{label}/{primary}/aliases/{alias}
# ---------------------------------------------------------------------------


@router.delete(
    "/{label}/{primary_canonical_id}/aliases/{alias_canonical_id}",
    status_code=204,
    dependencies=[Depends(_require_internal_key)],
)
async def unmerge_alias(
    label: str = Path(..., min_length=1, max_length=64),
    primary_canonical_id: str = Path(..., min_length=1, max_length=512),
    alias_canonical_id: str = Path(..., min_length=1, max_length=512),
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> None:
    """Remove a single alias from its cluster."""
    if not x_prbe_customer:
        raise HTTPException(
            status_code=400,
            detail="X-Prbe-Customer header is required",
        )
    customer_id = x_prbe_customer

    async with with_tenant(customer_id) as conn:
        # 1. Look up the merge_id; 404 if no routing row.
        existing = await conn.fetchrow(
            """
            SELECT merge_id FROM entity_aliases
            WHERE customer_id = $1 AND label = $2
              AND alias_canonical_id = $3 AND primary_canonical_id = $4
            """,
            customer_id, label, alias_canonical_id, primary_canonical_id,
        )
        if existing is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no alias '{alias_canonical_id}' under cluster "
                    f"'{primary_canonical_id}' for label '{label}'"
                ),
            )
        merge_id = existing["merge_id"]

        # 2. Re-INSERT the alias node from snapshot.
        new_alias = await conn.fetchrow(
            """
            INSERT INTO graph_nodes
              (customer_id, label, canonical_id, properties,
               degree, community_id, created_at, updated_at)
            SELECT customer_id, label, canonical_id, properties,
                   degree, community_id, created_at, NOW()
              FROM entity_merge_node_snapshot
             WHERE merge_id = $1 AND label = $2 AND canonical_id = $3
             RETURNING node_id
            """,
            merge_id, label, alias_canonical_id,
        )
        if new_alias is None:
            # Should not happen — snapshot is captured on merge.
            raise HTTPException(
                status_code=500,
                detail="missing node snapshot for alias",
            )
        new_alias_node_id = new_alias["node_id"]

        # 3. Restore alias provenance from inlined JSONB.
        await conn.execute(
            """
            INSERT INTO graph_node_provenance
              (node_id, customer_id, source_system, first_seen_at, last_seen_at)
            SELECT $1, customer_id,
                   p->>'source_system',
                   (p->>'first_seen_at')::timestamptz,
                   (p->>'last_seen_at')::timestamptz
              FROM entity_merge_node_snapshot,
                   LATERAL jsonb_array_elements(provenance) AS p
             WHERE merge_id = $2 AND label = $3 AND canonical_id = $4
            ON CONFLICT (node_id, source_system) DO NOTHING
            """,
            new_alias_node_id, merge_id, label, alias_canonical_id,
        )

        # 4. Rewrite edges back: from-side and to-side independently.
        await conn.execute(
            """
            UPDATE graph_edges
               SET from_node_id = $1,
                   aliased_from_canonical_id = NULL
             WHERE customer_id = $2 AND aliased_from_canonical_id = $3
            """,
            new_alias_node_id, customer_id, alias_canonical_id,
        )
        await conn.execute(
            """
            UPDATE graph_edges
               SET to_node_id = $1,
                   aliased_to_canonical_id = NULL
             WHERE customer_id = $2 AND aliased_to_canonical_id = $3
            """,
            new_alias_node_id, customer_id, alias_canonical_id,
        )

        # 5. Re-INSERT snapshotted self-loops involving this alias.
        await conn.execute(
            """
            INSERT INTO graph_edges
              (customer_id, edge_type, from_node_id, to_node_id,
               properties, valid_from, valid_to, source_system,
               confidence, extractor_id, extracted_at,
               aliased_from_canonical_id, aliased_to_canonical_id)
            SELECT s.customer_id, s.pre_edge_type,
                   $1, $1,  -- self-loop on the restored alias node
                   s.pre_properties, s.pre_valid_from, s.pre_valid_to,
                   s.pre_source_system, s.pre_confidence,
                   s.pre_extractor_id, s.pre_extracted_at,
                   s.pre_aliased_from_canonical_id, s.pre_aliased_to_canonical_id
              FROM entity_merge_edge_snapshot s
             WHERE s.merge_id = $2
               AND s.operation = 'deleted_self_loop'
               AND (s.pre_from_canonical_id = $3 OR s.pre_to_canonical_id = $3)
            """,
            new_alias_node_id, merge_id, alias_canonical_id,
        )

        # 6. Recompute degree on primary + restored alias.
        primary_row = await conn.fetchrow(
            """
            SELECT node_id FROM graph_nodes
            WHERE customer_id = $1 AND label = $2 AND canonical_id = $3
            """,
            customer_id, label, primary_canonical_id,
        )
        if primary_row is not None:
            await conn.execute(
                """
                UPDATE graph_nodes
                   SET degree = (
                       SELECT COUNT(*) FROM graph_edges
                        WHERE customer_id = $1
                          AND (from_node_id = graph_nodes.node_id
                               OR to_node_id = graph_nodes.node_id)
                   )
                 WHERE customer_id = $1 AND node_id IN ($2, $3)
                """,
                customer_id, primary_row["node_id"], new_alias_node_id,
            )

        # 7. Drop routing row + flip audit status if last.
        await conn.execute(
            """
            DELETE FROM entity_aliases
            WHERE customer_id = $1 AND label = $2 AND alias_canonical_id = $3
            """,
            customer_id, label, alias_canonical_id,
        )
        await conn.execute(
            """
            UPDATE entity_merge_audit SET status = 'reversed'
            WHERE merge_id = $1
              AND NOT EXISTS (
                  SELECT 1 FROM entity_aliases WHERE merge_id = $1
              )
            """,
            merge_id,
        )

    log.info(
        "entity_clusters.unmerge: customer=%s label=%s primary=%s alias=%s",
        customer_id, label, primary_canonical_id, alias_canonical_id,
    )
```

**Note:** the unmerge endpoint requires `X-Prbe-Customer` header because there's no body to carry customer_id. The BFF (PR 1c) sets it from the JWT session.

- [ ] **Step 2: Update the unmerge tests to send X-Prbe-Customer**

In `tests/test_entity_clusters_routes.py`, change `_headers()` to:

```python
def _headers() -> dict[str, str]:
    return {
        "X-Internal-Knowledge-Key": get_settings().internal_knowledge_api_key,
        "X-Prbe-Customer": CUSTOMER_ID,
    }
```

(This is also fine for the merge endpoint, which ignores unknown headers; the field for customer_id is in the body.)

- [ ] **Step 3: Run all entity-clusters tests, confirm green**

```bash
pytest tests/test_entity_clusters_routes.py -v
```

Expected: All tests PASS.

### Task B8: Run full prbe-knowledge test suite + commit PR 1b

- [ ] **Step 1: Run suite**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1
pytest -q
```

Expected: All tests pass.

- [ ] **Step 2: Stage + commit**

```bash
git add services/ingestion/entity_clusters_routes.py \
        services/ingestion/main.py \
        tests/test_entity_clusters_routes.py
git commit -m "$(cat <<'EOF'
feat(ingestion): /api/entity-clusters merge + unmerge endpoints (Phase 1b)

Internal API under /api/entity-clusters/* gated by X-Internal-Knowledge-Key.

  POST   /api/entity-clusters/merge
  DELETE /api/entity-clusters/{label}/{primary}/aliases/{alias}

Merge runs the full B-promote transaction:
  validate (404/409 on bad inputs) → lock alias edges → audit row →
  snapshot alias nodes + inline provenance → merge provenance into
  canonical → classify edges (rewrite vs. self-loop) → execute →
  hard-delete alias nodes → recompute canonical degree → routing rows.

Unmerge inverts via snapshot + aliased_from/to columns: re-INSERT alias
node, restore provenance, UPDATE rewritten edges back via
aliased_from/to_canonical_id, re-INSERT self-loop snapshots, recompute
degree on primary + restored alias, drop routing row, flip audit status
when last alias is removed.

Tests cover happy-path + every validation error (404/409 alias-already/
409 primary-is-alias/400 self-in-aliases/422 dup/401 missing-key).

Phase 1c adds the BFF thin wrappers that JWT-auth + forward here.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## PR 1c — prbe-backend: BFF thin wrappers

**Worktree:** `cd /Users/mahitnamburu/Desktop/prbe/prbe-backend-worktrees/entity-clusters-phase1`

### Task C1: Scaffold the entity_clusters BFF router

**Files:**
- Create: `apps/data_plane/routers/dashboard/entity_clusters.py`

- [ ] **Step 1: Write the wrapper**

```python
"""BFF thin wrappers around prbe-knowledge's /api/entity-clusters/* endpoints.

JWT-validated, admin-role-only. Forwards to prbe-knowledge with
X-Internal-Knowledge-Key + X-Prbe-Customer, injecting customer_id +
performed_by_user_id from the session.

Mirrors the proxy pattern used by /knowledge/query and /knowledge/retrieve
in apps/data_plane/routers/dashboard/knowledge.py.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, ConfigDict, Field, field_validator

from apps.data_plane.config import get_settings
from apps.data_plane.dependencies.jwt import Session, require_role

log = logging.getLogger(__name__)
router = APIRouter(prefix="/knowledge/entity-clusters", tags=["entity-clusters"])


def _customer_or_404(session: Session) -> str:
    if session.customer_id is None:
        raise HTTPException(
            status_code=409,
            detail="no customer linked to this organization",
        )
    return session.customer_id


# ---------------------------------------------------------------------------
# Pydantic models — mirror prbe-knowledge's MergeRequest / MergeResponse
# ---------------------------------------------------------------------------


class MergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label:                str        = Field(..., min_length=1, max_length=64)
    primary_canonical_id: str        = Field(..., min_length=1, max_length=512)
    alias_canonical_ids:  list[str]  = Field(..., min_length=1, max_length=64)
    reason:               str | None = Field(default=None, max_length=2000)

    @field_validator("alias_canonical_ids")
    @classmethod
    def _unique_non_blank(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("alias_canonical_ids must be unique")
        for a in v:
            if not a or not a.strip():
                raise ValueError("alias_canonical_ids may not contain blanks")
        return v


class MergeResponse(BaseModel):
    merge_id:                   uuid.UUID
    label:                      str
    primary_canonical_id:       str
    merged_alias_canonical_ids: list[str]


# ---------------------------------------------------------------------------
# POST /knowledge/entity-clusters/merge
# ---------------------------------------------------------------------------


@router.post("/merge", response_model=MergeResponse)
async def merge(
    body: MergeRequest,
    session: Session = Depends(require_role("admin")),
) -> MergeResponse:
    customer_id = _customer_or_404(session)
    settings = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.knowledge_base_url}/api/entity-clusters/merge",
            headers={
                "X-Internal-Knowledge-Key": settings.internal_knowledge_api_key,
            },
            json={
                **body.model_dump(),
                "customer_id":          customer_id,
                "performed_by_user_id": session.user_id,
            },
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return MergeResponse.model_validate(resp.json())


# ---------------------------------------------------------------------------
# DELETE /knowledge/entity-clusters/{label}/{primary}/aliases/{alias}
# ---------------------------------------------------------------------------


@router.delete(
    "/{label}/{primary_canonical_id}/aliases/{alias_canonical_id}",
    status_code=204,
)
async def unmerge(
    label: str = Path(..., min_length=1, max_length=64),
    primary_canonical_id: str = Path(..., min_length=1, max_length=512),
    alias_canonical_id: str = Path(..., min_length=1, max_length=512),
    session: Session = Depends(require_role("admin")),
) -> None:
    customer_id = _customer_or_404(session)
    settings = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.delete(
            f"{settings.knowledge_base_url}"
            f"/api/entity-clusters/{label}/{primary_canonical_id}"
            f"/aliases/{alias_canonical_id}",
            headers={
                "X-Internal-Knowledge-Key": settings.internal_knowledge_api_key,
                "X-Prbe-Customer":          customer_id,
            },
        )
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except ValueError:
            detail = resp.text
        raise HTTPException(status_code=resp.status_code, detail=detail)
    return None
```

**Note on `settings.knowledge_base_url`:** the BFF settings module may not yet have this attribute. If `from apps.data_plane.config import get_settings` doesn't expose `knowledge_base_url`, add the attribute to the Settings dataclass — likely a one-line change next to other knowledge-URL settings (e.g. `KNOWLEDGE_QUERY_BASE_URL`). Reuse whatever existing variable already points at the prbe-knowledge ingestion service. If unsure, grep:

```bash
grep -rn "knowledge.*base_url\|KNOWLEDGE.*BASE_URL" apps/data_plane/config.py
```

### Task C2: Mount the router

**Files:**
- Modify: `apps/data_plane/routers/dashboard/__init__.py`

- [ ] **Step 1: Import + include**

After the existing `from apps.data_plane.routers.dashboard.devices import router as devices_router` line (or any other dashboard router import), insert:

```python
from apps.data_plane.routers.dashboard.entity_clusters import (
    router as entity_clusters_router,
)
```

After the last existing `router.include_router(...)` (likely `router.include_router(workspace_prefs_router)`), insert:

```python
router.include_router(entity_clusters_router)
```

- [ ] **Step 2: Verify the app boots**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-backend-worktrees/entity-clusters-phase1
python -c "from apps.data_plane.main import app; print(sorted({r.path for r in app.routes if 'entity-clusters' in str(r.path)}))"
```

Expected: `['/knowledge/entity-clusters/merge', '/knowledge/entity-clusters/{label}/{primary_canonical_id}/aliases/{alias_canonical_id}']` (or similar — both endpoints registered).

### Task C3: Write the failing wrapper tests

**Files:**
- Create: `tests/test_dashboard_entity_clusters.py`

- [ ] **Step 1: Write the test file**

```python
"""Tests for /knowledge/entity-clusters/merge and /unmerge BFF wrappers.

Mirrors the proxy-test pattern from test_knowledge_query_proxy.py — we
patch httpx.AsyncClient so the wrapper doesn't hit the network, then
assert the forwarded request shape (URL, headers, body) and that the
upstream response bubbles up unchanged.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from apps.data_plane.dependencies.jwt import Session, require_session
from apps.data_plane.main import app

CUSTOMER_ID = "cust-test-ec"
USER_ID = "11111111-1111-1111-1111-111111111111"


async def _stub_admin_session() -> Session:
    return Session(
        user_id=USER_ID,
        email="admin@example.com",
        organization_id="org-ec",
        customer_id=CUSTOMER_ID,
        role="admin",
        dev_enabled=False,
    )


async def _stub_member_session() -> Session:
    return Session(
        user_id=USER_ID,
        email="member@example.com",
        organization_id="org-ec",
        customer_id=CUSTOMER_ID,
        role="member",
        dev_enabled=False,
    )


@pytest.fixture(autouse=True)
def _admin_session(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_BASE_URL", "http://knowledge.test")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    from apps.data_plane.config import get_settings
    get_settings.cache_clear()
    app.dependency_overrides[require_session] = _stub_admin_session
    yield
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def _mock_post(status_code: int = 200, body: dict | None = None):
    """Build a httpx.AsyncClient-like mock that returns the given response."""
    response = MagicMock()
    response.status_code = status_code
    response.json = MagicMock(return_value=body or {})
    response.text = "" if body else "error"
    client = MagicMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.post = AsyncMock(return_value=response)
    client.delete = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# Merge happy-path
# ---------------------------------------------------------------------------


def test_merge_forwards_to_knowledge_with_injected_fields():
    merge_id = str(uuid.uuid4())
    knowledge_resp = {
        "merge_id":                   merge_id,
        "label":                      "Person",
        "primary_canonical_id":       "richardwei6",
        "merged_alias_canonical_ids": ["mahit@prbe.ai"],
    }
    client_mock = _mock_post(status_code=200, body=knowledge_resp)
    with patch("httpx.AsyncClient", return_value=client_mock):
        resp = TestClient(app).post(
            "/knowledge/entity-clusters/merge",
            json={
                "label":                "Person",
                "primary_canonical_id": "richardwei6",
                "alias_canonical_ids":  ["mahit@prbe.ai"],
            },
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body == knowledge_resp

    # Assert forwarded request shape.
    call = client_mock.post.call_args
    assert call.args[0] == "http://knowledge.test/api/entity-clusters/merge"
    headers = call.kwargs["headers"]
    assert headers["X-Internal-Knowledge-Key"] == "test-internal-key"
    forwarded = call.kwargs["json"]
    assert forwarded["customer_id"] == CUSTOMER_ID
    assert forwarded["performed_by_user_id"] == USER_ID
    assert forwarded["label"] == "Person"
    assert forwarded["primary_canonical_id"] == "richardwei6"
    assert forwarded["alias_canonical_ids"] == ["mahit@prbe.ai"]


def test_merge_bubbles_up_409_from_knowledge():
    client_mock = _mock_post(
        status_code=409,
        body={"detail": {"error": "conflict", "conflicting_aliases": {"a": "x"}}},
    )
    with patch("httpx.AsyncClient", return_value=client_mock):
        resp = TestClient(app).post(
            "/knowledge/entity-clusters/merge",
            json={
                "label":                "Person",
                "primary_canonical_id": "richardwei6",
                "alias_canonical_ids":  ["a"],
            },
        )
    assert resp.status_code == 409
    assert resp.json()["detail"]["conflicting_aliases"] == {"a": "x"}


def test_merge_403_for_member_role():
    app.dependency_overrides[require_session] = _stub_member_session
    try:
        resp = TestClient(app).post(
            "/knowledge/entity-clusters/merge",
            json={
                "label":                "Person",
                "primary_canonical_id": "x",
                "alias_canonical_ids":  ["y"],
            },
        )
        assert resp.status_code == 403
    finally:
        app.dependency_overrides[require_session] = _stub_admin_session


def test_merge_422_on_duplicate_aliases():
    resp = TestClient(app).post(
        "/knowledge/entity-clusters/merge",
        json={
            "label":                "Person",
            "primary_canonical_id": "richardwei6",
            "alias_canonical_ids":  ["a", "a"],
        },
    )
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Unmerge
# ---------------------------------------------------------------------------


def test_unmerge_forwards_to_knowledge_with_customer_header():
    client_mock = _mock_post(status_code=204, body=None)
    with patch("httpx.AsyncClient", return_value=client_mock):
        resp = TestClient(app).delete(
            "/knowledge/entity-clusters/Person/richardwei6/aliases/mahit@prbe.ai",
        )
    assert resp.status_code == 204, resp.text
    call = client_mock.delete.call_args
    url = call.args[0]
    assert url.endswith(
        "/api/entity-clusters/Person/richardwei6/aliases/mahit@prbe.ai"
    )
    headers = call.kwargs["headers"]
    assert headers["X-Internal-Knowledge-Key"] == "test-internal-key"
    assert headers["X-Prbe-Customer"] == CUSTOMER_ID


def test_unmerge_bubbles_up_404_from_knowledge():
    client_mock = _mock_post(
        status_code=404,
        body={"detail": "no alias 'mahit@prbe.ai' under cluster 'richardwei6' for label 'Person'"},
    )
    with patch("httpx.AsyncClient", return_value=client_mock):
        resp = TestClient(app).delete(
            "/knowledge/entity-clusters/Person/richardwei6/aliases/mahit@prbe.ai",
        )
    assert resp.status_code == 404


def test_unmerge_403_for_member_role():
    app.dependency_overrides[require_session] = _stub_member_session
    try:
        resp = TestClient(app).delete(
            "/knowledge/entity-clusters/Person/richardwei6/aliases/mahit@prbe.ai",
        )
        assert resp.status_code == 403
    finally:
        app.dependency_overrides[require_session] = _stub_admin_session
```

- [ ] **Step 2: Run, confirm all pass**

```bash
pytest tests/test_dashboard_entity_clusters.py -v
```

Expected: All 7 tests PASS. (The wrapper from Task C1 + the dependency-overrides in this file cover the behavior.)

If a test fails because `KNOWLEDGE_BASE_URL` isn't a recognized setting on `apps/data_plane/config.py:Settings`, add the field there alongside other knowledge-URL settings (likely under a comment like "prbe-knowledge service URLs"). One-line `knowledge_base_url: str` with a sensible default for local dev. Re-run.

### Task C4: Run full prbe-backend test suite + commit PR 1c

- [ ] **Step 1: Run suite**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-backend-worktrees/entity-clusters-phase1
pytest -q
```

Expected: All tests pass.

- [ ] **Step 2: Stage + commit**

```bash
git add apps/data_plane/routers/dashboard/entity_clusters.py \
        apps/data_plane/routers/dashboard/__init__.py \
        apps/data_plane/config.py \
        tests/test_dashboard_entity_clusters.py
git commit -m "$(cat <<'EOF'
feat(bff): entity-cluster merge/unmerge thin wrappers (Phase 1c)

Adds two admin-only BFF endpoints:

  POST   /knowledge/entity-clusters/merge
  DELETE /knowledge/entity-clusters/{label}/{primary}/aliases/{alias}

Both validate the JWT session (admin role minimum) and forward to
prbe-knowledge's /api/entity-clusters/* with X-Internal-Knowledge-Key.
customer_id + performed_by_user_id are injected from the session so
clients cannot spoof them. Knowledge-side errors bubble up unchanged.

Depends on prbe-knowledge#XXXX (Phase 1a) and prbe-knowledge#XXXX
(Phase 1b) being merged and deployed. Once 1c deploys, the dashboard
becomes able to merge / unmerge entity clusters.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Replace `prbe-knowledge#XXXX` with the actual PR numbers once 1a and 1b are opened.)

---

## Stacking the PRs

### Task S1: Push and open three coordinated PRs

- [ ] **Step 1: Push PR 1a (prbe-knowledge)**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase1
git log --oneline origin/main..HEAD
# Expect two commits: 1a then 1b.
```

PR 1a and 1b are on the same branch. We need two stacked PRs in prbe-knowledge. Two options:

**Option Y — single combined PR for 1a + 1b.** Faster to merge; reviewer eats both at once. ~1200 LOC.

**Option Z — split into two branches.**
  1. `git branch entity-clusters-phase1a HEAD~1` (1a commit only).
  2. `git push -u origin entity-clusters-phase1a`.
  3. Open PR 1a targeting `main`.
  4. After 1a merges, rebase the `entity-clusters-phase1` branch onto `main` so it only carries the 1b commit.
  5. Push and open PR 1b targeting `main`.

Pick based on review appetite. Default recommendation: **Option Z** because 1a touches the hot write path and deserves a focused review.

- [ ] **Step 2: Push PR 1c (prbe-backend)**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-backend-worktrees/entity-clusters-phase1
git push -u origin entity-clusters-phase1
gh pr create --title "feat(bff): entity-cluster merge/unmerge thin wrappers (Phase 1c)" --body "$(cat <<'EOF'
## Summary
- Admin-only BFF endpoints proxying prbe-knowledge's /api/entity-clusters/*.
- customer_id + performed_by_user_id injected from JWT session.

## Test plan
- [ ] pytest tests/test_dashboard_entity_clusters.py -v — all 7 tests pass
- [ ] pytest -q — full suite green
- [ ] After deploy: smoke-test via dashboard cookies — merge two Person canonical_ids, confirm DB state, unmerge one, confirm reversal

## Depends on
- prbe-knowledge#XXXX (Phase 1a)
- prbe-knowledge#XXXX (Phase 1b)

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Replace the PR numbers once 1a and 1b are opened.

---

## Acceptance criteria (end of Phase 1)

- [ ] Five new tables exist in the data plane DB, all RLS-isolated.
- [ ] `graph_edges` has the two new alias provenance columns + composite UNIQUE index.
- [ ] Migration test suite passes (9 tests across the migration test file).
- [ ] graph_writer alias-resolution suite passes (4 tests).
- [ ] prbe-knowledge endpoint test suite passes (8 tests).
- [ ] prbe-backend wrapper test suite passes (7 tests).
- [ ] All three suites' full pytest runs are green.
- [ ] An operator with admin role can, end to end:
  - merge two duplicate Person canonical_ids; observe alias nodes gone, edges rewritten with `aliased_from_canonical_id` set, primary's degree updated, audit row inserted
  - have a post-merge webhook event for an alias canonical_id land on the primary node (verify via integration test or smoke check)
  - hit a 409 on attempting to re-merge the same alias (with the existing primary returned in the body)
  - hit a 404 on a typo'd canonical_id
  - unmerge an alias and see the underlying audit row transition to `'reversed'` once empty; alias node restored; edges back on the alias's NULL lane
- [ ] Pre-merge retrieval behavior is unchanged.
- [ ] No churn to services/retrieval/* (that's Phase 2).

---

*Plan written 2026-05-14 against the B-promote design. See `2026-05-13-entity-clusters-design.md` in this same directory for the full design rationale.*
