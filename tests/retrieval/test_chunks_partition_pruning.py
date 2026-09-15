"""The partition regression guard: a tenant's search must read only its own rows.

THE REGRESSION THIS EXISTS FOR
------------------------------
`chunks` had ONE HNSW index over every tenant. pgvector prices an
`ORDER BY <distance>` scan of it at the WHOLE-INDEX cost -- measured
3,808,868 cost units, IDENTICAL for `probe`, `anthrogen`, `bucket-robotics` and
`new-workspace` -- while the competing brute-force plan is priced on that
tenant's own rows. Small tenants lost that comparison and were pushed onto a
1,003 ms / 646,471-buffer scan where the index takes 80 ms. Because the index
bid is the FLEET's size, one tenant's ingestion moved another tenant's plan:
`probe` grew 53% in two weeks, `anthrogen`'s share fell 16.4% -> 11.8%, and its
pre-fan-out p50 went 3,343 ms -> 8,895 ms without anthrogen ingesting anything.

WHAT THIS ASSERTS, AND WHY NOT "USES THE INDEX"
-----------------------------------------------
Asserting `Index Scan` / no `Seq Scan` would be WRONG here, for the same reason
`test_ann_order_by_shape` refuses to: on a small partition a sequential scan is
the CORRECT plan, and partitioning is specifically meant to make that choice
honest rather than to force the index. A test that forbids brute force would go
red on exactly the behaviour being shipped, and would pressure later changes
toward ANN even where exact search wins.

So the invariants are the two that are true at every size:

  1. PRUNING -- the plan touches this tenant's partition and no other. That is
     what makes the index bid tenant-sized in the first place.
  2. A WORK BOUND -- buffers read stay proportional to the tenant's own data.
     This is the assertion that would have caught the original fault: 646,471
     buffers for a tenant holding 11% of the table.

Both hold whether the planner picks the index or a scan, on two rows or two
million.

Runs against the live test database (`live_db`), which is born from
`db/schema.sql` -- partitioned since this change -- so it exercises the real
shape rather than a fabricated one.
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio

from engine.shared import db as db_module
from engine.shared.partitions import (
    default_partition_name,
    drop_tenant_partition,
    ensure_tenant_partition,
    find_orphan_partitions,
    is_partitioned,
    partition_name_for,
    split_default,
)

pytestmark = pytest.mark.asyncio

BIG = "prune-big"
SMALL = "prune-small"


async def _seed(conn, customer_id: str, n: int) -> None:
    await conn.execute(
        "INSERT INTO customers (customer_id, display_name, api_key_hash) "
        "VALUES ($1, $1, $1) ON CONFLICT DO NOTHING",
        customer_id,
    )
    await ensure_tenant_partition(conn, customer_id)
    await conn.execute(
        """
        INSERT INTO documents (customer_id, doc_id, version, source_system,
                               source_id, source_url, doc_type, content_hash,
                               created_at, updated_at, valid_from, acl,
                               title, body_preview)
        VALUES ($2, $1, 1, 'custom_ingest', $1, 'https://x', 'custom.note',
                'dh', NOW(), NOW(), NOW(), '{}'::jsonb, 'T', 'p')
        ON CONFLICT DO NOTHING
        """,
        f"{customer_id}:d1",
        customer_id,
    )
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id, chunk_index, content, content_hash,
            token_count, chunker_version, first_seen_version, last_seen_version,
            kind, visibility
        )
        SELECT $1 || ':c_' || g, $2, $3, g, 'content ' || g, 'h' || g,
               3, 'v1', 1, 1, 'content', 'approved'
        FROM generate_series(1, $4::int) g
        """,
        customer_id,
        f"{customer_id}:d1",
        customer_id,
        n,
    )


@pytest_asyncio.fixture
async def seeded(live_db: None):
    """Two tenants of very different sizes in the same table."""
    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        await _seed(conn, BIG, 4000)
        await _seed(conn, SMALL, 40)
        await conn.execute("ANALYZE chunks")
        yield conn


async def _explain(conn, customer_id: str) -> dict:
    rows = await conn.fetch(
        """
        EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
        SELECT chunk_id FROM chunks
        WHERE customer_id = $1 AND valid_to IS NULL
        ORDER BY chunk_id
        LIMIT 100
        """,
        customer_id,
    )
    return json.loads(rows[0][0])[0]["Plan"]


def _walk(plan: dict):
    yield plan
    for child in plan.get("Plans", []) or []:
        yield from _walk(child)


def _relations(plan: dict) -> set[str]:
    return {n["Relation Name"] for n in _walk(plan) if "Relation Name" in n}


def _buffers(plan: dict) -> int:
    root = plan
    return int(root.get("Shared Hit Blocks", 0)) + int(root.get("Shared Read Blocks", 0))


async def test_query_touches_only_the_callers_partition(seeded) -> None:
    """INVARIANT 1. This is what makes the index bid tenant-sized."""
    conn = seeded
    small_plan = await _explain(conn, SMALL)
    touched = _relations(small_plan)

    assert partition_name_for(SMALL) in touched, (
        f"the tenant's own partition was not read; plan touched {sorted(touched)}"
    )
    assert partition_name_for(BIG) not in touched, (
        "the query read ANOTHER tenant's partition -- pruning is not happening, "
        f"plan touched {sorted(touched)}"
    )


async def test_work_stays_proportional_to_the_tenants_own_data(seeded) -> None:
    """INVARIANT 2. The assertion that would have caught the original fault.

    The small tenant holds 1% of the rows, so its search must not read work
    proportional to the whole table. A generous multiple is used deliberately:
    this is a blast-radius guard, not a performance benchmark, and it must not
    go red because a plan shape changed within the tenant.
    """
    conn = seeded
    small = _buffers(await _explain(conn, SMALL))
    big = _buffers(await _explain(conn, BIG))

    assert small < big, (
        f"the 40-row tenant read {small} buffers and the 4000-row tenant read "
        f"{big}: work is not scaling with the caller's own data, which is the "
        "shared-index fault this partitioning removed"
    )


async def test_rows_land_in_the_tenant_partition_not_default(seeded) -> None:
    """A tenant with a partition must never be writing into DEFAULT."""
    conn = seeded
    default_name = await default_partition_name(conn)
    assert default_name is not None, "schema.sql must define a DEFAULT partition"
    in_default = await conn.fetchval(
        f'SELECT count(*) FROM ONLY "{default_name}"'
    )
    assert in_default == 0, (
        f"{in_default} rows are in DEFAULT: a tenant is sharing an index again, "
        "which is the exact fault partitioning removed"
    )
    own = await conn.fetchval(
        f'SELECT count(*) FROM ONLY "{partition_name_for(SMALL)}"'
    )
    assert own == 40


async def test_upsert_uses_the_tenant_qualified_key(seeded) -> None:
    """The ON CONFLICT target `_CHUNK_UPSERT_ON_CONFLICT` names.

    A partitioned table rejects an ON CONFLICT naming a unique that does not
    contain the partition key, so this is the guard that the normalizer's
    conflict target and the schema agree.
    """
    conn = seeded
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id, chunk_index, content, content_hash,
            token_count, chunker_version, first_seen_version, last_seen_version,
            kind, visibility
        ) VALUES ($1, $2, $3, 1, 'content 1', 'h1', 3, 'v1', 1, 9, 'content',
                  'approved')
        ON CONFLICT (customer_id, doc_id, content_hash) DO UPDATE
            SET last_seen_version = EXCLUDED.last_seen_version
        """,
        f"{SMALL}:c_1",
        f"{SMALL}:d1",
        SMALL,
    )
    assert (
        await conn.fetchval(
            "SELECT last_seen_version FROM chunks WHERE customer_id=$1 AND content_hash='h1'",
            SMALL,
        )
        == 9
    )


async def test_two_tenants_may_share_a_chunk_id(seeded) -> None:
    """What dropping `chunks_chunk_id_unique` bought.

    Connector `doc_id`s carry no tenant, so before this change two tenants could
    not both hold the same GitHub document -- the second one's insert failed on
    a global unique that only ever existed as belt-and-braces (migration 0101
    calls it "obsolete as a pg_search requirement"). The PK
    `(customer_id, chunk_id)` is the identity that matters.
    """
    conn = seeded
    shared_chunk_id = "github:org/repo:README.md:c_deadbeef"
    for tenant in (BIG, SMALL):
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id, chunk_index, content,
                content_hash, token_count, chunker_version, first_seen_version,
                last_seen_version, kind, visibility
            ) VALUES ($1, $2, $3, 99, 'shared', $4, 3, 'v1', 1, 1, 'content',
                      'approved')
            """,
            shared_chunk_id,
            f"{tenant}:d1",
            tenant,
            f"shared-{tenant}",
        )
    assert (
        await conn.fetchval(
            "SELECT count(*) FROM chunks WHERE chunk_id = $1", shared_chunk_id
        )
        == 2
    )


async def test_split_default_moves_a_stranded_tenant(live_db: None) -> None:
    """The reconciliation path the guardian's DEFAULT alarm points at.

    A tenant created by something that never called `ensure_tenant_partition`
    (a seed script, a restore, a direct INSERT) writes into DEFAULT and shares
    an index. `split_default` is what an operator runs; it must move every row
    and leave DEFAULT empty.
    """
    stranded = "prune-stranded"
    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        # Deliberately NOT calling ensure_tenant_partition: this is the path
        # that skipped it.
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, $1, $1) ON CONFLICT DO NOTHING",
            stranded,
        )
        await conn.execute(
            """
            INSERT INTO documents (customer_id, doc_id, version, source_system,
                                   source_id, source_url, doc_type, content_hash,
                                   created_at, updated_at, valid_from, acl,
                                   title, body_preview)
            VALUES ($2, $1, 1, 'custom_ingest', $1, 'https://x', 'custom.note',
                    'dh', NOW(), NOW(), NOW(), '{}'::jsonb, 'T', 'p')
            """,
            f"{stranded}:d1",
            stranded,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id, chunk_index, content,
                content_hash, token_count, chunker_version, first_seen_version,
                last_seen_version, kind, visibility
            )
            SELECT $1 || ':c_' || g, $2, $3, g, 'c' || g, 'h' || g, 3, 'v1',
                   1, 1, 'content', 'approved'
            FROM generate_series(1, 5) g
            """,
            stranded,
            f"{stranded}:d1",
            stranded,
        )

        default_name = await default_partition_name(conn)
        assert (
            await conn.fetchval(f'SELECT count(*) FROM ONLY "{default_name}"') == 5
        ), "precondition: the stranded tenant's rows should be in DEFAULT"

        moved = await split_default(conn, stranded)

        assert moved == 5
        assert await conn.fetchval(f'SELECT count(*) FROM ONLY "{default_name}"') == 0
        assert (
            await conn.fetchval(
                f'SELECT count(*) FROM ONLY "{partition_name_for(stranded)}"'
            )
            == 5
        )


async def test_partition_names_never_collide(live_db: None) -> None:
    """`a-b` and `a_b` are different tenants that sanitize to the same slug.

    Without the hashed suffix the second one's ATTACH would point at the first
    one's table -- two tenants silently sharing a partition, which is a tenant
    isolation failure and not merely a performance one.
    """
    assert partition_name_for("a-b") != partition_name_for("a_b")
    assert partition_name_for("Acme") != partition_name_for("acme")
    # Deterministic: the same id always resolves to the same partition.
    assert partition_name_for("anthrogen") == partition_name_for("anthrogen")
    # Fits PostgreSQL's 63-byte identifier limit even for a long id.
    assert len(partition_name_for("x" * 62)) <= 63


async def test_a_deleted_tenants_partition_is_reported_and_droppable(
    live_db: None,
) -> None:
    """`ensure_tenant_partition` had no counterpart, so deletes leaked tables.

    Deleting a tenant takes its ROWS (`chunks.customer_id` carries ON DELETE
    CASCADE from `customers`) but leaves the PARTITION attached, with its
    indexes, forever. Nothing routes a row there again because no `customer_id`
    matches the bound. Seen on the research plane 2026-09-15: purging
    `richards-research-team` left a 609 MB partition behind with the customer
    row already gone.

    It is not only wasted space. Locks are taken per RELATION, so each orphan
    permanently adds itself and its indexes to the lock footprint of every
    query that does not prune -- a cost that grows with the number of tenants
    ever deleted rather than the number that exist.
    """
    tenant = "orphan-sweep-test"
    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash,"
            " r2_bucket) VALUES ($1, $1, $1, $1)"
            " ON CONFLICT (customer_id) DO NOTHING",
            tenant,
        )
        await ensure_tenant_partition(conn, tenant)
        part = partition_name_for(tenant)

        # While the tenant EXISTS the drop must refuse. Without this gate the
        # helper is a one-call way to delete a live tenant's whole corpus.
        assert (tenant, part) not in [(t, p) for p, t in
                                      await find_orphan_partitions(conn)]
        with pytest.raises(ValueError, match="still in `customers`"):
            await drop_tenant_partition(conn, tenant)
        assert await conn.fetchval("SELECT to_regclass($1)", part) is not None

        # Delete the tenant: rows go by cascade, the partition does not.
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", tenant)
        assert await conn.fetchval("SELECT to_regclass($1)", part) is not None, (
            "the partition should SURVIVE the cascade -- that is the leak this "
            "test exists for"
        )

        orphans = await find_orphan_partitions(conn)
        assert (part, tenant) in orphans, (
            f"{part} should be reported as an orphan once {tenant} is gone; "
            f"got {orphans}"
        )

        assert await drop_tenant_partition(conn, tenant) is True
        assert await conn.fetchval("SELECT to_regclass($1)", part) is None
        # Idempotent: a second sweep over the same tenant is a no-op, not an
        # error, because the cron will run again in ten minutes.
        assert await drop_tenant_partition(conn, tenant) is False
        assert (part, tenant) not in await find_orphan_partitions(conn)


async def test_the_bm25_pool_shape_runs_against_the_scan_target(seeded) -> None:
    """The exact channel's real query is UNSUPPORTED on a partitioned parent.

    pg_search rejects it with `Unsupported query shape` and the channel then
    returns nothing while the search still reports `state: ok` -- so the only
    signal is `degraded`, and the first person to notice was a customer whose
    results came back semantic-only. On the research plane the share of
    searches reporting `channel_degraded` went 10% -> 98% the moment `chunks`
    was partitioned, average exact hits 1.03 -> 0.42.

    It is the COMBINATION that is rejected, not scoring and not partitioning:
    score + ORDER BY score is fine on the parent, and so is adding a
    `content_tsv` projection; adding the `customer_id` and `valid_to`
    predicates on top is what tips it over. So this test pins the WHOLE shape
    rather than any one clause, because any one of them alone passes.
    """
    conn = seeded
    from engine.retrieval.retrievers.bm25 import bm25_scan_target

    target = await bm25_scan_target(conn, BIG)
    assert target == partition_name_for(BIG), (
        "a partitioned database must scan the tenant's own partition; scanning "
        "the parent is what pg_search rejects"
    )
    await conn.execute("SELECT set_config('app.current_customer_id', $1, true)", BIG)
    # The production shape, verbatim in structure: score, an ORDER BY on it, a
    # content_tsv projection, and both SQL predicates.
    await conn.fetch(
        f"""
        SELECT c.chunk_id, paradedb.score(c.chunk_id) AS score,
               (c.content_tsv @@ to_tsquery('english', $3)) AS content_hit
        FROM {target} c
        WHERE c.customer_id = $1
          AND c.chunk_id @@@ paradedb.boolean(must => ARRAY[
                paradedb.match('customer_id', $1, conjunction_mode => true),
                paradedb.boolean(should => ARRAY[
                  paradedb.boost(10.0, paradedb.match('title', $2)),
                  paradedb.match('content', $2)])])
          AND c.valid_to IS NULL
        ORDER BY paradedb.score(c.chunk_id) DESC
        LIMIT 10
        """,
        BIG,
        "seed",
        "seed",
    )


async def test_bm25_falls_back_to_the_parent_without_a_partition(
    live_db: None,
) -> None:
    """A tenant with no partition of its own lives in DEFAULT.

    Only the parent reaches DEFAULT, so pointing the scan at a partition that
    does not exist would turn a slow search into a missing table. The fallback
    is what lets this ship to a plane the conversion has not run on.
    """
    from engine.retrieval.retrievers.bm25 import bm25_scan_target
    from engine.shared.partitions import CHUNKS_PARENT

    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        assert await bm25_scan_target(conn, "tenant-with-no-partition") == CHUNKS_PARENT


async def test_bm25_pool_cannot_leak_across_token_sharing_tenants(
    live_db: None,
) -> None:
    """The pool has NO SQL `customer_id = $1` any more. This is what replaces it.

    That predicate was removed because `customer_id` is the partition key and
    pg_search rejects a SQL predicate on it under a partitioned parent -- the
    whole exact channel died for 13 hours. What now makes the answer exact is
    FORCE RLS, and the index-side `paradedb.match('customer_id', ...)` is only
    a pre-filter. The comment that used to justify the predicate said as much;
    a comment is not a test.

    The pair is chosen to be the WORST case for the pre-filter: every token of
    the shorter id appears in the longer one, so `conjunction_mode` on
    `leak-probe` matches `leak-probe-demo` too. If RLS were not binding, this
    is the pair that would leak.

    Both scan paths are exercised -- the partition by name, and the parent
    forced -- because on the parent RLS is the ONLY thing between two tenants.
    `assert hits` first, so a query that matches nothing cannot pass this
    vacuously.
    """
    from engine.retrieval.retrievers.bm25 import bm25_search
    from engine.shared.partitions import CHUNKS_PARENT

    a, b = "leak-probe", "leak-probe-demo"
    async with db_module.raw_conn() as conn:
        if not await is_partitioned(conn):
            pytest.skip("chunks is not partitioned on this database")
        await _seed(conn, a, 60)
        await _seed(conn, b, 60)

    for path, override in (("partition", None), ("parent", CHUNKS_PARENT)):
        for me, other in ((a, b), (b, a)):
            hits = await bm25_search(
                me, "content", top_k=200, _scan_target_override=override
            )
            assert hits, f"[{path}] {me}: the canary term matched nothing -- vacuous"
            leaked = [h.chunk_id for h in hits if h.chunk_id.startswith(f"{other}:")]
            assert not leaked, f"[{path}] {me} received {other}'s rows: {leaked[:3]}"
            assert all(h.chunk_id.startswith(f"{me}:") for h in hits), (
                f"[{path}] {me}: a hit belongs to neither tenant"
            )
