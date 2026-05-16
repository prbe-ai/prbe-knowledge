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

Deviation from plan B3/B5: the plan uses ``fastapi.testclient.TestClient``,
which spins its own event loop and clashes with the pool initialised by
the ``live_db`` fixture (asyncpg connections are loop-bound). We use the
``httpx.AsyncClient`` + ``ASGITransport`` + ``app.router.lifespan_context``
pattern already used in ``tests/test_internal_devices.py`` and
``tests/test_wiki_routes.py``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.ingestion.main import app
from shared.config import Settings, get_settings
from shared.db import close_pool, init_pool, raw_conn, with_tenant

CUSTOMER_ID = "ec-routes-cust"
USER_ID = "11111111-1111-1111-1111-111111111111"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


def _headers() -> dict[str, str]:
    return {
        "X-Internal-Knowledge-Key": "test-internal-key",
        "X-Prbe-Customer": CUSTOMER_ID,
    }


@pytest_asyncio.fixture
async def client(live_db: None, settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client bound to the app's own DB pool (via lifespan).

    Closes the ``live_db`` pool first so the app's pool — created inside
    ``app.router.lifespan_context`` — lives on the same event loop that the
    test will await against. After the test, re-init the ``live_db`` pool
    so its teardown (final TRUNCATE) succeeds.
    """
    await close_pool()
    transport = ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://t") as c,
        app.router.lifespan_context(app),
    ):
        yield c
    await init_pool(settings)


async def _seed_customer(conn: asyncpg.Connection) -> None:
    await conn.execute(
        "INSERT INTO customers(customer_id, display_name, api_key_hash) "
        "VALUES ($1, 'mig', 'mig-hash') ON CONFLICT DO NOTHING",
        CUSTOMER_ID,
    )


async def _seed_person(
    conn: asyncpg.Connection,
    canonical_id: str,
    *,
    props: dict[str, Any] | None = None,
) -> int:
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
async def test_merge_happy_path(client: httpx.AsyncClient) -> None:
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
    resp = await client.post(
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
        # COLLATE "C" pins byte-order so the assertion is locale-independent
        # (default en_US.UTF-8 sorts case-insensitively, which would flip
        # 'U07ABC123' vs 'mahit@prbe.ai').
        rows = await conn.fetch(
            """
            SELECT properties, aliased_from_canonical_id
            FROM graph_edges
            WHERE customer_id = $1 AND edge_type = 'AUTHORED'
            ORDER BY (aliased_from_canonical_id IS NULL) DESC,
                     aliased_from_canonical_id COLLATE "C"
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
            "WHERE customer_id = $1 ORDER BY alias_canonical_id COLLATE \"C\"",
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
            "WHERE customer_id = $1 ORDER BY canonical_id COLLATE \"C\"",
            CUSTOMER_ID,
        )
        assert [s["canonical_id"] for s in snaps] == ["U07ABC123", "mahit@prbe.ai"]


# ---------------------------------------------------------------------------
# Merge validation errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_merge_404_on_missing_canonical_id(client: httpx.AsyncClient) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "richardwei6")
        # 'unknown-id' does not exist.
    resp = await client.post(
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
async def test_merge_409_when_alias_already_in_cluster(client: httpx.AsyncClient) -> None:
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
    resp = await client.post(
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
async def test_merge_409_when_primary_is_already_alias(client: httpx.AsyncClient) -> None:
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
    resp = await client.post(
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


@pytest.mark.asyncio
async def test_merge_400_when_primary_in_aliases(client: httpx.AsyncClient) -> None:
    """No DB setup needed — request fails at the early body check."""
    resp = await client.post(
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


@pytest.mark.asyncio
async def test_merge_422_on_duplicate_aliases(client: httpx.AsyncClient) -> None:
    resp = await client.post(
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


@pytest.mark.asyncio
async def test_merge_401_without_internal_key(client: httpx.AsyncClient) -> None:
    resp = await client.post(
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


# ---------------------------------------------------------------------------
# Unmerge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unmerge_happy_path_restores_alias_and_edges(
    client: httpx.AsyncClient,
) -> None:
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

    merge_resp = await client.post(
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
    assert merge_resp.status_code == 200, merge_resp.text
    merge_id = merge_resp.json()["merge_id"]

    # Unmerge.
    unmerge_resp = await client.delete(
        "/api/entity-clusters/Person/richardwei6/aliases/mahit@prbe.ai",
        headers=_headers(),
    )
    assert unmerge_resp.status_code == 204, unmerge_resp.text

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
            ORDER BY gn.canonical_id COLLATE "C"
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


@pytest.mark.asyncio
async def test_unmerge_404_when_alias_not_in_cluster(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.delete(
        "/api/entity-clusters/Person/whatever/aliases/nothing",
        headers=_headers(),
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_unmerge_401_without_internal_key(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.delete(
        "/api/entity-clusters/Person/x/aliases/y",
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /api/entity-clusters — list
# ---------------------------------------------------------------------------


async def _merge(
    client: httpx.AsyncClient,
    *,
    primary: str,
    aliases: list[str],
) -> str:
    """POST a merge and return its `merge_id`. Caller seeds graph_nodes."""
    resp = await client.post(
        "/api/entity-clusters/merge",
        headers=_headers(),
        json={
            "customer_id":          CUSTOMER_ID,
            "performed_by_user_id": USER_ID,
            "label":                "Person",
            "primary_canonical_id": primary,
            "alias_canonical_ids":  aliases,
        },
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["merge_id"]


@pytest.mark.asyncio
async def test_list_empty_when_no_clusters(client: httpx.AsyncClient) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    resp = await client.get("/api/entity-clusters", headers=_headers())
    assert resp.status_code == 200, resp.text
    assert resp.json() == {"clusters": []}


@pytest.mark.asyncio
async def test_list_after_single_merge(client: httpx.AsyncClient) -> None:
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "richardwei6")
        await _seed_person(conn, "mahit@prbe.ai")
    await _merge(client, primary="richardwei6", aliases=["mahit@prbe.ai"])

    resp = await client.get("/api/entity-clusters", headers=_headers())
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert len(body["clusters"]) == 1
    cluster = body["clusters"][0]
    assert cluster["label"] == "Person"
    assert cluster["primary_canonical_id"] == "richardwei6"
    assert cluster["display_name"] is None
    assert [a["alias_canonical_id"] for a in cluster["aliases"]] == [
        "mahit@prbe.ai",
    ]
    # merge_id round-trips as a UUID string.
    uuid.UUID(cluster["aliases"][0]["merge_id"])


@pytest.mark.asyncio
async def test_list_groups_multiple_merges_into_same_primary(
    client: httpx.AsyncClient,
) -> None:
    """Two merges extending the same primary surface as ONE cluster row
    whose `aliases` list has both, each with its own merge_id."""
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "richardwei6")
        await _seed_person(conn, "mahit@prbe.ai")
        await _seed_person(conn, "U07ABC123")

    m1 = await _merge(client, primary="richardwei6", aliases=["mahit@prbe.ai"])
    m2 = await _merge(client, primary="richardwei6", aliases=["U07ABC123"])

    resp = await client.get("/api/entity-clusters", headers=_headers())
    body = resp.json()
    assert len(body["clusters"]) == 1
    aliases = body["clusters"][0]["aliases"]
    assert {a["alias_canonical_id"] for a in aliases} == {
        "mahit@prbe.ai",
        "U07ABC123",
    }
    merge_ids = {a["merge_id"] for a in aliases}
    assert merge_ids == {m1, m2}, "each alias should carry its own merge_id"


@pytest.mark.asyncio
async def test_list_hides_buried_clusters_under_chains(
    client: httpx.AsyncClient,
) -> None:
    """Chain `a → b → c`: only the outer cluster (c with alias b) shows.
    `b`'s own cluster (with alias a) is hidden until b is unmerged out
    of c."""
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "a")
        await _seed_person(conn, "b")
        await _seed_person(conn, "c")

    # Inner merge: a → b.
    await _merge(client, primary="b", aliases=["a"])
    # Outer merge: b → c. b's graph_node still exists here (a's was
    # deleted by the inner merge), so this is allowed by the current
    # server-side guard (which only blocks "alias already aliased",
    # not "primary already a primary of something else").
    await _merge(client, primary="c", aliases=["b"])

    resp = await client.get("/api/entity-clusters", headers=_headers())
    clusters = resp.json()["clusters"]
    primaries = [c["primary_canonical_id"] for c in clusters]
    assert primaries == ["c"], (
        f"expected only the outer cluster 'c' to surface; got {primaries}"
    )
    assert [a["alias_canonical_id"] for a in clusters[0]["aliases"]] == ["b"]


@pytest.mark.asyncio
async def test_list_resurfaces_inner_cluster_after_outer_unmerge(
    client: httpx.AsyncClient,
) -> None:
    """After unmerging b from c, b becomes a primary again and its inner
    cluster (b with alias a) reappears in the list."""
    async with raw_conn() as conn:
        await _seed_customer(conn)
    async with with_tenant(CUSTOMER_ID) as conn:
        await _seed_person(conn, "a")
        await _seed_person(conn, "b")
        await _seed_person(conn, "c")

    await _merge(client, primary="b", aliases=["a"])
    await _merge(client, primary="c", aliases=["b"])

    # Before unmerge: only outer cluster visible.
    before = await client.get("/api/entity-clusters", headers=_headers())
    assert [c["primary_canonical_id"] for c in before.json()["clusters"]] == [
        "c",
    ]

    # Unmerge b from c → b's node restored, b→c routing gone.
    resp = await client.delete(
        "/api/entity-clusters/Person/c/aliases/b",
        headers=_headers(),
    )
    assert resp.status_code == 204, resp.text

    # After unmerge: inner cluster (b with alias a) reappears. c has no
    # remaining aliases and drops out entirely.
    after = await client.get("/api/entity-clusters", headers=_headers())
    clusters = after.json()["clusters"]
    primaries = [c["primary_canonical_id"] for c in clusters]
    assert primaries == ["b"]
    assert [a["alias_canonical_id"] for a in clusters[0]["aliases"]] == ["a"]


@pytest.mark.asyncio
async def test_list_400_without_customer_header(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.get(
        "/api/entity-clusters",
        headers={"X-Internal-Knowledge-Key": "test-internal-key"},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_list_401_without_internal_key(
    client: httpx.AsyncClient,
) -> None:
    resp = await client.get(
        "/api/entity-clusters",
        headers={"X-Prbe-Customer": CUSTOMER_ID},
    )
    assert resp.status_code == 401
