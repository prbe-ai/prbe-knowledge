"""End-to-end test for /graph/explore anchor-mode alias translation.

When a user types an alias canonical_id and that alias has been merged,
the endpoint must translate to the primary before resolving the anchor
and BFS — otherwise the response is 404 (alias node was hard-deleted at
merge time).

Uses httpx.AsyncClient + ASGITransport against the in-process app, the
same pattern as ``tests/test_entity_clusters_routes.py`` (TestClient
clashes with the live_db pool).

Auth: X-Internal-Knowledge-Key + X-Prbe-Customer (the internal trust
boundary path that authenticate_query supports — see services/retrieval/auth.py).
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.retrieval.main import app
from shared.config import get_settings
from shared.db import raw_conn

pytestmark = pytest.mark.asyncio

CUSTOMER_ID = "graph-anchor-alias-cust"
PRIMARY = "richardwei6"
ALIAS = "mahit@prbe.ai"


@pytest.fixture(autouse=True)
def _patch_internal_key(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", "test-internal-key")
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _seed_cluster(customer_id: str) -> None:
    """Seed: customer + Person:PRIMARY graph_node (alias was hard-deleted
    at merge time) + entity_aliases row routing ALIAS to PRIMARY + a
    Repo:r1 node + TOUCHES edge so anchor_graph_query returns a 1-hop graph."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES
              ($1, 'Person', $2, '{"name":"Richard"}'::jsonb, 1),
              ($1, 'Repo',   'r1',  '{"name":"r1"}'::jsonb,    1)
            """,
            customer_id, PRIMARY,
        )
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type,
                from_node_id, to_node_id,
                confidence, properties
            )
            SELECT $1, 'TOUCHES',
                   p.node_id, r.node_id,
                   'EXTRACTED', '{}'::jsonb
            FROM graph_nodes p, graph_nodes r
            WHERE p.customer_id = $1 AND p.label = 'Person' AND p.canonical_id = $2
              AND r.customer_id = $1 AND r.label = 'Repo'   AND r.canonical_id = 'r1'
            """,
            customer_id, PRIMARY,
        )
        merge_id = str(uuid.uuid4())
        await conn.execute(
            """
            INSERT INTO entity_merge_audit (
                merge_id, customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, $2, 'Person', $3, ARRAY[$4]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            """,
            merge_id, customer_id, PRIMARY, ALIAS,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            customer_id, ALIAS, PRIMARY, merge_id,
        )


@pytest_asyncio.fixture
async def client(live_db) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as c:
            yield c


def _headers(customer_id: str) -> dict[str, str]:
    return {
        "X-Internal-Knowledge-Key": "test-internal-key",
        "X-Prbe-Customer": customer_id,
    }


async def test_anchor_alias_resolves_to_primary_graph(client):
    """Typing the alias resolves to the primary's 1-hop graph (not 404)."""
    await _seed_cluster(CUSTOMER_ID)
    resp = await client.post(
        "/graph/explore",
        headers=_headers(CUSTOMER_ID),
        json={"mode": "anchor", "anchor_node_id": ALIAS},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    node_ids = {n["id"] for n in body["nodes"]}
    # The primary's canonical_id is in the graph (the alias node was
    # hard-deleted at merge time so it's not).
    assert PRIMARY in node_ids
    assert ALIAS not in node_ids
    # 1-hop edge to Repo:r1 is present.
    assert "r1" in node_ids


async def test_anchor_unknown_returns_404(client):
    """An anchor that is neither a node nor an alias remains 404."""
    await _seed_cluster(CUSTOMER_ID)
    resp = await client.post(
        "/graph/explore",
        headers=_headers(CUSTOMER_ID),
        json={"mode": "anchor", "anchor_node_id": "nobody-here"},
    )
    assert resp.status_code == 404


async def test_anchor_primary_id_unchanged(client):
    """Typing the primary's canonical_id directly works without translation."""
    await _seed_cluster(CUSTOMER_ID)
    resp = await client.post(
        "/graph/explore",
        headers=_headers(CUSTOMER_ID),
        json={"mode": "anchor", "anchor_node_id": PRIMARY},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    node_ids = {n["id"] for n in body["nodes"]}
    assert PRIMARY in node_ids
