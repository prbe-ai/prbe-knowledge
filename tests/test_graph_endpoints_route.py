"""Route-level tests for POST /graph/explore and POST /graph/search.

Mirrors the request shape in tests/test_query_auth.py (ASGITransport +
httpx.AsyncClient against the FastAPI app). Validates the wire contract
the prbe-backend BFF and prbe-dashboard frontend code against:

  - 422 on bad enum values (edge_type, confidence, source_system)
  - 422 on missing anchor_node_id when mode='anchor'
  - 404 on anchor_node_id not present in the tenant
  - 200 with expected shape for default and anchor modes
  - 200 from /graph/search with matches array

Plus regression coverage that adding the new routes did NOT break the
existing /retrieve and /query routes (they still register and respond).
"""

from __future__ import annotations

import hashlib
import json
import secrets

import httpx
import pytest
from httpx import ASGITransport

from engine.shared.config import Settings, get_settings
from engine.shared.constants import EdgeType, NodeLabel
from engine.shared.db import close_pool, init_pool, raw_conn
from engine.shared.embeddings import reset_embedder
from engine.shared.storage import reset_store

INTERNAL_KEY = "test-internal-knowledge-key"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv(
        "TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value()
    )
    monkeypatch.setenv("ENVIRONMENT", "local")
    monkeypatch.setenv("INTERNAL_KNOWLEDGE_API_KEY", INTERNAL_KEY)
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _seed_customer_with_key(customer_id: str) -> str:
    """Insert a customer + return the plaintext api_key for Bearer auth."""
    api_key = secrets.token_urlsafe(32)
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, $2, $3)
            ON CONFLICT (customer_id) DO UPDATE
            SET api_key_hash = EXCLUDED.api_key_hash
            """,
            customer_id,
            f"{customer_id} display",
            api_key_hash,
        )
    return api_key


async def _seed_node(
    customer_id: str,
    *,
    label: str,
    canonical_id: str,
    degree: int = 0,
    name: str | None = None,
) -> None:
    properties: dict[str, str] = {}
    if name is not None:
        properties["name"] = name
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (
                customer_id, label, canonical_id, properties, degree
            )
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (customer_id, label, canonical_id)
            DO UPDATE SET degree = EXCLUDED.degree,
                          properties = EXCLUDED.properties
            """,
            customer_id, label, canonical_id, json.dumps(properties), degree,
        )


async def _seed_edge(
    customer_id: str,
    *,
    from_label: str, from_canonical_id: str,
    to_label: str, to_canonical_id: str,
    edge_type: str = EdgeType.DISCUSSES.value,
    confidence: str = "EXTRACTED",
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                confidence, valid_from
            )
            SELECT $1, $2, f.node_id, t.node_id, $7, NOW()
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = $3 AND f.canonical_id = $4
              AND t.customer_id = $1 AND t.label = $5 AND t.canonical_id = $6
            ON CONFLICT DO NOTHING
            """,
            customer_id, edge_type,
            from_label, from_canonical_id,
            to_label, to_canonical_id,
            confidence,
        )


async def _post(
    path: str,
    body: dict,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Spin up an ASGI client + lifespan context, post once, return response.

    The lifespan context init/closes the shared pool, so live_db's truncate
    teardown would fail with "pool is closed" without a re-init afterward.
    Mirrors the close_pool/init_pool dance in tests/test_query_auth.py.
    """
    from engine.retrieval.main import app as retrieval_app

    await close_pool()
    transport = ASGITransport(app=retrieval_app)
    try:
        async with (
            httpx.AsyncClient(transport=transport, base_url="http://t") as client,
            retrieval_app.router.lifespan_context(retrieval_app),
        ):
            return await client.post(path, json=body, headers=headers or {})
    finally:
        # Re-init unconditionally so the live_db fixture's truncate at
        # teardown finds a live pool. Without this, the next test's
        # live_db.acquire() call sees a closed pool and the failure
        # surfaces in the *next* test (which makes the actual cause
        # invisible).
        from engine.shared.config import get_settings as _get_settings
        await init_pool(_get_settings())


# ---- /graph/explore happy paths ------------------------------------------


@pytest.mark.asyncio
async def test_graph_explore_default_mode_shape(live_db, settings) -> None:
    customer_id = "cust-explore-default"
    api_key = await _seed_customer_with_key(customer_id)
    await _seed_node(customer_id, label=NodeLabel.SERVICE.value,
                     canonical_id="svc-1", degree=10, name="Service One")
    await _seed_node(customer_id, label=NodeLabel.SERVICE.value,
                     canonical_id="svc-2", degree=5, name="Service Two")
    await _seed_edge(customer_id,
                     from_label=NodeLabel.SERVICE.value, from_canonical_id="svc-1",
                     to_label=NodeLabel.SERVICE.value,   to_canonical_id="svc-2")

    resp = await _post(
        "/graph/explore",
        body={"mode": "default"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "nodes" in data
    assert "edges" in data
    assert "truncated" in data
    assert "total_nodes_available" in data
    assert "total_edges_available" in data
    assert {n["id"] for n in data["nodes"]} == {"svc-1", "svc-2"}
    assert len(data["edges"]) == 1
    edge = data["edges"][0]
    # Vendor-neutral field names: source/target (not from_node_id/to_node_id).
    assert edge["source"] == "svc-1"
    assert edge["target"] == "svc-2"
    assert edge["edge_type"] == EdgeType.DISCUSSES.value


@pytest.mark.asyncio
async def test_graph_explore_anchor_mode_shape(live_db, settings) -> None:
    customer_id = "cust-explore-anchor"
    api_key = await _seed_customer_with_key(customer_id)
    await _seed_node(customer_id, label=NodeLabel.SERVICE.value,
                     canonical_id="anchor", degree=0)
    await _seed_node(customer_id, label=NodeLabel.SERVICE.value,
                     canonical_id="neighbor", degree=0)
    await _seed_edge(customer_id,
                     from_label=NodeLabel.SERVICE.value, from_canonical_id="anchor",
                     to_label=NodeLabel.SERVICE.value,   to_canonical_id="neighbor")

    resp = await _post(
        "/graph/explore",
        body={"mode": "anchor", "anchor_node_id": "anchor"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)

    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert {n["id"] for n in data["nodes"]} == {"anchor", "neighbor"}


# ---- /graph/explore validation -------------------------------------------


@pytest.mark.asyncio
async def test_graph_explore_422_bad_edge_type(live_db, settings) -> None:
    customer_id = "cust-explore-bad-edge"
    api_key = await _seed_customer_with_key(customer_id)
    resp = await _post(
        "/graph/explore",
        body={
            "mode": "default",
            "filters": {"edge_types": ["NOT_A_REAL_EDGE_TYPE"]},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_graph_explore_422_bad_confidence(live_db, settings) -> None:
    customer_id = "cust-explore-bad-conf"
    api_key = await _seed_customer_with_key(customer_id)
    resp = await _post(
        "/graph/explore",
        body={
            "mode": "default",
            "filters": {"confidences": ["TOTALLY_FAKE"]},
        },
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_graph_explore_422_missing_anchor_in_anchor_mode(live_db, settings) -> None:
    customer_id = "cust-explore-no-anchor"
    api_key = await _seed_customer_with_key(customer_id)
    resp = await _post(
        "/graph/explore",
        body={"mode": "anchor"},  # missing anchor_node_id
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 422, resp.text


@pytest.mark.asyncio
async def test_graph_explore_404_anchor_not_found(live_db, settings) -> None:
    """Anchor mode with a canonical_id that doesn't exist in this tenant
    returns 404 (not 200 with empty nodes/edges)."""
    customer_id = "cust-explore-404"
    api_key = await _seed_customer_with_key(customer_id)
    resp = await _post(
        "/graph/explore",
        body={"mode": "anchor", "anchor_node_id": "does-not-exist"},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 404, resp.text


# ---- /graph/search happy path --------------------------------------------


@pytest.mark.asyncio
async def test_graph_search_returns_matches(live_db, settings) -> None:
    customer_id = "cust-graph-search"
    api_key = await _seed_customer_with_key(customer_id)
    await _seed_node(customer_id, label=NodeLabel.SERVICE.value,
                     canonical_id="prbe-backend", degree=20)
    await _seed_node(customer_id, label=NodeLabel.SERVICE.value,
                     canonical_id="other", degree=5)

    resp = await _post(
        "/graph/search",
        body={"q": "prbe", "limit": 5},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "matches" in data
    ids = {m["id"] for m in data["matches"]}
    assert ids == {"prbe-backend"}


# ---- Regression: existing routes still work ------------------------------


@pytest.mark.asyncio
async def test_retrieve_still_responds_after_graph_routes_added(live_db, settings) -> None:
    """Adding the new /graph/* routes must not break /retrieve. We don't
    care about the result contents -- a 200 status proves the route is
    still registered and the dependency injection chain is intact.
    """
    customer_id = "cust-retrieve-regression"
    api_key = await _seed_customer_with_key(customer_id)
    resp = await _post(
        "/retrieve",
        body={"query": "anything", "top_k": 1},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 200, resp.text
    assert "results" in resp.json()


@pytest.mark.asyncio
async def test_query_route_still_registered(live_db, settings) -> None:
    """Adding /graph/* routes must not break /query. We send an empty
    body and assert 422 (validation error) -- this proves the POST
    handler is still mounted; a missing route would return 404.

    We deliberately do NOT exercise the synthesis pipeline (would need
    LLM stubbing). The route-existence signal is what we care about.
    """
    customer_id = "cust-query-route-regression"
    api_key = await _seed_customer_with_key(customer_id)
    resp = await _post(
        "/query",
        body={},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    # 422 = body validation failed (route exists, parsed). 404 would mean
    # the route registration broke when /graph/* was added.
    assert resp.status_code == 422, (
        f"Expected 422 (validation error from empty body, proves route "
        f"is registered); got {resp.status_code}. Body: {resp.text!r}"
    )


@pytest.mark.asyncio
async def test_query_stream_route_still_registered(live_db, settings) -> None:
    """Same as test_query_route_still_registered but for /query/stream."""
    customer_id = "cust-query-stream-route-regression"
    api_key = await _seed_customer_with_key(customer_id)
    resp = await _post(
        "/query/stream",
        body={},
        headers={"Authorization": f"Bearer {api_key}"},
    )
    await init_pool(settings)
    assert resp.status_code == 422, (
        f"Expected 422 (validation error from empty body, proves route "
        f"is registered); got {resp.status_code}. Body: {resp.text!r}"
    )
