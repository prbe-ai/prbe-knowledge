"""Integration tests for services/retrieval/graph_explore.graph_search_query().

Lightweight typeahead used by the /graph/explore anchor picker. Prefix-only
match (no leading wildcard) so the lower-functional indexes survive.
"""

from __future__ import annotations

import json

import pytest

from services.retrieval.graph_explore import graph_search_query
from shared.constants import GRAPH_SEARCH_MAX_LIMIT, NodeLabel
from shared.db import raw_conn

pytestmark = pytest.mark.asyncio


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )


async def _seed_node(
    customer_id: str,
    *,
    label: str,
    canonical_id: str,
    name: str | None = None,
    title: str | None = None,
    source_system: str | None = None,
    degree: int = 0,
) -> None:
    properties: dict[str, str] = {}
    if name is not None:
        properties["name"] = name
    if title is not None:
        properties["title"] = title
    if source_system is not None:
        properties["source_system"] = source_system
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (
                customer_id, label, canonical_id, properties, degree
            )
            VALUES ($1, $2, $3, $4::jsonb, $5)
            ON CONFLICT (customer_id, label, canonical_id)
            DO UPDATE SET properties = EXCLUDED.properties,
                          degree = EXCLUDED.degree
            """,
            customer_id, label, canonical_id, json.dumps(properties), degree,
        )


async def test_empty_query_returns_empty(live_db) -> None:
    """An empty `q` short-circuits to [] without a SQL call."""
    await _seed_customer("cust-search-empty")
    assert await graph_search_query(customer_id="cust-search-empty", q="", limit=10) == []
    assert await graph_search_query(customer_id="cust-search-empty", q="   ", limit=10) == []


async def test_match_by_canonical_id_prefix(live_db) -> None:
    await _seed_customer("cust-cid")
    await _seed_node("cust-cid", label=NodeLabel.SERVICE.value,
                     canonical_id="prbe-backend", degree=10)
    await _seed_node("cust-cid", label=NodeLabel.SERVICE.value,
                     canonical_id="prbe-knowledge", degree=5)
    await _seed_node("cust-cid", label=NodeLabel.SERVICE.value,
                     canonical_id="other-svc", degree=20)

    hits = await graph_search_query(customer_id="cust-cid", q="prbe", limit=10)
    ids = {h.id for h in hits}
    assert ids == {"prbe-backend", "prbe-knowledge"}


async def test_match_by_properties_name_prefix(live_db) -> None:
    await _seed_customer("cust-name")
    await _seed_node("cust-name", label=NodeLabel.PERSON.value,
                     canonical_id="user-xyz", name="Richard Wei", degree=10)
    hits = await graph_search_query(customer_id="cust-name", q="richard", limit=10)
    assert len(hits) == 1
    assert hits[0].id == "user-xyz"


async def test_match_is_case_insensitive(live_db) -> None:
    await _seed_customer("cust-case")
    await _seed_node("cust-case", label=NodeLabel.SERVICE.value,
                     canonical_id="PRBE-Backend", degree=10)
    hits = await graph_search_query(customer_id="cust-case", q="PRBE", limit=10)
    assert len(hits) == 1
    hits_lower = await graph_search_query(customer_id="cust-case", q="prbe", limit=10)
    assert len(hits_lower) == 1


async def test_ordering_by_degree_desc(live_db) -> None:
    await _seed_customer("cust-order")
    await _seed_node("cust-order", label=NodeLabel.SERVICE.value,
                     canonical_id="prbe-a", degree=1)
    await _seed_node("cust-order", label=NodeLabel.SERVICE.value,
                     canonical_id="prbe-b", degree=100)
    await _seed_node("cust-order", label=NodeLabel.SERVICE.value,
                     canonical_id="prbe-c", degree=50)
    hits = await graph_search_query(customer_id="cust-order", q="prbe", limit=10)
    assert [h.id for h in hits] == ["prbe-b", "prbe-c", "prbe-a"]


async def test_limit_capped_at_max(live_db) -> None:
    """A request with limit > GRAPH_SEARCH_MAX_LIMIT is clamped down to the
    max -- the SQL never asks for more than that.
    """
    await _seed_customer("cust-limit")
    # Seed 30 matches; only GRAPH_SEARCH_MAX_LIMIT can come back.
    for i in range(30):
        await _seed_node(
            "cust-limit", label=NodeLabel.SERVICE.value,
            canonical_id=f"prbe-svc-{i:02d}", degree=i,
        )
    hits = await graph_search_query(
        customer_id="cust-limit", q="prbe-svc", limit=GRAPH_SEARCH_MAX_LIMIT + 50,
    )
    assert len(hits) == GRAPH_SEARCH_MAX_LIMIT


async def test_cross_tenant_isolation(live_db) -> None:
    """A typeahead query for tenant B must not surface tenant A's nodes
    even when the canonical_id prefix matches.
    """
    await _seed_customer("cust-A")
    await _seed_customer("cust-B")
    await _seed_node("cust-A", label=NodeLabel.SERVICE.value,
                     canonical_id="prbe-secret", degree=10)
    hits_b = await graph_search_query(customer_id="cust-B", q="prbe", limit=10)
    assert hits_b == []
