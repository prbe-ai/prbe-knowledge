"""Phase 2 search pipeline routed-entity translation.

When the router extracts an alias canonical_id, the corresponding
QueryEntityResult must land on the cluster's primary (not be dropped),
and two aliases of the same primary must collapse to one result.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from services.retrieval.router import RouterEntity
from services.retrieval.search_pipeline import _build_entity_results
from shared.db import raw_conn

pytestmark = pytest.mark.asyncio


CUSTOMER_ID = "search-cluster-cust"
PRIMARY = "richardwei6"
ALIAS_A = "mahit@prbe.ai"
ALIAS_B = "U07ABC123"


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'test', 'h-' || $1) ON CONFLICT (customer_id) DO NOTHING",
            customer_id,
        )


async def _seed_primary_node(customer_id: str) -> None:
    """Person:PRIMARY exists. Aliases were hard-deleted at merge time."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES ($1, 'Person', $2, '{"name":"Richard"}'::jsonb, 1)
            """,
            customer_id, PRIMARY,
        )


async def _seed_cluster(customer_id: str, aliases: list[str]) -> None:
    """Insert an entity_merge_audit + one entity_aliases row per alias."""
    merge_id = str(uuid.uuid4())
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO entity_merge_audit (
                merge_id, customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, $2, 'Person', $3, $4::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            """,
            merge_id, customer_id, PRIMARY, aliases,
        )
        for alias in aliases:
            await conn.execute(
                """
                INSERT INTO entity_aliases (
                    customer_id, label, alias_canonical_id,
                    primary_canonical_id, merge_id
                ) VALUES ($1, 'Person', $2, $3, $4)
                """,
                customer_id, alias, PRIMARY, merge_id,
            )


def _routed_person(canonical_id: str, *, confidence: float = 0.9) -> RouterEntity:
    return RouterEntity(
        entity_type="person",
        canonical_id=canonical_id,
        display_name=canonical_id,
        confidence=confidence,
    )


async def test_alias_input_lands_on_primary(live_db):
    """Router extracts mahit@prbe.ai; QueryEntityResult is for richardwei6."""
    await _seed_customer(CUSTOMER_ID)
    await _seed_primary_node(CUSTOMER_ID)
    await _seed_cluster(CUSTOMER_ID, aliases=[ALIAS_A])

    results = await _build_entity_results(
        customer_id=CUSTOMER_ID,
        routed_entities=[_routed_person(ALIAS_A)],
        timing={},
    )
    assert len(results) == 1
    assert results[0].canonical_id == PRIMARY


async def test_unmerged_input_unchanged(live_db):
    """Unmerged canonical_id flows through unchanged."""
    customer_id = "search-cluster-loner-cust"
    await _seed_customer(customer_id)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES ($1, 'Person', 'loner-id', '{"name":"Loner"}'::jsonb, 1)
            """,
            customer_id,
        )

    results = await _build_entity_results(
        customer_id=customer_id,
        routed_entities=[_routed_person("loner-id")],
        timing={},
    )
    assert len(results) == 1
    assert results[0].canonical_id == "loner-id"


async def test_two_aliases_of_same_primary_collapse(live_db):
    """Router extracts two aliases of the same primary; we emit ONE result."""
    await _seed_customer(CUSTOMER_ID)
    await _seed_primary_node(CUSTOMER_ID)
    await _seed_cluster(CUSTOMER_ID, aliases=[ALIAS_A, ALIAS_B])

    results = await _build_entity_results(
        customer_id=CUSTOMER_ID,
        routed_entities=[
            _routed_person(ALIAS_A, confidence=0.9),
            _routed_person(ALIAS_B, confidence=0.8),
        ],
        timing={},
    )
    assert len(results) == 1
    assert results[0].canonical_id == PRIMARY


async def test_mixed_alias_and_primary_inputs_collapse(live_db):
    """One input is an alias, another is the primary -- still one result."""
    await _seed_customer(CUSTOMER_ID)
    await _seed_primary_node(CUSTOMER_ID)
    await _seed_cluster(CUSTOMER_ID, aliases=[ALIAS_A])

    results = await _build_entity_results(
        customer_id=CUSTOMER_ID,
        routed_entities=[
            _routed_person(ALIAS_A, confidence=0.9),
            _routed_person(PRIMARY, confidence=0.8),
        ],
        timing={},
    )
    assert len(results) == 1
    assert results[0].canonical_id == PRIMARY


async def test_collapse_keeps_highest_confidence_alias(live_db):
    """Two aliases of the same primary with different confidences -
    the kept RouterEntity must be the higher-confidence one so
    downstream scoring isn't depressed by router extraction order.
    """
    await _seed_customer(CUSTOMER_ID)
    await _seed_primary_node(CUSTOMER_ID)
    await _seed_cluster(CUSTOMER_ID, aliases=[ALIAS_A, ALIAS_B])

    # Order matters: lower confidence FIRST (would be kept under first-wins).
    results = await _build_entity_results(
        customer_id=CUSTOMER_ID,
        routed_entities=[
            _routed_person(ALIAS_A, confidence=0.5),
            _routed_person(ALIAS_B, confidence=0.9),
        ],
        timing={},
    )
    assert len(results) == 1
    # Score = confidence * log1p(doc_count). doc_count is 0 here (no
    # attached docs seeded), so score collapses to `confidence * 0.5`
    # per search_pipeline.py's branch for `doc_count == 0`. Either way
    # the kept confidence is the higher one (0.9), so score > the
    # depressed-confidence (0.5) baseline.
    # We assert directly via the public attribute that the higher-
    # confidence entity won. The QueryEntityResult exposes `score`,
    # not raw confidence, so verify via score.
    assert results[0].canonical_id == PRIMARY
    # score = 0.9 * 0.5 = 0.45 (0-doc branch) vs first-wins 0.5*0.5=0.25
    assert results[0].score == pytest.approx(0.45)
