"""Alias routing follows chains; post-write work is enqueued only when it can matter.

Real Postgres (``live_db`` truncates between tests). Every entity is synthetic.

    merge a into b, then b into c  ->  entity_aliases: a -> b, b -> c
    upsert a  -> must land on c (one hop sent it to b, whose node the second
                 merge deleted, so the upsert resurrected b as a stray copy)
"""

from __future__ import annotations

import itertools
import uuid

import pytest

from engine.ingest.entity_clusters_routes import MergeRequest, merge_cluster, unmerge_alias
from engine.ingest.graph_writer import upsert_nodes
from engine.retrieval.helpers import (
    expand_to_author_id_set,
    expand_to_cluster_members,
    resolve_aliases,
)
from engine.shared.alias_sql import ALIAS_CHAIN_MAX_DEPTH
from engine.shared.db import raw_conn, with_tenant
from engine.shared.models import make_person

pytestmark = pytest.mark.asyncio

CUSTOMER = "alias-chain-cust"
USER = uuid.UUID("11111111-1111-1111-1111-111111111111")


async def _seed_customer() -> None:
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) VALUES ($1, 'test', 'h-' || $1) "
            "ON CONFLICT (customer_id) DO NOTHING",
            CUSTOMER,
        )


async def _people(*ids: str, **props) -> dict:
    async with with_tenant(CUSTOMER) as conn:
        return await upsert_nodes(conn, CUSTOMER, [make_person(i, {"name": i, **props}) for i in ids], "github")


async def _merge(primary: str, alias: str) -> None:
    await merge_cluster(MergeRequest(customer_id=CUSTOMER, performed_by_user_id=USER, label="Person",
                                     primary_canonical_id=primary, alias_canonical_ids=[alias]))


async def _fetch(sql: str, *args):
    async with raw_conn() as conn:
        return await conn.fetch(sql, *args)


async def _person_ids() -> set[str]:
    rows = await _fetch("SELECT canonical_id FROM graph_nodes WHERE customer_id = $1 AND label = 'Person'", CUSTOMER)
    return {r["canonical_id"] for r in rows}


async def _chain() -> None:
    """a -> b -> c, built the way a human merge builds it."""
    await _seed_customer()
    await _people("a", "b", "c")
    await _merge("b", "a")
    await _merge("c", "b")


# --------------------------------------------------------------------------- #
# routing
# --------------------------------------------------------------------------- #


async def test_an_upsert_of_a_chained_alias_lands_on_the_end_of_the_chain(live_db):
    await _chain()
    assert await _person_ids() == {"c"}

    ids = await _people("a")
    (node_id,) = ids.values()
    assert await _person_ids() == {"c"}  # no stray `b` resurrected
    (row,) = await _fetch("SELECT canonical_id FROM graph_nodes WHERE node_id = $1", node_id)
    assert row["canonical_id"] == "c"


async def test_the_read_path_resolves_and_expands_through_the_chain(live_db):
    await _chain()
    async with with_tenant(CUSTOMER) as conn:
        assert await resolve_aliases(conn, CUSTOMER, [("Person", "a"), ("Person", "b"), ("Person", "c")]) == {
            ("Person", "a"): "c",
            ("Person", "b"): "c",
        }
        clusters = await expand_to_cluster_members(conn, CUSTOMER, "Person", ["a", "c"])
        authors = await expand_to_author_id_set(conn, CUSTOMER, ["a"])
    assert clusters["a"][0] == "c" and sorted(clusters["a"]) == ["a", "b", "c"]
    assert clusters["c"][0] == "c" and sorted(clusters["c"]) == ["a", "b", "c"]
    assert {"a", "b", "c"} <= set(authors)


async def test_unmerging_the_outer_link_routes_the_inner_alias_to_its_own_primary_again(live_db):
    await _chain()
    await unmerge_alias(label="Person", primary_canonical_id="c", alias_canonical_id="b", x_prbe_customer=CUSTOMER)
    async with with_tenant(CUSTOMER) as conn:
        assert await resolve_aliases(conn, CUSTOMER, [("Person", "a")]) == {("Person", "a"): "b"}
    await _people("a")
    assert await _person_ids() == {"b", "c"}


async def _raw_aliases(pairs: list[tuple[str, str]]) -> None:
    """Routing rows written by hand (merge_cluster refuses these shapes)."""
    merge_id = uuid.uuid4()
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO entity_merge_audit (merge_id, customer_id, label, primary_canonical_id, "
            "merged_alias_canonical_ids, performed_by_user_id, status) "
            "VALUES ($1, $2, 'Person', 'x', ARRAY['y']::text[], $3, 'active')",
            merge_id, CUSTOMER, USER,
        )
        await conn.executemany(
            "INSERT INTO entity_aliases (customer_id, label, alias_canonical_id, primary_canonical_id, merge_id) "
            "VALUES ($1, 'Person', $2, $3, $4)",
            [(CUSTOMER, alias, primary, merge_id) for alias, primary in pairs],
        )


async def test_a_cycle_terminates(live_db):
    await _seed_customer()
    await _raw_aliases([("x", "y"), ("y", "x")])
    async with with_tenant(CUSTOMER) as conn:
        out = await resolve_aliases(conn, CUSTOMER, [("Person", "x"), ("Person", "y")])
        clusters = await expand_to_cluster_members(conn, CUSTOMER, "Person", ["x"])
    assert out == {("Person", "x"): "y", ("Person", "y"): "x"}
    assert set(clusters["x"]) == {"x", "y"}


async def test_a_chain_longer_than_the_cap_stops_at_the_cap(live_db):
    await _seed_customer()
    hops = [f"n{i}" for i in range(ALIAS_CHAIN_MAX_DEPTH + 3)]
    await _raw_aliases(list(itertools.pairwise(hops)))
    async with with_tenant(CUSTOMER) as conn:
        out = await resolve_aliases(conn, CUSTOMER, [("Person", "n0")])
    assert out == {("Person", "n0"): f"n{ALIAS_CHAIN_MAX_DEPTH}"}


# --------------------------------------------------------------------------- #
# enqueue only on insert, change, or a waiting pending edge
# --------------------------------------------------------------------------- #


async def _queued() -> set[int]:
    rows = await _fetch("SELECT node_id FROM node_post_write_queue WHERE customer_id = $1", CUSTOMER)
    return {r["node_id"] for r in rows}


async def _drain_queue() -> None:
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM node_post_write_queue WHERE customer_id = $1", CUSTOMER)


async def test_an_insert_is_enqueued(live_db):
    await _seed_customer()
    ids = await _people("ada")
    assert await _queued() == set(ids.values())


async def test_a_no_op_re_upsert_is_not_enqueued(live_db):
    await _seed_customer()
    await _people("ada", email="ada@example.com")
    await _drain_queue()
    await _people("ada", email="ada@example.com")  # same properties again
    assert await _queued() == set()


async def test_a_changed_property_is_enqueued(live_db):
    await _seed_customer()
    ids = await _people("ada", email="ada@example.com")
    await _drain_queue()
    await _people("ada", email="ada@example.com", title="CTO")
    assert await _queued() == set(ids.values())


async def test_a_no_op_re_upsert_with_a_parked_edge_waiting_is_enqueued(live_db):
    # An edge can park concurrently with the node's insert and miss that
    # insert's drain; the next touch of the node must still drain it.
    await _seed_customer()
    ids = await _people("ada", email="ada@example.com")
    await _drain_queue()
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id, edge_type, "
            "from_label, from_canonical_id, to_label, to_canonical_id, source_system) "
            "VALUES ($1, 'Person', 'ada', 'AUTHORED', 'Person', 'ada', 'Document', 'doc-1', 'github')",
            CUSTOMER,
        )
    await _people("ada", email="ada@example.com")
    assert await _queued() == set(ids.values())


async def test_only_the_changed_node_of_a_batch_is_enqueued_and_every_id_is_returned(live_db):
    await _seed_customer()
    first = await _people("ada", "grace", email="x@example.com")
    await _drain_queue()
    async with with_tenant(CUSTOMER) as conn:
        again = await upsert_nodes(conn, CUSTOMER, [
            make_person("ada", {"name": "ada", "email": "x@example.com"}),
            make_person("grace", {"name": "grace", "email": "x@example.com", "title": "RADM"}),
        ], "github")
    assert again == first  # ids for every node, as before
    assert await _queued() == {first[("Person", "grace")]}
