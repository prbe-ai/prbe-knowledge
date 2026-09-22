"""Retiring the `Experiment` graph nodes (experiments-as-subprojects T6).

research-os now names an experiment's node `Project` / `project:<uuid>` (same
uuid). The script folds each old node into that one when it exists and
relabels it in place when it does not -- never deletes it outright, because
the experiment-anchored artifact documents are never re-pushed and would lose
their parent edge for good.
"""

from __future__ import annotations

import pytest

from engine.shared import db as db_module
from scripts.retire_experiment_nodes import mapped_canonical_id, retire_customer

CID = "test-cust-retire-experiment"


async def _seed() -> dict[str, int]:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, status) "
            "VALUES ($1, 'Retire', 'hash-retire', 'active')",
            CID,
        )

        async def node(label: str, canonical_id: str, degree: int) -> int:
            return await conn.fetchval(
                "INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree) "
                "VALUES ($1, $2, $3, jsonb_build_object('name', $3::text), $4) RETURNING node_id",
                CID,
                label,
                canonical_id,
                degree,
            )

        ids = {
            # A: the experiment's project document has been re-pushed -> FOLD.
            "exp_a": await node("Experiment", "experiment:a", 3),
            "proj_a": await node("Project", "project:a", 1),
            # B: not re-pushed yet -> RELABEL in place.
            "exp_b": await node("Experiment", "experiment:b", 1),
            # An id this script does not understand -> left alone, reported.
            "exp_odd": await node("Experiment", "legacy-7", 0),
            "parent": await node("Project", "project:parent", 1),
            "run": await node("Run", "run:r1", 2),
            "run2": await node("Run", "run:r2", 0),
            "file_doc": await node("Document", "custom_ingest:x:file:f1", 1),
            "b_doc": await node("Document", "custom_ingest:x:file:f2", 1),
        }

        async def edge(edge_type: str, a: str, b: str) -> None:
            await conn.execute(
                "INSERT INTO graph_edges (customer_id, edge_type, from_node_id, to_node_id) "
                "VALUES ($1, $2, $3, $4)",
                CID,
                edge_type,
                ids[a],
                ids[b],
            )

        # The file edge only the old node has: it must SURVIVE, re-pointed.
        await edge("TOUCHES", "exp_a", "file_doc")
        # The run's edge exists on BOTH nodes (research-os re-pushed the run):
        # re-pointed, it would collide, so the old one is dropped.
        await edge("MEMBER_OF", "run", "exp_a")
        await edge("MEMBER_OF", "run", "proj_a")
        await edge("MEMBER_OF", "exp_a", "parent")
        await edge("TOUCHES", "exp_b", "b_doc")
        await conn.execute(
            "INSERT INTO graph_node_provenance (node_id, customer_id, source_system) "
            "VALUES ($1, $2, 'custom_ingest')",
            ids["exp_a"],
            CID,
        )
        await conn.execute(
            "INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id, "
            "edge_type, from_label, from_canonical_id, to_label, to_canonical_id, source_system) "
            "VALUES ($1, 'Experiment', 'experiment:c', 'MEMBER_OF', 'Experiment', 'legacy-7', "
            "'Experiment', 'experiment:c', 'custom_ingest')",
            CID,
        )
        # Waiting on experiment A, whose Project node has ALREADY landed -- and
        # stamped with a lease a failed drain left behind. Nothing would ever
        # claim it again unless the script clears the lease and drains it.
        await conn.execute(
            "INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id, "
            "edge_type, from_label, from_canonical_id, to_label, to_canonical_id, source_system, "
            "locked_until) "
            "VALUES ($1, 'Experiment', 'experiment:a', 'MEMBER_OF', 'Run', 'run:r2', "
            "'Experiment', 'experiment:a', 'custom_ingest', now() + interval '1 hour')",
            CID,
        )
    return ids


async def _nodes() -> dict[int, tuple[str, str, int]]:
    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT node_id, label, canonical_id, degree FROM graph_nodes WHERE customer_id = $1",
            CID,
        )
    return {r["node_id"]: (r["label"], r["canonical_id"], r["degree"]) for r in rows}


async def _edges() -> set[tuple[str, int, int]]:
    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            "SELECT edge_type, from_node_id, to_node_id FROM graph_edges WHERE customer_id = $1",
            CID,
        )
    return {(r["edge_type"], r["from_node_id"], r["to_node_id"]) for r in rows}


def test_only_the_uuid_form_is_mapped() -> None:
    assert mapped_canonical_id("experiment:0f9a") == "project:0f9a"
    assert mapped_canonical_id("legacy-7") is None


async def test_retire_folds_relabels_and_is_idempotent(live_db: None) -> None:
    ids = await _seed()
    before_nodes, before_edges = await _nodes(), await _edges()

    dry = await retire_customer(CID, dry_run=True)
    assert (dry.nodes, dry.merged, dry.relabelled, dry.pending_rewritten) == (3, 1, 1, 2)
    assert dry.skipped_ids == ["legacy-7"]
    assert await _nodes() == before_nodes and await _edges() == before_edges, "dry run wrote"

    stats = await retire_customer(CID)
    assert (stats.merged, stats.relabelled, stats.duplicate_edges_dropped) == (1, 1, 1)
    assert stats.edges_repointed == 2  # the file edge and the parent edge
    assert stats.pending_rewritten == 2
    assert stats.pending_drained == 1  # the row whose Project node had landed

    nodes = await _nodes()
    # A was FOLDED: gone, and its project node carries everything it had.
    assert ids["exp_a"] not in nodes
    assert await _edges() == {
        ("TOUCHES", ids["proj_a"], ids["file_doc"]),
        ("MEMBER_OF", ids["run"], ids["proj_a"]),
        ("MEMBER_OF", ids["proj_a"], ids["parent"]),
        ("TOUCHES", ids["exp_b"], ids["b_doc"]),
        # Drained from pending_edges onto the project node that had landed.
        ("MEMBER_OF", ids["run2"], ids["proj_a"]),
    }
    # Degrees RECOUNTED from the rows: the project node gained two re-pointed
    # edges (plus the drained one, which graph_writer counts), and the run
    # lost its duplicate.
    assert nodes[ids["proj_a"]] == ("Project", "project:a", 4)
    assert nodes[ids["run"]][2] == 1
    # B was RELABELLED in place: same node, same edge, new name.
    assert nodes[ids["exp_b"]] == ("Project", "project:b", 1)
    # The id this script does not understand is untouched.
    assert nodes[ids["exp_odd"]][:2] == ("Experiment", "legacy-7")

    async with db_module.raw_conn() as conn:
        provenance = await conn.fetchval(
            "SELECT count(*) FROM graph_node_provenance WHERE node_id = $1", ids["proj_a"]
        )
        pending = await conn.fetch(
            "SELECT missing_label, missing_canonical_id, from_label, from_canonical_id, "
            "to_label, to_canonical_id, locked_until FROM pending_edges WHERE customer_id = $1",
            CID,
        )
    assert provenance == 1
    # Only the row still waiting (project:c has not landed) remains, rewritten
    # per pair: the odd `legacy-7` endpoint is NOT mangled into `project:`.
    assert [dict(r) for r in pending] == [
        {
            "missing_label": "Project",
            "missing_canonical_id": "project:c",
            "from_label": "Experiment",
            "from_canonical_id": "legacy-7",
            "to_label": "Project",
            "to_canonical_id": "project:c",
            "locked_until": None,
        }
    ]

    again = await retire_customer(CID)
    assert (again.merged, again.relabelled, again.pending_rewritten) == (0, 0, 0)
    assert again.skipped_ids == ["legacy-7"]


@pytest.mark.parametrize("dry_run", [True, False])
async def test_a_tenant_with_no_experiment_nodes_is_a_no_op(live_db: None, dry_run: bool) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash, status) "
            "VALUES ($1, 'Empty', 'hash-empty', 'active')",
            CID,
        )
    stats = await retire_customer(CID, dry_run=dry_run)
    assert (stats.nodes, stats.merged, stats.relabelled, stats.pending_rewritten) == (0, 0, 0, 0)
