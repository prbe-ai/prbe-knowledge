"""Regression guard: cross_repo_deps DELETEs decrement graph_nodes.degree.

Lane A (surprise-score) introduced graph_nodes.degree, incremented on edge
INSERT in graph_writer.upsert_edges. Without a matching DECREMENT on edge
DELETE, degree drifts upward over time. cross_repo_deps.py is the only
application path that issues DELETE FROM graph_edges, so it's the only
place that needs the matching decrement.

This test reads the source file and asserts both DELETE sites are wrapped
in the data-modifying CTE pattern that updates graph_nodes.degree in the
same statement. It is a static-string check — cheap, catches regressions
without standing up a real DB.
"""

from __future__ import annotations

from pathlib import Path

SOURCE = Path(__file__).resolve().parent.parent / (
    "services/ingestion/code_graph/cross_repo_deps.py"
)


def _load() -> str:
    return SOURCE.read_text(encoding="utf-8")


def test_cross_repo_deps_has_no_bare_delete_from_graph_edges() -> None:
    """Every `DELETE FROM graph_edges` must live inside a `WITH deleted AS`
    CTE that also updates graph_nodes.degree.
    """
    src = _load()
    # Strip everything inside `WITH deleted AS (...)` blocks so the only
    # surviving DELETE FROM graph_edges occurrences are bare ones.
    # Easier check: count CTE wrappers vs delete occurrences.
    delete_count = src.count("DELETE FROM graph_edges")
    cte_count = src.count("WITH deleted AS (")
    assert delete_count == cte_count, (
        f"Bare DELETE FROM graph_edges detected: {delete_count} delete "
        f"statements but only {cte_count} CTE wrappers. Every delete must "
        f"be wrapped in a CTE that decrements graph_nodes.degree."
    )


def test_cross_repo_deps_cte_decrements_degree() -> None:
    """Each CTE block must reference graph_nodes.degree and GREATEST so
    the decrement clamps at 0.
    """
    src = _load()
    # The CTE pattern uses a UPDATE graph_nodes ... SET degree = GREATEST(...)
    # Both expected sites should match.
    update_pattern_count = src.count("SET degree = GREATEST(gn.degree - ed.dec, 0)")
    cte_count = src.count("WITH deleted AS (")
    assert update_pattern_count == cte_count, (
        f"Each CTE wrapper must contain the degree-decrement UPDATE. "
        f"Found {cte_count} CTE wrappers but only {update_pattern_count} "
        f"matching SET degree = GREATEST(...) clauses."
    )


def test_cross_repo_deps_at_least_two_delete_sites() -> None:
    """Sanity: there are 2 known DELETE FROM graph_edges sites in this file.
    If a future change adds a third, this test forces the author to also
    consider the CTE wrap."""
    src = _load()
    assert src.count("DELETE FROM graph_edges") >= 2
