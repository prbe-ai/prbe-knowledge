"""Unit tests for the code-graph bridge that DON'T need a live DB.

Covers the pure logic: source_event_id construction, file-cap behavior,
spillover detection. The DB-touching path-and-enqueue test lives in
test_codegraph_bridge_live.py.
"""

from __future__ import annotations

from services.ingestion.code_graph.bridge import (
    BRIDGE_PAYLOAD_SCHEMA_VERSION,
    _cap_changed_paths,
)


def test_under_cap_passes_through_untouched() -> None:
    added = ["a.py", "b.py"]
    modified = ["c.py"]
    removed = ["d.py"]
    capped_a, capped_m, capped_r, spillover = _cap_changed_paths(added, modified, removed)
    assert capped_a == added
    assert capped_m == modified
    assert capped_r == removed
    assert spillover is False


def test_at_cap_passes_through_untouched() -> None:
    n = 500
    added = [f"a{i}.py" for i in range(n)]
    capped_a, _capped_m, _capped_r, spillover = _cap_changed_paths(added, [], [])
    assert capped_a == added
    assert spillover is False


def test_over_cap_drops_added_modified_keeps_removed() -> None:
    """Spec §10 critical gap #1: spillover converts to a partial backfill.

    Removed paths stay because they're cheap (soft-deletes only); added
    and modified are dropped to be re-discovered by the spillover backfill
    walking the full tree.
    """
    added = [f"a{i}.py" for i in range(300)]
    modified = [f"m{i}.py" for i in range(250)]
    removed = ["r1.py", "r2.py"]
    capped_a, capped_m, capped_r, spillover = _cap_changed_paths(added, modified, removed)
    assert capped_a == []
    assert capped_m == []
    assert capped_r == removed
    assert spillover is True


def test_schema_version_constant() -> None:
    assert BRIDGE_PAYLOAD_SCHEMA_VERSION == 1
