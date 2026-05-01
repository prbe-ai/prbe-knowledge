"""Tests for the priority edge walk (spec §6 step 4).

The walk is pure Python — no DB, no async, no mocks. The only knobs are
priority filtering and the budget threshold. The four tests below pin:

1. P1-before-P2 ordering regardless of input order.
2. P3 dropped under tight budget.
3. P2 included when budget is non-zero.
4. P3 included when budget clears the 5000-token threshold.

The 5000 threshold is a tunable; test (4) exercises ``5001`` (one past
the boundary) so a future tweak from 5000 to e.g. 6000 fails this test
deliberately rather than silently shifting behavior.
"""

from __future__ import annotations

from services.kg.schema import (
    ContextSource,
    Evidence,
    Frontmatter,
    Related,
    Signature,
)
from services.kg.traversal.edge_walk import walk_priority_edges


def _fm(sources: list[ContextSource]) -> Frontmatter:
    """Build a minimal valid Frontmatter wrapping ``sources``.

    ``embedding_seed`` is at least 3 chars (schema validator); ``id`` is
    a slug matching ``^[a-z][a-z0-9-]{2,63}$``.
    """
    return Frontmatter(
        id="test-class",
        type="bug-class",
        description="test",
        signature=Signature(must_match=["x == 1"], embedding_seed="seed text"),
        related=Related(),
        context_sources=sources,
        evidence=Evidence(),
    )


def test_p1_loaded_first() -> None:
    """P1 sources come before P2 even when P2 was listed first in input."""
    fm = _fm(
        [
            ContextSource(priority=2, name="b", tool="t", params={}),
            ContextSource(priority=1, name="a", tool="t", params={}),
        ]
    )
    out = walk_priority_edges(fm, budget_remaining=10000)
    assert [s.name for s in out] == ["a", "b"]


def test_p3_skipped_under_tight_budget() -> None:
    """``budget_remaining=0`` drops both P2 and P3, keeps only P1."""
    fm = _fm(
        [
            ContextSource(priority=1, name="a", tool="t", params={}),
            ContextSource(priority=3, name="b", tool="t", params={}),
        ]
    )
    out = walk_priority_edges(fm, budget_remaining=0)
    assert [s.name for s in out] == ["a"]


def test_p2_loaded_when_budget_allows() -> None:
    """``budget_remaining=10000`` includes P2 sources behind P1."""
    fm = _fm(
        [
            ContextSource(priority=1, name="a", tool="t", params={}),
            ContextSource(priority=2, name="b", tool="t", params={}),
        ]
    )
    out = walk_priority_edges(fm, budget_remaining=10000)
    assert [s.name for s in out] == ["a", "b"]


def test_p3_loaded_when_budget_generous() -> None:
    """``budget_remaining=5001`` (just past the threshold) includes P3.

    Locks the 5000 threshold so a future tweak (e.g. 5000 -> 6000) fails
    this test rather than silently shifting load behavior.
    """
    fm = _fm(
        [
            ContextSource(priority=1, name="a", tool="t", params={}),
            ContextSource(priority=2, name="b", tool="t", params={}),
            ContextSource(priority=3, name="c", tool="t", params={}),
        ]
    )
    out = walk_priority_edges(fm, budget_remaining=5001)
    assert [s.name for s in out] == ["a", "b", "c"]
