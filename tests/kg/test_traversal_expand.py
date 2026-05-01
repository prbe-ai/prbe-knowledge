"""Tests for 1-hop expansion (spec §6 step 5).

The expansion step runs after the classifier picks a class: it gathers
the union of ``related.{analogous_to, overlaps_with, often_confused_with}``
and ranks those neighbors by cosine similarity to the matched class's
signature embedding — same pgvector similarity function the classifier
uses, so the scores are directly comparable.

These tests mock asyncpg.Connection.fetch directly and verify:

1. Returned rows are unwrapped into ``EmbeddingMatch`` in order.
2. An empty union short-circuits without hitting the DB.
3. Self-references in ``related`` are excluded from the union.
4. A DB error is logged at WARNING and re-raised as ``ExpandError``.

No live DB — the integration story is covered by
``test_embedding_query.py`` and ``test_tenant_isolation.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import asyncpg
import pytest

from services.kg.embedding_query import EmbeddingMatch
from services.kg.schema import (
    Evidence,
    Frontmatter,
    Related,
    Signature,
)
from services.kg.traversal.expand import ExpandError, expand_one_hop


class _FakeConn:
    """Captures fetch calls and returns scripted rows.

    The implementation calls ``conn.fetch(query, *params)`` so this fake
    matches that signature. ``rows`` is the canned response; raise via
    ``raise_on_fetch`` to simulate a DB error.
    """

    def __init__(
        self,
        rows: list[dict[str, Any]] | None = None,
        raise_on_fetch: BaseException | None = None,
    ) -> None:
        self._rows = rows or []
        self._raise = raise_on_fetch
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(
        self, query: str, *params: Any
    ) -> list[dict[str, Any]]:
        self.calls.append((query, params))
        if self._raise is not None:
            raise self._raise
        return self._rows


def _fm(
    *,
    class_id: str = "auth-401-jwt-refresh",
    analogous_to: list[str] | None = None,
    overlaps_with: list[str] | None = None,
    often_confused_with: list[str] | None = None,
    regressed_by: list[str] | None = None,
) -> Frontmatter:
    """Build a minimal valid Frontmatter with the given related ids."""
    return Frontmatter(
        id=class_id,
        type="bug-class",
        description="test",
        signature=Signature(must_match=["x == 1"], embedding_seed="seed text"),
        related=Related(
            analogous_to=analogous_to or [],
            overlaps_with=overlaps_with or [],
            often_confused_with=often_confused_with or [],
            regressed_by=regressed_by or [],
        ),
        context_sources=[],
        evidence=Evidence(),
    )


@pytest.mark.asyncio
async def test_returns_ranked_matches() -> None:
    """Three related ids -> three rows -> three matches in score order."""
    rows = [
        {"class_id": "auth-403-rbac", "score": 0.91},
        {"class_id": "auth-401-replay", "score": 0.74},
        {"class_id": "session-expired", "score": 0.55},
    ]
    conn = _FakeConn(rows=rows)
    fm = _fm(
        class_id="auth-401-jwt-refresh",
        analogous_to=["auth-403-rbac"],
        overlaps_with=["auth-401-replay"],
        often_confused_with=["session-expired"],
    )

    out = await expand_one_hop(
        conn=conn,
        customer_id="tA",
        frontmatter=fm,
        top_k=10,
    )

    assert out == [
        EmbeddingMatch(class_id="auth-403-rbac", score=0.91),
        EmbeddingMatch(class_id="auth-401-replay", score=0.74),
        EmbeddingMatch(class_id="session-expired", score=0.55),
    ]
    assert len(conn.calls) == 1


@pytest.mark.asyncio
async def test_empty_when_no_related() -> None:
    """No related ids -> ``[]`` without a DB round-trip."""
    conn = _FakeConn(rows=[])
    fm = _fm(class_id="auth-401-jwt-refresh")  # all related lists empty

    out = await expand_one_hop(
        conn=conn,
        customer_id="tA",
        frontmatter=fm,
    )

    assert out == []
    assert len(conn.calls) == 0


@pytest.mark.asyncio
async def test_self_reference_excluded() -> None:
    """``related`` may include the class's own id; the union must drop it.

    The DB query's ``$3::text[]`` parameter (the union of related ids)
    should not contain the matched class's own ``class_id``.
    """
    conn = _FakeConn(rows=[])
    fm = _fm(
        class_id="auth-401-jwt-refresh",
        analogous_to=["auth-401-jwt-refresh", "auth-403-rbac"],
        overlaps_with=["session-expired"],
    )

    await expand_one_hop(
        conn=conn,
        customer_id="tA",
        frontmatter=fm,
    )

    assert len(conn.calls) == 1
    _query, params = conn.calls[0]
    # Params are (customer_id, class_id, related_ids, top_k).
    related_ids = params[2]
    assert "auth-401-jwt-refresh" not in related_ids
    assert set(related_ids) == {"auth-403-rbac", "session-expired"}


@pytest.mark.asyncio
async def test_regressed_by_excluded_from_union() -> None:
    """Spec §6 step 5 names exactly three relation types; ``regressed_by``
    is NOT one of them. Verify it doesn't sneak into the union."""
    conn = _FakeConn(rows=[])
    fm = _fm(
        class_id="auth-401-jwt-refresh",
        analogous_to=["auth-403-rbac"],
        regressed_by=["pr-123-merged"],
    )

    await expand_one_hop(
        conn=conn,
        customer_id="tA",
        frontmatter=fm,
    )

    assert len(conn.calls) == 1
    _query, params = conn.calls[0]
    related_ids = params[2]
    assert "pr-123-merged" not in related_ids
    assert set(related_ids) == {"auth-403-rbac"}


@pytest.mark.asyncio
async def test_db_error_logged_and_reraised(
    _route_structlog_to_stdlib: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``asyncpg.PostgresError`` -> ``ExpandError`` + structured warning."""
    conn = _FakeConn(raise_on_fetch=asyncpg.PostgresError("fake"))
    fm = _fm(
        class_id="auth-401-jwt-refresh",
        analogous_to=["auth-403-rbac"],
    )

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(ExpandError, match="fake"),
    ):
        await expand_one_hop(
            conn=conn,
            customer_id="tA",
            frontmatter=fm,
        )

    assert any(
        "kg.traversal.expand_failed" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected warning event in caplog; got {[r.getMessage() for r in caplog.records]}"
