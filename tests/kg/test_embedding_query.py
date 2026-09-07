"""Tests for ``services.kg.embedding_query``.

Coverage:

- Unit (no DB needed): vector-format helper, frozen dataclass invariant,
  argument-validation gates, DB-error wrapping. These run without
  Postgres and exercise the validation/translation paths.
- Live-DB: round-trip a class with a known embedding through
  ``query_similar`` and confirm we get ourselves back. Skipped
  automatically when Postgres isn't reachable (see
  ``tests/kg/conftest.py``).
"""

from __future__ import annotations

import pytest

from services.kg.embedding_query import (
    EmbeddingMatch,
    EmbeddingQueryError,
    _format_vector,
    query_similar,
    query_similar_candidates,
)

# ---------------------------------------------------------------------------
# Unit tests — no DB required.
# ---------------------------------------------------------------------------


def test_format_vector_basic() -> None:
    assert _format_vector([1.0, 2.0, 3.0]) == "[1.0,2.0,3.0]"


def test_format_vector_coerces_ints_to_floats() -> None:
    """Ints round-trip as ``1.0`` (pgvector requires float syntax)."""
    assert _format_vector([1, 2, 3]) == "[1.0,2.0,3.0]"


def test_format_vector_preserves_precision() -> None:
    """``repr(float)`` keeps full precision; ``f"{x}"`` would not."""
    out = _format_vector([0.123456789012345])
    # The literal must round-trip the value exactly when pgvector parses it.
    assert "0.123456789012345" in out


def test_embedding_match_dataclass_is_frozen() -> None:
    """``frozen=True`` means assignment after construction must raise
    ``FrozenInstanceError`` (a subclass of ``AttributeError``)."""
    import dataclasses

    m = EmbeddingMatch(class_id="x", score=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.score = 0.9  # type: ignore[misc]


@pytest.mark.asyncio
async def test_query_similar_rejects_zero_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        await query_similar(
            conn=None,  # type: ignore[arg-type]
            customer_id="c",
            vector=[0.1] * 1536,
            top_k=0,
        )


@pytest.mark.asyncio
async def test_query_similar_rejects_negative_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        await query_similar(
            conn=None,  # type: ignore[arg-type]
            customer_id="c",
            vector=[0.1] * 1536,
            top_k=-3,
        )


@pytest.mark.asyncio
async def test_query_similar_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="vector"):
        await query_similar(
            conn=None,  # type: ignore[arg-type]
            customer_id="c",
            vector=[],
            top_k=5,
        )


@pytest.mark.asyncio
async def test_query_similar_candidates_rejects_threshold_above_one() -> None:
    with pytest.raises(ValueError, match="threshold"):
        await query_similar_candidates(
            conn=None,  # type: ignore[arg-type]
            customer_id="c",
            payload_hash="h",
            vector=[0.1] * 1536,
            threshold=1.5,
        )


@pytest.mark.asyncio
async def test_query_similar_candidates_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        await query_similar_candidates(
            conn=None,  # type: ignore[arg-type]
            customer_id="c",
            payload_hash="h",
            vector=[0.1] * 1536,
            threshold=-0.1,
        )


@pytest.mark.asyncio
async def test_query_similar_candidates_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="vector"):
        await query_similar_candidates(
            conn=None,  # type: ignore[arg-type]
            customer_id="c",
            payload_hash="h",
            vector=[],
            threshold=0.85,
        )


@pytest.mark.asyncio
async def test_query_similar_wraps_db_error_as_embedding_query_error() -> None:
    """``asyncpg.PostgresError`` from ``conn.fetch`` becomes ``EmbeddingQueryError``.

    The classifier's degrade-to-rules-only path keys on this exception
    type, so the translation is load-bearing.
    """
    import asyncpg

    class _BoomConn:
        async def fetch(self, *args: object, **kwargs: object) -> list[object]:
            raise asyncpg.PostgresError("boom")

    with pytest.raises(EmbeddingQueryError, match="boom"):
        await query_similar(
            conn=_BoomConn(),  # type: ignore[arg-type]
            customer_id="c",
            vector=[0.1] * 1536,
            top_k=5,
        )


@pytest.mark.asyncio
async def test_query_similar_candidates_wraps_db_error() -> None:
    """Same translation behavior on the candidates path."""
    import asyncpg

    class _BoomConn:
        async def fetch(self, *args: object, **kwargs: object) -> list[object]:
            raise asyncpg.PostgresError("kapow")

    with pytest.raises(EmbeddingQueryError, match="kapow"):
        await query_similar_candidates(
            conn=_BoomConn(),  # type: ignore[arg-type]
            customer_id="c",
            payload_hash="h",
            vector=[0.1] * 1536,
        )


@pytest.mark.asyncio
async def test_query_similar_wraps_oserror_as_embedding_query_error() -> None:
    """Connection-level OSError (socket reset etc.) also becomes ``EmbeddingQueryError``."""

    class _DropConn:
        async def fetch(self, *args: object, **kwargs: object) -> list[object]:
            raise OSError("connection reset")

    with pytest.raises(EmbeddingQueryError, match="connection reset"):
        await query_similar(
            conn=_DropConn(),  # type: ignore[arg-type]
            customer_id="c",
            vector=[0.1] * 1536,
            top_k=3,
        )


@pytest.mark.asyncio
async def test_query_similar_passes_expected_sql_args() -> None:
    """Confirm the helper passes ``(literal, customer_id, top_k)`` in order
    so the ``$1/$2/$3`` SQL placeholders map correctly. A regression in
    arg ordering would silently return wrong rows or raise a Postgres
    error at runtime; the unit test catches the wiring without a DB.
    """

    captured: list[tuple[object, ...]] = []

    class _CapturingConn:
        async def fetch(self, sql: str, *args: object) -> list[object]:
            captured.append(args)
            return []

    out = await query_similar(
        conn=_CapturingConn(),  # type: ignore[arg-type]
        customer_id="cust-x",
        vector=[0.5, 0.25],
        top_k=7,
    )
    assert out == []
    assert captured == [("[0.5,0.25]", "cust-x", 7)]


@pytest.mark.asyncio
async def test_query_similar_candidates_passes_expected_sql_args() -> None:
    """Same arg-order check on the dedup path. ``statuses`` must be cast
    to ``list`` so asyncpg binds it as a text array; a tuple works in
    most cases but ``list`` is the documented contract."""

    captured: list[tuple[object, ...]] = []

    class _CapturingConn:
        async def fetch(self, sql: str, *args: object) -> list[object]:
            captured.append(args)
            return []

    out = await query_similar_candidates(
        conn=_CapturingConn(),  # type: ignore[arg-type]
        customer_id="cust-y",
        payload_hash="abc123",
        vector=[0.1, 0.2],
        threshold=0.9,
        statuses=("pending", "merged"),
    )
    assert out == []
    assert captured == [
        ("[0.1,0.2]", "cust-y", "abc123", ["pending", "merged"], 0.9),
    ]


# ---------------------------------------------------------------------------
# Live-DB tests — auto-skipped when Postgres isn't reachable.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_similar_live(live_db_conn) -> None:  # type: ignore[no-untyped-def]
    """End-to-end: insert a class with a known embedding, query for it,
    expect ourselves back at the top of the list."""
    customer_id = "cust-emb-test"
    vec = [0.1] + [0.0] * 1535
    vec_literal = "[" + ",".join(repr(float(v)) for v in vec) + "]"

    # Seed a customer (FK requirement) and a class. ``ON CONFLICT DO
    # NOTHING`` lets the test rerun cleanly if the cleanup at the end
    # didn't fire (e.g., earlier assertion failure mid-test).
    await live_db_conn.execute(
        "INSERT INTO customers (customer_id, display_name, api_key_hash, status) "
        "VALUES ($1, 'emb test', 'hash', 'active') ON CONFLICT DO NOTHING",
        customer_id,
    )
    # Set the GUC inline so RLS sees the right tenant. The connection
    # isn't inside ``with_tenant()`` here — that's the helper's
    # documented contract for callers, but seeding requires the same
    # GUC since RLS is FORCEd on kg_classes.
    await live_db_conn.execute(
        "SELECT set_config('app.current_customer_id', $1, true)",
        customer_id,
    )
    try:
        await live_db_conn.execute(
            "INSERT INTO kg_classes "
            "(customer_id, class_id, frontmatter, body, signature_embedding) "
            "VALUES ($1, 'emb-self-test', '{\"id\":\"emb-self-test\"}'::jsonb, "
            "'', $2::vector)",
            customer_id,
            vec_literal,
        )

        matches = await query_similar(
            conn=live_db_conn,
            customer_id=customer_id,
            vector=vec,
            top_k=5,
        )
        ids = {m.class_id for m in matches}
        assert "emb-self-test" in ids
        # Self-similarity should be ~1.0 for an identical vector.
        self_score = next(m.score for m in matches if m.class_id == "emb-self-test")
        assert self_score == pytest.approx(1.0, abs=1e-5)
    finally:
        # Clean up so the test is idempotent — the live_db fixture's
        # TRUNCATE_SQL doesn't currently include kg_classes.
        await live_db_conn.execute(
            "DELETE FROM kg_classes "
            "WHERE customer_id = $1 AND class_id = 'emb-self-test'",
            customer_id,
        )
        await live_db_conn.execute(
            "DELETE FROM customers WHERE customer_id = $1",
            customer_id,
        )
