"""Tests for the classifier's embedding-similarity step (spec §6 step 2).

The wrapper (``rank_by_embedding``) is a thin async shim around
``services.kg.embedding_query.query_similar``. The integration story for
the underlying pgvector query is already covered by
``tests/kg/test_embedding_query.py`` and ``tests/kg/test_tenant_isolation.py``;
the unit-level concerns here are:

1. Forwarding semantics — the wrapper hands the right kwargs through and
   returns whatever ``query_similar`` returned, untouched.
2. Degraded-mode signal — an ``EmbeddingQueryError`` from the underlying
   helper produces a structured ``kg.classifier.embedding_unavailable``
   warning before re-raising. The classifier orchestrator (Task 17/18)
   keys on the exception type to fall back to rules-only.

Both behaviors are exercised by monkeypatching ``query_similar`` so the
tests never touch a DB.
"""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
import structlog

from services.kg.classifier import embedding as embedding_mod
from services.kg.classifier.embedding import rank_by_embedding
from services.kg.embedding_query import EmbeddingMatch, EmbeddingQueryError


@pytest.fixture(autouse=True)
def _route_structlog_to_stdlib() -> None:
    """Route structlog output through stdlib logging so ``caplog`` sees it.

    The production ``configure_logging`` uses ``make_filtering_bound_logger``,
    which writes through ``structlog``'s own logger factory and bypasses
    stdlib logging entirely — meaning ``caplog`` would catch nothing. For
    tests, swap to ``structlog.stdlib.BoundLogger`` + ``LoggerFactory`` so
    each ``log.warning(...)`` becomes a stdlib ``LogRecord`` that ``caplog``
    can assert on.
    """
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


@pytest.mark.asyncio
async def test_returns_top_k_matches_from_query_similar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Wrapper passes kwargs through and returns the underlying matches verbatim."""
    expected = [
        EmbeddingMatch(class_id="auth-401-jwt-refresh", score=0.91),
        EmbeddingMatch(class_id="db-timeout-replica-lag", score=0.74),
    ]
    captured: dict[str, object] = {}

    async def _fake_query_similar(
        conn: object,
        *,
        customer_id: str,
        vector: list[float],
        top_k: int,
    ) -> list[EmbeddingMatch]:
        captured["conn"] = conn
        captured["customer_id"] = customer_id
        captured["vector"] = vector
        captured["top_k"] = top_k
        return expected

    monkeypatch.setattr(embedding_mod, "query_similar", _fake_query_similar)

    sentinel_conn = MagicMock(name="conn")
    vec = [0.1] * 1536
    out = await rank_by_embedding(
        conn=sentinel_conn,
        customer_id="tA",
        vector=vec,
        top_k=2,
    )
    assert out == expected
    assert captured["conn"] is sentinel_conn
    assert captured["customer_id"] == "tA"
    assert captured["vector"] is vec
    assert captured["top_k"] == 2


@pytest.mark.asyncio
async def test_embedding_query_error_is_logged_and_reraised(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``EmbeddingQueryError`` propagates after a structured warning is emitted.

    The orchestrator catches ``EmbeddingQueryError`` to enter degraded
    mode (rules-only against the full class set, spec §6 step 2). The
    log line is the only operational signal that the system entered
    degraded mode, so its event name is part of the contract.
    """

    async def _fake_query_similar(
        conn: object,
        *,
        customer_id: str,
        vector: list[float],
        top_k: int,
    ) -> list[EmbeddingMatch]:
        raise EmbeddingQueryError("fake")

    monkeypatch.setattr(embedding_mod, "query_similar", _fake_query_similar)

    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(EmbeddingQueryError, match="fake"),
    ):
        await rank_by_embedding(
            conn=MagicMock(name="conn"),
            customer_id="tA",
            vector=[0.1] * 1536,
            top_k=5,
        )

    assert any(
        "kg.classifier.embedding_unavailable" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    ), f"expected warning event in caplog; got {[r.getMessage() for r in caplog.records]}"
