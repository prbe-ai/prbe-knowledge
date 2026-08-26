"""The per-source ANN queries must BIND against a real Postgres.

WHY THIS FILE EXISTS
--------------------
The first deploy of the per-source path failed on every request with

    could not determine data type of parameter $3

because its pool and top-up queries appended their limits as NEW parameters
and left $3 (top_k, bound by `_build_inner_query`) referenced by nothing.
Postgres infers a statement's parameter list from the $n symbols it actually
contains, so a supplied-but-unreferenced parameter has no inferable type and
the BIND fails -- before any row is touched, on every call.

Every shape test passed throughout, because a fake connection accepts any
(sql, params) pair: bind-time errors only exist on a real protocol. So this
file sends the real statements through real asyncpg. Zero rows is fine --
the properties under test (parameter binding, SQL validity) surface on an
empty database exactly as on a full one.

Needs a live Postgres with the schema applied (the standard integration
setup); skips without one, hard error in CI like every other live-DB test.
"""

from __future__ import annotations

import pytest

from engine.retrieval.retrievers import vector as vector_mod
from engine.shared.models import TemporalSpec

pytestmark = pytest.mark.integration


class _FakeEmbedder:
    async def embed_query(self, text: str) -> list[float]:
        return [0.0] * 3072


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vector_mod, "get_embedder_v2", lambda: _FakeEmbedder())


async def test_per_source_path_binds(live_db) -> None:
    """Pool + source-discovery statements reach Postgres and come back."""
    hits = await vector_mod.vector_search(
        "cust-bind", "q", top_k=30, per_source_top_k=5, temporal=TemporalSpec()
    )
    assert hits == []


async def test_topup_statement_binds(live_db) -> None:
    """Force the top-up branch: a caller-supplied source list makes every
    listed source short on an empty database, so the top-up SQL executes --
    the statement whose dangling $3 shipped broken."""
    hits = await vector_mod.vector_search(
        "cust-bind",
        "q",
        top_k=30,
        per_source_top_k=5,
        sources=["github", "custom_ingest"],
        temporal=TemporalSpec(),
    )
    assert hits == []


async def test_default_and_recency_paths_still_bind(live_db) -> None:
    """The refactor moved $3's meaning around; the untouched paths must keep
    binding too."""
    assert await vector_mod.vector_search("cust-bind", "q", top_k=10) == []
    assert (
        await vector_mod.vector_search(
            "cust-bind", "q", top_k=10, sort_by="recency", per_source_top_k=5
        )
        == []
    )
