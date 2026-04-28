"""SQL retriever integration tests against the live test DB.

Verifies the deterministic list pipeline's three operations and the
filter combinations they support. Uses the same `live_db` fixture pattern
as the rest of the test suite — requires Postgres + the schema applied.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.retrieval.retrievers.sql import (
    sql_count,
    sql_group_by,
    sql_list,
)
from shared.config import Settings, get_settings
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import TemporalMode, TemporalSpec
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---- Seed helpers ---------------------------------------------------------


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', $2)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
            "hash-" + customer_id,
        )


async def _seed_doc(
    customer_id: str,
    doc_id: str,
    *,
    source_system: str,
    doc_type: str,
    title: str,
    content: str,
    updated_at: datetime,
    author_id: str | None = None,
    version: int = 1,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                author_id,
                created_at, updated_at, valid_from, ingested_at,
                acl
            ) VALUES (
                $1, $2, $3,
                $4, $5, $6,
                'raw_source', $7, 'text/plain',
                $8, $9, $10, 0,
                $11,
                $12, $12, $12, $12,
                '{}'::jsonb
            )
            """,
            doc_id,
            version,
            customer_id,
            source_system,
            doc_id + "-src",
            f"https://example/{doc_id}",
            doc_type,
            f"hash-{doc_id}",
            title,
            len(content.encode()),
            author_id,
            updated_at,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1, $2, $3,
                0, $4, $5, 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                $6, $6
            )
            ON CONFLICT (doc_id, content_hash) DO NOTHING
            """,
            f"{doc_id}:c0:v{version}",
            doc_id,
            customer_id,
            content,
            f"chash-{doc_id}",
            version,
        )


# ---- sql_list -------------------------------------------------------------


async def test_sql_list_returns_recent_first(live_db) -> None:
    cust = "cust-A"
    await _seed_customer(cust)
    base = datetime(2026, 4, 28, 10, 0, tzinfo=UTC)
    # 5 commits, descending timestamps.
    for i in range(5):
        await _seed_doc(
            cust,
            f"github:repo:commit:{i}",
            source_system="github",
            doc_type="github.commit",
            title=f"commit {i}",
            content=f"commit {i} content",
            updated_at=base - timedelta(minutes=i),
        )

    hits = await sql_list(
        cust,
        top_k=3,
        sources=["github"],
        doc_types=["github.commit"],
    )
    assert len(hits) == 3
    # Newest first.
    assert hits[0].title == "commit 0"
    assert hits[1].title == "commit 1"
    assert hits[2].title == "commit 2"
    # All chunks present, score decays with rank.
    assert hits[0].score > hits[1].score > hits[2].score


async def test_sql_list_filters_by_doc_type_excluding_other_types(live_db) -> None:
    """The bug-fix regression: 'commits' query must NOT return PRs."""
    cust = "cust-B"
    await _seed_customer(cust)
    now = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        cust,
        "github:repo:commit:1",
        source_system="github",
        doc_type="github.commit",
        title="real commit",
        content="commit content",
        updated_at=now,
    )
    await _seed_doc(
        cust,
        "github:repo:pr:1",
        source_system="github",
        doc_type="github.pull_request",
        title="a PR (should NOT show up)",
        content="pr content",
        updated_at=now + timedelta(minutes=1),  # newer, but wrong type
    )

    hits = await sql_list(cust, top_k=10, doc_types=["github.commit"])
    assert len(hits) == 1
    assert "PR" not in (hits[0].title or "")
    assert hits[0].title == "real commit"


async def test_sql_list_no_filters_returns_anything_recent(live_db) -> None:
    cust = "cust-C"
    await _seed_customer(cust)
    now = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        cust,
        "slack:T:C:1.0",
        source_system="slack",
        doc_type="slack.message",
        title="msg",
        content="hi",
        updated_at=now,
    )

    hits = await sql_list(cust, top_k=10)
    assert len(hits) == 1


async def test_sql_list_temporal_changed_between(live_db) -> None:
    cust = "cust-D"
    await _seed_customer(cust)
    base = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        cust,
        "doc:old",
        source_system="github",
        doc_type="github.commit",
        title="old",
        content="old content",
        updated_at=base - timedelta(days=10),
    )
    await _seed_doc(
        cust,
        "doc:new",
        source_system="github",
        doc_type="github.commit",
        title="new",
        content="new content",
        updated_at=base - timedelta(days=1),
    )

    hits = await sql_list(
        cust,
        top_k=10,
        doc_types=["github.commit"],
        temporal=TemporalSpec(
            mode=TemporalMode.CHANGED_BETWEEN,
            since=base - timedelta(days=5),
            until=base + timedelta(days=1),
        ),
    )
    assert {h.title for h in hits} == {"new"}


async def test_sql_list_filters_by_author(live_db) -> None:
    cust = "cust-E"
    await _seed_customer(cust)
    now = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        cust,
        "doc:alice",
        source_system="github",
        doc_type="github.commit",
        title="alice's",
        content="x",
        author_id="alice",
        updated_at=now,
    )
    await _seed_doc(
        cust,
        "doc:bob",
        source_system="github",
        doc_type="github.commit",
        title="bob's",
        content="y",
        author_id="bob",
        updated_at=now + timedelta(minutes=1),
    )

    hits = await sql_list(cust, top_k=10, author_ids=["alice"])
    assert {h.title for h in hits} == {"alice's"}


async def test_sql_list_invalid_sort_raises() -> None:
    """Defense-in-depth — even if someone bypasses the dispatcher's
    validation, sql_list itself should reject SQL-injection vectors."""
    with pytest.raises(ValueError):
        await sql_list("anyone", sort_field="DROP TABLE", sort_direction="desc")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await sql_list("anyone", sort_field="updated_at", sort_direction="--")  # type: ignore[arg-type]


# ---- sql_count ------------------------------------------------------------


async def test_sql_count_with_filters(live_db) -> None:
    cust = "cust-F"
    await _seed_customer(cust)
    now = datetime(2026, 4, 28, tzinfo=UTC)
    for i in range(3):
        await _seed_doc(
            cust,
            f"github:repo:commit:{i}",
            source_system="github",
            doc_type="github.commit",
            title=f"c{i}",
            content="x",
            updated_at=now,
        )
    await _seed_doc(
        cust,
        "github:repo:pr:1",
        source_system="github",
        doc_type="github.pull_request",
        title="pr",
        content="x",
        updated_at=now,
    )

    n_commits = await sql_count(cust, doc_types=["github.commit"])
    assert n_commits == 3
    n_prs = await sql_count(cust, doc_types=["github.pull_request"])
    assert n_prs == 1
    n_total = await sql_count(cust)
    assert n_total == 4


# ---- sql_group_by ---------------------------------------------------------


async def test_sql_group_by_author(live_db) -> None:
    cust = "cust-G"
    await _seed_customer(cust)
    now = datetime(2026, 4, 28, tzinfo=UTC)
    for i, author in enumerate(["alice", "alice", "alice", "bob", "bob"]):
        await _seed_doc(
            cust,
            f"doc:{i}",
            source_system="github",
            doc_type="github.commit",
            title=str(i),
            content="x",
            author_id=author,
            updated_at=now,
        )

    groups = await sql_group_by(cust, key="author_id", top_k=10)
    assert groups[0]["key"] == "alice"
    assert groups[0]["n"] == 3
    assert groups[1]["key"] == "bob"
    assert groups[1]["n"] == 2


async def test_sql_group_by_invalid_key_raises() -> None:
    with pytest.raises(ValueError):
        await sql_group_by("anyone", key="content")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        await sql_group_by("anyone", key="title; DROP TABLE")  # type: ignore[arg-type]


# ---- Multi-tenant isolation (R2 mandatory regression) ---------------------


async def test_sql_list_does_not_leak_across_tenants(live_db) -> None:
    """RLS/tenant filter regression — tenant A's list query must never
    surface tenant B's docs even when filters would otherwise match."""
    await _seed_customer("tenant-A")
    await _seed_customer("tenant-B")
    now = datetime(2026, 4, 28, tzinfo=UTC)
    await _seed_doc(
        "tenant-A",
        "doc:A",
        source_system="github",
        doc_type="github.commit",
        title="A's commit",
        content="secret-A",
        updated_at=now,
    )
    await _seed_doc(
        "tenant-B",
        "doc:B",
        source_system="github",
        doc_type="github.commit",
        title="B's commit",
        content="secret-B",
        updated_at=now,
    )

    hits_a = await sql_list("tenant-A", top_k=10, doc_types=["github.commit"])
    titles_a = {h.title for h in hits_a}
    assert titles_a == {"A's commit"}
    assert "B's commit" not in titles_a

    n_a = await sql_count("tenant-A", doc_types=["github.commit"])
    n_b = await sql_count("tenant-B", doc_types=["github.commit"])
    assert n_a == 1
    assert n_b == 1
