"""Unit + live-DB tests for scripts/backfill_embedding_v2.py.

The script populates the chunks.embedding_v2 column for rows where it's
NULL. It runs against a live Postgres (docker-compose) — without that,
the live tests skip via the live_db fixture.

What's covered:
  - _estimate_tokens basic sanity bounds (unit, no DB)
  - End-to-end populates a NULL chunk (live_db)
  - Idempotent re-run is a no-op against fully-populated rows (live_db)
  - --dry-run writes nothing (live_db)
  - --customer filter scopes the work (live_db)

Stub mode: conftest sets GOOGLE_API_KEY="" implicitly via the missing env,
so GeminiEmbedder._ensure_client() returns None and stub _hash_vector
runs. That's enough to verify the pipeline -- the actual API call is
already covered by shared/embeddings unit tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio  # noqa: F401  # required for live_db fixture

from engine.shared.db import raw_conn
from engine.shared.embeddings import reset_embedder
from scripts.backfill_embedding_v2 import (
    RC_CAP_HIT_REMAINING,
    RC_CLEAN,
    _estimate_tokens,
    main,
)

# ---------------------------------------------------------------------------
# Unit tests (no DB)
# ---------------------------------------------------------------------------


def test_estimate_tokens_floors_at_one() -> None:
    # Empty content + no title still costs at least 1 token of overhead.
    assert _estimate_tokens("", None) >= 1


def test_estimate_tokens_grows_with_content() -> None:
    # 4000-char content should exceed 1000-char content roughly proportionally.
    short = _estimate_tokens("a" * 1000, None)
    long_ = _estimate_tokens("a" * 4000, None)
    assert long_ > short
    # Char/4 estimator: 4000 chars ~= 1000 tokens, 1000 chars ~= 250 tokens.
    # Allow 20% slack for the prefix scaffolding constant.
    assert 950 <= long_ <= 1050
    assert 240 <= short <= 280


def test_estimate_tokens_caps_title_at_200_chars() -> None:
    # A 5000-char title doesn't blow the estimate -- internal cap is 200.
    huge_title = "X" * 5000
    short_title = "X" * 200
    assert _estimate_tokens("body", huge_title) == _estimate_tokens(
        "body", short_title
    )


# ---- argparse validators reject nonsense values ------------------------


@pytest.mark.asyncio
async def test_argparse_rejects_negative_cost_cap(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        await main(["--cost-cap", "-1"])
    assert exc_info.value.code == 2  # argparse uses 2 for usage errors


@pytest.mark.asyncio
async def test_argparse_rejects_zero_batch_size(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        await main(["--batch-size", "0"])
    assert exc_info.value.code == 2


@pytest.mark.asyncio
async def test_argparse_rejects_negative_max_batches(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        await main(["--max-batches", "-5"])
    assert exc_info.value.code == 2


@pytest.mark.asyncio
async def test_argparse_rejects_worker_id_out_of_range(capsys) -> None:
    # --worker-id >= --workers catches typos like `--workers 4 --worker-id 4`
    # before they silently fetch zero rows and exit looking like success.
    with pytest.raises(SystemExit) as exc_info:
        await main(["--workers", "4", "--worker-id", "4"])
    assert exc_info.value.code == 2


@pytest.mark.asyncio
async def test_argparse_rejects_negative_worker_id(capsys) -> None:
    with pytest.raises(SystemExit) as exc_info:
        await main(["--workers", "4", "--worker-id", "-1"])
    assert exc_info.value.code == 2


@pytest.mark.asyncio
async def test_argparse_rejects_zero_workers(capsys) -> None:
    # --workers must be > 0; mod(_, 0) is a Postgres error.
    with pytest.raises(SystemExit) as exc_info:
        await main(["--workers", "0"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Live-DB tests
# ---------------------------------------------------------------------------

# Each test cleans up after itself by deleting its customer + cascading rows.


async def _insert_customer(conn, customer_id: str) -> None:
    await conn.execute(
        """
        INSERT INTO customers (customer_id, display_name, api_key_hash, status)
        VALUES ($1, $1, 'test-' || $1, 'active')
        ON CONFLICT (customer_id) DO NOTHING
        """,
        customer_id,
    )


async def _insert_doc(conn, customer_id: str, doc_id: str, title: str) -> None:
    now = datetime.now(UTC)
    await conn.execute(
        """
        INSERT INTO documents (
            doc_id, customer_id, version, source_system, source_id, source_url,
            doc_class, doc_type, content_hash, title, body_preview,
            created_at, updated_at, valid_from, ingested_at, acl
        )
        VALUES (
            $1, $2, 1, 'slack', $1, 'https://example.test/' || $1,
            'message', 'slack.message', md5($3 || $1), $3, '...',
            $4, $4, $4, $4, '{}'::jsonb
        )
        ON CONFLICT (customer_id, doc_id, version) DO NOTHING
        """,
        doc_id,
        customer_id,
        title,
        now,
    )


async def _insert_chunk_no_v2(
    conn, customer_id: str, doc_id: str, idx: int, content: str
) -> str:
    chunk_id = f"{doc_id}:c_{idx}"
    await conn.execute(
        """
        INSERT INTO chunks (
            chunk_id, doc_id, customer_id, chunk_index, content, content_hash,
            token_count, embedding, embedding_model, embedding_dim,
            chunker_version, first_seen_version, last_seen_version, valid_from
        ) VALUES (
            $1, $2, $3, $4, $5, md5($5),
            length($5) / 4,
            array_fill(0::real, ARRAY[3072])::halfvec,
            'openai/text-embedding-3-large', 3072,
            'naive-v1', 1, 1, NOW()
        )
        ON CONFLICT (customer_id, chunk_id) DO NOTHING
        """,
        chunk_id,
        doc_id,
        customer_id,
        idx,
        content,
    )
    return chunk_id


async def _cleanup(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute("DELETE FROM customers WHERE customer_id = $1", customer_id)


@pytest.fixture(autouse=True)
def _reset_embedders() -> None:
    # The Gemini singleton caches its (lack of) API key; reset between tests
    # so a test that monkeypatches the env var sees fresh state.
    reset_embedder()


@pytest.mark.asyncio
async def test_backfill_populates_null_chunk(live_db) -> None:
    cust = "cust-bf-v2-pop"
    try:
        async with raw_conn() as conn:
            await _insert_customer(conn, cust)
            await _insert_doc(conn, cust, "doc-1", "Title for doc 1")
            chunk_id = await _insert_chunk_no_v2(
                conn, cust, "doc-1", 0, "the content of chunk zero"
            )
            row = await conn.fetchrow(
                "SELECT embedding_v2 FROM chunks WHERE chunk_id = $1 AND customer_id = $2",
                chunk_id,
                cust,
            )
            assert row["embedding_v2"] is None

        rc = await main(["--customer", cust, "--cost-cap", "1.0"])
        assert rc == 0

        async with raw_conn() as conn:
            row = await conn.fetchrow(
                """
                SELECT embedding_v2, embedding_v2_model, embedding_v2_dim
                FROM chunks WHERE chunk_id = $1 AND customer_id = $2
                """,
                chunk_id,
                cust,
            )
            assert row["embedding_v2"] is not None
            assert row["embedding_v2_model"] == "google/gemini-embedding-2"
            assert row["embedding_v2_dim"] == 3072
    finally:
        await _cleanup(cust)


@pytest.mark.asyncio
async def test_backfill_partitions_are_disjoint_and_complete(live_db) -> None:
    """Two parallel workers split the rows by hashtext modulo. Together they
    cover every NULL row in the customer; neither one alone covers all.
    Without disjoint partitioning, multi-process runs would either
    double-bill (overlap) or skip rows (gap).
    """
    cust = "cust-bf-v2-part"
    try:
        async with raw_conn() as conn:
            await _insert_customer(conn, cust)
            await _insert_doc(conn, cust, "doc-1", "T")
            # 20 chunks gives both partitions a non-trivial slice with high
            # probability under hashtext's even distribution.
            for i in range(20):
                await _insert_chunk_no_v2(conn, cust, "doc-1", i, f"content {i}")

        # Worker 0 only.
        rc0 = await main(
            ["--customer", cust, "--workers", "2", "--worker-id", "0", "--cost-cap", "1.0"]
        )
        assert rc0 == 0

        # Some rows remain (partition 1 unfilled). Verify both:
        #   - Worker 0 is fully drained (no NULL rows in its partition).
        #   - Customer-level NULL count is > 0 (worker 1 still owns some).
        async with raw_conn() as conn:
            null_after_w0 = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE customer_id=$1 AND embedding_v2 IS NULL",
                cust,
            )
            null_in_w0_partition = await conn.fetchval(
                """
                SELECT COUNT(*) FROM chunks
                WHERE customer_id=$1 AND embedding_v2 IS NULL
                  AND mod(abs(hashtext(chunk_id)), 2) = 0
                """,
                cust,
            )
        assert null_in_w0_partition == 0, "worker 0 left NULLs in its own partition"
        assert null_after_w0 > 0, "if worker 0 covered everything, partitioning is broken"
        assert null_after_w0 < 20, "worker 0 covered nothing — partition predicate broken"

        # Worker 1 finishes the rest.
        rc1 = await main(
            ["--customer", cust, "--workers", "2", "--worker-id", "1", "--cost-cap", "1.0"]
        )
        assert rc1 == 0

        async with raw_conn() as conn:
            final_null = await conn.fetchval(
                "SELECT COUNT(*) FROM chunks WHERE customer_id=$1 AND embedding_v2 IS NULL",
                cust,
            )
        assert final_null == 0, "two workers together left NULLs — partitions don't tile"
    finally:
        await _cleanup(cust)


@pytest.mark.asyncio
async def test_backfill_idempotent_rerun(live_db) -> None:
    """A second run on a customer with no NULL rows is a no-op (zero updates)."""
    cust = "cust-bf-v2-idem"
    try:
        async with raw_conn() as conn:
            await _insert_customer(conn, cust)
            await _insert_doc(conn, cust, "doc-1", "T")
            await _insert_chunk_no_v2(conn, cust, "doc-1", 0, "content")

        await main(["--customer", cust, "--cost-cap", "1.0"])
        # All NULLs filled. Second run should hit the no_more_chunks branch.
        rc = await main(["--customer", cust, "--cost-cap", "1.0"])
        assert rc == 0
    finally:
        await _cleanup(cust)


@pytest.mark.asyncio
async def test_backfill_dry_run_writes_nothing(live_db) -> None:
    cust = "cust-bf-v2-dry"
    try:
        async with raw_conn() as conn:
            await _insert_customer(conn, cust)
            await _insert_doc(conn, cust, "doc-1", "T")
            chunk_id = await _insert_chunk_no_v2(conn, cust, "doc-1", 0, "content")

        rc = await main(["--customer", cust, "--dry-run"])
        assert rc == 0

        async with raw_conn() as conn:
            row = await conn.fetchrow(
                "SELECT embedding_v2 FROM chunks WHERE chunk_id = $1 AND customer_id = $2",
                chunk_id,
                cust,
            )
            # Dry-run must not mutate the column.
            assert row["embedding_v2"] is None
    finally:
        await _cleanup(cust)


@pytest.mark.asyncio
async def test_backfill_returns_rc_2_when_cap_hit_with_rows_remaining(live_db) -> None:
    """The operator-meaningful rc contract: cap-hit with NULL rows still
    in the table returns 2, NOT 0. Without this, an operator sees rc=0
    and assumes Stage 3 (HNSW build) is safe — but it would build over a
    NULL-heavy column. The contract is the gate.
    """
    cust = "cust-bf-v2-cap"
    try:
        async with raw_conn() as conn:
            await _insert_customer(conn, cust)
            await _insert_doc(conn, cust, "doc-1", "T")
            # Insert two chunks; --max-batches=1 + batch-size=1 stops after
            # processing only the first one, leaving one NULL row behind.
            await _insert_chunk_no_v2(conn, cust, "doc-1", 0, "first content")
            await _insert_chunk_no_v2(conn, cust, "doc-1", 1, "second content")

        rc = await main(
            ["--customer", cust, "--max-batches", "1", "--batch-size", "1"]
        )
        assert rc == RC_CAP_HIT_REMAINING

        # Sanity: a follow-up run drains the remaining NULLs and returns
        # RC_CLEAN, proving the rc=2 was about NULL rows specifically.
        rc = await main(["--customer", cust])
        assert rc == RC_CLEAN
    finally:
        await _cleanup(cust)


@pytest.mark.asyncio
async def test_backfill_customer_filter_isolates(live_db) -> None:
    """A run scoped to customer A leaves customer B's NULL chunks untouched."""
    cust_a = "cust-bf-v2-a"
    cust_b = "cust-bf-v2-b"
    try:
        async with raw_conn() as conn:
            await _insert_customer(conn, cust_a)
            await _insert_customer(conn, cust_b)
            await _insert_doc(conn, cust_a, "doc-a", "T")
            await _insert_doc(conn, cust_b, "doc-b", "T")
            cid_a = await _insert_chunk_no_v2(conn, cust_a, "doc-a", 0, "content a")
            cid_b = await _insert_chunk_no_v2(conn, cust_b, "doc-b", 0, "content b")

        await main(["--customer", cust_a, "--cost-cap", "1.0"])

        async with raw_conn() as conn:
            row_a = await conn.fetchrow(
                "SELECT embedding_v2 FROM chunks WHERE chunk_id = $1 AND customer_id = $2",
                cid_a,
                cust_a,
            )
            row_b = await conn.fetchrow(
                "SELECT embedding_v2 FROM chunks WHERE chunk_id = $1 AND customer_id = $2",
                cid_b,
                cust_b,
            )
            assert row_a["embedding_v2"] is not None
            assert row_b["embedding_v2"] is None
    finally:
        await _cleanup(cust_a)
        await _cleanup(cust_b)
