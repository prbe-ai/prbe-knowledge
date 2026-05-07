"""Unit tests for the exact-id retriever and the BM25 OR-of-tokens query
builder, plus a live-DB integration test for the id_lookup SQL.

The pure helpers (`is_lookup_candidate`, `_build_or_tsquery_string`) cover
the gates that decide whether a SQL pass even runs. The integration test
exercises the SELECT/JOIN/LIKE-ANY shape so kind filtering, DISTINCT ON
ordering, and the source_id-prefix fallback don't silently regress.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval.retrievers.bm25 import _build_or_tsquery_string
from services.retrieval.retrievers.id_lookup import id_lookup_search, is_lookup_candidate
from shared.db import raw_conn
from shared.models import TemporalSpec

# ---- is_lookup_candidate ---------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        "3c325e11-2008-46a9-83f7-fc40d11eaf82",  # session UUID
        "3C325E11-2008-46A9-83F7-FC40D11EAF82",  # uppercase UUID
        "PRB-17",  # Linear ticket
        "PRBE-2049",  # short Linear-ish ticket
        "prbe-backend#49",  # GitHub repo issue/PR ref
        "prbe-ai/prbe-backend#49",  # owner/repo issue/PR ref
        "abc1234567890abcdef1234567890abcdef12345",  # 40-char hex (commit SHA)
        "1234567890ab",  # 12-char hex
    ],
)
def test_is_lookup_candidate_accepts_stable_identifiers(value: str) -> None:
    assert is_lookup_candidate(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "",
        "auth",
        "prbe-backend",  # service slug
        "prbe-knowledge",  # repo slug
        "Richard Wei",  # display name
        "richard@prbe.ai",  # email
        "session 3c325e11",  # display string, not a canonical id
        "3c325e11",  # 8-char prefix is too ambiguous (matches dozens of words)
        "abc",  # too short to be a hash prefix
    ],
)
def test_is_lookup_candidate_rejects_words_and_short_tokens(value: str) -> None:
    assert is_lookup_candidate(value) is False


# ---- _build_or_tsquery_string ----------------------------------------------


def test_or_tsquery_keeps_uuid_parts_as_separate_tokens() -> None:
    """A query with a hyphenated UUID becomes an OR of every hex component
    so a metadata chunk that lexes the URL into the same parts can match.
    """
    out = _build_or_tsquery_string(
        "agent session 3c325e11-2008-46a9-83f7-fc40d11eaf82"
    )
    parts = [p.strip() for p in out.split("|")]
    # Every word is present, ORed (not ANDed).
    assert "agent" in parts
    assert "session" in parts
    assert "3c325e11" in parts
    assert "2008" in parts
    assert "46a9" in parts
    assert "83f7" in parts
    assert "fc40d11eaf82" in parts
    # The query string uses Postgres' OR operator.
    assert " | " in out


def test_or_tsquery_drops_one_char_tokens() -> None:
    """Postgres' english config strips 1-char lexemes on the index side, so
    matching them on the query side is dead weight."""
    out = _build_or_tsquery_string("a auth b")
    assert "auth" in out
    parts = [p.strip() for p in out.split("|")]
    assert "a" not in parts
    assert "b" not in parts


def test_or_tsquery_returns_empty_for_no_usable_tokens() -> None:
    assert _build_or_tsquery_string("") == ""
    assert _build_or_tsquery_string("   ") == ""
    assert _build_or_tsquery_string("a b c") == ""  # all 1-char
    assert _build_or_tsquery_string("!!!") == ""


def test_or_tsquery_handles_punctuation() -> None:
    out = _build_or_tsquery_string("auth.refactor: needs review!")
    parts = [p.strip() for p in out.split("|")]
    # Tokens are alnum/underscore runs only; punctuation drops away.
    assert "auth" in parts
    assert "refactor" in parts
    assert "needs" in parts
    assert "review" in parts


# ---- id_lookup_search (live DB) --------------------------------------------


_NOW = datetime(2026, 5, 7, tzinfo=UTC)


async def _seed_session_doc(
    customer_id: str,
    *,
    doc_id: str,
    source_id: str,
    body: str = "session transcript content",
) -> None:
    """Seed a claude_code session doc + one content chunk.

    Mirrors the shape the claude_code handler produces so the lookup SQL
    runs against realistic data (kind='content', valid version range,
    matching source_id).
    """
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $1, 1, $2,
                'claude_code', $3, 'https://prbe.ai/dashboard/agent-sessions/' || $3,
                'raw_source', 'claude_code.session', 'application/json',
                'h-' || $1, 'Session ' || substr($3, 1, 8), 100, 0,
                $4, $4, $4, $4, '{}'::jsonb
            )
            """,
            doc_id, customer_id, source_id, _NOW,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count, kind,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1, $2, $3, 0, $4, $5, 5, 'content',
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1
            )
            """,
            f"{doc_id}:c0", doc_id, customer_id,
            body, f"chash-{doc_id}",
        )


@pytest.mark.asyncio
async def test_id_lookup_returns_matched_doc_for_bare_uuid(live_db) -> None:
    """The router emits a bare UUID; the lookup matches a doc whose
    `source_id` equals that UUID and returns its first content chunk."""
    cust = "cust-id-lookup"
    target_uuid = "3c325e11-2008-46a9-83f7-fc40d11eaf82"
    other_uuid = "ffffffff-1111-2222-3333-444444444444"
    await _seed_session_doc(
        cust,
        doc_id=f"claude_code:{cust}:{target_uuid}",
        source_id=target_uuid,
        body="hit content",
    )
    await _seed_session_doc(
        cust,
        doc_id=f"claude_code:{cust}:{other_uuid}",
        source_id=other_uuid,
        body="other content",
    )

    hits = await id_lookup_search(cust, [target_uuid], temporal=TemporalSpec())

    assert len(hits) == 1
    assert hits[0].doc_id == f"claude_code:{cust}:{target_uuid}"
    assert hits[0].content == "hit content"
    assert hits[0].kind == "content"
    assert hits[0].score == 1.0


@pytest.mark.asyncio
async def test_id_lookup_matches_prefixed_source_id(live_db) -> None:
    """Handlers that encode a kind prefix (`issue:<uuid>`) per the
    documents.source_id format memo still match when the router emits
    the bare UUID — the suffix-LIKE branch catches them."""
    cust = "cust-id-lookup-prefix"
    bare = "8bcb1234-1111-2222-3333-444444444444"
    await _seed_session_doc(
        cust,
        doc_id=f"linear:{cust}:issue:{bare}",
        source_id=f"issue:{bare}",
    )

    hits = await id_lookup_search(cust, [bare], temporal=TemporalSpec())

    assert len(hits) == 1
    assert hits[0].doc_id == f"linear:{cust}:issue:{bare}"


@pytest.mark.asyncio
async def test_id_lookup_skips_non_candidate_inputs(live_db) -> None:
    """Plain words like service slugs never trigger a SQL pass — saves
    a query and keeps the retriever scoped to stable identifiers."""
    cust = "cust-id-lookup-skip"
    await _seed_session_doc(
        cust,
        doc_id=f"claude_code:{cust}:aaaa1111-1111-1111-1111-111111111111",
        source_id="aaaa1111-1111-1111-1111-111111111111",
    )

    hits = await id_lookup_search(cust, ["prbe-backend", "auth"], temporal=TemporalSpec())

    assert hits == []
