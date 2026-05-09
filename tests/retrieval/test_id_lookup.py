"""Unit tests for the exact-id retriever, BM25 OR-of-tokens query builder,
and post-fusion pin pass — plus a live-DB integration test for the
id_lookup SQL.

The pure helpers (`is_lookup_candidate`, `_build_or_tsquery_string`,
`_pin_id_lookup_matches`) cover the gates and reordering that wrap the
real SQL pass. The integration test exercises the SELECT/JOIN/LIKE-ANY
shape so kind filtering, DISTINCT ON ordering, and the source_id-prefix
fallback don't silently regress.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from services.retrieval.retrievers.bm25 import _build_or_tsquery_string
from services.retrieval.retrievers.id_lookup import id_lookup_search, is_lookup_candidate
from services.retrieval.search_pipeline import _pin_id_lookup_matches
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


# ---- _pin_id_lookup_matches + _inject_id_lookup_hits -----------------------


@dataclass
class _StubFused:
    chunk_id: str
    doc_id: str
    doc_version: int = 1
    source_system: str = "claude_code"
    source_url: str = "https://example/x"
    title: str | None = "t"
    content: str = "c"
    score: float = 0.5
    author_id: str | None = None
    retriever_scores: dict | None = None
    kind: str = "content"
    created_at: datetime = datetime(2026, 5, 7, tzinfo=UTC)
    updated_at: datetime = datetime(2026, 5, 7, tzinfo=UTC)


@dataclass
class _StubIdHit:
    chunk_id: str
    doc_id: str
    doc_version: int = 1
    source_system: str = "claude_code"
    source_url: str = "https://example/x"
    title: str | None = "t"
    content: str = "c"
    author_id: str | None = None
    score: float = 1.0
    kind: str = "content"
    created_at: datetime = datetime(2026, 5, 7, tzinfo=UTC)
    updated_at: datetime = datetime(2026, 5, 7, tzinfo=UTC)


def test_pin_no_op_when_no_id_hits() -> None:
    fused = [_StubFused(chunk_id="c1", doc_id="d1"), _StubFused(chunk_id="c2", doc_id="d2")]
    out = _pin_id_lookup_matches(fused, [])
    assert [f.doc_id for f in out] == ["d1", "d2"]


def test_pin_floats_matched_doc_to_top() -> None:
    """When id_lookup hit a doc that's also in fused, it goes to position 0
    regardless of its prior fused rank."""
    fused = [
        _StubFused(chunk_id="c1", doc_id="d1"),
        _StubFused(chunk_id="c2", doc_id="d2"),
        _StubFused(chunk_id="c3", doc_id="d3"),
    ]
    id_hits = [_StubIdHit(chunk_id="c3", doc_id="d3")]
    out = _pin_id_lookup_matches(fused, id_hits)
    assert [f.doc_id for f in out] == ["d3", "d1", "d2"]


def test_pin_preserves_id_lookup_order_for_multiple_matches() -> None:
    """Multiple id_lookup matches keep the order id_lookup_search returned
    them in — caller can rely on the SQL's DISTINCT ON ordering."""
    fused = [
        _StubFused(chunk_id="c1", doc_id="d1"),
        _StubFused(chunk_id="c2", doc_id="d2"),
        _StubFused(chunk_id="c3", doc_id="d3"),
    ]
    id_hits = [
        _StubIdHit(chunk_id="c3", doc_id="d3"),
        _StubIdHit(chunk_id="c1", doc_id="d1"),
    ]
    out = _pin_id_lookup_matches(fused, id_hits)
    assert [f.doc_id for f in out] == ["d3", "d1", "d2"]


def test_pin_skips_id_hits_not_present_in_fused() -> None:
    """If an id_lookup-matched doc was filtered out by ACL/dedupe before
    pin sees it, pin silently skips that doc (no synthesis here — that
    happens upstream in _inject_id_lookup_hits before ACL)."""
    fused = [_StubFused(chunk_id="c1", doc_id="d1")]
    id_hits = [_StubIdHit(chunk_id="c-missing", doc_id="d-missing")]
    out = _pin_id_lookup_matches(fused, id_hits)
    assert [f.doc_id for f in out] == ["d1"]


def test_pin_dedupes_repeated_id_hits() -> None:
    """Defensive: id_lookup_search's DISTINCT ON should prevent doc dupes,
    but if duplicates slip through, the pin keeps each doc once."""
    fused = [
        _StubFused(chunk_id="c1", doc_id="d1"),
        _StubFused(chunk_id="c2", doc_id="d2"),
    ]
    id_hits = [
        _StubIdHit(chunk_id="c1", doc_id="d1"),
        _StubIdHit(chunk_id="c1", doc_id="d1"),
    ]
    out = _pin_id_lookup_matches(fused, id_hits)
    assert [f.doc_id for f in out] == ["d1", "d2"]


def test_inject_no_op_when_no_id_hits() -> None:
    from services.retrieval.search_pipeline import _inject_id_lookup_hits

    fused = [_StubFused(chunk_id="c1", doc_id="d1")]
    out = _inject_id_lookup_hits(fused, [])
    assert out is fused or [f.doc_id for f in out] == ["d1"]


def test_inject_appends_missing_id_hit_as_synthetic_fused() -> None:
    """The MCP-default top_k=5 → pool=10 case: the matched doc didn't
    survive fuse()'s cap. Inject must add it so dedupe/ACL/pin can act."""
    from services.retrieval.search_pipeline import _inject_id_lookup_hits

    fused = [_StubFused(chunk_id=f"c{i}", doc_id=f"d{i}") for i in range(10)]
    id_hits = [_StubIdHit(chunk_id="c-target", doc_id="d-target")]
    out = _inject_id_lookup_hits(fused, id_hits)
    assert "d-target" in [f.doc_id for f in out]
    target = next(f for f in out if f.doc_id == "d-target")
    assert target.score == 1.0
    assert target.retriever_scores == {"id_lookup": 1.0}
    # Synthetic doc carries one chunk with the same retriever signal.
    assert len(target.chunks) == 1
    assert target.chunks[0].chunk_id == "c-target"
    assert target.chunks[0].retriever_scores == {"id_lookup": 1.0}


def test_inject_skips_id_hit_already_in_fused() -> None:
    """Don't double-count docs that fuse already surfaced — they keep
    their fused entry (with its real combined score), pin still floats
    them via the dedup-by-doc-id path."""
    from services.retrieval.search_pipeline import _inject_id_lookup_hits

    fused = [_StubFused(chunk_id="c1", doc_id="d1", score=0.42)]
    id_hits = [_StubIdHit(chunk_id="c1", doc_id="d1")]
    out = _inject_id_lookup_hits(fused, id_hits)
    assert len(out) == 1
    assert out[0].score == 0.42  # keeps fused score, no synthetic overwrite


def test_inject_then_pin_round_trip_for_small_top_k() -> None:
    """Full flow simulation: fuse returns 10 docs without the matched one,
    inject appends it, pin floats it to position 0. Final top[0] is the
    matched doc — the behavior the MCP top_k=5 case needs."""
    from services.retrieval.search_pipeline import _inject_id_lookup_hits

    fused = [_StubFused(chunk_id=f"c{i}", doc_id=f"code{i}") for i in range(10)]
    id_hits = [_StubIdHit(chunk_id="c-session", doc_id="d-session")]

    fused = _inject_id_lookup_hits(fused, id_hits)
    pinned = _pin_id_lookup_matches(fused, id_hits)

    assert pinned[0].doc_id == "d-session"
    assert [f.doc_id for f in pinned[1:]] == [f"code{i}" for i in range(10)]
