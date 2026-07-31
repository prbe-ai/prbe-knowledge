"""BM25 can match a document's TITLE, not only its chunk content (0099).

Before this, `chunks.content_tsv` was the only thing BM25 read. It is a
GENERATED column, and a generation expression may only reference its own row,
so it could never reach `documents.title` -- the two live in different tables.
A file named `model.ckpt`, or a PR titled "Fix the retry loop", was findable by
keyword ONLY if those words also appeared in the body.

Three invariants here:

1. A title-only match surfaces the document at all (the regression).
2. It surfaces exactly ONE chunk -- the `chunk_index = 0` representative.
   A title belongs to the document, not to any chunk, so matching on it without
   that restriction would return every chunk of the document at an identical
   score and a long document would bury everything else.
3. Filename tokenization actually aligns. Postgres' `english` parser emits
   `model.ckpt` as ONE `file` lexeme while the retriever splits queries on
   alphanumeric runs, so a verbatim-only index would have missed exactly the
   case this exists for. The generated column indexes the punctuation-flattened
   form alongside the verbatim one.

Scores are asserted as ordering and non-zero-ness, never as exact values --
ts_rank_cd's adjacent scores shift with Postgres point releases and stemmer
changes, and pinning them produces flaky failures on drift that is not a bug.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from engine.retrieval.retrievers.bm25 import bm25_search
from engine.shared.db import raw_conn
from engine.shared.models import TemporalSpec

_NOW = datetime.now(UTC)

# Two documents. Neither BODY mentions the checkpoint file; only one TITLE does.
_DOCS: list[tuple[str, str, list[str]]] = [
    (
        "doc-title-hit",
        "checkpoints/model.ckpt",
        # Three chunks, none containing "model" or "ckpt". Only chunk_index 0
        # may surface on a title-only match.
        [
            "binary artifact produced by the nightly training job",
            "second chunk of the same artifact record",
            "third chunk of the same artifact record",
        ],
    ),
    (
        "doc-body-hit",
        "unrelated notes",
        ["the model was rotated and the ckpt written to disk"],
    ),
]


def _doc(customer_id: str, suffix: str) -> str:
    """Doc ids are namespaced per customer: `documents` is keyed on
    (doc_id, version), so four tests sharing a literal id would collide on
    the first insert and silently reuse another test's corpus."""
    return f"{customer_id}-{suffix}"


async def _seed(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        for suffix, title, chunks in _DOCS:
            doc_id = _doc(customer_id, suffix)
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
                    'manual_upload', $1, 'https://prbe.ai/' || $1,
                    'raw_source', 'manual_upload.note', 'text/plain',
                    'h-' || $1, $3, 100, 0,
                    $4, $4, $4, $4, '{}'::jsonb
                )
                """,
                doc_id, customer_id, title, _NOW,
            )
            for idx, body in enumerate(chunks):
                await conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, customer_id,
                        chunk_index, content, content_hash, token_count, kind,
                        embedding, first_seen_version, last_seen_version
                    ) VALUES (
                        $1, $2, $3, $4, $5, $6, 5, 'content',
                        array_fill(0::real, ARRAY[3072])::halfvec,
                        1, 1
                    )
                    """,
                    f"{doc_id}-c{idx}", doc_id, customer_id, idx, body,
                    f"chash-{doc_id}-{idx}",
                )


@pytest.mark.asyncio
async def test_title_only_match_surfaces_the_document(live_db) -> None:
    """The regression: a filename in the TITLE is findable even though no
    chunk body contains it."""
    cust = "cust-bm25-title-title-surfaces"
    await _seed(cust)

    hits = await bm25_search(
        cust, "model.ckpt", top_k=20, temporal=TemporalSpec(mode="all")
    )
    doc_ids = {h.doc_id for h in hits}
    assert _doc(cust, "doc-title-hit") in doc_ids, (
        "a title-only match did not surface; BM25 is ignoring documents.title_tsv"
    )


@pytest.mark.asyncio
async def test_title_only_match_returns_one_representative_chunk(live_db) -> None:
    """A title belongs to the DOCUMENT. Matching on it must not return every
    chunk of that document at an identical score."""
    cust = "cust-bm25-title-one-chunk"
    await _seed(cust)

    hits = await bm25_search(
        cust, "model.ckpt", top_k=20, temporal=TemporalSpec(mode="all")
    )
    title_hits = [h for h in hits if h.doc_id == _doc(cust, "doc-title-hit")]
    assert len(title_hits) == 1, (
        f"expected one representative chunk, got {len(title_hits)} -- a long "
        "document would bury every other result"
    )
    assert title_hits[0].chunk_id == f'{_doc(cust, "doc-title-hit")}-c0'


@pytest.mark.asyncio
async def test_title_match_outranks_a_body_only_match(live_db) -> None:
    """title_tsv is setweight'd 'A' and content_tsv is unweighted ('D'), so a
    title hit is worth ~10x a body hit under ts_rank_cd's default weights.
    Someone typing a filename wants the file, not a doc that mentions it."""
    cust = "cust-bm25-title-outranks"
    await _seed(cust)

    hits = await bm25_search(
        cust, "model.ckpt", top_k=20, temporal=TemporalSpec(mode="all")
    )
    by_doc = {h.doc_id: h for h in hits}
    title_doc, body_doc = _doc(cust, "doc-title-hit"), _doc(cust, "doc-body-hit")
    assert {title_doc, body_doc} <= set(by_doc), by_doc.keys()
    assert by_doc[title_doc].score > by_doc[body_doc].score


@pytest.mark.asyncio
async def test_body_only_matches_still_return_every_matching_chunk(live_db) -> None:
    """The chunk_index restriction applies ONLY to title-only matches. A chunk
    that matches on its own content is unaffected."""
    cust = "cust-bm25-title-body-chunks"
    await _seed(cust)

    hits = await bm25_search(
        cust, "rotated", top_k=20, temporal=TemporalSpec(mode="all")
    )
    assert any(h.doc_id == _doc(cust, "doc-body-hit") for h in hits)
