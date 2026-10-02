"""Live chunks carry an open-ended last_seen_version (LIVE_CHUNK_LAST_SEEN).

Real Postgres, the real Phase A (`Normalizer._plan_chunks`, fed pre-chunked
pieces so every chunk's text is exact) and the real Phase B
(`_upsert_document` then `_apply_chunk_plan` inside one `with_tenant`
transaction, the calls `process_queue_row` makes per document). Only the
embedding network call is local: `GeminiEmbedder` without a key embeds by hash,
and a subclass records what reached it.

Pinned:
  * a reused chunk is written at most once -- a row still carrying an exact
    version (born before the sentinel, or written by an older pod) moves onto
    it on first reuse -- and never again; checked by `xmin` on both the
    version-bump path and the in-place (live session) path;
  * inserted and revived chunks get the sentinel, removed ones are capped to
    version - 1;
  * LATEST, AS_OF and ALL over three versions return exactly each version's
    content, and exactly what the pre-sentinel row shape returns;
  * removal is plan-authoritative: a live chunk the plan never saw (another
    worker's write between plan and apply, or the same document planned twice
    in one batch) is retired;
  * a chunk re-added after a CHUNKER_VERSION change ends live, and the next
    ingest reuses it instead of embedding it again;
  * the coding-agent metadata backfill script caps the row it replaces.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import asyncpg
import httpx
import pytest

from engine.ingest.chunker import ChunkPiece
from engine.ingest.handlers.base import ConnectorContext
from engine.ingest.normalizer import (
    Normalizer,
    _apply_chunk_plan,
    _ChunkPlan,
    _upsert_document,
)
from engine.retrieval.temporal import build_predicate, live_version_join
from engine.shared import db as db_module
from engine.shared.config import Settings
from engine.shared.constants import (
    CHUNKER_VERSION,
    LIVE_CHUNK_LAST_SEEN,
    DocType,
    SourceSystem,
)
from engine.shared.db import with_tenant
from engine.shared.embeddings import GeminiEmbedder
from engine.shared.models import (
    METADATA_CHUNK_INDEX,
    ACLSnapshot,
    Document,
    TemporalMode,
    TemporalSpec,
)

CUSTOMER = "cust-open-ended"


class _RecordingEmbedder(GeminiEmbedder):
    """The production embedder in its keyless (hash-vector) mode, recording
    every text it is asked to embed."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings=settings)
        self.texts: list[str] = []

    async def embed_documents(self, items):  # type: ignore[override]
        self.texts.extend(item.content for item in items)
        return await super().embed_documents(items)


class _NoStore:
    """Phase A and B never read the object store; the constructor needs one."""


def _normalizer() -> tuple[Normalizer, _RecordingEmbedder]:
    settings = Settings(environment="local", llm_gateway_url="", google_api_key="")
    embedder = _RecordingEmbedder(settings)
    ctx = ConnectorContext(settings=settings, http=httpx.AsyncClient())
    return Normalizer(ctx, store=_NoStore(), embedder=embedder), embedder  # type: ignore[arg-type]


async def _seed_customer() -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'test-hash') ON CONFLICT DO NOTHING
            """,
            CUSTOMER,
        )


def _doc(doc_id: str, tag: str, *, in_place: bool = False, title: str = "T") -> Document:
    """One incoming version. `tag` makes the content hash differ per version
    (an identical hash is the no-op path, which writes nothing at all).
    `in_place` is the live-session shape: coalesce into the live version."""
    now = datetime.now(UTC)
    return Document(
        doc_id=doc_id,
        customer_id=CUSTOMER,
        source_system=SourceSystem.SLACK,
        source_id=doc_id,
        source_url=f"https://example.invalid/{doc_id}",
        doc_type="slack.message",
        content_hash=f"doc-{doc_id}-{tag}",
        title=title,
        body_preview="preview",
        created_at=now,
        updated_at=now,
        valid_from=now,
        ingested_at=now,
        acl=ACLSnapshot(principals=[], captured_at=now),
        metadata={"session_complete": False} if in_place else {},
        coalesce_into_live=in_place,
    )


def _pieces(contents: list[str]) -> list[ChunkPiece]:
    return [ChunkPiece(chunk_index=i, content=c, token_count=2) for i, c in enumerate(contents)]


def _meta(text: str | None) -> ChunkPiece | None:
    if text is None:
        return None
    return ChunkPiece(chunk_index=METADATA_CHUNK_INDEX, content=text, token_count=2)


async def _plan(
    n: Normalizer, doc: Document, contents: list[str], meta: str | None = None
) -> _ChunkPlan:
    return await n._plan_chunks(CUSTOMER, doc, _pieces(contents), _meta(meta))


async def _apply(*pairs: tuple[Document, _ChunkPlan]) -> None:
    """Phase B for one or more documents in ONE transaction, as a batch is."""
    async with with_tenant(CUSTOMER) as conn:
        for doc, plan in pairs:
            assert await _upsert_document(conn, doc)
            await _apply_chunk_plan(conn, doc, plan)


async def _ingest(
    n: Normalizer, doc: Document, contents: list[str], meta: str | None = None
) -> Document:
    await _apply((doc, await _plan(n, doc, contents, meta)))
    return doc


async def _rows(doc_id: str) -> dict[str, Any]:
    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT content, first_seen_version, last_seen_version, valid_to,
                   chunker_version, xmin::text AS xmin
            FROM chunks WHERE customer_id = $1 AND doc_id = $2
            """,
            CUSTOMER,
            doc_id,
        )
    return {r["content"]: r for r in rows}


async def _retrieve(doc_id: str, spec: TemporalSpec) -> set[tuple[int, str]]:
    """(version, content) pairs through the retrievers' own temporal
    predicate and version join (engine/retrieval/temporal.py)."""
    pred = build_predicate(spec, "d", "c", 3)
    async with db_module.raw_conn() as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.version, c.content
            FROM documents d
            JOIN chunks c ON c.customer_id = d.customer_id AND c.doc_id = d.doc_id
            WHERE d.customer_id = $1 AND d.doc_id = $2
              {pred.doc_sql} {pred.chunk_sql}
              {live_version_join("d", "c")}
            """,
            CUSTOMER,
            doc_id,
            *pred.params,
        )
    return {(r["version"], r["content"]) for r in rows}


async def _mark_legacy(doc_id: str, content: str, version: int) -> None:
    """Give one live row the exact-version shape an older release wrote."""
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE chunks SET last_seen_version = $4"
            " WHERE customer_id = $1 AND doc_id = $2 AND content = $3",
            CUSTOMER,
            doc_id,
            content,
            version,
        )


# ---------------------------------------------------------------------------
# no write for an unchanged chunk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("in_place", [False, True], ids=["version-bump", "in-place-session"])
async def test_reused_chunks_are_written_at_most_once(live_db, in_place: bool) -> None:
    await _seed_customer()
    n, embedder = _normalizer()
    doc_id = f"slack:C1:reuse-{in_place}"

    await _ingest(n, _doc(doc_id, "1", in_place=in_place), ["alpha", "bravo"], "meta one")
    born = await _rows(doc_id)
    assert {c: r["last_seen_version"] for c, r in born.items()} == {
        "alpha": LIVE_CHUNK_LAST_SEEN,
        "bravo": LIVE_CHUNK_LAST_SEEN,
        "meta one": LIVE_CHUNK_LAST_SEEN,
    }

    # The shape a pre-sentinel pod left (last_seen == the live version) can no
    # longer exist: chunks_live_sentinel_chk (migration 0145) refuses it.
    with pytest.raises(asyncpg.exceptions.CheckViolationError, match="chunks_live_sentinel_chk"):
        await _mark_legacy(doc_id, "alpha", 1)
    before = await _rows(doc_id)

    v2 = await _ingest(
        n, _doc(doc_id, "2", in_place=in_place), ["alpha", "bravo", "charlie"], "meta one"
    )
    after_v2 = await _rows(doc_id)
    assert v2.version == (1 if in_place else 2)
    # Already open-ended: not written at all, content or metadata chunk.
    assert after_v2["alpha"]["last_seen_version"] == LIVE_CHUNK_LAST_SEEN
    assert after_v2["alpha"]["xmin"] == before["alpha"]["xmin"]
    assert after_v2["bravo"]["xmin"] == before["bravo"]["xmin"]
    assert after_v2["meta one"]["xmin"] == before["meta one"]["xmin"]
    assert after_v2["charlie"]["last_seen_version"] == LIVE_CHUNK_LAST_SEEN

    embedder.texts.clear()
    await _ingest(
        n,
        _doc(doc_id, "3", in_place=in_place),
        ["alpha", "bravo", "charlie", "delta"],
        "meta one",
    )
    after_v3 = await _rows(doc_id)
    for content in ("alpha", "bravo", "charlie", "meta one"):
        assert after_v3[content]["xmin"] == after_v2[content]["xmin"], (
            f"{content!r} was rewritten by a re-ingest that did not change it"
        )
        assert after_v3[content]["valid_to"] is None
    assert len(embedder.texts) == 1 and embedder.texts[0].endswith("delta")


# ---------------------------------------------------------------------------
# sentinel on insert and revival, cap on removal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("in_place", "capped_to"), [(False, 1), (True, 0)], ids=["version-bump", "in-place-session"]
)
async def test_removed_chunk_is_capped_and_a_revived_one_reopens(
    live_db, in_place: bool, capped_to: int
) -> None:
    await _seed_customer()
    n, _ = _normalizer()
    doc_id = f"slack:C1:revive-{in_place}"

    await _ingest(n, _doc(doc_id, "1", in_place=in_place), ["alpha", "bravo"])
    v2 = await _ingest(n, _doc(doc_id, "2", in_place=in_place), ["alpha"])
    removed = (await _rows(doc_id))["bravo"]
    assert removed["valid_to"] is not None
    # version - 1: on the bump path the last version that held it; on the
    # in-place path an empty range (first_seen 1 > 0), as before the sentinel.
    assert removed["last_seen_version"] == capped_to == v2.version - 1

    v3 = await _ingest(n, _doc(doc_id, "3", in_place=in_place), ["alpha", "bravo"])
    revived = (await _rows(doc_id))["bravo"]
    assert revived["valid_to"] is None
    assert revived["last_seen_version"] == LIVE_CHUNK_LAST_SEEN
    assert await _retrieve(doc_id, TemporalSpec(mode=TemporalMode.LATEST)) == {
        (v3.version, "alpha"),
        (v3.version, "bravo"),
    }


# ---------------------------------------------------------------------------
# retrieval: each version's content, identical to the pre-sentinel row shape
# ---------------------------------------------------------------------------


async def test_three_versions_retrieve_exactly_what_they_contained(live_db) -> None:
    await _seed_customer()
    n, _ = _normalizer()
    doc_id = "slack:C1:three-versions"
    contents = {
        1: (["alpha", "bravo", "charlie"], "meta one"),
        2: (["alpha", "charlie", "delta"], "meta one"),  # -bravo +delta
        3: (["alpha", "delta", "echo"], "meta two"),  # -charlie +echo, retitled
    }
    as_of: dict[int, datetime] = {}
    for version, (body, meta) in contents.items():
        doc = await _ingest(n, _doc(doc_id, str(version)), body, meta)
        assert doc.version == version
        await asyncio.sleep(0.02)
        as_of[version] = datetime.now(UTC)
        await asyncio.sleep(0.02)

    rows = await _rows(doc_id)
    assert {c: (r["first_seen_version"], r["last_seen_version"]) for c, r in rows.items()} == {
        "alpha": (1, LIVE_CHUNK_LAST_SEEN),
        "bravo": (1, 1),
        "charlie": (1, 2),
        "delta": (2, LIVE_CHUNK_LAST_SEEN),
        "echo": (3, LIVE_CHUNK_LAST_SEEN),
        "meta one": (1, 2),
        "meta two": (3, LIVE_CHUNK_LAST_SEEN),
    }

    def held(version: int) -> set[tuple[int, str]]:
        body, meta = contents[version]
        return {(version, c) for c in [*body, meta]}

    specs = {
        "latest": TemporalSpec(mode=TemporalMode.LATEST),
        "all": TemporalSpec(mode=TemporalMode.ALL),
        **{
            f"as_of_v{v}": TemporalSpec(mode=TemporalMode.AS_OF, as_of=t)
            for v, t in as_of.items()
        },
    }
    expected = {
        "latest": held(3),
        "all": held(1) | held(2) | held(3),
        "as_of_v1": held(1),
        "as_of_v2": held(2),
        "as_of_v3": held(3),
    }
    now_results = {name: await _retrieve(doc_id, spec) for name, spec in specs.items()}
    assert now_results == expected

    # The rows as the pre-sentinel code left them (a live chunk carrying the
    # current version) are refused since migration 0145, which is what lets
    # BM25 read "live" off the indexed last_seen_version.
    async with db_module.raw_conn() as conn:
        with pytest.raises(asyncpg.exceptions.CheckViolationError, match="chunks_live_sentinel_chk"):
            await conn.execute(
                "UPDATE chunks SET last_seen_version = 3"
                " WHERE customer_id = $1 AND doc_id = $2 AND last_seen_version = $3",
                CUSTOMER,
                doc_id,
                LIVE_CHUNK_LAST_SEEN,
            )


# ---------------------------------------------------------------------------
# removal is plan-authoritative
# ---------------------------------------------------------------------------


async def test_removal_retires_a_live_chunk_another_worker_wrote_after_the_plan(
    live_db,
) -> None:
    await _seed_customer()
    n, _ = _normalizer()
    doc_id = "slack:C1:cross-worker"
    await _ingest(n, _doc(doc_id, "1"), ["alpha"])

    # This worker reads and embeds (Phase A) ...
    mine = _doc(doc_id, "mine")
    my_plan = await _plan(n, mine, ["alpha", "bravo"])
    assert my_plan.removed_hashes == set(), "the plan saw nothing to remove"

    # ... while another worker writes a version holding `xray` ...
    await _ingest(n, _doc(doc_id, "other"), ["alpha", "xray"])
    assert (await _rows(doc_id))["xray"]["valid_to"] is None

    # ... and then this worker writes (Phase B).
    await _apply((mine, my_plan))

    assert mine.version == 3
    xray = (await _rows(doc_id))["xray"]
    assert xray["valid_to"] is not None, "a chunk the plan never saw must not stay live"
    assert xray["last_seen_version"] == 2
    assert await _retrieve(doc_id, TemporalSpec(mode=TemporalMode.LATEST)) == {
        (3, "alpha"),
        (3, "bravo"),
    }
    assert {c for v, c in await _retrieve(doc_id, TemporalSpec(mode=TemporalMode.ALL))
            if v == 2} == {"alpha", "xray"}


async def test_same_document_twice_in_one_batch_ends_with_the_second(live_db) -> None:
    await _seed_customer()
    n, _ = _normalizer()
    doc_id = "slack:C1:twice"
    await _ingest(n, _doc(doc_id, "1"), ["alpha"])

    # Both plans are read against version 1, as a batch's Phase A is.
    first, second = _doc(doc_id, "a"), _doc(doc_id, "b")
    first_plan = await _plan(n, first, ["alpha", "yankee"])
    second_plan = await _plan(n, second, ["alpha", "zulu"])
    await _apply((first, first_plan), (second, second_plan))

    assert (first.version, second.version) == (2, 3)
    rows = await _rows(doc_id)
    assert rows["yankee"]["valid_to"] is not None
    assert rows["yankee"]["last_seen_version"] == 2
    assert await _retrieve(doc_id, TemporalSpec(mode=TemporalMode.LATEST)) == {
        (3, "alpha"),
        (3, "zulu"),
    }


# ---------------------------------------------------------------------------
# CHUNKER_VERSION change
# ---------------------------------------------------------------------------


async def test_chunk_readded_after_a_chunker_version_change_ends_live(live_db) -> None:
    """Same content, older chunker: Phase A puts the hash in BOTH the removed
    (stale) set and the added set. Step 2 revives the row through ON CONFLICT;
    retiring by the removed set then closed it again, so the version that holds
    the content served none of it."""
    await _seed_customer()
    n, embedder = _normalizer()
    doc_id = "slack:C1:rechunked"
    await _ingest(n, _doc(doc_id, "1"), ["alpha", "bravo"])
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "UPDATE chunks SET chunker_version = 'naive-v0'"
            " WHERE customer_id = $1 AND doc_id = $2",
            CUSTOMER,
            doc_id,
        )

    embedder.texts.clear()
    v2 = _doc(doc_id, "2")
    plan = await _plan(n, v2, ["alpha", "bravo"])
    assert plan.reused_content_hashes == set()
    assert len(plan.removed_hashes) == 2 and len(plan.added_pieces) == 2
    await _apply((v2, plan))
    assert len(embedder.texts) == 2

    rows = await _rows(doc_id)
    for content in ("alpha", "bravo"):
        assert rows[content]["valid_to"] is None, f"{content!r} was retired right after re-adding"
        assert rows[content]["last_seen_version"] == LIVE_CHUNK_LAST_SEEN
        assert rows[content]["chunker_version"] == CHUNKER_VERSION
    assert await _retrieve(doc_id, TemporalSpec(mode=TemporalMode.LATEST)) == {
        (v2.version, "alpha"),
        (v2.version, "bravo"),
    }

    # And it is current now: the next ingest reuses it rather than paying to
    # embed it again.
    embedder.texts.clear()
    await _ingest(n, _doc(doc_id, "3"), ["alpha", "bravo", "charlie"])
    assert len(embedder.texts) == 1 and embedder.texts[0].endswith("charlie")


# ---------------------------------------------------------------------------
# scripts/backfill_cc_metadata_chunks.py
# ---------------------------------------------------------------------------


async def test_metadata_backfill_caps_the_row_it_replaces(live_db) -> None:
    from scripts import backfill_cc_metadata_chunks as backfill

    await _seed_customer()
    doc_id = "claude_code:cust-open-ended:82861aa0"
    now = datetime.now(UTC)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (customer_id, doc_id, version, source_system, source_id,
                                   source_url, doc_type, content_hash, created_at,
                                   updated_at, valid_from, ingested_at, acl, title,
                                   body_preview, metadata)
            VALUES ($1, $2, 3, 'claude_code', '82861aa0', 'https://x', $3, 'h',
                    $4, $4, $4, $4, '{}'::jsonb, 'Claude Code session 82861aa0',
                    'USER: hi', '{}'::jsonb)
            """,
            CUSTOMER,
            doc_id,
            DocType.CLAUDE_CODE_SESSION.value,
            now,
        )
        await conn.execute(
            """
            INSERT INTO chunks (customer_id, chunk_id, doc_id, chunk_index, content,
                                content_hash, token_count, first_seen_version,
                                last_seen_version, kind)
            VALUES ($1, $2, $3, $4, 'old metadata text', 'old-hash', 3, 1, $5, 'metadata')
            """,
            CUSTOMER,
            f"{doc_id}:m_old",
            doc_id,
            METADATA_CHUNK_INDEX,
            LIVE_CHUNK_LAST_SEEN,
        )

    [row] = await backfill._list_live_agent_docs(CUSTOMER, SourceSystem.CLAUDE_CODE)
    _, embedder = _normalizer()
    assert await backfill._process_doc(asyncio.Semaphore(1), embedder, row, False) == "updated"

    async with db_module.raw_conn() as conn:
        rows = {
            r["content_hash"]: r
            for r in await conn.fetch(
                "SELECT content_hash, first_seen_version, last_seen_version, valid_to"
                " FROM chunks WHERE customer_id = $1 AND doc_id = $2",
                CUSTOMER,
                doc_id,
            )
        }
    old = rows.pop("old-hash")
    [new] = rows.values()
    assert old["valid_to"] is not None
    assert old["last_seen_version"] == 2, "replaced in place at version 3: capped to 3 - 1"
    assert new["valid_to"] is None
    assert (new["first_seen_version"], new["last_seen_version"]) == (3, LIVE_CHUNK_LAST_SEEN)


async def test_a_real_delete_leaves_chunks_the_tombstone_purge_deletes(live_db) -> None:
    """End to end (review of #604): a deletion through the real plan/apply caps
    every live chunk below the tombstone, and the purge's own DELETE statement
    removes them all. An open-ended chunk here would outlive the purge."""
    import scripts.cron_tombstone_purge as purge

    await _seed_customer()
    n, _ = _normalizer()
    doc_id = "slack:C1:deleted-end-to-end"
    await _ingest(n, _doc(doc_id, "1"), ["alpha", "bravo"], meta="meta v1")
    await _ingest(n, _doc(doc_id, "2"), ["alpha", "charlie"], meta="meta v1")
    tomb = _doc(doc_id, "3")
    tomb.deleted_at = datetime.now(UTC)
    await _ingest(n, tomb, [])

    rows = await _rows(doc_id)
    assert rows and all(r["valid_to"] is not None for r in rows.values())
    assert all(r["last_seen_version"] < tomb.version for r in rows.values())

    async with db_module.raw_conn() as conn:
        out = await conn.fetchrow(
            purge._DELETE_CHUNKS_SQL, CUSTOMER, [doc_id], [tomb.version], 1000,
            LIVE_CHUNK_LAST_SEEN,
        )
    assert out["deleted"] == len(rows)
    assert await _rows(doc_id) == {}


async def test_an_old_pod_close_is_refused(live_db) -> None:
    """A pod on the pre-sentinel code (mid-rollout, or after a rollback) closed
    chunks with valid_to only, leaving LIVE_CHUNK_LAST_SEEN on a closed row.
    Since migration 0145 that write is refused (chunks_live_sentinel_chk), so
    the purge's handling of such rows has nothing left to find; the live
    chunk stays untouched."""
    import scripts.cron_tombstone_purge as purge

    await _seed_customer()
    n, _ = _normalizer()
    doc_id = "slack:C1:old-pod-close"
    await _ingest(n, _doc(doc_id, "1"), ["alpha", "bravo"])
    async with db_module.raw_conn() as conn:
        # The old-pod close: valid_to only, sentinel kept.
        with pytest.raises(asyncpg.exceptions.CheckViolationError, match="chunks_live_sentinel_chk"):
            await conn.execute(
                "UPDATE chunks SET valid_to = now() WHERE customer_id = $1 AND doc_id = $2"
                " AND content = 'alpha'",
                CUSTOMER,
                doc_id,
            )
        out = await conn.fetchrow(
            purge._DELETE_CHUNKS_SQL, CUSTOMER, [doc_id], [2], 1000, LIVE_CHUNK_LAST_SEEN
        )
    assert out["deleted"] == 0
    left = await _rows(doc_id)
    assert set(left) == {"alpha", "bravo"}, "live open-ended chunks are not the purge's to take"
