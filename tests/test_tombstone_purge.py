"""Tombstone purge: a customer's deletion finishes in the engine, and nothing else goes.

Every purge here runs through a pool connected as a NON-superuser role, the way
production connects as `app`: a superuser bypasses RLS, so a delete that forgot
the tenant GUC would pass these tests against the fixture's own superuser pool
while matching zero rows in production. Seeding and assertions use the
superuser pool, which sees every row.

The boundaries pinned:
  * an old tombstone loses every version, its chunks, every row naming it by
    id, its graph node (and the nodes only it held) and its raw payloads;
  * a young tombstone, a live document's history, a re-created document, and
    another tenant's same-named rows are untouched;
  * a held or non-active tenant is untouched, including when the hold lands in
    the middle of a run -- between groups and between batches of one document;
  * a raw payload that cannot be deleted keeps the rows (they are the retry
    handle), and an in-flight re-create keeps both.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from urllib.parse import urlsplit, urlunsplit

import pytest
import pytest_asyncio

import scripts.cron_tombstone_purge as purge
from engine.shared import db as db_module
from engine.shared import storage as storage_module
from engine.shared.config import Settings
from engine.shared.constants import (
    BACKUP_TAIL_DAYS,
    DELETION_DEADLINE_DAYS,
    TOMBSTONE_PURGE_DAYS,
    EdgeType,
    NodeLabel,
    SourceSystem,
)
from engine.shared.custom_ingest import (
    custom_ingest_doc_id,
    document_event_prefix,
    document_payload_key,
)
from engine.shared.exceptions import StorageUnavailable
from engine.shared.legal_hold import purge_blocked_reason

APP_ROLE = "prbe_tombstone_purge_app"
APP_PASSWORD = "tombstone-purge-test"
# A ':' in the key exercises the doc_id encoding the raw prefix is derived from.
SOURCE_KEY = "research-os:runs"
NOW = datetime.now(UTC)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def app_settings(live_db, settings: Settings) -> Settings:
    """Settings whose DSN is a LOGIN role with no superuser and no BYPASSRLS."""
    async with db_module.raw_conn() as conn:
        await conn.execute(
            f"""
            DO $$
            BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{APP_ROLE}') THEN
                    CREATE ROLE {APP_ROLE} LOGIN PASSWORD '{APP_PASSWORD}'
                        NOSUPERUSER NOBYPASSRLS;
                END IF;
            END $$;
            """
        )
        await conn.execute(f"GRANT USAGE ON SCHEMA public TO {APP_ROLE}")
        await conn.execute(f"GRANT ALL ON ALL TABLES IN SCHEMA public TO {APP_ROLE}")
        await conn.execute(f"GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {APP_ROLE}")
    parts = urlsplit(settings.database_url)
    netloc = f"{APP_ROLE}:{APP_PASSWORD}@{parts.hostname}:{parts.port or 5432}"
    dsn = urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return settings.model_copy(update={"database_url": dsn})


@pytest_asyncio.fixture
async def bucket_for(live_db) -> AsyncIterator:
    """Create (and afterwards empty and remove) a tenant's MinIO bucket."""
    storage_module.reset_store()
    storage_module._reset_bucket_cache_for_tests()
    store = storage_module.get_store()
    created: list[str] = []

    async def make(customer_id: str) -> str:
        bucket = await store.bucket_for(customer_id)
        await store.ensure_bucket(bucket)
        created.append(bucket)
        return bucket

    yield make
    for bucket in created:
        await store.delete_bucket_recursive(bucket)


async def _run(app: Settings, settings: Settings, **kwargs) -> int:
    """run_once through the app-role pool, then hand the superuser pool back."""
    await db_module.close_pool()
    await db_module.init_pool(app)
    try:
        return await purge.run_once(**kwargs)
    finally:
        await db_module.close_pool()
        await db_module.init_pool(settings)


# ---------------------------------------------------------------------------
# seeding
# ---------------------------------------------------------------------------


async def _tenant(customer_id: str, *, status: str = "active", metadata=None) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash, status, metadata)
            VALUES ($1, $1, 'h', $2, $3::jsonb)
            """,
            customer_id,
            status,
            json.dumps(metadata or {}),
        )


async def _doc(
    customer_id: str,
    doc_id: str,
    *,
    versions: int,
    tombstone_days: float | None,
    source_system: str = SourceSystem.SLACK.value,
    source_id: str | None = None,
    metadata: dict | None = None,
    closed: bool = False,
    tombstone_at: int | None = None,
) -> None:
    """Versions 1..n. Every version but the last is superseded. The last is
    live, or -- with `tombstone_days` -- a tombstone that old. `closed` also
    closes the tombstone row (the code-graph disconnect shape). `tombstone_at`
    makes that one version a tombstone instead of the last (a re-created doc)."""
    tomb = tombstone_at if tombstone_at is not None else versions
    async with db_module.raw_conn() as conn:
        for v in range(1, versions + 1):
            last = v == versions
            is_tomb = tombstone_days is not None and v == tomb
            deleted_at = NOW - timedelta(days=tombstone_days) if is_tomb else None
            valid_to = None if last and not closed else NOW - timedelta(days=40 - v)
            if is_tomb and closed:
                valid_to = deleted_at  # closed by the same UPDATE that tombstoned it
            elif is_tomb and not last:
                valid_to = deleted_at + timedelta(hours=1)  # re-created an hour later
            await conn.execute(
                """
                INSERT INTO documents (customer_id, doc_id, version, source_system,
                                       source_id, source_url, doc_type, content_hash,
                                       created_at, updated_at, valid_from, valid_to,
                                       deleted_at, acl, title, body_preview, metadata)
                VALUES ($1, $2, $3, $4, $5, 'https://x', 't', $6,
                        $7, $7, $7, $8, $9, '{}'::jsonb, $10, $11, $12::jsonb)
                """,
                customer_id,
                doc_id,
                v,
                source_system,
                source_id or doc_id,
                f"h-{doc_id}-{v}",
                NOW - timedelta(days=60 - v),
                valid_to,
                deleted_at,
                None if deleted_at else f"title of {doc_id}",
                None if deleted_at else "a body preview the customer deleted",
                json.dumps(metadata or {}),
            )


async def _chunk(
    customer_id: str, doc_id: str, chunk_id: str, first: int, last: int, *, live: bool
) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO chunks (customer_id, chunk_id, doc_id, chunk_index, content,
                                content_hash, token_count, first_seen_version,
                                last_seen_version, valid_to)
            VALUES ($1, $2, $3, 0, 'content', $2, 1, $4, $5, $6)
            """,
            customer_id,
            chunk_id,
            doc_id,
            first,
            last,
            None if live else NOW - timedelta(days=9),
        )


async def _node(customer_id: str, label: str, canonical_id: str, *, source: str,
                degree: int = 0) -> int:
    async with db_module.raw_conn() as conn:
        node_id = await conn.fetchval(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES ($1, $2, $3, '{"name": "a name the customer deleted"}'::jsonb, $4)
            RETURNING node_id
            """,
            customer_id,
            label,
            canonical_id,
            degree,
        )
        await conn.execute(
            """
            INSERT INTO graph_node_provenance (node_id, customer_id, source_system)
            VALUES ($1, $2, $3)
            """,
            node_id,
            customer_id,
            source,
        )
        await conn.execute(
            "INSERT INTO node_post_write_queue (customer_id, node_id) VALUES ($1, $2)",
            customer_id,
            node_id,
        )
    return node_id


async def _edge(customer_id: str, edge_type: str, from_id: int, to_id: int) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (customer_id, edge_type, from_node_id, to_node_id)
            VALUES ($1, $2, $3, $4)
            """,
            customer_id,
            edge_type,
            from_id,
            to_id,
        )


async def _count(sql: str, *args) -> int:
    async with db_module.raw_conn() as conn:
        return int(await conn.fetchval(sql, *args))


async def _doc_rows(customer_id: str, doc_id: str) -> int:
    return await _count(
        "SELECT count(*) FROM documents WHERE customer_id = $1 AND doc_id = $2",
        customer_id,
        doc_id,
    )


async def _chunk_rows(customer_id: str, doc_id: str) -> int:
    return await _count(
        "SELECT count(*) FROM chunks WHERE customer_id = $1 AND doc_id = $2",
        customer_id,
        doc_id,
    )


async def _set_hold(customer_id: str, reason: object) -> None:
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            UPDATE customers
            SET metadata = metadata || jsonb_build_object('legal_hold', $2::jsonb)
            WHERE customer_id = $1
            """,
            customer_id,
            json.dumps(reason),
        )


# ---------------------------------------------------------------------------
# the purge
# ---------------------------------------------------------------------------


async def test_old_tombstone_is_purged_everywhere(app_settings, settings, bucket_for) -> None:
    cid = "t-purge"
    await _tenant(cid)
    store = storage_module.get_store()
    bucket = await bucket_for(cid)

    gone = custom_ingest_doc_id(cid, SOURCE_KEY, "run:1")
    kept = custom_ingest_doc_id(cid, SOURCE_KEY, "run:2")
    ci = SourceSystem.CUSTOM_INGEST.value
    gone_source_id = f"{SOURCE_KEY}:run:1"
    kept_source_id = f"{SOURCE_KEY}:run:2"
    await _doc(cid, gone, versions=3, tombstone_days=TOMBSTONE_PURGE_DAYS + 1,
               source_system=ci, source_id=gone_source_id)
    await _doc(cid, kept, versions=2, tombstone_days=None,
               source_system=ci, source_id=kept_source_id)
    for chunk_id, first, last in (("g1", 1, 1), ("g2", 1, 2), ("g-meta", 2, 2)):
        await _chunk(cid, gone, chunk_id, first, last, live=False)
    await _chunk(cid, kept, "k-old", 1, 1, live=False)
    await _chunk(cid, kept, "k-live", 1, 2, live=True)

    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO failed_chunks (customer_id, doc_id, doc_version, chunk_index,
                                       content_preview, error)
            VALUES ($1, $2, 2, 0, 'deleted words', 'embed failed')
            """,
            cid,
            gone,
        )
        await conn.execute(
            """
            INSERT INTO inferred_edges_queue (customer_id, anchor_doc_id, extractor_id)
            VALUES ($1, $2, 'inferred_edges:v1'), ($1, $3, 'inferred_edges:v1')
            """,
            cid,
            gone,
            kept,
        )
        for source_id in (gone_source_id, kept_source_id):
            for permission in ("read", "write"):
                await conn.execute(
                    """
                    INSERT INTO acl_snapshots (customer_id, source_system, principal_type,
                                               principal_id, resource_type, resource_id,
                                               permission, valid_from)
                    VALUES ($1, $2, 'workspace', $1, 'custom.document', $3, $4, now())
                    """,
                    cid,
                    ci,
                    source_id,
                    permission,
                )
        await conn.execute(
            """
            INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id,
                                       edge_type, from_label, from_canonical_id,
                                       to_label, to_canonical_id, source_system)
            VALUES ($1, 'Run', 'run:9', 'TOUCHES', 'Run', 'run:9', 'Document', $2, $3)
            """,
            cid,
            gone,
            ci,
        )
        await conn.execute(
            "INSERT INTO github_installations (customer_id, installation_id) VALUES ($1, 'i1')",
            cid,
        )
        await conn.execute(
            """
            INSERT INTO github_document_bindings (customer_id, installation_id, doc_id,
                                                  repository)
            VALUES ($1, 'i1', $2, 'acme/app')
            """,
            cid,
            gone,
        )

    gone_node = await _node(cid, NodeLabel.DOCUMENT.value, gone, source=ci)
    kept_node = await _node(cid, NodeLabel.DOCUMENT.value, kept, source=ci)
    run_node = await _node(cid, NodeLabel.RUN.value, "run:1", source=ci)
    author = await _node(cid, NodeLabel.PERSON.value, "person:alice", source=ci)
    project = await _node(cid, NodeLabel.PROJECT.value, "project:p", source=ci, degree=2)
    # Bob's only edge is to the deleted document too, but Slack also asserts
    # him, so he is not this deletion's to remove.
    slack_person = await _node(cid, NodeLabel.PERSON.value, "person:bob", source=ci)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            "INSERT INTO graph_node_provenance (node_id, customer_id, source_system)"
            " VALUES ($1, $2, 'slack')",
            slack_person,
            cid,
        )
    await _edge(cid, EdgeType.TOUCHES.value, run_node, gone_node)
    await _edge(cid, EdgeType.AUTHORED.value, author, gone_node)
    await _edge(cid, EdgeType.AUTHORED.value, slack_person, gone_node)
    await _edge(cid, EdgeType.TOUCHES.value, project, gone_node)
    await _edge(cid, EdgeType.TOUCHES.value, project, kept_node)

    gone_keys = [
        document_payload_key(cid, SOURCE_KEY, "run:1", h) for h in ("aaa", "bbb")
    ]
    kept_keys = [
        document_payload_key(cid, SOURCE_KEY, "run:2", "ccc"),
        # Event-addressed: many items per object, never attributable.
        f"raw/slack/{cid}/2026/09/01/evt-1.json",
    ]
    for key in gone_keys + kept_keys:
        await store.put(bucket, key, b"{}")

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows(cid, gone) == 0
    assert await _chunk_rows(cid, gone) == 0
    for table, column in (
        ("failed_chunks", "doc_id"),
        ("inferred_edges_queue", "anchor_doc_id"),
        ("github_document_bindings", "doc_id"),
    ):
        assert await _count(
            f"SELECT count(*) FROM {table} WHERE customer_id = $1 AND {column} = $2",
            cid,
            gone,
        ) == 0, table
    assert await _count(
        "SELECT count(*) FROM acl_snapshots WHERE customer_id = $1 AND resource_id = $2",
        cid,
        gone_source_id,
    ) == 0
    assert await _count(
        "SELECT count(*) FROM pending_edges WHERE customer_id = $1", cid
    ) == 0

    async with db_module.raw_conn() as conn:
        nodes = {
            r["node_id"]: r["degree"]
            for r in await conn.fetch(
                "SELECT node_id, degree FROM graph_nodes WHERE customer_id = $1", cid
            )
        }
    # The document's node, the run only it touched and the author who
    # wrote nothing else are gone; the shared project, the live document
    # and the person another source asserts stay.
    assert set(nodes) == {kept_node, project, slack_person}
    assert nodes[project] == 1  # recounted: its edge to the deleted doc went
    assert await _count(
        "SELECT count(*) FROM graph_edges WHERE customer_id = $1"
        " AND (from_node_id = $2 OR to_node_id = $2)",
        cid,
        gone_node,
    ) == 0
    assert await _count(
        "SELECT count(*) FROM graph_node_provenance WHERE node_id = ANY($1::bigint[])",
        [gone_node, run_node, author],
    ) == 0
    assert await _count(
        "SELECT count(*) FROM node_post_write_queue WHERE node_id = ANY($1::bigint[])",
        [gone_node, run_node, author],
    ) == 0

    # The live document keeps its whole history, chunks and side rows.
    assert await _doc_rows(cid, kept) == 2
    assert await _chunk_rows(cid, kept) == 2
    assert await _count(
        "SELECT count(*) FROM acl_snapshots WHERE customer_id = $1 AND resource_id = $2",
        cid,
        kept_source_id,
    ) == 2
    assert await _count(
        "SELECT count(*) FROM inferred_edges_queue WHERE anchor_doc_id = $1", kept
    ) == 1

    remaining = set(await store.list_keys(bucket, "raw/"))
    assert remaining == set(kept_keys)

    # Idempotent: a second run finds nothing and changes nothing.
    assert await _run(app_settings, settings) == purge.EXIT_OK
    assert set(await store.list_keys(bucket, "raw/")) == set(kept_keys)
    assert await _doc_rows(cid, kept) == 2


async def test_young_tombstone_is_untouched(app_settings, settings) -> None:
    cid = "t-young"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=2, tombstone_days=TOMBSTONE_PURGE_DAYS - 1)
    await _chunk(cid, "slack:C1:1.0", "c1", 1, 1, live=False)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows(cid, "slack:C1:1.0") == 2
    assert await _chunk_rows(cid, "slack:C1:1.0") == 1


async def test_live_and_recreated_documents_keep_their_history(app_settings, settings) -> None:
    """Superseded versions of a LIVE document are history the customer still
    has. A document deleted long ago and then re-created is live again, and its
    old tombstone version is part of that history too."""
    cid = "t-live"
    await _tenant(cid)
    await _doc(cid, "slack:C1:live", versions=3, tombstone_days=None)
    await _chunk(cid, "slack:C1:live", "old", 1, 1, live=False)
    await _doc(cid, "slack:C1:back", versions=3, tombstone_days=30, tombstone_at=2)
    await _chunk(cid, "slack:C1:back", "before-delete", 1, 1, live=False)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows(cid, "slack:C1:live") == 3
    assert await _chunk_rows(cid, "slack:C1:live") == 1
    assert await _doc_rows(cid, "slack:C1:back") == 3
    assert await _chunk_rows(cid, "slack:C1:back") == 1


async def test_code_graph_disconnect_shape_is_purged(app_settings, settings) -> None:
    """A repo disconnect tombstones in place and CLOSES the row, so the
    document has no live row at all. Its current version is still a tombstone."""
    cid = "t-codegraph"
    await _tenant(cid)
    doc_id = "code_graph:acme/app:src/x.py:f"
    await _doc(cid, doc_id, versions=2, tombstone_days=20, closed=True,
               source_system=SourceSystem.CODE_GRAPH.value)
    await _chunk(cid, doc_id, "cg", 1, 2, live=False)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows(cid, doc_id) == 0
    assert await _chunk_rows(cid, doc_id) == 0


async def test_other_tenants_same_ids_are_untouched(app_settings, settings) -> None:
    """Connector doc ids carry no tenant, so two tenants can hold the same one.
    Only the tenant whose copy is an old tombstone loses it."""
    doc_id = "slack:C1:1.0"
    await _tenant("t-a")
    await _tenant("t-b")
    await _doc("t-a", doc_id, versions=2, tombstone_days=10)
    await _doc("t-b", doc_id, versions=2, tombstone_days=None)
    for cid in ("t-a", "t-b"):
        await _chunk(cid, doc_id, "same-chunk", 1, 1, live=False)
        await _node(cid, NodeLabel.DOCUMENT.value, doc_id, source="slack")

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows("t-a", doc_id) == 0
    assert await _chunk_rows("t-a", doc_id) == 0
    assert await _doc_rows("t-b", doc_id) == 2
    assert await _chunk_rows("t-b", doc_id) == 1
    assert await _count(
        "SELECT count(*) FROM graph_nodes WHERE customer_id = 't-b'"
    ) == 1


async def test_many_batches_delete_everything(app_settings, settings, monkeypatch) -> None:
    """Batch size 2 forces the chunk and version loops to commit several times;
    everything must still go and the loop must stop."""
    monkeypatch.setattr(purge, "TOMBSTONE_PURGE_BATCH_SIZE", 2)
    monkeypatch.setattr(purge, "TOMBSTONE_PURGE_DOCS_PER_GROUP", 1)
    cid = "t-batches"
    await _tenant(cid)
    for n in range(3):
        doc_id = f"slack:C1:{n}"
        await _doc(cid, doc_id, versions=4, tombstone_days=9)
        for c in range(5):
            await _chunk(cid, doc_id, f"{doc_id}:{c}", 1, 3, live=False)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _count("SELECT count(*) FROM documents WHERE customer_id = $1", cid) == 0
    assert await _count("SELECT count(*) FROM chunks WHERE customer_id = $1", cid) == 0


# ---------------------------------------------------------------------------
# tenant state and legal hold
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "metadata"),
    [
        ("active", {"legal_hold": "litigation hold, ticket #12"}),
        ("terminated", {}),
        ("deleted", {}),
    ],
)
async def test_held_or_inactive_tenant_is_untouched(
    app_settings, settings, status, metadata
) -> None:
    cid = "t-skip"
    await _tenant(cid, status=status, metadata=metadata)
    await _doc(cid, "slack:C1:1.0", versions=2, tombstone_days=40)
    await _chunk(cid, "slack:C1:1.0", "c1", 1, 1, live=False)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows(cid, "slack:C1:1.0") == 2
    assert await _chunk_rows(cid, "slack:C1:1.0") == 1


def _hold_on_gate_entry(monkeypatch, customer_id: str, entry: int) -> None:
    """Set the hold just before the `entry`-th deleting transaction checks it --
    i.e. after the earlier ones have committed."""
    real = purge._gated
    calls = 0

    @asynccontextmanager
    async def gated(cid: str) -> AsyncIterator:
        nonlocal calls
        calls += 1
        if calls == entry:
            await _set_hold(customer_id, "set mid-run")
        async with real(cid) as conn:
            yield conn

    monkeypatch.setattr(purge, "_gated", gated)


async def test_hold_set_between_documents_stops_the_rest(
    app_settings, settings, monkeypatch
) -> None:
    monkeypatch.setattr(purge, "TOMBSTONE_PURGE_DOCS_PER_GROUP", 1)
    cid = "t-hold-docs"
    await _tenant(cid)
    await _doc(cid, "slack:C1:first", versions=1, tombstone_days=20)
    await _doc(cid, "slack:C1:second", versions=1, tombstone_days=10)
    _hold_on_gate_entry(monkeypatch, cid, entry=2)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows(cid, "slack:C1:first") == 0  # oldest goes first
    assert await _doc_rows(cid, "slack:C1:second") == 1


async def test_hold_set_between_batches_of_one_document_stops_it(
    app_settings, settings, monkeypatch
) -> None:
    monkeypatch.setattr(purge, "TOMBSTONE_PURGE_BATCH_SIZE", 2)
    cid = "t-hold-batches"
    await _tenant(cid)
    await _doc(cid, "slack:C1:big", versions=2, tombstone_days=20)
    for c in range(5):
        await _chunk(cid, "slack:C1:big", f"c{c}", 1, 1, live=False)
    _hold_on_gate_entry(monkeypatch, cid, entry=2)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _chunk_rows(cid, "slack:C1:big") == 3  # one batch of 2 committed
    assert await _doc_rows(cid, "slack:C1:big") == 2


@pytest.mark.parametrize(
    ("value", "held"),
    [
        (None, False),  # key absent
        ("__json_null__", False),  # explicitly cleared
        ("litigation", True),
        ("", True),  # fail closed: malformed still holds
        (False, True),
        ({"reason": "x"}, True),
    ],
)
async def test_legal_hold_fails_closed_on_its_shape(live_db, value, held) -> None:
    cid = "t-shape"
    await _tenant(cid)
    if value == "__json_null__":
        await _set_hold(cid, None)
    elif value is not None:
        await _set_hold(cid, value)
    async with db_module.raw_conn() as conn:
        reason = await purge_blocked_reason(conn, cid)
    assert (reason == "legal_hold") is held
    assert reason in (None, "legal_hold")


# ---------------------------------------------------------------------------
# raw payloads, in-flight work, budget
# ---------------------------------------------------------------------------


async def test_manual_upload_objects_and_row_go(app_settings, settings, bucket_for) -> None:
    cid = "t-manual"
    await _tenant(cid)
    store = storage_module.get_store()
    bucket = await bucket_for(cid)
    doc_id = "manual_upload:up-1"
    payload = f"raw/manual_upload/{cid}/2026/09/01/up-1.json"
    staging = f"manual_uploads/staging/{cid}/2026/09/01/up-1/report.pdf"
    other = f"manual_uploads/staging/{cid}/2026/09/01/up-2/other.pdf"
    for key in (payload, staging, other):
        await store.put(bucket, key, b"bytes")
    await _doc(cid, doc_id, versions=1, tombstone_days=8,
               source_system=SourceSystem.MANUAL_UPLOAD.value)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO manual_uploads (upload_id, customer_id, filename, file_sha256,
                                        staging_object_key, payload_object_key,
                                        status, doc_id)
            VALUES ('up-1', $1, 'report.pdf', 'sha', $2, $3, 'indexed', $4)
            """,
            cid,
            staging,
            payload,
            doc_id,
        )
    assert await _run(app_settings, settings) == purge.EXIT_OK
    assert await _doc_rows(cid, doc_id) == 0
    assert await _count(
        "SELECT count(*) FROM manual_uploads WHERE customer_id = $1", cid
    ) == 0
    assert set(await store.list_keys(bucket, "")) == {other}


async def test_storage_failure_keeps_the_rows(
    app_settings, settings, bucket_for, monkeypatch
) -> None:
    """The rows are the only handle a retry has on the payloads."""
    cid = "t-r2-down"
    await _tenant(cid)
    store = storage_module.get_store()
    bucket = await bucket_for(cid)
    doc_id = custom_ingest_doc_id(cid, SOURCE_KEY, "run:1")
    await _doc(cid, doc_id, versions=1, tombstone_days=9,
               source_system=SourceSystem.CUSTOM_INGEST.value)
    key = document_payload_key(cid, SOURCE_KEY, "run:1", "aaa")
    await store.put(bucket, key, b"{}")

    async def unavailable(self, bucket: str, keys: list[str]):
        raise StorageUnavailable("r2 down")

    monkeypatch.setattr(storage_module.ObjectStore, "delete_keys", unavailable)
    assert await _run(app_settings, settings) == purge.EXIT_FAILED
    assert await _doc_rows(cid, doc_id) == 1
    assert await store.list_keys(bucket, "raw/") == [key]


async def test_in_flight_recreate_is_left_alone(app_settings, settings, bucket_for) -> None:
    """A queued custom-ingest row for the document means it is being pushed
    again right now; clearing its raw prefix would delete the new payload."""
    cid = "t-in-flight"
    await _tenant(cid)
    store = storage_module.get_store()
    bucket = await bucket_for(cid)
    doc_id = custom_ingest_doc_id(cid, SOURCE_KEY, "run:1")
    await _doc(cid, doc_id, versions=2, tombstone_days=9,
               source_system=SourceSystem.CUSTOM_INGEST.value)
    new_payload = document_payload_key(cid, SOURCE_KEY, "run:1", "fresh")
    await store.put(bucket, new_payload, b"{}")
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO ingestion_queue (customer_id, source_system, source_event_id,
                                         payload_s3_key, payload_s3_keys, status)
            VALUES ($1, 'custom_ingest', $2, $3, ARRAY[$3], 'pending')
            """,
            cid,
            document_event_prefix(SOURCE_KEY, "run:1") + "0123456789abcdef",
            new_payload,
        )
    assert await _run(app_settings, settings) == purge.EXIT_OK
    assert await _doc_rows(cid, doc_id) == 2
    assert await store.list_keys(bucket, "raw/") == [new_payload]


async def test_dry_run_deletes_nothing(app_settings, settings) -> None:
    cid = "t-dry"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=2, tombstone_days=9)

    assert await _run(app_settings, settings, dry_run=True) == purge.EXIT_OK

    assert await _doc_rows(cid, "slack:C1:1.0") == 2


async def test_budget_exhausted_stops_and_says_so(app_settings, settings) -> None:
    cid = "t-budget"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=1, tombstone_days=9)

    assert await _run(app_settings, settings, max_seconds=0) == purge.EXIT_BUDGET_EXHAUSTED

    assert await _doc_rows(cid, "slack:C1:1.0") == 1


def test_window_fits_the_deletion_deadline() -> None:
    """Eligible at TOMBSTONE_PURGE_DAYS, deleted by the next daily run, gone
    from the last backup BACKUP_TAIL_DAYS later: inside the policy's deadline.
    Lengthening the window or the backup tail past it has to fail here."""
    daily_run = 1
    assert TOMBSTONE_PURGE_DAYS + daily_run + BACKUP_TAIL_DAYS <= DELETION_DEADLINE_DAYS
