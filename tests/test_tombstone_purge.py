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

import asyncio
import importlib.util
import json
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import asyncpg
import pytest
import pytest_asyncio
from structlog.testing import capture_logs

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


async def _recreate(conn, customer_id: str, doc_id: str, version: int, source_system: str) -> None:
    """What an applied re-push leaves: the tombstone closed, a live version on top."""
    await conn.execute(
        "UPDATE documents SET valid_to = now()"
        " WHERE customer_id = $1 AND doc_id = $2 AND valid_to IS NULL",
        customer_id,
        doc_id,
    )
    await conn.execute(
        """
        INSERT INTO documents (customer_id, doc_id, version, source_system, source_id,
                               source_url, doc_type, content_hash, created_at, updated_at,
                               valid_from, acl, title, body_preview, metadata)
        VALUES ($1, $2, $3, $4, $2, 'https://x', 't', 'h-back', now(), now(), now(),
                '{}'::jsonb, 'back again', 'the customer re-created it', '{}'::jsonb)
        """,
        customer_id,
        doc_id,
        version,
        source_system,
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


async def _wait_for_lock_wait(fragment: str) -> None:
    """Until a backend whose query mentions `fragment` waits on a lock -- the
    interleaving a race test needs, asserted rather than slept for."""
    for _ in range(200):
        async with db_module.raw_conn() as conn:
            if await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity"
                " WHERE wait_event_type = 'Lock' AND query ILIKE $1",
                f"%{fragment}%",
            ):
                return
        await asyncio.sleep(0.05)
    pytest.fail(f"nothing running {fragment!r} ever waited on a lock")


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


async def test_dry_run_deletes_nothing_and_sizes_the_work(app_settings, settings) -> None:
    """An attended first run faces the whole historical backlog; the dry run is
    what sizes it, so it reports rows and sources, not just a document count."""
    cid = "t-dry"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=2, tombstone_days=9)
    await _chunk(cid, "slack:C1:1.0", "c1", 1, 1, live=False)
    await _doc(cid, "slack:C1:live", versions=2, tombstone_days=None)
    await _chunk(cid, "slack:C1:live", "c2", 1, 1, live=False)

    assert await _run(app_settings, settings, dry_run=True) == purge.EXIT_OK

    assert await _doc_rows(cid, "slack:C1:1.0") == 2
    assert await _chunk_rows(cid, "slack:C1:1.0") == 1

    result = await purge.purge_tenant(
        cid,
        cutoff=datetime.now(UTC) - timedelta(days=TOMBSTONE_PURGE_DAYS),
        deadline=float("inf"),
        dry_run=True,
    )
    assert result.eligible == 1
    assert result.eligible_rows == {"documents": 2, "chunks": 1}
    assert result.by_source == {"slack": 1}
    assert result.oldest_deleted_at is not None
    assert result.documents == 0 and not result.rows


async def test_budget_exhausted_stops_and_says_so(app_settings, settings) -> None:
    cid = "t-budget"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=1, tombstone_days=9)

    assert await _run(app_settings, settings, max_seconds=0) == purge.EXIT_BUDGET_EXHAUSTED

    assert await _doc_rows(cid, "slack:C1:1.0") == 1


async def test_revived_chunk_survives_a_racing_delete(live_db) -> None:
    """A closed code-graph tombstone gives a re-create no row lock to wait on,
    so its chunk upsert can revive a chunk IN PLACE (ON CONFLICT ... SET
    last_seen_version) while the purge's DELETE waits on that row. The DELETE
    must re-check the version bound on the row it deletes, not trust the
    snapshot it selected from."""
    cid = "t-revive"
    await _tenant(cid)
    doc_id = "code_graph:acme/app:src/x.py:f"
    await _doc(cid, doc_id, versions=2, tombstone_days=20, closed=True,
               source_system=SourceSystem.CODE_GRAPH.value)
    await _chunk(cid, doc_id, "revived", 1, 2, live=False)
    await _chunk(cid, doc_id, "dead", 1, 1, live=False)

    pool = db_module.get_pool()
    writer = await pool.acquire()
    try:
        tx = writer.transaction()
        await tx.start()
        await writer.execute(
            "UPDATE chunks SET last_seen_version = 3, valid_to = NULL"
            " WHERE customer_id = $1 AND chunk_id = 'revived'",
            cid,
        )

        async def delete():
            async with db_module.with_tenant(cid) as conn:
                return await conn.fetchrow(purge._DELETE_CHUNKS_SQL, cid, [doc_id], [2], 100)

        task = asyncio.create_task(delete())
        await _wait_for_lock_wait("DELETE FROM chunks")
        await tx.commit()
        row = await asyncio.wait_for(task, 10)
    finally:
        await pool.release(writer)

    assert (row["selected"], row["deleted"]) == (2, 1)
    assert await _count(
        "SELECT count(*) FROM chunks WHERE customer_id = $1 AND chunk_id = 'revived'", cid
    ) == 1
    assert await _count(
        "SELECT count(*) FROM chunks WHERE customer_id = $1 AND chunk_id = 'dead'", cid
    ) == 0


async def test_recreated_after_scan_keeps_its_payloads(
    app_settings, settings, bucket_for, monkeypatch
) -> None:
    """A re-push applied between the candidate scan and the raw delete has no
    queue row in flight any more, and the listing holds the LIVE document's
    payload. Eligibility is re-read before the objects go."""
    cid = "t-recreated"
    await _tenant(cid)
    store = storage_module.get_store()
    bucket = await bucket_for(cid)
    ci = SourceSystem.CUSTOM_INGEST.value
    doc_id = custom_ingest_doc_id(cid, SOURCE_KEY, "run:1")
    await _doc(cid, doc_id, versions=1, tombstone_days=9, source_system=ci)
    live_payload = document_payload_key(cid, SOURCE_KEY, "run:1", "back")
    await store.put(bucket, live_payload, b"{}")

    real_list = storage_module.ObjectStore.list_keys
    recreated = False

    async def list_then_recreate(self, bucket_name: str, prefix: str) -> list[str]:
        nonlocal recreated
        keys = await real_list(self, bucket_name, prefix)
        if not recreated:
            recreated = True
            async with db_module.with_tenant(cid) as conn:
                await _recreate(conn, cid, doc_id, 2, ci)
        return keys

    monkeypatch.setattr(storage_module.ObjectStore, "list_keys", list_then_recreate)
    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert recreated
    assert await real_list(store, bucket, "raw/") == [live_payload]
    assert await _doc_rows(cid, doc_id) == 2


async def test_node_of_a_recreated_document_survives_the_final_batch(live_db) -> None:
    """A re-connect re-creates a closed code-graph tombstone without touching
    the row the purge locked: it inserts version N+1 and upserts the document's
    node. Once that commits, the node and the edges written with it belong to a
    live document again, and the final batch must not delete them."""
    cid = "t-node-race"
    await _tenant(cid)
    cg = SourceSystem.CODE_GRAPH.value
    doc_id = "code_graph:acme/app:src/x.py"
    await _doc(cid, doc_id, versions=2, tombstone_days=20, closed=True, source_system=cg)
    node = await _node(cid, NodeLabel.DOCUMENT.value, doc_id, source=cg)
    symbol = await _node(cid, NodeLabel.CODE_SYMBOL.value, "acme/app:f", source=cg)
    await _edge(cid, EdgeType.COMPILED_FROM.value, node, symbol)

    pool = db_module.get_pool()
    writer = await pool.acquire()
    try:
        tx = writer.transaction()
        await tx.start()
        await _recreate(writer, cid, doc_id, 3, cg)
        await writer.execute(
            "UPDATE graph_nodes SET properties = properties || '{\"back\": true}'::jsonb"
            " WHERE node_id = $1",
            node,
        )

        async def finish() -> int:
            doc = purge._Doc(doc_id, 2, cg, doc_id, None, None)
            async with purge._gated(cid) as conn:
                return await purge._finish(conn, cid, [doc], [2], Counter())

        task = asyncio.create_task(finish())
        await _wait_for_lock_wait("graph_nodes")
        await tx.commit()
        gone = await asyncio.wait_for(task, 10)
    finally:
        await pool.release(writer)

    assert gone == 0
    assert await _count(
        "SELECT count(*) FROM graph_nodes WHERE node_id = ANY($1::bigint[])", [node, symbol]
    ) == 2
    assert await _count("SELECT count(*) FROM graph_edges WHERE from_node_id = $1", node) == 1
    assert await _doc_rows(cid, doc_id) == 1  # version 3; the old tombstone is history


async def test_hold_set_between_raw_deletes_stops_the_rest(
    app_settings, settings, bucket_for, monkeypatch
) -> None:
    """Deleting a raw payload is the one step nothing can undo, so the hold is
    re-checked before each document's, not once per group."""
    monkeypatch.setattr(purge, "TOMBSTONE_PURGE_DOCS_PER_GROUP", 2)
    cid = "t-hold-raw"
    await _tenant(cid)
    store = storage_module.get_store()
    bucket = await bucket_for(cid)
    ci = SourceSystem.CUSTOM_INGEST.value
    first = custom_ingest_doc_id(cid, SOURCE_KEY, "run:1")
    second = custom_ingest_doc_id(cid, SOURCE_KEY, "run:2")
    await _doc(cid, first, versions=1, tombstone_days=20, source_system=ci)
    await _doc(cid, second, versions=1, tombstone_days=10, source_system=ci)
    first_key = document_payload_key(cid, SOURCE_KEY, "run:1", "aaa")
    second_key = document_payload_key(cid, SOURCE_KEY, "run:2", "bbb")
    for key in (first_key, second_key):
        await store.put(bucket, key, b"{}")
    _hold_on_gate_entry(monkeypatch, cid, entry=2)

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await store.list_keys(bucket, "raw/") == [second_key]
    # Rows go only after the whole group's payloads, so both are still here.
    assert await _doc_rows(cid, first) == 1
    assert await _doc_rows(cid, second) == 1


async def test_acl_shared_with_a_live_document_is_kept(app_settings, settings) -> None:
    """ACL resource ids are per source, and a container id (a channel, a repo)
    names every document in it. The deleted document's own row goes; the one a
    surviving document of the same source still answers to stays."""
    cid = "t-acl"
    await _tenant(cid)
    await _doc(cid, "slack:C1:gone", versions=1, tombstone_days=9, source_id="C1")
    await _doc(cid, "slack:C1:live", versions=1, tombstone_days=None, source_id="C1")
    async with db_module.raw_conn() as conn:
        for resource_id in ("C1", "slack:C1:gone"):
            await conn.execute(
                """
                INSERT INTO acl_snapshots (customer_id, source_system, principal_type,
                                           principal_id, resource_type, resource_id,
                                           permission, valid_from)
                VALUES ($1, 'slack', 'workspace', $1, 'slack.channel', $2, 'read', now())
                """,
                cid,
                resource_id,
            )

    assert await _run(app_settings, settings) == purge.EXIT_OK

    assert await _doc_rows(cid, "slack:C1:gone") == 0
    assert await _count(
        "SELECT count(*) FROM acl_snapshots WHERE customer_id = $1 AND resource_id = 'C1'", cid
    ) == 1
    assert await _count(
        "SELECT count(*) FROM acl_snapshots WHERE customer_id = $1"
        " AND resource_id = 'slack:C1:gone'",
        cid,
    ) == 0


# The side deletes as #593 merged them, kept as the oracle for the index-shaped
# rewrites (migration 0141): the rewrite must delete exactly these rows. Both
# read the tenant's whole share of the table per batch in production.
_OR_FORM_ACL_SQL = """
    DELETE FROM acl_snapshots a
    USING unnest($2::text[], $3::text[], $4::text[]) AS g(doc_id, source_system, source_id)
    WHERE a.customer_id = $1
      AND a.source_system = g.source_system
      AND a.resource_id IN (g.doc_id, g.source_id)
      AND NOT EXISTS (
          SELECT 1 FROM documents x
          WHERE x.customer_id = $1
            AND x.source_system = a.source_system
            AND (x.doc_id = a.resource_id OR x.source_id = a.resource_id)
      )
"""
_OR_FORM_PENDING_SQL = """
    DELETE FROM pending_edges
    WHERE customer_id = $1
      AND ((from_label = 'Document' AND from_canonical_id = ANY($2::text[]))
        OR (to_label = 'Document' AND to_canonical_id = ANY($2::text[])))
"""


async def _deleted_by(statements: list[tuple[str, tuple]], table: str, key: str) -> set:
    """The `key`s of `table` rows the statements delete, rolled back after.

    The WHOLE table, every tenant, read as the superuser (no RLS): a statement
    that lost its customer_id predicate shows up here as another tenant's key,
    which a count taken after the rollback could never see."""
    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            sql = f"SELECT {key} FROM {table}"
            before = {r[0] for r in await conn.fetch(sql)}
            for statement, args in statements:
                await conn.execute(statement, *args)
            after = {r[0] for r in await conn.fetch(sql)}
        finally:
            await tx.rollback()
    return before - after


async def _acl(cid: str, source_system: str, resource_id: str) -> int:
    async with db_module.raw_conn() as conn:
        return await conn.fetchval(
            """
            INSERT INTO acl_snapshots (customer_id, source_system, principal_type,
                                       principal_id, resource_type, resource_id,
                                       permission, valid_from)
            VALUES ($1, $2, 'workspace', $1, 'x', $3, 'read', now())
            RETURNING snapshot_id
            """,
            cid,
            source_system,
            resource_id,
        )


async def test_acl_delete_matches_the_or_form_it_replaced(live_db) -> None:
    """Each id of a purged document is its own lookup, and the survivor check
    is two NOT EXISTS (by doc_id, by source_id): the same rows as the OR form,
    on both arms, only within the document's own source."""
    cid = "t-acl-forms"
    await _tenant(cid)
    await _tenant("t-acl-other")
    # Survivors: two Slack documents, and a Linear one sharing gone-1's source_id.
    await _doc(cid, "slack:C1:live", versions=1, tombstone_days=None, source_id="C1:live")
    await _doc(cid, "slack:C9:live", versions=1, tombstone_days=None, source_id="C9")
    await _doc(cid, "linear:ISS-1", versions=1, tombstone_days=None,
               source_system=SourceSystem.LINEAR.value, source_id="C1:gone")
    # Purged (already gone from documents, as they are when the delete runs):
    # gone-2's source_id happens to be a live Slack document's doc_id, and
    # gone-3's is a container id (a channel) a live Slack document also has.
    gone = [
        ("slack:C1:gone", "slack", "C1:gone"),
        ("slack:C2:gone", "slack", "slack:C1:live"),
        ("slack:C9:gone", "slack", "C9"),
    ]
    rows = {
        "gone-1 by doc_id": await _acl(cid, "slack", "slack:C1:gone"),
        "gone-1 by source_id, a Linear doc shares it": await _acl(cid, "slack", "C1:gone"),
        "same id, another source": await _acl(cid, "linear", "C1:gone"),
        "gone-2 by source_id, a live doc's doc_id": await _acl(cid, "slack", "slack:C1:live"),
        "gone-2 by doc_id": await _acl(cid, "slack", "slack:C2:gone"),
        "gone-3 by source_id, a live doc's source_id": await _acl(cid, "slack", "C9"),
        "gone-3 by doc_id": await _acl(cid, "slack", "slack:C9:gone"),
        "a live doc's own id": await _acl(cid, "slack", "C1:live"),
    }
    other_tenant = await _acl("t-acl-other", "slack", "slack:C1:gone")
    args = (cid, [g[0] for g in gone], [g[1] for g in gone], [g[2] for g in gone])

    rewritten = await _deleted_by([(purge._DELETE_ACL_SQL, args)], "acl_snapshots",
                                  "snapshot_id")
    or_form = await _deleted_by([(_OR_FORM_ACL_SQL, args)], "acl_snapshots",
                                "snapshot_id")

    # Read across every tenant, so the same-named row of another tenant would
    # appear here if either form reached it.
    assert other_tenant not in rewritten | or_form
    assert rewritten == or_form == {
        rows["gone-1 by doc_id"],
        rows["gone-1 by source_id, a Linear doc shares it"],
        rows["gone-2 by doc_id"],
        rows["gone-3 by doc_id"],
    }


async def test_pending_edges_delete_matches_the_or_form_it_replaced(live_db) -> None:
    """Split per side for the per-side partial indexes: still every row with a
    purged Document on EITHER end (both ends once), and nothing that merely
    shares the id under another label."""
    cid = "t-pending-forms"
    await _tenant(cid)
    await _tenant("t-pending-other")
    doc, other = "custom_ingest:t:k:gone", "custom_ingest:t:k:live"
    shapes = [
        ("from", cid, "Document", doc, "Run", "run:1"),
        ("to", cid, "Run", "run:2", "Document", doc),
        ("both", cid, "Document", doc, "Document", doc),
        ("other doc", cid, "Run", "run:3", "Document", other),
        ("same id, not a Document", cid, "Run", doc, "Project", doc),
        ("same id, another tenant", "t-pending-other", "Document", doc, "Document", doc),
    ]
    ids = {}
    async with db_module.raw_conn() as conn:
        for name, tenant, from_label, from_id, to_label, to_id in shapes:
            ids[name] = await conn.fetchval(
                """
                INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id,
                                           edge_type, from_label, from_canonical_id,
                                           to_label, to_canonical_id, source_system)
                VALUES ($1, $2, $3, 'TOUCHES', $2, $3, $4, $5, 'custom_ingest')
                RETURNING id
                """,
                tenant,
                from_label,
                from_id,
                to_label,
                to_id,
            )
    split = [(sql, (cid, [doc])) for _stage, table, sql in purge._BY_DOC_ID
             if table == "pending_edges"]
    assert len(split) == 2

    rewritten = await _deleted_by(split, "pending_edges", "id")
    or_form = await _deleted_by([(_OR_FORM_PENDING_SQL, (cid, [doc]))], "pending_edges", "id")

    assert rewritten == or_form == {ids["from"], ids["to"], ids["both"]}


async def test_side_deletes_can_use_their_indexes(app_settings) -> None:
    """Every per-document side delete on a big table must be SERVABLE by its
    index: the first production run timed out because three could not be (an
    OR, an IN list, a partial index whose predicate the query did not restate)
    and the ACL check fell back to the trigram index for every row.

    `enable_seqscan = off` makes this ask "can an index serve it", not "is a
    scan wrong here" -- on a test-sized table a scan is the right plan
    (engine/retrieval/index_contracts.py says why a plain EXPLAIN test would be
    wrong). A few thousand analyzed rows, because on EMPTY tables the planner
    picks index scans on customer_id alone and filters the rest, which would
    prove nothing. Run as the app role, under the tenant GUC, as the purge
    runs: FORCE RLS adds a security qual that must not keep the index out."""
    cid = "t-plans"
    await _tenant(cid)
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (customer_id, doc_id, version, source_system, source_id,
                                   source_url, doc_type, content_hash, created_at,
                                   updated_at, valid_from, acl)
            SELECT $1, 'custom_ingest:t-plans:k:' || i, 1, 'custom_ingest', 'k:' || i,
                   'https://x', 't', 'h', now(), now(), now(), '{}'::jsonb
            FROM generate_series(1, 2000) i
            """,
            cid,
        )
        await conn.execute(
            """
            INSERT INTO acl_snapshots (customer_id, source_system, principal_type,
                                       principal_id, resource_type, resource_id,
                                       permission, valid_from)
            SELECT $1, 'custom_ingest', 'workspace', $1, 'custom.document', 'k:' || i,
                   'read', now()
            FROM generate_series(1, 2000) i
            """,
            cid,
        )
        await conn.execute(
            """
            INSERT INTO pending_edges (customer_id, missing_label, missing_canonical_id,
                                       edge_type, from_label, from_canonical_id,
                                       to_label, to_canonical_id, source_system)
            SELECT $1, 'Run', 'run:' || i, 'TOUCHES', 'Run', 'run:' || i, 'Document',
                   'custom_ingest:t-plans:k:' || i, 'custom_ingest'
            FROM generate_series(1, 2000) i
            """,
            cid,
        )
        await conn.execute(
            """
            INSERT INTO inferred_edges_queue (customer_id, anchor_doc_id, extractor_id,
                                              done_at)
            SELECT $1, 'custom_ingest:t-plans:k:' || i, 'inferred_edges:v1', now()
            FROM generate_series(1, 2000) i
            """,
            cid,
        )
        await conn.execute(
            """
            INSERT INTO manual_uploads (upload_id, customer_id, filename, file_sha256,
                                        status, doc_id)
            SELECT 'u' || i, $1, 'f', 'h', 'indexed', 'custom_ingest:t-plans:k:' || i
            FROM generate_series(1, 2000) i
            """,
            cid,
        )
        for table in ("documents", "acl_snapshots", "pending_edges", "inferred_edges_queue",
                      "manual_uploads"):
            await conn.execute(f"ANALYZE {table}")
    ids = ["custom_ingest:t-plans:k:1"]
    conn = await asyncpg.connect(app_settings.database_url)
    try:
        async with conn.transaction():
            await conn.execute("SELECT set_config('app.current_customer_id', $1, true)", cid)
            await conn.execute("SET LOCAL enable_seqscan = off")

            async def plan(sql: str, *args) -> str:
                return "\n".join(r[0] for r in await conn.fetch("EXPLAIN " + sql, *args))

            plans = {stage: await plan(sql, cid, ids) for stage, _t, sql in purge._BY_DOC_ID}
            acl_args = (cid, ids, [SourceSystem.CUSTOM_INGEST.value], ["k:1"])
            plans["acl_snapshots"] = await plan(purge._DELETE_ACL_SQL, *acl_args)
            or_form_acl = await plan(_OR_FORM_ACL_SQL, *acl_args)
    finally:
        await conn.close()

    assert "idx_inferred_edges_queue_anchor" in plans["inferred_edges_queue"]
    assert "idx_pending_edges_from_document" in plans["pending_edges.from"]
    assert "idx_pending_edges_to_document" in plans["pending_edges.to"]
    assert "idx_manual_uploads_doc" in plans["manual_uploads"]
    # github_document_bindings is left unasserted on purpose: GitHub documents
    # only, and its primary key has doc_id third.
    # The control: the OR form that timed out does show the trigram probe, so
    # its absence below is a finding, not a blind spot.
    assert "trgm" in or_form_acl, or_form_acl
    acl = plans["acl_snapshots"]
    # Each lookup keyed on the id itself, not on customer_id with a filter.
    assert "idx_acl_snapshots_resource" in acl, acl
    assert "(resource_id = " in acl, acl
    assert "documents_pkey" in acl, acl
    assert "(doc_id = a.resource_id)" in acl, acl
    assert "idx_documents_customer_source" in acl, acl
    assert "(source_id = a.resource_id)" in acl, acl
    assert "trgm" not in acl, acl


_MIGRATION_0141 = (
    Path(__file__).resolve().parents[1]
    / "db/migrations/versions/20260925_0141_tombstone_purge_side_indexes.py"
)


async def test_migration_0141_builds_the_indexes_schema_sql_declares(live_db) -> None:
    """CI builds its database from db/schema.sql and stamps alembic head, so the
    migration that ships these indexes to both planes never runs here, and the
    plan test above proves only schema.sql's copies. A drifted migration -- a
    column, or the literal 'Document' a partial predicate must restate -- would
    pass every test and give production an index the purge cannot use. So each
    index is rebuilt from the migration's own definition, in a transaction that
    is rolled back, and Postgres's rendering of the two must be identical."""
    spec = importlib.util.spec_from_file_location("migration_0141", _MIGRATION_0141)
    assert spec is not None and spec.loader is not None
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    indexdef = "SELECT pg_get_indexdef(to_regclass($1)::oid)"

    async with db_module.raw_conn() as conn:
        tx = conn.transaction()
        await tx.start()
        try:
            for name, definition in migration._INDEXES:
                from_schema = await conn.fetchval(indexdef, name)
                assert from_schema is not None, f"db/schema.sql does not declare {name}"
                await conn.execute(f"DROP INDEX {name}")
                await conn.execute(f"CREATE INDEX {name} {definition}")
                assert await conn.fetchval(indexdef, name) == from_schema
        finally:
            await tx.rollback()


async def test_a_statement_timeout_names_its_stage(app_settings, settings, monkeypatch) -> None:
    """asyncpg's client command_timeout raises a TimeoutError whose message is
    empty; the first production failure logged `error=""` and no statement.
    The failure names the stage it was in and says what timed out."""
    cid = "t-timeout"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=1, tombstone_days=9)
    monkeypatch.setattr(
        purge,
        "_DELETE_ACL_SQL",
        "SELECT pg_sleep(30), $1::text, $2::text[], $3::text[], $4::text[]",
    )
    # 2 s, not less: the timeout covers every statement of the run, and a slow
    # CI runner must not time out an earlier one and fail on the wrong stage.
    impatient = app_settings.model_copy(update={"db_statement_timeout_ms": 2000})
    # In production one Settings builds the pool and names its timeout; the
    # test pool is built from `impatient`, so the message must read it too.
    monkeypatch.setattr(purge, "get_settings", lambda: impatient)

    with capture_logs() as logs:
        assert await _run(impatient, settings) == purge.EXIT_FAILED

    failed = [r for r in logs if r["event"] == "tombstone_purge.tenant_failed"]
    assert len(failed) == 1, logs
    assert failed[0]["stage"] == "acl_snapshots"
    assert failed[0]["error_type"] == "TimeoutError"
    assert "command_timeout (2000 ms" in failed[0]["error"]
    assert await _doc_rows(cid, "slack:C1:1.0") == 1  # the batch rolled back


async def test_a_lock_wait_names_its_stage(app_settings, settings, monkeypatch) -> None:
    """A group given up on a lock wait (a hold being placed holds the customers
    row) is logged contended with the stage it was waiting in, and keeps its
    rows for the next run."""
    cid = "t-contended"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=1, tombstone_days=9)
    monkeypatch.setattr(purge, "_LOCK_TIMEOUT", "200ms")
    # Its own connection, not the fixture pool: _run closes that pool.
    holder = await asyncpg.connect(settings.database_url)
    try:
        async with holder.transaction():
            await holder.execute(
                "SELECT 1 FROM customers WHERE customer_id = $1 FOR UPDATE", cid
            )
            with capture_logs() as logs:
                await _run(app_settings, settings)
    finally:
        await holder.close()

    contended = [r for r in logs if r["event"] == "tombstone_purge.group_contended"]
    assert [r["stage"] for r in contended] == ["gate"], logs
    assert contended[0]["error"] == "LockNotAvailableError"
    assert await _doc_rows(cid, "slack:C1:1.0") == 1


def test_an_error_without_a_message_is_still_named() -> None:
    assert purge._error_text(ValueError("bad row")) == "bad row"
    assert purge._error_text(RuntimeError()) == "RuntimeError()"


async def test_tombstone_a_writer_holds_waits_for_the_next_run(app_settings, settings) -> None:
    """SKIP LOCKED: a tombstone row a writer holds (a re-create closing it) is
    left alone, and the next run finishes it."""
    cid = "t-held-row"
    await _tenant(cid)
    await _doc(cid, "slack:C1:1.0", versions=2, tombstone_days=9)
    await _chunk(cid, "slack:C1:1.0", "c1", 1, 1, live=False)
    # Outside the pool: _run swaps pools, and closing one waits for its
    # connections to come back.
    writer = await asyncpg.connect(settings.database_url)
    try:
        async with writer.transaction():
            await writer.execute(
                "SELECT 1 FROM documents WHERE customer_id = $1 AND version = 2 FOR UPDATE",
                cid,
            )
            assert await _run(app_settings, settings) == purge.EXIT_OK
            assert await _doc_rows(cid, "slack:C1:1.0") == 2
            assert await _chunk_rows(cid, "slack:C1:1.0") == 1
    finally:
        await writer.close()

    assert await _run(app_settings, settings) == purge.EXIT_OK
    assert await _doc_rows(cid, "slack:C1:1.0") == 0
    assert await _chunk_rows(cid, "slack:C1:1.0") == 0


async def test_one_tenants_discovery_failure_does_not_stop_the_rest(
    app_settings, settings, monkeypatch
) -> None:
    await _tenant("t-bad")
    await _tenant("t-good")
    await _doc("t-bad", "slack:C1:1.0", versions=1, tombstone_days=9)
    await _doc("t-good", "slack:C1:1.0", versions=1, tombstone_days=9)
    real = purge.with_tenant

    @asynccontextmanager
    async def flaky(customer_id: str) -> AsyncIterator:
        if customer_id == "t-bad":
            raise RuntimeError("canceling statement due to statement timeout")
        async with real(customer_id) as conn:
            yield conn

    monkeypatch.setattr(purge, "with_tenant", flaky)
    assert await _run(app_settings, settings) == purge.EXIT_FAILED

    assert await _doc_rows("t-good", "slack:C1:1.0") == 0
    assert await _doc_rows("t-bad", "slack:C1:1.0") == 1


def test_window_fits_the_deletion_deadline() -> None:
    """Eligible at TOMBSTONE_PURGE_DAYS, deleted by the next daily run, gone
    from the last backup BACKUP_TAIL_DAYS later: inside the policy's deadline.
    Lengthening the window or the backup tail past it has to fail here."""
    daily_run = 1
    assert TOMBSTONE_PURGE_DAYS + daily_run + BACKUP_TAIL_DAYS <= DELETION_DEADLINE_DAYS
