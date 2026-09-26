"""Hard-delete what a customer deleted, once its tombstone is a week old.

WHY THIS EXISTS. A delete reaches the engine as a TOMBSTONE: a custom-ingest
entry with `"deleted": true` (every research-os run, project or note deleted in
the product), or a source-side delete (Slack, Linear, Notion, GitHub, a
code-graph file or repo). The normalizer writes a new document version with
`deleted_at` set and an empty body, closes the previous one and retires its
chunks. That makes the item unsearchable and nothing more: `cron_chunk_retention`
drops the retired chunks after 30 days, but every `documents` version (title,
280-char `body_preview`, metadata, author, ACL) and every raw payload in R2
stayed forever. Probe's policy is that a deletion is complete within 30 days,
backups included, and backups trail by 8 days -- TOMBSTONE_PURGE_DAYS in
engine/shared/constants.py carries the arithmetic.

WHAT IS ELIGIBLE. A document whose CURRENT version -- its highest `version` --
has `deleted_at` older than the window, and which has no live undeleted version.
"Current", not `valid_to IS NULL`: a code-graph repo disconnect tombstones in
place and closes the row too (kb/handlers/codegraph.py), so those tombstones
have no live row at all. A document deleted and then re-created is live again
and is NOT touched, history included: while a customer is active, superseded
versions of live documents are kept. So are tombstones inside the window.

WHAT IS DELETED, per eligible document, in this order:

  1. Raw payloads that belong to this document alone:
       custom_ingest   every object under raw/custom_ingest/<customer>/
                       <source_key>/<doc_hash>/ (one per content hash pushed)
       manual uploads  the manual_uploads row's payload and staging objects
     Listed, then deleted by exact key, so a payload a re-create writes after
     the listing survives. A document with a queued or in-flight custom-ingest
     row is skipped for this run: it is being re-created right now. So is one
     that stopped being eligible after the scan (its re-create already
     applied): eligibility is re-read, and the row locked, right before each
     document's objects are deleted.
  2. chunks, in batches.
  3. Superseded document versions, in batches.
  4. In one final transaction: failed_chunks, the tombstone version, then --
     only once no version of the document is left -- the rows that reference
     it by id. No foreign key reaches `documents`, so nothing cascades from it:
       inferred_edges_queue      anchor_doc_id
       github_document_bindings  doc_id
       manual_uploads            doc_id
       acl_snapshots             resource_id = the doc's doc_id or source_id in
                                 the same source, unless a surviving document of
                                 that source still has that id
       pending_edges             a Document endpoint whose canonical id is it
       graph_nodes               its Document node (label 'Document',
                                 canonical_id = doc_id); graph_edges and
                                 graph_node_provenance follow by FK CASCADE
       graph_nodes (neighbours)  a node the Document node touched that is left
                                 with no edges, whose only provenance is the
                                 deleted document's source, and that is not a
                                 surviving document's node -- a run's own entity
                                 node, an author who wrote nothing else. Shared
                                 nodes keep their other edges and stay; the
                                 survivors get `degree` recounted.
       node_post_write_queue     rows for the nodes deleted above (no FK)

WHAT IS NOT DELETED, and why:
  * EVENT-ADDRESSED raw payloads: webhook and poll batches under
    raw/<source>/<customer>/<YYYY/MM/DD>/, backfill pages under .../backfill/,
    code-graph batches raw/code_graph/<customer>/<uuid>.json. One object holds
    many items, so it cannot be attributed to one document. They go with a
    source disconnect (engine/ingest/purge.py) or the tenant purge.
  * SESSION-ADDRESSED raw payloads (raw/<agent>/<customer>/sessions-v2/<sid>/,
    .../<sid>/extraction-cache/): nothing tombstones a session document today;
    deleting a session is its own job (retention plan T8).
  * ingestion_queue / ingestion_events rows: event records carrying R2 keys and
    doc ids, not content. The custom-ingest route already treats a missing
    document as deleted, so a later re-push of the same item is accepted.
  * INFERRED edges between two SURVIVING entities: their `why` text came from a
    bundle that may have included this document, and nothing records which.
  * query_traces, usage_events, retrieve_pages (self-cleaning within a day) and
    entity-merge history.

LEGAL HOLD AND TENANT STATE. Only tenants whose status is 'active' and that have
no `metadata.legal_hold` are considered (engine/shared/legal_hold.py), and both
are re-checked under FOR SHARE on the customers row at the start of EVERY
committed batch and around every R2 delete, so a hold set mid-run stops the next
delete. Terminated tenants are purged wholesale by research-os.

RLS. documents, chunks, the graph and most side tables are FORCE RLS and this
runs as the `app` role that owns them, so every statement runs in with_tenant():
an unscoped DELETE matches zero rows and reports success (migrations 0119/0125).

CONCURRENCY. A batch locks the tombstone rows it works on FOR UPDATE SKIP
LOCKED: a document a writer holds waits for the next run, and a writer
re-creating a document blocks until the batch commits, then writes its new
version. Every delete is bounded by the tombstone's version (chunks by
last_seen_version, documents by version), and the by-id rows and the graph node
go only once no version is left, so a re-create racing the purge keeps its new
version, chunks and node. Neighbour nodes are locked SKIP LOCKED too, so one a
writer is re-asserting is left alone. lock_timeout bounds every wait; a lock
timeout or deadlock skips that group for this run rather than failing it.

BATCHES AND RESUMING. Raw payloads go first because the documents row is the
only handle a retry has. Rows then go in batches of TOMBSTONE_PURGE_BATCH_SIZE
per committed transaction, the tombstone version last, so a killed run leaves a
still-eligible document for the next one. Documents are walked in
(deleted_at, doc_id) order, each at most once per run, over the partial index
idx_documents_tombstones (migration 0139). No new batch starts after
--max-seconds. Each batch's closing deletes on the big side tables
(acl_snapshots, pending_edges, inferred_edges_queue) are index lookups
(migration 0141); before it, the first production run spent ~5 minutes per
batch scanning them and timed out.

SCHEDULE. research-os schedules engine crons in its own chart; this needs one
CronJob shaped like engine-cron-chunk-retention PLUS R2 credentials
(engineR2Env), because step 1 deletes objects:

    command: ["python", "-m", "scripts.cron_tombstone_purge"]
    workingDir: /opt/upstreams/prbe-knowledge
    schedule: "40 4 * * *"        # daily -- the window arithmetic assumes it
    concurrencyPolicy: Forbid
    activeDeadlineSeconds: 1800   # above the default --max-seconds 1500

EXIT CODES: 0 = done, including nothing to do. 1 = a database or storage
operation failed for at least one tenant (the other tenants still ran).
3 = stopped at --max-seconds with eligible documents left; the next run resumes,
but a job that exits 3 every day is a backlog that will miss the deadline.

    python -m scripts.cron_tombstone_purge [--dry-run] [--customer ID] [--max-seconds N]
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import sys
import time
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime

import asyncpg

from engine.shared.config import get_settings
from engine.shared.constants import (
    TOMBSTONE_PURGE_BATCH_SIZE,
    TOMBSTONE_PURGE_DAYS,
    TOMBSTONE_PURGE_DOCS_PER_GROUP,
    TOMBSTONE_PURGE_MAX_SECONDS,
    NodeLabel,
    QueueStatus,
    SourceSystem,
)
from engine.shared.custom_ingest import document_event_prefix, document_payload_prefix
from engine.shared.db import close_pool, get_pool, init_pool, with_tenant
from engine.shared.exceptions import StorageUnavailable
from engine.shared.legal_hold import purge_blocked_reason, purge_eligible_tenant_sql
from engine.shared.logging import configure_logging, get_logger
from engine.shared.storage import ObjectStore

log = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_BUDGET_EXHAUSTED = 3

#: How long one batch waits on any lock before giving its group up for the run.
_LOCK_TIMEOUT = "5s"
#: A breath between committed batches, as in cron_chunk_retention: nothing else
#: (replication, autovacuum, ingestion) should queue behind this job.
_BATCH_PAUSE_SECONDS = 0.2
#: Keyset start for the (deleted_at, doc_id) walk.
_WALK_START = (datetime(1970, 1, 1, tzinfo=UTC), "")

_DOC_LABEL = NodeLabel.DOCUMENT.value

# Eligibility over `documents d`, shared by the scan and the per-batch lock so
# the two can never disagree about what may be deleted. {cutoff} is a $n.
_ELIGIBLE = """
    d.deleted_at IS NOT NULL
    AND d.deleted_at < {cutoff}
    AND NOT EXISTS (
        SELECT 1 FROM documents newer
        WHERE newer.customer_id = d.customer_id
          AND newer.doc_id = d.doc_id
          AND newer.version > d.version
    )
    AND NOT EXISTS (
        SELECT 1 FROM documents live
        WHERE live.customer_id = d.customer_id
          AND live.doc_id = d.doc_id
          AND live.valid_to IS NULL
          AND live.deleted_at IS NULL
    )
"""

_CANDIDATES_SQL = f"""
    SELECT d.doc_id, d.version, d.deleted_at, d.source_system, d.source_id,
           d.metadata->>'source_key' AS source_key,
           d.metadata->>'custom_document_id' AS custom_document_id
    FROM documents d
    WHERE d.customer_id = $1
      AND {_ELIGIBLE.format(cutoff="$2")}
      AND (d.deleted_at, d.doc_id) > ($3::timestamptz, $4::text)
    ORDER BY d.deleted_at, d.doc_id
    LIMIT $5
"""

# What a real run would delete, for --dry-run: an attended first run faces the
# whole historical backlog, and its size in chunk rows (26 GB table) and raw
# prefixes (one listing per custom-ingest document) is what sizes the job.
_DRY_RUN_SQL = f"""
    WITH eligible AS (
        SELECT d.doc_id, d.version, d.deleted_at, d.source_system
        FROM documents d
        WHERE d.customer_id = $1 AND {_ELIGIBLE.format(cutoff="$2")}
    )
    SELECT
        (SELECT count(*) FROM eligible) AS documents,
        (SELECT min(deleted_at) FROM eligible) AS oldest_deleted_at,
        (SELECT count(*) FROM eligible e
           JOIN documents d
             ON d.customer_id = $1 AND d.doc_id = e.doc_id AND d.version < e.version
        ) AS superseded_versions,
        (SELECT count(*) FROM eligible e
           JOIN chunks c
             ON c.customer_id = $1 AND c.doc_id = e.doc_id
            AND c.last_seen_version <= e.version
        ) AS chunks,
        (SELECT coalesce(jsonb_object_agg(source_system, n), '{{}}'::jsonb)
           FROM (SELECT source_system, count(*) AS n FROM eligible GROUP BY 1) s
        ) AS by_source
"""

# Re-validates and locks at the start of every batch. SKIP LOCKED: a document a
# writer is holding is left for the next run rather than waited on.
_LOCK_SQL = f"""
    SELECT d.doc_id, d.version
    FROM documents d
    WHERE d.customer_id = $1
      AND d.doc_id = ANY($2::text[])
      AND {_ELIGIBLE.format(cutoff="$3")}
    ORDER BY d.doc_id
    FOR UPDATE OF d SKIP LOCKED
"""

# Is a custom-ingest row for this ONE document still to be applied? Then it is
# being re-created and its raw prefix must not be cleared under it. Every
# source_event_id of a document starts with its document_event_prefix.
_IN_FLIGHT_SQL = """
    SELECT EXISTS (
        SELECT 1 FROM ingestion_queue
        WHERE customer_id = $1 AND source_system = $2 AND status = ANY($3::text[])
          AND starts_with(source_event_id, $4)
    )
"""

_MANUAL_UPLOAD_KEYS_SQL = """
    SELECT doc_id, payload_object_key, staging_object_key
    FROM manual_uploads
    WHERE customer_id = $1 AND doc_id = ANY($2::text[])
"""

# Chunks carry the version range they were live in. Bounding by the
# tombstone's version keeps a chunk a racing re-create just revived -- and the
# bound is repeated on the DELETE itself, not only in `doomed`: the chunk
# upsert revives a row IN PLACE (ON CONFLICT ... SET last_seen_version), and a
# revival that commits while this statement waits on the row is re-checked
# against the DELETE's own quals (EvalPlanQual), never against `doomed`'s
# snapshot. A closed code-graph tombstone gives the re-create no row lock to
# wait on, so this re-check is its only protection.
# Returns (selected, deleted): the loop moves on from chunks only when fewer
# than the batch were SELECTED, so rows a concurrent delete took, or a revival
# kept, never read as "no chunks left".
_DELETE_CHUNKS_SQL = """
    WITH doomed AS (
        SELECT c.chunk_id, t.version
        FROM unnest($2::text[], $3::int[]) AS t(doc_id, version)
        JOIN chunks c ON c.customer_id = $1 AND c.doc_id = t.doc_id
        WHERE c.last_seen_version <= t.version
        LIMIT $4
    ),
    gone AS (
        DELETE FROM chunks c
        USING doomed
        WHERE c.customer_id = $1
          AND c.chunk_id = doomed.chunk_id
          AND c.last_seen_version <= doomed.version
        RETURNING 1
    )
    SELECT (SELECT count(*) FROM doomed) AS selected,
           (SELECT count(*) FROM gone) AS deleted
"""

_DELETE_OLD_VERSIONS_SQL = """
    WITH doomed AS (
        SELECT d.doc_id, d.version
        FROM unnest($2::text[], $3::int[]) AS t(doc_id, version)
        JOIN documents d ON d.customer_id = $1 AND d.doc_id = t.doc_id
        WHERE d.version < t.version
        LIMIT $4
    ),
    gone AS (
        DELETE FROM documents d
        USING doomed
        WHERE d.customer_id = $1
          AND d.doc_id = doomed.doc_id
          AND d.version = doomed.version
        RETURNING 1
    )
    SELECT count(*) FROM gone
"""

_DELETE_FAILED_CHUNKS_SQL = """
    DELETE FROM failed_chunks f
    USING unnest($2::text[], $3::int[]) AS t(doc_id, version)
    WHERE f.customer_id = $1 AND f.doc_id = t.doc_id AND f.doc_version <= t.version
"""

_DELETE_TOMBSTONES_SQL = """
    DELETE FROM documents d
    USING unnest($2::text[], $3::int[]) AS t(doc_id, version)
    WHERE d.customer_id = $1 AND d.doc_id = t.doc_id AND d.version <= t.version
"""

_GONE_SQL = """
    SELECT t.doc_id FROM unnest($2::text[]) AS t(doc_id)
    WHERE NOT EXISTS (
        SELECT 1 FROM documents d
        WHERE d.customer_id = $1 AND d.doc_id = t.doc_id
    )
"""

#: Rows that name a document by id and nothing else. Run only for documents
#: with no version left. (stage, table, statement). inferred_edges_queue,
#: manual_uploads and both pending_edges statements are lookups on an index
#: keyed (customer_id, <that id>) -- without one, a delete reads the whole
#: table once per batch, which is what timed the first production run out
#: (migration 0141). github_document_bindings has only its primary key
#: (customer_id, installation_id, doc_id) and is small: GitHub documents only.
#:
#: pending_edges is TWO statements, not one `from ... OR to ...`: each side has
#: its own partial index (WHERE <side>_label = 'Document'), and the label is a
#: literal here because a partial index serves only a query that restates its
#: predicate. A row with both endpoints gone is deleted by the first and simply
#: not found by the second.
_BY_DOC_ID: tuple[tuple[str, str, str], ...] = (
    (
        "inferred_edges_queue",
        "inferred_edges_queue",
        "DELETE FROM inferred_edges_queue "
        "WHERE customer_id = $1 AND anchor_doc_id = ANY($2::text[])",
    ),
    (
        "github_document_bindings",
        "github_document_bindings",
        "DELETE FROM github_document_bindings "
        "WHERE customer_id = $1 AND doc_id = ANY($2::text[])",
    ),
    (
        "manual_uploads",
        "manual_uploads",
        "DELETE FROM manual_uploads WHERE customer_id = $1 AND doc_id = ANY($2::text[])",
    ),
    (
        "pending_edges.from",
        "pending_edges",
        f"DELETE FROM pending_edges WHERE customer_id = $1"
        f" AND from_label = '{_DOC_LABEL}' AND from_canonical_id = ANY($2::text[])",
    ),
    (
        "pending_edges.to",
        "pending_edges",
        f"DELETE FROM pending_edges WHERE customer_id = $1"
        f" AND to_label = '{_DOC_LABEL}' AND to_canonical_id = ANY($2::text[])",
    ),
)

# ACL resource ids are per-source: sometimes the doc_id, sometimes the
# source_id, sometimes a container (a GitHub repo). Only a row naming THIS
# document by one of its own ids goes, and only while no surviving document of
# the same source answers to that id.
#
# Written as equality lookups, because the natural spelling cannot use an
# index: `a.resource_id IN (g.doc_id, g.source_id)` joined against the tenant's
# whole acl_snapshots, and a surviving-document check of
# `x.doc_id = a.resource_id OR x.source_id = a.resource_id` became a BitmapOr
# over the TRIGRAM index on source_id for every matched row. So each document
# contributes its two ids as separate (source, id) rows, each an index lookup
# on idx_acl_snapshots_resource, and "no survivor answers to it" is two
# NOT EXISTS: by doc_id (documents_pkey) and by source_id
# (idx_documents_customer_source, whose source_system is the row's own).
# NOT EXISTS (p OR q) is NOT EXISTS p AND NOT EXISTS q, and a row matched by
# both ids is still deleted once, so what goes is exactly what went before.
_DELETE_ACL_SQL = """
    WITH ids AS (
        SELECT g.source_system, r.resource_id
        FROM unnest($2::text[], $3::text[], $4::text[]) AS g(doc_id, source_system, source_id)
        CROSS JOIN LATERAL (VALUES (g.doc_id), (g.source_id)) AS r(resource_id)
    )
    DELETE FROM acl_snapshots a
    USING ids
    WHERE a.customer_id = $1
      AND a.resource_id = ids.resource_id
      AND a.source_system = ids.source_system
      AND NOT EXISTS (
          SELECT 1 FROM documents x
          WHERE x.customer_id = $1
            AND x.doc_id = a.resource_id
            AND x.source_system = a.source_system
      )
      AND NOT EXISTS (
          SELECT 1 FROM documents x
          WHERE x.customer_id = $1
            AND x.source_system = a.source_system
            AND x.source_id = a.resource_id
      )
"""

# Endpoints on the far side of each Document node's edges, tagged with the
# deleted document's source. Read BEFORE the node goes: the cascade takes the
# edges with it.
_NEIGHBOURS_SQL = """
    WITH doc_nodes AS (
        SELECT n.node_id, g.source_system
        FROM unnest($2::text[], $3::text[]) AS g(doc_id, source_system)
        JOIN graph_nodes n
          ON n.customer_id = $1 AND n.label = $4 AND n.canonical_id = g.doc_id
    )
    SELECT e.to_node_id AS node_id, dn.source_system
    FROM doc_nodes dn
    JOIN graph_edges e ON e.customer_id = $1 AND e.from_node_id = dn.node_id
    UNION
    SELECT e.from_node_id, dn.source_system
    FROM doc_nodes dn
    JOIN graph_edges e ON e.customer_id = $1 AND e.to_node_id = dn.node_id
"""

# Taken BEFORE deciding which documents are gone, and the decision is a later
# statement, so it reads everything committed up to it (READ COMMITTED). A
# closed code-graph tombstone gives a re-connect no row lock to wait on: it
# inserts version N+1 and upserts the document's node alongside it. Holding the
# node first, either that writer is already past it -- its commit is what we
# waited for, so _GONE_SQL sees N+1 and the node stays -- or it waits for this
# batch, then re-creates a fresh node. Deciding first and deleting later would
# delete the node the writer just updated, and every edge it wrote with it.
_LOCK_DOC_NODES_SQL = """
    SELECT n.node_id FROM graph_nodes n
    WHERE n.customer_id = $1 AND n.label = $3 AND n.canonical_id = ANY($2::text[])
    ORDER BY n.node_id
    FOR UPDATE
"""

_DELETE_DOC_NODES_SQL = """
    DELETE FROM graph_nodes n
    USING unnest($2::text[]) AS g(doc_id)
    WHERE n.customer_id = $1 AND n.label = $3 AND n.canonical_id = g.doc_id
    RETURNING n.node_id
"""

_LOCK_NODES_SQL = """
    SELECT node_id FROM graph_nodes
    WHERE customer_id = $1 AND node_id = ANY($2::bigint[])
    ORDER BY node_id
    FOR UPDATE SKIP LOCKED
"""

# A neighbour dies only when the deletion left it with nothing: no edge either
# way, at least one provenance row and every one of them from a source whose
# document it was attached to (a node no source claims is not ours to judge --
# cross_repo_deps writes some without provenance), and not the node of a
# document that still exists.
_DELETE_ORPHANS_SQL = """
    WITH cand AS (
        SELECT * FROM unnest($2::bigint[], $3::text[]) AS c(node_id, source_system)
    )
    DELETE FROM graph_nodes n
    WHERE n.customer_id = $1
      AND n.node_id IN (SELECT node_id FROM cand)
      AND NOT EXISTS (SELECT 1 FROM graph_edges e WHERE e.from_node_id = n.node_id)
      AND NOT EXISTS (SELECT 1 FROM graph_edges e WHERE e.to_node_id = n.node_id)
      AND EXISTS (SELECT 1 FROM graph_node_provenance p WHERE p.node_id = n.node_id)
      AND NOT EXISTS (
          SELECT 1 FROM graph_node_provenance p
          WHERE p.node_id = n.node_id
            AND p.source_system NOT IN (
                SELECT c.source_system FROM cand c WHERE c.node_id = n.node_id
            )
      )
      AND NOT (
          n.label = $4
          AND EXISTS (
              SELECT 1 FROM documents d
              WHERE d.customer_id = $1 AND d.doc_id = n.canonical_id
          )
      )
    RETURNING n.node_id
"""

# Degree is maintained incrementally by the graph writer; the cascade removed
# edges behind its back, so survivors are recounted from the rows (the shape
# scripts/retire_experiment_nodes.py uses).
_RECOUNT_DEGREE_SQL = """
    UPDATE graph_nodes n
       SET degree = (SELECT count(*) FROM graph_edges e WHERE e.from_node_id = n.node_id)
                  + (SELECT count(*) FROM graph_edges e WHERE e.to_node_id = n.node_id)
     WHERE n.customer_id = $1 AND n.node_id = ANY($2::bigint[])
"""

_DELETE_NODE_QUEUE_SQL = """
    DELETE FROM node_post_write_queue
    WHERE customer_id = $1 AND node_id = ANY($2::bigint[])
"""


#: What the purge was doing when a tenant failed or a group was contended,
#: named in `tombstone_purge.tenant_failed` / `.group_contended`. Set before
#: every statement and storage call. The first production failure logged only
#: `error_type=TimeoutError error=""` -- asyncpg's client command_timeout raises
#: an empty TimeoutError -- and nothing said which of ~20 statements it was.
_STAGE: ContextVar[str] = ContextVar("tombstone_purge_stage", default="start")


def _at(stage: str) -> None:
    _STAGE.set(stage)


def _error_text(exc: BaseException) -> str:
    """str(exc), never empty: a bare TimeoutError from asyncpg's client-side
    timeouts stringifies to ''. Two raise it: command_timeout on a statement,
    and the connect timeout when the pool opens a new connection."""
    text = str(exc)
    if text:
        return text
    if isinstance(exc, TimeoutError):
        settings = get_settings()
        return (
            "no reply within the client command_timeout "
            f"({settings.db_statement_timeout_ms} ms, db_statement_timeout_ms), "
            f"or no connection within {settings.db_connect_timeout_seconds} s "
            "(db_connect_timeout_seconds)"
        )
    return repr(exc)


class _TenantBlocked(Exception):
    """The tenant went on hold or stopped being active: stop deleting."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class _BudgetExhausted(Exception):
    """--max-seconds passed: stop between batches, resume next run."""


@dataclass
class _Doc:
    doc_id: str
    version: int
    source_system: str
    source_id: str
    source_key: str | None
    custom_document_id: str | None


@dataclass
class TenantResult:
    customer_id: str
    documents: int = 0
    rows: Counter[str] = field(default_factory=Counter)
    r2_objects: int = 0
    failed_documents: int = 0
    in_flight_skipped: int = 0
    contended_groups: int = 0
    eligible: int = 0
    eligible_rows: Counter[str] = field(default_factory=Counter)
    oldest_deleted_at: datetime | None = None
    by_source: dict[str, int] = field(default_factory=dict)
    blocked: str | None = None
    budget_exhausted: bool = False


@asynccontextmanager
async def _gated(customer_id: str) -> AsyncIterator[asyncpg.Connection]:
    """A tenant-scoped transaction allowed to delete.

    The tenant GUC is bound (FORCE RLS), lock waits are bounded, and the tenant
    is re-checked under FOR SHARE on its customers row: a hold committed
    before this point stops it, and one set while it runs waits for it.
    """
    _at("gate")
    async with with_tenant(customer_id) as conn:
        await conn.execute(f"SET LOCAL lock_timeout = '{_LOCK_TIMEOUT}'")
        reason = await purge_blocked_reason(conn, customer_id, lock=True)
        if reason is not None:
            raise _TenantBlocked(reason)
        yield conn
        # Reached only when the block ended cleanly: with_tenant commits next,
        # under the same client timeout, and a stalled commit must not be
        # logged as the batch's last statement. A failure at this stage leaves
        # the outcome UNKNOWN (the server may have committed before the client
        # gave up): the run's counts miss that batch, and the next run's
        # eligibility re-check finds whatever is still there.
        _at("commit")


@functools.cache
def _store() -> ObjectStore:
    """Short storage timeouts, unlike get_store()'s botocore defaults (60 s,
    4 tries): each delete runs while this job holds the tenant's customers row
    FOR SHARE and the tombstone row FOR UPDATE, so a stalled store must cost
    seconds of those locks, not minutes, and must not carry the run past the
    chart's activeDeadlineSeconds. A failed call keeps the rows and retries
    next run. Built once per run (run_once clears it)."""
    return ObjectStore(connect_timeout=5, read_timeout=30, total_max_attempts=3)


def _rows(status: str) -> int:
    """asyncpg returns 'DELETE <n>'."""
    try:
        return int(status.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return 0


def _custom_ingest_identity(customer_id: str, doc: _Doc) -> tuple[str, str] | None:
    """(source_key, client document id), which address the raw payloads.

    From the doc_id first -- the engine composed it
    (shared.custom_ingest.custom_ingest_doc_id), so it cannot drift -- with the
    copies the handler stamps into metadata as the fallback.
    """
    head = f"{SourceSystem.CUSTOM_INGEST.value}:{customer_id}:"
    if doc.doc_id.startswith(head):
        encoded_key, sep, document_id = doc.doc_id[len(head):].partition(":")
        if sep and encoded_key and document_id:
            # Inverse of encode_source_key_for_doc_id: '%' is outside the
            # source_key charset, so '%3A' can only be an encoded ':'.
            return encoded_key.replace("%3A", ":"), document_id
    if doc.source_key and doc.custom_document_id:
        return doc.source_key, doc.custom_document_id
    return None


async def _purge_raw(
    customer_id: str,
    docs: list[_Doc],
    cutoff: datetime,
    deadline: float,
    result: TenantResult,
) -> list[_Doc]:
    """Step 1. Delete each document's own raw payloads; return the documents
    whose rows may now go. A document whose payloads could not be removed keeps
    its rows -- they are the only way the next run finds the payloads again."""
    failed: set[str] = set()
    try:
        return await _purge_raw_inner(customer_id, docs, cutoff, deadline, result, failed)
    finally:
        # Counted even when a lock timeout or a hold ends the group early, so a
        # storage failure earlier in it still makes the run exit 1.
        result.failed_documents += len(failed)


async def _purge_raw_inner(
    customer_id: str,
    docs: list[_Doc],
    cutoff: datetime,
    deadline: float,
    result: TenantResult,
    failed: set[str],
) -> list[_Doc]:
    custom = [d for d in docs if d.source_system == SourceSystem.CUSTOM_INGEST.value]
    _at("raw.manual_upload_keys")
    async with with_tenant(customer_id) as conn:
        manual_rows = await conn.fetch(
            _MANUAL_UPLOAD_KEYS_SQL, customer_id, [d.doc_id for d in docs]
        )
    if not custom and not manual_rows:
        return docs

    keys: dict[str, list[str]] = {}
    event_prefix: dict[str, str] = {}
    store = _store()
    _at("raw.bucket")
    try:
        bucket = await store.bucket_for(customer_id)
    except StorageUnavailable as exc:
        log.error("tombstone_purge.bucket_failed", customer_id=customer_id, error=str(exc))
        failed.update(d.doc_id for d in custom)
        failed.update(r["doc_id"] for r in manual_rows)
        return [d for d in docs if d.doc_id not in failed]

    # List first; in-flight and eligibility are read per document afterwards,
    # right before its delete. A re-create PUTs its payload and then enqueues,
    # so a payload this listing saw has its queue row visible to a later read
    # -- unless that read falls inside the route's own put-to-enqueue gap (a
    # few milliseconds; closing it needs a lock the route shares). The read
    # being per document and last keeps it as far from the listing as it can.
    for doc in custom:
        # One storage round trip per document, up to a group of them: the
        # budget is checked here as well as between row batches, so a slow
        # store stops the run cleanly instead of running into the pod's
        # activeDeadlineSeconds.
        if time.monotonic() >= deadline:
            raise _BudgetExhausted
        identity = _custom_ingest_identity(customer_id, doc)
        if identity is None:
            log.error(
                "tombstone_purge.unaddressable", customer_id=customer_id, doc_id=doc.doc_id
            )
            failed.add(doc.doc_id)
            continue
        event_prefix[doc.doc_id] = document_event_prefix(*identity)
        _at("raw.list")
        try:
            keys[doc.doc_id] = await store.list_keys(
                bucket, document_payload_prefix(customer_id, *identity)
            )
        except StorageUnavailable as exc:
            log.error(
                "tombstone_purge.list_failed",
                customer_id=customer_id,
                doc_id=doc.doc_id,
                error=str(exc),
            )
            failed.add(doc.doc_id)
    for row in manual_rows:
        for key in (row["payload_object_key"], row["staging_object_key"]):
            if key:
                keys.setdefault(row["doc_id"], []).append(key)

    ready: list[_Doc] = []
    for doc in docs:
        if doc.doc_id in failed:
            continue
        doc_keys = keys.get(doc.doc_id, [])
        prefix = event_prefix.get(doc.doc_id)
        if not doc_keys and prefix is None:
            ready.append(doc)  # nothing in R2 is this document's alone
            continue
        if time.monotonic() >= deadline:
            raise _BudgetExhausted
        try:
            outcome = await _clear_payloads(
                customer_id, doc.doc_id, doc_keys, prefix, store, bucket, cutoff
            )
        except StorageUnavailable as exc:
            log.error(
                "tombstone_purge.delete_failed",
                customer_id=customer_id,
                doc_id=doc.doc_id,
                error=str(exc),
            )
            failed.add(doc.doc_id)
            continue
        if outcome is None:
            result.in_flight_skipped += 1
            continue
        deleted, errors = outcome
        if errors:
            log.error(
                "tombstone_purge.delete_partial",
                customer_id=customer_id,
                doc_id=doc.doc_id,
                deleted=deleted,
                errors=errors,
            )
            failed.add(doc.doc_id)
            continue
        result.r2_objects += deleted
        ready.append(doc)
    return ready


async def _clear_payloads(
    customer_id: str,
    doc_id: str,
    keys: list[str],
    event_prefix: str | None,
    store: ObjectStore,
    bucket: str,
    cutoff: datetime,
) -> tuple[int, int] | None:
    """Delete one document's listed payloads: (deleted, errors), or None when
    the document must be left alone this run.

    Inside the gate, so a hold set now waits for this one document and the next
    one sees it. Then, re-read now and with the row locked: the document is
    still eligible (one re-created since the scan, whose new version is already
    applied, has no queue row in flight and the listing holds its LIVE payload;
    one a writer holds is left for the next run), and no custom-ingest row for
    it is waiting to be applied.
    """
    async with _gated(customer_id) as conn:
        _at("raw.lock")
        if not await conn.fetch(_LOCK_SQL, customer_id, [doc_id], cutoff):
            return None
        _at("raw.in_flight")
        if event_prefix is not None and await conn.fetchval(
            _IN_FLIGHT_SQL,
            customer_id,
            SourceSystem.CUSTOM_INGEST.value,
            [QueueStatus.PENDING.value, QueueStatus.PROCESSING.value],
            event_prefix,
        ):
            return None
        if not keys:
            return 0, 0
        _at("raw.delete")
        return await store.delete_keys(bucket, keys)


async def _purge_graph(
    conn: asyncpg.Connection, customer_id: str, docs: list[_Doc], counts: Counter[str]
) -> None:
    ids = [d.doc_id for d in docs]
    sources = [d.source_system for d in docs]
    _at("graph.neighbours")
    neighbours = await conn.fetch(_NEIGHBOURS_SQL, customer_id, ids, sources, _DOC_LABEL)
    _at("graph.doc_nodes")
    doc_nodes = [
        r["node_id"]
        for r in await conn.fetch(_DELETE_DOC_NODES_SQL, customer_id, ids, _DOC_LABEL)
    ]
    counts["graph_nodes"] += len(doc_nodes)
    gone_nodes = list(doc_nodes)

    deleted = set(doc_nodes)
    candidates = [
        (r["node_id"], r["source_system"]) for r in neighbours if r["node_id"] not in deleted
    ]
    if candidates:
        _at("graph.lock_neighbours")
        locked = {
            r["node_id"]
            for r in await conn.fetch(
                _LOCK_NODES_SQL, customer_id, sorted({c[0] for c in candidates})
            )
        }
        candidates = [c for c in candidates if c[0] in locked]
        if candidates:
            _at("graph.orphans")
            orphans = [
                r["node_id"]
                for r in await conn.fetch(
                    _DELETE_ORPHANS_SQL,
                    customer_id,
                    [c[0] for c in candidates],
                    [c[1] for c in candidates],
                    _DOC_LABEL,
                )
            ]
            counts["graph_nodes"] += len(orphans)
            gone_nodes.extend(orphans)
            survivors = sorted(locked - set(orphans))
            if survivors:
                _at("graph.recount_degree")
                await conn.execute(_RECOUNT_DEGREE_SQL, customer_id, survivors)
    if gone_nodes:
        _at("node_post_write_queue")
        counts["node_post_write_queue"] += _rows(
            await conn.execute(_DELETE_NODE_QUEUE_SQL, customer_id, gone_nodes)
        )


async def _finish(
    conn: asyncpg.Connection,
    customer_id: str,
    docs: list[_Doc],
    versions: list[int],
    counts: Counter[str],
) -> int:
    """Step 4, inside the batch's transaction. Returns documents fully gone."""
    ids = [d.doc_id for d in docs]
    _at("failed_chunks")
    counts["failed_chunks"] += _rows(
        await conn.execute(_DELETE_FAILED_CHUNKS_SQL, customer_id, ids, versions)
    )
    _at("tombstones")
    counts["documents"] += _rows(
        await conn.execute(_DELETE_TOMBSTONES_SQL, customer_id, ids, versions)
    )
    _at("graph.lock_doc_nodes")
    await conn.execute(_LOCK_DOC_NODES_SQL, customer_id, ids, _DOC_LABEL)
    _at("gone")
    gone_ids = {r["doc_id"] for r in await conn.fetch(_GONE_SQL, customer_id, ids)}
    gone = [d for d in docs if d.doc_id in gone_ids]
    if not gone:
        return 0
    gone_list = [d.doc_id for d in gone]
    for stage, table, sql in _BY_DOC_ID:
        _at(stage)
        counts[table] += _rows(await conn.execute(sql, customer_id, gone_list))
    _at("acl_snapshots")
    counts["acl_snapshots"] += _rows(
        await conn.execute(
            _DELETE_ACL_SQL,
            customer_id,
            gone_list,
            [d.source_system for d in gone],
            [d.source_id for d in gone],
        )
    )
    await _purge_graph(conn, customer_id, gone, counts)
    return len(gone)


async def _purge_rows(
    customer_id: str,
    docs: list[_Doc],
    cutoff: datetime,
    deadline: float,
    result: TenantResult,
) -> None:
    """Steps 2-4, one committed transaction per batch of rows."""
    pending = {d.doc_id: d for d in docs}
    while pending:
        if time.monotonic() >= deadline:
            raise _BudgetExhausted
        batch = Counter[str]()
        started = time.monotonic()
        async with _gated(customer_id) as conn:
            _at("rows.lock")
            locked = await conn.fetch(_LOCK_SQL, customer_id, list(pending), cutoff)
            if not locked:
                return
            ids = [r["doc_id"] for r in locked]
            versions = [r["version"] for r in locked]
            budget = TOMBSTONE_PURGE_BATCH_SIZE
            _at("chunks")
            chunks = await conn.fetchrow(
                _DELETE_CHUNKS_SQL, customer_id, ids, versions, budget
            )
            batch["chunks"] += chunks["deleted"]
            budget -= chunks["selected"]
            if budget > 0:
                _at("old_versions")
                n = await conn.fetchval(
                    _DELETE_OLD_VERSIONS_SQL, customer_id, ids, versions, budget
                )
                batch["documents"] += n
                budget -= n
            finished = 0
            if budget > 0:
                finished = await _finish(
                    conn, customer_id, [pending[i] for i in ids], versions, batch
                )
                for doc_id in ids:
                    pending.pop(doc_id)
        # Committed: only now do the counts describe something that happened.
        result.rows.update(batch)
        result.documents += finished
        log.info(
            "tombstone_purge.batch_committed",
            customer_id=customer_id,
            documents=finished,
            rows=dict(batch),
            seconds=round(time.monotonic() - started, 2),
        )
        await asyncio.sleep(_BATCH_PAUSE_SECONDS)


async def purge_tenant(
    customer_id: str,
    *,
    cutoff: datetime,
    deadline: float,
    dry_run: bool = False,
) -> TenantResult:
    """Purge one tenant's eligible tombstoned documents."""
    result = TenantResult(customer_id)
    if dry_run:
        _at("dry_run")
        async with with_tenant(customer_id) as conn:
            row = await conn.fetchrow(_DRY_RUN_SQL, customer_id, cutoff)
        result.eligible = int(row["documents"])
        result.eligible_rows.update(
            documents=result.eligible + int(row["superseded_versions"]),
            chunks=int(row["chunks"]),
        )
        result.oldest_deleted_at = row["oldest_deleted_at"]
        result.by_source = dict(json.loads(row["by_source"]))
        log.info(
            "tombstone_purge.would_delete",
            customer_id=customer_id,
            documents=result.eligible,
            rows=dict(result.eligible_rows),
            by_source=result.by_source,
            oldest_deleted_at=(
                result.oldest_deleted_at.isoformat() if result.oldest_deleted_at else None
            ),
            window_days=TOMBSTONE_PURGE_DAYS,
        )
        return result

    after = _WALK_START
    while True:
        if time.monotonic() >= deadline:
            result.budget_exhausted = True
            break
        _at("scan")
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(
                _CANDIDATES_SQL,
                customer_id,
                cutoff,
                after[0],
                after[1],
                TOMBSTONE_PURGE_DOCS_PER_GROUP,
            )
        if not rows:
            break
        after = (rows[-1]["deleted_at"], rows[-1]["doc_id"])
        docs = [
            _Doc(
                doc_id=r["doc_id"],
                version=r["version"],
                source_system=r["source_system"],
                source_id=r["source_id"],
                source_key=r["source_key"],
                custom_document_id=r["custom_document_id"],
            )
            for r in rows
        ]
        try:
            ready = await _purge_raw(customer_id, docs, cutoff, deadline, result)
            if ready:
                await _purge_rows(customer_id, ready, cutoff, deadline, result)
        except _TenantBlocked as exc:
            result.blocked = exc.reason
            log.warning(
                "tombstone_purge.tenant_blocked", customer_id=customer_id, reason=exc.reason
            )
            break
        except _BudgetExhausted:
            result.budget_exhausted = True
            break
        except (asyncpg.LockNotAvailableError, asyncpg.DeadlockDetectedError) as exc:
            # A writer holds rows this group needs. Nothing is lost: the
            # batch rolled back and the documents stay eligible.
            result.contended_groups += 1
            log.warning(
                "tombstone_purge.group_contended",
                customer_id=customer_id,
                stage=_STAGE.get(),
                error=type(exc).__name__,
            )
    log.info(
        "tombstone_purge.tenant_done",
        customer_id=customer_id,
        documents=result.documents,
        rows=dict(result.rows),
        r2_objects=result.r2_objects,
        failed_documents=result.failed_documents,
        in_flight_skipped=result.in_flight_skipped,
        contended_groups=result.contended_groups,
        blocked=result.blocked,
        budget_exhausted=result.budget_exhausted,
    )
    return result


async def run_once(
    *,
    dry_run: bool = False,
    customer: str | None = None,
    max_seconds: float = TOMBSTONE_PURGE_MAX_SECONDS,
) -> int:
    deadline = time.monotonic() + max_seconds
    _store.cache_clear()
    async with get_pool().acquire() as conn:
        # The database's clock, fixed once per run, so every batch agrees on
        # what "older than the window" means.
        cutoff = await conn.fetchval(
            "SELECT now() - make_interval(days => $1)", TOMBSTONE_PURGE_DAYS
        )
        # `customers` is deliberately not row-secured -- the readable tenant
        # list is what makes the per-tenant GUC loop possible (0119).
        rows = await conn.fetch(
            f"""
            SELECT c.customer_id, {purge_eligible_tenant_sql("c")} AS eligible
            FROM customers c
            ORDER BY 1
            """
        )
    candidates = [
        r["customer_id"]
        for r in rows
        if r["eligible"] and (customer is None or r["customer_id"] == customer)
    ]
    # Most overdue first. A run that stops at --max-seconds must not always
    # spend its budget on the same tenant while the last one in a fixed order
    # never gets reached.
    if customer is not None and customer not in candidates:
        log.warning(
            "tombstone_purge.customer_not_eligible",
            customer_id=customer,
            reason="absent, not active, or on legal hold",
        )
    oldest: dict[str, datetime] = {}
    failed = False
    for customer_id in candidates:
        try:
            async with with_tenant(customer_id) as conn:
                first = await conn.fetchrow(
                    _CANDIDATES_SQL, customer_id, cutoff, *_WALK_START, 1
                )
        except Exception as exc:
            # Same rule as the purge loop below: one tenant's failure makes
            # the run red but must not stop every other tenant's deletions.
            failed = True
            log.error(
                "tombstone_purge.tenant_failed",
                customer_id=customer_id,
                stage="discover",
                error_type=type(exc).__name__,
                error=_error_text(exc),
            )
            continue
        if first is not None:
            oldest[customer_id] = first["deleted_at"]
    tenants = sorted(oldest, key=lambda c: (oldest[c], c))
    log.info(
        "tombstone_purge.start",
        tenants=len(tenants),
        tenants_without_work=len(candidates) - len(tenants),
        skipped_tenants=sum(1 for r in rows if not r["eligible"]),
        window_days=TOMBSTONE_PURGE_DAYS,
        cutoff=cutoff.isoformat(),
        dry_run=dry_run,
    )
    budget_exhausted = False
    totals = Counter[str]()
    documents = 0
    eligible = 0
    for customer_id in tenants:
        _at("start")
        try:
            result = await purge_tenant(
                customer_id, cutoff=cutoff, deadline=deadline, dry_run=dry_run
            )
        except Exception as exc:
            # One tenant's failure must not starve the rest, and it DOES make
            # the run red: a silent partial purge reads as "nothing to do".
            failed = True
            log.error(
                "tombstone_purge.tenant_failed",
                customer_id=customer_id,
                stage=_STAGE.get(),
                error_type=type(exc).__name__,
                error=_error_text(exc),
            )
            continue
        documents += result.documents
        totals.update(result.rows)
        totals.update(result.eligible_rows)
        eligible += result.eligible
        failed = failed or result.failed_documents > 0
        if result.budget_exhausted:
            budget_exhausted = True
            break
    log.info(
        "tombstone_purge.done",
        tenants=len(tenants),
        # A dry run deletes nothing: its totals are what a real run WOULD
        # delete, reported as `would_delete_documents` so the two never mix.
        documents=documents,
        would_delete_documents=eligible if dry_run else None,
        rows=dict(totals),
        failed=failed,
        budget_exhausted=budget_exhausted,
        dry_run=dry_run,
    )
    if failed:
        return EXIT_FAILED
    if budget_exhausted:
        log.warning("tombstone_purge.budget_exhausted", max_seconds=max_seconds)
        return EXIT_BUDGET_EXHAUSTED
    return EXIT_OK


async def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Hard-delete documents whose tombstone is older than "
        f"{TOMBSTONE_PURGE_DAYS} days."
    )
    parser.add_argument("--dry-run", action="store_true", help="count, delete nothing")
    parser.add_argument(
        "--customer",
        help="only this tenant (still skipped unless active and not on legal hold)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=TOMBSTONE_PURGE_MAX_SECONDS,
        help="stop starting new batches after this long (default %(default)s)",
    )
    args = parser.parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    await init_pool(settings)
    try:
        return await run_once(
            dry_run=args.dry_run, customer=args.customer, max_seconds=args.max_seconds
        )
    finally:
        await close_pool()


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
