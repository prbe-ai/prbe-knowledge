"""Delete captured coding-agent sessions: chosen by id, or every session one person authored.

A customer may ask for specific Claude Code / Codex / pi sessions to be erased,
or for everything one person captured, and it has to be finished within 30
days (backups trail by 8). The source purge (kb/purge_routes.py) can only drop
a whole integration; this removes single sessions, everywhere they live, and
keeps them from coming back.

Route contract (internal, same auth as every internal route: the
X-Internal-Knowledge-Key header, tenant from X-Prbe-Customer, never the body)

    POST /api/session-deletions
        {
          "session_ids": ["<id>", ...],             # exactly one of these two
          "author": {"employee_id": "<uuid>", "email": "a@b.c"},   # either or both keys
          "sources": ["claude_code", "codex", "pi"],  # optional filter, default all three
          "dry_run": true,                          # DEFAULT: report, change nothing
          "reason": "customer request",             # required to apply
          "ticket": "prbe-ai/research-os#123",      # optional
          "deep_scan": false                        # also list raw/<source>/<customer>/
        }
        dry run     -> 200 {"dry_run": true, "legal_hold": null|"<why>", "sessions": [
                             {"source", "session_id", "already_deleted", "rows": {table: n},
                              "r2_objects": n, "r2_referenced_keys": n}], "totals": {...},
                             "unattributed_sessions": n, "not_covered": [...]}
        apply       -> 202 {"deletion_id", "status": "running", "sessions": [{"source", "session_id"}]}
                       200 {"status": "nothing_to_delete", "sessions": []}
        legal hold  -> 423 {"detail": {"reason": "legal_hold", "legal_hold": "<why>"}}
        no tenant   -> 404
    GET /api/session-deletions/{deletion_id}
        -> 200 {"deletion_id", "status": running|done|failed|held, "sessions": [
                 {"source", "session_id", "status", "requested_at", "attempted_at",
                  "deleted_at", "result", "error"}]}

An explicit id that matches nothing is still recorded, for every requested
source: a session not uploaded yet must not be accepted later either.

What one session's deletion covers
----------------------------------
Rows: every version of the session document and of every unit document
extracted from it (`parent_doc_id`), their chunks, failed_chunks, acl_snapshots
and inferred_edges_queue rows; the session's Document and AgentSession graph
nodes with every edge touching them (surviving endpoints' `degree` decremented),
their provenance, post-write-queue and pending_edges rows; its ingestion_queue
row(s), ingestion_events, session_streams and session_batch_receipts. By author,
the person's Person node too, when nothing but these sessions asserts it.

Raw objects: `raw/<src>/<customer>/sessions-v2/<sid>/` (protocol-2 batches),
`raw/<src>/<customer>/<sid>/` (the idle sweep's finalize marker and the
extraction cache), and every protocol-1 batch. Those live under a DATE folder
(`raw/<src>/<customer>/YYYY/MM/DD/<sid>:<seq>.json`, kb/ingestion_app._payload_key)
shared with every other session, so they are found through the rows that
reference them -- queue `payload_s3_key(s)`, receipts, ingestion_events -- and
written to `session_deletions.pending_keys` BEFORE those rows are removed, in
the same transaction. `deep_scan` also lists the whole `raw/<src>/<customer>/`
folder and matches file names, for an object no row ever referenced (a crash
between the upload and its queue insert).

Order: record the session (from then on every writer refuses it, see
engine/shared/session_suppression.py) -> remove rows (under the session lock,
journaling keys in the same commit) -> remove objects -> verify. Rows go before
objects so the idle sweep and the worker, which act on the queue row, have
nothing left to act on while objects are removed. A crash anywhere leaves a
`pending` row holding the key list; re-POSTing resumes. A worker already
mining the session when the deletion lands can still write extraction-cache
objects until its pass ends (its row writes are refused), so a run that found
the session in flight waits `IN_FLIGHT_GRACE_S` and sweeps again.

NOT covered here (listed in every dry run as `not_covered`): see NOT_COVERED.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from engine.shared.constants import (
    AGENT_SESSION_SOURCES,
    NodeLabel,
    agent_session_canonical_id,
)
from engine.shared.db import raw_conn, with_tenant
from engine.shared.exceptions import StorageNotFound, StorageUnavailable
from engine.shared.session_signals import is_cron_marker_key, storage_id
from engine.shared.session_suppression import lock_session
from engine.shared.storage import get_store
from kb.admin_routes import verify_internal_knowledge_key

log = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/api/session-deletions",
    tags=["session-deletions"],
    dependencies=[Depends(verify_internal_knowledge_key)],
)

AGENT_SOURCES: tuple[str, ...] = tuple(s.value for s in AGENT_SESSION_SOURCES)

#: The shape research-os accepts for a session id (app/runs/agent_session.py),
#: which is also what makes the raw prefixes below safe to delete under: no `/`,
#: nothing short enough to be a date folder.
_SESSION_ID_RE = re.compile(r"\A[A-Za-z0-9._:-]{8,200}\Z")
#: Names that sit at the session-folder level of `raw/<src>/<customer>/` and
#: are NOT sessions. Deleting `raw/<src>/<customer>/sessions-v2/` would erase
#: every protocol-2 session of the tenant.
_RESERVED_IDS = frozenset({"sessions-v2"})

#: How long a run waits before sweeping again when a worker was mid-pass on the
#: session: a pass reclaims after 300 s without a heartbeat (worker.py), so by
#: then it has either finished (and been refused) or been abandoned.
IN_FLIGHT_GRACE_S = 360.0
_R2_CONCURRENCY = 16
_UNINDEXED_SCAN_LIMIT = 5000
_UNINDEXED_KEYS_READ = 3

#: Stores that can hold a trace of a deleted session and are NOT removed here.
NOT_COVERED: tuple[dict[str, str], ...] = (
    {
        "store": "query_traces.request/response, usage_events.summary",
        "why": "search logs can quote a session passage; no per-document key to find it by "
        "(retention of these logs is its own task)",
    },
    {
        "store": "retrieve_pages.items",
        "why": "paging snapshots of search results; not keyed by document, self-delete after 1 day",
    },
    {
        "store": "graph_edges with confidence INFERRED between OTHER nodes",
        "why": "the side-worker's `why` text was generated from a bundle that may have included "
        "the session, but an edge does not record its anchor document; edges touching the "
        "session's own nodes ARE deleted",
    },
    {
        "store": "entity_merge_node_snapshot / entity_merge_edge_snapshot",
        "why": "merge-undo snapshots of node properties; not keyed by document",
    },
    {
        "store": "serve_ledger.session_id",
        "why": "append-only exposure log (session id + clause ids, no content); RLS denies DELETE by design",
    },
    {
        "store": "Person graph node (by-id deletions)",
        "why": "a person is shared by their other sessions; removed only by an author deletion, "
        "and only when nothing else asserts it",
    },
    {
        "store": "Postgres index/heap residue, the standby's pg_search copy, backups",
        "why": "gone after VACUUM / index rebuild (retention T23) and backup expiry (<= 8 days)",
    },
    {
        "store": "research-os content derived from the session (overview pages, session "
        "updates, digests, agent_session_activity)",
        "why": "lives in research-os; its half of this deletion runs there",
    },
)


class SessionDeletionError(Exception):
    """A deletion could not proceed; `status` is the HTTP answer."""

    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def valid_session_id(session_id: object) -> bool:
    return (
        isinstance(session_id, str)
        and bool(_SESSION_ID_RE.match(session_id))
        and session_id not in _RESERVED_IDS
    )


@dataclass(frozen=True, order=True)
class SessionRef:
    source: str
    session_id: str

    def session_doc_id(self, customer_id: str) -> str:
        # kb/handlers/claude_code.py _build_session_doc; codex/pi share it.
        return f"{self.source}:{customer_id}:{self.session_id}"

    def agent_node_id(self) -> str:
        return agent_session_canonical_id(self.source, self.session_id)

    def prefixes(self, customer_id: str) -> tuple[str, str]:
        """Folders holding only this session's objects. Trailing `/` is load-bearing."""
        return (
            # kb/session_receipts.accept (protocol-2 batches)
            f"raw/{self.source}/{customer_id}/sessions-v2/{self.session_id}/",
            # session_signals.cron_marker_key + extraction_cache.SegmentCache
            f"raw/{self.source}/{customer_id}/{self.session_id}/",
        )

    def as_dict(self) -> dict[str, str]:
        return {"source": self.source, "session_id": self.session_id}


@dataclass
class Inventory:
    """Everything the database holds for one session, read under its lock."""

    doc_ids: list[str] = field(default_factory=list)
    node_ids: list[int] = field(default_factory=list)
    canonical_ids: list[str] = field(default_factory=list)
    queue_ids: list[int] = field(default_factory=list)
    event_ids: list[int] = field(default_factory=list)
    keys: set[str] = field(default_factory=set)
    in_flight: bool = False
    counts: dict[str, int] = field(default_factory=dict)


# --- tenant state -------------------------------------------------------------


async def legal_hold(customer_id: str) -> str | None:
    """The tenant's legal-hold marker, or None. Raises 404 for an unknown tenant.

    `customers.metadata.legal_hold` (retention plan T3). Any value but absent,
    null, false or "" counts as held: deleting under an ambiguous hold is the
    failure that cannot be undone.
    """
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT metadata->>'legal_hold' AS hold FROM customers WHERE customer_id = $1",
            customer_id,
        )
    if row is None:
        raise SessionDeletionError(404, f"unknown customer {customer_id!r}")
    hold: str | None = row["hold"]
    if hold is None or hold.strip().lower() in ("", "false", "null"):
        return None
    return hold


def _held(hold: str) -> SessionDeletionError:
    return SessionDeletionError(
        423,
        {
            "reason": "legal_hold",
            "message": "this customer is under legal hold; nothing was deleted",
            "legal_hold": hold,
        },
    )


# --- selection ----------------------------------------------------------------


@dataclass
class Selection:
    refs: list[SessionRef]
    #: Employee ids the matched sessions were authored or uploaded by; an author
    #: deletion removes their Person node when nothing else asserts it.
    person_ids: set[str] = field(default_factory=set)
    unattributed: int = 0
    skipped_invalid: list[str] = field(default_factory=list)


async def select_by_ids(
    customer_id: str, session_ids: list[str], sources: list[str]
) -> Selection:
    """Each id under the sources it exists in; under every requested source if none."""
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT source_system, source_id AS session_id FROM documents
             WHERE customer_id = $1 AND source_system = ANY($2::text[])
               AND source_id = ANY($3::text[])
            UNION
            SELECT source_system, session_id FROM session_streams
             WHERE customer_id = $1 AND source_system = ANY($2::text[])
               AND session_id = ANY($3::text[])
            UNION
            SELECT source_system, source_event_id FROM ingestion_queue
             WHERE customer_id = $1 AND source_system = ANY($2::text[])
               AND source_event_id = ANY($3::text[])
            UNION
            SELECT source_system, session_id FROM session_deletions
             WHERE customer_id = $1 AND source_system = ANY($2::text[])
               AND session_id = ANY($3::text[])
            """,
            customer_id,
            sources,
            session_ids,
        )
    found: dict[str, set[str]] = {}
    for r in rows:
        found.setdefault(r["session_id"], set()).add(r["source_system"])
    refs = [
        SessionRef(source, sid)
        for sid in session_ids
        for source in sorted(found.get(sid) or sources)
    ]
    return Selection(refs=sorted(set(refs)))


async def select_by_author(
    customer_id: str,
    sources: list[str],
    *,
    employee_id: str | None,
    email: str | None,
) -> Selection:
    """Sessions whose documents name this person, plus unprocessed ones whose raw
    batches do.

    Protocol 1 records the author as `author_id` (+ `employee_email`); protocol 2
    records only who UPLOADED it (`uploader_id`, `uploader_email`, author_id NULL)
    because a copied transcript's author is unverified. For a deletion request
    both mean "this person's capture", so both match.
    """
    selection = Selection(refs=[])
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT source_system,
                   CASE WHEN parent_doc_id IS NULL THEN source_id
                        ELSE split_part(source_id, ':', 1) END AS session_id,
                   COALESCE(author_id, metadata->>'uploader_id') AS person_id
              FROM documents
             WHERE customer_id = $1 AND source_system = ANY($2::text[])
               AND (
                    ($3::text IS NOT NULL AND (author_id = $3 OR metadata->>'uploader_id' = $3))
                 OR ($4::text IS NOT NULL AND (
                        lower(metadata->>'employee_email') = lower($4)
                     OR lower(metadata->>'uploader_email') = lower($4)))
               )
            """,
            customer_id,
            sources,
            employee_id,
            email,
        )
        unindexed = await conn.fetch(
            """
            SELECT q.source_system, q.source_event_id, q.payload_s3_key, q.payload_s3_keys
              FROM ingestion_queue q
             WHERE q.customer_id = $1 AND q.source_system = ANY($2::text[])
               AND NOT EXISTS (
                     SELECT 1 FROM documents d
                      WHERE d.customer_id = q.customer_id
                        AND d.source_system = q.source_system
                        AND d.source_id = split_part(q.source_event_id, ':', 1))
             ORDER BY q.queue_id
             LIMIT $3
            """,
            customer_id,
            sources,
            _UNINDEXED_SCAN_LIMIT,
        )
    refs: set[SessionRef] = set()
    for r in rows:
        if valid_session_id(r["session_id"]):
            refs.add(SessionRef(r["source_system"], r["session_id"]))
            if r["person_id"]:
                selection.person_ids.add(r["person_id"])
        else:
            selection.skipped_invalid.append(r["session_id"])
    if employee_id:
        selection.person_ids.add(employee_id)

    # Sessions never processed (pending, dead-lettered, protocol-2 streams
    # still open) have no document to say who they belong to. Their batches do.
    store = get_store()
    bucket = await store.bucket_for(customer_id) if unindexed else ""
    sem = asyncio.Semaphore(_R2_CONCURRENCY)

    async def identify(row: Any) -> tuple[str, str | None, str | None] | None:
        async with sem:
            return await _batch_identity(store, bucket, row)

    identities = await asyncio.gather(*(identify(r) for r in unindexed))
    for r, identity in zip(unindexed, identities, strict=True):
        if identity is None:
            selection.unattributed += 1
            continue
        sid, emp, mail = identity
        if (employee_id and emp == employee_id) or (
            email and mail and mail.lower() == email.lower()
        ):
            if valid_session_id(sid):
                refs.add(SessionRef(r["source_system"], sid))
                if emp:
                    selection.person_ids.add(emp)
            else:
                selection.skipped_invalid.append(sid)
    selection.refs = sorted(refs)
    return selection


async def _batch_identity(
    store: Any, bucket: str, row: Any
) -> tuple[str, str | None, str | None] | None:
    """(session_id, employee_id, employee_email) from a queue row's raw batches."""
    keys = [k for k in (row["payload_s3_keys"] or []) if not is_cron_marker_key(k)]
    if not keys and row["payload_s3_key"]:
        keys = [row["payload_s3_key"]]
    for key in keys[:_UNINDEXED_KEYS_READ]:
        try:
            envelope = json.loads(await store.get(bucket, key))
        except (StorageNotFound, StorageUnavailable, ValueError):
            continue
        payload = envelope.get("payload", envelope) if isinstance(envelope, dict) else None
        if not isinstance(payload, dict):
            continue
        sid = payload.get("session_id") or row["source_event_id"].split(":", 1)[0]
        emp = payload.get("employee_id")
        mail = payload.get("employee_email")
        if emp or mail:
            return str(sid), emp if isinstance(emp, str) else None, mail if isinstance(mail, str) else None
    return None


# --- inventory ------------------------------------------------------------------

_ID_OR_CHILD = "({col} = $3 OR left({col}, length($3) + 1) = $3 || ':')"


async def inventory(
    conn: Any, customer_id: str, ref: SessionRef, *, with_counts: bool = True
) -> Inventory:
    """Read every row of one session. Caller holds the session lock (or accepts a
    dry run's snapshot) and the tenant GUC. `with_counts=False` reads only what
    a deletion needs to act on: ids and raw keys."""
    inv = Inventory()
    session_doc = ref.session_doc_id(customer_id)
    inv.doc_ids = [
        r["doc_id"]
        for r in await conn.fetch(
            f"""
            SELECT DISTINCT doc_id FROM documents
             WHERE customer_id = $1 AND source_system = $2
               AND ({_ID_OR_CHILD.format(col="source_id")} OR parent_doc_id = $4 OR doc_id = $4)
            """,
            customer_id,
            ref.source,
            ref.session_id,
            session_doc,
        )
    ]
    inv.canonical_ids = sorted({*inv.doc_ids, session_doc, ref.agent_node_id()})
    nodes = await conn.fetch(
        """
        SELECT node_id FROM graph_nodes
         WHERE customer_id = $1
           AND ((label = $2 AND canonical_id = ANY($4::text[])) OR (label = $3 AND canonical_id = $5))
        """,
        customer_id,
        NodeLabel.DOCUMENT.value,
        NodeLabel.AGENT_SESSION.value,
        [*inv.doc_ids, session_doc],
        ref.agent_node_id(),
    )
    inv.node_ids = sorted(r["node_id"] for r in nodes)
    queue = await conn.fetch(
        f"""
        SELECT queue_id, status, payload_s3_key, payload_s3_keys FROM ingestion_queue
         WHERE customer_id = $1 AND source_system = $2 AND {_ID_OR_CHILD.format(col="source_event_id")}
        """,
        customer_id,
        ref.source,
        ref.session_id,
    )
    inv.queue_ids = [r["queue_id"] for r in queue]
    inv.in_flight = any(r["status"] == "processing" for r in queue)
    for r in queue:
        inv.keys.update(k for k in (r["payload_s3_keys"] or []) if k)
        if r["payload_s3_key"]:
            inv.keys.add(r["payload_s3_key"])
    events = await conn.fetch(
        f"""
        SELECT event_id, payload_s3_key FROM ingestion_events
         WHERE customer_id = $1 AND source_system = $2 AND {_ID_OR_CHILD.format(col="source_event_id")}
        """,
        customer_id,
        ref.source,
        ref.session_id,
    )
    inv.event_ids = [r["event_id"] for r in events]
    inv.keys.update(r["payload_s3_key"] for r in events if r["payload_s3_key"])
    receipts = await conn.fetch(
        "SELECT payload_key FROM session_batch_receipts "
        "WHERE customer_id = $1 AND source_system = $2 AND session_id = $3",
        customer_id,
        ref.source,
        ref.session_id,
    )
    inv.keys.update(r["payload_key"] for r in receipts if r["payload_key"])
    # Never journal a key outside this tenant's raw folder for this source: a
    # corrupt reference must not become a delete of someone else's object.
    own = f"raw/{ref.source}/{customer_id}/"
    inv.keys = {k for k in inv.keys if k.startswith(own)}
    if not with_counts:
        return inv

    c = inv.counts
    c["documents"] = await conn.fetchval(
        "SELECT count(*) FROM documents WHERE customer_id = $1 AND doc_id = ANY($2::text[])",
        customer_id,
        inv.doc_ids,
    )
    for table, column in (
        ("chunks", "doc_id"),
        ("failed_chunks", "doc_id"),
        ("inferred_edges_queue", "anchor_doc_id"),
    ):
        c[table] = await conn.fetchval(
            f"SELECT count(*) FROM {table} WHERE customer_id = $1 AND {column} = ANY($2::text[])",
            customer_id,
            inv.doc_ids,
        )
    c["acl_snapshots"] = await conn.fetchval(
        "SELECT count(*) FROM acl_snapshots WHERE customer_id = $1 "
        "AND resource_type = 'document' AND resource_id = ANY($2::text[])",
        customer_id,
        inv.doc_ids,
    )
    c["graph_nodes"] = len(inv.node_ids)
    c["graph_edges"] = await conn.fetchval(
        "SELECT count(*) FROM graph_edges WHERE customer_id = $1 "
        "AND (from_node_id = ANY($2::bigint[]) OR to_node_id = ANY($2::bigint[]))",
        customer_id,
        inv.node_ids,
    )
    c["graph_node_provenance"] = await conn.fetchval(
        "SELECT count(*) FROM graph_node_provenance WHERE customer_id = $1 AND node_id = ANY($2::bigint[])",
        customer_id,
        inv.node_ids,
    )
    c["node_post_write_queue"] = await conn.fetchval(
        "SELECT count(*) FROM node_post_write_queue WHERE customer_id = $1 AND node_id = ANY($2::bigint[])",
        customer_id,
        inv.node_ids,
    )
    c["pending_edges"] = await conn.fetchval(
        "SELECT count(*) FROM pending_edges WHERE customer_id = $1 AND ("
        "missing_canonical_id = ANY($2::text[]) OR from_canonical_id = ANY($2::text[]) "
        "OR to_canonical_id = ANY($2::text[]))",
        customer_id,
        inv.canonical_ids,
    )
    c["ingestion_queue"] = len(inv.queue_ids)
    c["ingestion_events"] = len(inv.event_ids)
    c["session_streams"] = await conn.fetchval(
        "SELECT count(*) FROM session_streams WHERE customer_id = $1 AND source_system = $2 AND session_id = $3",
        customer_id,
        ref.source,
        ref.session_id,
    )
    c["session_batch_receipts"] = len(receipts)
    inv.counts = {k: int(v or 0) for k, v in c.items()}
    return inv


# --- the deletion ---------------------------------------------------------------


def _n(status: str) -> int:
    try:
        return int(status.rsplit(" ", 1)[1])
    except (IndexError, ValueError):
        return 0


async def _delete_rows(conn: Any, customer_id: str, ref: SessionRef, inv: Inventory) -> dict[str, int]:
    """Remove one session's rows. Inside the caller's transaction and lock.

    Queue rows first: nothing (idle sweep, worker CAS, reclaim) acts on the
    session once its row is gone. Chunks and failed_chunks have no FK to
    documents, so they go before the documents that name them.
    """
    d: dict[str, int] = {}

    async def run(table: str, sql: str, *args: Any) -> None:
        d[table] = d.get(table, 0) + _n(await conn.execute(sql, customer_id, *args))

    await run("ingestion_queue", "DELETE FROM ingestion_queue WHERE customer_id = $1 AND queue_id = ANY($2::bigint[])", inv.queue_ids)
    await run(
        "inferred_edges_queue",
        "DELETE FROM inferred_edges_queue WHERE customer_id = $1 AND anchor_doc_id = ANY($2::text[])",
        inv.doc_ids,
    )
    await run(
        "pending_edges",
        "DELETE FROM pending_edges WHERE customer_id = $1 AND (missing_canonical_id = ANY($2::text[]) "
        "OR from_canonical_id = ANY($2::text[]) OR to_canonical_id = ANY($2::text[]))",
        inv.canonical_ids,
    )
    await run("chunks", "DELETE FROM chunks WHERE customer_id = $1 AND doc_id = ANY($2::text[])", inv.doc_ids)
    await run("failed_chunks", "DELETE FROM failed_chunks WHERE customer_id = $1 AND doc_id = ANY($2::text[])", inv.doc_ids)
    await run(
        "acl_snapshots",
        "DELETE FROM acl_snapshots WHERE customer_id = $1 AND resource_type = 'document' "
        "AND resource_id = ANY($2::text[])",
        inv.doc_ids,
    )
    if inv.node_ids:
        # Edges first, with the surviving endpoints' materialized degree brought
        # down the way graph_writer does it; then the nodes (provenance cascades).
        removed = await conn.fetchval(
            """
            WITH deleted AS (
                DELETE FROM graph_edges
                 WHERE customer_id = $1
                   AND (from_node_id = ANY($2::bigint[]) OR to_node_id = ANY($2::bigint[]))
             RETURNING from_node_id, to_node_id
            ),
            endpoint_decs AS (
                SELECT node_id, count(*) AS dec FROM (
                    SELECT from_node_id AS node_id FROM deleted
                    UNION ALL SELECT to_node_id FROM deleted
                ) e
                 WHERE NOT (node_id = ANY($2::bigint[]))
                 GROUP BY node_id
            ),
            bumped AS (
                UPDATE graph_nodes gn SET degree = GREATEST(gn.degree - ed.dec, 0)
                  FROM endpoint_decs ed
                 WHERE gn.customer_id = $1 AND gn.node_id = ed.node_id
                RETURNING 1
            )
            SELECT count(*) FROM deleted
            """,
            customer_id,
            inv.node_ids,
        )
        d["graph_edges"] = int(removed or 0)
        await run(
            "graph_node_provenance",
            "DELETE FROM graph_node_provenance WHERE customer_id = $1 AND node_id = ANY($2::bigint[])",
            inv.node_ids,
        )
        await run(
            "node_post_write_queue",
            "DELETE FROM node_post_write_queue WHERE customer_id = $1 AND node_id = ANY($2::bigint[])",
            inv.node_ids,
        )
        await run("graph_nodes", "DELETE FROM graph_nodes WHERE customer_id = $1 AND node_id = ANY($2::bigint[])", inv.node_ids)
    await run("documents", "DELETE FROM documents WHERE customer_id = $1 AND doc_id = ANY($2::text[])", inv.doc_ids)
    await run("ingestion_events", "DELETE FROM ingestion_events WHERE customer_id = $1 AND event_id = ANY($2::bigint[])", inv.event_ids)
    # Receipts would cascade from their stream; deleted first so they are counted.
    await run(
        "session_batch_receipts",
        "DELETE FROM session_batch_receipts WHERE customer_id = $1 AND source_system = $2 AND session_id = $3",
        ref.source,
        ref.session_id,
    )
    await run(
        "session_streams",
        "DELETE FROM session_streams WHERE customer_id = $1 AND source_system = $2 AND session_id = $3",
        ref.source,
        ref.session_id,
    )
    return {k: v for k, v in d.items() if v}


async def record_sessions(
    customer_id: str,
    refs: list[SessionRef],
    *,
    deletion_id: str,
    reason: str,
    ticket: str | None,
    selector: dict[str, Any],
) -> None:
    """Phase 1: record every session as deleted. From the commit on, every
    writer refuses it. Also journals the raw keys found so far."""
    for ref in refs:
        async with with_tenant(customer_id) as conn:
            await lock_session(conn, customer_id, ref.source, ref.session_id)
            inv = await inventory(conn, customer_id, ref, with_counts=False)
            await conn.execute(
                """
                INSERT INTO session_deletions
                    (customer_id, source_system, session_id, deletion_id, reason, ticket,
                     selector, status, pending_keys)
                VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb, 'pending', $8::text[])
                ON CONFLICT (customer_id, source_system, session_id) DO UPDATE SET
                    deletion_id  = EXCLUDED.deletion_id,
                    ticket       = COALESCE(session_deletions.ticket, EXCLUDED.ticket),
                    status       = 'pending',
                    error        = NULL,
                    pending_keys = ARRAY(SELECT DISTINCT unnest(
                                     session_deletions.pending_keys || EXCLUDED.pending_keys))
                """,
                customer_id,
                ref.source,
                ref.session_id,
                uuid.UUID(deletion_id),
                reason,
                ticket,
                json.dumps(selector),
                sorted(inv.keys),
            )


async def _journal_and_delete_rows(customer_id: str, ref: SessionRef) -> tuple[dict[str, int], list[str], bool]:
    """Phase 2: under the lock, add any newly referenced keys to the journal and
    remove the rows -- one commit, so the keys are durable exactly when the rows
    that pointed at them are gone."""
    async with with_tenant(customer_id) as conn:
        await lock_session(conn, customer_id, ref.source, ref.session_id)
        inv = await inventory(conn, customer_id, ref, with_counts=False)
        keys = await conn.fetchval(
            """
            UPDATE session_deletions
               SET pending_keys = ARRAY(SELECT DISTINCT unnest(pending_keys || $4::text[])),
                   attempted_at = now()
             WHERE customer_id = $1 AND source_system = $2 AND session_id = $3
         RETURNING pending_keys
            """,
            customer_id,
            ref.source,
            ref.session_id,
            sorted(inv.keys),
        )
        if keys is None:
            raise RuntimeError("session is not recorded as deleted; record it before erasing")
        deleted = await _delete_rows(conn, customer_id, ref, inv)
    return deleted, list(keys), inv.in_flight


async def _deep_scan(
    store: Any, bucket: str, customer_id: str, source: str, session_ids: set[str]
) -> dict[str, set[str]]:
    """Protocol-1 objects of these sessions found by NAME in the date folders.

    `raw/<src>/<customer>/YYYY/MM/DD/<storage id>[:<seq>...].json`. One listing
    of the source folder, however many sessions are asked about.
    """
    wanted = {storage_id(s): s for s in session_ids}
    found: dict[str, set[str]] = {}
    date_key = re.compile(
        rf"\Araw/{re.escape(source)}/{re.escape(customer_id)}/\d{{4}}/\d{{2}}/\d{{2}}/(?P<name>[^/]+)\.json\Z"
    )
    for key in await store.list_keys(bucket, f"raw/{source}/{customer_id}/"):
        m = date_key.match(key)
        if not m:
            continue
        name = m.group("name")
        while name not in wanted and ":" in name and name.rsplit(":", 1)[1].isdigit():
            name = name.rsplit(":", 1)[0]
        if name in wanted:
            found.setdefault(wanted[name], set()).add(key)
    return found


async def _delete_objects(
    store: Any, bucket: str, customer_id: str, ref: SessionRef, keys: list[str]
) -> tuple[int, list[str], int]:
    """Phase 3: journaled keys, then the session's own folders. Returns
    (objects deleted, keys that failed, objects still under the folders)."""
    deleted, failed = await store.delete_keys(bucket, keys)
    errors = 0
    for prefix in ref.prefixes(customer_id):
        d, e = await store.delete_prefix(bucket, prefix)
        deleted += d
        errors += e
    remaining = 0
    for prefix in ref.prefixes(customer_id):
        remaining += await store.count_prefix(bucket, prefix)
    return deleted, failed, remaining + errors


async def _residue(customer_id: str, ref: SessionRef) -> dict[str, int]:
    async with with_tenant(customer_id) as conn:
        inv = await inventory(conn, customer_id, ref)
    return {k: v for k, v in inv.counts.items() if v}


async def erase_session(
    customer_id: str,
    ref: SessionRef,
    *,
    deep_keys: set[str] | None = None,
    grace_s: float | None = None,
) -> dict[str, Any]:
    """Phases 2-4 for one recorded session. Idempotent; never raises for a
    storage or database failure -- the outcome is written to the row."""
    grace = IN_FLIGHT_GRACE_S if grace_s is None else grace_s
    result: dict[str, Any] = {"rows_deleted": {}, "r2_objects_deleted": 0}
    status, error = "failed", None
    try:
        hold = await legal_hold(customer_id)
        if hold is not None:
            status, error = "held", f"legal hold: {hold}"
            return result
        store = get_store()
        bucket = await store.bucket_for(customer_id)
        in_flight_seen = False
        failed: list[str] = []
        r2_left = 0
        for round_ in (1, 2):
            deleted, keys, in_flight = await _journal_and_delete_rows(customer_id, ref)
            in_flight_seen = in_flight_seen or in_flight
            for table, n in deleted.items():
                result["rows_deleted"][table] = result["rows_deleted"].get(table, 0) + n
            n, failed, r2_left = await _delete_objects(
                store, bucket, customer_id, ref, sorted(set(keys) | (deep_keys or set()))
            )
            result["r2_objects_deleted"] += n
            if not failed:
                await _clear_journal(customer_id, ref)
            if not in_flight or round_ == 2:
                break
            # A worker was mid-pass: its row writes are refused, but it can
            # still save extraction-cache objects until its pass ends.
            log.info("session_deletion.in_flight_wait", customer=customer_id, **ref.as_dict(), wait_s=grace)
            await _wait_for_in_flight(grace)
        residue = await _residue(customer_id, ref)
        result.update(
            in_flight=in_flight_seen,
            residue=residue,
            r2_residue=r2_left,
            r2_failed_keys=failed[:50],
            r2_failed_key_count=len(failed),
            verified=not residue and not r2_left and not failed,
        )
        status = "done" if result["verified"] else "failed"
        if not result["verified"]:
            error = "residue remains; re-run the deletion"
        return result
    except asyncio.CancelledError:
        # Shutdown mid-run. The row keeps its key journal; a re-POST resumes.
        error = "cancelled (worker shutdown); re-POST the request to resume"
        raise
    except Exception as exc:  # the outcome is the row, never a lost exception
        log.exception("session_deletion.failed", customer=customer_id, **ref.as_dict())
        error = f"{type(exc).__name__}: {exc}"[:2000]
        return result
    finally:
        with contextlib.suppress(Exception):
            await _finish(customer_id, ref, status, result, error)


async def _wait_for_in_flight(seconds: float) -> None:
    """The pause before the second sweep. A seam: tests replace it with a writer."""
    await asyncio.sleep(seconds)


async def _clear_journal(customer_id: str, ref: SessionRef) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            "UPDATE session_deletions SET pending_keys = '{}' "
            "WHERE customer_id = $1 AND source_system = $2 AND session_id = $3",
            customer_id,
            ref.source,
            ref.session_id,
        )


async def _finish(
    customer_id: str, ref: SessionRef, status: str, result: dict[str, Any], error: str | None
) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE session_deletions
               SET status = $4, result = $5::jsonb, error = $6,
                   -- When the session was last found and removed. A re-run that
                   -- finds nothing keeps the original completion time.
                   deleted_at = CASE WHEN $4 <> 'done' THEN deleted_at
                                     WHEN deleted_at IS NULL OR $7 THEN now()
                                     ELSE deleted_at END
             WHERE customer_id = $1 AND source_system = $2 AND session_id = $3
            """,
            customer_id,
            ref.source,
            ref.session_id,
            status,
            json.dumps(result),
            error,
            bool(result.get("rows_deleted") or result.get("r2_objects_deleted")),
        )


async def remove_orphaned_persons(customer_id: str, person_ids: set[str]) -> int:
    """After an AUTHOR deletion: drop the person's Person node when nothing else
    holds it -- no edge left, no provenance from a non-session source, and no
    surviving document authored or uploaded by them."""
    if not person_ids:
        return 0
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            DELETE FROM graph_nodes n
             WHERE n.customer_id = $1 AND n.label = $2 AND n.canonical_id = ANY($3::text[])
               AND NOT EXISTS (SELECT 1 FROM graph_edges e WHERE e.customer_id = $1
                                 AND (e.from_node_id = n.node_id OR e.to_node_id = n.node_id))
               AND NOT EXISTS (SELECT 1 FROM graph_node_provenance p
                                WHERE p.customer_id = $1 AND p.node_id = n.node_id
                                  AND NOT (p.source_system = ANY($4::text[])))
               AND NOT EXISTS (SELECT 1 FROM documents d WHERE d.customer_id = $1
                                 AND (d.author_id = n.canonical_id
                                      OR d.metadata @> jsonb_build_object('uploader_id', n.canonical_id)))
            RETURNING node_id
            """,
            customer_id,
            NodeLabel.PERSON.value,
            sorted(person_ids),
            list(AGENT_SOURCES),
        )
        ids = [r["node_id"] for r in rows]
        if ids:
            await conn.execute(
                "DELETE FROM node_post_write_queue WHERE customer_id = $1 AND node_id = ANY($2::bigint[])",
                customer_id,
                ids,
            )
    return len(ids)


async def run_deletion(
    customer_id: str,
    refs: list[SessionRef],
    *,
    deep_scan: bool = False,
    person_ids: set[str] | None = None,
    grace_s: float | None = None,
) -> dict[str, Any]:
    """Erase every recorded session in `refs`. Returns per-session outcomes."""
    deep: dict[str, set[str]] = {}
    if deep_scan and refs:
        store = get_store()
        bucket = await store.bucket_for(customer_id)
        for source in sorted({r.source for r in refs}):
            wanted = {r.session_id for r in refs if r.source == source}
            for sid, keys in (await _deep_scan(store, bucket, customer_id, source, wanted)).items():
                deep[f"{source}\0{sid}"] = keys
    outcomes: dict[str, Any] = {}
    for ref in refs:
        outcomes[f"{ref.source}:{ref.session_id}"] = await erase_session(
            customer_id, ref, deep_keys=deep.get(f"{ref.source}\0{ref.session_id}"), grace_s=grace_s
        )
    persons = 0
    if person_ids:
        with contextlib.suppress(Exception):
            persons = await remove_orphaned_persons(customer_id, person_ids)
    return {"sessions": outcomes, "person_nodes_deleted": persons}


async def plan(customer_id: str, refs: list[SessionRef], *, deep_scan: bool = False) -> dict[str, Any]:
    """The dry run: what a deletion of `refs` would remove. Writes nothing."""
    store = get_store()
    bucket = await store.bucket_for(customer_id)
    sessions: list[dict[str, Any]] = []
    already: set[SessionRef] = set()
    async with with_tenant(customer_id) as conn:
        for row in await conn.fetch(
            "SELECT source_system, session_id FROM session_deletions WHERE customer_id = $1 "
            "AND session_id = ANY($2::text[])",
            customer_id,
            sorted({r.session_id for r in refs}),
        ):
            already.add(SessionRef(row["source_system"], row["session_id"]))
        inventories = [(ref, await inventory(conn, customer_id, ref)) for ref in refs]
    deep: dict[tuple[str, str], set[str]] = {}
    if deep_scan:
        for source in sorted({r.source for r in refs}):
            wanted = {r.session_id for r in refs if r.source == source}
            for sid, keys in (await _deep_scan(store, bucket, customer_id, source, wanted)).items():
                deep[(source, sid)] = keys
    sem = asyncio.Semaphore(_R2_CONCURRENCY)

    async def listed(ref: SessionRef) -> set[str]:
        keys: set[str] = set()
        for prefix in ref.prefixes(customer_id):
            async with sem:
                keys.update(await store.list_keys(bucket, prefix))
        return keys

    folder_keys = await asyncio.gather(*(listed(ref) for ref, _ in inventories))
    totals: dict[str, int] = {}
    r2_total = referenced_total = 0
    for (ref, inv), in_folders in zip(inventories, folder_keys, strict=True):
        referenced = inv.keys - in_folders
        extra = deep.get((ref.source, ref.session_id), set()) - in_folders - referenced
        rows = {k: v for k, v in inv.counts.items() if v}
        for table, n in rows.items():
            totals[table] = totals.get(table, 0) + n
        r2_total += len(in_folders) + len(extra)
        referenced_total += len(referenced)
        sessions.append(
            {
                **ref.as_dict(),
                "already_deleted": ref in already,
                "in_flight": inv.in_flight,
                "rows": rows,
                # Listed, so these exist right now.
                "r2_objects": len(in_folders) + len(extra),
                # Protocol-1 keys named by rows; existence not checked (their
                # delete reports per key).
                "r2_referenced_keys": len(referenced),
            }
        )
    return {
        "sessions": sessions,
        "totals": {"sessions": len(sessions), "rows": totals, "r2_objects": r2_total,
                   "r2_referenced_keys": referenced_total},
    }


# --- HTTP ---------------------------------------------------------------------------

_INFLIGHT: set[asyncio.Task[Any]] = set()


def _require_customer(
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> str:
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
    return x_prbe_customer


class AuthorSelector(BaseModel):
    employee_id: str | None = Field(default=None, min_length=1, max_length=200)
    email: str | None = Field(default=None, min_length=3, max_length=320)

    @model_validator(mode="after")
    def one_identity(self) -> AuthorSelector:
        if not self.employee_id and not self.email:
            raise ValueError("author needs employee_id or email")
        return self


class SessionDeletionRequest(BaseModel):
    session_ids: list[str] | None = Field(default=None, min_length=1, max_length=1000)
    author: AuthorSelector | None = None
    sources: list[str] | None = Field(default=None, min_length=1)
    dry_run: bool = True
    reason: str | None = Field(default=None, max_length=2000)
    ticket: str | None = Field(default=None, max_length=500)
    deep_scan: bool = False

    @field_validator("session_ids")
    @classmethod
    def ids_are_session_shaped(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        bad = [s for s in v if not valid_session_id(s)]
        if bad:
            raise ValueError(f"not session ids: {bad[:5]}")
        return list(dict.fromkeys(v))

    @field_validator("sources")
    @classmethod
    def known_sources(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        unknown = sorted(set(v) - set(AGENT_SOURCES))
        if unknown:
            raise ValueError(f"not coding-agent sources: {unknown}")
        return sorted(set(v))

    @model_validator(mode="after")
    def one_selector(self) -> SessionDeletionRequest:
        if (self.session_ids is None) == (self.author is None):
            raise ValueError("give exactly one of session_ids or author")
        if not self.dry_run and not (self.reason and self.reason.strip()):
            raise ValueError("reason is required to apply a deletion")
        return self


async def _select(customer_id: str, body: SessionDeletionRequest) -> Selection:
    sources = body.sources or list(AGENT_SOURCES)
    if body.session_ids is not None:
        return await select_by_ids(customer_id, body.session_ids, sources)
    assert body.author is not None
    return await select_by_author(
        customer_id, sources, employee_id=body.author.employee_id, email=body.author.email
    )


async def _background(
    customer_id: str,
    deletion_id: str,
    refs: list[SessionRef],
    deep_scan: bool,
    person_ids: set[str],
) -> None:
    try:
        outcome = await run_deletion(
            customer_id, refs, deep_scan=deep_scan, person_ids=person_ids
        )
        log.info(
            "session_deletion.completed",
            customer=customer_id,
            deletion_id=deletion_id,
            sessions=len(refs),
            verified=sum(1 for o in outcome["sessions"].values() if o.get("verified")),
            person_nodes_deleted=outcome["person_nodes_deleted"],
        )
    except asyncio.CancelledError:
        # Shutdown mid-run: rows stay `pending` with their journal; a re-POST resumes.
        log.warning("session_deletion.cancelled", customer=customer_id, deletion_id=deletion_id)
        raise
    except Exception:
        log.exception("session_deletion.run_failed", customer=customer_id, deletion_id=deletion_id)


@router.post("")
async def delete_sessions(
    body: SessionDeletionRequest,
    customer_id: str = Depends(_require_customer),
) -> Any:
    try:
        hold = await legal_hold(customer_id)
        if hold is not None and not body.dry_run:
            raise _held(hold)
        selection = await _select(customer_id, body)
        if body.dry_run:
            report = await plan(customer_id, selection.refs, deep_scan=body.deep_scan)
            return {
                "dry_run": True,
                "customer_id": customer_id,
                "legal_hold": hold,
                **report,
                "unattributed_sessions": selection.unattributed,
                "skipped_invalid_ids": selection.skipped_invalid[:50],
                "not_covered": list(NOT_COVERED),
            }
    except SessionDeletionError as exc:
        raise HTTPException(exc.status, exc.detail) from exc
    if not selection.refs:
        return {"dry_run": False, "status": "nothing_to_delete", "sessions": [],
                "unattributed_sessions": selection.unattributed}
    deletion_id = str(uuid.uuid4())
    selector = {"by": "id" if body.session_ids is not None else "author"}
    assert body.reason is not None
    await record_sessions(
        customer_id,
        selection.refs,
        deletion_id=deletion_id,
        reason=body.reason.strip(),
        ticket=body.ticket,
        selector=selector,
    )
    log.info(
        "session_deletion.started",
        customer=customer_id,
        deletion_id=deletion_id,
        sessions=len(selection.refs),
        selector=selector["by"],
        ticket=body.ticket,
    )
    person_ids = selection.person_ids if body.author is not None else set()
    task = asyncio.create_task(
        _background(customer_id, deletion_id, selection.refs, body.deep_scan, person_ids)
    )
    _INFLIGHT.add(task)
    task.add_done_callback(_INFLIGHT.discard)
    return JSONResponse(
        status_code=202,
        content={
            "dry_run": False,
            "deletion_id": deletion_id,
            "status": "running",
            "sessions": [r.as_dict() for r in selection.refs],
            "unattributed_sessions": selection.unattributed,
        },
    )


@router.get("/{deletion_id}")
async def deletion_status(
    deletion_id: str,
    customer_id: str = Depends(_require_customer),
) -> dict[str, Any]:
    try:
        wanted = uuid.UUID(deletion_id)
    except ValueError as exc:
        raise HTTPException(422, "malformed deletion_id") from exc
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT source_system, session_id, status, requested_at, attempted_at, deleted_at,
                   result, error
              FROM session_deletions
             WHERE customer_id = $1 AND deletion_id = $2
             ORDER BY source_system, session_id
            """,
            customer_id,
            wanted,
        )
    if not rows:
        raise HTTPException(404, "unknown deletion_id")
    statuses = {r["status"] for r in rows}
    overall = (
        "running" if "pending" in statuses
        else "held" if "held" in statuses
        else "failed" if "failed" in statuses
        else "done"
    )
    return {
        "deletion_id": deletion_id,
        "customer_id": customer_id,
        "status": overall,
        "sessions": [
            {
                "source": r["source_system"],
                "session_id": r["session_id"],
                "status": r["status"],
                "requested_at": _iso(r["requested_at"]),
                "attempted_at": _iso(r["attempted_at"]),
                "deleted_at": _iso(r["deleted_at"]),
                "result": _decode(r["result"]),
                "error": r["error"],
            }
            for r in rows
        ],
    }


def _iso(value: Any) -> str | None:
    return value.isoformat() if value is not None else None


def _decode(raw: Any) -> Any:
    # No jsonb codec is registered on the pool (see kb/purge_routes._decode_result).
    if isinstance(raw, str):
        with contextlib.suppress(ValueError):
            return json.loads(raw)
    return raw
