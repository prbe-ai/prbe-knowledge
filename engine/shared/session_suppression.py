"""A deleted coding-agent session stays deleted: one lock, one check, shared by every writer.

Deleting a session (kb/session_deletion.py) records it in `session_deletions`.
From then on nothing may write that session again, and there are four writers:

    ingest door, protocol 2      kb/session_receipts.accept
    ingest door, protocol 1      kb/session_receipts.accept_legacy
    idle sweep                   kb/session_completer._end_one
    worker's write transaction   engine/ingest/normalizer (_persist, persist_batch)

Each takes the session's advisory lock and asks `deleted_sessions` under it.
The deletion takes the same lock before it records the session and before it
removes rows, so the two can only happen one after the other: a writer that
got in first has its rows removed by the deletion; a writer that comes second
sees the record and refuses. Without the lock, a worker that read "not
deleted" a moment before the deletion committed would write the transcript
back afterwards.

The worker's refusal matters more than it looks: a session's queue row is
re-read from raw storage on every pass (kb/handlers/claude_code.py
fetch_supplementary), so a deletion that removed only the derived rows would
be undone by the next pass.

Caller contract: the connection is inside a transaction with the tenant GUC
bound (`with_tenant`). `session_deletions` is FORCE RLS, so without the GUC the
check silently answers "not deleted".
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import asyncpg

from engine.shared.constants import AGENT_SESSION_SOURCES
from engine.shared.db import with_tenant
from engine.shared.exceptions import DuplicateEventIgnored

#: The HTTP status and machine-readable reason an ingest door answers with for
#: a deleted session. 410 Gone: the resource existed and will not come back, so
#: a client must stop re-sending it. NOTE: research-os's gateway today forwards
#: only 409 (and 422 for protocol 2) and turns every other engine 4xx into a
#: 502, which the tap retries forever; the research-os half of this deletion
#: must pass 410 through and have the tap drop the batch.
DELETED_STATUS = 410
DELETED_REASON = "session_deleted"


class SessionDeleted(DuplicateEventIgnored):
    """The session this work belongs to has been deleted; write nothing.

    A DuplicateEventIgnored so the single-row worker path marks the row
    skipped instead of failing it. `transient` so the coalesced batch path,
    which fails every sibling on any error, returns the healthy siblings to
    pending rather than dead-lettering them. (They are re-claimed with an
    attempt spent; the worker's pre-check then drops the deleted row, so this
    happens at most once per race, never in a loop.)

    Carries what the worker needs to clean up after its own pass: see
    `sweep_session_folders`.
    """

    transient = True

    def __init__(self, message: str, *, customer_id: str, source: str, session_ids: set[str]) -> None:
        super().__init__(message)
        self.customer_id = customer_id
        self.source = source
        self.session_ids = set(session_ids)


def session_lock_key(customer_id: str, source: str, session_id: str) -> str:
    """The advisory-lock name every writer of one session serializes on.

    Also taken by kb/session_receipts (ingest) and kb/session_completer (idle
    sweep) through `lock_session`; changing it splits them apart silently.
    """
    return f"session-stream:{customer_id}:{source}:{session_id}"


async def lock_session(conn: Any, customer_id: str, source: str, session_id: str) -> None:
    await conn.execute(
        "SELECT pg_advisory_xact_lock(hashtextextended($1, 0))",
        session_lock_key(customer_id, source, session_id),
    )


async def deleted_sessions(
    conn: Any, customer_id: str, source: str, session_ids: Iterable[str]
) -> set[str]:
    """Which of `session_ids` are recorded as deleted. Caller holds their locks.

    No table means nothing was ever recorded: code can reach a plane before
    migration 0140 does (the research plane pulls `latest` and migrates on
    research-os's deploy), and every session writer asks this -- an error here
    would stop all session capture. Asked inside a savepoint so the caller's
    transaction survives the miss.
    """
    ids = sorted(set(session_ids))
    if not ids:
        return set()
    try:
        async with conn.transaction():
            rows = await conn.fetch(
                "SELECT session_id FROM session_deletions "
                "WHERE customer_id = $1 AND source_system = $2 AND session_id = ANY($3::text[])",
                customer_id,
                source,
                ids,
            )
    except asyncpg.UndefinedTableError:
        return set()
    return {r["session_id"] for r in rows}


#: The only suffixes a session's OWN queue / ingestion_events rows carry:
#: pre-0026 `<session>:<batch_seq>` and the old synthetic `<session>:finalize`.
#: Nothing else may be stripped: protocol 1 accepts any session id, `:`
#: included, so `X:branch` is a session of its own, not a child of `X`.
LEGACY_EVENT_SUFFIX_SQL = "^([0-9]+|finalize)$"
_LEGACY_EVENT_SUFFIX = re.compile(r":(?:[0-9]+|finalize)\Z")


def session_of_event_id(event_id: str) -> str:
    """The session a queue / ingestion_events `source_event_id` belongs to."""
    return _LEGACY_EVENT_SUFFIX.sub("", event_id, count=1)


def session_ids_of(documents: Iterable[Any]) -> set[str]:
    """The session ids a coding-agent NormalizationResult writes.

    The session document's `source_id` is the session id. A unit document
    names its session through `parent_doc_id` (`<source>:<customer>:<session>`),
    NOT by splitting its own `<session>:<kind>:<n>` source_id at the first `:`
    -- that would read session `X:branch`'s units as session `X`'s.
    """
    ids: set[str] = set()
    for doc in documents:
        source_id = getattr(doc, "source_id", None)
        if not isinstance(source_id, str) or not source_id:
            continue
        parent = getattr(doc, "parent_doc_id", None)
        if parent is None:
            ids.add(source_id)
            continue
        system = getattr(doc, "source_system", None)
        prefix = f"{getattr(system, 'value', system)}:{getattr(doc, 'customer_id', '')}:"
        if isinstance(parent, str) and parent.startswith(prefix):
            ids.add(parent[len(prefix):])
        else:
            ids.add(source_id.split(":", 1)[0])
    return ids


async def refuse_deleted_sessions(
    conn: Any, customer_id: str, source: str, session_ids: Iterable[str]
) -> None:
    """Lock each session (sorted: two writers never lock in opposite orders),
    then raise SessionDeleted if any of them is recorded as deleted."""
    ids = sorted(set(session_ids))
    for session_id in ids:
        await lock_session(conn, customer_id, source, session_id)
    deleted = await deleted_sessions(conn, customer_id, source, ids)
    if deleted:
        raise SessionDeleted(
            f"{source} session deleted; nothing written ({len(deleted)} session(s))",
            customer_id=customer_id,
            source=source,
            session_ids=deleted,
        )


async def refuse_deleted_graph_refs(
    conn: Any, customer_id: str, refs: Iterable[tuple[str, str]]
) -> None:
    """The fence for a writer keyed by graph (label, canonical_id) pairs: the
    inferred-edges worker, which reads documents, spends an LLM call on them,
    and then upserts the nodes and edges it inferred -- `upsert_nodes` would
    re-create a session node a deletion removed meanwhile, and the edges would
    carry `why` text drawn from the transcript. Its anchor AND every edge
    endpoint are checked: a bundle holds neighbouring documents, so an edge
    can name a session other than the anchor's. Same lock and check as every
    other writer; a ref of no coding-agent session passes untouched.

    A Document `<source>:<customer>:<session>` is a session document and
    `...:<kind>:<n>` one of its units (both readings are checked, since a
    protocol-1 id may hold `:`); an AgentSession is
    `agent_session:<source>:<session>` (constants.agent_session_canonical_id).
    """
    sources = sorted(s.value for s in AGENT_SESSION_SOURCES)
    wanted: dict[str, set[str]] = {}
    for label, canonical_id in refs:
        for source in sources:
            if label == "Document" and canonical_id.startswith(f"{source}:{customer_id}:"):
                rest = canonical_id[len(f"{source}:{customer_id}:"):]
                wanted.setdefault(source, set()).update({rest, rest.rsplit(":", 2)[0]})
            elif label == "AgentSession" and canonical_id.startswith(f"agent_session:{source}:"):
                wanted.setdefault(source, set()).add(canonical_id[len(f"agent_session:{source}:"):])
    for source in sorted(wanted):
        await refuse_deleted_sessions(conn, customer_id, source, wanted[source])


async def is_session_deleted(customer_id: str, source: str, queue_event_id: str) -> bool:
    """The worker's check BEFORE it reads and mines a session (no lock: a pass
    that starts a moment before the deletion is caught by the write fence).

    Saves the extraction a deleted session would otherwise pay for, and keeps a
    re-pended row of a deleted session (the idle sweep's partial-pass retry
    does not look at deletions) from being mined again.
    """
    session_id = session_of_event_id(queue_event_id)
    async with with_tenant(customer_id) as conn:
        return bool(await deleted_sessions(conn, customer_id, source, [session_id]))


def session_folder(source: str, customer_id: str, session_id: str) -> str:
    """`raw/<src>/<customer>/<session>/`: the idle sweep's `finalize.marker`
    (session_signals.cron_marker_key) and the extraction cache
    (extraction_cache.SegmentCache). Trailing `/` is load-bearing."""
    return f"raw/{source}/{customer_id}/{session_id}/"


def is_own_folder_key(key: str, folder: str) -> bool:
    """Is `key`, listed under `folder`, this session's own object?

    Both writers build the folder from the RAW session id, and protocol 1
    accepts `/` in one: session `X/y` keeps its marker and cache under
    `raw/<src>/<customer>/X/y/`, which is inside session `X`'s folder. Only the
    two shapes this session writes itself are its own.
    """
    if not key.startswith(folder):
        return False
    rest = key[len(folder):]
    if rest == "finalize.marker":
        return True
    cache = rest.removeprefix("extraction-cache/")
    return cache != rest and cache.endswith(".json") and "/" not in cache


async def own_folder_keys(store: Any, bucket: str, folder: str) -> list[str]:
    return [k for k in await store.list_keys(bucket, folder) if is_own_folder_key(k, folder)]


def as_legal_hold(value: str | None) -> str | None:
    """`customers.metadata->>'legal_hold'` as a hold, or None. Anything but
    absent, null, false or "" is a hold: deleting under an ambiguous hold is
    the failure that cannot be undone."""
    if value is None or value.strip().lower() in ("", "false", "null"):
        return None
    return value


async def tenant_legal_hold(customer_id: str) -> str | None:
    async with with_tenant(customer_id) as conn:
        return as_legal_hold(await conn.fetchval(
            "SELECT metadata->>'legal_hold' FROM customers WHERE customer_id = $1", customer_id
        ))


async def reopen_deletion(exc: SessionDeleted, why: str, *, status: str = "failed") -> None:
    """A late sweep did not finish (or must not run, under a hold): mark the
    recorded deletion so its status says so and /resume sweeps again. Nothing
    else would -- the queue row is gone and the deletion may read `done`."""
    async with with_tenant(exc.customer_id) as conn:
        await conn.execute(
            "UPDATE session_deletions SET status = $5, error = $4 "
            "WHERE customer_id = $1 AND source_system = $2 "
            "AND session_id = ANY($3::text[]) AND status <> 'held'",
            exc.customer_id,
            exc.source,
            sorted(exc.session_ids),
            why[:2000],
            status,
        )


async def sweep_session_folders(store: Any, exc: SessionDeleted) -> tuple[int, int]:
    """Delete each refused session's own objects under
    `raw/<src>/<customer>/<session>/`. Returns (deleted, failed).

    A worker already mining a session when it was deleted keeps saving
    extraction-cache answers there until its write is refused, which can be
    after the deletion's own last sweep. The worker is the last writer, so it
    cleans up after itself. Only ids read back from `session_deletions` reach
    here, and those passed the deletion's id-shape check.
    """
    bucket = await store.bucket_for(exc.customer_id)
    deleted = failed = 0
    for session_id in sorted(exc.session_ids):
        folder = session_folder(exc.source, exc.customer_id, session_id)
        n, bad = await store.delete_keys(bucket, await own_folder_keys(store, bucket, folder))
        deleted += n
        failed += len(bad)
    return deleted, failed
