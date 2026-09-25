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

from collections.abc import Iterable
from typing import Any

from engine.shared.exceptions import DuplicateEventIgnored

#: The HTTP status and machine-readable reason an ingest door answers with for
#: a deleted session. 410 Gone: the resource existed and will not come back, so
#: a client must stop re-sending it (research-os forwards engine 4xx; see the
#: route contract in kb/session_deletion.py).
DELETED_STATUS = 410
DELETED_REASON = "session_deleted"


class SessionDeleted(DuplicateEventIgnored):
    """The session this work belongs to has been deleted; write nothing.

    A DuplicateEventIgnored so the single-row worker path marks the row
    skipped instead of failing it. `transient` so the coalesced batch path,
    which fails every sibling on any error, sends the healthy siblings back to
    pending instead of burning an attempt on them.
    """

    transient = True


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
    """Which of `session_ids` are recorded as deleted. Caller holds their locks."""
    ids = sorted(set(session_ids))
    if not ids:
        return set()
    rows = await conn.fetch(
        "SELECT session_id FROM session_deletions "
        "WHERE customer_id = $1 AND source_system = $2 AND session_id = ANY($3::text[])",
        customer_id,
        source,
        ids,
    )
    return {r["session_id"] for r in rows}


def session_ids_of(documents: Iterable[Any]) -> set[str]:
    """The session ids a coding-agent NormalizationResult writes.

    The session document's `source_id` is the session id; a unit document's is
    `<session_id>:<kind>:<suffix>` and carries a `parent_doc_id`. Normalize
    always emits the session document too, so an id holding `:` is still
    covered through it.
    """
    ids: set[str] = set()
    for doc in documents:
        source_id = getattr(doc, "source_id", None)
        if not isinstance(source_id, str) or not source_id:
            continue
        if getattr(doc, "parent_doc_id", None) is None:
            ids.add(source_id)
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
            f"{source} session deleted; nothing written ({len(deleted)} session(s))"
        )
