"""CRUD over the wiki_review_queue table.

Per-artifact review-state rows for post-approval wiki artifacts
(postmortem, knowledge_page, correction). Each artifact has exactly
one row keyed by ``(customer_id, artifact_doc_id)``; re-runs after a
reject create a NEW row whose ``parent_artifact_doc_id`` points at
the prior version, so version lineage is a parent-link chain rather
than a jsonb history blob (cf. ``services/ingestion/investigation_state.py``
which uses the jsonb-list pattern).

State machine:
- ``pending_writeback`` — orchestrator is writing the document body.
- ``pending_review``    — body persisted; reviewer attention required.
- ``failed_pending_review`` — orchestrator hit a non-retryable error
  but produced a partial artifact that still needs reviewer attention.
- ``approved`` / ``rejected`` — terminal reviewer outcomes.

RLS: every query goes through ``with_tenant(customer_id)``, which sets
``app.current_customer_id`` and lets the ``tenant_isolation`` policy
scope the row set (and reject cross-tenant writes via WITH CHECK).
"""
from __future__ import annotations

import json
from typing import Any

from shared.db import with_tenant
from shared.schemas.wiki_artifact import (
    ArtifactKind,
    ArtifactState,
    WikiArtifactDetail,
    WikiArtifactListItem,
    WikiArtifactVersionEntry,
)

# Cap the upward parent walk so a malformed cycle in parent_artifact_doc_id
# (would require a bypass of the parent-link insertion path) can't pin the
# event loop forever.
_MAX_LINEAGE_WALK = 100


def _coerce_metadata(raw: Any) -> dict[str, Any]:
    """Asyncpg returns jsonb as either str (no decoder set) or dict (decoder set)."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except (TypeError, ValueError):
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def _row_to_detail(
    row: Any, versions: list[WikiArtifactVersionEntry],
) -> WikiArtifactDetail:
    return WikiArtifactDetail(
        customer_id=row["customer_id"],
        artifact_doc_id=row["artifact_doc_id"],
        incident_doc_id=row["incident_doc_id"],
        artifact_kind=row["artifact_kind"],
        target_doc_id=row["target_doc_id"],
        parent_artifact_doc_id=row["parent_artifact_doc_id"],
        state=row["state"],
        versions=versions,
        reviewer_id=row["reviewer_id"],
        reviewed_at=row["reviewed_at"],
        metadata=_coerce_metadata(row["metadata"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_list_item(row: Any) -> WikiArtifactListItem:
    return WikiArtifactListItem(
        artifact_doc_id=row["artifact_doc_id"],
        incident_doc_id=row["incident_doc_id"],
        artifact_kind=row["artifact_kind"],
        target_doc_id=row["target_doc_id"],
        state=row["state"],
        parent_artifact_doc_id=row["parent_artifact_doc_id"],
        updated_at=row["updated_at"],
    )


def _version_entry_from_row(row: Any) -> WikiArtifactVersionEntry:
    """Convert a wiki_review_queue row into a version-history entry.

    The ``decision`` field is a coarse-grained UI label:
      - ``"approved"`` / ``"rejected"`` map directly from those terminal
        states.
      - ``"pending"`` is used for any non-terminal state
        (``pending_writeback``, ``pending_review``,
        ``failed_pending_review``). Callers that need to distinguish
        these should read the underlying ``state`` via ``get_detail``'s
        ``state`` field; this version entry intentionally hides the
        in-flight orchestrator detail.

    ``pending_writeback`` rows are transient (the writeback completes
    in <1s after the agent run), so they normally only appear in a
    version history if a writeback is currently in flight or failed.
    UI should consider surfacing those distinctly if
    ``failed_pending_review`` becomes common.
    """
    metadata = _coerce_metadata(row["metadata"])
    state = row["state"]
    decision: str
    if state == "approved":
        decision = "approved"
    elif state == "rejected":
        decision = "rejected"
    else:
        decision = "pending"
    return WikiArtifactVersionEntry(
        artifact_doc_id=row["artifact_doc_id"],
        decision=decision,  # type: ignore[arg-type]
        reviewed_by=row["reviewer_id"],
        reviewed_at=row["reviewed_at"],
        feedback=metadata.get("last_feedback"),
        created_at=row["created_at"],
    )


def _validate_kind_target_consistency(
    artifact_kind: ArtifactKind, target_doc_id: str | None,
) -> None:
    """Mirror the DB check constraint in Python so callers get a typed error
    rather than an asyncpg ``CheckViolationError`` after a roundtrip.
    """
    if artifact_kind == "correction" and target_doc_id is None:
        raise ValueError("correction artifact requires target_doc_id")
    if artifact_kind != "correction" and target_doc_id is not None:
        raise ValueError(
            f"{artifact_kind} artifact must have null target_doc_id"
        )


async def upsert_pending_review(
    *,
    customer_id: str,
    artifact_doc_id: str,
    incident_doc_id: str,
    artifact_kind: ArtifactKind,
    target_doc_id: str | None,
    parent_artifact_doc_id: str | None,
    initial_state: ArtifactState,
    metadata: dict[str, Any],
) -> WikiArtifactDetail:
    """Insert/upsert a wiki_review_queue row at ``initial_state``.

    Idempotent on ``(customer_id, artifact_doc_id)`` — re-issuing the
    same artifact_doc_id (e.g. retry after a partial failure) updates
    ``state``, ``metadata``, and ``updated_at``. The kind/target/parent
    columns are NOT updated on conflict because they're identity-shaped:
    re-running the writeback for the same artifact_doc_id with a
    different ``artifact_kind`` would be a caller bug, not something
    to silently overwrite.

    The returned detail carries ``versions=[]`` — only the row itself,
    not the full lineage. Callers that need the full lineage call
    ``get_detail`` (which runs a recursive CTE) after the upsert.
    """
    _validate_kind_target_consistency(artifact_kind, target_doc_id)
    payload = json.dumps(metadata)
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO wiki_review_queue
                (customer_id, artifact_doc_id, incident_doc_id,
                 artifact_kind, target_doc_id, parent_artifact_doc_id,
                 state, metadata)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::jsonb)
            ON CONFLICT (customer_id, artifact_doc_id) DO UPDATE
              SET state = EXCLUDED.state,
                  metadata = EXCLUDED.metadata,
                  updated_at = now()
            RETURNING *;
            """,
            customer_id,
            artifact_doc_id,
            incident_doc_id,
            artifact_kind,
            target_doc_id,
            parent_artifact_doc_id,
            initial_state,
            payload,
        )
    assert row is not None  # INSERT … RETURNING always returns a row
    return _row_to_detail(row, versions=[])


async def mark_approved(
    *,
    customer_id: str,
    artifact_doc_id: str,
    reviewer_id: str,
) -> WikiArtifactDetail:
    """Transition a row to 'approved'.

    Idempotent: if the row is already 'approved', returns the existing
    detail (with full version lineage). Raises ``LookupError`` if the
    row does not exist, and ``ValueError`` if the row is in a terminal
    non-approved state (i.e. 'rejected').
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            UPDATE wiki_review_queue
            SET state = 'approved',
                reviewer_id = $3,
                reviewed_at = now(),
                updated_at = now()
            WHERE customer_id = $1
              AND artifact_doc_id = $2
              AND state IN ('pending_review', 'failed_pending_review')
            RETURNING *;
            """,
            customer_id, artifact_doc_id, reviewer_id,
        )
        if row is not None:
            # Fresh approval — return full lineage detail.
            detail = await _get_detail_inner(conn, customer_id, artifact_doc_id)
            assert detail is not None
            return detail

        # No row returned: either missing, or in a state outside the
        # filter (already approved/rejected/pending_writeback).
        existing = await conn.fetchrow(
            "SELECT state FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            customer_id, artifact_doc_id,
        )
        if existing is None:
            raise LookupError(
                f"no wiki_review_queue row for {artifact_doc_id}"
            )
        if existing["state"] == "approved":
            # Idempotent re-approval — reuse the live connection so the
            # whole operation stays inside one transaction (matches the
            # fresh-approval branch above).
            detail = await _get_detail_inner(
                conn, customer_id, artifact_doc_id,
            )
            assert detail is not None
            return detail
        raise ValueError(
            f"cannot approve from terminal state {existing['state']}"
        )


async def mark_rejected(
    *,
    customer_id: str,
    artifact_doc_id: str,
    reviewer_id: str,
    feedback: str,
) -> WikiArtifactDetail:
    """Transition a row to 'rejected', stashing feedback in metadata.

    Not idempotent: re-rejecting an already-rejected row raises
    ``ValueError``. The writeback route is responsible for the
    idempotency probe (it reads the current state via ``get_detail``
    before calling here, so two parallel reject submissions race only
    inside this UPDATE, and the loser sees the ValueError).

    Raises ``LookupError`` if no row exists.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            UPDATE wiki_review_queue
            SET state = 'rejected',
                reviewer_id = $3,
                reviewed_at = now(),
                metadata = metadata
                    || jsonb_build_object('last_feedback', $4::text),
                updated_at = now()
            WHERE customer_id = $1
              AND artifact_doc_id = $2
              AND state IN ('pending_review', 'failed_pending_review')
            RETURNING *;
            """,
            customer_id, artifact_doc_id, reviewer_id, feedback,
        )
        if row is not None:
            detail = await _get_detail_inner(conn, customer_id, artifact_doc_id)
            assert detail is not None
            return detail
        existing = await conn.fetchrow(
            "SELECT state FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            customer_id, artifact_doc_id,
        )
    if existing is None:
        raise LookupError(
            f"no wiki_review_queue row for {artifact_doc_id}"
        )
    raise ValueError(
        f"cannot reject from terminal state {existing['state']}"
    )


async def list_for_customer(
    customer_id: str,
    *,
    state: ArtifactState | None = None,
    artifact_kind: ArtifactKind | None = None,
    incident_doc_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[WikiArtifactListItem]:
    """Paginated list with optional state/kind/incident filters.

    Ordered by ``updated_at DESC`` to surface freshly-updated artifacts
    first (matches the review-queue UI's expected ordering). ``limit``
    clamped to ``[1, 200]``; ``offset`` clamped to ``[0, ...)``.
    """
    limit = max(1, min(limit, 200))
    offset = max(0, offset)

    sql = (
        "SELECT artifact_doc_id, incident_doc_id, artifact_kind, "
        "target_doc_id, state, parent_artifact_doc_id, updated_at "
        "FROM wiki_review_queue WHERE customer_id = $1"
    )
    args: list[Any] = [customer_id]
    if state is not None:
        args.append(state)
        sql += f" AND state = ${len(args)}"
    if artifact_kind is not None:
        args.append(artifact_kind)
        sql += f" AND artifact_kind = ${len(args)}"
    if incident_doc_id is not None:
        args.append(incident_doc_id)
        sql += f" AND incident_doc_id = ${len(args)}"
    args.append(limit)
    args.append(offset)
    sql += (
        f" ORDER BY updated_at DESC "
        f"LIMIT ${len(args) - 1} OFFSET ${len(args)}"
    )
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, *args)
    return [_row_to_list_item(r) for r in rows]


async def _get_detail_inner(
    conn: Any, customer_id: str, artifact_doc_id: str,
) -> WikiArtifactDetail | None:
    """Shared body for ``get_detail`` and the post-mutation lineage fetch
    inside ``mark_approved`` / ``mark_rejected``, so callers don't have
    to re-acquire a tenant connection.
    """
    row = await conn.fetchrow(
        "SELECT * FROM wiki_review_queue "
        "WHERE customer_id = $1 AND artifact_doc_id = $2",
        customer_id, artifact_doc_id,
    )
    if row is None:
        return None

    # Walk up the parent chain to find the lineage root, bounded by
    # _MAX_LINEAGE_WALK so a hypothetical cycle (would require a manual
    # row hack — the insert path always uses a fresh artifact_doc_id)
    # can't spin forever.
    root_id: str = row["artifact_doc_id"]
    parent: str | None = row["parent_artifact_doc_id"]
    seen: set[str] = {root_id}
    for _ in range(_MAX_LINEAGE_WALK):
        if parent is None:
            break
        if parent in seen:
            # Cycle. Treat the current node as the root and stop.
            break
        parent_row = await conn.fetchrow(
            "SELECT artifact_doc_id, parent_artifact_doc_id "
            "FROM wiki_review_queue "
            "WHERE customer_id = $1 AND artifact_doc_id = $2",
            customer_id, parent,
        )
        if parent_row is None:
            # Parent row missing (cross-version retention deleted it
            # or the chain was broken by hand). Treat the parent_id as
            # a degenerate root.
            break
        seen.add(parent_row["artifact_doc_id"])
        root_id = parent_row["artifact_doc_id"]
        parent = parent_row["parent_artifact_doc_id"]

    lineage_rows = await conn.fetch(
        """
        WITH RECURSIVE lineage AS (
            SELECT artifact_doc_id, reviewer_id, reviewed_at, state,
                   metadata, created_at, parent_artifact_doc_id
            FROM wiki_review_queue
            WHERE customer_id = $1 AND artifact_doc_id = $2
          UNION
            -- UNION (not UNION ALL) so the recursive step dedupes rows
            -- by their full tuple; this terminates naturally if the
            -- parent_artifact_doc_id graph happens to contain a cycle
            -- (migration 0084 doesn't forbid parent = self at the DB
            -- level, and a hand-edited cycle elsewhere would otherwise
            -- spin the CTE forever). Each row is unique by
            -- artifact_doc_id in non-cyclic chains, so the result set
            -- is identical to UNION ALL in the normal case.
            SELECT w.artifact_doc_id, w.reviewer_id, w.reviewed_at, w.state,
                   w.metadata, w.created_at, w.parent_artifact_doc_id
            FROM wiki_review_queue w
            JOIN lineage l ON w.parent_artifact_doc_id = l.artifact_doc_id
            WHERE w.customer_id = $1
        )
        SELECT * FROM lineage ORDER BY created_at ASC;
        """,
        customer_id, root_id,
    )
    versions = [_version_entry_from_row(r) for r in lineage_rows]
    return _row_to_detail(row, versions=versions)


async def get_detail(
    customer_id: str, artifact_doc_id: str,
) -> WikiArtifactDetail | None:
    """Fetch a single artifact's full detail, including its version lineage.

    Lineage discovery:
      1. Walk up the parent_artifact_doc_id chain to find the root
         (the row whose parent is NULL).
      2. Run a recursive CTE rooted at that ancestor to collect every
         descendant (the chain is single-thread — re-runs after a reject
         produce one child per parent — but the CTE handles fanout
         defensively in case future code introduces branching).

    Returns ``None`` if the requested row does not exist.
    """
    async with with_tenant(customer_id) as conn:
        return await _get_detail_inner(conn, customer_id, artifact_doc_id)
