"""Shared persistence layer for wiki triage + synthesis workers.

Both workers (`triage_worker.py`, `synthesis_worker.py`) live in different
fly apps but operate on the same `wiki_synthesis_queue` and
`wiki_synthesis_runs` rows. This module is the single SQL boundary —
every read and write against those tables flows through here so the
two workers stay coherent on row-level state machine transitions:

    pending  ──[claim_pending_batch]──────────> triaging  (attempts++)
    triaging ──[mark_batch_triaged_and_notify]─> triaged   (UPDATE+NOTIFY same txn)
    triaging ──[mark_rejected]────────────────> rejected  (terminal: triage threshold)
    triaging ──[mark_for_retry]───────────────> pending | failed
    triaging ──[mark_batch_triage_error]──────> pending | failed
    triaged  ──[claim_triaged_rows]───────────> synthesizing  (attempts++)
    synthesizing ──[mark_synthesis_done]────────> done
    synthesizing ──[mark_synthesis_skipped]─────> synthesis_skipped (agent skip / no-op rewrite)
    synthesizing ──[mark_verifier_rejected]─────> verifier_rejected (terminal)
    synthesizing ──[mark_synthesis_error]───────> triaged | failed (retry)

NOTIFY discipline: when the triage worker marks rows triaged, it fires
`pg_notify(WIKI_TRIAGED_CHANNEL, customer_id)` from the same transaction
that committed the UPDATE. Postgres queues the NOTIFY at NOTIFY-time and
delivers it on COMMIT, so the synthesis worker's listener cannot wake
before the rows are visible to other connections. See
`mark_batch_triaged_and_notify` for the canonical pattern.
"""

from __future__ import annotations

from typing import Any

import asyncpg

from services.ingestion.normalizer import fetch_body_from_chunks_for_version
from services.synthesis.models import TriageInput, TriageVerdict
from shared.constants import WIKI_SYNTHESIS_MAX_ATTEMPTS
from shared.db import raw_conn, with_tenant
from shared.logging import get_logger

log = get_logger(__name__)


__all__ = [
    "claim_pending_batch",
    "claim_triaged_rows",
    "close_run",
    "dlq_agent_synthesizing_rows",
    "dlq_customer_for_triage_failure",
    "fetch_bodies",
    "fetch_existing_page",
    "fetch_triaged_manifest",
    "fetch_wiki_index",
    "get_event_body_for_agent",
    "list_pending_customers",
    "list_triaged_customers",
    "mark_batch_triage_error",
    "mark_batch_triaged_and_notify",
    "mark_for_retry",
    "mark_rejected",
    "mark_synthesis_done",
    "mark_synthesis_error",
    "mark_synthesis_skipped",
    "mark_triaged",
    "mark_verifier_rejected",
    "open_run",
    "reclaim_stuck_rows",
    "render_index_markdown",
]


# ---------------------------------------------------------------------------
# Customer discovery
# ---------------------------------------------------------------------------


async def list_pending_customers(conn: asyncpg.Connection) -> list[str]:
    """Customers with at least one row at status='pending' AND wiki opt-in.

    Defense-in-depth: the Normalizer enqueue path is gated on
    `customers.preferences->>'wiki_generation_enabled'`, but a tenant who
    flipped the flag off after enqueue could leave 'pending' rows behind
    — those must NOT drain. JSONB path-text comparison avoids casting a
    missing key (NULL) through ::boolean.
    """
    rows = await conn.fetch(
        """
        SELECT DISTINCT q.customer_id
        FROM wiki_synthesis_queue q
        JOIN customers c ON c.customer_id = q.customer_id
        WHERE q.status = 'pending'
          AND c.preferences->>'wiki_generation_enabled' = 'true'
        """
    )
    return [row["customer_id"] for row in rows]


async def list_triaged_customers(conn: asyncpg.Connection) -> list[str]:
    """Customers with at least one row at status='triaged' AND wiki opt-in.

    Used by the synthesis worker to pick which customers to drain on a
    NOTIFY wake. Same opt-in gate as `list_pending_customers` so a
    tenant who flipped off mid-pipeline doesn't get a partial drain.
    """
    rows = await conn.fetch(
        """
        SELECT DISTINCT q.customer_id
        FROM wiki_synthesis_queue q
        JOIN customers c ON c.customer_id = q.customer_id
        WHERE q.status = 'triaged'
          AND c.preferences->>'wiki_generation_enabled' = 'true'
        """
    )
    return [row["customer_id"] for row in rows]


# ---------------------------------------------------------------------------
# Run rows
# ---------------------------------------------------------------------------


async def open_run(customer_id: str, *, kind: str, stage: str) -> int:
    """Open a wiki_synthesis_runs row.

    `kind` is the trigger flavor ('wake' / 'scheduled' / 'onboarding').
    `stage` is the worker writing this row ('triage' / 'synthesis').
    Triage and synthesis each open their own run per drain; the status
    endpoint filters by stage='synthesis' for pages_* counts.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO wiki_synthesis_runs (customer_id, kind, stage)
            VALUES ($1, $2, $3)
            RETURNING run_id
            """,
            customer_id,
            kind,
            stage,
        )
    return int(row["run_id"])


async def close_run(
    run_id: int,
    *,
    customer_id: str,
    status: str,
    events_total: int,
    events_triaged: int,
    events_kept: int,
    pages_updated: int,
    pages_created: int,
    error: str | None,
) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_runs
            SET finished_at = NOW(),
                status = $2,
                events_total = $3,
                events_triaged = $4,
                events_kept = $5,
                pages_updated = $6,
                pages_created = $7,
                error = $8
            WHERE run_id = $1
            """,
            run_id,
            status,
            events_total,
            events_triaged,
            events_kept,
            pages_updated,
            pages_created,
            error,
        )


# ---------------------------------------------------------------------------
# Claim batches
# ---------------------------------------------------------------------------


async def claim_pending_batch(customer_id: str, *, limit: int) -> list[asyncpg.Record]:
    """Claim up to `limit` pending rows and flip them to 'triaging'.

    `FOR UPDATE SKIP LOCKED` lets multiple wiki-worker machines drain the
    same customer in parallel without double-claiming. Increments
    `attempts` so the retry/failed bookkeeping in `mark_*` works.

    Stamps `heartbeat_at = NOW()` so reclaim_stuck_rows can detect a
    crashed worker and reset/dead-letter rows whose heartbeat goes
    stale.
    """
    async with with_tenant(customer_id) as conn:
        return await conn.fetch(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'triaging',
                attempts = attempts + 1,
                heartbeat_at = NOW()
            WHERE queue_id IN (
                SELECT queue_id FROM wiki_synthesis_queue
                WHERE customer_id = $1
                  AND status = 'pending'
                ORDER BY enqueued_at
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            RETURNING queue_id, doc_id, doc_version, source_system, doc_type,
                      attempts, triage_score, source_ts
            """,
            customer_id,
            limit,
        )


async def claim_triaged_rows(customer_id: str, *, limit: int) -> list[asyncpg.Record]:
    """Claim up to `limit` triaged rows and flip them to 'synthesizing'.

    Increments `attempts` so the synthesis-stage retry loop
    (`mark_synthesis_error` -> 'triaged' -> re-claim) actually advances
    the counter and dead-letters at WIKI_SYNTHESIS_MAX_ATTEMPTS instead
    of looping forever and burning LLM spend.

    v4: ordered by source_ts ASC, queue_id ASC so the wiki agent reads
    the day in time order. The composite index ix_wsq_drain_cursor
    backs this scan.
    """
    async with with_tenant(customer_id) as conn:
        return await conn.fetch(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'synthesizing',
                attempts = attempts + 1,
                heartbeat_at = NOW()
            WHERE queue_id IN (
                SELECT queue_id FROM wiki_synthesis_queue
                WHERE customer_id = $1
                  AND status = 'triaged'
                ORDER BY source_ts, queue_id
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            RETURNING queue_id, doc_id, doc_version, source_system, doc_type,
                      attempts, triage_score, source_ts
            """,
            customer_id,
            limit,
        )


# ---------------------------------------------------------------------------
# Body fetching
# ---------------------------------------------------------------------------


async def fetch_bodies(
    customer_id: str,
    queue_rows: list[asyncpg.Record],
) -> list[TriageInput]:
    """Pull the FULL body of every queued doc from chunks (not preview).

    Body is reconstructed by joining live content chunks per (doc_id,
    version). A missing body falls back to body_preview so triage at
    least has *something* to score on; mid-version doc deletes still
    short-circuit via the `meta_lookup` miss.
    """
    if not queue_rows:
        return []
    doc_ids = [row["doc_id"] for row in queue_rows]
    versions = [row["doc_version"] for row in queue_rows]
    by_queue_id: dict[int, dict[str, Any]] = {
        row["queue_id"]: {
            "doc_id": row["doc_id"],
            "doc_version": row["doc_version"],
            "doc_type": row["doc_type"],
            "source_system": row["source_system"],
        }
        for row in queue_rows
    }

    triage_inputs: list[TriageInput] = []
    async with with_tenant(customer_id) as conn:
        # Filter to live, non-deleted versions. A doc enqueued at
        # version N then soft-deleted at N+1 (deleted_at set on the
        # superseding row, valid_to set on N) must NOT be compiled
        # into a wiki page — that's a privacy / right-to-delete
        # contract. Mid-version deletes drop out of meta_lookup and
        # are handled by the orphan path in synthesis_worker.
        rows = await conn.fetch(
            """
            SELECT q.doc_id, q.version, d.title, d.author_id,
                   d.body_token_count, d.body_preview
            FROM unnest($2::text[], $3::int[]) AS q(doc_id, version)
            JOIN documents d
              ON d.customer_id = $1
             AND d.doc_id = q.doc_id
             AND d.version = q.version
             AND d.valid_to IS NULL
             AND d.deleted_at IS NULL
            """,
            customer_id,
            doc_ids,
            versions,
        )
        meta_lookup: dict[tuple[str, int], asyncpg.Record] = {
            (row["doc_id"], row["version"]): row for row in rows
        }
        for queue_id, info in by_queue_id.items():
            key = (info["doc_id"], info["doc_version"])
            doc_row = meta_lookup.get(key)
            if doc_row is None:
                # Doc was deleted between enqueue and drain. Skip — the
                # queue row's missing-verdict path will mark it.
                continue
            body = await fetch_body_from_chunks_for_version(
                conn,
                customer_id,
                info["doc_id"],
                info["doc_version"],
            )
            if not body:
                body = doc_row["body_preview"] or ""
            triage_inputs.append(
                TriageInput(
                    queue_id=queue_id,
                    doc_id=info["doc_id"],
                    doc_type=info["doc_type"],
                    source_system=info["source_system"],
                    title=doc_row["title"],
                    author_id=doc_row["author_id"],
                    body=body,
                    body_token_count=doc_row["body_token_count"] or 0,
                )
            )
    return triage_inputs


# ---------------------------------------------------------------------------
# State transitions — triage stage
# ---------------------------------------------------------------------------


async def mark_batch_triage_error(
    customer_id: str,
    batch: list[TriageInput],
    error: str,
) -> None:
    queue_ids = [event.queue_id for event in batch]
    if not queue_ids:
        return
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = CASE
                  WHEN attempts >= $3 THEN 'failed'
                  ELSE 'pending'
                END,
                triage_error = $2
            WHERE customer_id = $1 AND queue_id = ANY($4::bigint[])
            """,
            customer_id,
            error,
            WIKI_SYNTHESIS_MAX_ATTEMPTS,
            queue_ids,
        )


async def mark_for_retry(customer_id: str, queue_id: int) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = CASE
                  WHEN attempts >= $2 THEN 'failed'
                  ELSE 'pending'
                END,
                triage_error = COALESCE(triage_error, 'no verdict from triage batch')
            WHERE customer_id = $1 AND queue_id = $3
            """,
            customer_id,
            WIKI_SYNTHESIS_MAX_ATTEMPTS,
            queue_id,
        )


async def mark_rejected(
    customer_id: str,
    queue_id: int,
    verdict: TriageVerdict,
) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'rejected',
                triage_score = $2,
                triage_completed_at = NOW()
            WHERE customer_id = $1 AND queue_id = $3
            """,
            customer_id,
            verdict.score,
            queue_id,
        )


async def mark_triaged(
    customer_id: str,
    queue_id: int,
    verdict: TriageVerdict,
) -> None:
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'triaged',
                triage_score = $2,
                triage_completed_at = NOW()
            WHERE customer_id = $1 AND queue_id = $3
            """,
            customer_id,
            verdict.score,
            queue_id,
        )


async def mark_batch_triaged_and_notify(
    customer_id: str,
    triaged_verdicts: list[tuple[int, TriageVerdict]],
    *,
    notify_channel: str,
) -> None:
    """Mark a batch's triaged rows and wake the synthesis worker atomically.

    Both the bulk UPDATE and the pg_notify run inside a single
    `with_tenant` transaction. Postgres queues NOTIFY at NOTIFY-time and
    delivers it on COMMIT, so the synthesis worker's listener cannot
    wake before the rows are visible to other connections.

    This function exists separately from `mark_triaged` so the per-row
    UPDATE pattern (used for one-off rejects / retries) doesn't accidentally
    fire a NOTIFY. NOTIFY only matters at batch boundaries.

    v4: no longer writes triage_targets — the wiki agent picks pages
    downstream by reading the day in time order.
    """
    if not triaged_verdicts:
        return
    queue_ids = [qid for qid, _ in triaged_verdicts]
    scores = [verdict.score for _, verdict in triaged_verdicts]
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'triaged',
                triage_score = u.score,
                triage_completed_at = NOW()
            FROM unnest($2::bigint[], $3::float[]) AS u(qid, score)
            WHERE customer_id = $1 AND queue_id = u.qid
            """,
            customer_id,
            queue_ids,
            scores,
        )
        # NOTIFY in the same transaction as the UPDATE — Postgres holds
        # the notify in its queue until COMMIT, so the listener wakes
        # only after the triaged rows are visible to other connections.
        await conn.execute(
            "SELECT pg_notify($1, $2)",
            notify_channel,
            customer_id,
        )


async def dlq_customer_for_triage_failure(
    customer_id: str,
    *,
    reason: str,
) -> int:
    """DLQ all pending + triaging rows for a customer after triage crash.

    v4 halt policy: an unrecoverable batch failure (Anthropic outage,
    Gemini outage, repeated parse error) parks the customer's whole
    in-flight slice in DLQ. Admin reset (POST .../dlq/reset) flips
    them back to pending.

    Returns the number of rows DLQ'd. Does NOT use with_tenant because
    wiki_synthesis_queue has RLS disabled (see migration 0034); the
    explicit WHERE customer_id = $1 enforces the scoping.
    """
    async with raw_conn() as conn:
        result = await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'dlq',
                dlq_reason = $2,
                dlq_at = NOW()
            WHERE customer_id = $1
              AND status IN ('pending', 'triaging')
            """,
            customer_id,
            reason,
        )
    # asyncpg returns 'UPDATE N' as the command tag string.
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# State transitions — synthesis stage
# ---------------------------------------------------------------------------


async def mark_synthesis_error(
    customer_id: str,
    queue_ids: list[int],
    error: str,
) -> None:
    if not queue_ids:
        return
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = CASE
                  WHEN attempts >= $3 THEN 'failed'
                  ELSE 'triaged'
                END,
                synthesis_error = $2
            WHERE customer_id = $1 AND queue_id = ANY($4::bigint[])
            """,
            customer_id,
            error,
            WIKI_SYNTHESIS_MAX_ATTEMPTS,
            queue_ids,
        )


async def mark_synthesis_done(
    customer_id: str,
    queue_ids: list[int],
    run_id: int,
) -> None:
    if not queue_ids:
        return
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'done',
                synthesis_run_id = $2,
                synthesis_completed_at = NOW()
            WHERE customer_id = $1 AND queue_id = ANY($3::bigint[])
            """,
            customer_id,
            run_id,
            queue_ids,
        )


async def mark_synthesis_skipped(
    customer_id: str,
    queue_ids: list[int],
    run_id: int,
    *,
    reason: str,
) -> None:
    """Mark events as terminal-but-no-page-change.

    Used by the wiki agent when (a) it explicitly skip_events()'d an
    event as agent-reviewed-but-not-page-changing, or (b) the rewriter
    no-op'd a cluster (should_rewrite=False). Status moves to
    'synthesis_skipped' (terminal v4 state — distinct from 'done',
    which means "page actually rewrote based on this event"). Reason
    is captured in synthesis_error for the audit trail.
    """
    if not queue_ids:
        return
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'synthesis_skipped',
                synthesis_run_id = $2,
                synthesis_completed_at = NOW(),
                synthesis_error = $3
            WHERE customer_id = $1 AND queue_id = ANY($4::bigint[])
            """,
            customer_id,
            run_id,
            f"skipped: {reason}",
            queue_ids,
        )


async def mark_verifier_rejected(
    customer_id: str,
    queue_ids: list[int],
    run_id: int,
    *,
    reason: str,
) -> None:
    """Mark a cluster's queue rows as `verifier_rejected`.

    Distinct from 'done' (synthesized) and 'rejected' (triage threshold
    miss). The verifier decided the cluster doesn't actually change the
    target page after a closer look. Terminal state — these rows do not
    re-enter triage.
    """
    if not queue_ids:
        return
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'verifier_rejected',
                synthesis_run_id = $2,
                synthesis_completed_at = NOW(),
                synthesis_error = $3
            WHERE customer_id = $1 AND queue_id = ANY($4::bigint[])
            """,
            customer_id,
            run_id,
            reason,
            queue_ids,
        )


# ---------------------------------------------------------------------------
# Existing-page lookup + clustering
# ---------------------------------------------------------------------------


async def fetch_existing_page(
    customer_id: str,
    wiki_type: str,
    slug: str,
) -> dict[str, Any] | None:
    """Return the live wiki page for `(wiki_type, slug)`, or None.

    The returned `doc_class` is what the synthesis worker checks before
    deciding whether the cron is allowed to rewrite the body.
    MANUAL_ENTRY pages are read-only; only COMPILED_WIKI / AGENT_ARTIFACT
    pages are open for regeneration.
    """
    from services.ingestion.normalizer import fetch_live_body_from_chunks

    doc_id = f"wiki:{wiki_type}:{slug}"
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT title, doc_class, metadata
            FROM documents
            WHERE customer_id = $1
              AND doc_id = $2
              AND valid_to IS NULL
              AND deleted_at IS NULL
            """,
            customer_id,
            doc_id,
        )
        if row is None:
            return None
        body = await fetch_live_body_from_chunks(conn, customer_id, doc_id)
    metadata = row["metadata"] or {}
    if isinstance(metadata, (str, bytes, bytearray)):
        import orjson

        metadata = orjson.loads(metadata)
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "title": row["title"],
        "doc_class": row["doc_class"],
        "body": body or None,
        "frontmatter": metadata.get("frontmatter") or {},
        "summary": metadata.get("summary"),
    }


# ---------------------------------------------------------------------------
# Index regeneration
# ---------------------------------------------------------------------------


def render_index_markdown(rows: list[asyncpg.Record]) -> str:
    """Deterministic markdown TOC. One entry per live wiki page.

    Falls back to body_preview when the cron-stored summary is absent
    (manual uploads can omit it). Sections grouped by wiki_type.
    """
    if not rows:
        return "# Wiki\n\nNo pages yet.\n"
    by_type: dict[str, list[asyncpg.Record]] = {}
    metas: dict[int, dict[str, Any]] = {}
    for idx, row in enumerate(rows):
        meta = row["metadata"] or {}
        if isinstance(meta, (str, bytes, bytearray)):
            import orjson

            meta = orjson.loads(meta)
        if not isinstance(meta, dict):
            meta = {}
        metas[idx] = meta
        wiki_type = meta.get("wiki_type") or row["source_id"].split(":", 1)[0]
        by_type.setdefault(wiki_type, []).append(row)
    parts = ["# Wiki", ""]
    for wiki_type in sorted(by_type):
        parts.append(f"## {wiki_type.replace('_', ' ').title()}")
        for row in by_type[wiki_type]:
            idx = rows.index(row)
            meta = metas[idx]
            slug = meta.get("slug") or row["source_id"].split(":", 1)[-1]
            title = row["title"] or slug
            summary = meta.get("summary") or row["body_preview"] or ""
            summary = summary.strip().splitlines()[0] if summary else ""
            line = f"- [[{title}]] — {summary}" if summary else f"- [[{title}]]"
            parts.append(line)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Wiki agent helpers (v4)
# ---------------------------------------------------------------------------


async def fetch_wiki_index(customer_id: str) -> list[dict[str, Any]]:
    """Return live COMPILED_WIKI / MANUAL_ENTRY pages for the agent's index.

    The agent reads this once at drain start and keeps it in CachedContent
    so it can pick (wiki_type, slug) targets without paying a round-trip.
    Includes only user-authored types (no auto-index page); the agent
    can call read_page(...) for any individual body.
    """
    from shared.constants import INDEXABLE_WIKI_DOC_TYPES, SourceSystem

    wiki_doc_types = [dt.value for dt in INDEXABLE_WIKI_DOC_TYPES]
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT title, source_id, version, updated_at, metadata
            FROM documents
            WHERE customer_id = $1
              AND source_system = $2
              AND doc_type = ANY($3::text[])
              AND valid_to IS NULL
              AND deleted_at IS NULL
            ORDER BY updated_at DESC
            """,
            customer_id,
            SourceSystem.WIKI.value,
            wiki_doc_types,
        )
    out: list[dict[str, Any]] = []
    for row in rows:
        meta = row["metadata"] or {}
        if isinstance(meta, (str, bytes, bytearray)):
            import orjson

            meta = orjson.loads(meta)
        if not isinstance(meta, dict):
            meta = {}
        wiki_type = meta.get("wiki_type") or row["source_id"].split(":", 1)[0]
        slug = meta.get("slug") or row["source_id"].split(":", 1)[-1]
        out.append(
            {
                "wiki_type": wiki_type,
                "slug": slug,
                "title": row["title"] or slug,
                "summary": meta.get("summary"),
                "last_updated": row["updated_at"],
                "version": row["version"],
            }
        )
    return out


async def fetch_triaged_manifest(
    customer_id: str,
    *,
    excluded_queue_ids: list[int],
    count: int,
) -> tuple[list[dict[str, Any]], int]:
    """Fetch the next manifest window for the agent's next_events tool.

    Reads up to `count` triaged rows ordered by source_ts ASC, queue_id
    ASC; excludes any queue_ids the runtime has already applied or
    skipped this drain. Body is replaced by body_preview to keep
    CachedContent / per-turn token cost bounded.

    Returns (events, remaining) where `remaining` is the count of
    additional triaged rows beyond this window.
    """
    excluded = excluded_queue_ids or [0]
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT q.queue_id, q.doc_id, q.doc_type, q.source_system,
                   q.source_ts, d.title, d.author_id, d.body_preview,
                   d.body_token_count
            FROM wiki_synthesis_queue q
            JOIN documents d
              ON d.customer_id = q.customer_id
             AND d.doc_id = q.doc_id
             AND d.version = q.doc_version
             AND d.valid_to IS NULL
             AND d.deleted_at IS NULL
            WHERE q.customer_id = $1
              AND q.status = 'synthesizing'
              AND NOT (q.queue_id = ANY($2::bigint[]))
            ORDER BY q.source_ts, q.queue_id
            LIMIT $3
            """,
            customer_id,
            excluded,
            count,
        )
        # Get remaining count (cheap; status='synthesizing' is the
        # in-flight slice for this drain).
        remaining = await conn.fetchval(
            """
            SELECT COUNT(*) FROM wiki_synthesis_queue
            WHERE customer_id = $1
              AND status = 'synthesizing'
              AND NOT (queue_id = ANY($2::bigint[]))
            """,
            customer_id,
            excluded,
        )
    events: list[dict[str, Any]] = []
    for row in rows:
        events.append(
            {
                "queue_id": int(row["queue_id"]),
                "doc_id": row["doc_id"],
                "doc_type": row["doc_type"],
                "source_system": row["source_system"],
                "source_ts": row["source_ts"],
                "title": row["title"],
                "author_id": row["author_id"],
                "body_preview": row["body_preview"] or "",
                "body_token_count": int(row["body_token_count"] or 0),
            }
        )
    after_window = max(int(remaining or 0) - len(events), 0)
    return events, after_window


async def get_event_body_for_agent(
    customer_id: str,
    queue_id: int,
) -> tuple[str, dict[str, Any]] | None:
    """Fetch the full body of one triaged event for the agent.

    Returns (body, metadata) where metadata contains doc_id, version,
    title, source_system, source_ts. None if the queue row is missing
    or the doc was deleted between triage and the agent's read.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT q.doc_id, q.doc_version, q.source_ts, q.source_system,
                   d.title, d.author_id
            FROM wiki_synthesis_queue q
            JOIN documents d
              ON d.customer_id = q.customer_id
             AND d.doc_id = q.doc_id
             AND d.version = q.doc_version
             AND d.valid_to IS NULL
             AND d.deleted_at IS NULL
            WHERE q.customer_id = $1 AND q.queue_id = $2
            """,
            customer_id,
            queue_id,
        )
        if row is None:
            return None
        body = await fetch_body_from_chunks_for_version(
            conn, customer_id, row["doc_id"], row["doc_version"]
        )
    return body, {
        "doc_id": row["doc_id"],
        "version": int(row["doc_version"]),
        "title": row["title"],
        "source_system": row["source_system"],
        "source_ts": row["source_ts"],
    }


async def dlq_agent_synthesizing_rows(
    customer_id: str,
    *,
    reason: str,
) -> int:
    """DLQ all 'synthesizing' rows after a wiki agent halt.

    v4 halt policy: an agent halt (turn cap, stall, update cap, Gemini
    outage, compactor crash) parks the customer's whole in-flight slice
    in DLQ. The pending_updates / pending_creates the agent had staged
    are dropped (they were never persisted). Admin reset flips them
    back to triaged for the next drain.

    Returns the number of rows DLQ'd. Does NOT use with_tenant because
    wiki_synthesis_queue has RLS disabled (migration 0034); the
    explicit WHERE customer_id enforces the scoping.
    """
    async with raw_conn() as conn:
        result = await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'dlq',
                dlq_reason = $2,
                dlq_at = NOW(),
                attempts = attempts + 1
            WHERE customer_id = $1
              AND status = 'synthesizing'
            """,
            customer_id,
            reason,
        )
    try:
        return int(result.split()[-1])
    except (ValueError, IndexError):
        return 0


# ---------------------------------------------------------------------------
# Reclaim — recover rows wedged after a worker SIGKILL/OOM
# ---------------------------------------------------------------------------


async def reclaim_stuck_rows(
    *,
    threshold_seconds: int,
    max_attempts: int,
) -> tuple[int, int]:
    """Sweep stuck rows whose heartbeat has gone stale.

    Two terminal paths:
      - `attempts < max_attempts` → reset to prior state ('triaging'
        rows go back to 'pending', 'synthesizing' rows go back to
        'triaged'). The next claim picks them up and bumps attempts.
      - `attempts >= max_attempts` → mark 'failed'. Surfaces in the
        dashboard so ops can investigate. Manual recovery: SQL UPDATE
        back to 'pending' once the root cause is fixed.

    Returns (retried_count, failed_count). Cross-tenant — uses
    `raw_conn` because wiki_synthesis_queue has RLS disabled (see
    migration 0034). The single statement covers both states + both
    branches via CASE so the sweep is one round-trip per cycle.
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            UPDATE wiki_synthesis_queue
            SET status = CASE
                  WHEN attempts >= $2 THEN 'failed'
                  WHEN status = 'triaging'    THEN 'pending'
                  WHEN status = 'synthesizing' THEN 'triaged'
                END,
                heartbeat_at = NULL,
                synthesis_error = CASE
                  WHEN status = 'synthesizing' AND attempts >= $2
                    THEN COALESCE(synthesis_error, '')
                         || ' | reclaimed: heartbeat stale, attempts exhausted'
                  WHEN status = 'synthesizing'
                    THEN COALESCE(synthesis_error, '')
                         || ' | reclaimed: heartbeat stale'
                  ELSE synthesis_error
                END,
                triage_error = CASE
                  WHEN status = 'triaging' AND attempts >= $2
                    THEN COALESCE(triage_error, '')
                         || ' | reclaimed: heartbeat stale, attempts exhausted'
                  WHEN status = 'triaging'
                    THEN COALESCE(triage_error, '')
                         || ' | reclaimed: heartbeat stale'
                  ELSE triage_error
                END
            WHERE status IN ('triaging', 'synthesizing')
              AND heartbeat_at IS NOT NULL
              AND heartbeat_at < NOW() - make_interval(secs => $1)
            RETURNING queue_id, customer_id, status, attempts
            """,
            threshold_seconds,
            max_attempts,
        )
    retried = sum(1 for r in rows if r["status"] in ("pending", "triaged"))
    failed = sum(1 for r in rows if r["status"] == "failed")
    if rows:
        log.warning(
            "wiki_reclaim.reclaimed",
            total=len(rows),
            retried=retried,
            failed=failed,
        )
    return retried, failed
