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
    synthesizing ──[mark_synthesis_skipped]─────> done (MANUAL_ENTRY guard, cluster cap)
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
from shared.db import with_tenant
from shared.logging import get_logger

log = get_logger(__name__)


__all__ = [
    "claim_pending_batch",
    "claim_triaged_rows",
    "close_run",
    "cluster_kept_rows",
    "fetch_bodies",
    "fetch_existing_page",
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
    "render_index_markdown",
    "verdict_targets_json",
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


async def open_run(customer_id: str, *, kind: str) -> int:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO wiki_synthesis_runs (customer_id, kind)
            VALUES ($1, $2)
            RETURNING run_id
            """,
            customer_id,
            kind,
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
    """
    async with with_tenant(customer_id) as conn:
        return await conn.fetch(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'triaging',
                attempts = attempts + 1
            WHERE queue_id IN (
                SELECT queue_id FROM wiki_synthesis_queue
                WHERE customer_id = $1
                  AND status = 'pending'
                ORDER BY enqueued_at
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            RETURNING queue_id, doc_id, doc_version, source_system, doc_type,
                      attempts, triage_score, triage_targets
            """,
            customer_id,
            limit,
        )


async def claim_triaged_rows(customer_id: str, *, limit: int) -> list[asyncpg.Record]:
    """Claim up to `limit` triaged rows and flip them to 'synthesizing'.

    Returns the rows along with their stored `triage_targets` JSON so the
    synthesis worker can rebuild the cluster map without re-running
    triage.

    Increments `attempts` so the synthesis-stage retry loop
    (`mark_synthesis_error` → 'triaged' → re-claim) actually advances
    the counter and dead-letters at WIKI_SYNTHESIS_MAX_ATTEMPTS instead
    of looping forever and burning LLM spend.
    """
    async with with_tenant(customer_id) as conn:
        return await conn.fetch(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'synthesizing',
                attempts = attempts + 1
            WHERE queue_id IN (
                SELECT queue_id FROM wiki_synthesis_queue
                WHERE customer_id = $1
                  AND status = 'triaged'
                ORDER BY triage_completed_at NULLS FIRST, enqueued_at
                LIMIT $2
                FOR UPDATE SKIP LOCKED
            )
            RETURNING queue_id, doc_id, doc_version, source_system, doc_type,
                      attempts, triage_score, triage_targets
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
                triage_targets = $3::jsonb,
                triage_completed_at = NOW()
            WHERE customer_id = $1 AND queue_id = $4
            """,
            customer_id,
            verdict.score,
            verdict_targets_json(verdict),
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
                triage_targets = $3::jsonb,
                triage_completed_at = NOW()
            WHERE customer_id = $1 AND queue_id = $4
            """,
            customer_id,
            verdict.score,
            verdict_targets_json(verdict),
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
    """
    if not triaged_verdicts:
        return
    queue_ids = [qid for qid, _ in triaged_verdicts]
    scores = [verdict.score for _, verdict in triaged_verdicts]
    targets = [verdict_targets_json(verdict) for _, verdict in triaged_verdicts]
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'triaged',
                triage_score = u.score,
                triage_targets = u.targets::jsonb,
                triage_completed_at = NOW()
            FROM unnest($2::bigint[], $3::float[], $4::text[])
                 AS u(qid, score, targets)
            WHERE customer_id = $1 AND queue_id = u.qid
            """,
            customer_id,
            queue_ids,
            scores,
            targets,
        )
        # NOTIFY in the same transaction as the UPDATE — Postgres holds
        # the notify in its queue until COMMIT, so the listener wakes
        # only after the triaged rows are visible to other connections.
        await conn.execute(
            "SELECT pg_notify($1, $2)",
            notify_channel,
            customer_id,
        )


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
    """Mark events 'done' without firing synthesis.

    Used when the synthesis worker declines to clobber a page (e.g.
    MANUAL_ENTRY) or when the cluster cap drops oldest events. The
    events still complete — they don't keep re-driving the cron — but
    the audit trail records why no synthesis occurred via
    synthesis_error.
    """
    if not queue_ids:
        return
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE wiki_synthesis_queue
            SET status = 'done',
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


def cluster_kept_rows(
    kept_rows: list[tuple[dict[str, Any], TriageInput, TriageVerdict]],
) -> dict[tuple[str, str], list[tuple[dict[str, Any], TriageInput]]]:
    out: dict[tuple[str, str], list[tuple[dict[str, Any], TriageInput]]] = {}
    for row, inp, verdict in kept_rows:
        for target in verdict.targets:
            key = (target.wiki_type, target.slug)
            out.setdefault(key, []).append((row, inp))
    return out


def verdict_targets_json(verdict: TriageVerdict) -> str:
    """Serialize triage_targets as a JSONB-friendly string."""
    import orjson

    return orjson.dumps(
        {
            "important": verdict.important,
            "score": verdict.score,
            "reason": verdict.reason,
            "targets": [
                {"wiki_type": t.wiki_type, "slug": t.slug, "action": t.action}
                for t in verdict.targets
            ],
        }
    ).decode("utf-8")


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
