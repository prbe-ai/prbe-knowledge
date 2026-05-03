"""WikiSynthesisCron — drain the queue, triage with Haiku, synthesize with Sonnet.

Long-running asyncio task that runs concurrently with the existing
ingestion+backfill workers (gathered into `run_worker_forever`). Wakes on
`pg_notify('wiki_synthesize', customer_id)` or on a periodic defensive
timer (`WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS`).

Per tick:
  1. Find every customer with pending queue rows.
  2. Per customer:
     a. Open a wiki_synthesis_runs row.
     b. Loop: claim batch → fetch full doc bodies → token-budget batch →
        Haiku triage → cluster triaged rows by target page → Sonnet
        synthesize per cluster → persist via build_normalization_result +
        Normalizer._persist.
     c. Regenerate the wiki.index page from the live set of wiki pages.
     d. Close the run row.

Triage and synthesis read FULL document bodies from `chunks.content`
(joined in chunk_index order). Until the storage-cleanup migration the
body was duplicated into `documents.metadata->>'body'`; that key is no
longer written. The chunk-join may include ~10-15% overlap inflation
between adjacent chunks, which is harmless for LLM synthesis input.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import asyncpg
from anthropic import AsyncAnthropic

from services.ingestion.handlers.base import ConnectorContext
from services.ingestion.handlers.wiki import (
    INDEX_SLUG,
    WIKI_PAYLOAD_KEY,
    build_normalization_result,
)
from services.ingestion.normalizer import (
    Normalizer,
    fetch_body_from_chunks_for_version,
    fetch_live_body_from_chunks,
)
from services.synthesis.models import (
    SynthesisInput,
    TriageInput,
    TriageVerdict,
    VerifierInput,
)
from services.synthesis.synthesize import (
    SynthesisParseError,
    VerifierParseError,
    call_synthesize,
    call_verifier,
    synthesis_to_normalization,
)
from services.synthesis.triage import (
    TriageParseError,
    call_triage,
    pack_into_batches,
)
from shared.config import get_settings
from shared.constants import (
    WIKI_SYNTHESIS_CLAIM_BATCH,
    WIKI_SYNTHESIS_CLUSTER_MAX_EVENTS,
    WIKI_SYNTHESIS_MAX_ATTEMPTS,
    WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
    WIKI_TRIAGE_SCORE_THRESHOLD,
    CompileTrigger,
    DocClass,
    DocType,
    SourceSystem,
)
from shared.db import raw_conn, with_tenant
from shared.embeddings import Embedder
from shared.logging import get_logger
from shared.models import NormalizationResult, WebhookEvent
from shared.storage import ObjectStore

log = get_logger(__name__)


_INDEX_USER_DOC_TYPES: list[str] = [
    DocType.WIKI_SERVICE_CARD.value,
    DocType.WIKI_DECISION.value,
    DocType.WIKI_FEATURE.value,
    DocType.WIKI_RUNBOOK.value,
]


class WikiSynthesisCron:
    """Orchestrator. Single-instance via the existing worker fly-app discipline."""

    def __init__(
        self,
        ctx: ConnectorContext,
        store: ObjectStore,
        wake_event: asyncio.Event,
        *,
        embedder: Embedder | None = None,
        anthropic_client: AsyncAnthropic | None = None,
        periodic_wake_seconds: float = WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
    ) -> None:
        self._ctx = ctx
        self._store = store
        self._wake = wake_event
        self._normalizer = Normalizer(ctx, store=store, embedder=embedder)
        self._anthropic_client = anthropic_client
        self._periodic = periodic_wake_seconds
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info("wiki_synthesis_cron.start")
        while not self._shutdown.is_set():
            woken_by_notify = await self._wait()
            try:
                await self._tick(woken_by_notify=woken_by_notify)
            except Exception:
                log.exception("wiki_synthesis_cron.tick_failed")
        log.info("wiki_synthesis_cron.stop")

    def shutdown(self) -> None:
        self._shutdown.set()

    # ------------------------------------------------------------------
    # Per-tick scheduling
    # ------------------------------------------------------------------

    async def _wait(self) -> bool:
        """Return True if the tick was woken by NOTIFY, False on periodic timer."""
        shutdown_task = asyncio.create_task(self._shutdown.wait())
        wake_task = asyncio.create_task(self._wake.wait())
        try:
            done, pending = await asyncio.wait(
                {shutdown_task, wake_task},
                timeout=self._periodic,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await t
            return wake_task in done
        finally:
            if self._wake.is_set():
                self._wake.clear()

    async def _tick(self, *, woken_by_notify: bool) -> None:
        if self._shutdown.is_set():
            return
        # Defense-in-depth opt-in gate: only drain customers whose
        # `preferences->>'wiki_generation_enabled'` is explicitly true.
        # The Normalizer enqueue path is gated too, but a tenant who
        # toggled the flag off after enqueue could leave 'pending' rows
        # behind — those must NOT drain. JSONB path-text comparison
        # avoids casting a missing key (NULL) through ::boolean.
        async with raw_conn() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT q.customer_id
                FROM wiki_synthesis_queue q
                JOIN customers c ON c.customer_id = q.customer_id
                WHERE q.status = 'pending'
                  AND c.preferences->>'wiki_generation_enabled' = 'true'
                """
            )
        customer_ids = [row["customer_id"] for row in rows]
        if not customer_ids:
            return
        client = self._resolve_client()
        if client is None:
            log.warning(
                "wiki_synthesis_cron.no_anthropic_key",
                pending_customers=len(customer_ids),
            )
            return
        kind = "wake" if woken_by_notify else "scheduled"
        for customer_id in customer_ids:
            try:
                await self._drain_customer(customer_id, client, run_kind=kind)
            except Exception:
                log.exception("wiki_synthesis_cron.drain_failed", customer=customer_id)

    def _resolve_client(self) -> AsyncAnthropic | None:
        if self._anthropic_client is not None:
            return self._anthropic_client
        settings = get_settings()
        secret = settings.anthropic_api_key
        if secret is None:
            return None
        key = secret.get_secret_value()
        if not key:
            return None
        return AsyncAnthropic(api_key=key)

    # ------------------------------------------------------------------
    # Per-customer drain
    # ------------------------------------------------------------------

    async def _drain_customer(
        self,
        customer_id: str,
        client: AsyncAnthropic,
        *,
        run_kind: str,
    ) -> None:
        run_id = await _open_run(customer_id, kind=run_kind)
        log.info(
            "wiki_synthesis_cron.run_open",
            customer=customer_id,
            run_id=run_id,
            kind=run_kind,
        )
        events_total = events_triaged = events_kept = 0
        pages_updated = pages_created = 0
        run_status = "complete"
        run_error: str | None = None
        try:
            while not self._shutdown.is_set():
                queue_rows = await _claim_batch(customer_id, limit=WIKI_SYNTHESIS_CLAIM_BATCH)
                if not queue_rows:
                    break
                events_total += len(queue_rows)

                triage_inputs = await _fetch_bodies(customer_id, queue_rows)
                # Pack by token budget so each Haiku call stays under context.
                batches = pack_into_batches(triage_inputs)
                verdicts: dict[int, TriageVerdict] = {}
                now = datetime.now(UTC)
                for batch in batches:
                    try:
                        output = await call_triage(client, batch, now=now)
                    except (TriageParseError, Exception) as exc:
                        log.warning(
                            "wiki_synthesis_cron.triage_failed",
                            customer=customer_id,
                            batch_size=len(batch),
                            error=str(exc),
                        )
                        await _mark_batch_triage_error(customer_id, batch, str(exc))
                        continue
                    for qid_str, verdict in output.verdicts.items():
                        try:
                            verdicts[int(qid_str)] = verdict
                        except ValueError:
                            log.warning(
                                "wiki_synthesis_cron.verdict_bad_qid",
                                qid=qid_str,
                            )
                events_triaged += len(verdicts)

                # Apply verdicts to queue rows.
                kept_rows: list[tuple[dict[str, Any], TriageInput, TriageVerdict]] = []
                inputs_by_qid: dict[int, TriageInput] = {inp.queue_id: inp for inp in triage_inputs}
                for row in queue_rows:
                    qid = row["queue_id"]
                    verdict = verdicts.get(qid)
                    if verdict is None:
                        await _mark_for_retry(customer_id, qid)
                        continue
                    if (
                        not verdict.important
                        or verdict.score < WIKI_TRIAGE_SCORE_THRESHOLD
                        or not verdict.targets
                    ):
                        await _mark_rejected(customer_id, qid, verdict)
                        continue
                    await _mark_triaged(customer_id, qid, verdict)
                    inp = inputs_by_qid.get(qid)
                    if inp is not None:
                        kept_rows.append((row, inp, verdict))
                events_kept += len(kept_rows)

                # Cluster triaged rows by (wiki_type, slug).
                clusters = _cluster_kept_rows(kept_rows)
                for (wiki_type, slug), cluster in clusters.items():
                    existing = await _fetch_existing_page(customer_id, wiki_type, slug)
                    cluster_inputs = [item[1] for item in cluster]
                    queue_ids_in_cluster = [item[0]["queue_id"] for item in cluster]

                    # MANUAL_ENTRY pages are read-only to the cron. The design
                    # invariant: humans own pages they author; the synthesizer
                    # observes events that *would* affect them but does not
                    # silently regenerate the body. Mark queue rows as 'done'
                    # with a skip note so they don't keep re-firing the cron.
                    if (
                        existing is not None
                        and existing.get("doc_class") == DocClass.MANUAL_ENTRY.value
                    ):
                        log.info(
                            "wiki_synthesis_cron.skipped_manual_entry",
                            customer=customer_id,
                            wiki_type=wiki_type,
                            slug=slug,
                            event_count=len(queue_ids_in_cluster),
                        )
                        await _mark_synthesis_skipped(
                            customer_id,
                            queue_ids_in_cluster,
                            run_id,
                            reason="page is MANUAL_ENTRY (human-authored)",
                        )
                        continue

                    # Cluster cap. Cluster items arrive in enqueued_at ASC
                    # (oldest first); keep the newest N so the verifier +
                    # synthesize stages stay under the Pro pricing tier
                    # (200K-token boundary). Dropped events still mark
                    # 'done' with a synthesis_error noting the truncation —
                    # they don't keep re-driving the cron, but the audit
                    # trail records the drop.
                    if len(cluster_inputs) > WIKI_SYNTHESIS_CLUSTER_MAX_EVENTS:
                        keep_n = WIKI_SYNTHESIS_CLUSTER_MAX_EVENTS
                        dropped_qids = queue_ids_in_cluster[:-keep_n]
                        cluster_inputs = cluster_inputs[-keep_n:]
                        queue_ids_in_cluster = queue_ids_in_cluster[-keep_n:]
                        log.info(
                            "wiki_synthesis_cron.cluster_truncated",
                            customer=customer_id,
                            wiki_type=wiki_type,
                            slug=slug,
                            dropped=len(dropped_qids),
                            kept=keep_n,
                        )
                        await _mark_synthesis_skipped(
                            customer_id,
                            dropped_qids,
                            run_id,
                            reason=(
                                f"cluster capped at {keep_n} events; "
                                f"dropped {len(dropped_qids)} oldest"
                            ),
                        )

                    action = "update" if existing is not None else "create"

                    # Verifier stage: cheap second look between triage and
                    # synthesize. Empty kept_doc_ids → mark cluster
                    # verifier_rejected (no synthesize). Non-empty → filter
                    # cluster.events to the kept set, then synthesize.
                    verifier_input = VerifierInput(
                        wiki_type=wiki_type,  # type: ignore[arg-type]
                        slug=slug,
                        action=action,  # type: ignore[arg-type]
                        current_title=existing.get("title") if existing else None,
                        current_body=(existing or {}).get("body"),
                        current_summary=(existing or {}).get("summary"),
                        events=cluster_inputs,
                    )
                    try:
                        verifier_output = await call_verifier(
                            client, verifier_input, now=datetime.now(UTC)
                        )
                    except (VerifierParseError, Exception) as exc:
                        log.warning(
                            "wiki_synthesis_cron.verifier_failed",
                            customer=customer_id,
                            wiki_type=wiki_type,
                            slug=slug,
                            error=str(exc),
                        )
                        await _mark_synthesis_error(customer_id, queue_ids_in_cluster, str(exc))
                        continue

                    if not verifier_output.kept_doc_ids:
                        log.info(
                            "wiki_synthesis_cron.verifier_rejected_cluster",
                            customer=customer_id,
                            wiki_type=wiki_type,
                            slug=slug,
                            event_count=len(queue_ids_in_cluster),
                            reason=verifier_output.drop_reason,
                        )
                        await _mark_verifier_rejected(
                            customer_id,
                            queue_ids_in_cluster,
                            run_id,
                            reason=(verifier_output.drop_reason or "verifier rejected cluster"),
                        )
                        continue

                    kept_set = set(verifier_output.kept_doc_ids)
                    filtered_inputs = [inp for inp in cluster_inputs if inp.doc_id in kept_set]
                    if not filtered_inputs:
                        # Verifier returned doc_ids that don't match any
                        # event in the cluster — treat as full rejection.
                        log.warning(
                            "wiki_synthesis_cron.verifier_kept_unknown_docs",
                            customer=customer_id,
                            wiki_type=wiki_type,
                            slug=slug,
                            kept_doc_ids=list(kept_set),
                        )
                        await _mark_verifier_rejected(
                            customer_id,
                            queue_ids_in_cluster,
                            run_id,
                            reason="verifier kept_doc_ids did not match cluster",
                        )
                        continue

                    synth_input = SynthesisInput(
                        wiki_type=wiki_type,  # type: ignore[arg-type]
                        slug=slug,
                        action=action,  # type: ignore[arg-type]
                        current_title=existing.get("title") if existing else None,
                        current_body=(existing or {}).get("body"),
                        current_frontmatter=(existing or {}).get("frontmatter") or {},
                        current_summary=(existing or {}).get("summary"),
                        events=filtered_inputs,
                    )
                    try:
                        synth_output = await call_synthesize(
                            client, synth_input, now=datetime.now(UTC)
                        )
                    except (SynthesisParseError, Exception) as exc:
                        log.warning(
                            "wiki_synthesis_cron.synthesize_failed",
                            customer=customer_id,
                            wiki_type=wiki_type,
                            slug=slug,
                            error=str(exc),
                        )
                        await _mark_synthesis_error(customer_id, queue_ids_in_cluster, str(exc))
                        continue
                    compile_trigger = (
                        CompileTrigger.SOURCE_UPDATE
                        if run_kind == "wake"
                        else CompileTrigger.SCHEDULED
                    )
                    norm = synthesis_to_normalization(
                        customer_id,
                        synth_input,
                        synth_output,
                        run_id=run_id,
                        compile_trigger=compile_trigger,
                    )
                    try:
                        await self._normalizer._persist(customer_id, SourceSystem.WIKI, norm)
                    except Exception as exc:
                        log.warning(
                            "wiki_synthesis_cron.persist_failed",
                            customer=customer_id,
                            wiki_type=wiki_type,
                            slug=slug,
                            error=str(exc),
                        )
                        await _mark_synthesis_error(customer_id, queue_ids_in_cluster, str(exc))
                        continue
                    if existing is None:
                        pages_created += 1
                    else:
                        pages_updated += 1
                    # Mark all queue rows in the cluster done — including
                    # those whose doc_ids were dropped by the verifier.
                    # Their participation in the cluster is complete; if
                    # they're also in another cluster, that cluster's run
                    # will handle them independently.
                    await _mark_synthesis_done(customer_id, queue_ids_in_cluster, run_id)

            # ----- regenerate the index after the drain finishes
            try:
                await self._regenerate_index(customer_id, run_id)
            except Exception as exc:
                log.warning(
                    "wiki_synthesis_cron.index_regen_failed",
                    customer=customer_id,
                    error=str(exc),
                )
        except Exception as exc:
            run_status = "failed"
            run_error = str(exc)
            raise
        finally:
            await _close_run(
                run_id,
                customer_id=customer_id,
                status=run_status,
                events_total=events_total,
                events_triaged=events_triaged,
                events_kept=events_kept,
                pages_updated=pages_updated,
                pages_created=pages_created,
                error=run_error,
            )
            log.info(
                "wiki_synthesis_cron.run_close",
                customer=customer_id,
                run_id=run_id,
                status=run_status,
                events_total=events_total,
                events_triaged=events_triaged,
                events_kept=events_kept,
                pages_updated=pages_updated,
                pages_created=pages_created,
            )

    # ------------------------------------------------------------------
    # Index regeneration
    # ------------------------------------------------------------------

    async def _regenerate_index(self, customer_id: str, run_id: int) -> None:
        """Aggregate all live wiki pages into a single WIKI_INDEX document.

        Deterministic — no LLM call. Each entry pulls the cron-stored
        per-page summary; falls back to body_preview when the summary is
        absent (manual uploads can omit it).
        """
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(
                """
                SELECT title, body_preview, source_id, version, updated_at,
                       metadata
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
                _INDEX_USER_DOC_TYPES,
            )
        body = _render_index_markdown(rows)
        received_at = datetime.now(UTC)
        raw_payload: dict[str, Any] = {
            WIKI_PAYLOAD_KEY: {
                "wiki_type": "index",
                "slug": INDEX_SLUG,
                "title": "Wiki — Table of Contents",
                "body": body,
                "frontmatter": {"page_count": len(rows)},
                "doc_class": DocClass.AGENT_ARTIFACT.value,
                "is_delete": False,
                "updated_at": received_at.isoformat(),
                "summary": f"Auto-generated table of contents ({len(rows)} pages).",
                "commit_message": (
                    f"Regenerate index ({len(rows)} pages) from synthesis run #{run_id}"
                ),
                "commit_author": "agent:wiki-synthesis-cron",
                "commit_run_id": run_id,
                "author_id": "agent:wiki-synthesis-cron",
            }
        }
        event = WebhookEvent(
            customer_id=customer_id,
            source_system=SourceSystem.WIKI,
            source_event_id=f"index:{INDEX_SLUG}:edit:{received_at.isoformat()}",
            received_at=received_at,
            payload_s3_key="",
            payload_s3_keys=[],
            raw_payload=raw_payload,
            headers={},
        )
        norm: NormalizationResult = build_normalization_result(event)
        await self._normalizer._persist(customer_id, SourceSystem.WIKI, norm)


# ----------------------------------------------------------------------
# Module-level helpers — testable independently
# ----------------------------------------------------------------------


async def _open_run(customer_id: str, *, kind: str) -> int:
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


async def _close_run(
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


async def _claim_batch(customer_id: str, *, limit: int) -> list[asyncpg.Record]:
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
                      attempts
            """,
            customer_id,
            limit,
        )


async def _fetch_bodies(
    customer_id: str,
    queue_rows: list[asyncpg.Record],
) -> list[TriageInput]:
    """Pull the FULL body of every queued doc — no chunks, no preview."""
    if not queue_rows:
        return []
    doc_ids = [row["doc_id"] for row in queue_rows]
    versions = [row["doc_version"] for row in queue_rows]
    queue_ids = [row["queue_id"] for row in queue_rows]
    doc_types = [row["doc_type"] for row in queue_rows]
    sources = [row["source_system"] for row in queue_rows]
    by_queue_id: dict[int, dict[str, Any]] = {
        qid: {
            "doc_id": doc_id,
            "doc_version": version,
            "doc_type": doc_type,
            "source_system": source,
        }
        for qid, doc_id, version, doc_type, source in zip(
            queue_ids, doc_ids, versions, doc_types, sources, strict=True
        )
    }

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT q.doc_id, q.version, d.title, d.author_id,
                   d.body_token_count, d.body_preview
            FROM unnest($2::text[], $3::int[]) AS q(doc_id, version)
            JOIN documents d
              ON d.customer_id = $1
             AND d.doc_id = q.doc_id
             AND d.version = q.version
            """,
            customer_id,
            doc_ids,
            versions,
        )

        meta_lookup: dict[tuple[str, int], asyncpg.Record] = {
            (row["doc_id"], row["version"]): row for row in rows
        }
        # Body is reconstructed by joining live content chunks per (doc_id,
        # version). Fetched inside the same with_tenant block so RLS sees the
        # tenant GUC. A missing body falls back to body_preview so triage at
        # least has *something* to score on; mid-version doc deletes still
        # short-circuit via the meta_lookup miss above.
        triage_inputs: list[TriageInput] = []
        for queue_id, info in by_queue_id.items():
            key = (info["doc_id"], info["doc_version"])
            doc_row = meta_lookup.get(key)
            if doc_row is None:
                # Doc was deleted out from under us between enqueue and drain.
                # Skip — the queue row will be marked rejected by the missing
                # verdict path.
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


async def _mark_batch_triage_error(
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


async def _mark_for_retry(customer_id: str, queue_id: int) -> None:
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


async def _mark_rejected(
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
            _verdict_targets_json(verdict),
            queue_id,
        )


async def _mark_triaged(
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
            _verdict_targets_json(verdict),
            queue_id,
        )


async def _mark_synthesis_error(
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


async def _mark_synthesis_done(
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


async def _mark_synthesis_skipped(
    customer_id: str,
    queue_ids: list[int],
    run_id: int,
    *,
    reason: str,
) -> None:
    """Mark events 'done' without firing synthesis.

    Used when the cron declines to clobber a page (e.g. MANUAL_ENTRY) or
    when the cluster cap drops oldest events. The events still complete —
    they don't keep re-driving the cron — but the audit trail records
    why no synthesis occurred via synthesis_error.
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


async def _mark_verifier_rejected(
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


async def _fetch_existing_page(
    customer_id: str,
    wiki_type: str,
    slug: str,
) -> dict[str, Any] | None:
    """Return the live wiki page for `(wiki_type, slug)`, or None.

    The returned `doc_class` is what the cluster loop checks before deciding
    whether the cron is allowed to rewrite the body. MANUAL_ENTRY pages are
    read-only to the cron; only COMPILED_WIKI / AGENT_ARTIFACT pages are
    open for regeneration.
    """
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


def _cluster_kept_rows(
    kept_rows: list[tuple[dict[str, Any], TriageInput, TriageVerdict]],
) -> dict[tuple[str, str], list[tuple[dict[str, Any], TriageInput]]]:
    out: dict[tuple[str, str], list[tuple[dict[str, Any], TriageInput]]] = {}
    for row, inp, verdict in kept_rows:
        for target in verdict.targets:
            key = (target.wiki_type, target.slug)
            out.setdefault(key, []).append((row, inp))
    return out


def _verdict_targets_json(verdict: TriageVerdict) -> str:
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


def _render_index_markdown(rows: list[asyncpg.Record]) -> str:
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
