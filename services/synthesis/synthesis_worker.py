"""SynthesisWorker — drain triaged rows through verifier + synthesize.

Runs in the prbe-knowledge-wiki-synthesis fly app (2 x 3GB).

Per tick:
  1. SELECT customers with triaged rows AND wiki_generation_enabled.
  2. Fan out per-customer drains via asyncio.gather + semaphore.
  3. Per customer:
     a. Open a wiki_synthesis_runs row (kind='wake'/'scheduled'). The
        triage worker opened its own run row for the triage half — this
        run row tracks the synthesize half. Two run rows per end-to-end
        drain is an audit feature, not a bug.
     b. Loop: claim triaged batch → cluster by (wiki_type, slug) →
        fan-out per-cluster verifier+synthesize via semaphore.
     c. Per cluster:
          - Fetch existing page; skip with reason if MANUAL_ENTRY.
          - Apply cluster cap (oldest-first drop above max).
          - Verifier call. Empty kept_doc_ids → mark verifier_rejected.
            Non-empty → filter cluster.events to kept set.
          - Synthesize call → persist via Normalizer._persist (same
            path as the manual-upload route) → mark queue rows done.
     d. Regenerate the wiki.index page from the live set.
     e. Close the run row.

Reads FULL doc bodies via `persistence.fetch_bodies` (chunks join), not
chunks-as-retrieval.
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import UTC, datetime
from typing import Any

import asyncpg
import orjson
from anthropic import AsyncAnthropic

from services.ingestion.handlers.base import ConnectorContext, make_default_context
from services.ingestion.handlers.wiki import (
    INDEX_SLUG,
    WIKI_PAYLOAD_KEY,
    build_normalization_result,
)
from services.ingestion.normalizer import Normalizer
from services.synthesis import persistence
from services.synthesis.models import (
    SynthesisInput,
    TriageInput,
    VerifierInput,
)
from services.synthesis.synthesize import (
    call_synthesize,
    call_verifier,
    synthesis_to_normalization,
)
from shared.config import get_settings
from shared.constants import (
    WIKI_SYNTHESIS_CLAIM_BATCH,
    WIKI_SYNTHESIS_CLUSTER_CONCURRENCY,
    WIKI_SYNTHESIS_CLUSTER_MAX_EVENTS,
    WIKI_SYNTHESIS_CUSTOMER_CONCURRENCY,
    WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
    CompileTrigger,
    DocClass,
    DocType,
    SourceSystem,
)
from shared.db import raw_conn, with_tenant
from shared.embeddings import Embedder
from shared.logging import get_logger
from shared.models import NormalizationResult, WebhookEvent
from shared.storage import ObjectStore, get_store

log = get_logger(__name__)


_INDEX_USER_DOC_TYPES: list[str] = [
    DocType.WIKI_SERVICE_CARD.value,
    DocType.WIKI_DECISION.value,
    DocType.WIKI_FEATURE.value,
    DocType.WIKI_RUNBOOK.value,
]


class SynthesisWorker:
    """Drain triaged rows through verifier + synthesize."""

    def __init__(
        self,
        wake_event: asyncio.Event,
        *,
        ctx: ConnectorContext | None = None,
        store: ObjectStore | None = None,
        embedder: Embedder | None = None,
        anthropic_client: AsyncAnthropic | None = None,
        periodic_wake_seconds: float = WIKI_SYNTHESIS_PERIODIC_WAKE_SECONDS,
        customer_concurrency: int = WIKI_SYNTHESIS_CUSTOMER_CONCURRENCY,
        cluster_concurrency: int = WIKI_SYNTHESIS_CLUSTER_CONCURRENCY,
    ) -> None:
        self._wake = wake_event
        self._ctx = ctx or make_default_context()
        self._store = store or get_store()
        self._normalizer = Normalizer(self._ctx, store=self._store, embedder=embedder)
        self._anthropic_client = anthropic_client
        self._periodic = periodic_wake_seconds
        self._customer_sem = asyncio.Semaphore(customer_concurrency)
        self._cluster_sem = asyncio.Semaphore(cluster_concurrency)
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info("synthesis_worker.start")
        while not self._shutdown.is_set():
            woken_by_notify = await self._wait()
            try:
                await self._tick(woken_by_notify=woken_by_notify)
            except Exception:
                log.exception("synthesis_worker.tick_failed")
        log.info("synthesis_worker.stop")

    def shutdown(self) -> None:
        self._shutdown.set()

    async def _wait(self) -> bool:
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
        async with raw_conn() as conn:
            customer_ids = await persistence.list_triaged_customers(conn)
        if not customer_ids:
            return
        client = self._resolve_client()
        if client is None:
            log.warning(
                "synthesis_worker.no_anthropic_key",
                triaged_customers=len(customer_ids),
            )
            return
        kind = "wake" if woken_by_notify else "scheduled"

        async def _drain(cid: str) -> None:
            async with self._customer_sem:
                try:
                    await self._drain_customer(cid, client, run_kind=kind)
                except Exception:
                    log.exception("synthesis_worker.drain_failed", customer=cid)

        await asyncio.gather(*[_drain(cid) for cid in customer_ids])

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

    async def _drain_customer(
        self,
        customer_id: str,
        client: AsyncAnthropic,
        *,
        run_kind: str,
    ) -> None:
        run_id = await persistence.open_run(customer_id, kind=run_kind)
        log.info(
            "synthesis_worker.run_open",
            customer=customer_id,
            run_id=run_id,
            kind=run_kind,
        )
        pages_updated = pages_created = 0
        run_status = "complete"
        run_error: str | None = None
        try:
            while not self._shutdown.is_set():
                claimed = await persistence.claim_triaged_rows(
                    customer_id, limit=WIKI_SYNTHESIS_CLAIM_BATCH
                )
                if not claimed:
                    break
                # Reconstruct cluster -> events mapping from triage_targets.
                cluster_map = await self._build_clusters(
                    customer_id, claimed, run_id=run_id
                )

                # Fan out cluster processing.
                async def _process(
                    key: tuple[str, str],
                    items: list[tuple[asyncpg.Record, TriageInput, list[dict[str, Any]]]],
                ) -> tuple[int, int]:
                    async with self._cluster_sem:
                        return await self._process_cluster(
                            customer_id,
                            key,
                            items,
                            client=client,
                            run_id=run_id,
                            run_kind=run_kind,
                        )

                results = await asyncio.gather(
                    *[_process(k, v) for k, v in cluster_map.items()],
                    return_exceptions=True,
                )
                for r in results:
                    if isinstance(r, BaseException):
                        log.warning(
                            "synthesis_worker.cluster_exception",
                            customer=customer_id,
                            error=str(r),
                        )
                        continue
                    created, updated = r
                    pages_created += created
                    pages_updated += updated

            # Regenerate the index after drain.
            try:
                await self._regenerate_index(customer_id, run_id)
            except Exception as exc:
                log.warning(
                    "synthesis_worker.index_regen_failed",
                    customer=customer_id,
                    error=str(exc),
                )
        except Exception as exc:
            run_status = "failed"
            run_error = str(exc)
            raise
        finally:
            await persistence.close_run(
                run_id,
                customer_id=customer_id,
                status=run_status,
                events_total=0,
                events_triaged=0,
                events_kept=0,
                pages_updated=pages_updated,
                pages_created=pages_created,
                error=run_error,
            )
            log.info(
                "synthesis_worker.run_close",
                customer=customer_id,
                run_id=run_id,
                status=run_status,
                pages_updated=pages_updated,
                pages_created=pages_created,
            )

    async def _build_clusters(
        self,
        customer_id: str,
        claimed: list[asyncpg.Record],
        *,
        run_id: int,
    ) -> dict[
        tuple[str, str],
        list[tuple[asyncpg.Record, TriageInput, list[dict[str, Any]]]],
    ]:
        """Fetch bodies + parse stored triage_targets JSON into cluster map.

        Each claimed row's `triage_targets` JSON has the shape produced
        by `persistence.verdict_targets_json`. The list of `targets`
        determines which (wiki_type, slug) clusters this row joins.

        `run_id` stamps any orphan-row skip-marks so the audit trail
        links back to the actual open run.
        """
        triage_inputs = await persistence.fetch_bodies(customer_id, claimed)
        inputs_by_qid: dict[int, TriageInput] = {inp.queue_id: inp for inp in triage_inputs}
        clusters: dict[
            tuple[str, str],
            list[tuple[asyncpg.Record, TriageInput, list[dict[str, Any]]]],
        ] = {}
        orphan_qids: list[int] = []
        no_target_qids: list[int] = []
        for row in claimed:
            qid = row["queue_id"]
            inp = inputs_by_qid.get(qid)
            if inp is None:
                # Doc was deleted between claim and body-fetch.
                orphan_qids.append(qid)
                continue
            targets_json = row["triage_targets"]
            if isinstance(targets_json, (str, bytes, bytearray)):
                try:
                    parsed = orjson.loads(targets_json)
                except orjson.JSONDecodeError:
                    parsed = {}
            elif isinstance(targets_json, dict):
                parsed = targets_json
            else:
                parsed = {}
            targets = parsed.get("targets") or []
            if not isinstance(targets, list) or not targets:
                # Triage said "yes important" but didn't pin a target —
                # shouldn't happen given the threshold gate, but mark
                # done so the row doesn't loop.
                no_target_qids.append(qid)
                continue
            for target in targets:
                if not isinstance(target, dict):
                    continue
                wiki_type = target.get("wiki_type")
                slug = target.get("slug")
                if not wiki_type or not slug:
                    continue
                clusters.setdefault((wiki_type, slug), []).append((row, inp, targets))

        # Bulk-mark orphan rows in one round-trip each, stamped with the
        # actual open run_id (not 0) so the audit trail links correctly.
        if orphan_qids:
            await persistence.mark_synthesis_skipped(
                customer_id,
                orphan_qids,
                run_id,
                reason="document missing at synthesize",
            )
        if no_target_qids:
            await persistence.mark_synthesis_skipped(
                customer_id,
                no_target_qids,
                run_id,
                reason="no triage targets",
            )
        return clusters

    async def _process_cluster(
        self,
        customer_id: str,
        key: tuple[str, str],
        cluster_items: list[tuple[asyncpg.Record, TriageInput, list[dict[str, Any]]]],
        *,
        client: AsyncAnthropic,
        run_id: int,
        run_kind: str,
    ) -> tuple[int, int]:
        """Run one cluster through verifier + synthesize.

        Returns (pages_created, pages_updated) — both 0 if the cluster
        was skipped (MANUAL_ENTRY) or rejected (verifier).
        """
        wiki_type, slug = key
        existing = await persistence.fetch_existing_page(customer_id, wiki_type, slug)
        cluster_inputs = [item[1] for item in cluster_items]
        queue_ids_in_cluster = [item[0]["queue_id"] for item in cluster_items]

        if existing is not None and existing.get("doc_class") == DocClass.MANUAL_ENTRY.value:
            log.info(
                "synthesis_worker.skipped_manual_entry",
                customer=customer_id,
                wiki_type=wiki_type,
                slug=slug,
                event_count=len(queue_ids_in_cluster),
            )
            await persistence.mark_synthesis_skipped(
                customer_id,
                queue_ids_in_cluster,
                run_id,
                reason="page is MANUAL_ENTRY (human-authored)",
            )
            return (0, 0)

        # Cluster cap. Items arrive in claim order (triage_completed_at
        # NULLS FIRST, then enqueued_at) — newest are at the end. Drop
        # oldest (front of list) past the cap.
        if len(cluster_inputs) > WIKI_SYNTHESIS_CLUSTER_MAX_EVENTS:
            keep_n = WIKI_SYNTHESIS_CLUSTER_MAX_EVENTS
            dropped_qids = queue_ids_in_cluster[:-keep_n]
            cluster_inputs = cluster_inputs[-keep_n:]
            queue_ids_in_cluster = queue_ids_in_cluster[-keep_n:]
            log.info(
                "synthesis_worker.cluster_truncated",
                customer=customer_id,
                wiki_type=wiki_type,
                slug=slug,
                dropped=len(dropped_qids),
                kept=keep_n,
            )
            await persistence.mark_synthesis_skipped(
                customer_id,
                dropped_qids,
                run_id,
                reason=(f"cluster capped at {keep_n} events; dropped {len(dropped_qids)} oldest"),
            )

        action = "update" if existing is not None else "create"
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
            verifier_output = await call_verifier(client, verifier_input, now=datetime.now(UTC))
        except Exception as exc:
            log.warning(
                "synthesis_worker.verifier_failed",
                customer=customer_id,
                wiki_type=wiki_type,
                slug=slug,
                error=str(exc),
            )
            await persistence.mark_synthesis_error(customer_id, queue_ids_in_cluster, str(exc))
            return (0, 0)

        if not verifier_output.kept_doc_ids:
            log.info(
                "synthesis_worker.verifier_rejected_cluster",
                customer=customer_id,
                wiki_type=wiki_type,
                slug=slug,
                event_count=len(queue_ids_in_cluster),
                reason=verifier_output.drop_reason,
            )
            await persistence.mark_verifier_rejected(
                customer_id,
                queue_ids_in_cluster,
                run_id,
                reason=(verifier_output.drop_reason or "verifier rejected cluster"),
            )
            return (0, 0)

        kept_set = set(verifier_output.kept_doc_ids)
        filtered_inputs = [inp for inp in cluster_inputs if inp.doc_id in kept_set]
        if not filtered_inputs:
            log.warning(
                "synthesis_worker.verifier_kept_unknown_docs",
                customer=customer_id,
                wiki_type=wiki_type,
                slug=slug,
                kept_doc_ids=list(kept_set),
            )
            await persistence.mark_verifier_rejected(
                customer_id,
                queue_ids_in_cluster,
                run_id,
                reason="verifier kept_doc_ids did not match cluster",
            )
            return (0, 0)

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
            synth_output = await call_synthesize(client, synth_input, now=datetime.now(UTC))
        except Exception as exc:
            log.warning(
                "synthesis_worker.synthesize_failed",
                customer=customer_id,
                wiki_type=wiki_type,
                slug=slug,
                error=str(exc),
            )
            await persistence.mark_synthesis_error(customer_id, queue_ids_in_cluster, str(exc))
            return (0, 0)

        compile_trigger = (
            CompileTrigger.SOURCE_UPDATE if run_kind == "wake" else CompileTrigger.SCHEDULED
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
                "synthesis_worker.persist_failed",
                customer=customer_id,
                wiki_type=wiki_type,
                slug=slug,
                error=str(exc),
            )
            await persistence.mark_synthesis_error(customer_id, queue_ids_in_cluster, str(exc))
            return (0, 0)

        await persistence.mark_synthesis_done(customer_id, queue_ids_in_cluster, run_id)
        if existing is None:
            return (1, 0)
        return (0, 1)

    async def _regenerate_index(self, customer_id: str, run_id: int) -> None:
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
        body = persistence.render_index_markdown(rows)
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


__all__ = ["SynthesisWorker"]
