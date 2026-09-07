"""Protocol-v2 drain composed into the existing ingestion-worker process.

DB leases survive restarts. Provider and embedding calls are bounded and outside
write transactions. Separate queue/history slices reserve capacity for both live
changes and history, and tenant iteration prevents a large installation starving
other tenants. No task is launched from an HTTP request.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import replace
from uuid import uuid4

from engine.ingest.normalizer import Normalizer
from engine.shared.constants import SourceSystem
from engine.shared.db import raw_conn, with_tenant
from engine.shared.exceptions import DuplicateEventIgnored, UnsupportedEventType
from engine.shared.logging import get_logger
from engine.shared.models import WebhookEvent
from kb.github_control import decoded, enqueue_event, installation_token
from kb.github_control_routes import list_repositories
from kb.handlers.github import GitHubConnector

log = get_logger(__name__)


class GitHubControlWorker:
    def __init__(self, ctx):
        self.ctx = ctx
        self.worker_id = str(uuid4())
        self.shutdown_event = asyncio.Event()
        self.normalizer = Normalizer(ctx)
        # Native cursor stores the current repository's page cursor, not every
        # concurrently running repository. Serial walks make checkpoints exact.
        self.connector = GitHubConnector(
            replace(
                ctx,
                settings=ctx.settings.model_copy(update={"github_backfill_repo_concurrency": 1}),
            )
        )
        self.connector.strict_backfill = True

    def shutdown(self):
        self.shutdown_event.set()

    async def _heartbeat(self):
        while not self.shutdown_event.is_set():
            async with raw_conn() as conn:
                await conn.execute(
                    """INSERT INTO github_worker_capabilities(worker_id,protocol_version)
                    VALUES ($1,2) ON CONFLICT(worker_id) DO UPDATE SET heartbeat_at=now()""",
                    self.worker_id,
                )
                await conn.execute(
                    "DELETE FROM github_worker_capabilities WHERE heartbeat_at < now()-interval '1 day'"
                )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.shutdown_event.wait(), 15)

    async def run(self):
        heartbeat = asyncio.create_task(self._heartbeat())
        try:
            while not self.shutdown_event.is_set():
                if heartbeat.done():
                    heartbeat.result()  # never advertise readiness with a dead heartbeat/drain
                async with raw_conn() as conn:
                    tenants = await conn.fetch(
                        "SELECT customer_id FROM customers ORDER BY customer_id"
                    )
                did_work = False
                for tenant in tenants:
                    customer = tenant["customer_id"]
                    try:
                        await self.reconcile(customer)
                        # Live has a reserved slot, history still gets one even
                        # under continuous webhook pressure.
                        did_work |= await self.queue_step(customer, live=True)
                        did_work |= await self.history_step(customer)
                        did_work |= await self.queue_step(customer, live=False)
                    except Exception:
                        log.exception("github_v2.tick_failed", customer=customer)
                    if self.shutdown_event.is_set():
                        break
                if not did_work:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self.shutdown_event.wait(), 2)
        finally:
            heartbeat.cancel()
            await asyncio.gather(heartbeat, return_exceptions=True)
            async with raw_conn() as conn:
                await conn.execute(
                    "DELETE FROM github_worker_capabilities WHERE worker_id=$1", self.worker_id
                )

    async def reconcile(self, customer_id):
        async with with_tenant(customer_id) as conn:
            # A worker that crashes on its last allowed attempt still becomes
            # terminal. Queued envelopes are removed, including after a crash.
            await conn.execute(
                """UPDATE ingestion_queue SET status=CASE WHEN attempts>=5 THEN 'v2_failed' ELSE 'v2_pending' END,
                github_lease_id=NULL,error='Indexing worker interrupted',
                github_payload=CASE WHEN attempts>=5 THEN NULL ELSE github_payload END
                WHERE customer_id=$1 AND status='v2_processing' AND heartbeat_at < now()-interval '120 seconds'""",
                customer_id,
            )
            await conn.execute(
                """UPDATE github_backfill_jobs SET state=CASE WHEN attempts>=5 THEN 'failed' ELSE 'queued' END,
                lease_id=NULL,last_error='History worker interrupted',attempts=attempts+1,
                finished_at=CASE WHEN attempts>=5 THEN now() ELSE NULL END
                WHERE customer_id=$1 AND state='running' AND lease_id IS NOT NULL
                AND heartbeat_at < now()-interval '120 seconds'""",
                customer_id,
            )
            await conn.execute(
                """UPDATE ingestion_queue q SET status='v2_canceled',github_payload=NULL,github_lease_id=NULL,
                completed_at=now() FROM github_backfill_jobs j WHERE q.customer_id=$1 AND j.customer_id=$1
                AND q.github_job_id=j.id AND j.state IN ('failed','canceled')
                AND q.status IN ('v2_pending','v2_processing')""",
                customer_id,
            )
            rows = await conn.fetch(
                """SELECT id,installation_id,kind FROM github_backfill_jobs WHERE customer_id=$1
                AND state='running' AND enumeration_complete=TRUE AND lease_id IS NULL""",
                customer_id,
            )
            for row in rows:
                counts = await conn.fetch(
                    """SELECT status,count(*) AS n FROM ingestion_queue
                    WHERE customer_id=$1 AND github_job_id=$2 GROUP BY status""",
                    customer_id,
                    row["id"],
                )
                by_state = {r["status"]: r["n"] for r in counts}
                if by_state.get("v2_pending") or by_state.get("v2_processing"):
                    continue
                failed = by_state.get("v2_failed", 0)
                state = "failed" if failed else "completed"
                receipt = {
                    "indexed_events": by_state.get("v2_completed", 0),
                    "failed_events": failed,
                    "skipped_events": by_state.get("v2_skipped", 0),
                    "canceled_events": by_state.get("v2_canceled", 0),
                }
                await conn.execute(
                    """UPDATE github_backfill_jobs SET state=$3,counts=$4::jsonb,
                    finished_at=now(),last_error=$5 WHERE customer_id=$1 AND id=$2 AND state='running'""",
                    customer_id,
                    row["id"],
                    state,
                    json.dumps(receipt),
                    "Some events failed to index; retry to recover them" if failed else None,
                )

    async def history_step(self, customer_id: str) -> bool:
        lease = uuid4()
        async with with_tenant(customer_id) as conn:
            job = await conn.fetchrow(
                """SELECT j.* FROM github_backfill_jobs j JOIN github_installations i
                USING(customer_id,installation_id) WHERE j.customer_id=$1 AND i.active AND j.state='queued'
                AND NOT j.enumeration_complete AND (SELECT count(*) FROM ingestion_queue q
                  WHERE q.customer_id=j.customer_id AND q.github_installation_id=j.installation_id
                  AND q.status IN ('v2_pending','v2_processing')) < 200
                ORDER BY j.heartbeat_at NULLS FIRST,j.created_at FOR UPDATE OF j SKIP LOCKED LIMIT 1""",
                customer_id,
            )
            if not job:
                return False
            job = await conn.fetchrow(
                """UPDATE github_backfill_jobs SET state='running',lease_id=$3,
                heartbeat_at=now(),started_at=coalesce(started_at,now()),attempts=GREATEST(attempts,1)
                WHERE customer_id=$1 AND id=$2 RETURNING *""",
                customer_id,
                job["id"],
                lease,
            )
        try:
            async with asyncio.timeout(55):
                cursor = job["cursor"]
                if cursor is None:
                    repos = await list_repositories(
                        self.ctx.http, customer_id, job["installation_id"]
                    )
                    wanted = [r["external_id"] for r in decoded(job["scope"])]
                    available = {r["full_name"].lower(): r for r in repos if r.get("full_name")}
                    if any(name.lower() not in available for name in wanted):
                        raise ValueError(
                            "Selected repository access changed; reconnect or revise scope"
                        )
                    cursor = json.dumps(
                        {
                            "version": 2,
                            "engine": "graphql",
                            "current_repo": wanted[0],
                            "repos_remaining": wanted[1:],
                            "repo_objs": {name: available[name.lower()] for name in wanted},
                        }
                    )
                stream = self.connector.backfill(
                    customer_id, installation_token(customer_id, job["installation_id"]), cursor
                )
                new_events = 0
                complete = True
                try:
                    async for event in stream:
                        envelope = {
                            "payload": dict(event.raw_payload),
                            "_headers": dict(event.headers),
                        }
                        async with with_tenant(customer_id) as conn:
                            installation = await conn.fetchrow(
                                """SELECT * FROM github_installations
                                WHERE customer_id=$1 AND installation_id=$2 FOR UPDATE""",
                                customer_id,
                                job["installation_id"],
                            )
                            current = await conn.fetchrow(
                                "SELECT * FROM github_backfill_jobs WHERE customer_id=$1 AND id=$2 FOR UPDATE",
                                customer_id,
                                job["id"],
                            )
                            if (
                                not installation["active"]
                                or current["state"] != "running"
                                or current["lease_id"] != lease
                            ):
                                return True
                            inserted = await enqueue_event(
                                conn,
                                installation=installation,
                                envelope=envelope,
                                source_event_id=event.source_event_id,
                                job=current,
                            )
                            new_events += int(inserted)
                            await conn.execute(
                                """UPDATE github_backfill_jobs SET cursor=$3,heartbeat_at=now(),
                                processed_count=processed_count+$4 WHERE customer_id=$1 AND id=$2""",
                                customer_id,
                                job["id"],
                                event.raw_payload.get("_cursor", cursor),
                                int(inserted),
                            )
                        if new_events >= 100:
                            complete = False
                            break
                finally:
                    await stream.aclose()
                async with with_tenant(customer_id) as conn:
                    await conn.execute(
                        """UPDATE github_backfill_jobs SET state=$4,enumeration_complete=$5,
                        lease_id=NULL,heartbeat_at=now() WHERE customer_id=$1 AND id=$2 AND lease_id=$3 AND state='running'""",
                        customer_id,
                        job["id"],
                        lease,
                        "running" if complete else "queued",
                        complete,
                    )
        except TimeoutError:
            # Never leave an unresponsive provider cycling green forever. A
            # manual retry retains the checkpoint and completed queue receipts.
            async with with_tenant(customer_id) as conn:
                await conn.execute(
                    """UPDATE github_backfill_jobs SET state='failed',lease_id=NULL,finished_at=now(),
                    last_error='GitHub history timed out; retry resumes the checkpoint'
                    WHERE customer_id=$1 AND id=$2 AND lease_id=$3 AND state='running'""",
                    customer_id,
                    job["id"],
                    lease,
                )
        except Exception as exc:
            async with with_tenant(customer_id) as conn:
                await conn.execute(
                    """UPDATE github_backfill_jobs SET state='failed',finished_at=now(),
                    last_error=$4,lease_id=NULL WHERE customer_id=$1 AND id=$2 AND lease_id=$3 AND state='running'""",
                    customer_id,
                    job["id"],
                    lease,
                    f"GitHub history failed ({type(exc).__name__}); check access and retry",
                )
        return True

    async def queue_step(self, customer_id: str, *, live: bool) -> bool:
        lease = uuid4()
        async with with_tenant(customer_id) as conn:
            row = await conn.fetchrow(
                """SELECT * FROM ingestion_queue WHERE customer_id=$1
                AND status='v2_pending' AND (github_job_id IS NULL)=$2
                ORDER BY enqueued_at FOR UPDATE SKIP LOCKED LIMIT 1""",
                customer_id,
                live,
            )
            if not row:
                return False
            row = await conn.fetchrow(
                """UPDATE ingestion_queue SET status='v2_processing',attempts=attempts+1,
                github_lease_id=$3,heartbeat_at=now(),started_at=now() WHERE customer_id=$1 AND queue_id=$2 RETURNING *""",
                customer_id,
                row["queue_id"],
                lease,
            )
        try:
            async with asyncio.timeout(60):
                envelope = decoded(row["github_payload"])
                headers = dict(envelope.get("_headers", {}))
                headers["x-probe-github-protocol"] = "2"
                payload = envelope["payload"]
                parsed = self.connector.parse_webhook_event(customer_id, headers, payload)
                if parsed is None:
                    raise UnsupportedEventType("Unsupported GitHub event")
                event = WebhookEvent(
                    customer_id=customer_id,
                    source_system=SourceSystem.GITHUB,
                    source_event_id=parsed.source_event_id,
                    received_at=parsed.received_at,
                    payload_s3_key="",
                    raw_payload=payload,
                    headers=headers,
                )
                token = installation_token(customer_id, row["github_installation_id"])
                hydrated = await self.connector.fetch_supplementary(event, token)
                result = await self.connector.normalize(event, hydrated)
                for doc in result.documents:
                    doc.metadata["_github_installation_id"] = row["github_installation_id"]
                    doc.metadata["_github_operation"] = (
                        "history" if row["github_job_id"] else "live"
                    )
                if result.is_empty:
                    raise UnsupportedEventType(
                        result.skipped_reason or "No supported GitHub content"
                    )
                outcome = await self.normalizer._persist(
                    customer_id,
                    SourceSystem.GITHUB,
                    result,
                    queue_id=row["queue_id"],
                    github_lease_id=lease,
                )
                if outcome.failed_chunk_count or outcome.quarantined_doc_ids:
                    raise ValueError("Some GitHub content could not be indexed")
                await self._finish(row, lease, "v2_completed")
        except (DuplicateEventIgnored, UnsupportedEventType):
            await self._finish(row, lease, "v2_skipped")
        except Exception as exc:
            async with with_tenant(customer_id) as conn:
                terminal = row["attempts"] >= 5
                await conn.execute(
                    """UPDATE ingestion_queue SET status=$4,error=$5,github_lease_id=NULL,
                    github_payload=CASE WHEN $6 THEN NULL ELSE github_payload END,
                    completed_at=CASE WHEN $6 THEN now() ELSE NULL END
                    WHERE customer_id=$1 AND queue_id=$2 AND github_lease_id=$3 AND status='v2_processing'""",
                    customer_id,
                    row["queue_id"],
                    lease,
                    "v2_failed" if terminal else "v2_pending",
                    f"GitHub indexing failed ({type(exc).__name__})",
                    terminal,
                )
            if row["github_job_id"] is None:
                async with with_tenant(customer_id) as conn:
                    await conn.execute(
                        """UPDATE github_installations SET last_attempt_at=now(),last_error=$4
                        WHERE customer_id=$1 AND installation_id=$2 AND active AND generation=$3""",
                        customer_id,
                        row["github_installation_id"],
                        row["github_generation"],
                        f"GitHub indexing failed ({type(exc).__name__}); retrying"
                        if not terminal
                        else "GitHub indexing failed repeatedly; pause and resume to catch up",
                    )
        return True

    async def _finish(self, row, lease, state):
        async with with_tenant(row["customer_id"]) as conn:
            await conn.fetchrow(
                "SELECT 1 FROM github_installations WHERE customer_id=$1 AND installation_id=$2 FOR UPDATE",
                row["customer_id"],
                row["github_installation_id"],
            )
            changed = await conn.fetchval(
                """UPDATE ingestion_queue SET status=$4,github_payload=NULL,
                completed_at=now(),github_lease_id=NULL WHERE customer_id=$1 AND queue_id=$2
                AND github_lease_id=$3 AND status='v2_processing' RETURNING queue_id""",
                row["customer_id"],
                row["queue_id"],
                lease,
                state,
            )
            if changed and row["github_job_id"] is None:
                await conn.execute(
                    """UPDATE github_installations SET last_attempt_at=now(),last_success_at=now(),last_error=NULL
                    WHERE customer_id=$1 AND installation_id=$2 AND active AND generation=$3""",
                    row["customer_id"],
                    row["github_installation_id"],
                    row["github_generation"],
                )
