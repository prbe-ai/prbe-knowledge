"""Durable installation controls. No provider or object-store I/O under locks.

Lock order is installation -> job -> queue. Every accepted projection write uses
the same installation lock as pause/disconnect, plus its independent job lock for
history. Changing live scope must not invalidate unrelated historical work.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any
from uuid import UUID

from fastapi import HTTPException

from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant
from engine.shared.locks import advisory_lock_key
from engine.shared.models import IntegrationToken


class JobState(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCEL_REQUESTED = "cancel_requested"
    CANCELED = "canceled"
    COMPLETED = "completed"
    FAILED = "failed"


ACTIVE_STATES = (JobState.QUEUED, JobState.RUNNING, JobState.CANCEL_REQUESTED)
MAX_ENVELOPE_BYTES = 1024 * 1024
MAX_INSTALLATION_QUEUE = 200
MAX_HISTORY_QUEUE = 180


class GitHubQueueCapacityReached(HTTPException):
    """The installation queue is full; live delivery must be retried."""

    def __init__(self, *, history: bool) -> None:
        detail = (
            "GitHub history is waiting for indexing capacity"
            if history
            else "GitHub live indexing is at capacity; retry this delivery"
        )
        super().__init__(status_code=503, detail=detail, headers={"Retry-After": "5"})


async def adoption_lock(conn: Any, customer_id: str) -> None:
    """Serialize the legacy enqueue boundary with first v2 adoption."""
    await conn.execute(
        "SELECT pg_advisory_xact_lock($1)",
        advisory_lock_key("github-control-adoption", customer_id),
    )


async def source_purge_active(conn: Any, customer_id: str) -> bool:
    return bool(
        await conn.fetchval(
            "SELECT purge_in_progress FROM github_source_gates WHERE customer_id=$1",
            customer_id,
        )
    )


async def set_source_purge_gate(conn: Any, customer_id: str, active: bool) -> None:
    """Set the durable source-wide gate while holding ``adoption_lock``."""
    await conn.execute(
        """INSERT INTO github_source_gates(customer_id,purge_in_progress,updated_at)
        VALUES ($1,$2,now()) ON CONFLICT(customer_id) DO UPDATE
        SET purge_in_progress=EXCLUDED.purge_in_progress,updated_at=now()""",
        customer_id,
        active,
    )


async def retire_unrepresented_documents(
    conn: Any, customer_id: str, doc_ids: list[str]
) -> list[str]:
    """Retire, but do not delete, docs with no live/history installation membership.

    Callers hold every corresponding ``github-doc`` advisory lock. Inactive
    binding rows deliberately remain as deletion provenance for a later scoped
    disconnect; only the membership flags decide current visibility.
    """
    if not doc_ids:
        return []
    rows = await conn.fetch(
        """SELECT candidate.doc_id FROM unnest($2::text[]) candidate(doc_id)
        WHERE NOT EXISTS (
          SELECT 1 FROM github_document_bindings b
          WHERE b.customer_id=$1 AND b.doc_id=candidate.doc_id
            AND (b.live_present OR b.history_present))
        ORDER BY candidate.doc_id""",
        customer_id,
        doc_ids,
    )
    retired = [row["doc_id"] for row in rows]
    if not retired:
        return []
    await conn.execute(
        """UPDATE chunks SET valid_to=coalesce(valid_to,now())
        WHERE customer_id=$1 AND doc_id=ANY($2::text[]) AND valid_to IS NULL""",
        customer_id,
        retired,
    )
    await conn.execute(
        """UPDATE documents SET valid_to=coalesce(valid_to,now())
        WHERE customer_id=$1 AND doc_id=ANY($2::text[]) AND valid_to IS NULL""",
        customer_id,
        retired,
    )
    return retired


async def remove_repository_membership(
    conn: Any,
    customer_id: str,
    installation_id: str,
    repositories: set[str],
    *,
    remove_live: bool,
    remove_history: bool,
) -> list[str]:
    """Fence membership removal and retire docs no accepted lane still represents."""
    names = sorted(name.lower() for name in repositories if name)
    if not names:
        return []
    candidates = await conn.fetch(
        """SELECT DISTINCT doc_id FROM github_document_bindings
        WHERE customer_id=$1 AND installation_id=$2 AND repository=ANY($3::text[])
        ORDER BY doc_id""",
        customer_id,
        installation_id,
        names,
    )
    doc_ids = [row["doc_id"] for row in candidates]
    for doc_id in doc_ids:
        await conn.execute(
            "SELECT pg_advisory_xact_lock($1)",
            advisory_lock_key("github-doc", customer_id, doc_id),
        )
    if remove_live:
        await conn.execute(
            """UPDATE github_document_bindings SET live_present=FALSE
            WHERE customer_id=$1 AND installation_id=$2 AND repository=ANY($3::text[])""",
            customer_id,
            installation_id,
            names,
        )
    if remove_history:
        await conn.execute(
            """UPDATE github_document_bindings SET history_present=FALSE
            WHERE customer_id=$1 AND installation_id=$2 AND repository=ANY($3::text[])""",
            customer_id,
            installation_id,
            names,
        )
    return await retire_unrepresented_documents(conn, customer_id, doc_ids)


async def enqueue_legacy_webhook(customer_id, installation_id, envelope, event_id, payload_key):
    async with with_tenant(customer_id) as conn:
        await adoption_lock(conn, customer_id)
        if await source_purge_active(conn, customer_id):
            return False, True
        row = await conn.fetchrow(
            "SELECT * FROM github_installations WHERE customer_id=$1 AND installation_id=$2 FOR UPDATE",
            customer_id,
            installation_id,
        )
        if row and row["managed"]:
            return await enqueue_event(
                conn, installation=row, envelope=envelope, source_event_id=event_id
            ), True
        active = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM integration_tokens WHERE customer_id=$1 AND source_system='github' AND status='active')",
            customer_id,
        )
        if not active:
            return False, True
        inserted = await conn.fetchval(
            """INSERT INTO ingestion_queue(customer_id,source_system,source_event_id,payload_s3_key,payload_s3_keys)
            VALUES ($1,'github',$2,$3,ARRAY[$3]) ON CONFLICT(customer_id,source_system,source_event_id) DO NOTHING RETURNING queue_id""",
            customer_id,
            event_id,
            payload_key,
        )
        return bool(inserted), False


def decoded(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def normalize_scope(scope: list[dict]) -> list[dict[str, str]]:
    if len(scope) > 500:
        raise HTTPException(422, "Select at most 500 repositories")
    normalized = {}
    for item in scope:
        name = item.get("external_id", "").strip()
        parts = name.split("/")
        if len(parts) != 2 or any(not p or p in (".", "..") for p in parts):
            raise HTTPException(422, "Repository scope must use owner/repository")
        if any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-/"
            for c in name
        ):
            raise HTTPException(422, "Invalid repository name")
        normalized[name.lower()] = {"external_id": name, "label": name}
    return [normalized[k] for k in sorted(normalized)]


def scope_names(scope: Any) -> set[str]:
    return {x["external_id"].lower() for x in decoded(scope)}


def installation_token(customer_id: str, installation_id: str) -> IntegrationToken:
    # The installation binding IS the credential: bearers are minted on demand
    # for this exact tenant+installation; never load the singleton latest token.
    return IntegrationToken(
        customer_id=customer_id,
        source_system=SourceSystem.GITHUB,
        access_token="installation-minted-on-demand",
        scope=f"installation:{installation_id}",
    )


async def get_installation(conn: Any, customer_id: str, installation_id: str, *, lock=False):
    row = await conn.fetchrow(
        "SELECT * FROM github_installations WHERE customer_id=$1 AND installation_id=$2"
        + (" FOR UPDATE" if lock else ""),
        customer_id,
        installation_id,
    )
    if row is None or not row["active"]:
        raise HTTPException(404, "GitHub installation is not connected")
    return row


def public_sync(row: Any) -> dict:
    return {
        "connection_id": row["installation_id"],
        "source": "github",
        "sync_enabled": row["sync_enabled"],
        "scope": decoded(row["scope"]),
        "revision": row["revision"],
        "generation": row["generation"],
        "state": "error" if row["last_error"] else ("idle" if row["sync_enabled"] else "paused"),
        "last_attempt_at": row["last_attempt_at"],
        "last_success_at": row["last_success_at"],
        "last_error": row["last_error"],
        "workspace_id": None,
        "configuration_required": not row["managed"],
    }


def public_job(row: Any) -> dict:
    return {
        "id": str(row["id"]),
        "connection_id": row["installation_id"],
        "source": "github",
        "state": row["state"],
        "scope": decoded(row["scope"]),
        "workspace_id": None,
        "created_at": row["created_at"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "processed_count": row["processed_count"],
        "counts": decoded(row["counts"]),
        "warnings": decoded(row["warnings"]),
        "last_error": row["last_error"],
        "attempts": row["attempts"],
        "retry_count": row["retry_count"],
    }


def job_fingerprint(scope: list[dict], kind: str) -> str:
    # GitHub names are case-insensitive, and labels do not change the selection.
    names = sorted(scope_names(scope))
    return hashlib.sha256(json.dumps([kind, names]).encode()).hexdigest()


async def existing_job(
    conn: Any,
    customer_id: str,
    installation_id: str,
    scope: list[dict],
    key: str,
    *,
    kind="backfill",
):
    existing = await conn.fetchrow(
        "SELECT * FROM github_backfill_jobs WHERE customer_id=$1 AND installation_id=$2 AND idempotency_key=$3",
        customer_id,
        installation_id,
        key,
    )
    if existing:
        if existing["request_fingerprint"] != job_fingerprint(scope, kind):
            raise HTTPException(409, "Idempotency key was used for a different selection")
        return existing
    return None


async def create_job(conn: Any, row: Any, scope: list[dict], key: str, *, kind="backfill"):
    if not scope:
        raise HTTPException(422, "Select at least one repository")
    existing = await existing_job(
        conn, row["customer_id"], row["installation_id"], scope, key, kind=kind
    )
    if existing:
        return existing
    fingerprint = job_fingerprint(scope, kind)
    return await conn.fetchrow(
        """INSERT INTO github_backfill_jobs(customer_id,installation_id,scope,generation,
              idempotency_key,request_fingerprint,kind)
           VALUES ($1,$2,$3::jsonb,$4,$5,$6,$7) RETURNING *""",
        row["customer_id"],
        row["installation_id"],
        json.dumps(scope),
        row["generation"],
        key,
        fingerprint,
        kind,
    )


async def managed_installation(customer_id: str, installation_id: str) -> bool:
    async with with_tenant(customer_id) as conn:
        return bool(
            await conn.fetchval(
                "SELECT managed FROM github_installations WHERE customer_id=$1 AND installation_id=$2",
                customer_id,
                installation_id,
            )
        )


async def enqueue_event(
    conn: Any, *, installation: Any, envelope: dict, source_event_id: str, job: Any = None
) -> bool:
    """Caller holds installation lock and (for history) job lock.

    Transient bounded envelopes live on the existing queue and are erased on
    terminalization. This avoids raw R2 writes racing disconnect before enqueue.
    """
    if not installation["active"]:
        return False
    scope = job["scope"] if job else installation["scope"]
    payload = envelope.get("payload", {})
    repo = (payload.get("repository") or {}).get("full_name", "").lower()
    if repo not in scope_names(scope):
        return False
    if job:
        if job["state"] not in (JobState.QUEUED, JobState.RUNNING):
            return False
        if job["kind"] == "catchup" and (
            not installation["sync_enabled"] or job["generation"] != installation["generation"]
        ):
            return False
    elif not installation["sync_enabled"]:
        return False
    encoded = json.dumps(envelope)
    if len(encoded.encode()) > MAX_ENVELOPE_BYTES:
        raise HTTPException(413, "GitHub event exceeds the 1 MiB indexing limit")
    job_id = job["id"] if job else None
    # A retry of one history job deduplicates, but history and live keep separate
    # queue receipts and converge at stable document identities/source versions.
    event_id = f"v2:{installation['installation_id']}:{job_id or 'live'}:{source_event_id}"
    if await conn.fetchval(
        """SELECT 1 FROM ingestion_queue WHERE customer_id=$1
        AND source_system='github' AND source_event_id=$2""",
        installation["customer_id"],
        event_id,
    ):
        return False
    # The caller holds the installation row lock, so count+insert is atomic
    # across replicas. History leaves twenty slots reserved for live webhooks;
    # a full live queue returns 503 so GitHub redelivers instead of losing data.
    outstanding = int(
        await conn.fetchval(
            """SELECT count(*) FROM ingestion_queue WHERE customer_id=$1
            AND github_installation_id=$2 AND status IN ('v2_pending','v2_processing')""",
            installation["customer_id"],
            installation["installation_id"],
        )
    )
    limit = MAX_HISTORY_QUEUE if job else MAX_INSTALLATION_QUEUE
    if outstanding >= limit:
        raise GitHubQueueCapacityReached(history=job is not None)
    return bool(
        await conn.fetchval(
            """INSERT INTO ingestion_queue(customer_id,source_system,source_event_id,status,
              github_installation_id,github_generation,github_job_id,github_payload,priority)
           VALUES ($1,'github',$2,'v2_pending',$3,$4,$5,$6::jsonb,$7)
           ON CONFLICT (customer_id,source_system,source_event_id) DO NOTHING RETURNING queue_id""",
            installation["customer_id"],
            event_id,
            installation["installation_id"],
            installation["generation"],
            job_id,
            encoded,
            50 if job else 100,
        )
    )


async def enqueue_live(
    customer_id: str, installation_id: str, envelope: dict, event_id: str
) -> bool:
    async with with_tenant(customer_id) as conn:
        await adoption_lock(conn, customer_id)
        if await source_purge_active(conn, customer_id):
            return False
        row = await conn.fetchrow(
            "SELECT * FROM github_installations WHERE customer_id=$1 AND installation_id=$2 FOR UPDATE",
            customer_id,
            installation_id,
        )
        if not row or not row["active"]:
            return False
        return await enqueue_event(
            conn, installation=row, envelope=envelope, source_event_id=event_id
        )


async def revoke_repository_access(
    customer_id: str,
    installation_id: str,
    repositories: list[str],
) -> list[str]:
    """Apply a provider-authoritative repository removal under the write fence.

    Unlike a user's live-scope deselection, provider revocation removes both
    live and explicit-history membership. Jobs that include a revoked root fail
    visibly; jobs for unrelated roots and their immutable history remain valid.
    """
    removed = {name.strip().lower() for name in repositories if name and name.strip()}
    if not removed:
        return []
    async with with_tenant(customer_id) as conn:
        installation = await conn.fetchrow(
            """SELECT * FROM github_installations WHERE customer_id=$1
            AND installation_id=$2 FOR UPDATE""",
            customer_id,
            installation_id,
        )
        if installation is None or not installation["active"]:
            return []
        old_scope = decoded(installation["scope"])
        new_scope = [
            item for item in old_scope if item["external_id"].lower() not in removed
        ]
        scope_changed = len(new_scope) != len(old_scope)
        if scope_changed:
            installation = await conn.fetchrow(
                """UPDATE github_installations SET scope=$3::jsonb,revision=revision+1,
                generation=generation+1,updated_at=now(),last_error=$4
                WHERE customer_id=$1 AND installation_id=$2 RETURNING *""",
                customer_id,
                installation_id,
                json.dumps(new_scope),
                "GitHub repository access changed; review the selected live scope",
            )
            # Cancel only live envelopes whose repository was revoked (or whose
            # payload is too malformed to prove it remains authorized). Retag
            # unaffected work with the fenced generation so it can still apply.
            await conn.execute(
                """UPDATE ingestion_queue SET status='v2_canceled',github_payload=NULL,
                github_lease_id=NULL,completed_at=now() WHERE customer_id=$1
                AND github_installation_id=$2 AND github_job_id IS NULL
                AND (coalesce(lower(github_payload#>>'{payload,repository,full_name}'),'')=''
                  OR lower(github_payload#>>'{payload,repository,full_name}')=ANY($3::text[]))
                AND status IN ('v2_pending','v2_processing')""",
                customer_id,
                installation_id,
                sorted(removed),
            )
            await conn.execute(
                """UPDATE ingestion_queue SET github_generation=$3 WHERE customer_id=$1
                AND github_installation_id=$2 AND github_job_id IS NULL
                AND status IN ('v2_pending','v2_processing')""",
                customer_id,
                installation_id,
                installation["generation"],
            )

        jobs = await conn.fetch(
            """SELECT * FROM github_backfill_jobs WHERE customer_id=$1
            AND installation_id=$2 AND state IN ('queued','running','cancel_requested')
            ORDER BY id FOR UPDATE""",
            customer_id,
            installation_id,
        )
        display = ", ".join(sorted(removed)[:3])
        if len(removed) > 3:
            display += f" and {len(removed) - 3} more"
        for job in jobs:
            if not (scope_names(job["scope"]) & removed):
                continue
            error = f"GitHub access was removed for {display}; revise scope and start again"
            await conn.execute(
                """UPDATE ingestion_queue SET status='v2_canceled',github_payload=NULL,
                github_lease_id=NULL,completed_at=now() WHERE customer_id=$1
                AND github_job_id=$2 AND status IN ('v2_pending','v2_processing')""",
                customer_id,
                job["id"],
            )
            await conn.execute(
                """UPDATE github_backfill_jobs SET state='failed',finished_at=now(),
                lease_id=NULL,last_error=$3 WHERE customer_id=$1 AND id=$2""",
                customer_id,
                job["id"],
                error,
            )
            if job["lease_id"] is not None:
                await conn.execute(
                    """UPDATE github_installations SET history_lease_id=NULL,
                    history_heartbeat_at=NULL WHERE customer_id=$1 AND installation_id=$2
                    AND history_lease_id=$3""",
                    customer_id,
                    installation_id,
                    job["lease_id"],
                )
        await remove_repository_membership(
            conn,
            customer_id,
            installation_id,
            removed,
            remove_live=True,
            remove_history=True,
        )
        if scope_changed and installation["sync_enabled"] and new_scope:
            await create_job(
                conn,
                installation,
                new_scope,
                f"catchup:{installation['generation']}",
                kind="catchup",
            )
    return sorted(removed)


async def admit_projection(
    conn: Any, customer_id: str, queue_id: int, lease_id: object | None = None
) -> bool:
    """Hold the accepted write fence through Normalizer's short apply txn."""
    queued = await conn.fetchrow(
        "SELECT * FROM ingestion_queue WHERE customer_id=$1 AND queue_id=$2", customer_id, queue_id
    )
    if queued is None:
        return False
    if queued["github_installation_id"] is None:
        return True  # protocol 1, explicitly retained until it drains
    row = await conn.fetchrow(
        "SELECT * FROM github_installations WHERE customer_id=$1 AND installation_id=$2 FOR UPDATE",
        customer_id,
        queued["github_installation_id"],
    )
    if not row or not row["active"]:
        return False
    if queued["github_job_id"]:
        job = await conn.fetchrow(
            "SELECT * FROM github_backfill_jobs WHERE customer_id=$1 AND id=$2 FOR UPDATE",
            customer_id,
            queued["github_job_id"],
        )
        permitted = bool(
            job
            and job["state"] in (JobState.QUEUED, JobState.RUNNING)
            and (
                job["kind"] != "catchup"
                or (row["sync_enabled"] and job["generation"] == row["generation"])
            )
        )
    else:
        permitted = bool(row["sync_enabled"] and row["generation"] == queued["github_generation"])
    # Re-read after taking the installation/job locks: reclaim could otherwise
    # replace the queue lease between the initial lookup and accepting a write.
    current = await conn.fetchrow(
        "SELECT status,github_lease_id FROM ingestion_queue WHERE customer_id=$1 AND queue_id=$2 FOR UPDATE",
        customer_id,
        queue_id,
    )
    return bool(
        permitted
        and current
        and current["status"] == "v2_processing"
        and current["github_lease_id"] == lease_id
    )


async def cancel_job(customer_id: str, installation_id: str, job_id: UUID) -> dict:
    async with with_tenant(customer_id) as conn:
        await get_installation(conn, customer_id, installation_id, lock=True)
        row = await conn.fetchrow(
            "SELECT * FROM github_backfill_jobs WHERE customer_id=$1 AND installation_id=$2 AND id=$3 FOR UPDATE",
            customer_id,
            installation_id,
            job_id,
        )
        if row is None:
            raise HTTPException(404, "Backfill not found")
        if row["state"] in ACTIVE_STATES:
            # The lock waits for every already accepted projection apply. Future
            # hydration may finish, but can no longer apply after this commits.
            await conn.execute(
                """UPDATE ingestion_queue SET status='v2_canceled',github_payload=NULL,
                   completed_at=now(),github_lease_id=NULL
                   WHERE customer_id=$1 AND github_job_id=$2 AND status IN ('v2_pending','v2_processing')""",
                customer_id,
                job_id,
            )
            job_lease = row["lease_id"]
            row = await conn.fetchrow(
                """UPDATE github_backfill_jobs SET state='canceled',finished_at=now(),lease_id=NULL
                   WHERE customer_id=$1 AND id=$2 RETURNING *""",
                customer_id,
                job_id,
            )
            if job_lease is not None:
                await conn.execute(
                    """UPDATE github_installations SET history_lease_id=NULL,
                    history_heartbeat_at=NULL WHERE customer_id=$1 AND installation_id=$2
                    AND history_lease_id=$3""",
                    customer_id,
                    installation_id,
                    job_lease,
                )
        return public_job(row)
