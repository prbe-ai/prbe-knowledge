"""Internal protocol-v2 GitHub endpoints; the gateway enforces member roles."""

from __future__ import annotations

import json
from uuid import UUID

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from engine.shared.backend_client import fetch_github_installation_token
from engine.shared.db import raw_conn, with_tenant
from kb.admin_routes import verify_internal_knowledge_key
from kb.backfill_routes import _require_customer
from kb.github_control import (
    adoption_lock,
    cancel_job,
    create_job,
    decoded,
    get_installation,
    normalize_scope,
    public_job,
    public_sync,
    scope_names,
)

router = APIRouter(
    prefix="/api/github",
    tags=["github-control"],
    dependencies=[Depends(verify_internal_knowledge_key)],
)


class ScopeItem(BaseModel):
    external_id: str = Field(min_length=3, max_length=220)
    label: str | None = Field(default=None, max_length=220)


class SyncPatch(BaseModel):
    scope: list[ScopeItem] | None = Field(default=None, max_length=500)
    sync_enabled: bool | None = None
    expected_revision: int | None = Field(default=None, ge=1)


class BackfillStart(BaseModel):
    scope: list[ScopeItem] = Field(min_length=1, max_length=500)
    idempotency_key: str = Field(min_length=1, max_length=200)


async def list_repositories(
    http: httpx.AsyncClient, customer_id: str, installation_id: str
) -> list[dict]:
    """Bounded enumeration, never interpret a partial/error page as empty scope."""
    bearer, _ = await fetch_github_installation_token(
        http, customer_id=customer_id, installation_id=installation_id
    )
    repos = []
    for page in range(1, 52):
        response = await http.get(
            "https://api.github.com/installation/repositories",
            params={"page": page, "per_page": 100},
            headers={"Authorization": f"Bearer {bearer}", "Accept": "application/vnd.github+json"},
            timeout=20,
        )
        if response.status_code != 200:
            raise HTTPException(
                502, "GitHub repository enumeration failed; existing scope was kept"
            )
        body = response.json()
        batch = body.get("repositories")
        if not isinstance(batch, list):
            raise HTTPException(502, "GitHub returned an invalid repository list")
        repos.extend(batch)
        if len(batch) < 100:
            return repos
    raise HTTPException(422, "This installation exceeds the 5,000-repository enumeration limit")


async def validate_scope(
    request: Request, customer_id: str, installation_id: str, scope: list[dict]
) -> None:
    repos = await list_repositories(request.app.state.ctx.http, customer_id, installation_id)
    allowed = {r["full_name"].lower() for r in repos if r.get("full_name")}
    if not scope_names(scope) <= allowed:
        raise HTTPException(
            422, "Some selected repositories are not accessible to this installation"
        )


@router.get("/capabilities")
async def capabilities() -> dict:
    async with raw_conn() as conn:
        ready = bool(
            await conn.fetchval("""SELECT EXISTS(SELECT 1 FROM github_worker_capabilities
            WHERE protocol_version=2 AND heartbeat_at > now()-interval '90 seconds')""")
        )
    return {
        "protocol_version": 2,
        "installation_controls": True,
        "durable_backfills": True,
        "worker_ready": ready,
    }


@router.get("/installations/{installation_id}/sync")
async def get_sync(installation_id: str, customer_id: str = Depends(_require_customer)) -> dict:
    async with with_tenant(customer_id) as conn:
        row = await get_installation(conn, customer_id, installation_id)
        result = public_sync(row)
        busy = await conn.fetchval(
            """SELECT EXISTS(SELECT 1 FROM ingestion_queue
            WHERE customer_id=$1 AND github_installation_id=$2 AND github_job_id IS NULL
            AND status IN ('v2_pending','v2_processing')) OR EXISTS(SELECT 1 FROM github_backfill_jobs
            WHERE customer_id=$1 AND installation_id=$2 AND kind='catchup' AND state IN ('queued','running'))""",
            customer_id,
            installation_id,
        )
        catchup = await conn.fetchrow(
            """SELECT state,started_at,finished_at,last_error
            FROM github_backfill_jobs WHERE customer_id=$1 AND installation_id=$2 AND kind='catchup'
            AND generation=$3 ORDER BY created_at DESC LIMIT 1""",
            customer_id,
            installation_id,
            row["generation"],
        )
        if catchup:
            result["last_attempt_at"] = row["last_attempt_at"] or catchup["started_at"]
            if catchup["state"] == "completed":
                result["last_success_at"] = row["last_success_at"] or catchup["finished_at"]
            elif catchup["state"] == "failed":
                result["last_error"] = catchup["last_error"]
                result["state"] = "error"
        if busy and row["sync_enabled"] and not result["last_error"]:
            result["state"] = "running"
        return result


@router.get("/installations/{installation_id}/scope-options")
async def scope_options(
    installation_id: str, request: Request, customer_id: str = Depends(_require_customer)
) -> dict:
    async with with_tenant(customer_id) as conn:
        await get_installation(conn, customer_id, installation_id)
    repos = await list_repositories(request.app.state.ctx.http, customer_id, installation_id)
    return {
        "options": [
            {"external_id": r["full_name"], "label": r["full_name"]}
            for r in repos
            if r.get("full_name")
        ],
        "complete": True,
    }


@router.patch("/installations/{installation_id}/sync")
async def patch_sync(
    installation_id: str,
    body: SyncPatch,
    request: Request,
    customer_id: str = Depends(_require_customer),
) -> dict:
    async with with_tenant(customer_id) as conn:
        initial = await get_installation(conn, customer_id, installation_id)
    scope = (
        normalize_scope([x.model_dump() for x in body.scope])
        if body.scope is not None
        else decoded(initial["scope"])
    )
    enabled = initial["sync_enabled"] if body.sync_enabled is None else body.sync_enabled
    if enabled and not scope:
        raise HTTPException(422, "Select at least one repository before enabling sync")
    if scope and (enabled or body.scope is not None):
        await validate_scope(request, customer_id, installation_id, scope)
    async with with_tenant(customer_id) as conn:
        await adoption_lock(conn, customer_id)
        current = await get_installation(conn, customer_id, installation_id, lock=True)
        if not current["managed"] and await conn.fetchval(
            """SELECT
            EXISTS(SELECT 1 FROM ingestion_queue WHERE customer_id=$1 AND source_system='github'
                AND github_installation_id IS NULL AND status IN ('pending','processing'))
            OR EXISTS(SELECT 1 FROM backfill_state WHERE customer_id=$1 AND source_system='github'
                AND status IN ('pending','running'))""",
            customer_id,
        ):
            raise HTTPException(
                409, "Existing GitHub work is still finishing; retry setup when it has drained"
            )
        expected = (
            body.expected_revision if body.expected_revision is not None else initial["revision"]
        )
        if current["revision"] != expected:
            raise HTTPException(409, "Settings changed; refresh and try again")
        row = await conn.fetchrow(
            """UPDATE github_installations SET managed=TRUE,
            sync_enabled=$3,scope=$4::jsonb,revision=revision+1,generation=generation+1,
            updated_at=now(),last_error=NULL WHERE customer_id=$1 AND installation_id=$2 RETURNING *""",
            customer_id,
            installation_id,
            enabled,
            json.dumps(scope),
        )
        # Only live work is invalidated. Explicit history jobs retain their
        # immutable scope, cursor and write permission while live is paused.
        await conn.execute(
            """UPDATE github_backfill_jobs SET state='canceled',finished_at=now(),lease_id=NULL
            WHERE customer_id=$1 AND installation_id=$2 AND kind='catchup'
            AND state IN ('queued','running','cancel_requested')""",
            customer_id,
            installation_id,
        )
        await conn.execute(
            """UPDATE ingestion_queue SET status='v2_canceled',github_payload=NULL,
            github_lease_id=NULL,completed_at=now() WHERE customer_id=$1 AND github_installation_id=$2
            AND (github_job_id IS NULL OR github_job_id IN (SELECT id FROM github_backfill_jobs
              WHERE customer_id=$1 AND installation_id=$2 AND kind='catchup'))
            AND status IN ('v2_pending','v2_processing')""",
            customer_id,
            installation_id,
        )
        if enabled:
            await create_job(conn, row, scope, f"catchup:{row['generation']}", kind="catchup")
        return public_sync(row)


@router.post("/installations/{installation_id}/backfills")
async def start_backfill(
    installation_id: str,
    body: BackfillStart,
    request: Request,
    customer_id: str = Depends(_require_customer),
) -> dict:
    scope = normalize_scope([x.model_dump() for x in body.scope])
    async with with_tenant(customer_id) as conn:
        await get_installation(conn, customer_id, installation_id)
    await validate_scope(request, customer_id, installation_id, scope)
    async with with_tenant(customer_id) as conn:
        row = await get_installation(conn, customer_id, installation_id, lock=True)
        if not row["managed"]:
            raise HTTPException(
                409, "Save repository scope and live settings before starting a backfill"
            )
        await conn.execute(
            "UPDATE github_installations SET managed=TRUE WHERE customer_id=$1 AND installation_id=$2",
            customer_id,
            installation_id,
        )
        return public_job(await create_job(conn, row, scope, body.idempotency_key))


@router.get("/installations/{installation_id}/backfills")
async def list_backfills(
    installation_id: str,
    customer_id: str = Depends(_require_customer),
    limit: int = Query(default=30, ge=1, le=100),
    cursor: UUID | None = None,
) -> dict:
    async with with_tenant(customer_id) as conn:
        await get_installation(conn, customer_id, installation_id)
        if cursor and not await conn.fetchval(
            "SELECT 1 FROM github_backfill_jobs WHERE customer_id=$1 AND installation_id=$2 AND id=$3",
            customer_id,
            installation_id,
            cursor,
        ):
            raise HTTPException(422, "Invalid history cursor")
        rows = await conn.fetch(
            """SELECT * FROM github_backfill_jobs WHERE customer_id=$1
            AND installation_id=$2 AND kind='backfill' AND ($3::uuid IS NULL OR (created_at,id) <
              (SELECT created_at,id FROM github_backfill_jobs WHERE customer_id=$1 AND id=$3))
            ORDER BY created_at DESC,id DESC LIMIT $4""",
            customer_id,
            installation_id,
            cursor,
            limit + 1,
        )
    return {
        "jobs": [public_job(r) for r in rows[:limit]],
        "next_cursor": str(rows[limit - 1]["id"]) if len(rows) > limit else None,
    }


@router.get("/installations/{installation_id}/backfills/{job_id}")
async def get_backfill(
    installation_id: str, job_id: UUID, customer_id: str = Depends(_require_customer)
) -> dict:
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            "SELECT * FROM github_backfill_jobs WHERE customer_id=$1 AND installation_id=$2 AND id=$3",
            customer_id,
            installation_id,
            job_id,
        )
        if row is None:
            raise HTTPException(404, "Backfill not found")
        return public_job(row)


@router.post("/installations/{installation_id}/backfills/{job_id}/cancel")
async def cancel_backfill(
    installation_id: str, job_id: UUID, customer_id: str = Depends(_require_customer)
) -> dict:
    return await cancel_job(customer_id, installation_id, job_id)


@router.post("/installations/{installation_id}/backfills/{job_id}/retry")
async def retry_backfill(
    installation_id: str, job_id: UUID, customer_id: str = Depends(_require_customer)
) -> dict:
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
        if row["state"] in ("failed", "canceled"):
            row = await conn.fetchrow(
                """UPDATE github_backfill_jobs SET state='queued',finished_at=NULL,attempts=attempts+1,
                last_error=NULL,lease_id=NULL,enumeration_complete=FALSE,cursor=NULL
                WHERE customer_id=$1 AND id=$2 RETURNING *""",
                customer_id,
                job_id,
            )
            # Keep successful receipts. A failed/canceled envelope was erased,
            # so replay traversal from the scope root: its last fetch checkpoint
            # may be newer than those unindexed items. Completed receipts dedupe.
            await conn.execute(
                "DELETE FROM ingestion_queue WHERE customer_id=$1 AND github_job_id=$2 AND status IN ('v2_failed','v2_canceled')",
                customer_id,
                job_id,
            )
        return public_job(row)


@router.get("/installations/{installation_id}/purge-preview")
async def purge_preview(
    installation_id: str, customer_id: str = Depends(_require_customer)
) -> dict:
    from kb.github_control_purge import preview

    return await preview(customer_id, installation_id)


@router.post("/installations/{installation_id}/purge")
async def purge_installation(
    installation_id: str, customer_id: str = Depends(_require_customer)
) -> dict:
    from kb.github_control_purge import purge

    return await purge(customer_id, installation_id)
