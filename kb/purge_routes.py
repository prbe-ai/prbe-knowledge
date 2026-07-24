"""Per-source purge endpoints — internal API.

    POST /purge            {"source": "github"}  -> 202 {"purge_id": ...}
    GET  /purge/status?purge_id=<id>             -> the recorded outcome
    GET  /purge/status?source=<source>           -> latest run for that source

Gated by X-Internal-Knowledge-Key like every other internal route; the tenant
comes from the X-Prbe-Customer header the caller sets, never from the body.

Why trigger + poll rather than one blocking call: the cascade spans every
source-tagged table plus an R2 sweep plus a verification pass, so its duration
scales with the corpus. A single synchronous request would put a timeout cliff
between the caller and the answer, and a caller that times out cannot tell a
finished purge from a failed one -- which matters because the caller drops its
own record of the integration only on `verified=true`. This mirrors the
existing backfill surface (POST to start, GET /backfill/status to follow).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from engine.ingest.purge import (
    cascade_for,
    create_purge_run,
    finish_purge_run,
    get_purge_run,
    latest_purge_run,
    purge_source,
)
from engine.shared.constants import SourceSystem
from kb.admin_routes import verify_internal_knowledge_key

log = structlog.get_logger(__name__)

router = APIRouter(prefix="/purge", tags=["purge"])

# Strong references to in-flight purge tasks. asyncio only holds a weak
# reference to a bare create_task(), so without this the garbage collector can
# cancel a running purge mid-cascade — leaving a half-deleted source and a
# `running` row that never resolves.
_INFLIGHT: set[asyncio.Task[None]] = set()

# Sources a caller may purge. Derived sources are cascaded INTO by their
# parent and have no integration of their own, so purging one directly would
# be a half-disconnect that leaves the parent's gate open.
_PURGEABLE: frozenset[SourceSystem] = frozenset(
    {
        SourceSystem.SLACK,
        SourceSystem.LINEAR,
        SourceSystem.GITHUB,
        SourceSystem.NOTION,
        SourceSystem.SENTRY,
        SourceSystem.GRANOLA,
    }
)


def _require_customer(
    x_prbe_customer: str | None = Header(default=None, alias="X-Prbe-Customer"),
) -> str:
    if not x_prbe_customer:
        raise HTTPException(status_code=400, detail="missing X-Prbe-Customer")
    return x_prbe_customer


class PurgeRequest(BaseModel):
    source: str


def _resolve_source(raw: str) -> SourceSystem:
    try:
        source = SourceSystem(raw)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail=f"unknown source '{raw}'"
        ) from exc
    if source not in _PURGEABLE:
        raise HTTPException(
            status_code=400,
            detail=(
                f"source '{raw}' is not independently purgeable "
                "(derived sources are cascaded by their parent)"
            ),
        )
    return source


def _decode_result(raw: Any) -> dict[str, Any]:
    """JSONB comes back as a string.

    The pool's connection setup pins search_path and nothing else — no jsonb
    codec is registered — so asyncpg hands back the raw JSON text. Treating it
    as a dict crashes the status endpoint on exactly the requests that matter
    (a purge that finished).
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw:
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


async def _run_and_record(
    customer_id: str, source: SourceSystem, purge_id: str
) -> None:
    """Background body. Never raises: the outcome is the purge_runs row."""
    try:
        result = await purge_source(customer_id, source, purge_id)
        await finish_purge_run(customer_id, purge_id, result)
    except asyncio.CancelledError:
        # Shutdown mid-purge. CancelledError is BaseException, so without this
        # the run would stay 'running' forever and a caller polling status
        # could wait on a purge nothing is executing. Mark it failed —
        # re-triggering is safe because the cascade is idempotent.
        log.warning(
            "purge.cancelled",
            customer=customer_id,
            source=source.value,
            purge_id=purge_id,
        )
        with contextlib.suppress(Exception):
            await finish_purge_run(
                customer_id, purge_id, None, error="cancelled (worker shutdown)"
            )
        raise
    except Exception as exc:
        log.exception(
            "purge.failed",
            customer=customer_id,
            source=source.value,
            purge_id=purge_id,
        )
        await finish_purge_run(
            customer_id, purge_id, None, error=f"{type(exc).__name__}: {exc}"
        )


@router.post("", status_code=202, dependencies=[Depends(verify_internal_knowledge_key)])
async def start_purge(
    body: PurgeRequest,
    customer_id: str = Depends(_require_customer),
) -> dict[str, Any]:
    """Start a purge. Returns immediately with the id to poll.

    Safe to call repeatedly: the cascade is idempotent, so a caller retrying
    after a lost response re-runs a delete that finds nothing and verifies
    clean.
    """
    source = _resolve_source(body.source)
    purge_id = await create_purge_run(customer_id, source)
    log.info(
        "purge.started",
        customer=customer_id,
        source=source.value,
        purge_id=purge_id,
        cascade=[s.value for s in cascade_for(source)],
    )
    # Detached: the request returns now and the caller follows /purge/status.
    # The task holds no request-scoped state, and its outcome is persisted, so
    # a pod restart mid-purge is recoverable by re-triggering.
    task = asyncio.create_task(_run_and_record(customer_id, source, purge_id))
    _INFLIGHT.add(task)
    task.add_done_callback(_INFLIGHT.discard)
    return {
        "purge_id": purge_id,
        "status": "running",
        "source": source.value,
        "cascade": [s.value for s in cascade_for(source)],
    }


@router.get("/status", dependencies=[Depends(verify_internal_knowledge_key)])
async def purge_status(
    customer_id: str = Depends(_require_customer),
    purge_id: str | None = Query(default=None),
    source: str | None = Query(default=None),
) -> dict[str, Any]:
    """Outcome of a purge, by id or the latest for a source."""
    if purge_id:
        row = await get_purge_run(customer_id, purge_id)
        if row is None:
            raise HTTPException(status_code=404, detail="unknown purge_id")
    elif source:
        row = await latest_purge_run(customer_id, _resolve_source(source))
        if row is None:
            return {"status": "none", "source": source}
    else:
        raise HTTPException(
            status_code=400, detail="provide purge_id or source"
        )

    result = _decode_result(row.get("result"))
    return {
        "purge_id": str(row["purge_id"]),
        "source": row["source_system"],
        "status": row["status"],
        "verified": bool(result.get("verified")),
        "result": result,
        "error": row.get("error"),
        "started_at": row["started_at"].isoformat() if row.get("started_at") else None,
        "finished_at": (
            row["finished_at"].isoformat() if row.get("finished_at") else None
        ),
    }
