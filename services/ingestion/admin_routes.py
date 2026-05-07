"""Internal API for prbe-backend BFF.

Gated by X-Internal-Knowledge-Key. The BFF (api.prbe.ai) handles all
customer-facing admin work directly against the Neon DB. The only thing
that still requires running here is the per-source OAuth
code-for-token exchange, because that calls connector classes that
live in this service.

Route layout:
    POST   /api/oauth/{source}/exchange   — token exchange + persist
    POST   /api/queue/replay-dlq          — re-enqueue DLQ rows
    POST   /api/code-graph/reindex        — re-enqueue full backfill per repo
"""

from __future__ import annotations

import hmac
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from services.ingestion.backfill_runner import enqueue_backfill
from services.ingestion.code_graph.reindex import (
    ReindexNotConnected,
    ReindexResult,
    reindex_customer,
)
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
)
from shared.config import get_settings
from shared.constants import QueueStatus, SourceSystem
from shared.customer_mapping import record_mapping, resolve_customer
from shared.db import raw_conn
from shared.exceptions import (
    GitHubAuthError,
    HandlerNotFound,
    NotSupportedByConnector,
    PrbeError,
    SourceAlreadyConnectedError,
)
from shared.logging import get_logger
from shared.tokens import save_token

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["internal-api"])


# ---------------------------------------------------------------------------
# Auth — shared X-Internal-Knowledge-Key gate
# ---------------------------------------------------------------------------


async def verify_internal_knowledge_key(
    x_internal_knowledge_key: str | None = Header(
        default=None, alias="X-Internal-Knowledge-Key"
    ),
) -> None:
    expected_secret = get_settings().internal_knowledge_api_key
    if expected_secret is None or not expected_secret.get_secret_value():
        raise HTTPException(
            status_code=503,
            detail="disabled — set INTERNAL_KNOWLEDGE_API_KEY",
        )
    if x_internal_knowledge_key is None:
        raise HTTPException(
            status_code=401, detail="missing X-Internal-Knowledge-Key"
        )
    if not hmac.compare_digest(
        x_internal_knowledge_key, expected_secret.get_secret_value()
    ):
        raise HTTPException(
            status_code=401, detail="invalid X-Internal-Knowledge-Key"
        )


# ---------------------------------------------------------------------------
# OAuth code-for-token exchange (gateway pattern).
#
# prbe-backend's gateway handles the public-facing OAuth callback at
# /oauth/{source}/callback. After verifying the HMAC-signed state and
# resolving customer_id, it POSTs here with {customer_id, code,
# redirect_uri, extra_params}. We run the per-source token exchange,
# persist the encrypted token, identify workspaces, and queue backfill.
# ---------------------------------------------------------------------------


class OAuthExchangeRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    # Empty string allowed — GitHub Apps installs authenticate via
    # installation_id from extra_params, no user-grant code required.
    # Slack/Linear/Notion connectors reject an empty code downstream when
    # they try to use it against the OAuth provider.
    code: str = Field(default="", max_length=4096)
    redirect_uri: str = Field(min_length=1)
    extra_params: dict[str, str] = Field(default_factory=dict)


class OAuthExchangeResponse(BaseModel):
    customer_id: str
    source: str
    workspaces: list[dict[str, Any]] = Field(default_factory=list)
    backfill_queued: bool = False


def _source_or_404(source: str) -> SourceSystem:
    try:
        return SourceSystem(source)
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail=f"unknown source: {source}"
        ) from exc


@router.post(
    "/oauth/{source}/exchange",
    response_model=OAuthExchangeResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def oauth_exchange(
    source: str,
    body: OAuthExchangeRequest,
    request: Request,
) -> OAuthExchangeResponse:
    source_enum = _source_or_404(source)
    try:
        get_connector_class(source_enum)
    except HandlerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    connector = build_connector(source_enum, request.app.state.ctx)

    try:
        token = await connector.exchange_oauth_code(
            code=body.code,
            redirect_uri=body.redirect_uri,
            extra_params=body.extra_params,
        )
    except NotSupportedByConnector as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PrbeError as exc:
        log.error(
            "oauth.exchange_failed",
            source=source,
            customer=body.customer_id,
            error=str(exc),
        )
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    token = token.model_copy(update={"customer_id": body.customer_id})

    # Identify workspaces BEFORE persisting the token so we can refuse the
    # install if any workspace is already connected to a different customer.
    # `identify_workspaces` only needs the in-memory token; nothing in the DB
    # yet. If we saved the token first and then discovered a conflict, we'd
    # leave a half-installed integration_tokens row behind.
    try:
        refs = await connector.identify_workspaces(token)
    except Exception as exc:
        log.warning(
            "oauth.identify_workspaces_failed",
            source=source,
            customer=body.customer_id,
            error=str(exc),
        )
        refs = []

    for ref in refs:
        existing = await resolve_customer(source_enum, ref.external_id)
        if existing is not None and existing != body.customer_id:
            log.warning(
                "oauth.workspace_already_connected",
                source=source,
                external_id=ref.external_id,
                external_name=ref.external_name,
                existing_customer=existing,
                attempted_customer=body.customer_id,
            )
            # Friendly message for the dashboard. Intentionally does not
            # name the existing customer — that would leak tenant identity
            # to an unrelated user attempting an install.
            label = ref.external_name or ref.external_id
            raise HTTPException(
                status_code=409,
                detail=(
                    f"This {source} workspace ({label}) is already connected "
                    f"to another Probe account. Ask an admin of that account "
                    f"to disconnect it before connecting it here."
                ),
            )

    await save_token(token)

    workspaces: list[dict[str, Any]] = []
    for ref in refs:
        try:
            await record_mapping(
                customer_id=body.customer_id,
                source_system=source_enum,
                external_id=ref.external_id,
                external_name=ref.external_name,
                metadata=ref.metadata,
            )
        except SourceAlreadyConnectedError as exc:
            # Defense in depth: should be unreachable because of the
            # pre-check above. If it fires, a parallel install raced us
            # between the pre-check and now.
            log.warning(
                "oauth.workspace_already_connected_race",
                source=source,
                external_id=exc.external_id,
                external_name=exc.external_name,
                existing_customer=exc.existing_customer_id,
                attempted_customer=exc.attempted_customer_id,
            )
            label = exc.external_name or exc.external_id
            raise HTTPException(
                status_code=409,
                detail=(
                    f"This {source} workspace ({label}) was just connected "
                    f"to another Probe account. Try again or contact support."
                ),
            ) from exc
        workspaces.append(
            {
                "external_id": ref.external_id,
                "external_name": ref.external_name,
            }
        )

    backfill_queued = False
    try:
        await enqueue_backfill(customer_id=body.customer_id, source=source_enum)
        backfill_queued = True
    except Exception as exc:
        log.warning(
            "oauth.backfill_enqueue_failed",
            customer=body.customer_id,
            source=source,
            error=str(exc),
        )

    log.info(
        "oauth.exchanged",
        customer=body.customer_id,
        source=source,
        workspaces=[r.external_id for r in refs],
    )
    return OAuthExchangeResponse(
        customer_id=body.customer_id,
        source=source,
        workspaces=workspaces,
        backfill_queued=backfill_queued,
    )


# ---------------------------------------------------------------------------
# Queue admin: re-enqueue DLQ rows.
#
# Source-platform redeliveries cannot rescue a DLQ'd row because the
# UNIQUE (customer_id, source_system, source_event_id) constraint on
# ingestion_queue makes the redelivery a silent "duplicate" 200. So once
# something lands in DLQ, the only way to retry it is to flip status back
# to 'pending'. This route is the supported path; manual SQL is the
# unsupported one.
# ---------------------------------------------------------------------------


class ReplayDLQRequest(BaseModel):
    customer_id: str | None = Field(
        default=None,
        description="If set, only replay this customer's DLQ rows.",
    )
    source_system: str | None = Field(
        default=None,
        description="If set, only replay rows for this source.",
    )
    since: datetime | None = Field(
        default=None,
        description="If set, only replay rows whose enqueued_at >= this timestamp.",
    )
    queue_ids: list[int] | None = Field(
        default=None,
        description="If set, only replay these specific queue_ids (overrides other filters).",
    )
    limit: int = Field(
        default=1000,
        ge=1,
        le=100_000,
        description="Maximum rows to replay in this call.",
    )


class ReplayDLQResponse(BaseModel):
    requeued: int
    queue_ids: list[int]


@router.post(
    "/queue/replay-dlq",
    response_model=ReplayDLQResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def replay_dlq(body: ReplayDLQRequest) -> ReplayDLQResponse:
    if body.source_system is not None:
        try:
            SourceSystem(body.source_system)
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail=f"unknown source: {body.source_system}"
            ) from exc

    sql = """
        UPDATE ingestion_queue
        SET status       = $1,
            attempts     = 0,
            error        = NULL,
            started_at   = NULL,
            heartbeat_at = NULL,
            completed_at = NULL
        WHERE queue_id IN (
            SELECT queue_id FROM ingestion_queue
            WHERE status = $2
              AND ($3::text IS NULL OR customer_id   = $3)
              AND ($4::text IS NULL OR source_system = $4)
              AND ($5::timestamptz IS NULL OR enqueued_at >= $5)
              AND ($6::bigint[] IS NULL OR queue_id = ANY($6))
            ORDER BY enqueued_at
            LIMIT $7
            FOR UPDATE SKIP LOCKED
        )
        RETURNING queue_id
    """
    async with raw_conn() as conn:
        rows = await conn.fetch(
            sql,
            QueueStatus.PENDING.value,
            QueueStatus.DLQ.value,
            body.customer_id,
            body.source_system,
            body.since,
            body.queue_ids,
            body.limit,
        )

    queue_ids = [r["queue_id"] for r in rows]
    log.info(
        "queue.replay_dlq",
        requeued=len(queue_ids),
        customer=body.customer_id,
        source=body.source_system,
    )
    return ReplayDLQResponse(requeued=len(queue_ids), queue_ids=queue_ids)


# ---------------------------------------------------------------------------
# Code-graph manual reindex.
#
# Re-enqueues an initial-backfill event per repo the customer's github
# installation can see at HEAD. Idempotent on the same SHAs (the bridge's
# UNIQUE on source_event_id dedupes), so re-clicking the dashboard's
# "Reindex" cog without new commits is a no-op.
# ---------------------------------------------------------------------------


class ReindexCodeGraphRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)


class ReindexCodeGraphResponse(BaseModel):
    enqueued: int
    skipped: int
    repos: list[str]


@router.post(
    "/code-graph/reindex",
    response_model=ReindexCodeGraphResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def code_graph_reindex(
    body: ReindexCodeGraphRequest,
) -> ReindexCodeGraphResponse:
    try:
        result: ReindexResult = await reindex_customer(body.customer_id)
    except ReindexNotConnected as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except GitHubAuthError as exc:
        # Backend installation-token endpoint is unhealthy or the install
        # was uninstalled on the GitHub side. 502 — upstream auth dependency.
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ReindexCodeGraphResponse(
        enqueued=result.enqueued,
        skipped=result.skipped,
        repos=result.repos,
    )
