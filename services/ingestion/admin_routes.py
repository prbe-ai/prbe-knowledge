"""Internal API for prbe-backend BFF.

Gated by X-Admin-Key. The BFF (api.prbe.ai) handles all customer-facing
admin work directly against the Neon DB. The only thing that still
requires running here is the per-source OAuth code-for-token exchange,
because that calls connector classes that live in this service.

Route layout:
    POST   /api/oauth/{source}/exchange — token exchange + persist
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from services.ingestion.backfill_runner import enqueue_backfill
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
)
from shared.config import get_settings
from shared.constants import SourceSystem
from shared.customer_mapping import record_mapping
from shared.exceptions import (
    HandlerNotFound,
    NotSupportedByConnector,
    PrbeError,
)
from shared.logging import get_logger
from shared.tokens import save_token

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["internal-api"])


# ---------------------------------------------------------------------------
# Auth — shared X-Admin-Key gate
# ---------------------------------------------------------------------------


async def verify_admin_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    expected_secret = get_settings().admin_api_key
    if expected_secret is None or not expected_secret.get_secret_value():
        raise HTTPException(
            status_code=503, detail="admin disabled — set ADMIN_API_KEY"
        )
    if x_admin_key is None:
        raise HTTPException(status_code=401, detail="missing X-Admin-Key")
    if not hmac.compare_digest(x_admin_key, expected_secret.get_secret_value()):
        raise HTTPException(status_code=401, detail="invalid X-Admin-Key")


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
    code: str = Field(min_length=1, max_length=4096)
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
    dependencies=[Depends(verify_admin_key)],
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
    await save_token(token)

    workspaces: list[dict[str, Any]] = []
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
        await record_mapping(
            customer_id=body.customer_id,
            source_system=source_enum,
            external_id=ref.external_id,
            external_name=ref.external_name,
            metadata=ref.metadata,
        )
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
