"""OAuth install + callback routes.

Generic across all connectors. Each connector that supports OAuth implements
`oauth_install_url(customer_id, redirect_uri)` and `exchange_oauth_code(code,
redirect_uri)`. This module wires those into FastAPI routes.

Flow:
    GET  /oauth/{source}/install?customer_id=...&redirect_uri=...
         → 302 redirect to the source's authorize URL
    GET  /oauth/{source}/callback?code=...&state=<customer_id>
         → exchange code → encrypt + persist token → return success page

`state` carries the customer_id (signed would be safer; Phase 0 accepts plain).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from services.ingestion.backfill_runner import enqueue_backfill
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
)
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

router = APIRouter(prefix="/oauth", tags=["oauth"])


@router.get("/{source}/install")
async def oauth_install(
    source: str,
    request: Request,
    customer_id: str = Query(...),
    redirect_uri: str = Query(...),
) -> RedirectResponse:
    source_enum = _source(source)
    try:
        get_connector_class(source_enum)
    except HandlerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    connector = build_connector(source_enum, request.app.state.ctx)
    try:
        url = connector.oauth_install_url(customer_id, redirect_uri)
    except NotSupportedByConnector as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    log.info("oauth.install_redirect", customer=customer_id, source=source)
    return RedirectResponse(url=url, status_code=302)


@router.get("/{source}/callback")
async def oauth_callback(
    source: str,
    request: Request,
    code: str | None = Query(default=None),
    state: str = Query(..., description="customer_id passed through from install"),
    error: str | None = Query(default=None),
) -> HTMLResponse:
    if error:
        raise HTTPException(status_code=400, detail=f"OAuth provider error: {error}")

    source_enum = _source(source)
    connector = build_connector(source_enum, request.app.state.ctx)

    # Redirect URI must match exactly what was sent in the install step.
    redirect_uri = str(request.url).split("?", 1)[0]
    extra_params = dict(request.query_params)

    try:
        token = await connector.exchange_oauth_code(
            code=code, redirect_uri=redirect_uri, extra_params=extra_params
        )
    except NotSupportedByConnector as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except PrbeError as exc:
        log.error("oauth.exchange_failed", source=source, error=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # Ensure customer_id is bound before persisting. Connector impls may leave
    # it empty because they don't know the tenant; we take it from `state`.
    token = token.model_copy(update={"customer_id": state})
    await save_token(token)

    # Identify the source-side workspace(s) this token grants access to so
    # incoming webhooks can resolve customer_id from the payload alone.
    workspaces_msg = ""
    try:
        refs = await connector.identify_workspaces(token)
    except Exception as exc:
        log.warning(
            "oauth.identify_workspaces_failed",
            source=source,
            customer=state,
            error=str(exc),
        )
        refs = []
    for ref in refs:
        await record_mapping(
            customer_id=state,
            source_system=source_enum,
            external_id=ref.external_id,
            external_name=ref.external_name,
            metadata=ref.metadata,
        )
    if refs:
        workspaces_msg = (
            "<p>Linked workspaces: "
            + ", ".join(
                f"<code>{r.external_name or r.external_id}</code>" for r in refs
            )
            + "</p>"
        )
    else:
        workspaces_msg = (
            "<p><small>No workspace mapping recorded. First webhook will "
            "fall back to single-tenant routing.</small></p>"
        )

    # Kick off historical backfill. The backfill worker will pick this up
    # asynchronously; webhooks arrive in parallel and are deduped by the
    # UNIQUE (customer, source, source_event_id) constraint.
    try:
        await enqueue_backfill(customer_id=state, source=source_enum)
        backfill_msg = (
            "<p><small>Historical backfill queued. Check "
            f"<code>/backfill/status?customer_id={state}</code> for progress.</small></p>"
        )
    except Exception as exc:
        log.warning(
            "oauth.backfill_enqueue_failed",
            customer=state,
            source=source,
            error=str(exc),
        )
        backfill_msg = (
            "<p><small>Backfill could not be queued (see server logs). "
            "Live webhooks will still flow.</small></p>"
        )

    log.info(
        "oauth.connected",
        customer=state,
        source=source,
        workspaces=[r.external_id for r in refs],
    )
    return HTMLResponse(
        f"<!doctype html><html><body><h2>Connected: {source}</h2>"
        f"<p>Integration active for customer <code>{state}</code>.</p>"
        f"{workspaces_msg}{backfill_msg}</body></html>"
    )


def _source(s: str) -> SourceSystem:
    try:
        return SourceSystem(s)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"unknown source '{s}'") from exc
