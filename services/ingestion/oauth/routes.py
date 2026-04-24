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

from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from services.ingestion.backfill_runner import enqueue_backfill
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
)
from shared.config import get_settings
from shared.constants import SourceSystem
from shared.customer_mapping import record_mapping, resolve_customer
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
    state: str | None = Query(
        default=None, description="customer_id passed through from install"
    ),
    error: str | None = Query(default=None),
) -> Response:
    dashboard_base = (get_settings().dashboard_base_url or "").rstrip("/")
    source_enum = _source(source)
    extra_params = dict(request.query_params)

    if error:
        if dashboard_base:
            return _landed_redirect(
                dashboard_base,
                source=source,
                customer_id=state or "",
                ok=False,
                error=error,
            )
        raise HTTPException(status_code=400, detail=f"OAuth provider error: {error}")

    # `state` is absent for GitHub marketplace installs and for post-install
    # "Redirect on update" fires (repo added/removed). Fall back to existing
    # (source, external_id) → customer_id mapping so those flows don't 422.
    customer_id = state
    resolved_from_mapping = False
    if customer_id is None:
        customer_id = await _resolve_customer_from_callback(source_enum, extra_params)
        resolved_from_mapping = customer_id is not None

    # Post-install "Redirect on update" fires on every repo add/remove for an
    # already-connected installation. Recognize it by: state absent (GitHub
    # drops it on update), mapping already on file, no `code`. Skip the full
    # save path — installation id is unchanged, tokens are minted on demand.
    # When `state` IS present we're in a first-time connect flow even if
    # GitHub happens to stamp setup_action=update (e.g. the app was installed
    # at the account level before this customer linked it).
    setup_action = extra_params.get("setup_action")
    if resolved_from_mapping and setup_action == "update" and not code:
        log.info("oauth.post_install_update", customer=customer_id, source=source)
        if dashboard_base:
            return _landed_redirect(
                dashboard_base, source=source, customer_id=customer_id, ok=True
            )
        return HTMLResponse(
            f"<!doctype html><html><body><h2>Updated: {source}</h2>"
            f"<p>Installation updated for customer <code>{customer_id}</code>.</p>"
            f"</body></html>"
        )

    # No state and no existing mapping: direct marketplace install without
    # a dashboard-initiated flow. We can't bind this to a tenant.
    if customer_id is None:
        if dashboard_base:
            return _landed_redirect(
                dashboard_base,
                source=source,
                customer_id="",
                ok=False,
                error="install_without_state",
            )
        raise HTTPException(
            status_code=400,
            detail="OAuth callback missing tenant context. Install via the dashboard.",
        )

    connector = build_connector(source_enum, request.app.state.ctx)

    # Redirect URI must match exactly what was sent in the install step.
    redirect_uri = str(request.url).split("?", 1)[0]

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
    token = token.model_copy(update={"customer_id": customer_id})
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
            customer=customer_id,
            error=str(exc),
        )
        refs = []
    for ref in refs:
        await record_mapping(
            customer_id=customer_id,
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
        await enqueue_backfill(customer_id=customer_id, source=source_enum)
        backfill_msg = (
            "<p><small>Historical backfill queued. Check "
            f"<code>/backfill/status?customer_id={customer_id}</code> for progress.</small></p>"
        )
    except Exception as exc:
        log.warning(
            "oauth.backfill_enqueue_failed",
            customer=customer_id,
            source=source,
            error=str(exc),
        )
        backfill_msg = (
            "<p><small>Backfill could not be queued (see server logs). "
            "Live webhooks will still flow.</small></p>"
        )

    log.info(
        "oauth.connected",
        customer=customer_id,
        source=source,
        workspaces=[r.external_id for r in refs],
    )
    if dashboard_base:
        return _landed_redirect(
            dashboard_base,
            source=source,
            customer_id=customer_id,
            ok=True,
            workspaces=[r.external_name or r.external_id for r in refs],
        )
    return HTMLResponse(
        f"<!doctype html><html><body><h2>Connected: {source}</h2>"
        f"<p>Integration active for customer <code>{customer_id}</code>.</p>"
        f"{workspaces_msg}{backfill_msg}</body></html>"
    )


async def _resolve_customer_from_callback(
    source: SourceSystem, extra_params: dict[str, str]
) -> str | None:
    """Derive customer_id when `state` was lost (marketplace install, update fire).

    Uses the connector-native workspace id already recorded in
    `customer_source_mapping` at first install. Returns None if no mapping
    exists yet — that case must be surfaced as a 'please install via the
    dashboard' flow, not a silent bind.
    """
    if source == SourceSystem.GITHUB:
        installation_id = extra_params.get("installation_id")
        if installation_id:
            return await resolve_customer(source, installation_id)
    return None


def _source(s: str) -> SourceSystem:
    try:
        return SourceSystem(s)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=f"unknown source '{s}'") from exc


def _landed_redirect(
    dashboard_base: str,
    *,
    source: str,
    customer_id: str,
    ok: bool,
    error: str | None = None,
    workspaces: list[str] | None = None,
) -> RedirectResponse:
    params: dict[str, str] = {
        "source": source,
        "customer_id": customer_id,
        "ok": "1" if ok else "0",
    }
    if error:
        params["error"] = error[:200]
    if workspaces:
        params["workspaces"] = ",".join(workspaces)[:200]
    url = f"{dashboard_base}/oauth-landed?{urlencode(params)}"
    return RedirectResponse(url=url, status_code=302)
