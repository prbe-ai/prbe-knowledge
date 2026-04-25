"""Internal API for the dashboard BFF (prbe-backend) to manage tenants.

Gated by a shared X-Admin-Key header. All routes live under /api and
are unavailable when `ADMIN_API_KEY` is unset (503). prbe-backend
calls these to bootstrap tenants, read integration status, and surface
ingestion stats — never served directly to end users.

Route layout:
    POST   /api/customers                           — create tenant
    POST   /api/customers/{id}/rotate_key           — issue new API key
    GET    /api/customers                           — list tenants
    GET    /api/customers/{id}/integrations         — per-source status
    GET    /api/customers/{id}/ingestion_stats      — per-source counters
    POST   /api/oauth/{source}/exchange             — token exchange
                                                       (callback handled by
                                                       prbe-backend gateway)
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel, Field

from services.ingestion.backfill_runner import (
    enqueue_backfill,
    re_enqueue_for_polling,
)
from services.ingestion.handlers.registry import (
    build_connector,
    get_connector_class,
)
from shared.config import get_settings
from shared.constants import (
    GRANOLA_REFRESH_CHANNEL,
    GRANOLA_REFRESH_DEBOUNCE_SECONDS,
    GRANOLA_SCOPE_ENTERPRISE,
    GRANOLA_SCOPE_PERSONAL,
    IntegrationStatus,
    SourceSystem,
)
from shared.customer_mapping import record_mapping
from shared.db import get_pool, raw_conn, with_tenant
from shared.encryption import encrypt_token
from shared.exceptions import (
    HandlerNotFound,
    NotSupportedByConnector,
    PrbeError,
)
from shared.logging import get_logger
from shared.provisioning import (
    CustomerAlreadyExists,
    CustomerNotFound,
    create_customer,
    delete_customer,
    ensure_bucket_for,
    rotate_customer_key,
)
from shared.tokens import save_token

# Sources that do NOT use OAuth — admin endpoints handle their connect flow
# explicitly (paste-an-API-key, etc.) and the OAuth install URL is omitted
# from the integrations response.
_NO_OAUTH_SOURCES: frozenset[SourceSystem] = frozenset({SourceSystem.GRANOLA})

log = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["internal-api"])


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def verify_admin_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    expected_secret = get_settings().admin_api_key
    # Treat an empty SecretStr as "unset" — otherwise a blank .env value
    # would silently allow the header check to pass against "" == "".
    if expected_secret is None or not expected_secret.get_secret_value():
        raise HTTPException(
            status_code=503, detail="admin disabled — set ADMIN_API_KEY"
        )
    if x_admin_key is None:
        raise HTTPException(status_code=401, detail="missing X-Admin-Key")
    # compare_digest avoids timing-based token recovery.
    if not hmac.compare_digest(x_admin_key, expected_secret.get_secret_value()):
        raise HTTPException(status_code=401, detail="invalid X-Admin-Key")


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CreateCustomerRequest(BaseModel):
    customer_id: str
    display_name: str
    redirect_uri_base: str


class CreateCustomerResponse(BaseModel):
    customer_id: str
    display_name: str
    api_key: str
    bucket: str
    install_urls: dict[str, str]


class RotateKeyResponse(BaseModel):
    api_key: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_install_urls(customer_id: str, redirect_uri_base: str) -> dict[str, str]:
    base = redirect_uri_base.rstrip("/")
    return {
        src.value: (
            f"{base}/oauth/{src.value}/install"
            f"?customer_id={customer_id}"
            f"&redirect_uri={base}/oauth/{src.value}/callback"
        )
        for src in SourceSystem
        if src not in _NO_OAUTH_SOURCES
    }


async def _customer_exists(customer_id: str) -> bool:
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            "SELECT 1 FROM customers WHERE customer_id = $1", customer_id
        )
    return row is not None


def _iso(ts: Any) -> str | None:
    return ts.isoformat() if ts is not None else None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/customers",
    response_model=CreateCustomerResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def create_customer_route(body: CreateCustomerRequest) -> CreateCustomerResponse:
    try:
        api_key = await create_customer(body.customer_id, body.display_name)
    except CustomerAlreadyExists as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    bucket = await ensure_bucket_for(body.customer_id)
    return CreateCustomerResponse(
        customer_id=body.customer_id,
        display_name=body.display_name,
        api_key=api_key,
        bucket=bucket,
        install_urls=_build_install_urls(body.customer_id, body.redirect_uri_base),
    )


@router.post(
    "/customers/{customer_id}/rotate_key",
    response_model=RotateKeyResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def rotate_key_route(customer_id: str) -> RotateKeyResponse:
    try:
        api_key = await rotate_customer_key(customer_id)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RotateKeyResponse(api_key=api_key)


@router.delete(
    "/customers/{customer_id}",
    status_code=204,
    dependencies=[Depends(verify_admin_key)],
)
async def delete_customer_route(customer_id: str) -> None:
    try:
        await delete_customer(customer_id)
    except CustomerNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/customers", dependencies=[Depends(verify_admin_key)])
async def list_customers_route() -> dict[str, Any]:
    async with raw_conn() as conn:
        rows = await conn.fetch(
            """
            SELECT customer_id, display_name, created_at
            FROM customers
            ORDER BY created_at ASC, customer_id ASC
            """
        )
    return {
        "customers": [
            {
                "customer_id": r["customer_id"],
                "display_name": r["display_name"],
                "created_at": _iso(r["created_at"]),
            }
            for r in rows
        ]
    }


@router.get(
    "/customers/{customer_id}/integrations",
    dependencies=[Depends(verify_admin_key)],
)
async def get_integrations_route(
    customer_id: str,
    request: Request,
    redirect_uri_base: str | None = Query(default=None),
) -> dict[str, Any]:
    if not await _customer_exists(customer_id):
        raise HTTPException(status_code=404, detail="customer not found")

    base = redirect_uri_base or str(request.base_url).rstrip("/")
    install_urls = _build_install_urls(customer_id, base)

    async with with_tenant(customer_id) as conn:
        token_rows = await conn.fetch(
            """
            SELECT source_system, status, scope, last_refresh_at, last_refresh_error
            FROM integration_tokens
            WHERE customer_id = $1
            """,
            customer_id,
        )
        mapping_rows = await conn.fetch(
            """
            SELECT source_system, external_id, external_name
            FROM customer_source_mapping
            WHERE customer_id = $1
            """,
            customer_id,
        )
        backfill_rows = await conn.fetch(
            """
            SELECT source_system, status, events_enqueued,
                   started_at, completed_at, last_progress_at, last_error
            FROM backfill_state
            WHERE customer_id = $1
            """,
            customer_id,
        )
        # failed_chunks: per-source count surfaced as a badge on the dashboard
        # card. Joins through documents because failed_chunks itself doesn't
        # carry source_system.
        failed_rows = await conn.fetch(
            """
            SELECT d.source_system, COUNT(*) AS n
            FROM failed_chunks f
            JOIN documents d ON d.doc_id = f.doc_id AND d.customer_id = f.customer_id
            WHERE f.customer_id = $1
            GROUP BY d.source_system
            """,
            customer_id,
        )
        # last_event_at: most recent webhook/poll arrival per source. The
        # signal users actually want when asking "is data flowing in?" —
        # different from `last_refresh_at` (OAuth token rotation only).
        # Also surface a 24h count so the card can flag silent integrations.
        event_rows = await conn.fetch(
            """
            SELECT source_system,
                   MAX(received_at) AS last_received_at,
                   COUNT(*) FILTER (
                     WHERE received_at > NOW() - INTERVAL '24 hours'
                   ) AS events_24h
            FROM ingestion_events
            WHERE customer_id = $1
            GROUP BY source_system
            """,
            customer_id,
        )

    token_by_source: dict[str, Any] = {r["source_system"]: r for r in token_rows}
    backfill_by_source: dict[str, Any] = {
        r["source_system"]: r for r in backfill_rows
    }
    failed_by_source: dict[str, int] = {
        r["source_system"]: int(r["n"]) for r in failed_rows
    }
    events_by_source: dict[str, dict[str, Any]] = {
        r["source_system"]: {
            "last_received_at": r["last_received_at"],
            "events_24h": int(r["events_24h"]),
        }
        for r in event_rows
    }
    mappings_by_source: dict[str, list[dict[str, Any]]] = {}
    for r in mapping_rows:
        mappings_by_source.setdefault(r["source_system"], []).append(
            {"external_id": r["external_id"], "external_name": r["external_name"]}
        )

    integrations = []
    for src in SourceSystem:
        tok = token_by_source.get(src.value)
        bf = backfill_by_source.get(src.value)
        ev = events_by_source.get(src.value)
        connected = bool(tok and tok["status"] == IntegrationStatus.ACTIVE.value)
        integrations.append(
            {
                "source": src.value,
                "connected": connected,
                "workspaces": mappings_by_source.get(src.value, []),
                "last_refresh_at": _iso(tok["last_refresh_at"]) if tok else None,
                "last_refresh_error": tok["last_refresh_error"] if tok else None,
                "last_event_at": _iso(ev["last_received_at"]) if ev else None,
                "events_24h": ev["events_24h"] if ev else 0,
                "install_url": install_urls.get(src.value),
                "scope": tok["scope"] if tok else None,
                "backfill_status": bf["status"] if bf else None,
                "backfill_events_enqueued": int(bf["events_enqueued"]) if bf else 0,
                "backfill_last_progress_at": _iso(bf["last_progress_at"]) if bf else None,
                "backfill_completed_at": _iso(bf["completed_at"]) if bf else None,
                "backfill_started_at": _iso(bf["started_at"]) if bf else None,
                "backfill_last_error": bf["last_error"] if bf else None,
                "failed_chunk_count": failed_by_source.get(src.value, 0),
            }
        )

    return {"integrations": integrations}


@router.get(
    "/customers/{customer_id}/ingestion_stats",
    dependencies=[Depends(verify_admin_key)],
)
async def get_ingestion_stats_route(customer_id: str) -> dict[str, Any]:
    if not await _customer_exists(customer_id):
        raise HTTPException(status_code=404, detail="customer not found")

    per_source: list[dict[str, Any]] = []

    async with with_tenant(customer_id) as conn:
        for src in SourceSystem:
            documents = await conn.fetchval(
                """
                SELECT COUNT(*) FROM documents
                WHERE customer_id = $1 AND source_system = $2 AND valid_to IS NULL
                """,
                customer_id,
                src.value,
            )
            # chunks has no source_system column — join via documents on doc_id.
            chunks = await conn.fetchval(
                """
                SELECT COUNT(*) FROM chunks c
                WHERE c.customer_id = $1 AND c.valid_to IS NULL
                  AND EXISTS (
                      SELECT 1 FROM documents d
                      WHERE d.doc_id = c.doc_id
                        AND d.customer_id = $1
                        AND d.source_system = $2
                  )
                """,
                customer_id,
                src.value,
            )
            queue_rows = await conn.fetch(
                """
                SELECT status, COUNT(*) AS n
                FROM ingestion_queue
                WHERE customer_id = $1 AND source_system = $2
                GROUP BY status
                """,
                customer_id,
                src.value,
            )
            queue_by_status = {r["status"]: r["n"] for r in queue_rows}
            last_ingested = await conn.fetchval(
                """
                SELECT MAX(ingested_at) FROM documents
                WHERE customer_id = $1 AND source_system = $2
                """,
                customer_id,
                src.value,
            )
            per_source.append(
                {
                    "source": src.value,
                    "documents": int(documents or 0),
                    "chunks": int(chunks or 0),
                    "queue_pending": int(queue_by_status.get("pending", 0)),
                    "queue_processing": int(queue_by_status.get("processing", 0)),
                    "queue_dlq": int(queue_by_status.get("dlq", 0)),
                    "last_ingested_at": _iso(last_ingested),
                }
            )

        backfill_rows = await conn.fetch(
            """
            SELECT source_system, status, events_enqueued, started_at,
                   completed_at, last_error
            FROM backfill_state
            WHERE customer_id = $1
            ORDER BY source_system ASC
            """,
            customer_id,
        )

    backfill = [
        {
            "source": r["source_system"],
            "status": r["status"],
            "events_enqueued": r["events_enqueued"],
            "started_at": _iso(r["started_at"]),
            "completed_at": _iso(r["completed_at"]),
            "last_error": r["last_error"],
        }
        for r in backfill_rows
    ]

    return {"per_source": per_source, "backfill": backfill}


# ---------------------------------------------------------------------------
# Granola — paste-an-API-key connect flow (no OAuth).
#
# Granola does not support OAuth or webhooks (as of 2026-04). The dashboard
# opens a modal that POSTs the user's Personal/Enterprise API key here.
# We validate by hitting Granola's /v1/notes endpoint, then store the key
# encrypted and enqueue an initial full backfill. Subsequent steady-state
# polling is driven by the prbe-knowledge-poller process (5-min cadence)
# plus pg_notify('granola_refresh') for sub-second manual-refresh wake.
# ---------------------------------------------------------------------------


_GRANOLA_VALIDATE_URL = "https://public-api.granola.ai/v1/notes"


class GranolaConnectRequest(BaseModel):
    api_key: str = Field(min_length=8, max_length=512)
    tier: str = Field(
        default="enterprise",
        description="'personal' or 'enterprise' — controls which scope label is "
        "stored on the integration_tokens row. Personal sees only the issuing "
        "user's notes + shared. Enterprise sees the whole workspace.",
    )


class GranolaConnectResponse(BaseModel):
    customer_id: str
    source: str = "granola"
    scope: str
    backfill_enqueued: bool


class GranolaRefreshResponse(BaseModel):
    customer_id: str
    source: str = "granola"
    triggered: bool
    next_check_in_seconds: int = 5


def _granola_scope(tier: str) -> str:
    t = tier.strip().lower()
    if t == "personal":
        return GRANOLA_SCOPE_PERSONAL
    if t == "enterprise":
        return GRANOLA_SCOPE_ENTERPRISE
    raise HTTPException(
        status_code=400,
        detail="tier must be 'personal' or 'enterprise'",
    )


async def _validate_granola_key(api_key: str) -> None:
    """Hit Granola's /v1/notes with the key to confirm it works.

    Raises HTTPException(400) on auth failure (so the dashboard can show a
    meaningful error inline). Treats network errors as 503 so the user knows
    to retry rather than thinking their key is bad.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "prbe-knowledge/0.1",
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                _GRANOLA_VALIDATE_URL,
                headers=headers,
                params={"limit": 1},
            )
    except (httpx.HTTPError, OSError) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"could not reach Granola API: {exc}",
        ) from exc

    if resp.status_code in {401, 403}:
        raise HTTPException(
            status_code=400,
            detail="Granola rejected the API key (check tier + that it's not revoked)",
        )
    if resp.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Granola rate limit hit during validation; retry in a few seconds",
        )
    if resp.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Granola returned unexpected status {resp.status_code}",
        )


@router.post(
    "/customers/{customer_id}/integrations/granola",
    response_model=GranolaConnectResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def connect_granola_route(
    customer_id: str, body: GranolaConnectRequest
) -> GranolaConnectResponse:
    if not await _customer_exists(customer_id):
        raise HTTPException(status_code=404, detail="customer not found")

    scope = _granola_scope(body.tier)
    await _validate_granola_key(body.api_key)
    encrypted = encrypt_token(body.api_key)

    async with raw_conn() as conn:
        # Don't set last_refresh_at — connect does NOT count as a refresh.
        # That field tracks the last time the user clicked "Refresh now" so
        # the debounce check works. Setting it here would block the user
        # from manually refreshing for 30s after connecting, which is the
        # opposite of what they'd want (they just hit Connect and want to
        # see data populate).
        await conn.execute(
            """
            INSERT INTO integration_tokens
                (customer_id, source_system, access_token_encrypted,
                 scope, status, last_refresh_error)
            VALUES ($1, $2, $3, $4, $5, NULL)
            ON CONFLICT (customer_id, source_system)
            DO UPDATE SET access_token_encrypted = EXCLUDED.access_token_encrypted,
                          scope                  = EXCLUDED.scope,
                          status                 = EXCLUDED.status,
                          last_refresh_error     = NULL,
                          updated_at             = NOW()
            """,
            customer_id,
            SourceSystem.GRANOLA.value,
            encrypted,
            scope,
            IntegrationStatus.ACTIVE.value,
        )

    # Initial backfill: clear cursor so we pull from epoch (full history).
    # Subsequent polls go through re_enqueue_for_polling which preserves the
    # watermark for incremental syncs.
    await enqueue_backfill(customer_id, SourceSystem.GRANOLA)
    log.info(
        "granola.connected",
        customer=customer_id,
        scope=scope,
    )
    return GranolaConnectResponse(
        customer_id=customer_id,
        scope=scope,
        backfill_enqueued=True,
    )


@router.delete(
    "/customers/{customer_id}/integrations/granola",
    status_code=204,
    dependencies=[Depends(verify_admin_key)],
)
async def disconnect_granola_route(customer_id: str) -> None:
    if not await _customer_exists(customer_id):
        raise HTTPException(status_code=404, detail="customer not found")
    async with raw_conn() as conn:
        await conn.execute(
            """
            UPDATE integration_tokens
            SET status = $1,
                last_refresh_error = 'disconnected by admin',
                updated_at = NOW()
            WHERE customer_id = $2 AND source_system = $3
            """,
            IntegrationStatus.REVOKED.value,
            customer_id,
            SourceSystem.GRANOLA.value,
        )


@router.post(
    "/customers/{customer_id}/integrations/granola/refresh",
    response_model=GranolaRefreshResponse,
    dependencies=[Depends(verify_admin_key)],
)
async def refresh_granola_route(customer_id: str) -> GranolaRefreshResponse:
    """Manually re-enqueue a Granola backfill, preserving the cursor.

    Debounced: returns 429 if a refresh fired in the last
    GRANOLA_REFRESH_DEBOUNCE_SECONDS. Sends pg_notify so the worker's
    BackfillWorker wakes immediately instead of waiting for its 5s tick.
    """
    if not await _customer_exists(customer_id):
        raise HTTPException(status_code=404, detail="customer not found")

    async with raw_conn() as conn:
        token = await conn.fetchrow(
            """
            SELECT status, last_refresh_at FROM integration_tokens
            WHERE customer_id = $1 AND source_system = $2
            """,
            customer_id,
            SourceSystem.GRANOLA.value,
        )
    if token is None:
        raise HTTPException(
            status_code=404, detail="granola integration not configured"
        )
    if token["status"] != IntegrationStatus.ACTIVE.value:
        raise HTTPException(
            status_code=409,
            detail=f"granola integration status is {token['status']}; reconnect first",
        )

    last = token["last_refresh_at"]
    if last is not None:
        # Tolerate naive vs aware timestamps in case the column ever drops tz.
        last_aware = last if last.tzinfo else last.replace(tzinfo=UTC)
        elapsed = datetime.now(UTC) - last_aware
        if elapsed < timedelta(seconds=GRANOLA_REFRESH_DEBOUNCE_SECONDS):
            retry_in = GRANOLA_REFRESH_DEBOUNCE_SECONDS - int(elapsed.total_seconds())
            raise HTTPException(
                status_code=429,
                detail=f"refresh debounced; retry in {retry_in}s",
                headers={"Retry-After": str(retry_in)},
            )

    triggered = await re_enqueue_for_polling(customer_id, SourceSystem.GRANOLA)

    # Update last_refresh_at regardless — the user clicked refresh, so the
    # debounce should apply to their click, not just to successful enqueues.
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            UPDATE integration_tokens
            SET last_refresh_at = NOW(), updated_at = NOW()
            WHERE customer_id = $1 AND source_system = $2
            """,
            customer_id,
            SourceSystem.GRANOLA.value,
        )
        # pg_notify() is the function form of NOTIFY that accepts parameters.
        # Sent on the same connection so the LISTEN side sees it after the
        # row update commits (not inside a transaction here, so commit is
        # at end-of-statement).
        await conn.execute(
            "SELECT pg_notify($1, $2)",
            GRANOLA_REFRESH_CHANNEL,
            customer_id,
        )

    log.info(
        "granola.refresh_triggered",
        customer=customer_id,
        re_enqueued=triggered,
    )
    return GranolaRefreshResponse(
        customer_id=customer_id,
        triggered=triggered,
    )


# ---------------------------------------------------------------------------
# OAuth code-for-token exchange (gateway pattern).
#
# prbe-backend handles the public-facing OAuth callback at
# /oauth/{source}/callback. After verifying the HMAC-signed state and
# resolving customer_id, it POSTs here with {customer_id, code,
# redirect_uri, extra_params}. We run the per-source token exchange,
# persist the encrypted token, identify workspaces, and queue backfill —
# the same logic as the legacy /oauth/{source}/callback under
# services/ingestion/oauth/routes.py, just split off the public surface.
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

    # Connectors don't know the tenant; bind it from the request.
    token = token.model_copy(update={"customer_id": body.customer_id})
    await save_token(token)

    # Identify workspaces this token grants access to so incoming webhooks
    # can resolve customer_id from payload alone.
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
