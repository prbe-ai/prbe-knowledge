"""Admin API for the internal provisioning dashboard.

Gated by a shared X-Admin-Key header. All routes live under /admin and
are unavailable when `ADMIN_API_KEY` is unset (503). The dashboard uses
these to bootstrap tenants, read integration status, and surface ingestion
stats — never to serve end-user traffic.

Route layout:
    POST   /admin/customers                           — create tenant
    POST   /admin/customers/{id}/rotate_key           — issue new API key
    GET    /admin/customers                           — list tenants
    GET    /admin/customers/{id}/integrations         — per-source status
    GET    /admin/customers/{id}/ingestion_stats      — per-source counters
"""

from __future__ import annotations

import hmac
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request
from pydantic import BaseModel

from shared.config import get_settings
from shared.constants import IntegrationStatus, SourceSystem
from shared.db import raw_conn, with_tenant
from shared.logging import get_logger
from shared.provisioning import (
    CustomerAlreadyExists,
    CustomerNotFound,
    create_customer,
    ensure_bucket_for,
    rotate_customer_key,
)

log = get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def verify_admin_key(
    x_admin_key: str | None = Header(default=None, alias="X-Admin-Key"),
) -> None:
    expected_secret = get_settings().admin_api_key
    if expected_secret is None:
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
            SELECT source_system, status, last_refresh_at, last_refresh_error
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

    token_by_source: dict[str, Any] = {r["source_system"]: r for r in token_rows}
    mappings_by_source: dict[str, list[dict[str, Any]]] = {}
    for r in mapping_rows:
        mappings_by_source.setdefault(r["source_system"], []).append(
            {"external_id": r["external_id"], "external_name": r["external_name"]}
        )

    integrations = []
    for src in SourceSystem:
        tok = token_by_source.get(src.value)
        connected = bool(tok and tok["status"] == IntegrationStatus.ACTIVE.value)
        integrations.append(
            {
                "source": src.value,
                "connected": connected,
                "workspaces": mappings_by_source.get(src.value, []),
                "last_refresh_at": _iso(tok["last_refresh_at"]) if tok else None,
                "last_refresh_error": tok["last_refresh_error"] if tok else None,
                "install_url": install_urls[src.value],
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
