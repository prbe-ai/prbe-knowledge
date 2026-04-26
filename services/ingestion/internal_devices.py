"""Internal /api/devices/* endpoints called by the prbe-backend BFF gateway.

Daemons (prbe-agent-tap) never reach this service. The flow is:

    daemon  →  api.prbe.ai/agent-tap/{pair,heartbeat,revoke}
              (prbe-backend; verifies pairing JWT or device bearer)
              ↓
              api-knowledge.prbe.ai/api/devices/*
              (this module; X-Internal-Knowledge-Key auth)

Endpoints:

    POST   /api/devices/register
        Persist a new device row (the gateway has already minted + hashed
        the device token). Also records the device in customer_source_mapping
        so the webhook handler can resolve customer_id by device_id.
    POST   /api/devices/verify-token
        Resolve a token_hash → {customer_id, employee_id, device_id, status}.
        Used by the gateway on every daemon webhook to authenticate the
        bearer device token without keeping its own copy of the hash store.
    POST   /api/devices/{device_id}/heartbeat
        Stamp last_heartbeat_at on the device row.
    POST   /api/devices/{device_id}/revoke
        Mark the device row revoked.
    GET    /api/devices?customer_id=...
        List active + revoked devices for a customer.
"""

from __future__ import annotations

from typing import Any

import orjson
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from services.ingestion.admin_routes import verify_internal_knowledge_key
from shared.constants import IntegrationStatus, SourceSystem
from shared.customer_mapping import record_mapping
from shared.db import get_pool
from shared.logging import get_logger
from shared.models import IntegrationToken
from shared.tokens import (
    list_devices_for_customer,
    revoke_device_token,
    save_device_token,
    update_device_heartbeat,
)

log = get_logger(__name__)

router = APIRouter(prefix="/api/devices", tags=["internal-devices"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class DeviceRegisterRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)
    employee_id: str = Field(min_length=1, max_length=128)
    device_id: str = Field(min_length=1, max_length=128)
    token_hash: str = Field(
        min_length=64,
        max_length=128,
        description="SHA-256 hex digest of the device token (gateway hashes the plaintext).",
    )
    os: str | None = Field(default=None, max_length=64)
    hostname: str | None = Field(default=None, max_length=256)


class DeviceRegisterResponse(BaseModel):
    customer_id: str
    device_id: str
    status: str


class DeviceVerifyTokenRequest(BaseModel):
    token_hash: str = Field(min_length=64, max_length=128)


class DeviceVerifyTokenResponse(BaseModel):
    customer_id: str
    employee_id: str
    device_id: str
    status: str


class DeviceCustomerOnlyRequest(BaseModel):
    customer_id: str = Field(min_length=1, max_length=64)


class DeviceMutationResponse(BaseModel):
    customer_id: str
    device_id: str
    updated: bool


class DeviceListResponse(BaseModel):
    customer_id: str
    devices: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post(
    "/register",
    response_model=DeviceRegisterResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def register_device(body: DeviceRegisterRequest) -> DeviceRegisterResponse:
    metadata: dict[str, Any] = {
        "employee_id": body.employee_id,
    }
    if body.os is not None:
        metadata["os"] = body.os
    if body.hostname is not None:
        metadata["hostname"] = body.hostname

    token = IntegrationToken(
        customer_id=body.customer_id,
        source_system=SourceSystem.CLAUDE_CODE,
        access_token="device-token",
        webhook_secret=body.token_hash,
        device_id=body.device_id,
        device_metadata=metadata,
    )
    await save_device_token(token)
    await record_mapping(
        customer_id=body.customer_id,
        source_system=SourceSystem.CLAUDE_CODE,
        external_id=body.device_id,
        external_name=body.hostname,
        metadata=metadata,
    )
    log.info(
        "devices.registered",
        customer=body.customer_id,
        employee=body.employee_id,
        device=body.device_id,
    )
    return DeviceRegisterResponse(
        customer_id=body.customer_id,
        device_id=body.device_id,
        status=IntegrationStatus.ACTIVE.value,
    )


@router.post(
    "/verify-token",
    response_model=DeviceVerifyTokenResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def verify_device_token(
    body: DeviceVerifyTokenRequest,
) -> DeviceVerifyTokenResponse:
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT customer_id, device_id, status, device_metadata
            FROM integration_tokens
            WHERE webhook_secret = $1
              AND source_system = $2
              AND device_id IS NOT NULL
            LIMIT 1
            """,
            body.token_hash,
            SourceSystem.CLAUDE_CODE.value,
        )
    if row is None:
        raise HTTPException(status_code=401, detail="unknown device token")
    if row["status"] != IntegrationStatus.ACTIVE.value:
        raise HTTPException(
            status_code=401, detail=f"device status is {row['status']}"
        )

    metadata = row["device_metadata"] or {}
    if isinstance(metadata, (str, bytes, bytearray)):
        metadata = orjson.loads(metadata)
    employee_id = metadata.get("employee_id") if isinstance(metadata, dict) else None
    if not employee_id:
        raise HTTPException(
            status_code=500,
            detail="device row is missing employee_id metadata",
        )

    return DeviceVerifyTokenResponse(
        customer_id=row["customer_id"],
        employee_id=employee_id,
        device_id=row["device_id"],
        status=row["status"],
    )


@router.post(
    "/{device_id}/heartbeat",
    response_model=DeviceMutationResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def heartbeat(
    device_id: str, body: DeviceCustomerOnlyRequest
) -> DeviceMutationResponse:
    updated = await update_device_heartbeat(
        body.customer_id, SourceSystem.CLAUDE_CODE, device_id
    )
    if not updated:
        raise HTTPException(status_code=404, detail="device not found or revoked")
    return DeviceMutationResponse(
        customer_id=body.customer_id, device_id=device_id, updated=True
    )


@router.post(
    "/{device_id}/revoke",
    response_model=DeviceMutationResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def revoke(
    device_id: str, body: DeviceCustomerOnlyRequest
) -> DeviceMutationResponse:
    updated = await revoke_device_token(
        body.customer_id, SourceSystem.CLAUDE_CODE, device_id
    )
    return DeviceMutationResponse(
        customer_id=body.customer_id, device_id=device_id, updated=updated
    )


@router.get(
    "",
    response_model=DeviceListResponse,
    dependencies=[Depends(verify_internal_knowledge_key)],
)
async def list_devices(
    customer_id: str = Query(min_length=1, max_length=64),
) -> DeviceListResponse:
    devices = await list_devices_for_customer(customer_id, SourceSystem.CLAUDE_CODE)
    return DeviceListResponse(customer_id=customer_id, devices=devices)
