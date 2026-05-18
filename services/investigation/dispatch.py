"""Dispatch the investigation kickoff to prbe-orchestrator.

Called by the knowledge worker after a successful normalize when the
NormalizeOutcome carries requires_investigation=True (Plan 4 worker
hook — F.4). Also called by the review-routes reject path on re-runs
(deferred to F.5 since it needs the review-routes module).

Bounded retry on transient (5xx, network) failures with exponential
backoff. On exhaustion, raises `DispatchExhausted`; the caller should
flag the incident doc so the dashboard can offer a re-trigger button.
"""
from __future__ import annotations

import asyncio
import logging

import httpx

from shared.config import get_settings


log = logging.getLogger(__name__)


class DispatchExhausted(Exception):
    """Retries used up. Caller should mark the incident doc as
    dispatch-failed so the dashboard can surface a re-trigger action."""


async def dispatch_investigation(
    payload: dict,
    *,
    max_attempts: int = 5,
    base_delay_s: float = 0.5,
) -> None:
    """POST the dispatch payload to orchestrator. Retries on 5xx +
    transport errors only — 4xx is treated as a permanent client error
    and short-circuits to DispatchExhausted on first attempt."""
    settings = get_settings()
    url = f"{settings.orchestrator_base_url.rstrip('/')}/internal/investigations"
    headers = {
        "x-internal-backend-key": settings.internal_backend_api_key.get_secret_value(),
        "x-prbe-customer": payload["customer_id"],
        "content-type": "application/json",
    }
    last_status: int | None = None
    last_text = ""
    for attempt in range(1, max_attempts + 1):
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(url, headers=headers, json=payload)
            last_status, last_text = resp.status_code, resp.text
            if 200 <= resp.status_code < 300:
                log.info(
                    "investigation.dispatched",
                    extra={
                        "customer_id": payload["customer_id"],
                        "incident_doc_id": payload["incident_doc_id"],
                        "version": payload.get("version", 1),
                        "attempt": attempt,
                        "status": resp.status_code,
                    },
                )
                return
            if 400 <= resp.status_code < 500:
                # 4xx is permanent — schema mismatch, auth failure, etc.
                # No point retrying.
                log.error(
                    "investigation.dispatch_rejected",
                    extra={
                        "customer_id": payload["customer_id"],
                        "incident_doc_id": payload["incident_doc_id"],
                        "status": resp.status_code,
                        "body_prefix": resp.text[:200],
                    },
                )
                break
        except httpx.HTTPError as exc:
            last_status = None
            last_text = str(exc)
            log.warning(
                "investigation.dispatch_transport_error",
                extra={"error": str(exc)[:200], "attempt": attempt},
            )
        if attempt < max_attempts:
            await asyncio.sleep(base_delay_s * (2 ** (attempt - 1)))
    log.error(
        "investigation.dispatch_failed",
        extra={
            "customer_id": payload["customer_id"],
            "incident_doc_id": payload["incident_doc_id"],
            "last_status": last_status,
            "last_text": last_text[:200],
        },
    )
    raise DispatchExhausted(
        f"status={last_status} text={last_text[:200]}"
    )
