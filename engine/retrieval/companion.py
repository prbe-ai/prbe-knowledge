"""HTTP surface for the companion transport: enqueue, poll, claim, ack, readback.

The engine half of `/v1/companion/*` (implementation spec v3 §2.4). research-os
proxies here and OWNS recipient authorization: it resolves both credential
kinds to `user:<uuid>`, checks session ownership, and never forwards a client-
supplied recipient it did not verify. This service is trusted-internal
(`authenticate_query` admits the internal key + customer pair), so `recipient`
IS accepted from the caller here for exactly the reason `actor_ref` is in
procedures.py: the trust boundary is one hop up. If this service ever becomes
reachable by an untrusted caller, that field is the first thing to revisit.

GATING IS PER-TENANT AND FAILS CLOSED. Without `companion_infra` every route
returns an empty result carrying the three-state envelope, and nothing is
written -- not a card, not an ack. Off is not 403: an HTTP status cannot spell
the difference between "nobody turned it on" and "not entitled".

THE WAIT IS A WAIT, NOT A HELD CONNECTION. `/companion/poll` sleeps between
short reads; each read is its own `with_tenant` (a connection and a
transaction per call), so nothing is held across the long-poll interval and
multiple engine replicas are correct by construction (no LISTEN/NOTIFY).
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from engine.retrieval.auth import authenticate_query
from engine.shared.companion.capability import companion_envelope
from engine.shared.companion.mailbox import (
    BODY_MAX,
    DEDUPE_KEY_MAX,
    SESSION_ID_MAX,
    TTL_MAX,
    TTL_MIN,
    AckRefused,
    Card,
    EnqueueConflict,
    EnqueueRefused,
    UnknownCard,
    ack,
    claim_for_actor,
    deliveries,
    enqueue,
    pending,
    report,
)
from engine.shared.logging import get_logger

log = get_logger(__name__)

companion_router = APIRouter()

#: Upper bound on one poll's wait; research-os proxies with a 45s timeout.
POLL_WAIT_MAX_S = 25
#: Sleep between short reads while waiting.
POLL_INTERVAL_S = 2.0
#: Config rider version. Bump when a field's meaning changes.
CONFIG_VERSION = 1


# --------------------------------------------------------------------------
# Wire shapes
# --------------------------------------------------------------------------


class _Strict(BaseModel):
    """Refuse unknown fields (spec §2.3 strict parse).

    Pydantic ignores extras by default, which would make a client sending
    `actor_ref`, `source` or `mode` look like it worked -- silently dropped while
    the server pins its own value. A 422 turns a misunderstanding into a
    visible error rather than a quiet mis-attribution.
    """

    model_config = ConfigDict(extra="forbid")


class CapabilityOut(BaseModel):
    """The house three-state shape. `entitled` is hardcoded True until Phase 3."""

    enabled: bool
    entitled: bool = True
    upgrade_url: str | None = None


class CardOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    mailbox_id: UUID
    recipient: str
    session_id: str | None
    class_: str = Field(alias="class")
    intended_seam: str | None
    body: str
    trial_id: UUID
    created_at: datetime
    expires_at: datetime


def _card_out(card: Card) -> CardOut:
    return CardOut.model_validate(
        {
            "mailbox_id": card.mailbox_id,
            "recipient": card.recipient,
            "session_id": card.session_id,
            "class": card.class_,
            "intended_seam": card.intended_seam,
            "body": card.body,
            "trial_id": card.trial_id,
            "created_at": card.created_at,
            "expires_at": card.expires_at,
        }
    )


class EnqueueRequest(_Strict):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    recipient: str = Field(min_length=6, max_length=200)
    session_id: str | None = Field(default=None, max_length=SESSION_ID_MAX)
    class_: str = Field(alias="class")
    body: str = Field(min_length=1, max_length=BODY_MAX)
    dedupe_key: str = Field(min_length=1, max_length=DEDUPE_KEY_MAX)
    ttl_seconds: int = Field(ge=TTL_MIN, le=TTL_MAX)
    intended_seam: str | None = None


class EnqueueResponse(BaseModel):
    capability: CapabilityOut
    card: CardOut | None = None
    created: bool = False


class KnownCard(_Strict):
    mailbox_id: UUID
    #: Informational: any state the client reports means "do not re-offer".
    state: str = Field(min_length=1, max_length=32)


class PollRequest(_Strict):
    recipient: str = Field(min_length=6, max_length=200)
    session_id: str = Field(min_length=1, max_length=SESSION_ID_MAX)
    known: list[KnownCard] = Field(default_factory=list, max_length=500)
    wait_seconds: int = Field(default=0, ge=0, le=POLL_WAIT_MAX_S)


class ConfigOut(BaseModel):
    version: int = CONFIG_VERSION
    enabled: bool
    hot_flush_patterns: list[str] = Field(default_factory=list)
    max_cards_per_emission: int = 3
    max_emission_chars: int = 8_000
    max_config_age_s: int = 60
    poll_wait_max_s: int = POLL_WAIT_MAX_S


class PollResponse(BaseModel):
    capability: CapabilityOut
    cards: list[CardOut] = Field(default_factory=list)
    config: ConfigOut


class ClaimRequest(_Strict):
    recipient: str = Field(min_length=6, max_length=200)
    claimed_by: str = Field(min_length=1, max_length=200)
    lease_seconds: int = Field(default=60, ge=1, le=3_600)


class ClaimResponse(BaseModel):
    capability: CapabilityOut
    card: CardOut | None = None


class AckRequest(_Strict):
    mailbox_id: UUID
    attempt_id: UUID
    seam: str = Field(min_length=1, max_length=32)
    outcome: str = Field(min_length=1, max_length=32)
    receiving_instance: str = Field(min_length=1, max_length=200)
    delivering_credential: str | None = Field(default=None, max_length=200)
    harness_version: str | None = Field(default=None, max_length=64)
    session_state: str | None = Field(default=None, max_length=16)
    # tz-aware only: asyncpg would store a naive value as server-local time.
    client_received_at: AwareDatetime | None = None
    client_emitted_at: AwareDatetime | None = None
    receipt_to_emission_ms: int | None = Field(default=None, ge=0)
    evidence: dict[str, Any] = Field(default_factory=dict)


class AckResponse(BaseModel):
    capability: CapabilityOut
    delivery_id: int | None = None
    created: bool = False


class DeliveryOut(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: int
    mailbox_id: UUID
    attempt_id: UUID
    seam: str
    outcome: str
    delivering_credential: str | None
    receiving_instance: str
    harness_version: str | None
    session_state: str | None
    client_received_at: datetime | None
    client_emitted_at: datetime | None
    receipt_to_emission_ms: int | None
    ack_received_at: datetime
    evidence: dict[str, Any]
    session_id: str | None
    recipient: str
    trial_id: UUID
    intended_seam: str | None
    class_: str = Field(alias="class")
    enqueued_at: datetime


class DeliveriesResponse(BaseModel):
    capability: CapabilityOut
    deliveries: list[DeliveryOut] = Field(default_factory=list)
    #: True when more rows exist beyond `limit` -- an actor-wide readback can
    #: otherwise hide earlier attempts behind the bound.
    truncated: bool = False


class LatencyOut(BaseModel):
    n: int
    p50: float | None
    p95: float | None
    min: int | None
    max: int | None


class SeamReportOut(BaseModel):
    seam: str
    outcome: str
    attempts: int
    harness_accepted: int
    observed_in_context: int
    latency_ms: LatencyOut
    first_at: datetime
    last_at: datetime


class ReportResponse(BaseModel):
    capability: CapabilityOut
    seams: list[SeamReportOut] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


async def _envelope(customer_id: str) -> CapabilityOut:
    return CapabilityOut(**await companion_envelope(customer_id))  # type: ignore[arg-type]


def _config(enabled: bool) -> ConfigOut:
    return ConfigOut(enabled=enabled)


@companion_router.post("/companion/enqueue", response_model=EnqueueResponse)
async def enqueue_card(
    req: EnqueueRequest,
    customer_id: str = Depends(authenticate_query),
) -> EnqueueResponse:
    capability = await _envelope(customer_id)
    if not capability.enabled:
        return EnqueueResponse(capability=capability)
    try:
        card, created = await enqueue(
            customer_id,
            recipient=req.recipient,
            session_id=req.session_id,
            class_=req.class_,
            body=req.body,
            dedupe_key=req.dedupe_key,
            ttl_seconds=req.ttl_seconds,
            intended_seam=req.intended_seam,
        )
    except EnqueueRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except EnqueueConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"reason": exc.reason, "mailbox_id": str(exc.existing.mailbox_id)},
        ) from exc
    return EnqueueResponse(capability=capability, card=_card_out(card), created=created)


@companion_router.post("/companion/poll", response_model=PollResponse)
async def poll_cards(
    req: PollRequest,
    request: Request,
    customer_id: str = Depends(authenticate_query),
) -> PollResponse:
    capability = await _envelope(customer_id)
    if not capability.enabled:
        return PollResponse(capability=capability, config=_config(False))
    exclude = {k.mailbox_id for k in req.known}
    loop = asyncio.get_running_loop()
    deadline = loop.time() + req.wait_seconds
    while True:
        cards = await pending(
            customer_id, session_id=req.session_id, recipient=req.recipient, exclude=exclude
        )
        remaining = deadline - loop.time()
        if cards or remaining <= 0:
            break
        if await request.is_disconnected():
            log.info("companion.poll.client_gone", customer=customer_id, session=req.session_id)
            break
        await asyncio.sleep(min(POLL_INTERVAL_S, remaining))
    # The gate can be withdrawn while a poll waits (consent revoked, tenant cell
    # flipped). Nothing goes out on a snapshot taken up to 25 seconds ago.
    capability = await _envelope(customer_id)
    if not capability.enabled:
        return PollResponse(capability=capability, config=_config(False))
    return PollResponse(
        capability=capability,
        cards=[_card_out(c) for c in cards],
        config=_config(True),
    )


@companion_router.post("/companion/claim", response_model=ClaimResponse)
async def claim_card(
    req: ClaimRequest,
    customer_id: str = Depends(authenticate_query),
) -> ClaimResponse:
    capability = await _envelope(customer_id)
    if not capability.enabled:
        return ClaimResponse(capability=capability)
    card = await claim_for_actor(
        customer_id,
        recipient=req.recipient,
        claimed_by=req.claimed_by,
        lease_seconds=req.lease_seconds,
    )
    return ClaimResponse(capability=capability, card=_card_out(card) if card else None)


@companion_router.post("/companion/ack", response_model=AckResponse)
async def ack_delivery(
    req: AckRequest,
    customer_id: str = Depends(authenticate_query),
) -> AckResponse:
    capability = await _envelope(customer_id)
    if not capability.enabled:
        return AckResponse(capability=capability)
    try:
        delivery_id, created = await ack(
            customer_id,
            mailbox_id=req.mailbox_id,
            attempt_id=req.attempt_id,
            seam=req.seam,
            outcome=req.outcome,
            receiving_instance=req.receiving_instance,
            delivering_credential=req.delivering_credential,
            harness_version=req.harness_version,
            session_state=req.session_state,
            client_received_at=req.client_received_at,
            client_emitted_at=req.client_emitted_at,
            receipt_to_emission_ms=req.receipt_to_emission_ms,
            evidence=req.evidence,
        )
    except UnknownCard as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AckRefused as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AckResponse(capability=capability, delivery_id=delivery_id, created=created)


@companion_router.get("/companion/deliveries", response_model=DeliveriesResponse)
async def list_deliveries(
    session_id: str | None = Query(default=None, min_length=1, max_length=SESSION_ID_MAX),
    recipient: str | None = Query(default=None, min_length=6, max_length=200),
    limit: int = Query(default=200, ge=1, le=1_000),
    customer_id: str = Depends(authenticate_query),
) -> DeliveriesResponse:
    if session_id is None and recipient is None:
        raise HTTPException(status_code=422, detail="filter by session_id or recipient")
    capability = await _envelope(customer_id)
    if not capability.enabled:
        return DeliveriesResponse(capability=capability)
    rows = await deliveries(
        customer_id, session_id=session_id, recipient=recipient, limit=limit + 1
    )
    truncated = len(rows) > limit
    return DeliveriesResponse(
        capability=capability,
        deliveries=[DeliveryOut.model_validate(r) for r in rows[:limit]],
        truncated=truncated,
    )


@companion_router.get("/companion/report", response_model=ReportResponse)
async def seam_report(
    session_id: str | None = Query(default=None, min_length=1, max_length=SESSION_ID_MAX),
    recipient: str | None = Query(default=None, min_length=6, max_length=200),
    customer_id: str = Depends(authenticate_query),
) -> ReportResponse:
    """Per (seam, outcome) numbers for the fault catalog (spec §8).

    Attempts, the two evidence counts (`harness_accepted`, `observed_in_context`
    -- JSON `true` in the ack's evidence), and the monotonic latency
    distribution. Cross-clock enqueue-to-emission deltas are left to the
    catalog author: they carry clock-offset uncertainty and should not be
    aggregated as if exact.
    """
    if session_id is None and recipient is None:
        raise HTTPException(status_code=422, detail="filter by session_id or recipient")
    capability = await _envelope(customer_id)
    if not capability.enabled:
        return ReportResponse(capability=capability)
    rows = await report(customer_id, session_id=session_id, recipient=recipient)
    return ReportResponse(
        capability=capability,
        seams=[
            SeamReportOut(
                seam=r["seam"],
                outcome=r["outcome"],
                attempts=r["attempts"],
                harness_accepted=r["harness_accepted"],
                observed_in_context=r["observed_in_context"],
                latency_ms=LatencyOut(
                    n=r["latency_n"],
                    p50=r["latency_p50"],
                    p95=r["latency_p95"],
                    min=r["latency_min"],
                    max=r["latency_max"],
                ),
                first_at=r["first_at"],
                last_at=r["last_at"],
            )
            for r in rows
        ],
    )
