"""Neon Auth webhook receiver.

Subscribes to Neon Auth events and routes them. Currently:
  * `user.created` (non-blocking) — log only; future hook for CRM sync
  * `user.before_create` (blocking) — return {"allowed": true}; future
    hook for domain allowlists or invite-pre-validation
  * `send.magic_link` (blocking) — for email-verification / password-reset
    style links if we ever enable email-password (currently disabled).
    Returns 200 immediately.

Note on invitations: the Organization plugin's invitation emails are
controlled by `sendInvitationEmail` in the org plugin config (currently
false). When we want our own branded emails, we'll override that path
either by intercepting the create-invite call at our backend (recommended)
or by extending the webhook event set if/when Neon Auth ships invite-send
events. For now, the dashboard reads the invitation list and surfaces
"resend invite" via our backend, which sends via Resend directly.

Signature verification: detached EdDSA JWS with double base64url-encoded
payload per Neon's spec. JWKS is fetched from
{NEON_AUTH_BASE_URL}/.well-known/jwks.json.
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_pem_public_key
from fastapi import APIRouter, HTTPException, Request

from shared.config import get_settings
from shared.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

_jwks_cache: dict[str, Any] = {}
_jwks_fetched_at: float = 0.0
_JWKS_TTL_SECONDS = 600.0
_TIMESTAMP_TOLERANCE_MS = 5 * 60 * 1000


def _b64url_decode(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


async def _fetch_jwks(force: bool = False) -> dict[str, Any]:
    global _jwks_cache, _jwks_fetched_at
    base = get_settings().neon_auth_base_url
    if not base:
        raise HTTPException(
            status_code=503,
            detail="NEON_AUTH_BASE_URL not configured",
        )
    now = time.monotonic()
    if not force and _jwks_cache and (now - _jwks_fetched_at) < _JWKS_TTL_SECONDS:
        return _jwks_cache
    async with httpx.AsyncClient(timeout=5.0) as client:
        resp = await client.get(f"{base.rstrip('/')}/.well-known/jwks.json")
        resp.raise_for_status()
    _jwks_cache = resp.json()
    _jwks_fetched_at = now
    return _jwks_cache


async def _ed25519_key_for(kid: str) -> Ed25519PublicKey:
    """Find the Ed25519 public key for the given kid in the JWKS.

    Per the Neon webhook spec, keys are JWK-encoded with kty=OKP, crv=Ed25519.
    We convert the raw `x` parameter to an Ed25519PublicKey via cryptography.
    """
    jwks = await _fetch_jwks()
    for k in jwks.get("keys", []):
        if k.get("kid") != kid:
            continue
        if k.get("kty") == "OKP" and k.get("crv") == "Ed25519":
            x = _b64url_decode(k["x"])
            return Ed25519PublicKey.from_public_bytes(x)
        # Some deployments may publish PEM in the x5c slot.
        x5c = k.get("x5c")
        if x5c:
            pem = base64.b64decode(x5c[0])
            key = load_pem_public_key(pem)
            if isinstance(key, Ed25519PublicKey):
                return key
    # Refresh JWKS once in case Neon rotated keys without changing the endpoint.
    jwks = await _fetch_jwks(force=True)
    for k in jwks.get("keys", []):
        if k.get("kid") != kid:
            continue
        if k.get("kty") == "OKP" and k.get("crv") == "Ed25519":
            x = _b64url_decode(k["x"])
            return Ed25519PublicKey.from_public_bytes(x)
    raise HTTPException(
        status_code=401,
        detail=f"unknown webhook key id: {kid}",
    )


async def _verify_signature(
    raw_body: bytes,
    signature_header: str,
    kid: str,
    timestamp_ms: str,
) -> None:
    """Verify the detached EdDSA JWS over (timestamp.payload).

    Per Neon's webhook spec:
      payloadB64       = base64url(rawBody)
      signaturePayload = timestamp + "." + payloadB64
      signaturePayloadB64 = base64url(signaturePayload)
      signingInput     = header + "." + signaturePayloadB64
    """
    parts = signature_header.split(".")
    if len(parts) != 3 or parts[1] != "":
        raise HTTPException(
            status_code=401,
            detail="signature header must be detached JWS (header..sig)",
        )
    header_b64, _, signature_b64 = parts
    signature = _b64url_decode(signature_b64)

    # Timestamp freshness (replay protection).
    try:
        ts_ms = int(timestamp_ms)
    except ValueError as exc:
        raise HTTPException(
            status_code=401,
            detail="invalid X-Neon-Timestamp",
        ) from exc
    age_ms = int(time.time() * 1000) - ts_ms
    if age_ms > _TIMESTAMP_TOLERANCE_MS:
        raise HTTPException(status_code=401, detail="webhook timestamp too old")

    payload_b64 = _b64url_encode(raw_body)
    signing_payload = f"{timestamp_ms}.{payload_b64}"
    signing_payload_b64 = _b64url_encode(signing_payload.encode())
    signing_input = f"{header_b64}.{signing_payload_b64}".encode()

    key = await _ed25519_key_for(kid)
    try:
        key.verify(signature, signing_input)
    except InvalidSignature as exc:
        raise HTTPException(status_code=401, detail="invalid webhook signature") from exc


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.post("/neon-auth")
async def neon_auth_webhook(request: Request) -> dict[str, Any]:
    raw = await request.body()
    sig = request.headers.get("x-neon-signature")
    kid = request.headers.get("x-neon-signature-kid")
    ts = request.headers.get("x-neon-timestamp")
    event_type = request.headers.get("x-neon-event-type")
    event_id = request.headers.get("x-neon-event-id")

    if not (sig and kid and ts and event_type):
        raise HTTPException(
            status_code=400,
            detail="missing required Neon webhook headers",
        )

    await _verify_signature(raw, sig, kid, ts)

    try:
        json.loads(raw.decode())  # well-formed check; payload not used yet
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid json") from exc

    log.info(
        "dashboard.webhook.received",
        event_type=event_type,
        event_id=event_id,
    )

    # Blocking events: return shape per the docs.
    if event_type == "user.before_create":
        # Future: domain allowlist, invite-pre-existence check.
        return {"allowed": True}

    if event_type == "send.magic_link":
        # Email/password is disabled today, so we shouldn't see these. Acknowledge.
        return {}

    if event_type == "send.otp":
        # Same — OTP is unused.
        return {}

    if event_type == "user.created":
        # Non-blocking; ack and process async (today: nothing to do).
        return {}

    log.warning("dashboard.webhook.unknown_event", event_type=event_type)
    return {}
