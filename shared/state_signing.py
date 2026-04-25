"""HMAC-signed OAuth `state` parameter helpers.

The OAuth `state` query parameter must round-trip from install → provider →
callback unmodified. Without signing, an attacker can craft a callback with
state=<victim_customer> and attach their own token to the victim's tenant
(token-attachment CSRF).

We wrap itsdangerous.URLSafeTimedSerializer with:
- HMAC-SHA256 (forced via signer_kwargs; library default is SHA1)
- salt="oauth-state-v1" for domain separation from any other use of
  TOKEN_ENCRYPTION_KEY
- 10-minute TTL — long enough for "click connect → auth → return", short
  enough that a leaked state isn't replayable next week

Returns None on any failure (forged, expired, wrong source, garbage). The
caller turns that into a 400 — never log the raw state in error responses,
only the first 8 chars for forensics.
"""

from __future__ import annotations

import hashlib

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from shared.config import get_settings

_MAX_AGE_SECONDS = 600  # 10 min — install + auth roundtrip
_SALT = "oauth-state-v1"


def _serializer() -> URLSafeTimedSerializer:
    key = get_settings().token_encryption_key.get_secret_value()
    return URLSafeTimedSerializer(
        key,
        salt=_SALT,
        signer_kwargs={"digest_method": hashlib.sha256},
    )


def sign_state(customer_id: str, source: str) -> str:
    """Return a signed, URL-safe state string binding customer_id + source."""
    return _serializer().dumps({"c": customer_id, "s": source})


def verify_state(state: str, source: str) -> str | None:
    """Return customer_id if state is valid, matches source, and not expired.

    None on any failure — forged signature, mismatched source, TTL exceeded,
    or unparseable input. Caller should respond 400 and log
    `oauth.state_verification_failed`.
    """
    if not state:
        return None
    try:
        payload = _serializer().loads(state, max_age=_MAX_AGE_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    except Exception:
        # Defensive: any deserialization issue (truncation, wrong shape) → reject.
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("s") != source:
        return None
    cid = payload.get("c")
    return cid if isinstance(cid, str) else None
