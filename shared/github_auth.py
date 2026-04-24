"""GitHub App installation-token minter.

GitHub Apps do not use OAuth refresh tokens. To call the GitHub API as an
installation, we sign a short-lived RS256 JWT with the App's private key,
POST it to `/app/installations/{id}/access_tokens`, and get back an
installation token valid for one hour.

This module centralises that dance so every connector code path (backfill,
CODEOWNERS hydration, future sync jobs) can obtain a bearer by calling
`mint_installation_token(...)`. Results are cached in-process per
`installation_id` with a 5-minute safety margin against the returned
`expires_at`; concurrent mints for the same installation are serialized
via a per-installation asyncio.Lock so we don't stampede GitHub.
"""

from __future__ import annotations

import asyncio
import base64
import json
from datetime import UTC, datetime, timedelta

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from shared.exceptions import GitHubAuthError
from shared.logging import get_logger

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
_JWT_LIFETIME_SECONDS = 9 * 60  # 540s — GitHub caps App JWTs at 10 minutes.
_CACHE_SAFETY_MARGIN = timedelta(minutes=5)

_token_cache: dict[str, tuple[str, datetime]] = {}
_mint_locks: dict[str, asyncio.Lock] = {}


def _b64url(data: bytes) -> bytes:
    return base64.urlsafe_b64encode(data).rstrip(b"=")


def _build_app_jwt(app_id: str, private_key_pem: str) -> str:
    private_key = serialization.load_pem_private_key(
        private_key_pem.encode("utf-8"), password=None
    )

    now = int(datetime.now(UTC).timestamp())
    header = {"alg": "RS256", "typ": "JWT"}
    # iat is backdated 60s to tolerate clock skew between us and GitHub.
    payload = {"iss": app_id, "iat": now - 60, "exp": now + _JWT_LIFETIME_SECONDS}

    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signing_input = header_b64 + b"." + payload_b64

    signature = private_key.sign(signing_input, padding.PKCS1v15(), hashes.SHA256())
    return (signing_input + b"." + _b64url(signature)).decode("ascii")


def _parse_expires_at(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


async def mint_installation_token(
    http: httpx.AsyncClient,
    app_id: str,
    private_key_pem: str,
    installation_id: str,
) -> tuple[str, datetime]:
    """Mint a GitHub App installation access token.

    Signs an RS256 JWT (iss=app_id, iat=now-60, exp=now+9min) with the
    provided PEM, POSTs it to /app/installations/{id}/access_tokens,
    returns (access_token, expires_at).

    Cached per installation_id in-process until 5 minutes before expiry.
    Concurrent mints for the same installation_id are serialized so we
    don't hit GitHub N times in parallel.
    """
    now = datetime.now(UTC)
    cached = _token_cache.get(installation_id)
    if cached is not None and cached[1] > now + _CACHE_SAFETY_MARGIN:
        log.debug("github_auth.cache_hit", installation=installation_id)
        return cached

    lock = _mint_locks.setdefault(installation_id, asyncio.Lock())
    async with lock:
        cached = _token_cache.get(installation_id)
        if cached is not None and cached[1] > datetime.now(UTC) + _CACHE_SAFETY_MARGIN:
            log.debug("github_auth.cache_hit", installation=installation_id)
            return cached

        log.info("github_auth.mint_start", installation=installation_id)
        jwt_token = _build_app_jwt(app_id, private_key_pem)

        url = f"{_GITHUB_API}/app/installations/{installation_id}/access_tokens"
        try:
            resp = await http.post(
                url,
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        except httpx.HTTPError as exc:
            log.warning(
                "github_auth.mint_failed",
                installation=installation_id,
                error=str(exc),
            )
            raise GitHubAuthError(f"http error minting installation token: {exc}") from exc

        if resp.status_code != 200:
            body = resp.text
            log.warning(
                "github_auth.mint_failed",
                installation=installation_id,
                status=resp.status_code,
                body=body,
            )
            raise GitHubAuthError(
                f"mint installation token failed: status={resp.status_code} body={body}"
            )

        data = resp.json()
        token = data.get("token")
        expires_raw = data.get("expires_at")
        if not isinstance(token, str) or not isinstance(expires_raw, str):
            raise GitHubAuthError(
                f"mint installation token returned malformed body: {data!r}"
            )

        expires_at = _parse_expires_at(expires_raw)
        _token_cache[installation_id] = (token, expires_at)
        log.info(
            "github_auth.mint_success",
            installation=installation_id,
            expires_at=expires_at.isoformat(),
        )
        return token, expires_at


def _reset_cache_for_tests() -> None:
    """Test helper. Clears the process-wide mint cache + locks."""
    _token_cache.clear()
    _mint_locks.clear()


__all__ = ["GitHubAuthError", "mint_installation_token"]
