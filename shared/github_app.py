"""Local GitHub App installation-token minting for standalone (community) mode.

Hosted mode fetches installation tokens from prbe-backend (the App private key
lives there) via `shared.backend_client.fetch_github_installation_token`. When
`BACKEND_BASE_URL` is unset, the self-hoster registers their own GitHub App and
this module mints installation tokens directly from `GITHUB_APP_ID` +
`GITHUB_APP_PRIVATE_KEY`, returning the same `(token, expires_at)` tuple so the
call sites are identical.

Installation resolution: the OAuth/install flow stores the installation id on
the customer's `integration_tokens` row as ``scope = "installation:<id>"``
(see `services/ingestion/handlers/github.py`). We read it back via
`shared.tokens.load_token` — no extra config needed.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import httpx
from jose import jwt

from shared.config import get_settings
from shared.constants import GITHUB_INSTALLATION_SCOPE_PREFIX, SourceSystem
from shared.exceptions import GitHubAuthError
from shared.logging import get_logger
from shared.tokens import load_token

log = get_logger(__name__)

_GITHUB_API = "https://api.github.com"
# GitHub installation tokens live ~60min; refresh a bit early. The App JWT
# itself is short-lived (GitHub caps it at 10min) so we mint it per call.
_REFRESH_MARGIN = timedelta(minutes=5)
# Cache keyed by (customer_id, installation_id) so repeated calls within the
# token lifetime don't re-hit GitHub (mirrors backend's server-side cache).
# The customer_id is part of the key purely defensively: this minter only runs
# in single-tenant standalone mode today, but keying per-tenant means it can
# never leak a token across tenants if it's ever reused multi-tenant.
_token_cache: dict[tuple[str, str], tuple[str, datetime]] = {}


def _build_app_jwt(app_id: str, private_key_pem: str) -> str:
    now = int(time.time())
    claims = {
        "iat": now - 60,  # clock-skew cushion
        "exp": now + 9 * 60,  # GitHub rejects exp > 10min out
        "iss": app_id,
    }
    # python-jose is untyped; jwt.encode returns str.
    return str(jwt.encode(claims, private_key_pem, algorithm="RS256"))


async def _resolve_installation_id(customer_id: str) -> str:
    token = await load_token(customer_id, SourceSystem.GITHUB)
    scope = (token.scope if token else None) or ""
    if not scope.startswith(GITHUB_INSTALLATION_SCOPE_PREFIX):
        raise GitHubAuthError(
            f"no GitHub installation for customer {customer_id} "
            "(standalone mode reads installation:<id> from integration_tokens.scope)"
        )
    return scope[len(GITHUB_INSTALLATION_SCOPE_PREFIX) :]


async def mint_installation_token(
    http: httpx.AsyncClient, *, customer_id: str
) -> tuple[str, datetime]:
    """Mint a GitHub App installation token locally (standalone community mode).

    Matches `shared.backend_client.fetch_github_installation_token`'s
    `(token, expires_at)` return so the four call sites are unchanged.
    """
    settings = get_settings()
    app_id = settings.github_app_id
    pem = settings.github_app_private_key.get_secret_value()
    if not app_id or not pem:
        raise GitHubAuthError(
            "GITHUB_APP_ID / GITHUB_APP_PRIVATE_KEY not configured for local minting"
        )

    installation_id = await _resolve_installation_id(customer_id)
    cache_key = (customer_id, installation_id)

    cached = _token_cache.get(cache_key)
    if cached and cached[1] - _REFRESH_MARGIN > datetime.now(UTC):
        return cached

    app_jwt = _build_app_jwt(app_id, pem)
    url = f"{_GITHUB_API}/app/installations/{installation_id}/access_tokens"
    try:
        resp = await http.post(
            url,
            headers={
                "Authorization": f"Bearer {app_jwt}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise GitHubAuthError(f"GitHub token endpoint unreachable: {exc}") from exc

    if resp.status_code == 404:
        raise GitHubAuthError(
            f"GitHub installation {installation_id} not found (check the App install)"
        )
    if resp.status_code >= 400:
        raise GitHubAuthError(
            f"GitHub token endpoint {resp.status_code}: {resp.text[:200]}"
        )

    body = resp.json()
    token = body["token"]
    expires_at = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
    _token_cache[cache_key] = (token, expires_at)
    return token, expires_at


__all__ = ["mint_installation_token"]
