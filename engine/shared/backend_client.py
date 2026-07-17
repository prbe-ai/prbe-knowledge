"""Thin client for calling prbe-backend's internal endpoints.

Knowledge is downstream of backend for identity/auth concerns: GitHub
App installation tokens are minted by backend so the App private key
only lives in one service. This client wraps the HTTP call.
"""
from __future__ import annotations

from datetime import datetime

import httpx

from engine.shared.config import Settings, get_settings
from engine.shared.exceptions import GitHubAuthError
from engine.shared.logging import get_logger

log = get_logger(__name__)


def github_mint_path(settings: Settings) -> str | None:
    """Which installation-token mint path the current config selects.

    ``"hosted"`` (prbe-backend mints; both backend settings present),
    ``"standalone"`` (local minting from the self-hoster's own App creds),
    or ``None``. Single source of truth for the branch below and for
    operator tooling (``scripts.github_seed_token``) — keep them from
    drifting.
    """
    base = (settings.backend_base_url or "").rstrip("/")
    api_key = (
        settings.internal_backend_api_key.get_secret_value()
        if settings.internal_backend_api_key
        else ""
    )
    if base and api_key:
        return "hosted"
    if settings.github_app_id and settings.github_app_private_key.get_secret_value():
        return "standalone"
    return None


async def fetch_github_installation_token(
    http: httpx.AsyncClient,
    *,
    customer_id: str,
) -> tuple[str, datetime]:
    """Fetch a fresh GitHub App installation token from prbe-backend.

    Backend handles minting + caching server-side; we just call the
    endpoint each time we need a bearer. Backend's per-installation
    cache (5min safety margin against the 60min token lifetime) means
    repeated calls within ~55 minutes return the same token cheaply.

    Raises GitHubAuthError on any failure mode so call sites that
    previously used `mint_installation_token` don't need to change
    their except clauses.
    """
    settings = get_settings()
    path = github_mint_path(settings)
    if path != "hosted":
        # Standalone (community) mode: no control plane. Mint the installation
        # token locally from the self-hoster's own GitHub App creds, preserving
        # the (token, expires_at) contract so call sites are unchanged.
        if path == "standalone":
            from engine.shared.github_app import mint_installation_token

            return await mint_installation_token(http, customer_id=customer_id)
        raise GitHubAuthError(
            "GitHub tokens unavailable: set BACKEND_BASE_URL + INTERNAL_BACKEND_API_KEY "
            "(hosted) or GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY (standalone)"
        )

    base = (settings.backend_base_url or "").rstrip("/")
    api_key = settings.internal_backend_api_key.get_secret_value()
    url = f"{base}/internal/github/installation_token"
    try:
        resp = await http.post(
            url,
            json={"customer_id": customer_id},
            headers={
                # Canonical header — prbe-backend retired the X-Internal-Key
                # alias when the Fly sunset closed (see
                # apps/data_plane/dependencies/internal.py).
                "X-Internal-Backend-Key": api_key,
                "Content-Type": "application/json",
            },
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        raise GitHubAuthError(f"backend token endpoint unreachable: {exc}") from exc

    if resp.status_code == 404:
        raise GitHubAuthError(f"no GitHub installation for customer {customer_id}")
    if resp.status_code == 503:
        raise GitHubAuthError("backend App credentials not configured (503)")
    if resp.status_code >= 500:
        raise GitHubAuthError(
            f"backend token endpoint {resp.status_code}: {resp.text[:200]}"
        )
    if resp.status_code >= 400:
        raise GitHubAuthError(
            f"backend token endpoint {resp.status_code}: {resp.text[:200]}"
        )

    body = resp.json()
    token = body["token"]
    expires_at = datetime.fromisoformat(body["expires_at"].replace("Z", "+00:00"))
    return token, expires_at


__all__ = ["fetch_github_installation_token"]
