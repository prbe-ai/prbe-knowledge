"""Thin client for calling prbe-backend's internal endpoints.

Knowledge is downstream of backend for identity/auth concerns: GitHub
App installation tokens are minted by backend so the App private key
only lives in one service. This client wraps the HTTP call.
"""
from __future__ import annotations

from datetime import UTC, datetime

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
    installation_id: str | None = None,
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

            return await mint_installation_token(http, customer_id=customer_id, installation_id=installation_id)
        raise GitHubAuthError(
            "GitHub token minting is not configured: set BACKEND_BASE_URL + INTERNAL_BACKEND_API_KEY "
            "(hosted) or GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY (standalone)"
        )

    base = (settings.backend_base_url or "").rstrip("/")
    api_key = settings.internal_backend_api_key.get_secret_value()
    url = f"{base}/internal/github/installation_token"
    try:
        resp = await http.post(
            url,
            json={"customer_id": customer_id, **({"installation_id": installation_id} if installation_id else {})},
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

    try:
        body = resp.json()
    except ValueError as exc:
        raise GitHubAuthError("backend returned an invalid GitHub token response") from exc
    if installation_id is not None and (
        not isinstance(body, dict)
        or str(body.get("installation_id", "")) != str(installation_id)
    ):
        # Older hosted backends ignore the selector and mint their latest
        # tenant installation. Never use that token for a different passport.
        # A response without its exact binding is equally unverifiable.
        raise GitHubAuthError("backend did not confirm the requested GitHub installation")
    token = body.get("token") if isinstance(body, dict) else None
    raw_expiry = body.get("expires_at") if isinstance(body, dict) else None
    if not isinstance(token, str) or not token.strip() or not isinstance(raw_expiry, str):
        raise GitHubAuthError("backend returned an invalid GitHub token response")
    try:
        expires_at = datetime.fromisoformat(raw_expiry.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GitHubAuthError("backend returned an invalid GitHub token expiry") from exc
    if expires_at.tzinfo is None or expires_at <= datetime.now(UTC):
        raise GitHubAuthError("backend returned an expired GitHub installation token")
    return token, expires_at


__all__ = ["fetch_github_installation_token"]
