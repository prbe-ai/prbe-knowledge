"""Per-tenant LiteLLM virtual key fetching + caching.

Background
----------
`shared/llm.py` already routes every LiteLLM call through the central
LiteLLM gateway when ``LLM_GATEWAY_URL`` is set, authenticated with a
single process-wide ``LLM_GATEWAY_KEY``. That's fine for self-host
(one virtual key per pod) but loses per-tenant attribution on the
**shared-managed** data plane where one process handles many tenants.

This module bridges that gap:

  - `get_tenant_virtual_key(customer_id)` fetches the customer's LiteLLM
    virtual key from the control plane and caches it per-customer with
    a TTL (default 5 minutes).
  - `tenant_virtual_key_context(key)` binds a key onto a context var for
    the duration of an `async with` block.
  - `current_tenant_virtual_key()` reads the bound key; `shared/llm.py`
    consults this on every call and prefers it over ``LLM_GATEWAY_KEY``.

Why a contextvar (not a thread-local, not a kwarg)
--------------------------------------------------
LLM call sites are deep — router → triage → providers → forced_tool_call
→ shared.llm.acompletion. Threading a `customer_id` (or pre-fetched key)
through every layer is a sprawling diff. ContextVars propagate through
`asyncio` automatically, so the scope is one decorator/context-manager
at the request/worker-task entrypoint and zero changes everywhere else.

Why fetch from the control plane (not from a DB column on `customers`)
----------------------------------------------------------------------
The control plane owns LiteLLM virtual-key lifecycle (create/rotate/
revoke). Mirroring the key into a data-plane column would mean two
sources of truth and a key-rotation race. The control plane endpoint is
the source of truth; we cache for latency, not authority.

In-cluster trust
----------------
The control-plane endpoint is gated by the shared
``INTERNAL_BACKEND_API_KEY`` secret, sent as ``X-Internal-Backend-Key``
(same header `shared/backend_client.py` uses for the GitHub
installation-token endpoint).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar

import httpx

from engine.shared.config import get_settings
from engine.shared.logging import get_logger

log = get_logger(__name__)

__all__ = [
    "LiteLLMKeyUnavailable",
    "current_tenant_virtual_key",
    "get_tenant_virtual_key",
    "invalidate_tenant_virtual_key",
    "tenant_virtual_key_context",
]

# Cache TTL: short enough to pick up rotations within ~5min, long enough
# to keep the control-plane RTT off every LLM call.
_CACHE_TTL_SECONDS = 300.0

# Per-call HTTP timeout (control-plane fetch is in-cluster).
_FETCH_TIMEOUT_SECONDS = 5.0

# ContextVar holding the *current* tenant's virtual key. ``None`` means
# "no per-tenant override — fall through to the process-wide
# ``LLM_GATEWAY_KEY``". Importantly, ContextVars propagate through
# ``asyncio.create_task`` automatically, so a worker that fans out to
# sub-tasks doesn't have to re-bind.
_CURRENT_KEY: ContextVar[str | None] = ContextVar(
    "prbe_litellm_tenant_virtual_key", default=None
)

# In-memory cache: customer_id -> (virtual_key, fetched_at_monotonic).
# Process-local — each replica fetches independently. That's fine: the
# cardinality is "active tenants per worker" (small), and a brief
# inconsistency window after a key rotation is acceptable.
_KEY_CACHE: dict[str, tuple[str, float]] = {}


class LiteLLMKeyUnavailable(Exception):
    """Raised when the control plane cannot return a key for the tenant.

    Distinct from a transport error — this means the control plane
    explicitly does not have a key for this customer (404, or the
    endpoint is unconfigured). Callers may want to degrade gracefully
    (use process-wide key, or surface as a user-visible error).
    """


def current_tenant_virtual_key() -> str | None:
    """Return the LiteLLM virtual key bound to the current async context,
    or ``None`` if no per-tenant key is active.

    ``shared.llm._maybe_inject_gateway`` consults this on every call and
    prefers it over the process-wide ``LLM_GATEWAY_KEY`` env var. A
    ``None`` return means "use whatever default is configured."
    """
    return _CURRENT_KEY.get()


def invalidate_tenant_virtual_key(customer_id: str) -> None:
    """Drop the cached virtual key for ``customer_id``.

    Call this after a 401/403 from the LiteLLM proxy — the cached key
    may have been rotated. Next ``get_tenant_virtual_key`` re-fetches
    from the control plane.
    """
    _KEY_CACHE.pop(customer_id, None)


async def get_tenant_virtual_key(
    customer_id: str,
    *,
    http: httpx.AsyncClient | None = None,
) -> str:
    """Fetch (or read from cache) the LiteLLM virtual key for ``customer_id``.

    Hits the control plane endpoint
    ``GET {backend_base_url}/routing/customer/{customer_id}/litellm-key``
    with ``X-Internal-Backend-Key`` set to ``INTERNAL_BACKEND_API_KEY``. The
    response body shape is ``{"litellm_key": "<sk-...>", ...}``.

    Parameters
    ----------
    customer_id : str
        The tenant whose key to fetch. Must be non-empty.
    http : httpx.AsyncClient | None
        Optional explicit client (tests inject a ``MockTransport`` here).
        If ``None``, a short-lived client is created per call.

    Returns
    -------
    str
        The bearer-token-shaped virtual key to set as LiteLLM ``api_key``.

    Raises
    ------
    LiteLLMKeyUnavailable
        Control plane returned 404, or required config is missing.
    httpx.HTTPError
        On transport failure (caller decides retry policy).
    """
    if not customer_id:
        raise LiteLLMKeyUnavailable("customer_id is required")

    cached = _KEY_CACHE.get(customer_id)
    now = time.monotonic()
    if cached is not None and (now - cached[1]) < _CACHE_TTL_SECONDS:
        return cached[0]

    settings = get_settings()
    base = (settings.backend_base_url or "").rstrip("/")
    api_key = (
        settings.internal_backend_api_key.get_secret_value()
        if settings.internal_backend_api_key
        else ""
    )
    if not base or not api_key:
        raise LiteLLMKeyUnavailable(
            "BACKEND_BASE_URL or INTERNAL_BACKEND_API_KEY is not configured"
        )

    url = f"{base}/routing/customer/{customer_id}/litellm-key"
    headers = {"X-Internal-Backend-Key": api_key}

    async def _do_fetch(client: httpx.AsyncClient) -> str:
        resp = await client.get(url, headers=headers, timeout=_FETCH_TIMEOUT_SECONDS)
        if resp.status_code == 404:
            raise LiteLLMKeyUnavailable(
                f"control plane has no LiteLLM key for customer {customer_id}"
            )
        if resp.status_code >= 400:
            raise LiteLLMKeyUnavailable(
                f"control plane returned {resp.status_code}: {resp.text[:200]}"
            )
        body = resp.json()
        key = body.get("litellm_key") or body.get("key")
        if not key or not isinstance(key, str):
            raise LiteLLMKeyUnavailable(
                f"control plane response missing 'litellm_key' for {customer_id}"
            )
        return key

    if http is not None:
        key = await _do_fetch(http)
    else:
        async with httpx.AsyncClient() as client:
            key = await _do_fetch(client)

    _KEY_CACHE[customer_id] = (key, now)
    log.debug(
        "litellm_key.fetched",
        customer_id=customer_id,
        cache_size=len(_KEY_CACHE),
    )
    return key


@asynccontextmanager
async def tenant_virtual_key_context(
    customer_id: str,
    *,
    http: httpx.AsyncClient | None = None,
) -> AsyncIterator[str]:
    """Bind the tenant's LiteLLM virtual key for the duration of the block.

    Usage at a request/worker entrypoint::

        async with tenant_virtual_key_context(customer_id):
            # Every shared.llm.acompletion() inside this block uses the
            # tenant's virtual key — per-tenant cost attribution kicks in
            # at the LiteLLM proxy.
            await run_retrieval(customer_id, query)

    The bound key is automatically cleared on exit (success or exception).
    Yields the resolved key string for callers that want to inspect it
    (e.g. for logging).
    """
    key = await get_tenant_virtual_key(customer_id, http=http)
    token = _CURRENT_KEY.set(key)
    try:
        yield key
    finally:
        _CURRENT_KEY.reset(token)
