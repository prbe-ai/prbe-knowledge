"""LiteLLM-backed provider abstraction (Phase 0a, plan D1).

Why this exists
---------------
Today the repo calls Anthropic, OpenAI and Google via three different
SDKs (see `docs/llm-migration-inventory.md` for the full call-site
table). Phase 0 of the managed-shared / self-host split requires:

  1. **Self-host customers bring their own LLM keys** and route through
     their own LiteLLM proxy via ``LLM_GATEWAY_URL``. We never see the
     traffic.
  2. **Managed-mode pricing** needs uniform per-call token accounting
     (queries metered, tokens passthrough+markup) — easier to do at one
     wrapper than three SDK shapes.

This module is the wrapper. It has TWO callable surfaces:

  - `acompletion(model, messages, **kwargs)` — thin async wrapper around
    `litellm.acompletion`.
  - `aembedding(model, input, **kwargs)` — thin async wrapper around
    `litellm.aembedding`.

It also exposes one error class, `LLMError`, that wraps every LiteLLM
exception so call sites never have to import LiteLLM internals.

Phase boundary
--------------
Phase 0a (this PR): wrapper + tests + inventory. No call sites change.
Phase 0b (next PR): migrate the call sites in
`docs/llm-migration-inventory.md` to use this module.

Model strings
-------------
LiteLLM accepts both *provider-prefixed* names (the recommended form) and
bare model ids when the SDK can infer the provider. The prefixed form is
preferred because it's unambiguous:

  - Anthropic   -> ``anthropic/claude-sonnet-4-6``  (or any Anthropic id)
  - OpenAI      -> ``openai/text-embedding-3-large`` /
                   ``openai/gpt-4o-mini`` etc.
  - Google      -> ``gemini/gemini-3-flash-preview`` /
                   ``gemini/gemini-embedding-2``

Bare ids that are unambiguous (e.g. ``text-embedding-3-large`` ->
OpenAI by SDK convention, ``claude-sonnet-4-6`` -> Anthropic) work too,
but Phase-0b call-site migrations should adopt the explicit prefix to
keep routing intent legible.

Per the gemini-3-flash-preview pin in plan D1 + memory
``project_gemini_dedupe_model_id``: the dedupe judge model id is fixed.
LiteLLM resolves ``gemini/gemini-3-flash-preview`` to the exact same
underlying Google API call shape; we do NOT add fallbacks or aliases
that could silently downgrade the model.

Gateway routing
---------------
If the env var ``LLM_GATEWAY_URL`` is set, every call (completion AND
embedding) is forwarded to that URL via LiteLLM's ``api_base`` parameter,
authenticated with ``LLM_GATEWAY_KEY`` (LiteLLM's ``api_key``) when that
env var is also set. Two modes use this:

  - **managed-shared**: the central LiteLLM proxy on the cluster; the
    per-tenant virtual key is bound by ``tenant_virtual_key_context``
    (see ``shared/litellm_key.py``) and wins over ``LLM_GATEWAY_KEY``,
    which remains as a process-wide fallback for bootstrap/cron calls
    that run without tenant context.
  - **self-host**: the customer runs their own LiteLLM / OpenAI-compatible
    gateway and we point at it with whatever key they configure.

Without ``LLM_GATEWAY_URL``, LiteLLM uses each provider's default
endpoint and picks credentials out of the standard env vars
(``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ``GOOGLE_API_KEY``) — the
dev / self-host-with-own-keys path. A
``LLM_GATEWAY_KEY`` without a ``LLM_GATEWAY_URL`` is ignored. A per-call
``api_base`` / ``api_key`` kwarg always wins over the env.
"""

from __future__ import annotations

import os
from typing import Any

# Importing litellm at module load is intentional: the abstraction is
# used on the hot retrieval path. A lazy import would push the
# ~1.5s litellm warm-up onto the first user request. The cost lives at
# process start where it belongs.
#
# `litellm` exposes its async surface as `acompletion` / `aembedding`.
# Exception classes live under `litellm.exceptions`. Bare `litellm`
# attributes (`drop_params`, `set_verbose`) are configuration knobs we
# leave at defaults for now; tuning lives in `shared/constants.py` per
# `feedback_prbe_knowledge_tuning_consts`.
#
# LiteLLM does NOT expose a single umbrella exception class — its
# hierarchy re-uses OpenAI's exception names + an `OpenAIError` root.
# We catch the broadest base (`OpenAIError`) for anything LiteLLM raises
# from a provider call, plus a generic `Exception` fallback for
# library-internal errors that don't subclass it.
import litellm
from litellm.exceptions import OpenAIError as _LiteLLMBaseError

__all__ = [
    "LLMError",
    "acompletion",
    "aembedding",
    "gateway_key",
    "gateway_url",
]


# Env vars read at every call (NOT cached at import) so a test or the
# self-host installer can flip them after the module is imported.
_GATEWAY_URL_ENV = "LLM_GATEWAY_URL"
_GATEWAY_KEY_ENV = "LLM_GATEWAY_KEY"


def gateway_url() -> str | None:
    """Return the configured LLM gateway URL, or None for direct calls.

    Reads ``LLM_GATEWAY_URL`` from the environment on every call (not
    cached). Empty string is treated as unset so a customer who clears
    the var at runtime falls back to direct provider calls without a
    process restart.
    """
    val = os.environ.get(_GATEWAY_URL_ENV)
    return val if val else None


def gateway_key() -> str | None:
    """Return the bearer key for the LLM gateway (``LLM_GATEWAY_KEY``),
    or None. Read fresh every call, like :func:`gateway_url`. Only
    meaningful when :func:`gateway_url` is also set — a key without a
    gateway URL is ignored (there's nothing to send it to)."""
    val = os.environ.get(_GATEWAY_KEY_ENV)
    return val if val else None


class LLMError(Exception):
    """Stable wrapper around LiteLLM's exception hierarchy.

    Call sites should `except LLMError` and never have to know about
    litellm internals (its exception hierarchy is wide and shifts
    across releases). The original LiteLLM exception is preserved on
    ``__cause__`` for callers that need provider-specific handling.

    Attributes
    ----------
    status_code : int | None
        HTTP-style status if LiteLLM exposed one (rate-limit, auth
        failure, server error). ``None`` for client-side / library
        errors.
    provider : str | None
        Provider name LiteLLM was routing to when the call failed.
        Useful for logging / observability.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider = provider


def _wrap_litellm_error(exc: BaseException) -> LLMError:
    """Translate a LiteLLM exception into the stable ``LLMError`` shape.

    Pulls status_code / llm_provider off the LiteLLM exception when
    present (``litellm.exceptions.*`` carry both fields on most
    subclasses). Falls back to ``str(exc)`` for the message.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, str) and status.isdigit():
        status = int(status)
    if not isinstance(status, int):
        status = None
    provider = getattr(exc, "llm_provider", None) or getattr(exc, "provider", None)
    if not isinstance(provider, str):
        provider = None
    return LLMError(str(exc), status_code=status, provider=provider)


def _maybe_inject_gateway(kwargs: dict[str, Any]) -> dict[str, Any]:
    """If ``LLM_GATEWAY_URL`` is set and the caller didn't override
    ``api_base``, inject it; and inject ``api_key`` (in precedence order:
    per-tenant virtual key bound on a ContextVar > ``LLM_GATEWAY_KEY``
    env var) when neither is None and the caller didn't override
    ``api_key``. Caller-supplied ``api_base`` / ``api_key`` always win,
    so per-call overrides remain possible (the eval harness pointing at
    a staging gateway, or an explicit direct provider call).

    A gateway key is only forwarded when ``LLM_GATEWAY_URL`` is set — a
    stray key with no URL would otherwise override the direct-provider
    credentials LiteLLM picks up from ``ANTHROPIC_API_KEY`` /
    ``OPENAI_API_KEY`` / ``GOOGLE_API_KEY`` and break the dev path.

    Per-tenant precedence (shared-managed data plane)
    -------------------------------------------------
    ``shared.litellm_key.current_tenant_virtual_key()`` returns the
    LiteLLM virtual key bound to the current async context by
    ``tenant_virtual_key_context(customer_id)``. When set, it wins over
    ``LLM_GATEWAY_KEY``: the env var is a process-wide fallback for
    self-host (1 tenant per pod) and bootstrap/cron calls without tenant
    context; the contextvar is the per-request override for shared-managed
    (N tenants per pod).
    """
    url_already_set = bool(kwargs.get("api_base"))
    if not url_already_set:
        url = gateway_url()
        if url:
            kwargs["api_base"] = url
            url_already_set = True
    if url_already_set and not kwargs.get("api_key"):
        # Per-tenant override first (shared-managed); env-var fallback
        # second (self-host / no-tenant-context). Lazy-import the helper so a
        # circular import (litellm_key imports config; config might one
        # day import llm) is impossible.
        from engine.shared.litellm_key import current_tenant_virtual_key

        key = current_tenant_virtual_key() or gateway_key()
        if key:
            kwargs["api_key"] = key
    return kwargs


async def acompletion(
    model: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> Any:
    """Async chat-completion via LiteLLM.

    Parameters
    ----------
    model : str
        LiteLLM model id. Use the provider-prefixed form
        (``anthropic/claude-sonnet-4-6``, ``openai/gpt-4o-mini``,
        ``gemini/gemini-3-flash-preview``) when possible. Bare ids work
        too where unambiguous.
    messages : list[dict]
        OpenAI chat-completion messages shape:
        ``[{"role": "system" | "user" | "assistant" | "tool", "content": ...}]``.
        LiteLLM normalizes this into each provider's native shape
        (Anthropic ``messages.create``, Gemini ``generate_content``,
        OpenAI ``chat.completions.create``).
    **kwargs
        Forwarded verbatim to ``litellm.acompletion``. Common keys:
        ``max_tokens``, ``temperature``, ``tools``, ``tool_choice``,
        ``stream``, ``response_format``, ``api_base`` (per-call gateway
        override), ``timeout``.

    Returns
    -------
    The provider-normalized response object that ``litellm.acompletion``
    returns. The shape mirrors OpenAI's ChatCompletion: ``.choices[0]
    .message.content``, ``.choices[0].message.tool_calls``,
    ``.usage.prompt_tokens`` / ``.usage.completion_tokens`` etc. See
    https://docs.litellm.ai/docs/completion/output for the full schema.

    Raises
    ------
    LLMError
        Wraps any LiteLLM exception. Inspect ``status_code`` / ``provider``
        for routing decisions; ``__cause__`` carries the original error.

    Gateway transport
    -----------------
    No global ``custom_llm_provider`` injection here (unlike
    ``aembedding`` which forces ``"openai"``). Chat-completion callers
    vary in what provider-native kwargs they pass — e.g. the gatherer
    (Fireworks gpt-oss-120B) needs the OpenAI wire shape so
    ``response_format`` survives, but the synthesizer (Gemini) passes
    ``reasoning_effort`` which OpenAI rejects with
    ``UnsupportedParamsError``. Forcing ``"openai"`` universally
    breaks Gemini callers.

    Callers that need a specific wire shape pass
    ``custom_llm_provider="..."`` in their own call kwargs (see
    ``services/retrieval/agent/loop.py`` for the gatherer's reasoning).
    Callers that need the provider-native wire shape pass nothing and
    let LiteLLM pick from the model prefix.
    """
    kwargs = _maybe_inject_gateway(kwargs)
    try:
        return await litellm.acompletion(model=model, messages=messages, **kwargs)
    except _LiteLLMBaseError as exc:
        raise _wrap_litellm_error(exc) from exc
    except Exception as exc:
        # Belt-and-suspenders: some LiteLLM error paths re-raise
        # provider-native or transport exceptions that don't subclass
        # OpenAIError (the abstraction is leaky in places — httpx
        # timeouts, JSON parse errors, etc). Wrapping here keeps the
        # call-site contract.
        raise _wrap_litellm_error(exc) from exc


async def aembedding(
    model: str,
    input: list[str],
    **kwargs: Any,
) -> Any:
    """Async embedding via LiteLLM.

    Parameters
    ----------
    model : str
        LiteLLM model id. Examples:
        ``openai/text-embedding-3-large``,
        ``gemini/gemini-embedding-2``.
    input : list[str]
        The texts to embed. LiteLLM batches these into a single
        provider call where the provider supports it.
    **kwargs
        Forwarded to ``litellm.aembedding``. Provider-specific knobs
        (e.g. ``dimensions`` for OpenAI's ``text-embedding-3-*``,
        Gemini's ``task_type`` if supported) pass through.

    Returns
    -------
    The OpenAI-shaped embedding response: ``.data[i].embedding`` is a
    ``list[float]`` for input ``i``; ``.usage.prompt_tokens`` carries
    the input token count.

    Raises
    ------
    LLMError
        See ``acompletion``.

    Gateway transport
    -----------------
    When ``LLM_GATEWAY_URL`` is set, embedding calls route through the
    central LiteLLM proxy. The proxy is OpenAI-compatible on
    ``POST /embeddings`` and resolves the requested model id via its own
    ``model_list`` (so a ``gemini-embedding-*`` request lands on Gemini
    upstream). The LiteLLM SDK, however, picks its wire format from the
    model's provider prefix: a ``gemini/...`` model would otherwise build
    the Gemini-native URL ``/v1beta/models/<m>:batchEmbedContents`` and
    POST that against the proxy — which doesn't serve that path and
    answers FastAPI 405 ``{"detail":"Method Not Allowed"}``. We force
    ``custom_llm_provider="openai"`` so the SDK uses the OpenAI wire
    shape against the proxy regardless of model prefix. Routing to the
    real upstream happens inside the proxy.

    Scope of the override: ONLY when ``_maybe_inject_gateway`` injects
    ``api_base`` from ``LLM_GATEWAY_URL``. A caller who passes an
    explicit ``api_base`` (e.g. pointing at a non-LiteLLM provider's
    OpenAI-compatible endpoint, or at the Google Vertex AI gateway with
    different wire semantics) is trusted to also pick the right
    ``custom_llm_provider`` and gets no override.
    """
    caller_set_api_base = "api_base" in kwargs
    kwargs = _maybe_inject_gateway(kwargs)
    gateway_injected = not caller_set_api_base and "api_base" in kwargs
    if gateway_injected and "custom_llm_provider" not in kwargs:
        kwargs["custom_llm_provider"] = "openai"
    try:
        return await litellm.aembedding(model=model, input=input, **kwargs)
    except _LiteLLMBaseError as exc:
        raise _wrap_litellm_error(exc) from exc
    except Exception as exc:
        raise _wrap_litellm_error(exc) from exc
