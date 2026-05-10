"""LiteLLM-backed provider abstraction (Phase 0a, plan D1).

Why this exists
---------------
Today the repo calls Anthropic, OpenAI and Google via three different
SDKs (see `docs/llm-migration-inventory.md` for the full call-site
table). Phase 0 of the managed-isolated / self-host split requires:

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
                   ``gemini/gemini-embedding-2-preview``

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
embedding) is forwarded to that URL via LiteLLM's ``api_base`` parameter.
This is the self-host pattern: customer runs their own LiteLLM proxy
(or an OpenAI-compatible gateway) and we point at it. Without the env
var, LiteLLM uses each provider's default endpoint and picks credentials
out of the standard env vars (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``,
``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``).
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
    "gateway_url",
]


# Env var read at every call (NOT cached at import) so a test or the
# self-host installer can flip it after the module is imported.
_GATEWAY_URL_ENV = "LLM_GATEWAY_URL"


def gateway_url() -> str | None:
    """Return the configured LLM gateway URL, or None for direct calls.

    Reads ``LLM_GATEWAY_URL`` from the environment on every call (not
    cached). Empty string is treated as unset so a customer who clears
    the var at runtime falls back to direct provider calls without a
    process restart.
    """
    val = os.environ.get(_GATEWAY_URL_ENV)
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
    ``api_base``, inject it. Caller-supplied ``api_base`` always wins
    so per-call overrides remain possible (e.g. the eval harness can
    point at a staging gateway without unsetting the global env var).
    """
    if kwargs.get("api_base"):
        return kwargs
    url = gateway_url()
    if url:
        kwargs["api_base"] = url
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
        ``gemini/gemini-embedding-2-preview``.
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
    """
    kwargs = _maybe_inject_gateway(kwargs)
    try:
        return await litellm.aembedding(model=model, input=input, **kwargs)
    except _LiteLLMBaseError as exc:
        raise _wrap_litellm_error(exc) from exc
    except Exception as exc:
        raise _wrap_litellm_error(exc) from exc
