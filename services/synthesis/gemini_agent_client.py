"""GeminiAgentClient — adapts google-genai to the AgentLoop's _LLMClient surface.

The harness expects two methods:

    create_cache(*, system_instruction, tools, seed_contents) -> str
    generate_with_cache(*, cache_name, contents, tools) -> dict

This module wraps the production Gemini SDK with that contract. Stays
out of the unit-test path (tests pass their own stub client into
SynthesisWorker via llm_client=...).

Phase-0b carve-out (managed-isolated / self-host)
-------------------------------------------------
This call site is the **one production call site that does NOT migrate
to `shared.llm.acompletion`** in Phase 0b. Two reasons:

  1. **LiteLLM does not expose Gemini's `CachedContent` API.** The agent
     loop relies on a per-drain cache (system prompt + wiki index +
     manifest seed, ~10K-100K tokens) that's re-used across up to 200
     turns. Dropping the cache and re-sending the seed every turn would
     multiply Gemini input-cost by ~200x — material at fleet scale.
  2. **Gemini 3.x request shape is not OpenAI-compatible.** The harness
     round-trips Gemini-native `function_call` / `function_response`
     parts AND the opaque `thought_signature` bytes the API requires on
     every echoed function_call. LiteLLM normalizes tools to OpenAI's
     `tool_calls` shape — `thought_signature` has no slot in that shape,
     so multi-turn agent runs would 400 on turn 2 (this is the same
     bug `agent.gemini_persistent_error` flagged before AFC was
     disabled, but with no clean workaround inside LiteLLM).

So `GeminiAgentClient` keeps the direct google-genai SDK. The
trade-off: **on managed tenants (where LLM_GATEWAY_URL is set), the
data-plane pod has no GOOGLE_API_KEY**, so wiki-agent runs cannot
execute. Constructing this client in gateway mode raises a clear
RuntimeError — surfaces loudly, doesn't silently downgrade to a
broken request.

TODO(phase-0b-cached-content): re-enable on managed tenants when
either (a) LiteLLM exposes Gemini's `caches.create` / `cached_content`
lifecycle plus `thought_signature` passthrough, or (b) the central
LiteLLM proxy gets per-customer GCP credentials so the call can route
through the proxy with provider-native shape. Until then, wiki-agent
is a direct-keys-only feature.

Critical Gemini constraint (caused v4 first-turn halt):

    When a request sets `cached_content`, it MUST NOT also set
    `system_instruction`, `tools`, or `tool_config`. The API rejects
    that combination with a 400 ("CachedContent can not be used with
    GenerateContent request setting system_instruction, tools or
    tool_config"); the SDK surfaces it as a ValueError.

    Tools/system_instruction live on the cache itself (they're set at
    create_cache time). The per-call config only carries the *new*
    user turn and the cache pointer.

Second constraint: `contents` must be non-empty. On turn 0 the
harness's conversation tail is `[]`, so we send a minimal nudge so
the model takes its first turn against the cached seed.

Reference shape (google-genai==1.x):

    client = google.genai.Client(api_key=...)
    cache = await client.aio.caches.create(
        model="gemini-3.1-pro-preview",
        config=CreateCachedContentConfig(
            contents=[...],
            system_instruction=...,
            tools=[Tool(function_declarations=[...])],
            ttl="3600s",
        ),
    )
    # WITH cache: tools/system_instruction omitted from per-call config.
    resp = await client.aio.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[...],   # non-empty
        config=GenerateContentConfig(cached_content=cache.name),
    )
"""

from __future__ import annotations

from typing import Any

from shared import llm as shared_llm
from shared.config import get_settings
from shared.constants import WIKI_AGENT_CACHE_TTL, WIKI_AGENT_MODEL
from shared.logging import get_logger

log = get_logger(__name__)


# Schema keys Gemini's strict OpenAPI subset rejects. Mirrors
# `services/synthesis/providers.py:_GEMINI_REJECTED_SCHEMA_KEYS` so
# any tool schema we hand to FunctionDeclaration is sanitized the
# same way response_schemas are. `additionalProperties` and `$ref`
# are the dangerous ones; the others are belt-and-suspenders for
# older SDK versions that were stricter.
_GEMINI_REJECTED_SCHEMA_KEYS: tuple[str, ...] = (
    "additionalProperties",
    "$ref",
    "$schema",
)


# Nudge message used when the harness's conversation tail is empty
# (turn 0 with a cache). The SDK requires `contents` to be non-empty
# even when the entire context lives in the cache.
_TURN_ZERO_NUDGE = "Begin the drain. Use the cached wiki index and manifest to decide your first action."


def _strip_keys_recursive(value: Any, keys: tuple[str, ...]) -> Any:
    """Recursively drop dict keys Gemini's Schema validator rejects."""
    if isinstance(value, dict):
        return {
            k: _strip_keys_recursive(v, keys) for k, v in value.items() if k not in keys
        }
    if isinstance(value, list):
        return [_strip_keys_recursive(v, keys) for v in value]
    return value


def _sanitize_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Return a Gemini-safe copy of a tool's `parameters` schema.

    Empty / None becomes `{"type": "object", "properties": {}}` so
    FunctionDeclaration always has a valid schema.
    """
    if not parameters:
        return {"type": "object", "properties": {}}
    return _strip_keys_recursive(parameters, _GEMINI_REJECTED_SCHEMA_KEYS)


class GeminiAgentClient:
    """Production wrapper around google-genai for the wiki agent loop."""

    def __init__(self, *, model: str = WIKI_AGENT_MODEL) -> None:
        self._model = model
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        # Phase-0b carve-out gate: in managed-isolated / self-host mode
        # the data plane has no GOOGLE_API_KEY (provider creds live in
        # the central LiteLLM proxy). Since CachedContent + thought-
        # signature round-tripping don't migrate through LiteLLM today
        # (see module docstring), refuse to construct the client rather
        # than emit a misleading "GOOGLE_API_KEY not configured" — that
        # error implies a config fix exists. The carve-out doesn't.
        if shared_llm.gateway_url():
            raise RuntimeError(
                "GeminiAgentClient is not available in LLM_GATEWAY_URL "
                "(managed-isolated / self-host) mode — Gemini CachedContent "
                "and thought_signature round-tripping have no LiteLLM "
                "equivalent. The wiki-agent loop is direct-keys-only "
                "until either LiteLLM exposes Gemini caches or the proxy "
                "gets GCP credentials. See module docstring for the full "
                "rationale + TODO(phase-0b-cached-content)."
            )
        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai not installed; cannot use GeminiAgentClient"
            ) from exc
        secret = get_settings().google_api_key
        api_key = secret.get_secret_value() if secret is not None else ""
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not configured for GeminiAgentClient")
        self._client = genai.Client(api_key=api_key)
        return self._client

    async def create_cache(
        self,
        *,
        system_instruction: str,
        tools: list[dict[str, Any]],
        seed_contents: list[dict[str, Any]],
    ) -> str:
        client = self._ensure_client()
        from google.genai.types import (
            CreateCachedContentConfig,
            FunctionDeclaration,
            Tool,
        )

        function_decls = [
            FunctionDeclaration(
                name=t["name"],
                description=t.get("description"),
                parameters=_sanitize_parameters(t.get("parameters")),
            )
            for t in tools
        ]
        cache = await client.aio.caches.create(
            model=self._model,
            config=CreateCachedContentConfig(
                contents=seed_contents,
                system_instruction=system_instruction,
                tools=[Tool(function_declarations=function_decls)],
                ttl=WIKI_AGENT_CACHE_TTL,
            ),
        )
        return getattr(cache, "name", "") or ""

    async def generate_with_cache(
        self,
        *,
        cache_name: str,
        contents: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Issue one cached `generate_content` call.

        When `cache_name` is set, tools/system_instruction live on the
        cache; the per-call config carries ONLY the cache pointer.
        Gemini rejects requests that set both `cached_content` and
        `tools` (400 -> SDK ValueError -> tenacity exhaustion).

        When `cache_name` is empty (cache creation failed; we logged
        a warning and fell through), we pay full input cost AND have
        to attach tools+system_instruction to every call ourselves.
        """
        client = self._ensure_client()
        from google.genai.types import (
            AutomaticFunctionCallingConfig,
            FunctionDeclaration,
            GenerateContentConfig,
            Tool,
        )

        # Gemini requires non-empty `contents`. Turn 0 with cache has
        # an empty conversation tail (the seed lives in the cache);
        # send a minimal nudge so the model takes its first turn.
        effective_contents: list[Any] = list(contents) if contents else [
            {"role": "user", "parts": [{"text": _TURN_ZERO_NUDGE}]}
        ]
        # AFC must be disabled. The google-genai SDK defaults Automatic
        # Function Calling ON whenever a request mentions functions —
        # including functions resolved through `cached_content`. Our
        # agent harness does manual dispatch (receive function_call ->
        # run tool -> send function_response back). AFC engaging on
        # turn 2+ produces a 400 on every multi-turn call:
        #   turn=1: HTTP 200 OK   (model returns function_call)
        #   turn=2: HTTP 400      (AFC + manual response collide)
        #   tenacity exhausts:    400 / 400 / 400
        #   agent.halt reason=gemini_persistent_error turns=1
        # Disabling AFC keeps the request shape unambiguous and lets
        # the harness own the dispatch loop end-to-end.
        afc_disabled = AutomaticFunctionCallingConfig(disable=True)

        if cache_name:
            config = GenerateContentConfig(
                cached_content=cache_name,
                automatic_function_calling=afc_disabled,
            )
        else:
            function_decls = [
                FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description"),
                    parameters=_sanitize_parameters(t.get("parameters")),
                )
                for t in tools
            ]
            config = GenerateContentConfig(
                tools=[Tool(function_declarations=function_decls)],
                automatic_function_calling=afc_disabled,
            )

        try:
            resp = await client.aio.models.generate_content(
                model=self._model,
                contents=effective_contents,
                config=config,
            )
        except Exception as exc:
            # Surface enough of the SDK's ClientError body to diagnose
            # future Gemini regressions without redeploying for logs.
            log.warning(
                "agent.gemini_call_failed",
                error_class=type(exc).__name__,
                error_message=str(exc)[:500],
                cache_name_set=bool(cache_name),
                conversation_length=len(effective_contents),
            )
            raise
        return _extract_response(resp)


def _extract_response(resp: Any) -> dict[str, Any]:
    """Normalize the SDK's response object into the harness's dict shape.

    The harness expects:
      {
        "text": str | None,
        "tool_calls": [
            {"name": ..., "args": {...}, "thought_signature": bytes | None},
            ...
        ],
        "usage_metadata": {
            "prompt_token_count": int,
            "cached_content_token_count": int,
            "candidates_token_count": int,
        },
      }

    `thought_signature` is required on Gemini 3.x models. The SDK
    attaches it to the part that holds a function_call. When the
    harness echoes the model's prior function_call back as part of
    the conversation history (turn 2+), the SAME thought_signature
    must accompany it. Otherwise Gemini rejects the request:

        400 INVALID_ARGUMENT: 'Function call is missing a
        thought_signature in functionCall parts. This is required
        for tools to work correctly...'

    See: https://ai.google.dev/gemini-api/docs/thought-signatures
    """
    text: str | None = getattr(resp, "text", None)
    tool_calls: list[dict[str, Any]] = []
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            fc = getattr(part, "function_call", None)
            if fc is None:
                continue
            args = getattr(fc, "args", None) or {}
            if not isinstance(args, dict):
                args = dict(args) if hasattr(args, "items") else {}
            thought_sig = getattr(part, "thought_signature", None)
            tool_calls.append(
                {
                    "name": getattr(fc, "name", ""),
                    "args": dict(args),
                    "thought_signature": thought_sig,
                }
            )
    usage = getattr(resp, "usage_metadata", None)
    usage_dict: dict[str, Any] = {}
    if usage is not None:
        usage_dict = {
            "prompt_token_count": getattr(usage, "prompt_token_count", 0) or 0,
            "cached_content_token_count": getattr(
                usage, "cached_content_token_count", 0
            )
            or 0,
            "candidates_token_count": getattr(usage, "candidates_token_count", 0)
            or 0,
        }
    return {
        "text": text,
        "tool_calls": tool_calls,
        "usage_metadata": usage_dict,
    }


__all__ = ["GeminiAgentClient"]
