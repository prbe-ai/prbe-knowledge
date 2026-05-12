"""Tool-use call helpers on top of `shared.llm.acompletion`.

Why this exists
---------------
Every Phase-0b tool-use migration target ultimately wants the same
three-step dance:

  1. Build an OpenAI-shaped `tools=[...]` + `tool_choice=...` request that
     forces the model to invoke exactly one named function.
  2. Pull the function call's JSON-string `arguments` out of
     `resp.choices[0].message.tool_calls[0].function.arguments`.
  3. Parse it as JSON.

LiteLLM normalizes Anthropic `tool_use`, Google `function_declarations`,
and OpenAI `function_call` into that single OpenAI-shaped surface. The
call sites (router, triage, claude-code extraction, retrieval synthesis,
directed phrases, inferred edges) all share this pattern — so the parse
+ error-shape lives here, not duplicated five times.

The companion helpers `usage_tokens` and `is_context_overflow` round out
the migration:

  * `usage_tokens` extracts the OpenAI-shaped token counts AND the
    Anthropic prompt-cache fields (which LiteLLM surfaces as private
    attrs on `usage` and via `usage.prompt_tokens_details`). The router
    hot path relies on these surfacing so cache-hit-rate telemetry
    doesn't go dark on migration.

  * `is_context_overflow` is the LiteLLM-shaped equivalent of the
    Anthropic SDK's `BadRequestError` + "prompt is too long" string
    match. The triage split-retry path drives recursion off this
    predicate; under LiteLLM the same 400 comes back as an
    `LLMError(status_code=400)` (or a `ContextWindowExceededError`-
    backed wrapper) whose message carries the provider's phrasing.

Response-shape adapter, one place
---------------------------------
Today the call sites parse three different SDK shapes:

  Anthropic SDK:   block.input  on `tool_use` blocks
  Google SDK:      result.parsed / result.text on `generate_content`
  OpenAI SDK:      already OpenAI-shaped

After Phase-0b they all converge on:

  resp.choices[0].message.tool_calls[0].function.arguments  # JSON string

`forced_tool_call` does that lookup + JSON-parse and raises a single
`LLMError` shape on every failure mode (no tool call emitted, malformed
arguments, etc.).

Tool-schema shape
-----------------
LiteLLM follows OpenAI's `tools` shape verbatim:

    {
      "type": "function",
      "function": {
        "name": "<tool_name>",
        "description": "...",
        "parameters": <json-schema>,
      }
    }

Anthropic's `input_schema` maps 1:1 to OpenAI's `parameters`.

For Google `response_schema` use sites that we want to migrate but the
underlying call is "constrained JSON output" rather than function
calling, see `forced_response_schema` — same idea, but uses
`response_format={"type": "json_schema", "json_schema": ...}` which
LiteLLM translates into Anthropic tool-use / Gemini `response_schema`
per provider. The OpenAI strict-JSON-schema mode passes through
unchanged.
"""

from __future__ import annotations

import re
from typing import Any

import orjson

from shared.llm import LLMError, acompletion

__all__ = [
    "ToolCallParseError",
    "forced_response_schema",
    "forced_tool_call",
    "is_context_overflow",
    "usage_tokens",
]


class ToolCallParseError(LLMError):
    """The model returned but didn't emit the forced tool call we asked
    for (or the arguments weren't parseable JSON).

    Subclass of `LLMError` so call sites that catch the broader type
    still catch this; call sites that want to special-case the
    "model declined the tool" failure mode can catch this directly.
    """


def _build_tool_kwargs(
    *,
    tool_name: str,
    tool_description: str,
    tool_schema: dict[str, Any],
) -> dict[str, Any]:
    """OpenAI-shaped `tools=[...]` + `tool_choice=...` kwargs that force
    exactly one named function call.

    LiteLLM forwards this verbatim to OpenAI and translates it to
    Anthropic `tool_use` / Google `function_declarations`.
    """
    return {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": tool_schema,
                },
            }
        ],
        "tool_choice": {
            "type": "function",
            "function": {"name": tool_name},
        },
    }


async def forced_tool_call(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tool_name: str,
    tool_schema: dict[str, Any],
    tool_description: str = "",
    max_tokens: int | None = None,
    **kwargs: Any,
) -> tuple[dict[str, Any], Any]:
    """Run an `acompletion` that forces one named function call. Return
    the parsed-arguments dict AND the raw LiteLLM response object.

    Parameters
    ----------
    model : str
        LiteLLM model id (provider-prefixed; e.g. ``anthropic/claude-haiku-4-5-20251001``).
    messages : list[dict]
        OpenAI chat-completion messages. Use the OpenAI `system` role —
        LiteLLM translates to Anthropic's separate `system` slot
        automatically when routing to Anthropic.
    tool_name : str
        The function name the model must invoke. We force it via
        ``tool_choice``; if the model emits text-only the helper raises
        `ToolCallParseError` (same failure mode the SDK-shaped callers
        handled before migration).
    tool_schema : dict
        JSON Schema for the function arguments. Maps 1:1 to Anthropic's
        `input_schema` and Google's `function_declarations.parameters`.
    tool_description : str
        Human-readable tool description. Anthropic's reference shows
        this matters for tool-use quality; pass through whatever the
        Anthropic-shaped prompt had.
    max_tokens : int | None
        Forwarded as `max_tokens=` to LiteLLM. None means "let LiteLLM /
        the provider pick the default".
    **kwargs
        Forwarded to `shared.llm.acompletion`. Common keys: ``temperature``,
        ``timeout``, ``api_base`` (per-call gateway override).

    Returns
    -------
    (args_dict, raw_response)
        ``args_dict`` is the JSON-parsed function arguments. The raw
        response is returned alongside so call sites that need usage
        telemetry (`usage_tokens`) or other response metadata don't
        have to re-do the call.

    Raises
    ------
    ToolCallParseError
        Model returned but didn't invoke ``tool_name`` (text-only
        response, wrong tool name, or unparseable JSON arguments).
    LLMError
        Any other LiteLLM/provider error, wrapped by `acompletion`.
    """
    call_kwargs: dict[str, Any] = dict(kwargs)
    call_kwargs.update(
        _build_tool_kwargs(
            tool_name=tool_name,
            tool_description=tool_description,
            tool_schema=tool_schema,
        )
    )
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    resp = await acompletion(model=model, messages=messages, **call_kwargs)
    args = _extract_tool_call_args(resp, tool_name=tool_name)
    return args, resp


def _extract_tool_call_args(resp: Any, *, tool_name: str) -> dict[str, Any]:
    """Pull the JSON-decoded function arguments out of a LiteLLM response.

    LiteLLM normalizes every provider into the OpenAI shape:

        resp.choices[0].message.tool_calls[i].function.{name, arguments}

    ``arguments`` is a JSON string (per OpenAI's spec). Anthropic and
    Gemini natively return dicts; LiteLLM JSON-serializes them on the
    way out so the OpenAI contract is preserved.
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        raise ToolCallParseError(
            f"response had no choices; expected forced tool call to {tool_name!r}"
        )
    message = getattr(choices[0], "message", None)
    tool_calls = getattr(message, "tool_calls", None) or []
    if not tool_calls:
        # Model declined the tool — same failure mode the old code
        # handled by checking for the absence of a tool_use block.
        raise ToolCallParseError(
            f"response had no tool_calls; expected forced call to {tool_name!r}"
        )
    for call in tool_calls:
        fn = getattr(call, "function", None)
        name = getattr(fn, "name", None)
        if name != tool_name:
            continue
        raw_args = getattr(fn, "arguments", None)
        return _parse_arguments(raw_args, tool_name=tool_name)
    # tool_calls present, but none matched — the model invoked a
    # different tool than the one we forced. Surface that distinctly.
    names = [
        getattr(getattr(c, "function", None), "name", "?") for c in tool_calls
    ]
    raise ToolCallParseError(
        f"forced tool {tool_name!r} not in response tool_calls; got {names!r}"
    )


def _parse_arguments(raw: Any, *, tool_name: str) -> dict[str, Any]:
    """Coerce the `function.arguments` field into a dict.

    OpenAI spec says it's a JSON string; LiteLLM follows that, but some
    versions/providers occasionally surface it as a dict already. Be
    tolerant of both.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"tool {tool_name!r} arguments were not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolCallParseError(
                f"tool {tool_name!r} arguments parsed to {type(parsed).__name__}, expected dict"
            )
        return parsed
    if isinstance(raw, str):
        if not raw.strip():
            raise ToolCallParseError(
                f"tool {tool_name!r} arguments were empty"
            )
        try:
            parsed = orjson.loads(raw)
        except orjson.JSONDecodeError as exc:
            raise ToolCallParseError(
                f"tool {tool_name!r} arguments were not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolCallParseError(
                f"tool {tool_name!r} arguments parsed to {type(parsed).__name__}, expected dict"
            )
        return parsed
    raise ToolCallParseError(
        f"tool {tool_name!r} arguments had unexpected type {type(raw).__name__}"
    )


async def forced_response_schema(
    model: str,
    messages: list[dict[str, Any]],
    *,
    schema_name: str,
    schema: dict[str, Any],
    max_tokens: int | None = None,
    strict: bool = True,
    **kwargs: Any,
) -> tuple[dict[str, Any], Any]:
    """Run an `acompletion` that constrains output to a JSON Schema.

    Uses OpenAI's `response_format={"type": "json_schema", ...}` shape;
    LiteLLM forwards it verbatim for OpenAI, translates it into
    Anthropic's tool-use round-trip, and routes Google `response_schema`
    per-provider. The OpenAI strict-mode `strict: true` flag passes
    through to OpenAI; on Gemini/Anthropic LiteLLM treats the schema as
    the structured-output spec.

    Use this for sites that today use Google `response_schema` or
    OpenAI strict json_schema — call sites that today use Anthropic
    forced tool-use should stay on ``forced_tool_call`` (the
    tool-use surface is more reliable on Anthropic per provider docs).

    Returns ``(parsed_dict, raw_response)``. Raises
    ``ToolCallParseError`` if the response content isn't valid JSON.
    """
    call_kwargs: dict[str, Any] = dict(kwargs)
    call_kwargs["response_format"] = {
        "type": "json_schema",
        "json_schema": {
            "name": schema_name,
            "schema": schema,
            "strict": strict,
        },
    }
    if max_tokens is not None:
        call_kwargs["max_tokens"] = max_tokens
    resp = await acompletion(model=model, messages=messages, **call_kwargs)
    content = _extract_content_text(resp)
    if not content:
        raise ToolCallParseError(
            f"response_format=json_schema for {schema_name!r} returned empty content"
        )
    try:
        parsed = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        raise ToolCallParseError(
            f"response_format=json_schema for {schema_name!r} was not valid JSON: {exc}"
        ) from exc
    if not isinstance(parsed, dict):
        raise ToolCallParseError(
            f"response_format=json_schema for {schema_name!r} parsed to "
            f"{type(parsed).__name__}, expected dict"
        )
    return parsed, resp


def _extract_content_text(resp: Any) -> str:
    """Pull `choices[0].message.content` out of a LiteLLM response, or
    an empty string if absent. Centralizes the OpenAI-shape access for
    structured-output paths.
    """
    choices = getattr(resp, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    return ""


# ---------------------------------------------------------------------------
# Usage telemetry — including Anthropic prompt-cache fields
# ---------------------------------------------------------------------------


def usage_tokens(resp: Any) -> dict[str, int]:
    """Extract per-call token counts from a LiteLLM response.

    Returns a dict with keys:
      ``prompt_tokens``        — input tokens billed
      ``completion_tokens``    — output tokens billed
      ``total_tokens``         — sum (provider-reported, may differ)
      ``cache_creation_input_tokens``
                               — Anthropic prompt-cache: tokens that
                                 wrote into a new cache entry. Zero if
                                 the request didn't use cache_control
                                 or the provider was not Anthropic.
      ``cache_read_input_tokens``
                               — Anthropic prompt-cache: tokens served
                                 from cache. Zero if no cache hit.

    All values default to 0 so call-site code can use them in arithmetic
    without None-guards.

    Why the dual-source lookup
    --------------------------
    LiteLLM surfaces Anthropic prompt-cache fields in TWO places
    depending on version + code path:

      1. ``usage._cache_creation_input_tokens`` / ``usage._cache_read_input_tokens``
         — private attrs set in `Usage.__init__` from the Anthropic
         response (current API, see litellm/types/utils.py).
      2. ``usage.prompt_tokens_details.cache_creation_tokens`` /
         ``usage.prompt_tokens_details.cached_tokens``
         — public OpenAI-shaped fields (new API, also populated by
         the same Usage init path).

    Older versions of LiteLLM exposed only (1); current versions
    populate both from the same source. Reading both with a
    sum-of-max-by-source semantic (prefer the public path, fall back
    to private) means cache telemetry survives LiteLLM version
    bumps without app-side breakage — exactly the survival contract
    the router hot path needs.
    """
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        }
    prompt = _coerce_int(getattr(usage, "prompt_tokens", 0))
    completion = _coerce_int(getattr(usage, "completion_tokens", 0))
    total = _coerce_int(getattr(usage, "total_tokens", 0))

    # Public path: usage.prompt_tokens_details.{cache_creation_tokens,cached_tokens}
    details = getattr(usage, "prompt_tokens_details", None)
    pub_create = _coerce_int(getattr(details, "cache_creation_tokens", 0))
    pub_read = _coerce_int(getattr(details, "cached_tokens", 0))

    # Private path: usage._cache_creation_input_tokens, usage._cache_read_input_tokens
    # (LiteLLM exposes these on the Anthropic provider call regardless of
    # whether prompt_tokens_details was populated.)
    priv_create = _coerce_int(getattr(usage, "_cache_creation_input_tokens", 0))
    priv_read = _coerce_int(getattr(usage, "_cache_read_input_tokens", 0))

    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        # Prefer whichever source LiteLLM populated for this version.
        # max() handles "both populated identically" and "only one set".
        "cache_creation_input_tokens": max(pub_create, priv_create),
        "cache_read_input_tokens": max(pub_read, priv_read),
    }


def _coerce_int(val: Any) -> int:
    if isinstance(val, bool):
        return 0
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        return int(val)
    return 0


# ---------------------------------------------------------------------------
# Context-overflow predicate (replaces Anthropic SDK BadRequestError match)
# ---------------------------------------------------------------------------


# Anthropic 400 phrasings we've seen — same set as services/synthesis/triage.py
# `_OVERSIZE_REGEXES`, mirrored here for the LiteLLM-shaped exception path.
# LiteLLM forwards the provider's error text in ``.message`` / ``str(exc)``
# but may also raise a ``ContextWindowExceededError`` subclass; we check
# both the class name and the message.
_OVERFLOW_MESSAGE_REGEXES = (
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"\btokens?\s*>\s*\d+\s*maximum\b", re.IGNORECASE),
    re.compile(r"context[\s_]?(window[\s_]?)?(length\s+)?exceeded", re.IGNORECASE),
    re.compile(r"maximum\s+context\s+length", re.IGNORECASE),
)


def is_context_overflow(exc: BaseException) -> bool:
    """True iff a wrapped LLMError represents a "context window
    exceeded" 400 from the underlying provider.

    Used by the triage split-retry path: a positive verdict halves the
    batch and recurses; a negative verdict propagates the exception
    unchanged. False on any other shape (auth failure, malformed
    schema, transport error, non-LLMError exception).

    Replaces the Anthropic-SDK-specific
    `services/synthesis/triage.py::is_anthropic_oversize_error`. The
    semantics are identical — only the input exception type changed.

    Detection strategy:
      1. Must be an ``LLMError``. Non-LLM exceptions (network blips,
         test asserts) MUST return False so the split-retry doesn't eat
         genuine failures.
      2. Status code 400 is required — overflow surfaces as a 400.
         A 4xx that isn't 400 (auth, rate-limit) is NOT overflow.
      3. The message OR the wrapped-exception class name signals
         overflow. LiteLLM raises ``ContextWindowExceededError`` for
         providers whose error response carries an explicit overflow
         signal; for others we fall back to the message regex.
    """
    if not isinstance(exc, LLMError):
        return False
    if exc.status_code != 400:
        return False
    msg = str(exc)
    if any(rx.search(msg) for rx in _OVERFLOW_MESSAGE_REGEXES):
        return True
    # Fallback: LiteLLM's ContextWindowExceededError class name in the
    # __cause__ chain. Cheaper than importing the class (avoids a hard
    # litellm.exceptions import here just for an isinstance check).
    cause = exc.__cause__
    while cause is not None:
        if type(cause).__name__ == "ContextWindowExceededError":
            return True
        cause = cause.__cause__
    return False
