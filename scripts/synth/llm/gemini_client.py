"""GeminiClient — google-genai async client implementing LlmClientProtocol."""

from __future__ import annotations

import json
import re
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse


def _safe_parse_json(raw: str) -> dict:
    """Parse Gemini's JSON output, attempting minor repair on common quirks.

    Despite ``response_mime_type='application/json'`` Gemini occasionally
    returns:

    1. Trailing commas before `}` or `]` (`{"a": 1,}`).
    2. Trailing commentary or markdown after the JSON value
       (e.g. ``{...}\n```\nNote: ...``).
    3. Truncated output when the response approaches max_output_tokens —
       the JSON ends mid-key or mid-value.

    For (1) and (2) we can recover. For (3) we can sometimes recover by
    closing unclosed structures, but the parsed object will be missing
    fields and downstream Pydantic validation will reject it cleanly —
    which is the right behavior (the operator should bump max_tokens
    or switch to a more capable model).

    Falls through to a clearer error than ``json.JSONDecodeError`` if
    repair fails, including a snippet of the raw response so the
    operator can see what came back.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    s = raw.strip()

    # Strip Markdown fences if present (```json ... ```).
    if s.startswith("```"):
        s = s.lstrip("`")
        if s.startswith("json"):
            s = s[4:]
        s = s.lstrip("\n")
    if s.endswith("```"):
        s = s.rstrip("`").rstrip()

    # Truncate any text after the last balanced top-level } or ].
    # Walk the string tracking depth + in-string state.
    depth = 0
    in_string = False
    escape = False
    last_balanced_end: int | None = None
    for i, ch in enumerate(s):
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            depth += 1
        elif ch in "}]":
            depth -= 1
            if depth == 0:
                last_balanced_end = i + 1
            elif depth < 0:
                # Stray close-bracket: stop here, take what we have.
                break
    if last_balanced_end is not None and last_balanced_end < len(s):
        s = s[:last_balanced_end]

    # Strip trailing commas before } or ] — `,}` and `,]` (with optional whitespace).
    s = re.sub(r",(\s*[}\]])", r"\1", s)

    try:
        return json.loads(s)
    except json.JSONDecodeError as exc:
        snippet = raw[:300] + ("..." if len(raw) > 300 else "")
        raise ValueError(
            f"Gemini returned malformed JSON that could not be repaired "
            f"(parse error: {exc}; raw length: {len(raw)}; first 300 chars: {snippet!r}). "
            f"If this recurs, try increasing max_output_tokens (current default 4096 in "
            f"planner.py / 1024 in validator_pass2.py) or switching to a more capable model "
            f"(e.g. gemini-2.5-pro instead of gemini-2.5-flash)."
        ) from exc


def _clean_schema_for_gemini(schema: Any) -> Any:
    """Strip JSON-schema fields that the Gemini API rejects at request-build time.

    Pydantic's ``model_json_schema()`` emits keys that are valid JSON
    Schema Draft 2020-12 but unsupported by Gemini's `response_schema`
    transformer (see google.genai._transformers._raise_for_unsupported_mldev_properties).
    The most common offender is ``additionalProperties`` — Pydantic
    emits it for both closed models (``additionalProperties: false``)
    and ``dict[str, T]`` fields (``additionalProperties: {<schema>}``).
    Both raise ValueError from the SDK before any HTTP call happens.

    This cleaner walks the schema recursively and strips ``additionalProperties``
    everywhere. The semantic loss is small: Gemini will accept extra keys
    in returned objects (as if additionalProperties were unset, which is
    the JSON Schema default), and dict[str, T] fields lose value-type
    enforcement at the schema level — downstream Pydantic validation in
    the caller still catches type mismatches on parse.

    Add other unsupported keys here if they bite. Known-unsupported (per
    google-genai source as of 2026-05): ``additionalProperties``. ``title``,
    ``default``, ``$defs``, and ``$ref`` are accepted by the transformer.
    """
    if isinstance(schema, dict):
        return {
            k: _clean_schema_for_gemini(v)
            for k, v in schema.items()
            if k != "additionalProperties"
        }
    if isinstance(schema, list):
        return [_clean_schema_for_gemini(item) for item in schema]
    return schema


class GeminiClient:
    """google-genai backed client implementing LlmClientProtocol."""

    def __init__(self, api_key: str) -> None:
        self._client = genai.Client(api_key=api_key)

    async def generate(self, req: LlmRequest) -> LlmResponse:
        config = GenerateContentConfig(
            system_instruction=req.system or None,
            max_output_tokens=req.max_tokens,
            temperature=req.temperature,
        )
        response = await self._client.aio.models.generate_content(
            model=req.model,
            contents=req.prompt,
            config=config,
        )
        return LlmResponse(text=response.text or "")

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        """Use Gemini's native structured-output mode via response_schema.

        We pass a CLEANED schema dict (not the Pydantic class directly)
        so that fields the Gemini API rejects — e.g. ``additionalProperties``
        — are stripped before the SDK's request-build transformer
        validates them. See ``_clean_schema_for_gemini``.
        """
        cleaned_schema = _clean_schema_for_gemini(schema.model_json_schema())
        config = GenerateContentConfig(
            system_instruction=req.system or None,
            max_output_tokens=req.max_tokens,
            temperature=req.temperature,
            response_mime_type="application/json",
            response_schema=cleaned_schema,
        )
        response = await self._client.aio.models.generate_content(
            model=req.model,
            contents=req.prompt,
            config=config,
        )
        raw = response.text
        if not raw:
            raise ValueError("Gemini returned empty response.text for structured output")
        return _safe_parse_json(raw)

    async def close(self) -> None:
        # google-genai Client does not expose an async close method in current SDK.
        pass
