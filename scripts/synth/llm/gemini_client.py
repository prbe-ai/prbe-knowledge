"""GeminiClient — google-genai async client implementing LlmClientProtocol."""

from __future__ import annotations

import json
from typing import Any

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse


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
        return json.loads(raw)

    async def close(self) -> None:
        # google-genai Client does not expose an async close method in current SDK.
        pass
