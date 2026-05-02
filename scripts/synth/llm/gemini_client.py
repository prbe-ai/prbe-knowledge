"""GeminiClient — google-genai async client implementing LlmClientProtocol."""

from __future__ import annotations

import json

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel

from scripts.synth.llm.base import LlmRequest, LlmResponse


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
        """Use Gemini's native structured-output mode via response_schema."""
        config = GenerateContentConfig(
            system_instruction=req.system or None,
            max_output_tokens=req.max_tokens,
            temperature=req.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        response = await self._client.aio.models.generate_content(
            model=req.model,
            contents=req.prompt,
            config=config,
        )
        raw = response.text or "{}"
        return json.loads(raw)

    async def close(self) -> None:
        # google-genai Client does not expose an async close method in current SDK.
        pass
