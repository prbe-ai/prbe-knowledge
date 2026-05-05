"""GeminiAgentClient — adapts google-genai to the AgentLoop's _LLMClient surface.

The harness expects two methods:

    create_cache(*, system_instruction, tools, seed_contents) -> str
    generate_with_cache(*, cache_name, contents, tools) -> dict

This module wraps the production Gemini SDK with that contract. Stays
out of the unit-test path (tests pass their own stub client into
SynthesisWorker via llm_client=...).

Reference shape (from google-genai==1.x):

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
    resp = await client.aio.models.generate_content(
        model="gemini-3.1-pro-preview",
        contents=[...],
        config=GenerateContentConfig(
            cached_content=cache.name,
            tools=[Tool(function_declarations=[...])],
        ),
    )
"""

from __future__ import annotations

from typing import Any

from shared.config import get_settings
from shared.constants import WIKI_AGENT_CACHE_TTL, WIKI_AGENT_MODEL
from shared.logging import get_logger

log = get_logger(__name__)


class GeminiAgentClient:
    """Production wrapper around google-genai for the wiki agent loop."""

    def __init__(self, *, model: str = WIKI_AGENT_MODEL) -> None:
        self._model = model
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
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
                parameters=t.get("parameters") or {"type": "object", "properties": {}},
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
        client = self._ensure_client()
        from google.genai.types import (
            FunctionDeclaration,
            GenerateContentConfig,
            Tool,
        )

        function_decls = [
            FunctionDeclaration(
                name=t["name"],
                description=t.get("description"),
                parameters=t.get("parameters") or {"type": "object", "properties": {}},
            )
            for t in tools
        ]
        kwargs: dict[str, Any] = {}
        if cache_name:
            kwargs["cached_content"] = cache_name
        config = GenerateContentConfig(
            tools=[Tool(function_declarations=function_decls)],
            **kwargs,
        )
        resp = await client.aio.models.generate_content(
            model=self._model,
            contents=contents,
            config=config,
        )
        return _extract_response(resp)


def _extract_response(resp: Any) -> dict[str, Any]:
    """Normalize the SDK's response object into the harness's dict shape.

    The harness expects:
      {
        "text": str | None,
        "tool_calls": [{"name": ..., "args": {...}}, ...],
        "usage_metadata": {
            "prompt_token_count": int,
            "cached_content_token_count": int,
            "candidates_token_count": int,
        },
      }
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
            tool_calls.append(
                {"name": getattr(fc, "name", ""), "args": dict(args)}
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
