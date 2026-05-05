"""PromptCache — wraps Plan 1's DiskCache with LLM-aware key derivation.

Cache key = sha256(provider | model | prompt | system | temperature | max_tokens | schema_json).
Stored value is a raw dict: {"text": "..."} for text responses, or the
structured output dict directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from scripts.synth.cache import DiskCache
from scripts.synth.llm.base import LlmRequest, Provider


def cache_key(provider: Provider, req: LlmRequest, schema_json: str | None = None) -> str:
    """Derive a deterministic sha256 cache key from all request fields.

    Args:
        provider: The LLM provider (affects cache namespace).
        req: The full LlmRequest (model, system, prompt, temperature, max_tokens).
        schema_json: Optional JSON string of the Pydantic schema for structured calls.

    Returns:
        64-character lowercase hex string.
    """
    parts = "|".join([
        provider.value,
        req.model,
        req.prompt,
        req.system,
        f"{req.temperature}",
        f"{req.max_tokens}",
        schema_json or "",
    ])
    return hashlib.sha256(parts.encode()).hexdigest()


class PromptCache:
    """Disk-backed cache for LLM responses keyed by request content hash."""

    def __init__(self, root: Path) -> None:
        self._disk = DiskCache(root=root)

    async def get(
        self,
        provider: Provider,
        req: LlmRequest,
        schema_json: str | None,
    ) -> dict | None:
        """Return cached response dict, or None on cache miss."""
        key = cache_key(provider, req, schema_json)
        return self._disk.get(key)

    async def put(
        self,
        provider: Provider,
        req: LlmRequest,
        schema_json: str | None,
        response_dict: dict,
    ) -> None:
        """Store a response dict under the derived cache key."""
        key = cache_key(provider, req, schema_json)
        self._disk.put(key, response_dict)
