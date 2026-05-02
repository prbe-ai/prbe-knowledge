"""MockLlmClient — fixture-keyed replay/record client.

In replay mode, looks up fixtures by cache key and raises
FixtureNotFoundError with a --record-llm hint on miss.

In record mode, forwards to real_client, writes the fixture, and
returns the real response. real_client is mandatory in record mode.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from scripts.synth.llm.base import LlmClientProtocol, LlmRequest, LlmResponse, provider_from_model
from scripts.synth.llm.cache import cache_key
from scripts.synth.llm.fixtures import FixtureStore


class FixtureNotFoundError(Exception):
    """Raised when a fixture is missing in replay mode."""


class MockLlmClient:
    """Fixture-backed LLM client for deterministic testing.

    Args:
        store: FixtureStore pointing at the fixture root directory.
        mode: 'replay' (default) or 'record'.
        real_client: Required when mode='record'; must implement LlmClientProtocol.
    """

    def __init__(
        self,
        store: FixtureStore,
        *,
        mode: Literal["replay", "record"] = "replay",
        real_client: LlmClientProtocol | None = None,
    ) -> None:
        if mode == "record" and real_client is None:
            raise ValueError("real_client is required when mode='record'")
        self._store = store
        self._mode = mode
        self._real = real_client

    async def generate(self, req: LlmRequest) -> LlmResponse:
        provider = provider_from_model(req.model)
        key = cache_key(provider, req)
        if self._mode == "replay":
            fixture = self._store.load(provider, key)
            if fixture is None:
                raise FixtureNotFoundError(
                    f"No fixture for key={key!r} provider={provider.value!r}. "
                    "Re-run with --record-llm to record real responses."
                )
            return LlmResponse(text=fixture.get("text", ""))
        # record mode
        assert self._real is not None  # guaranteed by __init__
        resp = await self._real.generate(req)
        self._store.record(provider, key, {"text": resp.text})
        return resp

    async def generate_structured(self, req: LlmRequest, schema: type[BaseModel]) -> dict:
        provider = provider_from_model(req.model)
        schema_json = json.dumps(schema.model_json_schema(), sort_keys=True)
        key = cache_key(provider, req, schema_json)
        if self._mode == "replay":
            fixture = self._store.load(provider, key)
            if fixture is None:
                raise FixtureNotFoundError(
                    f"No structured fixture for key={key!r} provider={provider.value!r}. "
                    "Re-run with --record-llm to record real responses."
                )
            return fixture
        assert self._real is not None
        result = await self._real.generate_structured(req, schema)
        self._store.record(provider, key, result)
        return result

    async def close(self) -> None:
        if self._real is not None:
            await self._real.close()
