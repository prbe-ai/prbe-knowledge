"""Anthropic SDK wrapper. Plan 1 needs only basic generate(). Plan 3
adds prompt-caching block + mock-mode fixture loader."""

from __future__ import annotations

from typing import Any, ClassVar

import pytest

from scripts.synth.llm_client import (
    LlmClient,
    LlmRequest,
    StaticLlmClient,
)


@pytest.mark.asyncio
async def test_static_client_returns_canned_response() -> None:
    client = StaticLlmClient({"hello": "hi there"})
    resp = await client.generate(LlmRequest(model="m", system="", prompt="hello"))
    assert resp.text == "hi there"


@pytest.mark.asyncio
async def test_static_client_raises_on_unmapped_prompt() -> None:
    client = StaticLlmClient({})
    with pytest.raises(KeyError):
        await client.generate(LlmRequest(model="m", system="", prompt="??"))


@pytest.mark.asyncio
async def test_real_client_passes_prompt_to_anthropic_sdk(monkeypatch) -> None:
    """Verify the real client wires prompt → anthropic.messages.create
    (we don't actually call the network)."""

    class FakeMessages:
        called_with: ClassVar[dict[str, Any]] = {}

        async def create(self, **kwargs):
            FakeMessages.called_with = kwargs
            class _R:
                content: ClassVar[list[Any]] = [type("B", (), {"text": "ok", "type": "text"})()]
            return _R()

    class FakeClient:
        def __init__(self, **_): self.messages = FakeMessages()
        async def aclose(self): pass

    import scripts.synth.llm_client as mod
    monkeypatch.setattr(mod, "AsyncAnthropic", FakeClient)

    client = LlmClient(api_key="test-key")
    resp = await client.generate(LlmRequest(model="claude-x", system="sys", prompt="hi"))
    await client.close()

    assert resp.text == "ok"
    assert FakeMessages.called_with["model"] == "claude-x"
    assert FakeMessages.called_with["system"] == "sys"
    assert FakeMessages.called_with["messages"][0]["content"] == "hi"
