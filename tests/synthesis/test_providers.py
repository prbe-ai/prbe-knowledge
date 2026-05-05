"""Provider Protocol dispatch tests for the triage stage.

The triage stage selects a provider via the constant `WIKI_TRIAGE_MODEL`.
v4 dropped the verifier + synthesis providers (the wiki agent uses
Gemini directly via `services.synthesis.gemini_agent_client`); only
the triage provider abstraction remains.

These tests verify:
  - default model name resolves to the Anthropic implementation.
  - explicit `model_override` flips dispatch to Gemini Flash Lite.
  - unknown model names raise.
  - Anthropic round-trip works with a mocked AsyncAnthropic client.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.synthesis.models import TriageInput
from services.synthesis.providers import (
    TriageParseError,
    _AnthropicTriage,
    get_triage_provider,
)


def _ev(qid: int, doc_id: str = "doc:1") -> TriageInput:
    return TriageInput(
        queue_id=qid,
        doc_id=doc_id,
        doc_type="github.commit",
        source_system="github",
        title="t",
        author_id="alice",
        body="body",
        body_token_count=10,
    )


def _tool_use(name: str, payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name=name, input=payload)
    return SimpleNamespace(content=[block])


def _mock_anthropic(payload_by_tool: dict[str, dict]) -> SimpleNamespace:
    async def create(*, model: str, **kwargs):
        tool_name = (kwargs.get("tools") or [{}])[0].get("name", "")
        return _tool_use(tool_name, payload_by_tool[tool_name])

    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=create))
    return client


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_default_triage_provider_is_anthropic() -> None:
    client = _mock_anthropic({})
    provider = get_triage_provider(client)
    assert isinstance(provider, _AnthropicTriage)


def test_triage_provider_dispatches_to_gemini_on_override() -> None:
    from services.synthesis.providers import _GeminiTriage

    provider = get_triage_provider(model_override="gemini-flash-lite-preview")
    assert isinstance(provider, _GeminiTriage)


def test_unknown_triage_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown WIKI_TRIAGE_MODEL"):
        get_triage_provider(model_override="not-a-real-model")


def test_anthropic_provider_requires_client() -> None:
    with pytest.raises(ValueError, match="requires an AsyncAnthropic client"):
        get_triage_provider(anthropic_client=None, model_override="haiku")


# ---------------------------------------------------------------------------
# Anthropic round-trip (Protocol-shape compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_triage_raises_on_validation_error() -> None:
    """Provider parses tool input through Pydantic; missing fields raise."""
    client = _mock_anthropic({"record_triage": {"verdicts": {"1": {"important": True}}}})
    provider = get_triage_provider(client)
    with pytest.raises(TriageParseError):
        await provider.triage([_ev(1)], now=datetime.now(UTC))
