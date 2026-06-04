"""Provider Protocol dispatch tests for the triage stage.

Phase-0b: provider impls now route through `shared.llm.acompletion`.
The `client` parameter to `_AnthropicTriage` / `get_triage_provider`
is preserved for API compatibility but unused — `shared.llm`
owns transport. Tests mock the wrapper instead of a fake
`AsyncAnthropic`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from services.synthesis.models import TriageInput
from services.synthesis.providers import (
    TriageParseError,
    _AnthropicTriage,
    _GeminiTriage,
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


def _tool_response(tool_name: str, payload: dict) -> SimpleNamespace:
    """LiteLLM-shaped response carrying a single forced tool call."""
    func = SimpleNamespace(
        name=tool_name,
        arguments=orjson.dumps(payload).decode("utf-8"),
    )
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_default_triage_provider_is_gemini() -> None:
    provider = get_triage_provider()
    assert isinstance(provider, _GeminiTriage)


def test_triage_provider_dispatches_to_gemini_on_override() -> None:
    provider = get_triage_provider(model_override="gemini-3.1-flash-lite")
    assert isinstance(provider, _GeminiTriage)


def test_unknown_triage_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown WIKI_TRIAGE_MODEL"):
        get_triage_provider(model_override="not-a-real-model")


def test_anthropic_provider_accepts_no_client_post_litellm_migration() -> None:
    """Pre-Phase-0b this raised ValueError when no `anthropic_client` was
    passed. After the LiteLLM migration the worker doesn't own transport
    — `shared.llm.acompletion` owns it — so the constructor accepts
    None. Pin the new contract."""
    provider = get_triage_provider(anthropic_client=None, model_override="haiku")
    assert isinstance(provider, _AnthropicTriage)


# ---------------------------------------------------------------------------
# Anthropic round-trip (Protocol-shape compatibility)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_triage_raises_on_validation_error(monkeypatch) -> None:
    """Provider parses tool input through Pydantic; missing fields raise."""
    # Verdict missing required `score` — Pydantic will reject.
    fake = AsyncMock(
        return_value=_tool_response(
            "record_triage",
            {"verdicts": {"1": {"important": True}}},
        )
    )
    monkeypatch.setattr("shared.llm_tools.acompletion", fake)
    provider = get_triage_provider()
    with pytest.raises(TriageParseError):
        await provider.triage([_ev(1)], now=datetime.now(UTC))
