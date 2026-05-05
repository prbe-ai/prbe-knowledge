"""Unit tests for agent_compactor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.synthesis.agent_compactor import call_summarizer, extract_state_for_summary
from shared.exceptions import AgentCompactionError


def _runtime_state(**overrides) -> dict:
    base = {
        "pending_updates": [
            {"wiki_type": "decision", "slug": "a", "applied_queue_ids": [1, 2]},
        ],
        "pending_creates": [
            {"wiki_type": "runbook", "slug": "b", "applied_queue_ids": [3]},
        ],
        "applied_queue_ids": [1, 2, 3],
        "skipped_queue_ids": [4, 5],
    }
    base.update(overrides)
    return base


def _stub_client(text: str = "summary text"):
    """Build a mock genai client whose generate_content returns `text`."""
    client = SimpleNamespace()
    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(text=text)
    )
    return client


@pytest.mark.asyncio
async def test_call_summarizer_happy_path() -> None:
    """Stubbed Gemini returns a summary; output contains the runtime
    state block + the LLM text."""
    state = _runtime_state()
    client = _stub_client("conversation summary here")
    out = await call_summarizer(
        [{"role": "user", "parts": [{"text": "hello"}]}],
        state,
        client=client,
    )
    assert "RUNTIME STATE:" in out
    # Either appended (because summary text didn't include it) OR the
    # summary includes it — either way, the runtime block must be in.
    assert "applied_queue_ids: [1, 2, 3]" in out


def test_extract_state_for_summary_correctness() -> None:
    """The state block round-trips structured fields verbatim."""
    state = _runtime_state()
    block = extract_state_for_summary(state)
    assert "pending_updates: 1 pages" in block
    assert "[decision/a]" in block
    assert "[runbook/b]" in block
    assert "applied_queue_ids: [1, 2, 3]" in block
    assert "skipped_queue_ids: [4, 5]" in block


@pytest.mark.asyncio
async def test_summarizer_fails_raises_AgentCompactionError() -> None:
    """An empty Gemini response should raise AgentCompactionError; the
    harness catches it and re-raises as AgentHaltError."""
    client = _stub_client("")
    with pytest.raises(AgentCompactionError):
        await call_summarizer(
            [{"role": "user", "parts": [{"text": "x"}]}],
            _runtime_state(),
            client=client,
        )
