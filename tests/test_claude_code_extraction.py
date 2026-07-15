"""Unit tests for `shared.claude_code_extraction.extract_units_from_session`.

Phase-0b: the extractor now routes through `shared.llm.acompletion`
(LiteLLM-backed). Tests mock the wrapper rather than constructing a fake
`AsyncAnthropic`. We assert the same observable behavior the SDK-shaped
tests asserted: typed unit dataclasses come back, and oversized event
lists are truncated to the most-recent slice before being sent.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from engine.shared.claude_code_extraction import (
    QA,
    CodeChange,
    Decision,
    FileRef,
    extract_units_from_session,
)


def _litellm_tool_response(tool_name: str, payload: dict) -> SimpleNamespace:
    """Build a LiteLLM-shaped response carrying a single forced tool call.

    Mirrors what `litellm.acompletion` returns: a ChatCompletion-style
    object with `choices[0].message.tool_calls[0].function.{name,
    arguments}`. `arguments` is a JSON string per OpenAI spec; LiteLLM
    normalises Anthropic `tool_use` blocks into this shape.
    """
    func = SimpleNamespace(
        name=tool_name,
        arguments=orjson.dumps(payload).decode("utf-8"),
    )
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


@pytest.mark.asyncio
async def test_extract_units_dispatches_via_litellm_and_parses_tool_call(
    monkeypatch,
) -> None:
    """The extractor sends events to the model and returns typed unit dataclasses."""
    fake_tool_input = {
        "qa": [
            {
                "prompt": "Why is /ingest 422?",
                "outcome": "Pydantic v2 list[dict] coercion fix",
                "tags": ["pydantic", "422"],
            },
        ],
        "code_change": [
            {
                "file": "app/schemas/ingest.py",
                "before": "events: list[dict]",
                "after": "events: list[Event]",
                "intent": "tighten payload typing",
            },
        ],
        "decision": [
            {
                "question": "loosen schema or fix caller?",
                "options_considered": ["loosen", "tighten"],
                "chosen": "tighten",
                "rationale": "validation is the point",
            },
        ],
        "file_ref": [
            {
                "files": ["app/routes/ingest.py", "app/schemas/ingest.py"],
                "context": "Pydantic v2 fix",
            },
        ],
    }
    fake = AsyncMock(return_value=_litellm_tool_response("emit_units", fake_tool_input))
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)

    bundle = await extract_units_from_session(
        session_id="s1",
        events=[
            {"line_no": 0, "raw": {"role": "user", "content": "Why is /ingest 422?"}}
        ],
        cwd="/tmp/p",
    )

    assert len(bundle.qa) == 1 and isinstance(bundle.qa[0], QA)
    assert len(bundle.code_change) == 1 and isinstance(bundle.code_change[0], CodeChange)
    assert len(bundle.decision) == 1 and isinstance(bundle.decision[0], Decision)
    assert len(bundle.file_ref) == 1 and isinstance(bundle.file_ref[0], FileRef)
    assert bundle.qa[0].outcome.startswith("Pydantic v2")
    # Sanity: the forced tool-call wiring fired.
    fake.assert_awaited_once()
    kwargs = fake.await_args.kwargs
    assert kwargs["tool_choice"]["function"]["name"] == "emit_units"
    assert kwargs["tools"][0]["function"]["name"] == "emit_units"


@pytest.mark.asyncio
async def test_extract_units_truncates_oversized_event_list(monkeypatch) -> None:
    """Defensive guard: events lists larger than _MAX_EVENTS are truncated to
    the most recent slice before being sent to the LLM. Prevents context
    overflow from blowing up the worker."""
    from engine.shared import claude_code_extraction as ext_mod

    empty_payload = {"qa": [], "code_change": [], "decision": [], "file_ref": []}
    captured: dict = {}

    async def fake_acompletion(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _litellm_tool_response("emit_units", empty_payload)

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)

    huge_events = [{"line_no": i, "raw": {}} for i in range(ext_mod._MAX_EVENTS + 500)]
    await ext_mod.extract_units_from_session(
        session_id="big",
        events=huge_events,
        cwd=None,
    )

    # The user message contains the truncated event list; check it's
    # bounded by reading the captured kwargs.
    user_msg_content = captured["messages"][-1]["content"]
    assert isinstance(user_msg_content, str)
    assert "big" in user_msg_content  # session_id present
    # The truncation logic keeps the most recent _MAX_EVENTS, so line_no=500 is
    # the smallest line_no that should be retained (since we created 2500
    # events 0-2499, truncating to the last 2000 keeps indices 500-2499).
    assert '"line_no":500' in user_msg_content or '"line_no": 500' in user_msg_content
    # And line_no=2499 is the largest.
    assert '"line_no":2499' in user_msg_content or '"line_no": 2499' in user_msg_content
