from __future__ import annotations

from typing import ClassVar

import pytest

from shared.claude_code_extraction import (
    QA,
    CodeChange,
    Decision,
    FileRef,
    extract_units_from_session,
)


@pytest.mark.asyncio
async def test_extract_units_dispatches_to_anthropic_and_parses_tool_use(monkeypatch) -> None:
    """The extractor sends events to the model and returns typed unit dataclasses."""
    fake_tool_input = {
        "qa": [
            {"prompt": "Why is /ingest 422?", "outcome": "Pydantic v2 list[dict] coercion fix", "tags": ["pydantic", "422"]},
        ],
        "code_change": [
            {"file": "app/schemas/ingest.py",
             "before": "events: list[dict]",
             "after": "events: list[Event]",
             "intent": "tighten payload typing"},
        ],
        "decision": [
            {"question": "loosen schema or fix caller?",
             "options_considered": ["loosen", "tighten"],
             "chosen": "tighten",
             "rationale": "validation is the point"},
        ],
        "file_ref": [
            {"files": ["app/routes/ingest.py", "app/schemas/ingest.py"], "context": "Pydantic v2 fix"},
        ],
    }

    class FakeContent:
        type = "tool_use"
        name = "emit_units"
        input = fake_tool_input

    class FakeMessage:
        content: ClassVar[list[FakeContent]] = [FakeContent()]

    class FakeMessages:
        async def create(self, **kwargs):
            return FakeMessage()

    class FakeAsyncAnthropic:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    monkeypatch.setattr("shared.claude_code_extraction.AsyncAnthropic", FakeAsyncAnthropic)

    bundle = await extract_units_from_session(
        session_id="s1",
        events=[{"line_no": 0, "raw": {"role": "user", "content": "Why is /ingest 422?"}}],
        cwd="/tmp/p",
    )

    assert len(bundle.qa) == 1 and isinstance(bundle.qa[0], QA)
    assert len(bundle.code_change) == 1 and isinstance(bundle.code_change[0], CodeChange)
    assert len(bundle.decision) == 1 and isinstance(bundle.decision[0], Decision)
    assert len(bundle.file_ref) == 1 and isinstance(bundle.file_ref[0], FileRef)
    assert bundle.qa[0].outcome.startswith("Pydantic v2")


@pytest.mark.asyncio
async def test_extract_units_truncates_oversized_event_list(monkeypatch) -> None:
    """Defensive guard: events lists larger than _MAX_EVENTS are truncated to
    the most recent slice before being sent to Anthropic. Prevents context
    overflow from blowing up the worker."""
    from shared import claude_code_extraction as ext_mod

    captured_payload = {}

    class FakeContent:
        type: ClassVar = "tool_use"
        name: ClassVar = "emit_units"
        input: ClassVar = {"qa": [], "code_change": [], "decision": [], "file_ref": []}

    class FakeMessage:
        content: ClassVar = [FakeContent()]

    class FakeMessages:
        async def create(self, **kwargs):
            captured_payload["messages"] = kwargs["messages"]
            return FakeMessage()

    class FakeAsyncAnthropic:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    monkeypatch.setattr(ext_mod, "AsyncAnthropic", FakeAsyncAnthropic)

    huge_events = [{"line_no": i, "raw": {}} for i in range(ext_mod._MAX_EVENTS + 500)]
    await ext_mod.extract_units_from_session(
        session_id="big",
        events=huge_events,
        cwd=None,
    )

    # The user message body contains the truncated event list; check it's bounded
    # by reading the captured kwargs.
    user_msg = captured_payload["messages"][0]["content"]
    assert "big" in user_msg  # session_id present
    # The truncation logic keeps the most recent _MAX_EVENTS, so line_no=500 is
    # the smallest line_no that should be retained (since we created 2500 events
    # 0-2499, truncating to the last 2000 keeps indices 500-2499).
    assert '"line_no":500' in user_msg or '"line_no": 500' in user_msg
    # And line_no=2499 is the largest.
    assert '"line_no":2499' in user_msg or '"line_no": 2499' in user_msg
