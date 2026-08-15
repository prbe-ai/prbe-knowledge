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
                "summary": "Tighten the ingest payload typing",
                "kind": "fix",
                "files": ["app/schemas/ingest.py"],
                "rationale": "validation is the point of the endpoint",
                "evidence": "events: list[dict] -> events: list[Event]",
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
            {"line_no": 0, "raw": {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Why is /ingest 422?"}]}}}
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
async def test_extract_units_gateway_preserves_alias_and_uses_openai_wire(
    monkeypatch,
) -> None:
    """Gateway aliases stay bare while LiteLLM speaks the proxy's OpenAI wire."""
    empty_payload = {"qa": [], "code_change": [], "decision": [], "file_ref": []}
    fake = AsyncMock(return_value=_litellm_tool_response("emit_units", empty_payload))
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://litellm.example/v1")
    monkeypatch.setattr(
        "engine.shared.claude_code_extraction.get_settings",
        lambda: SimpleNamespace(claude_code_extraction_model="gemini-3.5-flash"),
    )
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)

    await extract_units_from_session(session_id="gateway", events=[{"line_no": 0, "raw": {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Why is /ingest 422?"}]}}}])

    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "gemini-3.5-flash"
    assert kwargs["custom_llm_provider"] == "openai"


@pytest.mark.asyncio
async def test_extract_units_direct_prefixes_anthropic_without_wire_override(
    monkeypatch,
) -> None:
    """Direct calls keep the legacy Anthropic routing contract."""
    empty_payload = {"qa": [], "code_change": [], "decision": [], "file_ref": []}
    fake = AsyncMock(return_value=_litellm_tool_response("emit_units", empty_payload))
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.setattr(
        "engine.shared.claude_code_extraction.get_settings",
        lambda: SimpleNamespace(claude_code_extraction_model="claude-sonnet-4-6"),
    )
    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake)

    await extract_units_from_session(session_id="direct", events=[{"line_no": 0, "raw": {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "Why is /ingest 422?"}]}}}])

    kwargs = fake.await_args.kwargs
    assert kwargs["model"] == "anthropic/claude-sonnet-4-6"
    assert "custom_llm_provider" not in kwargs


def _user(text: str, line_no: int = 0) -> dict:
    return {
        "line_no": line_no,
        "raw": {
            "type": "user",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        },
    }


def _boundary(line_no: int) -> dict:
    return {
        "line_no": line_no,
        "raw": {"type": "system", "subtype": "compact_boundary",
                "content": "Conversation compacted"},
    }


def _summary(line_no: int) -> dict:
    return {
        "line_no": line_no,
        "raw": {
            "type": "user",
            "isCompactSummary": True,
            "message": {"role": "user",
                        "content": [{"type": "text", "text": "Summary of earlier work"}]},
        },
    }


def test_session_splits_at_every_compaction_boundary() -> None:
    """The pre-compaction conversation is never lost — it is on disk, shipped,
    and in the indexed document. Only the extractor used to miss it, because it
    kept the last 2000 events: in four of six measured compacted sessions that
    meant ZERO pre-compaction events were ever mined."""
    from engine.shared.claude_code_extraction import _segment_session

    events = [_user("a", 0), _boundary(1), _summary(2), _user("b", 3),
              _boundary(4), _summary(5), _user("c", 6)]
    segments, capped = _segment_session(events)
    assert len(segments) == 3
    assert capped is False
    # Boundaries open the following segment, so no turn is orphaned.
    assert segments[0][0][0] is events[0]
    assert segments[1][0][0] is events[1]


def test_oversized_segment_is_split_on_a_user_turn() -> None:
    """A session that never compacted still needs a size guard — one measured
    session had a single 1.1M-character stretch and no boundary at all."""
    from engine.shared import claude_code_extraction as ext_mod

    big = "x" * 40_000
    events = [_user(big, i) for i in range(20)]
    segments, _ = ext_mod._segment_session(events)
    assert len(segments) > 1
    # Every cut lands on a user turn, so a request stays with the work it caused.
    for seg, _why in segments[1:]:
        assert ext_mod._renders_as_user_turn(seg[0])


@pytest.mark.asyncio
async def test_every_segment_is_extracted_and_merged(monkeypatch) -> None:
    """One call per segment, and the units of all of them come back."""
    from engine.shared import claude_code_extraction as ext_mod

    seen: list[str] = []

    async def fake_acompletion(**kwargs):
        seen.append(kwargs["messages"][-1]["content"])
        return _litellm_tool_response("emit_units", {
            "qa": [{"prompt": f"q{len(seen)}", "outcome": "o"}],
            "code_change": [], "decision": [], "file_ref": [],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)

    events = [_user("first", 0), _boundary(1), _summary(2), _user("second", 3)]
    bundle = await ext_mod.extract_units_from_session(session_id="s", events=events)

    assert len(seen) == 2, "one extraction call per segment"
    assert len(bundle.qa) == 2, "units from every segment are merged"
    # The model is sent RENDERED prose, not raw event JSON: measured 3.7x-9.3x
    # smaller, and 476k-929k tokens of JSON against a 200k context was a
    # guaranteed failure on any large session.
    assert "USER: first" in seen[0]
    assert '"line_no"' not in seen[0]
    # With the originals of every segment in hand, the compaction summary is a
    # second telling of what we are already reading.
    assert "Summary of earlier work" not in seen[1]


@pytest.mark.asyncio
async def test_one_failing_segment_does_not_lose_the_others(monkeypatch) -> None:
    """A partial bundle beats none, and the loss is logged rather than silent."""
    from engine.shared import claude_code_extraction as ext_mod

    calls = {"n": 0}

    async def flaky(**kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("context_length_exceeded")
        return _litellm_tool_response("emit_units", {
            "qa": [{"prompt": "survived", "outcome": "o"}],
            "code_change": [], "decision": [], "file_ref": [],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", flaky)

    events = [_user("first", 0), _boundary(1), _user("second", 2)]
    bundle = await ext_mod.extract_units_from_session(session_id="s", events=events)
    assert [q.prompt for q in bundle.qa] == ["survived"]


def test_segments_record_why_each_one_began() -> None:
    """`compaction` is a chapter break the AGENT chose when it ran out of
    context; `size` is a cut we made to fit a call and means nothing about the
    work. Collapsing the two would make a mechanical split look like a real
    narrative boundary."""
    from engine.shared.claude_code_extraction import _segment_session

    events = [_user("a", 0), _boundary(1), _user("b", 2), _boundary(3), _user("c", 4)]
    segments, _ = _segment_session(events)
    assert [why for _, why in segments] == ["session_start", "compaction", "compaction"]


def test_size_subsplits_do_not_masquerade_as_compactions() -> None:
    """Only the FIRST piece of a compaction stretch inherits the real
    boundary; the rest are size cuts through continuous work."""
    from engine.shared.claude_code_extraction import _segment_session

    big = "y" * 40_000
    events = [_boundary(0)] + [_user(big, i + 1) for i in range(20)]
    segments, _ = _segment_session(events)
    reasons = [why for _, why in segments]
    assert reasons[0] == "session_start"
    assert reasons.count("size") >= 1
    assert reasons.count("compaction") == 0, (
        "one compaction must not become several"
    )


@pytest.mark.asyncio
async def test_units_carry_their_place_in_the_session(monkeypatch) -> None:
    """Without this a unit is a free-floating fact about a session that may
    have run for hours; with it the units can be put back in order and located
    in the transcript they came from."""
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "qa": [{"prompt": "q", "outcome": "o"}],
            "code_change": [],
            "decision": [{"question": "a or b?", "options_considered": ["a", "b"],
                          "chosen": "a", "rationale": "why"}],
            "file_ref": [],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)

    events = [_user("first", 10), _boundary(11), _user("second", 12)]
    bundle = await ext_mod.extract_units_from_session(session_id="s", events=events)

    assert len(bundle.qa) == 2
    refs = sorted((q.segment for q in bundle.qa), key=lambda r: r.index)
    assert [r.index for r in refs] == [1, 2]
    assert all(r.total == 2 for r in refs)
    assert [r.boundary for r in refs] == ["session_start", "compaction"]
    # Line anchors locate the segment back in the transcript.
    assert refs[0].start_line_no == 10
    assert refs[1].start_line_no == 11
    # Every unit type is stamped, not just qa.
    assert all(d.segment is not None for d in bundle.decision)


@pytest.mark.asyncio
async def test_a_change_spanning_files_is_one_unit(monkeypatch) -> None:
    """One fix that touched six files is ONE change, not six.

    The old file-shaped unit produced 93 documents for a single session — 32% of
    everything it emitted — each telling a fraction of the work. Nobody
    describes their own work that way.
    """
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "qa": [], "decision": [], "file_ref": [],
            "code_change": [{
                "summary": "Reject writes that would blow the per-run series cap",
                "kind": "feature",
                "files": ["app/store.py", "app/routes.py", "tests/test_store.py"],
                "rationale": "A runaway dimension silently turned one curve into "
                             "thousands of one-point series.",
                "evidence": "",
            }],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("go", 0)]
    )

    assert len(bundle.code_change) == 1
    change = bundle.code_change[0]
    assert len(change.files) == 3
    assert change.kind == "feature"
    assert change.evidence == "", "no diff was shown, so none is claimed"
    assert change.segment is not None


@pytest.mark.asyncio
async def test_a_stale_unit_shape_does_not_cost_the_segment(monkeypatch) -> None:
    """The schema changes as we learn what a unit should be. A model answering
    with an older shape must lose the extra field, not the whole segment."""
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "qa": [], "decision": [], "file_ref": [],
            "code_change": [{
                "summary": "Widen the field",
                "kind": "fix",
                "files": ["app/models.py"],
                "rationale": "ints could not hold the new ids",
                "before": "x: int",      # gone from the schema
                "after": "x: str",       # gone from the schema
            }],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("go", 0)]
    )
    assert [c.summary for c in bundle.code_change] == ["Widen the field"]
