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


@pytest.mark.asyncio
async def test_units_carry_the_goal_their_segment_served(monkeypatch) -> None:
    """Audited over 72 real decisions, every one read as a well-argued LOCAL
    tradeoff orphaned from the work it served: you could tell what was decided
    and why that option won, never why the work was happening.

    The goal is extracted once per segment and stamped onto every unit, so one
    sentence gives all of them their context instead of the model repeating
    itself per unit.
    """
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "goal": {
                "objective": "Capture Harbor sandbox end-state without touching the task image",
                "motivation": "Without it a failed trial leaves no evidence of what the agent did",
            },
            "qa": [{"prompt": "q", "outcome": "o"}],
            "code_change": [], "file_ref": [],
            "decision": [{
                "question": "exec into the container or observe from the host?",
                "options_considered": ["exec", "docker diff from host"],
                "chosen": "exec, hardened to be ephemeral",
                "rationale": "host observation needs the Docker socket, which Harbor's "
                             "remote providers do not expose",
                "serves": "Makes the capture work on every provider, which the objective "
                          "requires since Nebius runs remote sandboxes",
            }],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("go", 0)]
    )

    decision = bundle.decision[0]
    # Both halves of "why did we decide this".
    assert "Docker socket" in decision.rationale        # why this option won
    assert "every provider" in decision.serves          # why we were choosing
    assert decision.segment.objective.startswith("Capture Harbor sandbox")
    assert decision.segment.motivation

    # Every unit type inherits it, not just decisions.
    assert bundle.qa[0].segment.objective == decision.segment.objective


@pytest.mark.asyncio
async def test_concurrent_segments_do_not_swap_goals(monkeypatch) -> None:
    """Segments extract concurrently. Holding the goal in module state would
    hand one segment's objective to another's units — worse than having none,
    because it reads as true."""
    from engine.shared import claude_code_extraction as ext_mod

    seen = {"n": 0}

    async def fake_acompletion(**kwargs):
        seen["n"] += 1
        which = "first" if "first" in kwargs["messages"][-1]["content"] else "second"
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": f"objective for the {which} part", "motivation": ""},
            "qa": [{"prompt": which, "outcome": "o"}],
            "code_change": [], "decision": [], "file_ref": [],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    events = [_user("first", 0), _boundary(1), _user("second", 2)]
    bundle = await ext_mod.extract_units_from_session(session_id="s", events=events)

    assert seen["n"] == 2
    for unit in bundle.qa:
        assert unit.segment.objective == f"objective for the {unit.prompt} part"


@pytest.mark.asyncio
async def test_decisions_record_who_actually_decided(monkeypatch) -> None:
    """Authorship was not merely absent, it was actively lost: over 72 real
    decisions only 8 rationales credited the user, and in one session 0 of 12
    did while the transcript showed roughly five were the user's call outright.

    It is the field that makes per-person norms possible — a decision the agent
    took alone says nothing about how a researcher works, and one they overrode
    says a great deal.
    """
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "make liveness honest", "motivation": ""},
            "qa": [], "code_change": [], "file_ref": [],
            "decision": [{
                "question": "server, local locks, or both?",
                "options_considered": ["server", "local locks", "both"],
                "chosen": "both",
                "rationale": "local locks cannot see a remote run",
                "serves": "liveness is correct on every machine",
                "decided_by": "user_directed",
            }],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("go", 0)]
    )
    assert bundle.decision[0].decided_by == "user_directed"
    assert bundle.decision[0].decided_by in ext_mod._DECIDED_BY


@pytest.mark.asyncio
async def test_a_quote_that_is_not_in_the_transcript_is_caught(monkeypatch) -> None:
    """The only fabrication signal available without a human reading the
    session. A model asked for a quote and answering with one that appears
    nowhere has told us something its own prose never would.

    Confidence is DOWNGRADED on a failed match and never raised on a passing
    one — matching a quote proves the words were said, not that the unit read
    them correctly.
    """
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "o", "motivation": ""},
            "code_change": [], "decision": [], "file_ref": [],
            "qa": [
                {"prompt": "real", "outcome": "o", "confidence": "high",
                 "evidence": "pin the tokenizer"},
                {"prompt": "invented", "outcome": "o", "confidence": "high",
                 "evidence": "we agreed to rewrite the scheduler"},
            ],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("we should pin the tokenizer version", 4)]
    )

    grounded = {q.prompt: q for q in bundle.qa}
    assert grounded["real"].evidence_verified is True
    assert grounded["real"].confidence == "high"
    assert grounded["real"].anchor_line_no == 4, "the quote locates its own line"

    assert grounded["invented"].evidence_verified is False
    assert grounded["invented"].confidence == "low", "claimed high, capped at low"


@pytest.mark.asyncio
async def test_quote_matching_ignores_line_wrapping(monkeypatch) -> None:
    """A model reproduces the words reliably and the newlines less so. Failing
    a quote over a wrapped line would measure formatting, not fabrication."""
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "o", "motivation": ""},
            "code_change": [], "decision": [], "file_ref": [],
            "qa": [{"prompt": "p", "outcome": "o", "confidence": "medium",
                    "evidence": "pin the tokenizer version"}],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("we should pin the\ntokenizer version", 9)]
    )
    assert bundle.qa[0].evidence_verified is True
    assert bundle.qa[0].confidence == "medium", "a match never raises confidence"


@pytest.mark.asyncio
async def test_the_model_cannot_mark_its_own_evidence_verified(monkeypatch) -> None:
    """evidence_verified is the whole point of the check; a model that could
    assert it would be grading its own homework."""
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "o", "motivation": ""},
            "code_change": [], "decision": [], "file_ref": [],
            "qa": [{"prompt": "p", "outcome": "o", "confidence": "high",
                    "evidence": "nowhere in the transcript",
                    "evidence_verified": True, "anchor_line_no": 999}],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("something else entirely", 0)]
    )
    assert bundle.qa[0].evidence_verified is False
    assert bundle.qa[0].anchor_line_no is None


@pytest.mark.asyncio
async def test_a_long_quote_that_drifts_late_still_verifies(monkeypatch) -> None:
    """Models reproduce the opening of a passage faithfully and drift later.
    Measured on a real session, every failing quote matched its first several
    words and diverged after — failing those outright made the check measure
    verbosity rather than fabrication."""
    from engine.shared import claude_code_extraction as ext_mod

    real = ("Qwen3-8B is the minimal-diff switch because its chat template is "
            "byte-for-byte the same mechanic as the model we started from")

    async def fake_acompletion(**kwargs):
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "o", "motivation": ""},
            "code_change": [], "decision": [], "file_ref": [],
            "qa": [{"prompt": "p", "outcome": "o", "confidence": "high",
                    # faithful opening, drifted tail
                    "evidence": real[:70] + " and everything else carried over"}],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user(real, 3)]
    )
    assert bundle.qa[0].evidence_verified is True
    assert bundle.qa[0].anchor_line_no == 3


def test_a_short_invented_quote_cannot_ride_the_prefix_rule() -> None:
    """The fallback must not become a way in. Sixty contiguous characters of a
    transcript are not reproduced by accident."""
    from engine.shared.claude_code_extraction import _collapse, _locate

    transcript = _collapse("we should pin the tokenizer version before training")
    assert _locate(transcript, "we should pin the tokenizer") >= 0
    assert _locate(transcript, "we decided to rewrite the entire training scheduler "
                               "from scratch this afternoon") < 0


@pytest.mark.asyncio
async def test_a_cross_segment_reversal_is_found(monkeypatch) -> None:
    """Segments extract in separate concurrent calls, so the model handling
    segment 7 has never seen segment 3. The clearest real reversal in the
    corpus spans exactly that gap — per-segment extraction cannot see it even
    in principle, which is why this runs once over the assembled bundle."""
    from engine.shared import claude_code_extraction as ext_mod

    def decision(q, chosen):
        return {"question": q, "options_considered": ["a", "b"], "chosen": chosen,
                "rationale": "r", "serves": "s", "decided_by": "agent_unilateral",
                "status": "implemented", "trigger": "user_request",
                "evidence": "go", "confidence": "high"}

    async def fake_acompletion(**kwargs):
        tool = kwargs["tool_choice"]["function"]["name"]
        if tool == "emit_supersessions":
            return _litellm_tool_response("emit_supersessions", {
                "links": [{"earlier": 0, "later": 1, "reason": "reversed"}]
            })
        which = "first" if "first" in kwargs["messages"][-1]["content"] else "second"
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "o", "motivation": ""},
            "qa": [], "code_change": [], "file_ref": [], "directive": [],
            "decision": [decision(f"{which} question", which)],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    events = [_user("first", 0), _boundary(1), _user("second", 2)]
    bundle = await ext_mod.extract_units_from_session(session_id="s", events=events)

    assert len(bundle.decision) == 2
    assert bundle.decision[0].superseded_by == 1
    assert bundle.decision[1].supersedes == 0


@pytest.mark.asyncio
async def test_a_backwards_or_out_of_range_link_is_refused(monkeypatch) -> None:
    """A wrong link is worse than a missing one — it rewrites the session's
    history. Only strictly forward links into the list we actually sent."""
    from engine.shared import claude_code_extraction as ext_mod

    def decision(q):
        return {"question": q, "options_considered": ["a", "b"], "chosen": "a",
                "rationale": "r", "serves": "s", "decided_by": "agent_unilateral",
                "status": "implemented", "trigger": "user_request",
                "evidence": "go", "confidence": "high"}

    async def fake_acompletion(**kwargs):
        if kwargs["tool_choice"]["function"]["name"] == "emit_supersessions":
            return _litellm_tool_response("emit_supersessions", {"links": [
                {"earlier": 1, "later": 0, "reason": "backwards"},
                {"earlier": 0, "later": 99, "reason": "out of range"},
                {"earlier": 0, "later": 0, "reason": "itself"},
            ]})
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "o", "motivation": ""},
            "qa": [], "code_change": [], "file_ref": [], "directive": [],
            "decision": [decision("one"), decision("two")],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("go", 0)]
    )
    assert all(d.supersedes is None and d.superseded_by is None
               for d in bundle.decision)


@pytest.mark.asyncio
async def test_a_failed_supersede_pass_keeps_the_decisions(monkeypatch) -> None:
    """Best-effort: a failure loses the links, never the decisions."""
    from engine.shared import claude_code_extraction as ext_mod

    async def fake_acompletion(**kwargs):
        if kwargs["tool_choice"]["function"]["name"] == "emit_supersessions":
            raise RuntimeError("gateway down")
        return _litellm_tool_response("emit_units", {
            "goal": {"objective": "o", "motivation": ""},
            "qa": [], "code_change": [], "file_ref": [], "directive": [],
            "decision": [
                {"question": f"q{i}", "options_considered": ["a", "b"],
                 "chosen": "a", "rationale": "r", "serves": "s",
                 "decided_by": "agent_unilateral", "status": "implemented",
                 "trigger": "user_request", "evidence": "go",
                 "confidence": "high"} for i in range(2)
            ],
        })

    monkeypatch.setattr("engine.shared.llm_tools.acompletion", fake_acompletion)
    bundle = await ext_mod.extract_units_from_session(
        session_id="s", events=[_user("go", 0)]
    )
    assert len(bundle.decision) == 2
