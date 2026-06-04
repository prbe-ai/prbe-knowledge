"""Unit tests for `call_triage_with_split_retry`.

Defense-in-depth on top of PR #100's tightened triage packer. Even with
the Anthropic tokenizer multiplier and per-event framing accounted for,
tokenizer drift, a prompt-template change, or an unusually dense doc
could still push a wire request past Anthropic's 200K hard limit. When
that happens we want to halve-and-retry instead of DLQ'ing the whole
batch.

Phase-0b: the call sites route through `shared.llm.acompletion`, which
wraps every LiteLLM exception in `shared.llm.LLMError`. The split-retry
detector lives in `shared.llm_tools.is_context_overflow` and matches
the same Anthropic 400 phrasings the SDK-shape detector did.

These tests pin the wrapper behavior:

  1. A single oversize 400 on the full batch → split into halves, both
     halves succeed → all verdicts returned, exactly 3 LLM calls.
  2. Repeated overflows recursing all the way down to single events →
     every event's verdict still surfaces (some as success, some as
     rejected).
  3. A single-event batch that overflows is reported as
     `triage.oversized_event_at_call_time` rejected, no further recursion.
  4. A non-oversize 400 (e.g. authentication) propagates unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import orjson
import pytest

from services.synthesis.models import TriageInput, TriageOutput
from services.synthesis.triage import (
    OVERSIZED_AT_CALL_TIME_REASON,
    TRIAGE_FINAL_MILE_SYNTHESIS_REASON,
    TRIAGE_RECURSION_FAILED_REASON,
    call_triage_with_split_retry,
    is_anthropic_oversize_error,
)
from shared.llm import LLMError

NOW = datetime(2026, 5, 5, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _ev(qid: int) -> TriageInput:
    return TriageInput(
        queue_id=qid,
        doc_id=f"doc:{qid}",
        doc_type="github.commit",
        source_system="github",
        title=f"Doc {qid}",
        author_id="alice",
        body="x",
        body_token_count=100,
    )


def _overflow_error(message: str) -> LLMError:
    """Build the LiteLLM-wrapped 400 the split-retry path keys off.
    Same overflow phrasings the legacy Anthropic SDK used.
    """
    return LLMError(message, status_code=400, provider="anthropic")


def _bad_request(message: str) -> LLMError:
    """Backwards-compat alias used by the existing tests; same shape."""
    return _overflow_error(message)


def _success_response(events: list[TriageInput]) -> SimpleNamespace:
    """LiteLLM-shaped response carrying a tool_call that scores every
    event in `events`.
    """
    payload = {
        "verdicts": {
            str(ev.queue_id): {
                "important": True,
                "score": 7.0,
                "reason": "ok",
            }
            for ev in events
        }
    }
    func = SimpleNamespace(
        name="record_triage",
        arguments=orjson.dumps(payload).decode("utf-8"),
    )
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


def _gemini_success_response(events: list[TriageInput]) -> SimpleNamespace:
    """Gemini structured-output response: JSON text in message.content."""
    payload = {
        "verdicts": {
            str(ev.queue_id): {
                "important": True,
                "score": 7.0,
                "reason": "ok",
            }
            for ev in events
        }
    }
    message = SimpleNamespace(
        content=orjson.dumps(payload).decode("utf-8"),
        tool_calls=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _gemini_malformed_json_response() -> SimpleNamespace:
    """Gemini response_schema response with invalid JSON content."""
    message = SimpleNamespace(
        content=(
            '{"verdicts":{"0":{"important":true,"score":7.0,"reason":"ok"}'
            '"1":{"important":true,"score":7.0,"reason":"ok"}}}'
        ),
        tool_calls=None,
    )
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _empty_tool_call_response() -> SimpleNamespace:
    """Forced tool call with empty `{}` arguments → Pydantic surfaces
    `verdicts: Field required`. Mirrors Haiku stopping at max_tokens
    mid tool-use payload."""
    func = SimpleNamespace(name="record_triage", arguments="{}")
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


def _no_tool_call_response() -> SimpleNamespace:
    """Model returned text only, no tool_calls — Haiku stopped at
    max_tokens before opening the tool_use at all."""
    message = SimpleNamespace(content="...", tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def _partial_success_response(
    events_to_score: list[TriageInput],
) -> SimpleNamespace:
    """LiteLLM-shaped response that scores ONLY `events_to_score` —
    a partial response that omits whichever input events aren't in
    this list. Mirrors what Haiku does when max_tokens cuts it off."""
    payload = {
        "verdicts": {
            str(ev.queue_id): {
                "important": True,
                "score": 7.0,
                "reason": "ok",
            }
            for ev in events_to_score
        }
    }
    func = SimpleNamespace(
        name="record_triage",
        arguments=orjson.dumps(payload).decode("utf-8"),
    )
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


def _dict_type_error_response() -> SimpleNamespace:
    """Tool call whose arguments are valid JSON but the `verdicts`
    field is a JSON-STRING instead of a dict — Pydantic raises
    type=dict_type with input_type=str. Same overflow-shaped recovery
    applies."""
    args = orjson.dumps(
        {
            "verdicts": (
                '{"1": {"important": false, "score": 0, "reason": "n/a"}}'
            )
        }
    ).decode("utf-8")
    func = SimpleNamespace(name="record_triage", arguments=args)
    call = SimpleNamespace(type="function", function=func)
    message = SimpleNamespace(content=None, tool_calls=[call])
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    return SimpleNamespace(choices=[choice], usage=None)


def _patch_acompletion(monkeypatch, responses: list[object]) -> AsyncMock:
    """Patch `shared.llm_tools.acompletion` to return / raise the given
    sequence of responses. Each entry is either a response object
    (returned) or an Exception subclass instance (raised). Returns the
    AsyncMock so the test can assert call counts.
    """
    iterator = iter(responses)

    async def _create(**kwargs):
        item = next(iterator)
        if isinstance(item, BaseException):
            raise item
        return item

    fake = AsyncMock(side_effect=_create)
    monkeypatch.setattr("shared.llm_tools.acompletion", fake)
    return fake


def _patch_shared_acompletion(monkeypatch, responses: list[object]) -> AsyncMock:
    """Patch `shared.llm.acompletion` for Gemini provider tests."""
    iterator = iter(responses)

    async def _create(*args, **kwargs):
        item = next(iterator)
        if isinstance(item, BaseException):
            raise item
        return item

    fake = AsyncMock(side_effect=_create)
    monkeypatch.setattr("shared.llm.acompletion", fake)
    return fake


@pytest.fixture(autouse=True)
def _force_anthropic_triage_for_split_retry(monkeypatch) -> None:
    """These tests mostly exercise Anthropic tool-call parser shapes.

    The production default is Gemini now, so pin Haiku for the legacy
    split-retry cases and override explicitly in Gemini-specific tests.
    """
    monkeypatch.setattr("services.synthesis.providers.WIKI_TRIAGE_MODEL", "haiku")


# ---------------------------------------------------------------------------
# is_anthropic_oversize_error — detector unit tests
# ---------------------------------------------------------------------------


def test_detector_matches_prompt_too_long() -> None:
    err = _overflow_error("prompt is too long: 211739 tokens > 200000 maximum")
    assert is_anthropic_oversize_error(err) is True


def test_detector_matches_tokens_maximum_phrasing() -> None:
    err = _overflow_error("input has 250000 tokens > 200000 maximum context window")
    assert is_anthropic_oversize_error(err) is True


def test_detector_skips_unrelated_400() -> None:
    err = _overflow_error("invalid api key")
    assert is_anthropic_oversize_error(err) is False


def test_detector_skips_non_bad_request() -> None:
    assert is_anthropic_oversize_error(RuntimeError("boom")) is False


# ---------------------------------------------------------------------------
# Split-on-overflow behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_batch_splits_into_two_successful_halves(monkeypatch) -> None:
    """4 events: full batch overflows once, both halves succeed."""
    events = [_ev(i) for i in range(4)]
    overflow = _overflow_error("prompt is too long: 211739 tokens > 200000 maximum")
    fake = _patch_acompletion(
        monkeypatch,
        [
            overflow,
            _success_response(events[:2]),
            _success_response(events[2:]),
        ],
    )

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    assert isinstance(out, TriageOutput)
    assert sorted(int(k) for k in out.verdicts) == [0, 1, 2, 3]
    # 1 failed call + 2 successful calls = 3 wire calls total.
    assert fake.await_count == 3


@pytest.mark.asyncio
async def test_repeated_overflows_recurse_to_single_events(monkeypatch) -> None:
    """8 events: full batch overflows, first half overflows, first
    quarter overflows. Eventually every leaf either succeeds or gets
    surfaced as a rejected single-event verdict.
    """
    events = [_ev(i) for i in range(8)]
    overflow = _overflow_error("prompt is too long: 250000 tokens > 200000 maximum")

    fake = _patch_acompletion(
        monkeypatch,
        [
            overflow,                          # [0..7]
            overflow,                          # [0..3]
            overflow,                          # [0,1]
            overflow,                          # [0] alone -> rejected, no further call
            _success_response([events[1]]),    # [1]
            _success_response(events[2:4]),    # [2,3]
            _success_response(events[4:8]),    # [4..7]
        ],
    )

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    # All 8 queue_ids must appear in the merged output.
    assert sorted(int(k) for k in out.verdicts) == list(range(8))
    # qid 0 should be tagged with the call-time-oversize reason.
    assert out.verdicts["0"].important is False
    assert out.verdicts["0"].score == 0.0
    assert out.verdicts["0"].reason == OVERSIZED_AT_CALL_TIME_REASON
    # The other 7 came back as scored success verdicts.
    for qid in range(1, 8):
        assert out.verdicts[str(qid)].important is True
    assert fake is not None


@pytest.mark.asyncio
async def test_single_event_overflow_marked_rejected(monkeypatch) -> None:
    """A single-event batch that overflows is returned as a rejected
    verdict tagged with the call-time-oversize reason; no recursion
    happens beyond this leaf."""
    events = [_ev(42)]
    overflow = _overflow_error("prompt is too long: 220000 tokens > 200000 maximum")
    fake = _patch_acompletion(monkeypatch, [overflow])

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    assert list(out.verdicts.keys()) == ["42"]
    verdict = out.verdicts["42"]
    assert verdict.important is False
    assert verdict.score == 0.0
    assert verdict.reason == OVERSIZED_AT_CALL_TIME_REASON
    # Exactly one wire call — no retry on single-event leaf.
    assert fake.await_count == 1


@pytest.mark.asyncio
async def test_non_oversize_400_propagates(monkeypatch) -> None:
    """A 400 that is NOT the oversize-prompt variant (e.g. bad api key)
    must propagate unchanged. We must not split-retry on it — that would
    just multiply the failures."""
    events = [_ev(1), _ev(2), _ev(3)]
    other_400 = _overflow_error("authentication_error: invalid x-api-key header")
    fake = _patch_acompletion(monkeypatch, [other_400])

    with pytest.raises(LLMError) as exc_info:
        await call_triage_with_split_retry(object(), events, now=NOW)
    assert "invalid x-api-key" in str(exc_info.value)
    # No retry: the wrapper saw a non-oversize 400 and re-raised.
    assert fake.await_count == 1


@pytest.mark.asyncio
async def test_non_llm_error_exception_propagates(monkeypatch) -> None:
    """Non-LLMError exceptions (network, 5xx that escaped wrapping) must
    not trigger split-retry either. (Note: a TriageParseError whose
    message matches the overflow signature DOES trigger a split — see
    the parse-overflow section below.)"""
    events = [_ev(1), _ev(2)]
    fake = _patch_acompletion(monkeypatch, [RuntimeError("kaboom")])

    with pytest.raises(RuntimeError, match="kaboom"):
        await call_triage_with_split_retry(object(), events, now=NOW)
    assert fake.await_count == 1


# ---------------------------------------------------------------------------
# Output-side overflow: max_tokens-truncated response triggers split too
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_failure_empty_input_triggers_split(monkeypatch) -> None:
    """LiteLLM returns a 200 with an empty `{}` tool_call arguments —
    that means max_tokens cut Haiku off mid-output. Same recovery as a
    400: halve and retry."""
    events = [_ev(i) for i in range(4)]
    fake = _patch_acompletion(
        monkeypatch,
        [
            _empty_tool_call_response(),    # full batch fails Pydantic
            _success_response(events[:2]),  # left half: ok
            _success_response(events[2:]),  # right half: ok
        ],
    )
    out = await call_triage_with_split_retry(object(), events, now=NOW)
    assert isinstance(out, TriageOutput)
    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    assert fake.await_count == 3


@pytest.mark.asyncio
async def test_parse_failure_no_tool_use_triggers_split(monkeypatch) -> None:
    """Same overflow shape, different SDK surface: no tool_calls at
    all. The parser raises 'response had no record_triage tool_use
    block', which the wrapper recognizes as overflow."""
    events = [_ev(i) for i in range(4)]
    fake = _patch_acompletion(
        monkeypatch,
        [
            _no_tool_call_response(),
            _success_response(events[:2]),
            _success_response(events[2:]),
        ],
    )
    out = await call_triage_with_split_retry(object(), events, now=NOW)
    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    assert fake.await_count == 3


@pytest.mark.asyncio
async def test_single_event_parse_overflow_marked_rejected(monkeypatch) -> None:
    """Bottom-out: 1 event whose response is empty/truncated gets the
    same call-time-oversize rejection as a 400 single-event leaf."""
    events = [_ev(99)]
    fake = _patch_acompletion(monkeypatch, [_empty_tool_call_response()])
    out = await call_triage_with_split_retry(object(), events, now=NOW)
    assert list(out.verdicts.keys()) == ["99"]
    assert out.verdicts["99"].reason == OVERSIZED_AT_CALL_TIME_REASON
    assert fake.await_count == 1


# ---------------------------------------------------------------------------
# Partial response: parse succeeds but verdicts cover only some events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_partial_response_recurses_on_missing_only(monkeypatch) -> None:
    """4 events sent. Haiku scored only events 0,1 (truncation cut off
    2,3). Wrapper detects the 2 missing qids and recurses on JUST those
    two (not the full batch — the verdicts we got are kept).
    """
    events = [_ev(i) for i in range(4)]
    fake = _patch_acompletion(
        monkeypatch,
        [
            _partial_success_response([events[0], events[1]]),
            _success_response([events[2], events[3]]),
        ],
    )
    out = await call_triage_with_split_retry(object(), events, now=NOW)
    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    # 2 calls only: the original + 1 recurse on the missing subset.
    assert fake.await_count == 2


@pytest.mark.asyncio
async def test_partial_response_single_missing_marked_rejected(monkeypatch) -> None:
    """Bottom-out for partial path: 1 event in the recurse subset and
    Haiku again returns no verdict for it -> mark rejected with the
    same call-time-oversize reason. Same recovery as the 400 leaf."""
    events = [_ev(99)]
    # Haiku returns a verdicts dict with NOTHING (skipped the 1 input).
    fake = _patch_acompletion(monkeypatch, [_partial_success_response([])])
    out = await call_triage_with_split_retry(object(), events, now=NOW)
    assert list(out.verdicts.keys()) == ["99"]
    assert out.verdicts["99"].reason == OVERSIZED_AT_CALL_TIME_REASON
    assert fake.await_count == 1


@pytest.mark.asyncio
async def test_empty_partial_response_splits_instead_of_retrying_same_batch(
    monkeypatch,
) -> None:
    """A parsed response with `verdicts={}` for a multi-event batch must
    split the full batch. Recursing on the missing subset would pass the
    identical event list again and can loop forever if the provider keeps
    returning an empty dict for that size.
    """
    events = [_ev(i) for i in range(4)]
    fake = _patch_acompletion(
        monkeypatch,
        [
            _partial_success_response([]),
            _success_response(events[:2]),
            _success_response(events[2:]),
        ],
    )

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    assert fake.await_count == 3


@pytest.mark.asyncio
async def test_partial_response_with_recurse_into_further_partial(monkeypatch) -> None:
    """Recurse can itself produce partial responses. 8 events; first
    call scores 3, recurse on missing 5 scores 2, recurse on missing 3
    scores all 3. Final output covers all 8."""
    events = [_ev(i) for i in range(8)]
    fake = _patch_acompletion(
        monkeypatch,
        [
            _partial_success_response(events[:3]),  # 0,1,2
            _partial_success_response([events[3], events[4]]),  # 3,4
            _success_response(events[5:]),  # 5,6,7
        ],
    )
    out = await call_triage_with_split_retry(object(), events, now=NOW)
    assert sorted(out.verdicts.keys(), key=int) == [str(i) for i in range(8)]
    assert fake.await_count == 3


# ---------------------------------------------------------------------------
# dict-type Pydantic error: Haiku returned `verdicts` as a JSON-string
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dict_type_pydantic_error_triggers_split(monkeypatch) -> None:
    """When verdicts arrives as str instead of dict, Pydantic raises
    type=dict_type. Wrapper now treats this as overflow-shaped and
    halves the batch."""
    events = [_ev(i) for i in range(4)]
    fake = _patch_acompletion(
        monkeypatch,
        [
            _dict_type_error_response(),     # full batch fails Pydantic
            _success_response(events[:2]),   # left half ok
            _success_response(events[2:]),   # right half ok
        ],
    )
    out = await call_triage_with_split_retry(object(), events, now=NOW)
    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    assert fake.await_count == 3


# ---------------------------------------------------------------------------
# string_too_long: defense-in-depth in case the model field_validator is
# ever removed.
# ---------------------------------------------------------------------------


def test_parse_overflow_matches_string_too_long_type_tag() -> None:
    """The literal `type=string_too_long` tag in a Pydantic v2 error
    message is recognized as overflow-shaped."""
    from services.synthesis.providers import TriageParseError
    from services.synthesis.triage import _is_parse_overflow_error

    msg = (
        "triage tool input failed validation: 1 validation error for "
        "TriageOutput verdicts.10471.reason\n  String should have at "
        "most 240 characters [type=string_too_long, "
        "input_value='Detailed architectural d...n and system "
        "ownership.', input_type=str]"
    )
    err = TriageParseError(msg)
    assert _is_parse_overflow_error(err) is True


def test_parse_overflow_matches_string_should_have_at_most_phrasing() -> None:
    """Match the human-readable phrasing too — Pydantic occasionally
    surfaces the constraint message without the `type=` tag depending on
    surrounding context."""
    from services.synthesis.providers import TriageParseError
    from services.synthesis.triage import _is_parse_overflow_error

    err = TriageParseError(
        "validation failed: String should have at most 240 characters"
    )
    assert _is_parse_overflow_error(err) is True


def test_parse_overflow_matches_gemini_json_decode_error() -> None:
    """Malformed Gemini JSON should trigger split-retry, not DLQ."""
    from services.synthesis.providers import TriageParseError
    from services.synthesis.triage import _is_parse_overflow_error

    err = TriageParseError(
        "gemini triage call failed: gemini response was not JSON: "
        "Expecting ',' delimiter: line 4 column 9487 (char 15970)"
    )
    assert _is_parse_overflow_error(err) is True


def test_parse_overflow_matches_gemini_timeout() -> None:
    """A stalled Gemini structured-output call should split-retry."""
    from services.synthesis.providers import TriageParseError
    from services.synthesis.triage import _is_parse_overflow_error

    err = TriageParseError(
        "gemini triage call failed: gemini call timed out after 120s"
    )
    assert _is_parse_overflow_error(err) is True


def test_parse_overflow_skips_unrelated_parse_error() -> None:
    """Sanity: an unrelated TriageParseError (e.g. wrong tool name) MUST
    NOT be classified as overflow-shaped — those errors should propagate
    instead of triggering a wasteful split-retry."""
    from services.synthesis.providers import TriageParseError
    from services.synthesis.triage import _is_parse_overflow_error

    err = TriageParseError("response had unexpected tool name 'foo'")
    assert _is_parse_overflow_error(err) is False


@pytest.mark.asyncio
async def test_gemini_malformed_json_triggers_split_retry(monkeypatch) -> None:
    """Gemini response_schema can still return malformed JSON.

    The wrapper should classify that as output-side overflow/deviation,
    split the batch, and preserve sibling verdicts instead of DLQ'ing
    the whole customer.
    """
    monkeypatch.setattr(
        "services.synthesis.providers.WIKI_TRIAGE_MODEL",
        "gemini-3.5-flash",
    )
    events = [_ev(i) for i in range(4)]
    fake = _patch_shared_acompletion(
        monkeypatch,
        [
            _gemini_malformed_json_response(),
            _gemini_success_response(events[:2]),
            _gemini_success_response(events[2:]),
        ],
    )

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    assert fake.await_count == 3


@pytest.mark.asyncio
async def test_gemini_timeout_triggers_split_retry(monkeypatch) -> None:
    """A timed-out Gemini full batch should split instead of DLQ."""
    monkeypatch.setattr(
        "services.synthesis.providers.WIKI_TRIAGE_MODEL",
        "gemini-3.5-flash",
    )
    events = [_ev(i) for i in range(4)]
    fake = _patch_shared_acompletion(
        monkeypatch,
        [
            RuntimeError("gemini call timed out after 120s"),
            _gemini_success_response(events[:2]),
            _gemini_success_response(events[2:]),
        ],
    )

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    assert fake.await_count == 3


# ---------------------------------------------------------------------------
# Hardening: every input qid must have a verdict on every successful return.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_recursive_split_half_exception_preserves_sibling_verdicts(
    monkeypatch,
) -> None:
    """4 events: full batch overflows, left half succeeds, right half
    raises a generic RuntimeError. Expected: 4 verdicts — 2 real from
    left, 2 synthesized rejections for right tagged
    TRIAGE_RECURSION_FAILED_REASON. The exception MUST NOT bubble and
    discard the left half's progress."""
    events = [_ev(i) for i in range(4)]
    overflow = _overflow_error("prompt is too long: 211739 tokens > 200000 maximum")
    _patch_acompletion(
        monkeypatch,
        [
            overflow,                          # full batch [0..3]
            _success_response(events[:2]),     # left half [0,1] ok
            RuntimeError("transient 500"),     # right half [2,3] dies
        ],
    )

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    # All 4 input qids must be covered — no exception, no missing keys.
    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    # Left half: real verdicts.
    assert out.verdicts["0"].important is True
    assert out.verdicts["1"].important is True
    # Right half: synthesized rejections.
    assert out.verdicts["2"].important is False
    assert out.verdicts["2"].score == 0.0
    assert out.verdicts["2"].reason == TRIAGE_RECURSION_FAILED_REASON
    assert out.verdicts["3"].important is False
    assert out.verdicts["3"].score == 0.0
    assert out.verdicts["3"].reason == TRIAGE_RECURSION_FAILED_REASON


@pytest.mark.asyncio
async def test_partial_response_recursion_exception_preserves_parent_verdicts(
    monkeypatch,
) -> None:
    """4 events: parent call returns verdicts for events 0,1 (partial).
    Recursive call on the missing subset [2,3] raises a generic
    exception. Expected: 4 verdicts — 2 real for 0,1 plus 2 synthesized
    rejections for 2,3 tagged TRIAGE_RECURSION_FAILED_REASON."""
    events = [_ev(i) for i in range(4)]
    _patch_acompletion(
        monkeypatch,
        [
            # Parent call: scored 0,1 only (truncation cut off 2,3).
            _partial_success_response([events[0], events[1]]),
            # Recursive call on [2,3] dies.
            ConnectionError("network blip"),
        ],
    )

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    assert sorted(out.verdicts.keys()) == ["0", "1", "2", "3"]
    # Parent verdicts kept.
    assert out.verdicts["0"].important is True
    assert out.verdicts["1"].important is True
    # Synthesized rejections for the failed recursion.
    assert out.verdicts["2"].reason == TRIAGE_RECURSION_FAILED_REASON
    assert out.verdicts["3"].reason == TRIAGE_RECURSION_FAILED_REASON
    assert out.verdicts["2"].important is False
    assert out.verdicts["3"].important is False


@pytest.mark.asyncio
async def test_final_mile_guard_fires_when_recurse_returns_wrong_keys(
    monkeypatch,
) -> None:
    """Force the final-mile guard to fire. Patch `_safe_recurse` to
    return a TriageOutput whose verdicts are keyed on something OTHER
    than the input qids — simulating an unforeseen downstream bug.

    Expected: the orphaned input qid gets a synthesized rejection
    tagged TRIAGE_FINAL_MILE_SYNTHESIS_REASON; the wrong-keyed verdicts
    from the recurse may still be in the merged output (the worker
    filters by input qid downstream).
    """
    from services.synthesis import triage as triage_module
    from services.synthesis.models import TriageOutput as TO
    from services.synthesis.models import TriageVerdict as TV

    events = [_ev(i) for i in range(3)]
    # Parent call_triage returns partial: verdicts for 0,1; missing 2.
    _patch_acompletion(
        monkeypatch,
        [_partial_success_response([events[0], events[1]])],
    )

    async def _broken_recurse(
        client: object, sub_events: list[TriageInput], *, now: object
    ) -> TO:
        return TO(
            verdicts={
                "WRONG_KEY": TV(important=True, score=7.0, reason="bug")
            }
        )

    monkeypatch.setattr(triage_module, "_safe_recurse", _broken_recurse)

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    # All three input qids covered.
    assert "0" in out.verdicts
    assert "1" in out.verdicts
    assert "2" in out.verdicts
    # qid 2 is the orphan — only the final-mile guard could have
    # rescued it.
    assert out.verdicts["2"].reason == TRIAGE_FINAL_MILE_SYNTHESIS_REASON
    assert out.verdicts["2"].important is False
    assert out.verdicts["2"].score == 0.0
    # Real verdicts kept.
    assert out.verdicts["0"].important is True
    assert out.verdicts["1"].important is True


@pytest.mark.asyncio
async def test_final_mile_guard_no_op_on_happy_path(monkeypatch) -> None:
    """Sanity: a fully-covered single-call success returns unchanged.
    No synthesized rows, no extra log spam."""
    events = [_ev(i) for i in range(3)]
    fake = _patch_acompletion(monkeypatch, [_success_response(events)])

    out = await call_triage_with_split_retry(object(), events, now=NOW)

    assert sorted(out.verdicts.keys()) == ["0", "1", "2"]
    # All real success verdicts — no synthesized rejections.
    for qid in ["0", "1", "2"]:
        assert out.verdicts[qid].important is True
        assert out.verdicts[qid].reason not in (
            TRIAGE_FINAL_MILE_SYNTHESIS_REASON,
            TRIAGE_RECURSION_FAILED_REASON,
            OVERSIZED_AT_CALL_TIME_REASON,
        )
    assert fake.await_count == 1
