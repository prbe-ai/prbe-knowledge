"""Per-turn provider failover + whole-stage deadline.

The gatherer talks to Cerebras, whose latency is bimodal: 87.6% of turns
land in ~1s (p90 1.6s, max 3.9s) and 12.4% stall at 59.5-63.8s with nothing
in between. Waiting the stall out was the old strategy (70s per-turn
deadline); at a mean 2.23 turns/retrieval that compounds to ~30% of
retrievals degrading, and research-os hangs up at 30s regardless.

So: cut the primary at 5s and finish the run on the fallback provider. The
gateway's own route fallback does NOT do this -- stalled turns carry
`x-litellm-attempted-fallbacks: 0` -- so the loop owns the hop.

These tests drive `_run_turn` directly with a faked `acompletion` so the
failover is exercised as behaviour, not asserted as a constant.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from engine.retrieval.agent.loop import (
    _MIN_LOOP_BUDGET_SECONDS,
    LoopState,
    _remaining_loop_budget,
    _run_turn,
)
from engine.retrieval.agent.tools import TERMINAL_TOOL_NAME
from engine.shared.constants import (
    SEARCH_AGENT_FALLBACK_INFERENCE_MODEL,
    SEARCH_AGENT_FALLBACK_TIMEOUT_SECONDS,
    SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS,
    SEARCH_AGENT_INFERENCE_MODEL,
    SEARCH_AGENT_LOOP_TIMEOUT_SECONDS,
)
from engine.shared.llm import LLMError


def _resp() -> SimpleNamespace:
    """Minimal chat-completion response carrying one terminal tool call."""
    fn = SimpleNamespace(
        name=TERMINAL_TOOL_NAME,
        arguments=json.dumps({"entities": [], "chunks": [], "gatherer_notes": {}}),
    )
    msg = SimpleNamespace(
        tool_calls=[SimpleNamespace(id="t1", function=fn)],
        content=None,
        reasoning_content=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason="tool_calls")],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        system_fingerprint="fp_test",
    )


def _state() -> LoopState:
    st = LoopState(customer_id="cust-1", trace_id="tr-1", query="q")
    st.messages = [{"role": "user", "content": "q"}]
    return st


# ============================================================
# The stall cut is calibrated to the measured distribution
# ============================================================

def test_primary_deadline_separates_the_two_latency_modes() -> None:
    """5s sits above the healthy max (3.9s) and far below the stall floor
    (59.5s). Anything in 4..59 works; drifting outside that window either
    truncates healthy turns or stops cutting stalls."""
    assert 3.9 < SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS < 59.5


def test_stage_cap_leaves_room_for_a_multi_turn_run() -> None:
    """A 3-turn run worst case is 5s cut + 3 fallback turns; the stage cap
    must exceed that or the backstop becomes the routine path."""
    worst_turns = (
        SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS + 3 * SEARCH_AGENT_FALLBACK_TIMEOUT_SECONDS
    )
    assert worst_turns < SEARCH_AGENT_LOOP_TIMEOUT_SECONDS


# ============================================================
# Failover behaviour
# ============================================================

@pytest.mark.asyncio
async def test_primary_stall_fails_over_and_completes_the_turn() -> None:
    """A timeout on Cerebras must not surface as a degraded turn: the same
    turn is retried once on the fallback and returns a real response."""
    st = _state()
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            raise LLMError("litellm.Timeout: timeout value=5.0, time taken=5.01")
        return _resp()

    with patch("engine.retrieval.agent.loop.acompletion", new=AsyncMock(side_effect=fake)):
        resp = await _run_turn(st)

    assert resp is not None
    assert len(calls) == 2, "expected exactly one retry"
    assert calls[0]["model"] == SEARCH_AGENT_INFERENCE_MODEL
    assert calls[1]["model"] == SEARCH_AGENT_FALLBACK_INFERENCE_MODEL
    assert st.llm_failed_over is True
    assert st.llm_model == SEARCH_AGENT_FALLBACK_INFERENCE_MODEL


@pytest.mark.asyncio
async def test_failover_is_sticky_for_the_rest_of_the_run() -> None:
    """The whole point: turn 2 goes straight to the fallback. Re-trying the
    primary each turn would re-pay the 5s cut on every one."""
    st = _state()
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            raise LLMError("litellm.Timeout: timeout value=5.0")
        return _resp()

    with patch("engine.retrieval.agent.loop.acompletion", new=AsyncMock(side_effect=fake)):
        await _run_turn(st)   # turn 1: stall -> failover
        await _run_turn(st)   # turn 2: must NOT touch the primary again

    models = [c["model"] for c in calls]
    assert models == [
        SEARCH_AGENT_INFERENCE_MODEL,
        SEARCH_AGENT_FALLBACK_INFERENCE_MODEL,
        SEARCH_AGENT_FALLBACK_INFERENCE_MODEL,
    ]
    assert models.count(SEARCH_AGENT_INFERENCE_MODEL) == 1


@pytest.mark.asyncio
async def test_post_failover_turns_use_the_fallback_deadline() -> None:
    """5s is a Cerebras stall cut, not a universal SLA. Applying it to the
    fallback would cut healthy traffic on a provider without the stall."""
    st = _state()
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        if len(calls) == 1:
            raise LLMError("litellm.Timeout")
        return _resp()

    with patch("engine.retrieval.agent.loop.acompletion", new=AsyncMock(side_effect=fake)):
        await _run_turn(st)
        await _run_turn(st)

    assert calls[0]["timeout"] == SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS
    assert calls[1]["timeout"] == SEARCH_AGENT_FALLBACK_TIMEOUT_SECONDS
    assert calls[2]["timeout"] == SEARCH_AGENT_FALLBACK_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_fallback_failure_raises_instead_of_looping() -> None:
    """Both providers down is a real outage. Replaying a high-token turn a
    third time only spends the stage budget the degrade path needs."""
    st = _state()
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        raise LLMError("provider down")

    with (
        patch("engine.retrieval.agent.loop.acompletion", new=AsyncMock(side_effect=fake)),
        pytest.raises(LLMError),
    ):
        await _run_turn(st)

    assert len(calls) == 2, "one primary attempt + one fallback attempt, no more"


@pytest.mark.asyncio
async def test_no_failover_when_no_fallback_model_configured() -> None:
    """Self-hosted installs with a single provider must behave exactly as
    before: the error propagates, no phantom second call."""
    st = _state()
    calls: list[dict[str, Any]] = []

    async def fake(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        raise LLMError("litellm.Timeout")

    with (
        patch("engine.retrieval.agent.loop.SEARCH_AGENT_FALLBACK_INFERENCE_MODEL", ""),
        patch("engine.retrieval.agent.loop.acompletion", new=AsyncMock(side_effect=fake)),
        pytest.raises(LLMError),
    ):
        await _run_turn(st)

    assert len(calls) == 1
    assert st.llm_failed_over is False


# ============================================================
# Whole-stage deadline
# ============================================================

def test_setup_time_comes_out_of_the_stage_budget() -> None:
    """The cap covers the ENTIRE stage. Setup (grounding + extraction +
    pre-fan-out, ~4s) is subtracted, so the loop gets the remainder -- not a
    fresh full budget on top, which would make the real ceiling
    `setup + cap` and mislead any caller sizing its HTTP timeout off it."""
    spent = 4.0
    budget = _remaining_loop_budget(time.perf_counter() - spent)
    assert budget == pytest.approx(SEARCH_AGENT_LOOP_TIMEOUT_SECONDS - spent, abs=0.2)
    assert budget < SEARCH_AGENT_LOOP_TIMEOUT_SECONDS


def test_blown_setup_still_leaves_one_turn_rather_than_cancelling() -> None:
    """A non-positive wait_for timeout cancels the loop before it runs a
    single turn. Floor it so a pathological setup still degrades through the
    normal loop_timeout path, which backfills from the pre-fan-out."""
    budget = _remaining_loop_budget(
        time.perf_counter() - (SEARCH_AGENT_LOOP_TIMEOUT_SECONDS + 30)
    )
    assert budget == _MIN_LOOP_BUDGET_SECONDS
    assert budget > 0


@pytest.mark.asyncio
async def test_healthy_primary_never_touches_the_fallback() -> None:
    """87.6% of traffic. No extra call, no state flip."""
    st = _state()
    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_resp()),
    ) as m:
        await _run_turn(st)

    assert m.await_count == 1
    assert m.await_args.kwargs["model"] == SEARCH_AGENT_INFERENCE_MODEL
    assert st.llm_failed_over is False
