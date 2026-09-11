"""Hedging a slow turn: who wins, who gets cancelled, and what stays unchanged.

The whole risk of a hedge is that it changes behaviour for traffic that was
fine. So most of what is asserted here is that NOTHING happens: no second call
on a fast turn, none on a fast error, none once the run has already failed
over. The interesting cases are the two slow ones.

Timing is driven by a deliberately tiny hedge deadline and `asyncio.sleep`,
never by wall-clock guesses about a real provider.
"""

from __future__ import annotations

import asyncio

import pytest

from engine.retrieval.agent import loop as loop_mod
from engine.retrieval.agent.loop import LoopState


class _Resp:
    """Marker object standing in for a provider response."""

    def __init__(self, who: str) -> None:
        self.who = who


def _state() -> LoopState:
    return LoopState(customer_id="acme", trace_id="t-1", query="q")


@pytest.fixture
def fast_hedge(monkeypatch):
    """Hedge after 10ms so a 'slow' call is 50ms, not 5 seconds."""
    monkeypatch.setattr(loop_mod, "SEARCH_AGENT_HEDGE_AFTER_SECONDS", 0.01)
    monkeypatch.setattr(
        loop_mod, "SEARCH_AGENT_FALLBACK_INFERENCE_MODEL", "fallback/model"
    )


def _calls_recorded(monkeypatch, behaviour):
    """Patch acompletion with `behaviour(model) -> awaitable`, recording models."""
    seen: list[str] = []

    async def fake(**kwargs):
        seen.append(kwargs["model"])
        return await behaviour(kwargs["model"])

    monkeypatch.setattr(loop_mod, "acompletion", fake)
    return seen


@pytest.mark.asyncio
async def test_a_fast_primary_never_starts_a_second_call(fast_hedge, monkeypatch):
    """The 96% case. One call, no hedge, no duplicate tokens."""

    async def behaviour(model):
        return _Resp(model)

    seen = _calls_recorded(monkeypatch, behaviour)
    state = _state()

    resp = await loop_mod._acompletion_hedged({"model": "primary/model"}, state)

    assert resp.who == "primary/model"
    assert seen == ["primary/model"]
    assert state.hedge_fired == 0
    assert state.llm_failed_over is False


@pytest.mark.asyncio
async def test_a_slow_primary_that_still_answers_first_wins(fast_hedge, monkeypatch):
    """Hedge fired, primary won: the run must NOT be marked failed over.

    This is the case the old 5s cut got wrong -- it abandoned a turn that was
    going to return, and paid for it on every later turn of the run.
    """

    async def behaviour(model):
        if model == "primary/model":
            await asyncio.sleep(0.05)
            return _Resp(model)
        await asyncio.sleep(5)  # fallback never gets there
        return _Resp(model)

    seen = _calls_recorded(monkeypatch, behaviour)
    state = _state()

    resp = await loop_mod._acompletion_hedged({"model": "primary/model"}, state)

    assert resp.who == "primary/model"
    assert seen == ["primary/model", "fallback/model"]
    assert state.hedge_fired == 1
    assert state.hedge_fallback_won == 0
    assert state.llm_failed_over is False, "a won race is not a failover"
    assert state.hedge_discarded_ms > 0, "the cancelled fallback is a real bill"


@pytest.mark.asyncio
async def test_a_stalled_primary_loses_to_the_fallback(fast_hedge, monkeypatch):
    """The case this exists for: the 12s stall answered in ~50ms instead."""
    cancelled = asyncio.Event()

    async def behaviour(model):
        if model == "primary/model":
            try:
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return _Resp(model)
        await asyncio.sleep(0.05)
        return _Resp(model)

    _calls_recorded(monkeypatch, behaviour)
    state = _state()

    resp = await loop_mod._acompletion_hedged({"model": "primary/model"}, state)

    assert resp.who == "fallback/model"
    assert state.hedge_fired == 1
    assert state.hedge_fallback_won == 1
    assert cancelled.is_set(), "the stalled primary must be cancelled, not orphaned"
    assert state.llm_failed_over is True
    assert state.llm_model == "fallback/model"


@pytest.mark.asyncio
async def test_the_fallback_call_keeps_the_primarys_payload(fast_hedge, monkeypatch):
    """Only model and timeout differ -- same messages, same tools, same cap."""
    captured: list[dict] = []

    async def fake(**kwargs):
        captured.append(dict(kwargs))
        if kwargs["model"] == "primary/model":
            await asyncio.sleep(30)
        await asyncio.sleep(0.05)
        return _Resp(kwargs["model"])

    monkeypatch.setattr(loop_mod, "acompletion", fake)

    payload = {
        "model": "primary/model",
        "timeout": 12.0,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16000,
        "seed": 7,
    }
    await loop_mod._acompletion_hedged(payload, _state())

    primary, fallback = captured
    assert fallback["model"] == "fallback/model"
    assert fallback["timeout"] == loop_mod.SEARCH_AGENT_FALLBACK_TIMEOUT_SECONDS
    for key in ("messages", "max_tokens", "seed"):
        assert fallback[key] == primary[key]


@pytest.mark.asyncio
async def test_hedging_does_not_mutate_the_callers_kwargs(fast_hedge, monkeypatch):
    """`_run_turn` reuses call_kwargs on its failover path -- don't corrupt it."""

    async def fake(**kwargs):
        if kwargs["model"] == "primary/model":
            await asyncio.sleep(30)
        await asyncio.sleep(0.05)
        return _Resp(kwargs["model"])

    monkeypatch.setattr(loop_mod, "acompletion", fake)
    payload = {"model": "primary/model", "timeout": 12.0}

    await loop_mod._acompletion_hedged(payload, _state())

    assert payload == {"model": "primary/model", "timeout": 12.0}


@pytest.mark.asyncio
async def test_a_fast_error_is_left_to_the_sequential_failover(
    fast_hedge, monkeypatch
):
    """A 400 does not deserve a concurrent second copy of itself.

    The exception must reach `_run_turn` untouched so its existing one-retry
    failover handles it -- and no hedge may be spent on the way.
    """

    async def behaviour(model):
        raise loop_mod.LLMError("bad request")

    seen = _calls_recorded(monkeypatch, behaviour)
    state = _state()

    with pytest.raises(loop_mod.LLMError):
        await loop_mod._acompletion_hedged({"model": "primary/model"}, state)

    assert seen == ["primary/model"]
    assert state.hedge_fired == 0
    assert state.llm_failed_over is False, "_run_turn owns that transition"


@pytest.mark.asyncio
async def test_both_failing_marks_failed_over_so_nothing_replays(
    fast_hedge, monkeypatch
):
    """Both providers down is an outage. One attempt each, then stop."""

    async def behaviour(model):
        if model == "primary/model":
            await asyncio.sleep(0.05)
        raise loop_mod.LLMError(f"{model} down")

    _calls_recorded(monkeypatch, behaviour)
    state = _state()

    with pytest.raises(loop_mod.LLMError):
        await loop_mod._acompletion_hedged({"model": "primary/model"}, state)

    assert state.llm_failed_over is True, (
        "without this the caller replays the fallback a third time"
    )
    # ONE entry, not two: the helper records the fallback, and the caller's
    # except-handler records the primary. See the next test, which checks the
    # pair actually adds up through `_run_turn`.
    assert len(state.failed_turn_latencies_ms) == 1


@pytest.mark.asyncio
async def test_two_failed_calls_are_billed_exactly_twice(fast_hedge, monkeypatch):
    """Regression: the hedge must not double-count into agent_failed_llm_ms.

    `_run_turn` appends a failed latency of its own, so a helper that also
    recorded the primary produced THREE entries for TWO calls -- inflating the
    one metric used to decide whether hedging was worth its duplicate tokens.
    """
    calls: list[str] = []

    async def fake(**kwargs):
        calls.append(kwargs["model"])
        if kwargs["model"] == "primary/model":
            await asyncio.sleep(0.05)
        raise loop_mod.LLMError(f"{kwargs['model']} down")

    monkeypatch.setattr(loop_mod, "acompletion", fake)
    state = _state()

    with pytest.raises(loop_mod.LLMError):
        try:
            await loop_mod._acompletion_hedged({"model": "primary/model"}, state)
        except loop_mod.LLMError:
            # Stand in for `_run_turn`'s handler, which records the primary.
            state.failed_turn_latencies_ms.append(1.0)
            raise

    assert calls == ["primary/model", "fallback/model"]
    assert len(state.failed_turn_latencies_ms) == len(calls) == 2


@pytest.mark.asyncio
async def test_an_already_failed_over_run_does_not_hedge(fast_hedge, monkeypatch):
    """Nothing left to hedge ONTO -- the fallback is already the primary."""

    async def behaviour(model):
        await asyncio.sleep(0.05)
        return _Resp(model)

    seen = _calls_recorded(monkeypatch, behaviour)
    state = _state()
    state.llm_failed_over = True

    await loop_mod._acompletion_hedged({"model": "fallback/model"}, state)

    assert seen == ["fallback/model"]
    assert state.hedge_fired == 0


@pytest.mark.asyncio
async def test_no_fallback_configured_means_no_hedge(monkeypatch):
    """Self-hosted installs with one provider keep exactly today's behaviour."""
    monkeypatch.setattr(loop_mod, "SEARCH_AGENT_HEDGE_AFTER_SECONDS", 0.01)
    monkeypatch.setattr(loop_mod, "SEARCH_AGENT_FALLBACK_INFERENCE_MODEL", "")

    async def behaviour(model):
        await asyncio.sleep(0.05)
        return _Resp(model)

    seen = _calls_recorded(monkeypatch, behaviour)
    state = _state()

    await loop_mod._acompletion_hedged({"model": "primary/model"}, state)

    assert seen == ["primary/model"]
    assert state.hedge_fired == 0


def test_the_hedge_fires_well_above_the_healthy_p95() -> None:
    """Guard the sizing, not just the plumbing.

    Measured 2026-09-11 over 570 clean single-turn retrievals: p95 3,574ms,
    p99 4,839ms. A hedge at or below p95 would fire on healthy traffic and
    spend duplicate tokens for nothing, which is the failure mode that got the
    5s CUT reverted twice. It must also stay under the 12s deadline or it can
    never fire at all.
    """
    from engine.shared.constants import (
        SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS,
        SEARCH_AGENT_HEDGE_AFTER_SECONDS,
    )

    healthy_p95_s = 3.574
    assert healthy_p95_s < SEARCH_AGENT_HEDGE_AFTER_SECONDS
    assert SEARCH_AGENT_HEDGE_AFTER_SECONDS < SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_an_outer_cancellation_does_not_orphan_either_call(
    fast_hedge, monkeypatch
):
    """loop_timeout cancels us mid-race. Both providers must be let go.

    `_drive_loop` runs under `asyncio.wait_for(..., loop_budget)`, so this is a
    real path, not a hypothetical. Orphaning here would turn one stall into two
    calls still holding booked quota.
    """
    live: set[str] = set()

    async def fake(**kwargs):
        model = kwargs["model"]
        live.add(model)
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            live.discard(model)
            raise
        return _Resp(model)

    monkeypatch.setattr(loop_mod, "acompletion", fake)
    state = _state()

    race = asyncio.ensure_future(
        loop_mod._acompletion_hedged({"model": "primary/model"}, state)
    )
    await asyncio.sleep(0.05)  # let the hedge fire so both are in flight
    assert live == {"primary/model", "fallback/model"}

    race.cancel()
    with pytest.raises(asyncio.CancelledError):
        await race

    assert live == set(), "both provider calls must be cancelled, not orphaned"
