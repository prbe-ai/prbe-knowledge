"""Unit tests for AgentLoop.

Tests the harness with stub runtime + LLM. Mocked Gemini SDK so we
exercise turn loop / dispatch / compaction / stall / cap halts without
any live model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from services.synthesis.agent_harness import AgentLoop, AgentMetrics
from shared.constants import (
    WIKI_AGENT_STALL_TURNS,
    WIKI_AGENT_TURN_CAP,
    WIKI_AGENT_UPDATE_CAP,
)
from shared.exceptions import AgentCompactionError, AgentHaltError, ToolValidationError

# ---------------------------------------------------------------------------
# Stub runtime + LLM
# ---------------------------------------------------------------------------


class StubRuntime:
    """Minimal runtime — records dispatch calls, tracks done flag."""

    def __init__(
        self,
        *,
        customer_id: str = "stub-cust",
        agent_run_id: str = "run-1",
        index: list[dict[str, Any]] | None = None,
        manifest: dict[str, Any] | None = None,
        tool_handlers: dict[str, Any] | None = None,
    ) -> None:
        self.customer_id = customer_id
        self.agent_run_id = agent_run_id
        self.is_done = False
        self.pending_update_count = 0
        self._index = index or []
        self._manifest = manifest or {"events": [], "remaining": 0, "drain_complete": True}
        self._handlers: dict[str, Any] = tool_handlers or {}
        self.dispatch_calls: list[tuple[str, dict[str, Any]]] = []

    async def dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.dispatch_calls.append((name, args))
        handler = self._handlers.get(name)
        if handler is None:
            return {"status": "ok"}
        result = await handler(self, name, args) if callable(handler) else handler
        return result

    def state_snapshot_for_summary(self) -> dict[str, Any]:
        return {
            "pending_updates": [],
            "pending_creates": [],
            "applied_queue_ids": [],
            "skipped_queue_ids": [],
        }

    async def initial_manifest(self, count: int) -> dict[str, Any]:
        return self._manifest

    async def wiki_index(self) -> list[dict[str, Any]]:
        return self._index


class StubLLM:
    """Mock that yields a script of responses on each call."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self._responses = list(responses)
        self.create_cache_calls = 0
        self.generate_calls: list[dict[str, Any]] = []

    async def create_cache(
        self, *, system_instruction: str, tools: list[dict[str, Any]], seed_contents: list[dict[str, Any]]
    ) -> str:
        self.create_cache_calls += 1
        return "caches/stub-1"

    async def generate_with_cache(
        self,
        *,
        cache_name: str,
        contents: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]:
        self.generate_calls.append(
            {"cache_name": cache_name, "contents": list(contents), "tools": tools}
        )
        if not self._responses:
            return {"text": "", "tool_calls": [], "usage_metadata": {}}
        return self._responses.pop(0)


def _make_loop(
    runtime: StubRuntime,
    llm: StubLLM,
    *,
    summarizer: Any | None = None,
) -> AgentLoop:
    return AgentLoop(
        runtime=runtime,
        llm=llm,
        system_prompt="test sys",
        tool_schemas=[{"name": "done"}, {"name": "update_page"}],
        summarizer=summarizer,
    )


def _done_call() -> dict[str, Any]:
    return {
        "tool_calls": [{"name": "done", "args": {}}],
        "usage_metadata": {
            "prompt_token_count": 100,
            "cached_content_token_count": 900,
            "candidates_token_count": 50,
        },
    }


# ---------------------------------------------------------------------------
# run() happy path + halts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_happy_done() -> None:
    async def done_handler(rt: StubRuntime, name: str, args: dict[str, Any]) -> dict[str, Any]:
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_handler})
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm)
    metrics = await loop.run()
    assert isinstance(metrics, AgentMetrics)
    assert metrics.turns == 1
    assert metrics.gemini_call_count == 1
    assert ("done", {}) in runtime.dispatch_calls


@pytest.mark.asyncio
async def test_run_turn_cap_halts(monkeypatch) -> None:
    """Lower the turn cap so we can hit it cheaply."""
    import services.synthesis.agent_harness as h

    monkeypatch.setattr(h, "WIKI_AGENT_TURN_CAP", 2)

    runtime = StubRuntime()
    # 3 no-op model responses; loop never sees done().
    llm = StubLLM([
        {"text": "thinking", "tool_calls": [], "usage_metadata": {}},
        {"text": "still thinking", "tool_calls": [], "usage_metadata": {}},
        {"text": "more thinking", "tool_calls": [], "usage_metadata": {}},
    ])
    loop = _make_loop(runtime, llm)
    with pytest.raises(AgentHaltError) as exc_info:
        await loop.run()
    assert "turn_cap" in exc_info.value.reason


@pytest.mark.asyncio
async def test_run_stall_halts_after_threshold(monkeypatch) -> None:
    """No consequential tool for STALL_TURNS turns -> halt('agent.stall').

    Patches the threshold to a small value (3) for test speed — the
    production value (15) is high to accommodate normal read-heavy
    exploration before a decision. The behaviour we're pinning is
    threshold-relative, not the specific number.
    """
    import services.synthesis.agent_harness as h

    monkeypatch.setattr(h, "WIKI_AGENT_STALL_TURNS", 3)
    runtime = StubRuntime()
    llm = StubLLM(
        [
            {"text": "thinking 1", "tool_calls": [], "usage_metadata": {}},
            {"text": "thinking 2", "tool_calls": [], "usage_metadata": {}},
            {"text": "thinking 3", "tool_calls": [], "usage_metadata": {}},
            {"text": "thinking 4", "tool_calls": [], "usage_metadata": {}},
        ]
    )
    loop = _make_loop(runtime, llm)
    with pytest.raises(AgentHaltError) as exc_info:
        await loop.run()
    assert "stall" in exc_info.value.reason
    assert loop.metrics.turns >= 3


@pytest.mark.asyncio
async def test_stall_threshold_tolerates_realistic_read_exploration(
    monkeypatch,
) -> None:
    """A realistic agent decision flow on a chunk of triaged events
    looks like:
        next_events -> list_wiki_pages -> read_page x3 ->
        get_event_body x2 -> update_page (CONSEQUENTIAL) -> done

    That's 7 non-consequential turns before the first decision. The
    production stall threshold (15) must NOT halt this — that's what
    bit run 105 on probe-founders when STALL_TURNS=3 was the default.
    """
    import services.synthesis.agent_harness as h

    # Production value 15. Don't patch — exercise the real default.
    assert h.WIKI_AGENT_STALL_TURNS >= 10, (
        "STALL_TURNS regressed below the realistic-exploration floor; "
        "raise it back to >= 10 so a normal read-heavy chunk doesn't "
        "halt mid-decision."
    )

    async def next_events_h(rt, n, a):
        return {"events": [{"queue_id": 1, "title": "x"}], "drain_complete": True}

    async def list_h(rt, n, a):
        return {"index": []}

    async def read_h(rt, n, a):
        return {"body": "page body"}

    async def body_h(rt, n, a):
        return {"body": "event body"}

    async def update_h(rt, n, a):
        return {"committed": True}

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(
        tool_handlers={
            "next_events": next_events_h,
            "list_wiki_pages": list_h,
            "read_page": read_h,
            "get_event_body": body_h,
            "update_page": update_h,
            "done": done_h,
        }
    )
    llm = StubLLM(
        [
            {"tool_calls": [{"name": "next_events", "args": {"count": 50}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "list_wiki_pages", "args": {}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "read_page", "args": {"slug": "a"}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "read_page", "args": {"slug": "b"}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "read_page", "args": {"slug": "c"}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "get_event_body", "args": {"queue_id": 1}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "get_event_body", "args": {"queue_id": 2}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "update_page", "args": {"slug": "a", "applied_queue_ids": [1]}}], "usage_metadata": {}},
            {"tool_calls": [{"name": "done", "args": {}}], "usage_metadata": {}},
        ]
    )
    loop = _make_loop(runtime, llm)
    # Should complete without raising.
    await loop.run()
    # update_page reached on turn 8 -> consequential counter resets;
    # done() ends the loop cleanly.
    assert runtime.is_done is True


@pytest.mark.asyncio
async def test_run_update_cap_halts(monkeypatch) -> None:
    import services.synthesis.agent_harness as h

    monkeypatch.setattr(h, "WIKI_AGENT_UPDATE_CAP", 1)

    runtime = StubRuntime()
    runtime.pending_update_count = 5  # over the cap from the start
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm)
    with pytest.raises(AgentHaltError) as exc_info:
        await loop.run()
    assert "update_cap" in exc_info.value.reason


# ---------------------------------------------------------------------------
# Dispatch behaviors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_single_tool_use() -> None:
    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm)
    await loop.run()
    assert runtime.dispatch_calls == [("done", {})]


@pytest.mark.asyncio
async def test_function_call_thought_signature_round_trips_to_conversation() -> None:
    """Regression for `agent.gemini_persistent_error` at turn 1+.

    Gemini 3.x emits a thought_signature on every function_call part.
    When the harness echoes the function_call back as part of the
    conversation history (turn 2's request), the SAME signature must
    be on the part. Otherwise Gemini 400s:

        'Function call is missing a thought_signature in functionCall
        parts. This is required for tools to work correctly...'

    The harness lifts the signature off `tool_call["thought_signature"]`
    (set by gemini_agent_client._extract_response) and attaches it to
    the synthesized model-turn part.
    """
    sig = b"opaque-bytes-from-gemini"

    async def read_h(rt, n, a):
        return {"body": "..."}

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"read_page": read_h, "done": done_h})
    llm = StubLLM(
        [
            {
                "tool_calls": [
                    {
                        "name": "read_page",
                        "args": {"slug": "auth"},
                        "thought_signature": sig,
                    }
                ],
                "usage_metadata": {},
            },
            {
                "tool_calls": [{"name": "done", "args": {}}],
                "usage_metadata": {},
            },
        ]
    )
    loop = _make_loop(runtime, llm)
    await loop.run()

    # The model-turn part for read_page must carry the same signature.
    model_turn = next(
        c for c in loop._conversation if c.get("role") == "model"
    )
    parts = model_turn["parts"]
    assert len(parts) == 1
    assert parts[0]["function_call"]["name"] == "read_page"
    assert parts[0]["thought_signature"] == sig


@pytest.mark.asyncio
async def test_missing_thought_signature_omitted_from_part() -> None:
    """When the SDK omitted thought_signature (e.g. older model,
    text-only response upstream), the harness must NOT include the
    key as None on the echoed part — the field is bytes-typed at
    Gemini's API and JSON-encoding None there would fail validation
    differently from omitting it entirely."""

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = StubLLM(
        [
            {
                # No thought_signature key (older Gemini, no signature).
                "tool_calls": [{"name": "done", "args": {}}],
                "usage_metadata": {},
            }
        ]
    )
    loop = _make_loop(runtime, llm)
    await loop.run()

    model_turn = next(
        c for c in loop._conversation if c.get("role") == "model"
    )
    part = model_turn["parts"][0]
    assert part["function_call"]["name"] == "done"
    assert "thought_signature" not in part


@pytest.mark.asyncio
async def test_dispatch_multiple_tool_use_in_one_response() -> None:
    """Model emits multiple tool calls in one turn — all dispatched."""

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    async def read_h(rt, n, a):
        return {"body": "page body"}

    runtime = StubRuntime(tool_handlers={"done": done_h, "read_page": read_h})
    llm = StubLLM(
        [
            {
                "tool_calls": [
                    {"name": "read_page", "args": {"slug": "a"}},
                    {"name": "done", "args": {}},
                ],
                "usage_metadata": {},
            }
        ]
    )
    loop = _make_loop(runtime, llm)
    await loop.run()
    names = [c[0] for c in runtime.dispatch_calls]
    assert names == ["read_page", "done"]


@pytest.mark.asyncio
async def test_dispatch_tool_handler_raises_returns_typed_error() -> None:
    """Tool handler raises -> harness sends a typed error result back; loop continues."""

    async def boom(rt, n, a):
        raise RuntimeError("boom")

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"read_page": boom, "done": done_h})
    llm = StubLLM(
        [
            {"tool_calls": [{"name": "read_page", "args": {}}], "usage_metadata": {}},
            _done_call(),
        ]
    )
    loop = _make_loop(runtime, llm)
    metrics = await loop.run()
    assert metrics.turns == 2
    # The function_response from the failing tool should be in the
    # conversation as an error result (not raise out of the loop).
    fr_msg = loop._conversation[1]
    parts = fr_msg["parts"]
    assert parts[0]["function_response"]["response"]["error"] == "tool_exception"


@pytest.mark.asyncio
async def test_dispatch_unknown_tool_name() -> None:
    """An unknown tool falls through to the runtime's default handler
    (StubRuntime returns ok). In production WikiAgentRuntime raises
    ToolValidationError; here we just verify the harness routes it."""

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = StubLLM(
        [
            {"tool_calls": [{"name": "not_a_tool", "args": {}}], "usage_metadata": {}},
            _done_call(),
        ]
    )
    loop = _make_loop(runtime, llm)
    await loop.run()
    assert ("not_a_tool", {}) in runtime.dispatch_calls


@pytest.mark.asyncio
async def test_dispatch_tool_validation_error_continues_loop() -> None:
    """ToolValidationError -> typed error result, loop continues."""

    async def bad_args(rt, n, a):
        raise ToolValidationError("bad args")

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(
        tool_handlers={"update_page": bad_args, "done": done_h}
    )
    llm = StubLLM(
        [
            {"tool_calls": [{"name": "update_page", "args": {}}], "usage_metadata": {}},
            _done_call(),
        ]
    )
    loop = _make_loop(runtime, llm)
    metrics = await loop.run()
    assert metrics.turns == 2


# ---------------------------------------------------------------------------
# call_llm behaviors (cached content + retries + persistent error)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_llm_happy_with_cached_content() -> None:
    """Cache is created in build_initial_cache; generate_with_cache uses it."""

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm)
    await loop.run()
    assert llm.create_cache_calls == 1
    assert llm.generate_calls[0]["cache_name"] == "caches/stub-1"


@pytest.mark.asyncio
async def test_call_llm_gemini_timeout_retries_via_tenacity() -> None:
    """First two generate calls fail; third succeeds. The harness
    retries via tenacity and the loop completes."""

    class FlakyLLM(StubLLM):
        def __init__(self):
            super().__init__([])
            self.attempts = 0

        async def generate_with_cache(self, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise TimeoutError("flaky")
            return _done_call()

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = FlakyLLM()
    loop = _make_loop(runtime, llm)
    await loop.run()
    assert llm.attempts == 3


@pytest.mark.asyncio
async def test_call_llm_persistent_error_raises_AgentHaltError() -> None:
    """Tenacity exhaustion -> AgentHaltError('agent.gemini_persistent_error')."""

    class DeadLLM(StubLLM):
        def __init__(self):
            super().__init__([])

        async def generate_with_cache(self, **kwargs):
            raise TimeoutError("dead")

    runtime = StubRuntime()
    llm = DeadLLM()
    loop = _make_loop(runtime, llm)
    with pytest.raises(AgentHaltError) as exc:
        await loop.run()
    assert "gemini" in exc.value.reason


@pytest.mark.asyncio
async def test_call_llm_cache_miss_logs_warning() -> None:
    """If create_cache raises, the harness keeps going with empty cache_name."""

    class NoCacheLLM(StubLLM):
        async def create_cache(self, **kwargs):
            raise RuntimeError("cache rejected")

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = NoCacheLLM([_done_call()])
    loop = _make_loop(runtime, llm)
    await loop.run()
    assert llm.generate_calls[0]["cache_name"] == ""


# ---------------------------------------------------------------------------
# Compaction behaviors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_maybe_compact_below_threshold_no_op() -> None:
    """No summarizer call when token estimate < threshold."""
    summarizer = AsyncMock()

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm, summarizer=summarizer)
    await loop.run()
    assert summarizer.await_count == 0


@pytest.mark.asyncio
async def test_maybe_compact_above_threshold_triggers(monkeypatch) -> None:
    """Force the threshold low so the very-empty conversation crosses it."""
    import services.synthesis.agent_harness as h

    monkeypatch.setattr(h, "_MODEL_CONTEXT_TOKENS", 100)
    monkeypatch.setattr(h, "WIKI_AGENT_COMPACT_THRESHOLD", 0.0)

    summarizer = AsyncMock(return_value="compacted summary")

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm, summarizer=summarizer)
    await loop.run()
    assert summarizer.await_count >= 1
    assert loop.metrics.compaction_count >= 1


@pytest.mark.asyncio
async def test_maybe_compact_preserves_staged_state(monkeypatch) -> None:
    """Summarizer receives runtime_state; replacement conversation
    starts with the compacted text."""
    import services.synthesis.agent_harness as h

    monkeypatch.setattr(h, "_MODEL_CONTEXT_TOKENS", 100)
    monkeypatch.setattr(h, "WIKI_AGENT_COMPACT_THRESHOLD", 0.0)

    captured_state = {}

    async def summarize(messages, state):
        captured_state.update(state)
        return "compact summary text"

    async def done_h(rt, n, a):
        rt.is_done = True
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": done_h})
    runtime.pending_update_count = 0
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm, summarizer=summarize)
    await loop.run()
    # state_snapshot_for_summary returned by the stub has these keys.
    assert "pending_updates" in captured_state
    assert "applied_queue_ids" in captured_state


@pytest.mark.asyncio
async def test_maybe_compact_summarizer_fails_raises_AgentHaltError(monkeypatch) -> None:
    import services.synthesis.agent_harness as h

    monkeypatch.setattr(h, "_MODEL_CONTEXT_TOKENS", 100)
    monkeypatch.setattr(h, "WIKI_AGENT_COMPACT_THRESHOLD", 0.0)

    async def bad_summarize(messages, state):
        raise AgentCompactionError("summarizer crashed")

    runtime = StubRuntime()
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm, summarizer=bad_summarize)
    with pytest.raises(AgentHaltError) as exc:
        await loop.run()
    assert "compaction_failed" in exc.value.reason


# ---------------------------------------------------------------------------
# Sanity — turn cap constant and run terminate
# ---------------------------------------------------------------------------


def test_turn_cap_is_finite() -> None:
    """Sanity: the cap must be > stall threshold or stall-halt is dead code."""
    assert WIKI_AGENT_TURN_CAP > WIKI_AGENT_STALL_TURNS
    assert WIKI_AGENT_UPDATE_CAP >= 1


# ---------------------------------------------------------------------------
# CancelledError propagation regression guard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agent_loop_propagates_cancelled_error() -> None:
    """Required for the bootstrap force-cancel path to actually stop a
    running agent. The harness has three broad ``except Exception:``
    blocks (cache create, tool dispatch, summarizer); none must swallow
    ``asyncio.CancelledError``. ``CancelledError`` inherits from
    ``BaseException`` (not ``Exception``) on Python 3.8+ so that's the
    invariant being pinned here. If a future refactor changes any of
    those handlers to ``except BaseException:`` without a re-raise, this
    test fails.
    """
    import asyncio

    cancelled_inside_tool = asyncio.Event()
    waiter_started = asyncio.Event()

    async def slow_done_handler(rt: StubRuntime, name: str, args: dict[str, Any]) -> dict[str, Any]:
        # Park inside a tool dispatch; the outer task gets cancelled
        # while we're waiting here. The harness's per-tool
        # ``except Exception`` must let CancelledError propagate.
        waiter_started.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled_inside_tool.set()
            raise
        return {"committed": True}

    runtime = StubRuntime(tool_handlers={"done": slow_done_handler})
    llm = StubLLM([_done_call()])
    loop = _make_loop(runtime, llm)
    task = asyncio.create_task(loop.run())
    await asyncio.wait_for(waiter_started.wait(), timeout=2.0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cancelled_inside_tool.is_set(), (
        "the tool handler should have observed CancelledError; if this "
        "assertion fails the harness is swallowing CancelledError before "
        "the awaitable inside the tool sees it"
    )
