"""Wiki agent harness — Gemini 3.1 Pro turn loop with CachedContent + compaction.

The harness is provider-aware (Gemini-only) but runtime-agnostic — it
takes any object that exposes a `dispatch_tool(name, args) -> dict`
contract plus a few introspection hooks. `WikiAgentRuntime` in
`wiki_agent.py` is the only production runtime; tests pin a stub.

```
                AgentLoop.run()
                      |
                      v
              ___________________
             |                   |
             |  build cache      |
             |  - system prompt  |
             |  - tool defs      |
             |  - wiki index     |
             |  - first manifest |
             |___________________|
                      |
                      v   per turn
              ___________________
             |                   |
             |  maybe_compact()  | <-- Flash Lite summarizer
             |                   |
             |  call_llm()       | <-- Gemini Pro w/ CachedContent
             |                   |
             |  dispatch tools   | <-- runtime.dispatch_tool
             |                   |
             |  stall check      |
             |___________________|
                      |
              done() / halt -> exit
```

Halt conditions (any -> raise AgentHaltError, drain DLQs):
  * turn >= WIKI_AGENT_TURN_CAP
  * pending update count >= WIKI_AGENT_UPDATE_CAP
  * stalled (WIKI_AGENT_STALL_TURNS turns with no consequential tool)
  * call_llm tenacity-exhausted on Gemini errors
  * compactor failure
  * unknown tool name (caller bug)

State safety: snapshot-then-mutate — the runtime takes a snapshot
before each tool dispatch and restores on tool exception. The harness
just calls runtime.dispatch_tool; the runtime owns the snapshot.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from shared.constants import (
    WIKI_AGENT_BATCH_SIZE,
    WIKI_AGENT_COMPACT_THRESHOLD,
    WIKI_AGENT_MODEL,
    WIKI_AGENT_STALL_TURNS,
    WIKI_AGENT_TURN_CAP,
    WIKI_AGENT_UPDATE_CAP,
)
from shared.exceptions import AgentCompactionError, AgentHaltError, ToolValidationError
from shared.logging import get_logger

log = get_logger(__name__)

# Approximate token budget for the model (Gemini 3.1 Pro: 2M context).
# Used by _estimate_tokens to decide compaction. The threshold is a
# fraction of this; the actual prompt limit is enforced by Gemini.
_MODEL_CONTEXT_TOKENS = 2_000_000

# Approx tokens-per-character for the per-turn token estimator.
# Gemini's tokenizer averages ~4 chars per token for English; we use
# a conservative 3.5 to err on the side of compacting earlier.
_TOKENS_PER_CHAR = 1.0 / 3.5


# ---------------------------------------------------------------------------
# Public protocols + dataclasses
# ---------------------------------------------------------------------------


class _RuntimeProtocol(Protocol):
    """Minimum surface the harness needs from a runtime."""

    customer_id: str
    agent_run_id: str
    is_done: bool
    pending_update_count: int

    async def dispatch_tool(
        self, name: str, args: dict[str, Any]
    ) -> dict[str, Any]: ...

    def state_snapshot_for_summary(self) -> dict[str, Any]: ...

    async def initial_manifest(self, count: int) -> dict[str, Any]: ...

    async def wiki_index(self) -> list[dict[str, Any]]: ...


class _LLMClient(Protocol):
    """Minimum surface the harness needs from a Gemini client wrapper.

    `call_with_tools(messages, tools, cached_content_name) -> dict`
    returns the raw model output. The wrapper holds the cache + retries.
    """

    async def create_cache(
        self,
        *,
        system_instruction: str,
        tools: list[dict[str, Any]],
        seed_contents: list[dict[str, Any]],
    ) -> str: ...

    async def generate_with_cache(
        self,
        *,
        cache_name: str,
        contents: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class AgentMetrics:
    """Per-drain audit metrics.

    Captured into wiki_synthesis_runs after the drain by the worker.
    """

    turns: int = 0
    compaction_count: int = 0
    gemini_call_count: int = 0
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    consequential_turns: int = 0
    last_consequential_turn: int = 0
    halt_reason: str | None = None

    @property
    def cache_hit_rate(self) -> float | None:
        denom = self.total_input_tokens + self.total_cached_tokens
        if denom <= 0:
            return None
        return self.total_cached_tokens / denom


# ---------------------------------------------------------------------------
# AgentLoop
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AgentLoop:
    """One drain through the wiki agent for one customer.

    The loop is single-turn synchronous (no streaming). Each turn:
      1. maybe_compact()
      2. call_llm() with cache_name
      3. dispatch any tool calls in the response
      4. stall + cap checks; raise AgentHaltError on any tripwire
    """

    runtime: _RuntimeProtocol
    llm: _LLMClient
    system_prompt: str
    tool_schemas: list[dict[str, Any]]
    summarizer: Callable[..., Any] | None = None  # set to agent_compactor.call_summarizer
    model: str = WIKI_AGENT_MODEL

    # Mutable per-loop state.
    _conversation: list[dict[str, Any]] = field(default_factory=list)
    _cache_name: str = ""
    metrics: AgentMetrics = field(default_factory=AgentMetrics)

    async def run(self) -> AgentMetrics:
        """Drive the drain to either done() or AgentHaltError."""
        log.info(
            "agent.start",
            customer=self.runtime.customer_id,
            agent_run_id=self.runtime.agent_run_id,
            model=self.model,
        )
        await self._build_initial_cache()

        while True:
            self._check_caps()
            await self._maybe_compact()

            response = await self._call_llm()
            self.metrics.turns += 1

            consequential = await self._dispatch(response)
            if consequential:
                self.metrics.consequential_turns += 1
                self.metrics.last_consequential_turn = self.metrics.turns

            if self.runtime.is_done:
                log.info(
                    "agent.done_called",
                    customer=self.runtime.customer_id,
                    agent_run_id=self.runtime.agent_run_id,
                    turns=self.metrics.turns,
                )
                return self.metrics

            if self._stalled():
                self._halt("agent.stall")

    # -----------------------------------------------------------------------
    # Cache + manifest
    # -----------------------------------------------------------------------

    async def _build_initial_cache(self) -> None:
        index = await self.runtime.wiki_index()
        manifest = await self.runtime.initial_manifest(WIKI_AGENT_BATCH_SIZE)
        seed = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Wiki index (titles + slugs + summaries):\n"
                            f"{_render_index(index)}\n\n"
                            "First manifest window:\n"
                            f"{_render_manifest(manifest)}\n"
                        )
                    }
                ],
            }
        ]
        try:
            self._cache_name = await self.llm.create_cache(
                system_instruction=self.system_prompt,
                tools=self.tool_schemas,
                seed_contents=seed,
            )
            log.info(
                "agent.cache_created",
                customer=self.runtime.customer_id,
                agent_run_id=self.runtime.agent_run_id,
                cache_name=self._cache_name,
            )
        except Exception as exc:
            log.warning(
                "agent.cache_create_failed",
                customer=self.runtime.customer_id,
                agent_run_id=self.runtime.agent_run_id,
                error=str(exc),
            )
            # No cache is non-fatal; we just pay full input cost. Stamp a
            # sentinel so downstream code doesn't NPE.
            self._cache_name = ""

    # -----------------------------------------------------------------------
    # Per-turn LLM call (with tenacity retry)
    # -----------------------------------------------------------------------

    async def _call_llm(self) -> dict[str, Any]:
        retry = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type(Exception),
            reraise=False,
        )
        try:
            async for attempt in retry:
                with attempt:
                    started = time.monotonic()
                    resp = await self.llm.generate_with_cache(
                        cache_name=self._cache_name,
                        contents=self._conversation,
                        tools=self.tool_schemas,
                    )
                    latency = time.monotonic() - started
                    self._record_usage(resp)
                    log.info(
                        "agent.gemini_call",
                        customer=self.runtime.customer_id,
                        agent_run_id=self.runtime.agent_run_id,
                        turn=self.metrics.turns + 1,
                        latency_seconds=latency,
                        cached_input_tokens=self.metrics.total_cached_tokens,
                        new_input_tokens=self.metrics.total_input_tokens,
                        output_tokens=self.metrics.total_output_tokens,
                        cache_hit_rate=self.metrics.cache_hit_rate,
                    )
                    return resp
        except RetryError as exc:
            self._halt("agent.gemini_persistent_error", error=str(exc))
        # _halt always raises; this line is unreachable but keeps mypy happy.
        raise AgentHaltError("agent.gemini_persistent_error")

    def _record_usage(self, resp: dict[str, Any]) -> None:
        usage = resp.get("usage_metadata") or {}
        self.metrics.gemini_call_count += 1
        self.metrics.total_input_tokens += int(usage.get("prompt_token_count") or 0)
        self.metrics.total_cached_tokens += int(
            usage.get("cached_content_token_count") or 0
        )
        self.metrics.total_output_tokens += int(
            usage.get("candidates_token_count") or 0
        )

    # -----------------------------------------------------------------------
    # Tool dispatch
    # -----------------------------------------------------------------------

    async def _dispatch(self, resp: dict[str, Any]) -> bool:
        """Dispatch every tool call in `resp`. Returns True if any
        tool was consequential (update_page / create_page / skip_events
        / done). Snapshot-then-mutate happens inside the runtime.
        """
        tool_calls: list[dict[str, Any]] = resp.get("tool_calls") or []
        if not tool_calls:
            # Model returned text only — append to conversation so the
            # next turn sees its commentary, then short-circuit.
            text = resp.get("text") or ""
            if text:
                self._conversation.append(
                    {"role": "model", "parts": [{"text": text}]}
                )
            return False

        # Append the model turn (with tool_call parts) to conversation
        # before we record results, so the function_response parts are
        # in the right order in the trace.
        self._conversation.append(
            {
                "role": "model",
                "parts": [
                    {"function_call": {"name": tc["name"], "args": tc.get("args", {})}}
                    for tc in tool_calls
                ],
            }
        )

        consequential = False
        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            try:
                result = await self.runtime.dispatch_tool(name, args)
            except ToolValidationError as exc:
                # Pydantic validation failure on tool input. Tell the
                # model so it can re-decide on the next turn.
                result = {"error": "tool_validation_error", "detail": str(exc)}
                log.info(
                    "agent.tool_validation_error",
                    customer=self.runtime.customer_id,
                    agent_run_id=self.runtime.agent_run_id,
                    tool=name,
                    detail=str(exc),
                )
            except Exception as exc:
                # Snapshot rollback inside runtime. Surface a typed
                # error result so the next turn can retry / give up.
                result = {"error": "tool_exception", "detail": str(exc)}
                log.warning(
                    "agent.tool_exception",
                    customer=self.runtime.customer_id,
                    agent_run_id=self.runtime.agent_run_id,
                    tool=name,
                    error=str(exc),
                    error_class=type(exc).__name__,
                )
            results.append({"name": name, "result": result})
            if name in {"update_page", "create_page", "skip_events", "done"}:
                consequential = True

        self._conversation.append(
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": r["name"],
                            "response": r["result"],
                        }
                    }
                    for r in results
                ],
            }
        )
        return consequential

    # -----------------------------------------------------------------------
    # Compaction + stall + caps
    # -----------------------------------------------------------------------

    async def _maybe_compact(self) -> None:
        if self.summarizer is None:
            return
        est = self._estimate_tokens()
        if est < int(_MODEL_CONTEXT_TOKENS * WIKI_AGENT_COMPACT_THRESHOLD):
            return
        log.info(
            "agent.compaction_triggered",
            customer=self.runtime.customer_id,
            agent_run_id=self.runtime.agent_run_id,
            estimated_tokens=est,
            threshold=WIKI_AGENT_COMPACT_THRESHOLD,
        )
        try:
            summary = await self.summarizer(
                self._conversation,
                self.runtime.state_snapshot_for_summary(),
            )
        except AgentCompactionError as exc:
            self._halt("agent.compaction_failed", error=str(exc))
        except Exception as exc:
            self._halt(
                "agent.compaction_failed", error=f"{type(exc).__name__}: {exc}"
            )
        # Replace the conversation with the summary; runtime state is
        # preserved across the boundary because the summarizer reads it
        # from runtime.state_snapshot_for_summary() and writes it back
        # into the summary text.
        self._conversation = [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "[compacted history at "
                            f"{datetime.now(UTC).isoformat()}]\n\n"
                            f"{summary}\n"
                        )
                    }
                ],
            }
        ]
        self.metrics.compaction_count += 1

    def _stalled(self) -> bool:
        if self.metrics.last_consequential_turn == 0 and self.metrics.turns < WIKI_AGENT_STALL_TURNS:
            # Pre-first-consequential grace window.
            return False
        gap = self.metrics.turns - self.metrics.last_consequential_turn
        return gap >= WIKI_AGENT_STALL_TURNS

    def _check_caps(self) -> None:
        if self.metrics.turns >= WIKI_AGENT_TURN_CAP:
            self._halt("agent.turn_cap")
        if self.runtime.pending_update_count >= WIKI_AGENT_UPDATE_CAP:
            self._halt("agent.update_cap")

    def _halt(self, reason: str, **context: Any) -> None:
        self.metrics.halt_reason = reason
        log.warning(
            "agent.halt",
            customer=self.runtime.customer_id,
            agent_run_id=self.runtime.agent_run_id,
            reason=reason,
            turns=self.metrics.turns,
            **context,
        )
        raise AgentHaltError(reason, **context)

    # -----------------------------------------------------------------------
    # Token estimate (cheap; no Gemini round-trip)
    # -----------------------------------------------------------------------

    def _estimate_tokens(self) -> int:
        """Rough token estimate over the live conversation tail.

        The cache is roughly fixed-cost; the unbounded growth is the
        conversation tail. Estimate from char-length of every text /
        function_call / function_response part. ~3.5 chars/token (the
        Gemini average for English).
        """
        char_count = 0
        for msg in self._conversation:
            for part in msg.get("parts") or []:
                if "text" in part:
                    char_count += len(part["text"] or "")
                if "function_call" in part:
                    fc = part["function_call"]
                    char_count += len(str(fc.get("args") or "")) + len(fc.get("name", ""))
                if "function_response" in part:
                    fr = part["function_response"]
                    char_count += len(str(fr.get("response") or "")) + len(
                        fr.get("name", "")
                    )
        return int(char_count * _TOKENS_PER_CHAR)


# ---------------------------------------------------------------------------
# Helpers (rendering)
# ---------------------------------------------------------------------------


def _render_index(index: list[dict[str, Any]]) -> str:
    if not index:
        return "(no wiki pages yet)"
    lines = []
    for entry in index:
        wiki_type = entry.get("wiki_type", "?")
        slug = entry.get("slug", "?")
        title = entry.get("title", slug)
        summary = entry.get("summary") or ""
        summary_short = summary[:120].replace("\n", " ")
        lines.append(f"- [{wiki_type}/{slug}] {title}: {summary_short}")
    return "\n".join(lines)


def _render_manifest(manifest: dict[str, Any]) -> str:
    events = manifest.get("events") or []
    if not events:
        return "(no triaged events to read)"
    lines = []
    for ev in events:
        qid = ev.get("queue_id")
        ts = ev.get("source_ts")
        title = ev.get("title") or "(no title)"
        source = ev.get("source_system", "?")
        preview = (ev.get("body_preview") or "")[:200].replace("\n", " ")
        lines.append(f"  qid={qid} ts={ts} src={source} title={title!r} preview={preview!r}")
    remaining = manifest.get("remaining", 0)
    drain_complete = manifest.get("drain_complete", False)
    return (
        f"events ({len(events)}; remaining={remaining}, "
        f"drain_complete={drain_complete}):\n" + "\n".join(lines)
    )


__all__ = [
    "AgentLoop",
    "AgentMetrics",
    "new_agent_run_id",
]


def new_agent_run_id() -> str:
    """UUID4 used as the wiki_synthesis_runs row stamp + structlog field."""
    return str(uuid.uuid4())
