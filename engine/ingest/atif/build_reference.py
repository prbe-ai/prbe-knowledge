"""FROZEN: the probe-events/1 -> ATIF builder exactly as it was before the
fragment + fold split (prbe-knowledge e1632b2). DO NOT EDIT until the
protocol-2 sunset, when it is deleted.

It is the oracle: `fold(map(fragment, events))` (fold.py, fragment.py) must
build the same document from the same events, and every trajectory stored
before the split was written by this code. A change here would move the
reference that change is meant to be measured against; change fragment.py /
fold.py instead and let the equality tests (tests/test_atif_fragment_fold.py)
and the replay (`scripts/atif_sessions.py replay --compare-builders`) say
whether the documents still agree.

What follows is that module verbatim, less `build_and_render` (the API stays
in build.py).

A captured session's events -> an ATIF trajectory (Harbor's Agent Trajectory
Interchange Format, RFC 0001, v1.8).

INPUT is the merged event list `fetch_supplementary` returns: `probe-events/1`,
the shape every capture sanitizer emits (Claude Code's event shape, with the
other harnesses' own fields under `_codex_extras` / `_pi_extras` /
`_kimi_extras`). OUTPUT is a plain dict that validates against the vendored
models in `engine.ingest.atif.models`. Readers (the session API, the dashboard,
`probe session export`) get it from R2; nothing about search changes.

TWO VIEWS IN ONE DOCUMENT
-------------------------
The ATIF fields are the structured view: one user step per prompt, one agent
step per model inference (text, reasoning, tool calls, the tool calls' results
as `observation`), system steps for everything else.

`extra.probe` (root) and `extra.probe_parts` (per step) are the provenance that
lets `engine.ingest.atif.lines` rebuild the exact text the search index holds:
one record per source event, in order, and for every rendered piece the event
it came from and its position inside it. That is what keeps evidence spans
(one per event) and the extraction cache (keyed on rendered text) byte-for-byte
unchanged when the trajectory is the source. In `shadow` and `atif` modes the
ingest pass compares the two on every completing pass (kb/handlers/claude_code.py),
so a gap here costs a fallback, never a changed index. The provenance is
engine-internal: the stored copy and the API carry the ATIF fields only.

  root extra.probe.lines   [[line_no, flags], ...]    one per source event
  step extra.probe_parts   [[line, seq, kind, *args]] line = index into lines

Part kinds and what they render (engine/shared/transcript_render.py formats):
  u  user text        args: message index (None for a str message), speaker
  r  tool result      args: observation.results index
  a  assistant text   args: message index
  t  thinking         args: start, end offsets into reasoning_content
  c  tool call        args: tool_calls index
  s  stop reason      args: the raw stop_reason
  y  system event     (no args)
  e  other event type (no args)

WHAT IT NEVER DOES
------------------
Ship arguments: capture keeps a summary of a tool call, never its input, so
`arguments` is `{}` and the summary sits in the call's `extra`. Ship tool
output: results carry `is_error` and a size, never `content`. Drop an event:
every one is a line, and one it cannot map becomes an `unparsed` system step
(counted; a pass with any is not authoritative for extraction).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from engine.shared.transcript_render import renders_stop, speaker_for, strip_harness

SCHEMA_VERSION = "ATIF-v1.8"
#: Version of the provenance encoding in `extra.probe`. A reader that sees any
#: other value must not rebuild text from it.
RENDER_FORMAT = "probe-render/1"

PART_USER = "u"
PART_RESULT = "r"
PART_TEXT = "a"
PART_THINKING = "t"
PART_CALL = "c"
PART_STOP = "s"
PART_SYSTEM = "y"
PART_EVENT = "e"

FLAG_USER_TURN = 1
FLAG_COMPACT_BOUNDARY = 2
FLAG_COMPACT_SUMMARY = 4

#: Harness-only fields the sanitizers carry, kept verbatim on the step.
_HARNESS_EXTRAS = ("_codex_extras", "_pi_extras", "_kimi_extras")

#: Bounds on what a client-controlled value may put into a document readers
#: are served (GET /trajectory). Upload is the trust boundary: a patched or
#: compromised tap posts whatever it likes, and before this module nothing
#: served these fields to anyone.
_EXTRA_MAX_KEYS = 64
_EXTRA_KEY_CHARS = 64
_EXTRA_VALUE_CHARS = 256
_TYPE_NAME_CHARS = 64
#: probe-events/1 `origin` of a command the researcher typed, not the model.
USER_SHELL = "user_shell"
#: Unparsed events reported per build; the count covers the rest.
_UNPARSED_LOGGED = 20
_DROPPED_BLOCKS_MAX = 32

_USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass(slots=True)
class BuildResult:
    trajectory: dict[str, Any]
    #: Events that could not be mapped. Any at all makes the pass degraded.
    unparsed: int
    #: (line_no, event type, error class) of the first few, for the caller to
    #: log: the build may run in a pool process, whose logging is not the
    #: worker's.
    unparsed_events: list[tuple[int, str | None, str]] = field(default_factory=list)


def build_trajectory(
    events: list[dict[str, Any]],
    *,
    session_id: str,
    agent_name: str,
) -> BuildResult:
    builder = _Builder()
    for ev in events:
        if isinstance(ev, dict):
            builder.add(ev)
    return BuildResult(
        builder.finish(session_id=session_id, agent_name=agent_name),
        builder.unparsed,
        builder.unparsed_events,
    )


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _iso(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _usage(raw: dict[str, Any], msg: dict[str, Any]) -> dict[str, int] | None:
    """`probe-events/1` top-level `usage`, else Claude Code's own `message.usage`."""
    source = raw.get("usage") if isinstance(raw.get("usage"), dict) else msg.get("usage")
    if not isinstance(source, dict):
        return None
    out = {k: v for k in _USAGE_KEYS if (v := _int(source.get(k))) is not None}
    return out or None


def _short(value: Any, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def _safe_extras(value: dict[str, Any]) -> dict[str, Any]:
    """A harness's extras as flat scalars: strings capped, nested values gone.

    The sanitizers stash harness-only metadata under `_codex_extras` /
    `_pi_extras` / `_kimi_extras`; older taps put whole tool-call inputs there
    (Codex `action`, with `env`). Only short scalars reach a reader.
    """
    out: dict[str, Any] = {}
    for key, item in value.items():
        if len(out) >= _EXTRA_MAX_KEYS:
            break
        if not isinstance(key, str) or not key:
            continue
        if isinstance(item, bool | int | float):
            out[key[:_EXTRA_KEY_CHARS]] = item
        elif isinstance(item, str):
            out[key[:_EXTRA_KEY_CHARS]] = item[:_EXTRA_VALUE_CHARS]
    return out


def _call_stats(stats: Any) -> dict[str, Any] | None:
    """Exactly what the renderer reads from `stats` (transcript_render
    `_render_tool_use`): integer line counts and a literal True `replace_all`."""
    if not isinstance(stats, dict):
        return None
    out: dict[str, Any] = {
        k: stats[k] for k in ("added_lines", "removed_lines") if isinstance(stats.get(k), int)
    }
    if stats.get("replace_all") is True:
        out["replace_all"] = True
    return out or None


def _metrics(usage: dict[str, int]) -> dict[str, Any]:
    """ATIF counts every input token in prompt_tokens; cached_tokens is a subset."""
    prompt = sum(
        usage.get(k, 0)
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    metrics: dict[str, Any] = {"prompt_tokens": prompt}
    if "output_tokens" in usage:
        metrics["completion_tokens"] = usage["output_tokens"]
    if "cache_read_input_tokens" in usage:
        metrics["cached_tokens"] = usage["cache_read_input_tokens"]
    if "cache_creation_input_tokens" in usage:
        metrics["extra"] = {"cache_creation_input_tokens": usage["cache_creation_input_tokens"]}
    return metrics


class _Builder:
    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []
        self.lines: list[list[Any]] = []
        #: tool_call_id -> index of the agent step that made the call.
        self.calls: dict[str, int] = {}
        #: The agent step the next assistant event joins, while no other kind
        #: of event has come between.
        self.open_agent: int | None = None
        self.open_inference: str | None = None
        self.seen_inferences: set[str] = set()
        #: inference_id -> the step that carries its metrics.
        self.metrics_step: dict[str, int] = {}
        self.unparsed = 0
        self.unparsed_events: list[tuple[int, str | None, str]] = []
        self.agent_version: str | None = None
        self.model_name: str | None = None
        #: Per agent step: its thinking blocks, and the length they will have
        #: once joined with blank lines.
        self.reasoning: dict[int, list[str]] = {}
        self.reasoning_len: dict[int, int] = {}

    # -- per event ---------------------------------------------------------

    def add(self, ev: dict[str, Any]) -> None:
        raw = ev.get("raw")
        line_no = ev.get("line_no") if isinstance(ev.get("line_no"), int) else None
        facts = raw if isinstance(raw, dict) else ev
        flags = 0
        if facts.get("type") == "user" and not facts.get("isCompactSummary"):
            flags |= FLAG_USER_TURN
        if facts.get("type") == "system" and facts.get("subtype") == "compact_boundary":
            flags |= FLAG_COMPACT_BOUNDARY
        if facts.get("isCompactSummary"):
            flags |= FLAG_COMPACT_SUMMARY
        line = len(self.lines)
        self.lines.append([line_no, flags])
        if not isinstance(raw, dict):
            return

        # Per-event isolation: a shape this module has never seen costs that
        # one event, not the session. What it mapped before failing stays (a
        # reader is better served by part of an event than none); the marker
        # says the event is incomplete, and the count makes the pass degraded.
        try:
            self._map(line, raw)
        except Exception as exc:  # isolation is the point: never fail the session
            self.unparsed += 1
            self.open_agent = None
            self._add_step(
                {
                    "source": "system",
                    "message": "",
                    "extra": {
                        "subtype": "unparsed",
                        "event_type": raw.get("type") if isinstance(raw.get("type"), str) else None,
                        "probe_parts": [],
                    },
                }
            )
            if len(self.unparsed_events) < _UNPARSED_LOGGED:
                ev_type = raw.get("type")
                self.unparsed_events.append(
                    (line_no, _short(ev_type, _TYPE_NAME_CHARS), type(exc).__name__)
                )

    def _map(self, line: int, raw: dict[str, Any]) -> None:
        ev_type = raw.get("type")
        if ev_type == "assistant":
            self._assistant(line, raw)
            return
        # Anything but the model's own output ends the inference in progress.
        self.open_agent = None
        if ev_type == "user":
            self._user(line, raw)
        elif ev_type == "system":
            self._system(line, raw)
        else:
            self._other(line, raw)

    def _user(self, line: int, raw: dict[str, Any]) -> None:
        msg = raw.get("message")
        if not isinstance(msg, dict):
            return
        speaker = speaker_for(raw)
        summary = bool(raw.get("isCompactSummary"))
        content = msg.get("content")
        if isinstance(content, str) and content:
            cleaned = strip_harness(content)
            if cleaned:
                step = self._prompt_step(raw, summary, cleaned)
                self._part(step, line, 0, PART_USER, None, speaker)
            return
        if not isinstance(content, list):
            return

        step: int | None = None
        text_parts: list[dict[str, Any]] = []
        other_blocks: list[str] = []
        orphan: int | None = None
        for seq, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                cleaned = strip_harness(b.get("text") or "")
                if not cleaned:
                    continue
                if step is None:
                    step = self._prompt_step(raw, summary, text_parts)
                text_parts.append({"type": "text", "text": cleaned})
                self._part(step, line, seq, PART_USER, len(text_parts) - 1, speaker)
            elif bt == "tool_result":
                target = self._result_target(b.get("tool_use_id"))
                if target is None:
                    if orphan is None:
                        orphan = self._add_step(
                            {
                                "source": "system",
                                "message": "",
                                "extra": {"subtype": "orphan_tool_result", "probe_parts": []},
                                **self._timestamp(raw),
                            }
                        )
                    target = orphan
                index = self._add_result(target, b, linked=target != orphan)
                self._part(target, line, seq, PART_RESULT, index)
            elif isinstance(bt, str) and len(other_blocks) < _DROPPED_BLOCKS_MAX:
                other_blocks.append(bt[:_TYPE_NAME_CHARS])
        if other_blocks and step is not None:
            self.steps[step]["extra"]["dropped_blocks"] = other_blocks

    def _prompt_step(self, raw: dict[str, Any], summary: bool, message: Any) -> int:
        extra: dict[str, Any] = {"probe_parts": []}
        if summary:
            extra["subtype"] = "compaction_summary"
        elif raw.get("origin") == USER_SHELL:
            # Kimi's shell mode: the researcher typed a command, which arrives as
            # their own turn. Marked so readers can tell it from a prompt.
            extra["origin"] = USER_SHELL
        self._harness_extras(extra, raw)
        return self._add_step(
            {
                "source": "system" if summary else "user",
                "message": message,
                "extra": extra,
                **self._timestamp(raw),
            }
        )

    def _assistant(self, line: int, raw: dict[str, Any]) -> None:
        msg = raw.get("message")
        if not isinstance(msg, dict):
            return
        inference = _short(raw.get("inference_id"), _EXTRA_VALUE_CHARS)
        # A command the researcher typed (pi's `!`) arrives shaped as an
        # assistant tool call with no inference id. ATIF keeps tool
        # calls on agent steps, so it gets one of its own, marked, and never
        # joins a model call's step on either side.
        user_shell = raw.get("origin") == USER_SHELL
        if user_shell:
            self.open_agent = None
        step = self._agent_step(raw, msg, inference)
        record = self.steps[step]
        if user_shell:
            record["extra"]["origin"] = USER_SHELL
            self.open_agent = None

        content = msg.get("content")
        if isinstance(content, str) and content:
            self._append_text(step, line, 0, content)
            return
        if not isinstance(content, list):
            return
        for seq, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                text = b.get("text") or ""
                if text:
                    self._append_text(step, line, seq, text if isinstance(text, str) else f"{text}")
            elif bt == "thinking":
                text = b.get("thinking") or ""
                if isinstance(text, str) and text.strip():
                    # Joined once in finish(): appending to one growing string
                    # copied it per block, quadratic in a long inference.
                    blocks = self.reasoning.setdefault(step, [])
                    start = self.reasoning_len.get(step, -2) + 2
                    blocks.append(text)
                    self.reasoning_len[step] = start + len(text)
                    self._part(step, line, seq, PART_THINKING, start, start + len(text))
            elif bt == "tool_use":
                calls = record.setdefault("tool_calls", [])
                call_id = b.get("id") if isinstance(b.get("id"), str) and b.get("id") else None
                name = b.get("name") or "tool"
                call: dict[str, Any] = {
                    "tool_call_id": call_id or f"probe:{line}:{seq}",
                    "function_name": name if isinstance(name, str) else f"{name}",
                    # Capture never ships a tool's input; the summary is in extra.
                    "arguments": {},
                }
                call_extra: dict[str, Any] = {}
                summary = b.get("summary")
                if summary:
                    # As the renderer prints it; a non-string never reaches a reader.
                    call_extra["summary"] = summary if isinstance(summary, str) else f"{summary}"
                stats = _call_stats(b.get("stats"))
                if stats:
                    call_extra["stats"] = stats
                if call_extra:
                    call["extra"] = call_extra
                calls.append(call)
                if call_id:
                    self.calls[call_id] = step
                self._part(step, line, seq, PART_CALL, len(calls) - 1)
        stop = msg.get("stop_reason")
        if renders_stop(stop):
            stop = stop if isinstance(stop, str) else f"{stop}"
            record["extra"]["stop_reason"] = stop
            self._part(step, line, len(content), PART_STOP, stop)

    def _agent_step(self, raw: dict[str, Any], msg: dict[str, Any], inference: str | None) -> int:
        joins = self.open_agent is not None and (
            inference is None or self.open_inference is None or inference == self.open_inference
        )
        if joins:
            step = self.open_agent
            assert step is not None
            if inference and not self.open_inference:
                self.open_inference = inference
                self.steps[step]["extra"]["inference_id"] = inference
        else:
            extra: dict[str, Any] = {"probe_parts": []}
            if inference:
                extra["inference_id"] = inference
                if inference in self.seen_inferences:
                    # An event of another kind came between two pieces of one
                    # model response; it stays in place, so the response is two
                    # steps. Usage is counted on the first only.
                    extra["continues_inference"] = True
            step = self._add_step(
                {"source": "agent", "message": [], "extra": extra, **self._timestamp(raw)}
            )
            self.open_agent, self.open_inference = step, inference
        record = self.steps[step]
        if inference:
            self.seen_inferences.add(inference)
        model = _short(msg.get("model"), _EXTRA_VALUE_CHARS)
        if model:
            record.setdefault("model_name", model)
            self.model_name = self.model_name or model
        usage = _usage(raw, msg)
        if usage:
            # Last report wins within one inference: each streamed piece of a
            # response repeats the usage so far.
            owner = self.metrics_step.setdefault(inference, step) if inference else step
            self.steps[owner]["metrics"] = _metrics(usage)
        self._harness_extras(record["extra"], raw)
        return step

    def _append_text(self, step: int, line: int, seq: int, text: str) -> None:
        message = self.steps[step]["message"]
        message.append({"type": "text", "text": text})
        self._part(step, line, seq, PART_TEXT, len(message) - 1)

    def _system(self, line: int, raw: dict[str, Any]) -> None:
        content = raw.get("content")
        extra: dict[str, Any] = {"probe_parts": []}
        subtype = raw.get("subtype")
        if subtype:
            # As the renderer prints it (`SYSTEM (<subtype>)`).
            extra["subtype"] = subtype if isinstance(subtype, str) else f"{subtype}"
        self._harness_extras(extra, raw)
        step = self._add_step(
            {
                "source": "system",
                "message": content if isinstance(content, str) and content else "",
                "extra": extra,
                **self._timestamp(raw),
            }
        )
        self._part(step, line, 0, PART_SYSTEM)
        codex = raw.get("_codex_extras")
        if isinstance(codex, dict):
            self.agent_version = self.agent_version or _short(codex.get("cli_version"), _TYPE_NAME_CHARS)

    def _other(self, line: int, raw: dict[str, Any]) -> None:
        content = raw.get("content")
        extra: dict[str, Any] = {"probe_parts": []}
        # Uncapped: the renderer prints the whole type into the index.
        event_type = raw.get("type")
        if isinstance(event_type, str) and event_type:
            extra["event_type"] = event_type
        attachment = raw.get("attachment")
        if isinstance(attachment, dict) and _short(attachment.get("type"), _TYPE_NAME_CHARS):
            extra["attachment_type"] = attachment["type"][:_TYPE_NAME_CHARS]
        self._harness_extras(extra, raw)
        step = self._add_step(
            {
                "source": "system",
                "message": content if isinstance(content, str) and content else "",
                "extra": extra,
                **self._timestamp(raw),
            }
        )
        self._part(step, line, 0, PART_EVENT)

    # -- tool results ---------------------------------------------------------

    def _result_target(self, tool_use_id: Any) -> int | None:
        if isinstance(tool_use_id, str) and tool_use_id:
            return self.calls.get(tool_use_id)
        return None

    def _add_result(self, step: int, block: dict[str, Any], *, linked: bool) -> int:
        """`linked`: the step holds the call. ATIF requires `source_call_id` to
        name a call in the same step, so an orphan keeps its id in `extra`."""
        tool_use_id = block.get("tool_use_id")
        result: dict[str, Any] = {}
        extra: dict[str, Any] = {"is_error": bool(block.get("is_error"))}
        if linked:
            result["source_call_id"] = tool_use_id
        elif tool_use_id:
            # As the renderer prints it (`TOOL_RESULT (<id>)`).
            extra["tool_use_id"] = tool_use_id if isinstance(tool_use_id, str) else f"{tool_use_id}"
        if isinstance(block.get("result_bytes"), int):
            # The renderer prints a size only for an int; anything else is noise.
            extra["result_bytes"] = block.get("result_bytes")
        result["extra"] = extra
        observation = self.steps[step].setdefault("observation", {"results": []})
        observation["results"].append(result)
        return len(observation["results"]) - 1

    # -- bookkeeping ----------------------------------------------------------

    def _add_step(self, step: dict[str, Any]) -> int:
        self.steps.append(step)
        return len(self.steps) - 1

    def _part(self, step: int, line: int, seq: int, kind: str, *args: Any) -> None:
        self.steps[step]["extra"]["probe_parts"].append([line, seq, kind, *args])

    @staticmethod
    def _timestamp(raw: dict[str, Any]) -> dict[str, Any]:
        stamp = _iso(raw.get("timestamp"))
        return {"timestamp": stamp} if stamp else {}

    @staticmethod
    def _harness_extras(extra: dict[str, Any], raw: dict[str, Any]) -> None:
        for key in _HARNESS_EXTRAS:
            value = raw.get(key)
            if isinstance(value, dict):
                safe = _safe_extras(value)
                if safe:
                    extra.setdefault(key.strip("_"), []).append(safe)

    # -- the document ---------------------------------------------------------

    def finish(self, *, session_id: str, agent_name: str) -> dict[str, Any]:
        if not self.steps:
            # ATIF requires one step; a session whose events carry nothing
            # readable still gets a document, marked as such.
            self._add_step(
                {"source": "system", "message": "", "extra": {"subtype": "empty", "probe_parts": []}}
            )
        steps: list[dict[str, Any]] = []
        totals = {"prompt": 0, "completion": 0, "cached": 0}
        any_metrics = False
        for index, step in enumerate(self.steps):
            if index in self.reasoning:
                step["reasoning_content"] = "\n\n".join(self.reasoning[index])
        for number, step in enumerate(self.steps, start=1):
            out = {"step_id": number, **step}
            if out["source"] == "agent" and not out["message"]:
                out["message"] = ""
            if not out.get("reasoning_content"):
                out.pop("reasoning_content", None)
            metrics = out.get("metrics")
            if metrics:
                any_metrics = True
                totals["prompt"] += metrics.get("prompt_tokens") or 0
                totals["completion"] += metrics.get("completion_tokens") or 0
                totals["cached"] += metrics.get("cached_tokens") or 0
            steps.append(out)
        agent: dict[str, Any] = {"name": agent_name, "version": self.agent_version or "unknown"}
        if self.model_name:
            agent["model_name"] = self.model_name
        trajectory: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "session_id": session_id,
            "agent": agent,
            "steps": steps,
            "extra": {
                "probe": {
                    "format": RENDER_FORMAT,
                    "lines": self.lines,
                    "unparsed": self.unparsed,
                }
            },
        }
        if any_metrics:
            trajectory["final_metrics"] = {
                "total_prompt_tokens": totals["prompt"],
                "total_completion_tokens": totals["completion"],
                "total_cached_tokens": totals["cached"],
                "total_steps": len(steps),
            }
        return trajectory
