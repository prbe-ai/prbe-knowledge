"""A session's fragments -> its ATIF trajectory (Harbor's Agent Trajectory
Interchange Format, RFC 0001, v1.8).

The cross-event half of the probe-events/1 -> ATIF builder. `fragment.fragment`
maps one event; `fold` joins them: the pieces of one model call into one step
(`continues_inference` when another event came between), a late
`inference_id`, results onto the step that made the call (orphans, ids that
name nothing, results that arrive before their call), which step carries an
inference's metrics (the last report wins), step ids, generated call ids,
`final_metrics`, the agent, the empty-session marker, `unparsed`, and the
render provenance below, rebuilt from the fragments' order.
`fold(map(fragment, events))` equals the frozen builder's document for the
same events (build_reference.py; tests/test_atif_fragment_fold.py).

INPUT IS UNTRUSTED. Fragments will come from clients (protocol 3), so fold
reads each one as data: it checks every field's type, re-applies every bound
the builder applies (extras flattened and capped, call stats, usage counts,
id and name caps), copies only the fields it knows into the document (a tool
call's `arguments` is always `{}`, a result never carries content), ignores
unknown keys (and `lineage`, carried for later), and never raises on a
fragment's content. A fragment it cannot read is one line and one `unparsed`
step, counted like an event the builder could not map. Its ordinal must be a
non-negative int (or null, for events stored without one) that no earlier
fragment of the session holds: a repeated one is unreadable and its line has
no number, so the first fragment keeps the slot. Fragments are folded in the
order given; putting them in ordinal order is the caller's.

OUTPUT is a plain dict that validates against the vendored models in
`engine.ingest.atif.models`. Readers (the session API, the dashboard, `probe
session export`) get it from R2; nothing about search changes.

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

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any

from engine.ingest.atif.fragment import (
    DROPPED_BLOCKS_MAX,
    EXTRA_VALUE_CHARS,
    FRAGMENT_VERSION,
    HARNESS_EXTRAS,
    JSON_INT_MAX,
    TYPE_NAME_CHARS,
    USER_SHELL,
    Kind,
    PieceType,
    call_stats,
    safe_extras,
    short,
    usage_counts,
)
from engine.shared.transcript_render import Line, renders_stop, speaker_for

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

#: The fragment versions this fold reads; one session may mix them. Public: the
#: door accepts a protocol-3 batch only in versions it lists here (intersected
#: with whatever the door's own setting allows).
SUPPORTED_FRAGMENT_VERSIONS: frozenset[int] = frozenset({FRAGMENT_VERSION})

#: Unparsed events reported per build; the count covers the rest.
_UNPARSED_LOGGED = 20
#: Bound on an error name a fragment reports; it reaches the worker's log.
_ERROR_CHARS = 64
#: A part's position inside its event. Real ones are block indexes; the bound
#: only keeps a hostile one printable inside a generated call id.
_SEQ_MAX = 2**31 - 1

_KINDS = frozenset(k.value for k in Kind)
_FLAG_NAMES = ("user_turn", "compact_boundary", "compact_summary")


class Unreadable(StrEnum):
    """Why fold could not read a fragment: the error its unparsed step is counted under."""

    INVALID = "InvalidFragment"
    VERSION = "UnsupportedFragmentVersion"


@dataclass(slots=True)
class BuildResult:
    trajectory: dict[str, Any]
    #: Events that could not be mapped. Any at all makes the pass degraded.
    unparsed: int
    #: (line_no, event type, error class) of the first few, for the caller to
    #: log: the build may run in a pool process, whose logging is not the
    #: worker's.
    unparsed_events: list[tuple[int | None, str | None, str]] = field(default_factory=list)


def fold(fragments: Iterable[Any], *, session_id: str, agent_name: str) -> BuildResult:
    folder = _Folder()
    for fragment in fragments:
        folder.add(fragment)
    return BuildResult(
        folder.finish(session_id=session_id, agent_name=agent_name),
        folder.unparsed,
        folder.unparsed_events,
    )


def line_of(fragment: Any) -> Line | None:
    """A fragment's index Line, checked field by field; None when it carries
    none fold can read (fold then records the event with no line number and no
    text). Whether its ordinal repeats an earlier one is fold's to say."""
    if not isinstance(fragment, dict):
        return None
    line = fragment.get("line")
    if not isinstance(line, dict):
        return None
    line_no, text = line.get("line_no"), line.get("text")
    if line_no is not None and not (
        isinstance(line_no, int) and not isinstance(line_no, bool) and 0 <= line_no <= JSON_INT_MAX
    ):
        return None
    if not isinstance(text, str):
        return None
    flags: dict[str, bool] = {}
    for name in _FLAG_NAMES:
        value = line.get(name, False)
        if not isinstance(value, bool):
            return None
        flags[name] = value
    return Line(line_no=line_no, text=text, **flags)


def _flags(line: Line) -> int:
    return (
        (FLAG_USER_TURN if line.user_turn else 0)
        | (FLAG_COMPACT_BOUNDARY if line.compact_boundary else 0)
        | (FLAG_COMPACT_SUMMARY if line.compact_summary else 0)
    )


def iso_timestamp(value: Any) -> str | None:
    """A timestamp the ATIF model accepts, or None. Engine-side only: what
    `datetime.fromisoformat` accepts differs between Python versions."""
    if not isinstance(value, str) or not value:
        return None
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return value


def _metrics(usage: dict[str, int]) -> dict[str, Any]:
    """ATIF counts every input token in prompt_tokens; cached_tokens is a subset."""
    prompt = sum(
        usage.get(k, 0)
        for k in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    )
    # Each count is capped (fragment.non_negative_int); their sum is too, or a
    # JSON writer refuses the document.
    metrics: dict[str, Any] = {"prompt_tokens": min(prompt, JSON_INT_MAX)}
    if "output_tokens" in usage:
        metrics["completion_tokens"] = usage["output_tokens"]
    if "cache_read_input_tokens" in usage:
        metrics["cached_tokens"] = usage["cache_read_input_tokens"]
    if "cache_creation_input_tokens" in usage:
        metrics["extra"] = {"cache_creation_input_tokens": usage["cache_creation_input_tokens"]}
    return metrics


# -- reading one fragment --------------------------------------------------------------


class _Invalid(Exception):
    def __init__(self, reason: Unreadable = Unreadable.INVALID) -> None:
        super().__init__(reason.value)
        self.reason = reason


def _need(condition: bool) -> None:
    if not condition:
        raise _Invalid


def _opt_str(source: dict[str, Any], key: str) -> str | None:
    """A string field, None when absent or null; any other type is unreadable."""
    value = source.get(key)
    if value is None:
        return None
    _need(isinstance(value, str))
    return value


def _seq(value: Any) -> int:
    _need(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= _SEQ_MAX)
    return value


@dataclass(slots=True)
class _Piece:
    type: PieceType
    seq: int
    #: text / thinking
    text: str = ""
    #: tool_call `id`, tool_result `call_id`: a non-empty string or None
    call_id: str | None = None
    #: tool_call
    name: str = ""
    summary: str | None = None
    stats: dict[str, Any] | None = None
    #: tool_result
    id_text: str | None = None
    is_error: bool = False
    result_bytes: int | None = None


@dataclass(slots=True)
class _Event:
    """One fragment, read: every value checked and bounded."""

    line_no: int | None
    kind: Kind
    error: str | None = None
    timestamp: str | None = None
    #: (name on the step, flat scalars), in HARNESS_EXTRAS order
    extras: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    compaction: bool = False
    user_shell: bool = False
    message: str | None = None
    parts: list[_Piece] = field(default_factory=list)
    dropped_blocks: list[str] | None = None
    inference: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None
    stop: tuple[int, str] | None = None
    subtype: str | None = None
    agent_version: str | None = None
    event_type: str | None = None
    attachment_type: str | None = None

    def type_name(self) -> str | None:
        """The event's probe-events/1 type, as the builder reports an unparsed one."""
        return self.event_type if self.kind is Kind.OTHER else self.kind.value


#: Piece types (their wire strings) each kind may carry.
_PIECES_BY_KIND = {
    Kind.USER: frozenset({PieceType.TEXT.value, PieceType.TOOL_RESULT.value}),
    Kind.ASSISTANT: frozenset(
        {PieceType.TEXT.value, PieceType.THINKING.value, PieceType.TOOL_CALL.value}
    ),
}


def _read(fragment: dict[str, Any], line: Line) -> _Event:
    version = fragment.get("v")
    if not (isinstance(version, int) and not isinstance(version, bool)):
        raise _Invalid
    if version not in SUPPORTED_FRAGMENT_VERSIONS:
        raise _Invalid(Unreadable.VERSION)
    kind = fragment.get("kind")
    _need(isinstance(kind, str) and kind in _KINDS)
    event = _Event(line_no=line.line_no, kind=Kind(kind))
    error = _opt_str(fragment, "error")
    if error:
        # Only a non-empty name is a failure; `""` reports nothing.
        event.error = error[:_ERROR_CHARS]
    if event.kind is Kind.NONE:
        return event
    # Fields the builder itself filters are filtered the same way; any other
    # field of the wrong type makes the fragment unreadable.
    event.timestamp = iso_timestamp(fragment.get("timestamp"))
    extras = fragment.get("extras")
    if isinstance(extras, dict):
        for _key, name in HARNESS_EXTRAS:
            value = extras.get(name)
            if isinstance(value, dict):
                safe = safe_extras(value)
                if safe:
                    event.extras.append((name, safe))
    event.user_shell = fragment.get("origin") == USER_SHELL
    if event.kind is not Kind.ASSISTANT:
        # An assistant's string content is a text part.
        event.message = _opt_str(fragment, "message")
    if event.kind in _PIECES_BY_KIND:
        event.parts = _read_parts(fragment.get("parts"), _PIECES_BY_KIND[event.kind])
    if event.kind is Kind.USER:
        event.compaction = fragment.get("compaction") is True
        dropped = fragment.get("dropped_blocks")
        if dropped is not None:
            _need(isinstance(dropped, list) and all(isinstance(b, str) for b in dropped))
            event.dropped_blocks = [b[:TYPE_NAME_CHARS] for b in dropped[:DROPPED_BLOCKS_MAX]]
    elif event.kind is Kind.ASSISTANT:
        event.inference = short(_opt_str(fragment, "inference_id"), EXTRA_VALUE_CHARS)
        event.model = short(_opt_str(fragment, "model"), EXTRA_VALUE_CHARS)
        event.usage = usage_counts(fragment.get("usage"))
        stop = fragment.get("stop")
        if stop is not None:
            _need(isinstance(stop, dict))
            reason = _opt_str(stop, "reason")
            seq = _seq(stop.get("seq"))
            if renders_stop(reason):
                event.stop = (seq, reason)
    elif event.kind is Kind.SYSTEM:
        event.subtype = _opt_str(fragment, "subtype")
        event.agent_version = short(_opt_str(fragment, "agent_version"), TYPE_NAME_CHARS)
    else:
        event.event_type = _opt_str(fragment, "event_type")
        event.attachment_type = short(_opt_str(fragment, "attachment_type"), TYPE_NAME_CHARS)
    return event


def _read_parts(value: Any, allowed: frozenset[str]) -> list[_Piece]:
    if value is None:
        return []
    _need(isinstance(value, list))
    return [_read_piece(p, allowed) for p in value]


def _read_piece(value: Any, allowed: frozenset[str]) -> _Piece:
    _need(isinstance(value, dict))
    kind = value.get("type")
    _need(isinstance(kind, str) and kind in allowed)
    piece = _Piece(type=PieceType(kind), seq=_seq(value.get("seq")))
    if piece.type in (PieceType.TEXT, PieceType.THINKING):
        text = value.get("text")
        _need(isinstance(text, str))
        piece.text = text
    elif piece.type is PieceType.TOOL_CALL:
        name = value.get("name")
        _need(isinstance(name, str))
        piece.call_id = _opt_str(value, "id") or None
        piece.name = name
        piece.summary = _opt_str(value, "summary") or None
        piece.stats = call_stats(value.get("stats"))
    else:
        piece.call_id = _opt_str(value, "call_id") or None
        piece.id_text = _opt_str(value, "id_text") or None
        piece.is_error = bool(value.get("is_error"))
        result_bytes = value.get("result_bytes")
        # As the builder: the renderer prints a size only for an int.
        piece.result_bytes = result_bytes if isinstance(result_bytes, int) else None
    return piece


# -- folding --------------------------------------------------------------------------


class _Folder:
    """The builder's cross-event state (build_reference._Builder), fed fragments."""

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
        self.unparsed_events: list[tuple[int | None, str | None, str]] = []
        self.agent_version: str | None = None
        self.model_name: str | None = None
        #: Per agent step: its thinking blocks, and the length they will have
        #: once joined with blank lines.
        self.reasoning: dict[int, list[str]] = {}
        self.reasoning_len: dict[int, int] = {}
        #: Ordinals already given a line: each belongs to one fragment.
        self.ordinals: set[int] = set()

    # -- per fragment ----------------------------------------------------------

    def add(self, fragment: Any) -> None:
        line = line_of(fragment)
        if line is not None and line.line_no is not None and line.line_no in self.ordinals:
            # A repeated ordinal: the first fragment keeps the slot; this one is
            # recorded without a number, so no two lines claim one event.
            self._unreadable(None, Unreadable.INVALID, line_no=line.line_no)
            return
        try:
            if line is None:
                raise _Invalid
            event = _read(fragment, line)
        except _Invalid as exc:
            self._unreadable(line, exc.reason)
            return
        except Exception:  # a value no check above foresaw: still never the session
            self._unreadable(line, Unreadable.INVALID)
            return
        index = self._line(event.line_no, _flags(line))
        if event.kind is Kind.NONE:
            return
        try:
            self._map(index, event)
        except Exception as exc:  # isolation, as the builder's: never fail the session
            self._unparsed(event.line_no, event.type_name(), type(exc).__name__)
            return
        if event.error is not None:
            # The event failed where the fragment stops; what precedes is mapped.
            self._unparsed(event.line_no, event.type_name(), event.error)

    def _line(self, line_no: int | None, flags: int) -> int:
        """Record one fragment's line; returns its index (a part's `line`)."""
        if line_no is not None:
            self.ordinals.add(line_no)
        self.lines.append([line_no, flags])
        return len(self.lines) - 1

    def _unreadable(
        self, line: Line | None, reason: Unreadable, *, line_no: int | None = None
    ) -> None:
        """One line and one unparsed step for a fragment fold cannot read.
        `line_no`: the ordinal to report when the line itself goes unnumbered."""
        if line is not None:
            self._line(line.line_no, _flags(line))
            line_no = line.line_no
        else:
            self._line(None, 0)
        self._unparsed(line_no, None, reason.value)

    def _unparsed(self, line_no: int | None, event_type: str | None, error: str) -> None:
        self.unparsed += 1
        self.open_agent = None
        self._add_step(
            {
                "source": "system",
                "message": "",
                "extra": {"subtype": "unparsed", "event_type": event_type, "probe_parts": []},
            }
        )
        if len(self.unparsed_events) < _UNPARSED_LOGGED:
            self.unparsed_events.append((line_no, short(event_type, TYPE_NAME_CHARS), error))

    def _map(self, line: int, ev: _Event) -> None:
        if ev.kind is Kind.ASSISTANT:
            self._assistant(line, ev)
            return
        # Anything but the model's own output ends the inference in progress.
        self.open_agent = None
        if ev.kind is Kind.USER:
            self._user(line, ev)
        elif ev.kind is Kind.SYSTEM:
            self._system(line, ev)
        else:
            self._other(line, ev)

    def _user(self, line: int, ev: _Event) -> None:
        speaker = speaker_for({"isCompactSummary": ev.compaction})
        if ev.message is not None:
            # A string content: the prompt, and nothing else of the event.
            if ev.message:
                step = self._prompt_step(ev, ev.message)
                self._part(step, line, 0, PART_USER, None, speaker)
            return
        step: int | None = None
        text_parts: list[dict[str, Any]] = []
        orphan: int | None = None
        for piece in ev.parts:
            if piece.type is PieceType.TEXT:
                if not piece.text:
                    continue
                if step is None:
                    step = self._prompt_step(ev, text_parts)
                text_parts.append({"type": "text", "text": piece.text})
                self._part(step, line, piece.seq, PART_USER, len(text_parts) - 1, speaker)
            else:
                target = self.calls.get(piece.call_id) if piece.call_id else None
                if target is None:
                    if orphan is None:
                        orphan = self._add_step(
                            {
                                "source": "system",
                                "message": "",
                                "extra": {"subtype": "orphan_tool_result", "probe_parts": []},
                                **self._timestamp(ev),
                            }
                        )
                    target = orphan
                index = self._add_result(target, piece, linked=target != orphan)
                self._part(target, line, piece.seq, PART_RESULT, index)
        if ev.dropped_blocks and step is not None:
            self.steps[step]["extra"]["dropped_blocks"] = list(ev.dropped_blocks)

    def _prompt_step(self, ev: _Event, message: Any) -> int:
        extra: dict[str, Any] = {"probe_parts": []}
        if ev.compaction:
            extra["subtype"] = "compaction_summary"
        elif ev.user_shell:
            # Kimi's shell mode: the researcher typed a command, which arrives as
            # their own turn. Marked so readers can tell it from a prompt.
            extra["origin"] = USER_SHELL
        self._harness_extras(extra, ev)
        return self._add_step(
            {
                "source": "system" if ev.compaction else "user",
                "message": message,
                "extra": extra,
                **self._timestamp(ev),
            }
        )

    def _assistant(self, line: int, ev: _Event) -> None:
        # A command the researcher typed (pi's `!`) arrives shaped as an
        # assistant tool call with no inference id. ATIF keeps tool calls on
        # agent steps, so it gets one of its own, marked, and never joins a
        # model call's step on either side.
        if ev.user_shell:
            self.open_agent = None
        step = self._agent_step(ev)
        record = self.steps[step]
        if ev.user_shell:
            record["extra"]["origin"] = USER_SHELL
            self.open_agent = None
        for piece in ev.parts:
            if piece.type is PieceType.TEXT:
                if piece.text:
                    self._append_text(step, line, piece.seq, piece.text)
            elif piece.type is PieceType.THINKING:
                if piece.text.strip():
                    # Joined once in finish(): appending to one growing string
                    # copied it per block, quadratic in a long inference.
                    blocks = self.reasoning.setdefault(step, [])
                    start = self.reasoning_len.get(step, -2) + 2
                    blocks.append(piece.text)
                    self.reasoning_len[step] = start + len(piece.text)
                    self._part(step, line, piece.seq, PART_THINKING, start, start + len(piece.text))
            else:
                calls = record.setdefault("tool_calls", [])
                call: dict[str, Any] = {
                    "tool_call_id": piece.call_id or f"probe:{line}:{piece.seq}",
                    "function_name": piece.name,
                    # Capture never ships a tool's input; the summary is in extra.
                    "arguments": {},
                }
                call_extra: dict[str, Any] = {}
                if piece.summary:
                    call_extra["summary"] = piece.summary
                if piece.stats:
                    call_extra["stats"] = piece.stats
                if call_extra:
                    call["extra"] = call_extra
                calls.append(call)
                if piece.call_id:
                    self.calls[piece.call_id] = step
                self._part(step, line, piece.seq, PART_CALL, len(calls) - 1)
        if ev.stop is not None:
            seq, stop = ev.stop
            record["extra"]["stop_reason"] = stop
            self._part(step, line, seq, PART_STOP, stop)

    def _agent_step(self, ev: _Event) -> int:
        inference = ev.inference
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
                {"source": "agent", "message": [], "extra": extra, **self._timestamp(ev)}
            )
            self.open_agent, self.open_inference = step, inference
        record = self.steps[step]
        if inference:
            self.seen_inferences.add(inference)
        if ev.model:
            record.setdefault("model_name", ev.model)
            self.model_name = self.model_name or ev.model
        if ev.usage:
            # Last report wins within one inference: each streamed piece of a
            # response repeats the usage so far.
            owner = self.metrics_step.setdefault(inference, step) if inference else step
            self.steps[owner]["metrics"] = _metrics(ev.usage)
        self._harness_extras(record["extra"], ev)
        return step

    def _append_text(self, step: int, line: int, seq: int, text: str) -> None:
        message = self.steps[step]["message"]
        message.append({"type": "text", "text": text})
        self._part(step, line, seq, PART_TEXT, len(message) - 1)

    def _system(self, line: int, ev: _Event) -> None:
        extra: dict[str, Any] = {"probe_parts": []}
        if ev.subtype:
            # As the renderer prints it (`SYSTEM (<subtype>)`).
            extra["subtype"] = ev.subtype
        self._harness_extras(extra, ev)
        step = self._add_step(
            {
                "source": "system",
                "message": ev.message or "",
                "extra": extra,
                **self._timestamp(ev),
            }
        )
        self._part(step, line, 0, PART_SYSTEM)
        self.agent_version = self.agent_version or ev.agent_version

    def _other(self, line: int, ev: _Event) -> None:
        extra: dict[str, Any] = {"probe_parts": []}
        # Uncapped: the renderer prints the whole type into the index.
        if ev.event_type:
            extra["event_type"] = ev.event_type
        if ev.attachment_type:
            extra["attachment_type"] = ev.attachment_type
        self._harness_extras(extra, ev)
        step = self._add_step(
            {
                "source": "system",
                "message": ev.message or "",
                "extra": extra,
                **self._timestamp(ev),
            }
        )
        self._part(step, line, 0, PART_EVENT)

    # -- tool results ---------------------------------------------------------

    def _add_result(self, step: int, piece: _Piece, *, linked: bool) -> int:
        """`linked`: the step holds the call. ATIF requires `source_call_id` to
        name a call in the same step, so an orphan keeps its id in `extra`."""
        result: dict[str, Any] = {}
        extra: dict[str, Any] = {"is_error": piece.is_error}
        if linked:
            result["source_call_id"] = piece.call_id
        elif piece.call_id or piece.id_text:
            # As the renderer prints it (`TOOL_RESULT (<id>)`).
            extra["tool_use_id"] = piece.call_id or piece.id_text
        if piece.result_bytes is not None:
            extra["result_bytes"] = piece.result_bytes
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
    def _timestamp(ev: _Event) -> dict[str, Any]:
        return {"timestamp": ev.timestamp} if ev.timestamp else {}

    @staticmethod
    def _harness_extras(extra: dict[str, Any], ev: _Event) -> None:
        for name, safe in ev.extras:
            extra.setdefault(name, []).append(dict(safe))

    # -- the document ---------------------------------------------------------

    def finish(self, *, session_id: str, agent_name: str) -> dict[str, Any]:
        if not self.steps:
            # ATIF requires one step; a session whose events carry nothing
            # readable still gets a document, marked as such.
            self._add_step(
                {
                    "source": "system",
                    "message": "",
                    "extra": {"subtype": "empty", "probe_parts": []},
                }
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
        totals = {k: min(v, JSON_INT_MAX) for k, v in totals.items()}
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
