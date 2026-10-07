"""One stored event -> one fragment of its session's ATIF trajectory.

The per-event half of the probe-events/1 -> ATIF builder: everything the
builder (build_reference.py, the stateful builder this was split from) works
out about an event without looking at any other event. `fold.fold` does the
rest -- joining the pieces of one model call, pairing results with calls,
which step carries an inference's metrics, step ids, the document -- so
`fold(map(fragment, events))` is the trajectory the builder makes of `events`
(tests/test_atif_fragment_fold.py holds them equal).

PURE AND STATELESS. A fragment is a function of its event alone: JSON in, JSON
out, deterministic, no clock, no settings. A client that builds fragments from
the batch it is uploading therefore sends byte-identical bodies on a retry or a
resume, which is what per-batch receipts need. This module is VENDORED into
the tap together with engine/shared/transcript_render.py, so it imports the
standard library and that module, nothing else.

PORTABLE. The tap runs on Python 3.10 and up, the engine on 3.12 and up, and
both must build the same bytes from the same event: no syntax or standard
library newer than 3.10, and no call whose result moved between versions
(`datetime.fromisoformat` accepts more from 3.11 on, so a timestamp is carried
as written and fold, which runs only in the engine, decides whether it parses).
tests/test_atif_fragment_fold.py checks both files.

UNTRUSTED ONCE BUILT. Whoever made a fragment, fold re-reads every field,
re-applies every bound and copies only what it knows (fold.py).

THE FRAGMENT (FRAGMENT_VERSION 1)
---------------------------------
A JSON object, one per event, keyed by the event's ordinal (`line.line_no`).
A key that does not apply is absent. Unknown keys are ignored by fold.

  every fragment
    v               int      FRAGMENT_VERSION
    line            object   the event's index Line, as transcript_render.line_from_event
                             gives it: line_no int|null (the event ordinal), text str,
                             user_turn, compact_boundary, compact_summary bool
    line_error      str      the event renderer raised (exception class); line.text is ""
    kind            str      user | assistant | system | other | none (Kind)
    error           str      mapping raised (exception class): what precedes it in the
                             fragment was mapped, and the event counts as unparsed
  kinds user, assistant, system, other
    timestamp       str      the event's timestamp as written (a non-empty string);
                             fold keeps it only when it parses as ISO-8601
    extras          object   {codex_extras|pi_extras|kimi_extras: {key: scalar}}, each
                             flat, <= 64 keys, keys <= 64 and strings <= 256 chars
    lineage         object   Claude Code's event ids, carried for mapping sidechains
                             later; fold does not read it: uuid str, parentUuid
                             str|null, logicalParentUuid str|null (each <= 256 chars)
  user
    compaction      true     a compaction summary (`isCompactSummary`)
    origin          str      "user_shell": a command the researcher typed
    message         str      the prompt when the content is a string (harness-stripped)
    parts           list     in block order, each {type, seq, ...}; seq = block index:
                               text         text str (harness-stripped, non-empty)
                               tool_result  call_id str?  the id, when it is a string
                                            id_text str?  the id as printed, when it is not
                                            is_error bool, result_bytes int?
    dropped_blocks  list     other block types (<= 32, each <= 64 chars), only beside a
                             text part
  assistant
    inference_id    str      <= 256 chars
    origin          str      "user_shell": a command the researcher typed (pi's `!`)
    model           str      <= 256 chars
    usage           object   input_tokens, output_tokens, cache_read_input_tokens,
                             cache_creation_input_tokens: ints >= 0, each optional,
                             capped at 2**63 - 1
    parts           list     text      text str (a string content is one, at seq 0)
                             thinking  text str (not blank)
                             tool_call id str?, name str, summary str?,
                                       stats {added_lines int?, removed_lines int?,
                                              replace_all true?}?
    stop_reason     str      the message's `stop_reason` when it is a non-empty string,
                             whatever it is (<= 256 chars): `end_turn` and `tool_use`
                             too, and on any content. What ended the reply -- the
                             engine's live trajectory writes a turn end at once -- and
                             nothing renders it; fold does not read it
    stop            object   {seq: int, reason: str}: a stop reason the renderer prints,
                             as it prints it (whole), on block-list content only: the
                             one fold renders
  system
    message str (the content), subtype str, agent_version str (Codex `cli_version`, <= 64)
  other
    event_type str (the raw type, uncapped: the index prints it whole), message str,
    attachment_type str (<= 64)
  none
    the event adds its Line and nothing else: its raw is not an object, or it is
    an assistant event without a message object (which does not end a model call)

Never in a fragment: a tool call's input (arguments), a tool's output, an
image, any other block's content.
"""

from __future__ import annotations

from dataclasses import fields
from enum import Enum
from typing import Any

from engine.shared.transcript_render import Line, line_from_event, renders_stop, strip_harness

#: Bumped when a fragment's meaning changes. fold reads every version it lists
#: as supported, mixed within one session (a tap can update mid-session).
FRAGMENT_VERSION = 1


class Kind(str, Enum):
    """What an event contributes, by its probe-events/1 type."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    #: Any other type: a system step that names it.
    OTHER = "other"
    #: Its Line only.
    NONE = "none"


class PieceType(str, Enum):
    """One block of an event's message, in a fragment's `parts`."""

    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


#: probe-events/1 `origin` of a command the researcher typed, not the model.
USER_SHELL = "user_shell"

#: Harness-only fields the sanitizers carry: (event key, name in a fragment and
#: on a step), in the order a step lists them.
HARNESS_EXTRAS = (
    ("_codex_extras", "codex_extras"),
    ("_pi_extras", "pi_extras"),
    ("_kimi_extras", "kimi_extras"),
)

#: probe-events/1 event ids a fragment carries (`lineage`) but nothing folds yet:
#: the builder maps no subagent or sidechain, and these are what would.
LINEAGE_KEYS = ("uuid", "parentUuid", "logicalParentUuid")

#: Bounds on what a client-controlled value may put into a document readers
#: are served (GET /trajectory). Upload is the trust boundary: a patched or
#: compromised tap posts whatever it likes, so fold applies them again.
EXTRA_MAX_KEYS = 64
EXTRA_KEY_CHARS = 64
EXTRA_VALUE_CHARS = 256
TYPE_NAME_CHARS = 64
DROPPED_BLOCKS_MAX = 32

#: The largest count a fragment or a trajectory carries: a JSON writer such as
#: orjson refuses integers past 64 bits, and fold sums these.
JSON_INT_MAX = 2**63 - 1

USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def fragment(event: dict[str, Any]) -> dict[str, Any]:
    """The fragment of one merged event (`{"line_no", "raw"}`). Never raises on
    an event's content: a shape it cannot map is recorded in `error`, one the
    renderer cannot render in `line_error`."""
    if not isinstance(event, dict):
        # Every consumer skips a non-object event (no Line, no step).
        raise TypeError("fragment() takes one merged event object")
    out: dict[str, Any] = {"v": FRAGMENT_VERSION}
    try:
        line = line_from_event(event)
        line_error = None
    except Exception as exc:  # the legacy renderer's own gaps cost the text only
        line, line_error = _unrendered_line(event), type(exc).__name__
    out["line"] = line_fields(line)
    if line_error is not None:
        out["line_error"] = line_error
    raw = event.get("raw")
    kind = _kind(raw)
    out["kind"] = kind.value
    if kind is Kind.NONE:
        return out
    # Per-event isolation, as the builder's: a shape never seen before costs
    # this event, not the session. What was mapped before the failure stays.
    try:
        _MAPPERS[kind](out, raw)
    except Exception as exc:
        out["error"] = type(exc).__name__
    return out


def line_fields(line: Line) -> dict[str, Any]:
    """A `Line` as the fragment's `line` object: every field, by name."""
    return {f.name: getattr(line, f.name) for f in fields(Line)}


def fragment_line(fragment: dict[str, Any]) -> Line:
    """The index `Line` of a fragment this module built (fold.line_of reads an
    untrusted one)."""
    return Line(**fragment["line"])


# -- the bounds both halves apply ---------------------------------------------------


def non_negative_int(value: Any) -> int | None:
    """An int >= 0, capped at JSON_INT_MAX; None for anything else (bools included)."""
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        return None
    return min(value, JSON_INT_MAX)


def short(value: Any, limit: int) -> str | None:
    return value[:limit] if isinstance(value, str) and value else None


def usage_counts(source: Any) -> dict[str, int] | None:
    """The token counts of a usage report, ints >= 0 only."""
    if not isinstance(source, dict):
        return None
    out = {k: v for k in USAGE_KEYS if (v := non_negative_int(source.get(k))) is not None}
    return out or None


def safe_extras(value: dict[str, Any]) -> dict[str, Any]:
    """A harness's extras as flat scalars: strings capped, nested values gone.

    The sanitizers stash harness-only metadata under `_codex_extras` /
    `_pi_extras` / `_kimi_extras`; older taps put whole tool-call inputs there
    (Codex `action`, with `env`). Only short scalars reach a reader.
    """
    out: dict[str, Any] = {}
    for key, item in value.items():
        if len(out) >= EXTRA_MAX_KEYS:
            break
        if not isinstance(key, str) or not key:
            continue
        if isinstance(item, bool | int | float):
            out[key[:EXTRA_KEY_CHARS]] = item
        elif isinstance(item, str):
            out[key[:EXTRA_KEY_CHARS]] = item[:EXTRA_VALUE_CHARS]
    return out


def call_stats(stats: Any) -> dict[str, Any] | None:
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


# -- per kind -----------------------------------------------------------------------


def _unrendered_line(event: dict[str, Any]) -> Line:
    """`line_from_event`'s Line without its text, for an event it cannot render."""
    raw = event.get("raw")
    line_no = event.get("line_no")
    facts = raw if isinstance(raw, dict) else event
    return Line(
        line_no=line_no if isinstance(line_no, int) else None,
        text="",
        user_turn=facts.get("type") == "user" and not facts.get("isCompactSummary"),
        compact_boundary=(
            facts.get("type") == "system" and facts.get("subtype") == "compact_boundary"
        ),
        compact_summary=bool(facts.get("isCompactSummary")),
    )


def _kind(raw: Any) -> Kind:
    if not isinstance(raw, dict):
        return Kind.NONE
    ev_type = raw.get("type")
    if ev_type == "assistant":
        # No message: the builder maps nothing and leaves the model call open.
        return Kind.ASSISTANT if isinstance(raw.get("message"), dict) else Kind.NONE
    if ev_type == "user":
        return Kind.USER
    if ev_type == "system":
        return Kind.SYSTEM
    return Kind.OTHER


def _stamp_and_extras(out: dict[str, Any], raw: dict[str, Any]) -> None:
    stamp = raw.get("timestamp")
    if isinstance(stamp, str) and stamp:
        # As written: whether it parses is fold's call (see PORTABLE above).
        out["timestamp"] = stamp
    extras: dict[str, Any] = {}
    for key, name in HARNESS_EXTRAS:
        value = raw.get(key)
        if isinstance(value, dict):
            safe = safe_extras(value)
            if safe:
                extras[name] = safe
    if extras:
        out["extras"] = extras
    lineage: dict[str, str | None] = {}
    for key in LINEAGE_KEYS:
        value = raw.get(key)
        if isinstance(value, str):
            lineage[key] = value[:EXTRA_VALUE_CHARS]
        elif key in raw and value is None:
            # A null parent is a fact (the first event), not a missing one.
            lineage[key] = None
    if lineage:
        out["lineage"] = lineage


def _user(out: dict[str, Any], raw: dict[str, Any]) -> None:
    msg = raw.get("message")
    if not isinstance(msg, dict):
        # Still a user event: it ends the model call in progress.
        return
    if raw.get("isCompactSummary"):
        out["compaction"] = True
    if raw.get("origin") == USER_SHELL:
        out["origin"] = USER_SHELL
    _stamp_and_extras(out, raw)
    content = msg.get("content")
    if isinstance(content, str) and content:
        cleaned = strip_harness(content)
        if cleaned:
            out["message"] = cleaned
        return
    if not isinstance(content, list):
        return
    parts: list[dict[str, Any]] = []
    dropped: list[str] = []
    try:
        for seq, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                cleaned = strip_harness(b.get("text") or "")
                if cleaned:
                    parts.append({"type": PieceType.TEXT.value, "seq": seq, "text": cleaned})
            elif bt == "tool_result":
                parts.append(_result(seq, b))
            elif isinstance(bt, str) and len(dropped) < DROPPED_BLOCKS_MAX:
                dropped.append(bt[:TYPE_NAME_CHARS])
    finally:
        # A block that raised keeps the ones before it, as the builder does.
        if parts:
            out["parts"] = parts
    if dropped and any(p["type"] == PieceType.TEXT for p in parts):
        # The builder notes them on the prompt step, which only text makes.
        out["dropped_blocks"] = dropped


def _result(seq: int, block: dict[str, Any]) -> dict[str, Any]:
    piece: dict[str, Any] = {"type": PieceType.TOOL_RESULT.value, "seq": seq}
    tool_use_id = block.get("tool_use_id")
    if isinstance(tool_use_id, str) and tool_use_id:
        piece["call_id"] = tool_use_id
    elif tool_use_id:
        # Names no call; kept as the renderer prints it (`TOOL_RESULT (<id>)`).
        piece["id_text"] = f"{tool_use_id}"
    piece["is_error"] = bool(block.get("is_error"))
    if isinstance(block.get("result_bytes"), int):
        # The renderer prints a size only for an int; anything else is noise.
        piece["result_bytes"] = block["result_bytes"]
    return piece


def _assistant(out: dict[str, Any], raw: dict[str, Any]) -> None:
    msg = raw["message"]
    inference = short(raw.get("inference_id"), EXTRA_VALUE_CHARS)
    if inference:
        out["inference_id"] = inference
    if raw.get("origin") == USER_SHELL:
        out["origin"] = USER_SHELL
    model = short(msg.get("model"), EXTRA_VALUE_CHARS)
    if model:
        out["model"] = model
    # probe-events/1 top-level `usage`, else Claude Code's own `message.usage`.
    usage = usage_counts(
        raw.get("usage") if isinstance(raw.get("usage"), dict) else msg.get("usage")
    )
    if usage:
        out["usage"] = usage
    # Every stop reason, printed or not (`stop` below is only the printed one):
    # Claude Code's `end_turn` is how a reader knows the model is waiting for
    # the researcher. Before the content, so no content shape can lose it.
    stop_reason = short(msg.get("stop_reason"), EXTRA_VALUE_CHARS)
    if stop_reason:
        out["stop_reason"] = stop_reason
    _stamp_and_extras(out, raw)
    content = msg.get("content")
    if isinstance(content, str) and content:
        out["parts"] = [{"type": PieceType.TEXT.value, "seq": 0, "text": content}]
        return
    if not isinstance(content, list):
        return
    parts: list[dict[str, Any]] = []
    try:
        for seq, b in enumerate(content):
            if not isinstance(b, dict):
                continue
            bt = b.get("type")
            if bt == "text":
                text = b.get("text") or ""
                if text:
                    parts.append(
                        {
                            "type": PieceType.TEXT.value,
                            "seq": seq,
                            "text": text if isinstance(text, str) else f"{text}",
                        }
                    )
            elif bt == "thinking":
                text = b.get("thinking") or ""
                if isinstance(text, str) and text.strip():
                    parts.append({"type": PieceType.THINKING.value, "seq": seq, "text": text})
            elif bt == "tool_use":
                parts.append(_call(seq, b))
    finally:
        if parts:
            out["parts"] = parts
    stop = msg.get("stop_reason")
    if renders_stop(stop):
        out["stop"] = {"seq": len(content), "reason": stop if isinstance(stop, str) else f"{stop}"}


def _call(seq: int, block: dict[str, Any]) -> dict[str, Any]:
    piece: dict[str, Any] = {"type": PieceType.TOOL_CALL.value, "seq": seq}
    call_id = block.get("id")
    if isinstance(call_id, str) and call_id:
        piece["id"] = call_id
    name = block.get("name") or "tool"
    piece["name"] = name if isinstance(name, str) else f"{name}"
    summary = block.get("summary")
    if summary:
        # As the renderer prints it. Capture never ships a tool's input.
        piece["summary"] = summary if isinstance(summary, str) else f"{summary}"
    stats = call_stats(block.get("stats"))
    if stats:
        piece["stats"] = stats
    return piece


def _system(out: dict[str, Any], raw: dict[str, Any]) -> None:
    content = raw.get("content")
    if isinstance(content, str) and content:
        out["message"] = content
    subtype = raw.get("subtype")
    if subtype:
        # As the renderer prints it (`SYSTEM (<subtype>)`).
        out["subtype"] = subtype if isinstance(subtype, str) else f"{subtype}"
    _stamp_and_extras(out, raw)
    codex = raw.get("_codex_extras")
    if isinstance(codex, dict):
        version = short(codex.get("cli_version"), TYPE_NAME_CHARS)
        if version:
            out["agent_version"] = version


def _other(out: dict[str, Any], raw: dict[str, Any]) -> None:
    event_type = raw.get("type")
    if isinstance(event_type, str):
        out["event_type"] = event_type
    content = raw.get("content")
    if isinstance(content, str) and content:
        out["message"] = content
    attachment = raw.get("attachment")
    if isinstance(attachment, dict):
        attachment_type = short(attachment.get("type"), TYPE_NAME_CHARS)
        if attachment_type:
            out["attachment_type"] = attachment_type
    _stamp_and_extras(out, raw)


_MAPPERS = {
    Kind.USER: _user,
    Kind.ASSISTANT: _assistant,
    Kind.SYSTEM: _system,
    Kind.OTHER: _other,
}
