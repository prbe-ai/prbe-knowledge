"""A protocol-3 session's uploaded fragments -> what the ingest pass reads.

A protocol-3 client uploads one ATIF fragment per event (fragment.py builds
them; kb/session_receipts.py is the door) instead of the events. The ingest
pass then needs two things from them, and they come from different places:

  LINES (index, evidence spans, extraction). Each fragment carries its event's
  index Line, rendered by the client with the same renderer the engine uses
  (engine/shared/transcript_render.py, vendored with fragment.py). The pass
  reads those Lines directly, in ordinal order, and never waits on `fold`: the
  Lines of the same events are the same Lines protocol 2 renders, so search,
  evidence spans and the extraction cache see the same text whichever protocol
  a session came in on.

  TRAJECTORY. `fold(fragments)`, off the event loop (`fold_fragments` is the
  pool entry point), on the same live throttle and completing pass as a
  protocol-2 build.

UNTRUSTED. A fragment is the client's word. `fold` reads it as data; so does
`fragment_lines`: a Line it cannot read (not an object, a field of the wrong
type) becomes an empty Line at the fragment's ordinal, so every later Line
keeps its place, and is counted. Text past FRAGMENT_LINE_MAX_CHARS is cut and
counted. Nothing here raises on a fragment's content.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

from engine.ingest.atif.fold import BuildResult, fold, line_of
from engine.ingest.atif.fragment import Kind, fragment
from engine.shared.transcript_render import Line

#: The longest Line text the pass reads from one fragment. The event path has
#: no per-Line cap of its own: one event's text is bounded by the batch it came
#: in. The research-os gateway caps a protocol-3 batch at 6,000,000 bytes
#: (app/ingestion/sessions_router.py MAX_FRAGMENT_BODY_BYTES, research-os #2397;
#: a protocol-2 batch at 2,000,000), and a Line is no longer than the JSON
#: string it travels in. The same bound here never cuts a Line the gateway let
#: through, and still holds if that gateway changes, because this engine's own
#: webhook has no body cap.
FRAGMENT_LINE_MAX_CHARS = 6_000_000

#: Fragment kinds whose `message` is the event's top-level `content` (what a
#: session document previews): fragment.py `_system` / `_other`.
_CONTENT_KINDS = frozenset({Kind.SYSTEM.value, Kind.OTHER.value})


def fragment_ordinal(item: object) -> int | None:
    """The event ordinal a fragment covers: its Line's `line_no` (fragment.py).
    The door (kb/session_receipts.py) refuses a batch whose fragments do not
    carry contiguous ones, so every stored fragment has one."""
    line = item.get("line") if isinstance(item, dict) else None
    ordinal = line.get("line_no") if isinstance(line, dict) else None
    return ordinal if type(ordinal) is int else None


@dataclass(slots=True)
class FragmentLines:
    """A session's Lines read from its fragments, and what reading them cost."""

    lines: list[Line] = field(default_factory=list)
    #: A Line that could not be read (not an object, a field of the wrong
    #: type): served empty at the fragment's ordinal.
    invalid: int = 0
    #: The client's renderer raised on the event (`line_error`, a non-empty
    #: string): its text is empty by construction.
    line_errors: int = 0
    #: The client could not map the event (`error`, a non-empty string, which
    #: is what fold counts unparsed), including a client's fallback fragment
    #: for an event it could not fragment at all.
    errors: int = 0
    #: Text cut to FRAGMENT_LINE_MAX_CHARS.
    truncated: int = 0

    @property
    def degraded(self) -> bool:
        return bool(self.invalid or self.line_errors or self.errors or self.truncated)

    def counts(self) -> dict[str, int]:
        """For a log line: counts only, never content."""
        return {
            "fragments": len(self.lines),
            "invalid_lines": self.invalid,
            "line_errors": self.line_errors,
            "unmapped": self.errors,
            "truncated": self.truncated,
        }


def fragment_lines(fragments: Iterable[Any]) -> FragmentLines:
    """One Line per fragment, in the order given (the caller's is ordinal order)."""
    out = FragmentLines()
    for item in fragments:
        line = line_of(item)
        if line is None:
            out.invalid += 1
            line = Line(line_no=fragment_ordinal(item), text="")
        if len(line.text) > FRAGMENT_LINE_MAX_CHARS:
            out.truncated += 1
            line = replace(line, text=line.text[:FRAGMENT_LINE_MAX_CHARS])
        if isinstance(item, dict):
            if _named(item.get("line_error")):
                out.line_errors += 1
            if _named(item.get("error")):
                out.errors += 1
        out.lines.append(line)
    return out


def _named(value: object) -> bool:
    """An error a fragment reports: a non-empty string (as fold reads `error`)."""
    return isinstance(value, str) and bool(value)


def session_fragments(fragments: list[Any], events: list[Any]) -> list[Any]:
    """A protocol-3 session's fragments in ordinal order, with any event that
    came on another protocol's batch mapped by this engine's own `fragment`.

    The door pins a session to one protocol, so `events` is empty unless the
    stored batches somehow mix; then each batch is still read by its own
    protocol and the two kinds interleave by ordinal. An item with no ordinal
    sorts last, in the order given.
    """
    if not events:
        return list(fragments)
    items = [(fragment_ordinal(f), f) for f in fragments]
    for ev in events:
        if isinstance(ev, dict):
            line_no = ev.get("line_no")
            items.append((line_no if type(line_no) is int else None, fragment(ev)))
    items.sort(key=lambda item: (item[0] is None, item[0] or 0))
    return [f for _ordinal, f in items]


def first_content(fragments: list[Any]) -> str:
    """What a session document previews: the first event's top-level `content`,
    which a fragment carries as `message` for a system or other event only
    (kb/handlers/claude_code.py `_build_session_doc`, from the events)."""
    if not fragments or not isinstance(fragments[0], dict):
        return ""
    first = fragments[0]
    message = first.get("message")
    if first.get("kind") in _CONTENT_KINDS and isinstance(message, str):
        return message
    return ""


def fold_fragments(fragments: list[Any], session_id: str, agent_name: str) -> BuildResult:
    """`fold` as one plain-data call the ingest pass can run in `cpu_pool`'s
    processes."""
    return fold(fragments, session_id=session_id, agent_name=agent_name)
