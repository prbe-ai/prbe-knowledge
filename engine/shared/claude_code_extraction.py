"""Per-session knowledge-unit extraction using Sonnet tool-use as structured output.

A single tool `emit_units` with one parameter shape per unit type. The model
is forced to call this tool; tool input is the result. We use tool-use rather
than `response_format` JSON mode because tool-use enforces the schema more
reliably across longer context windows.

Phase-0b: routes through `shared.llm.acompletion` (chunk C migration) so
tenants without provider API keys can use it via the central LiteLLM
gateway.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, fields, replace
from typing import Any

from engine.shared.config import get_settings
from engine.shared.llm import gateway_url
from engine.shared.llm_tools import ToolCallParseError, forced_tool_call
from engine.shared.logging import get_logger
from engine.shared.transcript_render import (
    _events_to_text,
    line_for_offset,
    render_indexed,
)


@dataclass(slots=True)
class SegmentRef:
    """Where in its session a unit came from.

    A long session is extracted in parts, so without this every unit is a
    free-floating fact about a session that may have run for hours. Carrying
    the part number, the total, and WHY the part began lets a reader put the
    units back in order and tell a real narrative break from a mechanical one.

    `boundary` says what opened this segment:
      session_start — the beginning of the session
      compaction    — the agent ran out of context here, which is a genuine
                      chapter break the agent itself chose
      size          — no boundary was available and we split to fit; this is a
                      cut through continuous work and means nothing semantically
    """

    index: int          # 1-based position within the session
    total: int          # how many parts the session was split into
    boundary: str       # session_start | compaction | size
    #: What this stretch of work was FOR. Lives here because it is a property of
    #: the segment, not of any one unit, and every unit is already stamped with
    #: its segment — so one extracted sentence gives every unit its context
    #: instead of asking the model to repeat itself per unit.
    #:
    #: Without it a decision is a well-argued local tradeoff orphaned from the
    #: work it served: you can tell WHAT was decided and why that option won,
    #: but not why the work was happening at all. Audited over 72 real
    #: decisions, that was the one thing missing from every single one.
    objective: str = ""
    motivation: str = ""  # why the objective mattered, when the transcript says
    start_line_no: int | None = None
    end_line_no: int | None = None
    # Position of this unit among units of the SAME kind in the SAME segment.
    # Deliberately segment-local: a document id built from a session-wide
    # counter shifts every id after any segment that returns a different number
    # of units, so one changed segment silently repoints every later document.
    position: int = 0


@dataclass(slots=True)
class QA:
    prompt: str
    outcome: str
    tags: list[str] = field(default_factory=list)
    #: A short VERBATIM quote from the transcript that supports this unit.
    #: Checked server-side: a quote that matches nothing was not in the
    #: transcript, which is the one fabrication signal available without a
    #: human. `evidence_verified` and `anchor_line_no` are set from that check,
    #: never by the model.
    evidence: str = ""
    evidence_verified: bool = False
    anchor_line_no: int | None = None
    confidence: str = ""  # see _CONFIDENCE
    segment: SegmentRef | None = None


@dataclass(slots=True)
class CodeChange:
    """One logical change, however many files it touched.

    Was one unit per FILE, which is not how anybody describes their own work: a
    single fix that spanned six files came back as six documents that each told
    a sixth of it, and on a real backfill that shape produced 93 units for one
    session — 32% of everything it emitted. Nobody reads a codebase that way and
    nobody searches one that way either.

    `before`/`after` are gone. The tap deliberately ships no diffs, and 78 of
    those 93 came back with both sides empty; the rare case where the transcript
    really did contain the text is served by the shared `evidence` quote.
    """

    summary: str                          # the change, one line
    kind: str = ""                        # feature | fix | refactor | test | docs | config | infra
    files: list[str] = field(default_factory=list)
    rationale: str = ""
    #: A short VERBATIM quote from the transcript that supports this unit.
    #: Checked server-side: a quote that matches nothing was not in the
    #: transcript, which is the one fabrication signal available without a
    #: human. `evidence_verified` and `anchor_line_no` are set from that check,
    #: never by the model.
    evidence: str = ""
    evidence_verified: bool = False
    anchor_line_no: int | None = None
    confidence: str = ""  # see _CONFIDENCE
    segment: SegmentRef | None = None


@dataclass(slots=True)
class Decision:
    question: str
    options_considered: list[str]
    chosen: str
    rationale: str
    #: How this choice advances the segment's objective. `rationale` answers
    #: "why this option over the others"; `serves` answers "why we were
    #: choosing at all". Asking "why did we decide this?" needs both, and the
    #: second was the half that did not exist.
    serves: str = ""
    #: WHO made the call. Currently not just absent but actively lost: audited
    #: over 72 real decisions, only 8 rationales credited the user at all, and
    #: in one session 0 of 12 did while tracing the transcript showed roughly
    #: five were the user's call outright ("using the server as the source and
    #: keeping the local locks as a fast-path" came back as though the agent had
    #: reasoned its way there).
    #:
    #: This is the field that makes per-person norms possible: a decision the
    #: agent took alone says nothing about how a researcher works, and one they
    #: overrode says a great deal.
    decided_by: str = ""  # see _DECIDED_BY
    #: A short VERBATIM quote from the transcript that supports this unit.
    #: Checked server-side: a quote that matches nothing was not in the
    #: transcript, which is the one fabrication signal available without a
    #: human. `evidence_verified` and `anchor_line_no` are set from that check,
    #: never by the model.
    evidence: str = ""
    evidence_verified: bool = False
    anchor_line_no: int | None = None
    confidence: str = ""  # see _CONFIDENCE
    segment: SegmentRef | None = None


@dataclass(slots=True)
class FileRef:
    files: list[str]
    context: str
    #: A short VERBATIM quote from the transcript that supports this unit.
    #: Checked server-side: a quote that matches nothing was not in the
    #: transcript, which is the one fabrication signal available without a
    #: human. `evidence_verified` and `anchor_line_no` are set from that check,
    #: never by the model.
    evidence: str = ""
    evidence_verified: bool = False
    anchor_line_no: int | None = None
    confidence: str = ""  # see _CONFIDENCE
    segment: SegmentRef | None = None


@dataclass(slots=True)
class UnitBundle:
    #: The goal for this bundle's segment. Carried here rather than in module
    #: state because segments extract CONCURRENTLY — a shared slot would hand
    #: one segment's objective to another's units, which is worse than having
    #: none at all.
    objective: str = ""
    motivation: str = ""
    qa: list[QA] = field(default_factory=list)
    code_change: list[CodeChange] = field(default_factory=list)
    decision: list[Decision] = field(default_factory=list)
    file_ref: list[FileRef] = field(default_factory=list)


# Tool name + JSON Schema for the forced tool call. The schema is
# OpenAI-shaped (`parameters`); LiteLLM translates it to Anthropic
# `input_schema` and Google `function_declarations.parameters` per provider.
_TOOL_NAME = "emit_units"
_TOOL_DESCRIPTION = (
    "Emit structured knowledge units extracted from a coding-agent session."
)
_TOOL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "object",
            "description": (
                "What this stretch of the session was trying to achieve. One "
                "goal for the whole transcript below, not one per unit."
            ),
            "properties": {
                "objective": {
                    "type": "string",
                    "description": (
                        "The objective in one sentence, as it actually turned "
                        "out — not a restatement of the opening request. If the "
                        "work changed direction, say what it became."
                    ),
                },
                "motivation": {
                    "type": "string",
                    "description": (
                        "Why the objective mattered — what it unblocks, what "
                        "breaks without it. Only if the transcript says. Empty "
                        "otherwise; do not invent a business reason."
                    ),
                },
            },
            "required": ["objective"],
        },
        "qa": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "outcome": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "evidence": {
                        "type": "string",
                        "description": (
                            "A SHORT VERBATIM quote from the transcript above "
                            "that supports this unit — copied exactly, not "
                            "paraphrased, one or two lines. It is checked "
                            "against the transcript, so an approximate quote, "
                            "a tidied one, or two spans joined by an ellipsis "
                            "all fail the check."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high: the transcript states this outright. "
                            "medium: strongly implied but assembled from "
                            "several turns. low: a reasonable reading that "
                            "someone could disagree with. Prefer emitting a "
                            "low-confidence unit with honest evidence over "
                            "omitting it, and never raise confidence to make a "
                            "unit look better."
                        ),
                    },
                },
                "required": ["prompt", "outcome", "evidence", "confidence"],
            },
        },
        "code_change": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": (
                            "The change as one line, at the level a person would "
                            "describe their own work: 'Added a cardinality guard "
                            "to the metric write path', not 'edited store.py'."
                        ),
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["feature", "fix", "refactor", "test", "docs",
                                 "config", "infra"],
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Every file this one change spanned.",
                    },
                    "rationale": {"type": "string"},
                    "evidence": {
                        "type": "string",
                        "description": (
                            "A SHORT VERBATIM quote from the transcript above "
                            "that supports this unit — copied exactly, not "
                            "paraphrased, one or two lines. It is checked "
                            "against the transcript, so an approximate quote "
                            "fails the check."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high: the transcript states this outright. "
                            "medium: strongly implied but assembled from "
                            "several turns. low: a reasonable reading that "
                            "someone could disagree with. Prefer emitting a "
                            "low-confidence unit with honest evidence over "
                            "omitting it, and never raise confidence to make a "
                            "unit look better."
                        ),
                    },
                },
                "required": ["summary", "kind", "files", "rationale", "evidence", "confidence"],
            },
        },
        "decision": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options_considered": {"type": "array", "items": {"type": "string"}},
                    "chosen": {"type": "string"},
                    "rationale": {
                        "type": "string",
                        "description": "Why this option beat the others.",
                    },
                    "serves": {
                        "type": "string",
                        "description": (
                            "How this choice advances the goal above — the link "
                            "between the local tradeoff and what the work was "
                            "for. Empty if the decision is incidental to it."
                        ),
                    },
                    "decided_by": {
                        "type": "string",
                        "enum": [
                            "user_directed",
                            "user_confirmed",
                            "agent_proposed_user_approved",
                            "agent_unilateral",
                            "agent_overrode_user",
                        ],
                        "description": (
                            "Who actually made the call. Attribute to the USER "
                            "whenever a user turn states the choice, even if the "
                            "reasoning around it is the assistant's. "
                            "agent_unilateral is a real answer, not a fallback. "
                            "Use agent_overrode_user only when a user turn asked "
                            "for something else and it was not done."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "A SHORT VERBATIM quote from the transcript above "
                            "that supports this unit — copied exactly, not "
                            "paraphrased, one or two lines. It is checked "
                            "against the transcript, so an approximate quote "
                            "fails the check."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high: the transcript states this outright. "
                            "medium: strongly implied but assembled from "
                            "several turns. low: a reasonable reading that "
                            "someone could disagree with. Prefer emitting a "
                            "low-confidence unit with honest evidence over "
                            "omitting it, and never raise confidence to make a "
                            "unit look better."
                        ),
                    },
                },
                "required": [
                    "question", "options_considered", "chosen", "rationale",
                    "serves", "decided_by", "evidence", "confidence",
                ],
            },
        },
        "file_ref": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": "string"},
                    "evidence": {
                        "type": "string",
                        "description": (
                            "A SHORT VERBATIM quote from the transcript above "
                            "that supports this unit — copied exactly, not "
                            "paraphrased, one or two lines. It is checked "
                            "against the transcript, so an approximate quote "
                            "fails the check."
                        ),
                    },
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": (
                            "high: the transcript states this outright. "
                            "medium: strongly implied but assembled from "
                            "several turns. low: a reasonable reading that "
                            "someone could disagree with. Prefer emitting a "
                            "low-confidence unit with honest evidence over "
                            "omitting it, and never raise confidence to make a "
                            "unit look better."
                        ),
                    },
                },
                "required": ["files", "context", "evidence", "confidence"],
            },
        },
    },
    "required": ["goal", "qa", "code_change", "decision", "file_ref"],
}


# SEGMENTATION, not truncation.
#
# This used to keep the last 2000 events and send them as raw JSON. Both halves
# were wrong. Measured over six real sessions, that payload was 476k-929k
# tokens against a 200k-token context — every large session would have failed
# outright — and the tail-only window meant that in four of six compacted
# sessions ZERO pre-compaction events reached the extractor. The conversation
# was never lost (it is on disk, shipped, and in the indexed document); only
# the mining missed it.
#
# So: render the transcript instead of dumping JSON (3.7x-9.3x smaller, and the
# same prose the chunker already reads), then split the session and extract
# every part.
#
# Compaction boundaries are the natural split. One exists precisely BECAUSE
# that much conversation filled a context window, so the segments come
# pre-sized by the agent itself. Sessions that never compacted still need a
# size guard — one measured 1.1M-character session had no boundary at all — so
# an oversized segment is sub-split on turn boundaries.
_SEGMENT_CHAR_BUDGET = 260_000  # ~65-75k tokens of rendered transcript

# Bound on cost per session. A session needing more segments than this is
# pathological; the LAST _MAX_SEGMENTS are kept (the conclusion matters most)
# and the drop is logged rather than silent.
_MAX_SEGMENTS = 16

# Extraction calls run concurrently, but a chatty session should not saturate
# the gateway on its own.
_SEGMENT_CONCURRENCY = 4

# The agent name is interpolated rather than hardcoded. Every Codex session was
# being introduced to the model as a Claude Code session, because CodexConnector
# inherits this extraction path wholesale — and the two transcripts do not look
# alike once translated (Codex renders far more tool calls per unit of human
# text), so telling the model the wrong provenance is telling it to expect the
# wrong shape.
_SYSTEM_TEMPLATE = (
    "You extract structured knowledge from one {agent} session. "
    "Return only the emit_units tool call. Be conservative — emit a unit only "
    "when the session clearly demonstrates the corresponding kind of insight. "
    "Empty arrays are valid.\n"
    "\n"
    "Reading the transcript: TOOL_USE lines carry the tool, what it acted on, "
    "and for file edits a measured [+added/-removed lines] count. Those counts "
    "are FACTS computed at capture time. Successful tool results are not shown "
    "at all; a TOOL_RESULT line means that call FAILED.\n"
    "\n"
    "COMPACTION SUMMARY lines are Claude's own summary of earlier turns in this "
    "same session, written when it ran out of context — not something the user "
    "said. Read them for intent, and never quote one as a user prompt.\n"
    "\n"
    "EVERY unit needs `evidence`: one CONTIGUOUS verbatim span copied from the "
    "transcript, and a `confidence`. Do not stitch two places together with an "
    "ellipsis and do not tidy the wording — the quote is checked against the "
    "transcript, and a stitched or tidied one fails and caps the unit at low "
    "confidence. If no single span supports the unit, quote the closest one and "
    "say low.\n"
    "\n"
    "`goal` first: say what this stretch of work was actually FOR, in one "
    "sentence. Not a paraphrase of the opening message — the objective as it "
    "turned out, including if the work changed direction. Then, on each "
    "decision, `serves` says how that particular choice advances it. Together "
    "they answer both halves of \"why did we decide this\": the technical "
    "reason, and what it was in service of.\n"
    "\n"
    "code_change is one LOGICAL CHANGE, not one file. A feature or a fix that "
    "touched six files is ONE code_change listing six files — never six. Group "
    "by the unit of work a person would name in a commit or a standup, and give "
    "each one a `kind`. If two edits served the same goal they are one change; "
    "if one file was touched for two unrelated reasons they are two.\n"
    "\n"
    "`evidence` is for a snippet the transcript actually contained. If it does "
    "not contain the text, leave evidence EMPTY — never reconstruct code you "
    "were not shown. The summary and rationale carry the meaning."
)

#: Ordinal, not a float. A model asked for 0.87 produces fake precision; asked
#: to pick one of three it has to commit. The machine-checked signal is
#: `evidence_verified`, which is what actually distinguishes a grounded unit
#: from a fluent one.
_CONFIDENCE = ("high", "medium", "low")

_AGENT_LABELS = {"claude_code": "Claude Code", "codex": "Codex"}

#: Authorship of a decision. Ordered from most to least human involvement.
_DECIDED_BY = (
    "user_directed",              # the user stated the choice
    "user_confirmed",             # the user endorsed a choice already on the table
    "agent_proposed_user_approved",
    "agent_unilateral",           # a real answer, not a missing one
    "agent_overrode_user",        # rare, and the highest-signal value here
)


log = get_logger(__name__)


def _is_compact_boundary(event: dict[str, Any]) -> bool:
    raw = event.get("raw")
    raw = raw if isinstance(raw, dict) else event
    return raw.get("type") == "system" and raw.get("subtype") == "compact_boundary"


def _is_compact_summary(event: dict[str, Any]) -> bool:
    raw = event.get("raw")
    raw = raw if isinstance(raw, dict) else event
    return bool(raw.get("isCompactSummary"))


def _split_on_compaction(
    events: list[dict[str, Any]],
) -> list[tuple[list[dict[str, Any]], str]]:
    """One segment per stretch between compactions, tagged with what opened it."""
    segments: list[tuple[list[dict[str, Any]], str]] = []
    current: list[dict[str, Any]] = []
    boundary = "session_start"
    for event in events:
        if _is_compact_boundary(event) and current:
            segments.append((current, boundary))
            current, boundary = [], "compaction"
        current.append(event)
    if current:
        segments.append((current, boundary))
    return segments or [([], "session_start")]


def _split_to_budget(segment: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Sub-split a segment that is still too large to send in one call.

    Splits on USER turns where possible: a request and the work it produced
    belong in the same call, and cutting mid-exchange is how you get a
    `decision` whose rationale sits in the next chunk.
    """
    if _rendered_size(segment) <= _SEGMENT_CHAR_BUDGET:
        return [segment]

    out: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    size = 0
    for event in segment:
        one = _rendered_size([event])
        starts_turn = _renders_as_user_turn(event)
        if current and size + one > _SEGMENT_CHAR_BUDGET and starts_turn:
            out.append(current)
            current, size = [], 0
        elif current and size + one > _SEGMENT_CHAR_BUDGET * 1.5:
            # No user turn arrived in time — a single enormous exchange. Cut
            # anyway rather than send something the model will refuse.
            out.append(current)
            current, size = [], 0
        current.append(event)
        size += one
    if current:
        out.append(current)
    return out


def _renders_as_user_turn(event: dict[str, Any]) -> bool:
    raw = event.get("raw")
    raw = raw if isinstance(raw, dict) else event
    return raw.get("type") == "user" and not raw.get("isCompactSummary")


def _rendered_size(events: list[dict[str, Any]]) -> int:
    return len(_events_to_text(events))


def _segment_session(
    events: list[dict[str, Any]],
) -> tuple[list[tuple[list[dict[str, Any]], str]], bool]:
    """(segments, capped) — every part of the session, oldest first.

    Each segment is paired with the reason it began, so a downstream reader can
    tell a chapter break the agent chose from a cut we made to fit a context
    window. Sub-splits of one compaction stretch are `size`; only the first
    inherits the real boundary.
    """
    segments: list[tuple[list[dict[str, Any]], str]] = []
    for chunk, boundary in _split_on_compaction(events):
        for offset, piece in enumerate(_split_to_budget(chunk)):
            segments.append((piece, boundary if offset == 0 else "size"))
    segments = [(evs, why) for evs, why in segments if evs]
    capped = len(segments) > _MAX_SEGMENTS
    if capped:
        segments = segments[-_MAX_SEGMENTS:]
    return segments, capped


def _line_bounds(events: list[dict[str, Any]]) -> tuple[int | None, int | None]:
    """First and last transcript line numbers in a segment, when present.

    These are the anchors that let a unit be located back in the transcript it
    came from — without them a unit says which PART of a session it belongs to
    but not where.
    """
    nums = [e.get("line_no") for e in events if isinstance(e.get("line_no"), int)]
    return (min(nums), max(nums)) if nums else (None, None)


def _ground_units(bundle: UnitBundle, transcript: str, spans) -> UnitBundle:
    """Check every unit's quote against the transcript it was drawn from.

    This is the only fabrication signal available without a human reading the
    session. A model asked for a quote and answering with one that appears
    nowhere has told us something about the unit that the unit's own prose
    never would.

    Two things fall out of a match, and neither is taken from the model:
    `evidence_verified`, and `anchor_line_no` — the transcript line the quote
    sits on. That anchor is what lets two units inside one segment be ordered
    against each other; segment bounds alone put ~33 units in an identical
    1000-event window.

    Confidence is DOWNGRADED on a failed match, never upgraded on a passing
    one. Matching a quote proves the words were said, not that the unit read
    them correctly.
    """
    normalised = _collapse(transcript)
    for unit in (*bundle.qa, *bundle.code_change, *bundle.decision, *bundle.file_ref):
        quote = (unit.evidence or "").strip()
        if unit.confidence not in _CONFIDENCE:
            unit.confidence = "low"
        if not quote:
            unit.confidence = "low"
            continue
        offset = _locate(normalised, quote)
        if offset < 0:
            unit.evidence_verified = False
            # An unverifiable quote caps the unit at low no matter what the
            # model claimed for it.
            unit.confidence = "low"
            continue
        unit.evidence_verified = True
        unit.anchor_line_no = line_for_offset(spans, _expand(normalised, offset))
    return bundle


#: How much of a quote has to match. A whole-string match is the ideal and is
#: tried first, but models reproduce the OPENING of a passage faithfully and
#: drift later — measured on a real session, every failing quote matched its
#: first several words and diverged after. Failing those outright made the
#: check measure verbosity rather than fabrication.
#:
#: 60 contiguous characters is still a real check: a unit invented out of
#: nothing does not accidentally reproduce sixty characters of a transcript it
#: never saw.
_QUOTE_MATCH_PREFIX = 60


def _locate(normalised: str, quote: str) -> int:
    """Offset of `quote` in the transcript, or -1.

    Whole quote first, then its leading _QUOTE_MATCH_PREFIX characters.
    """
    collapsed = _collapse(quote)
    offset = normalised.find(collapsed)
    if offset >= 0:
        return offset
    if len(collapsed) <= _QUOTE_MATCH_PREFIX:
        return -1
    return normalised.find(collapsed[:_QUOTE_MATCH_PREFIX])


def _collapse(text: str) -> str:
    """Whitespace-insensitive form for quote matching.

    A model reproduces the words reliably and the line breaks less so; failing
    a quote over a wrapped newline would make the check measure formatting
    rather than fabrication. Character positions are preserved 1:1 so an offset
    in the collapsed text still maps back to the original.
    """
    return "".join(" " if c.isspace() else c for c in text)


def _expand(_normalised: str, offset: int) -> int:
    """_collapse preserves length, so offsets need no translation."""
    return offset


#: Fields the SERVER sets and the model may never supply. `evidence_verified`
#: is the whole point of the check — a model that could assert it would be
#: grading its own homework — and the other two are derived, not reported.
_SERVER_OWNED = frozenset({"segment", "evidence_verified", "anchor_line_no"})


def _only(raw: dict[str, Any], cls: type) -> dict[str, Any]:
    """Keep just the fields `cls` declares.

    The extraction schema changes as we learn what a unit should be, and a model
    occasionally answers with an older shape. Dropping the extras costs one
    field; letting them through raises TypeError and costs the segment.
    """
    allowed = {f.name for f in fields(cls)} - _SERVER_OWNED
    return {k: v for k, v in raw.items() if k in allowed}


def _stamp(bundle: UnitBundle, ref: SegmentRef) -> UnitBundle:
    """Give every unit its own ref, carrying its position within this segment.

    Each kind is enumerated separately: `position` only has to be unique among
    units of the same kind in the same segment, because the document id already
    carries both the kind and the segment.
    """
    for units in (bundle.qa, bundle.code_change, bundle.decision, bundle.file_ref):
        for position, unit in enumerate(units):
            unit.segment = replace(ref, position=position)
    return bundle


async def extract_units_from_session(
    session_id: str,
    events: list[dict[str, Any]],
    cwd: str | None = None,
    agent: str = "claude_code",
) -> UnitBundle:
    """Mine every part of the session, not just its tail."""
    segments, capped = _segment_session(events)

    # With the originals of every segment in hand, the compaction summaries are
    # a second telling of conversation we are already reading — drop them from
    # the model's input. When the cap DID drop early segments, they are the only
    # remaining account of those, so they stay.
    drop_summaries = not capped
    if capped:
        log.warning(
            "claude_code_extraction.segments_capped",
            extra={"session_id": session_id, "kept": _MAX_SEGMENTS},
        )

    bundle = UnitBundle()
    semaphore = asyncio.Semaphore(_SEGMENT_CONCURRENCY)

    async def _run(index: int, segment: list[dict[str, Any]], boundary: str) -> UnitBundle:
        start, end = _line_bounds(segment)
        ref = SegmentRef(
            index=index + 1,
            total=len(segments),
            boundary=boundary,
            start_line_no=start,
            end_line_no=end,
        )
        async with semaphore:
            result = await _extract_one(
                session_id=session_id,
                events=segment,
                cwd=cwd,
                agent=agent,
                part=(ref.index, ref.total),
                drop_summaries=drop_summaries,
            )
        return _stamp(
            result,
            replace(ref, objective=result.objective, motivation=result.motivation),
        )

    results = await asyncio.gather(
        *(_run(i, evs, why) for i, (evs, why) in enumerate(segments)),
        return_exceptions=True,
    )
    for index, result in enumerate(results):
        if isinstance(result, BaseException):
            # One segment failing must not cost the whole session. A partial
            # bundle is strictly better than none, and the loss is visible.
            log.warning(
                "claude_code_extraction.segment_failed",
                extra={
                    "session_id": session_id,
                    "part": index + 1,
                    "error": str(result),
                },
            )
            continue
        bundle.qa.extend(result.qa)
        bundle.code_change.extend(result.code_change)
        bundle.decision.extend(result.decision)
        bundle.file_ref.extend(result.file_ref)
    return bundle


async def _extract_one(
    *,
    session_id: str,
    events: list[dict[str, Any]],
    cwd: str | None,
    agent: str,
    part: tuple[int, int],
    drop_summaries: bool,
) -> UnitBundle:
    settings = get_settings()

    if drop_summaries:
        events = [e for e in events if not _is_compact_summary(e)]
    transcript, spans = render_indexed(events)
    if not transcript.strip():
        return UnitBundle()

    index, total = part
    where = f" (part {index} of {total})" if total > 1 else ""
    user_content = (
        f"Extract structured units from this session{where}.\n"
        f"session_id: {session_id}\n"
        f"cwd: {cwd or 'unknown'}\n\n"
        f"{transcript}"
    )

    # Gateway model ids are proxy-owned aliases and must pass through verbatim.
    # Force the OpenAI wire shape so LiteLLM calls the proxy's /chat/completions
    # endpoint instead of deriving a provider-native path from the model name.
    # Direct calls retain the legacy Anthropic prefix and native transport.
    gateway_enabled = gateway_url() is not None
    if gateway_enabled:
        model = settings.claude_code_extraction_model
        transport_kwargs: dict[str, Any] = {"custom_llm_provider": "openai"}
    else:
        model = _ensure_provider_prefix(
            settings.claude_code_extraction_model, default_provider="anthropic"
        )
        transport_kwargs = {}

    try:
        args, _resp = await forced_tool_call(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_TEMPLATE.format(
                        agent=_AGENT_LABELS.get(agent, "coding agent")
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            tool_name=_TOOL_NAME,
            tool_description=_TOOL_DESCRIPTION,
            tool_schema=_TOOL_PARAMETERS,
            max_tokens=8000,
            **transport_kwargs,
        )
    except ToolCallParseError:
        # Model declined to call the tool — return an empty bundle, the
        # same fallback the previous direct-SDK path used when no
        # `tool_use` block came back.
        return UnitBundle()

    # Unknown keys are dropped rather than raising: a model that answers with a
    # field the schema no longer has must not cost the whole segment its units.
    goal = args.get("goal") if isinstance(args.get("goal"), dict) else {}
    return _ground_units(UnitBundle(
        objective=str(goal.get("objective") or "").strip(),
        motivation=str(goal.get("motivation") or "").strip(),
        qa=[QA(**_only(x, QA)) for x in args.get("qa", [])],
        code_change=[CodeChange(**_only(x, CodeChange)) for x in args.get("code_change", [])],
        decision=[Decision(**_only(x, Decision)) for x in args.get("decision", [])],
        file_ref=[FileRef(**_only(x, FileRef)) for x in args.get("file_ref", [])],
    ), transcript, spans)


def _ensure_provider_prefix(model: str, *, default_provider: str) -> str:
    """Return a LiteLLM-compatible model id with a provider prefix.

    If the configured id already looks provider-prefixed
    (e.g. ``anthropic/...``, ``openai/...``, ``gemini/...``), pass it
    through unchanged. Otherwise prepend ``default_provider``. This lets
    callers configure either a bare model id (legacy shape) OR a fully
    qualified LiteLLM id without code edits.
    """
    if "/" in model:
        return model
    return f"{default_provider}/{model}"
