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
from dataclasses import dataclass, field
from typing import Any

from engine.shared.config import get_settings
from engine.shared.llm import gateway_url
from engine.shared.llm_tools import ToolCallParseError, forced_tool_call
from engine.shared.logging import get_logger
from engine.shared.transcript_render import _events_to_text


@dataclass(slots=True)
class QA:
    prompt: str
    outcome: str
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CodeChange:
    file: str
    before: str
    after: str
    intent: str


@dataclass(slots=True)
class Decision:
    question: str
    options_considered: list[str]
    chosen: str
    rationale: str


@dataclass(slots=True)
class FileRef:
    files: list[str]
    context: str


@dataclass(slots=True)
class UnitBundle:
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
        "qa": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string"},
                    "outcome": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["prompt", "outcome"],
            },
        },
        "code_change": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "file": {"type": "string"},
                    "before": {"type": "string"},
                    "after": {"type": "string"},
                    "intent": {"type": "string"},
                },
                "required": ["file", "before", "after", "intent"],
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
                    "rationale": {"type": "string"},
                },
                "required": ["question", "options_considered", "chosen", "rationale"],
            },
        },
        "file_ref": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "files": {"type": "array", "items": {"type": "string"}},
                    "context": {"type": "string"},
                },
                "required": ["files", "context"],
            },
        },
    },
    "required": ["qa", "code_change", "decision", "file_ref"],
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
    "For code_change, `before` and `after` must be quoted from the transcript. "
    "If the transcript does not contain the actual text on either side, leave "
    "them empty and describe the change in `intent` instead — never reconstruct "
    "code you were not shown."
)

_AGENT_LABELS = {"claude_code": "Claude Code", "codex": "Codex"}


log = get_logger(__name__)


def _is_compact_boundary(event: dict[str, Any]) -> bool:
    raw = event.get("raw")
    raw = raw if isinstance(raw, dict) else event
    return raw.get("type") == "system" and raw.get("subtype") == "compact_boundary"


def _is_compact_summary(event: dict[str, Any]) -> bool:
    raw = event.get("raw")
    raw = raw if isinstance(raw, dict) else event
    return bool(raw.get("isCompactSummary"))


def _split_on_compaction(events: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """One segment per stretch between compactions."""
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for event in events:
        if _is_compact_boundary(event) and current:
            segments.append(current)
            current = []
        current.append(event)
    if current:
        segments.append(current)
    return segments or [[]]


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


def _segment_session(events: list[dict[str, Any]]) -> tuple[list[list[dict[str, Any]]], bool]:
    """(segments, capped) — every part of the session, oldest first."""
    segments: list[list[dict[str, Any]]] = []
    for chunk in _split_on_compaction(events):
        segments.extend(_split_to_budget(chunk))
    segments = [s for s in segments if s]
    capped = len(segments) > _MAX_SEGMENTS
    if capped:
        segments = segments[-_MAX_SEGMENTS:]
    return segments, capped


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

    async def _run(index: int, segment: list[dict[str, Any]]) -> UnitBundle:
        async with semaphore:
            return await _extract_one(
                session_id=session_id,
                events=segment,
                cwd=cwd,
                agent=agent,
                part=(index + 1, len(segments)),
                drop_summaries=drop_summaries,
            )

    results = await asyncio.gather(
        *(_run(i, seg) for i, seg in enumerate(segments)), return_exceptions=True
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
    transcript = _events_to_text(events)
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

    return UnitBundle(
        qa=[QA(**x) for x in args.get("qa", [])],
        code_change=[CodeChange(**x) for x in args.get("code_change", [])],
        decision=[Decision(**x) for x in args.get("decision", [])],
        file_ref=[FileRef(**x) for x in args.get("file_ref", [])],
    )


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
