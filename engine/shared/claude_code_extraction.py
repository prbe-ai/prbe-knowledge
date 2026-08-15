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

from dataclasses import dataclass, field
from typing import Any

import orjson

from engine.shared.config import get_settings
from engine.shared.llm import gateway_url
from engine.shared.llm_tools import ToolCallParseError, forced_tool_call


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


# Hard cap to prevent context-length-exceeded errors. Real production sessions
# rarely exceed a few hundred events. Anything beyond this is suspicious; we
# keep the most recent _MAX_EVENTS so the extractor still sees the conclusion.
_MAX_EVENTS = 2000

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


async def extract_units_from_session(
    session_id: str,
    events: list[dict[str, Any]],
    cwd: str | None = None,
    agent: str = "claude_code",
) -> UnitBundle:
    if len(events) > _MAX_EVENTS:
        events = events[-_MAX_EVENTS:]
    settings = get_settings()

    user_payload = {
        "session_id": session_id,
        "cwd": cwd,
        "events": events,
    }
    user_content = (
        "Extract structured units from this session.\n\n"
        + orjson.dumps(user_payload).decode("utf-8")
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
