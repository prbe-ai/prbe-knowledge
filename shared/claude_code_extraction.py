"""Per-session knowledge-unit extraction using Sonnet tool-use as structured output.

A single tool `emit_units` with one parameter shape per unit type. The model
is forced to call this tool; tool input is the result. We use tool-use rather
than `response_format` JSON mode because tool-use enforces the schema more
reliably across longer context windows.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import orjson
from anthropic import AsyncAnthropic

from shared.config import get_settings


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


_TOOL_SCHEMA: dict[str, Any] = {
    "name": "emit_units",
    "description": "Emit structured knowledge units extracted from a Claude Code session.",
    "input_schema": {
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
    },
}


# Hard cap to prevent context-length-exceeded errors. Real production sessions
# rarely exceed a few hundred events. Anything beyond this is suspicious; we
# keep the most recent _MAX_EVENTS so the extractor still sees the conclusion.
_MAX_EVENTS = 2000

_SYSTEM = (
    "You extract structured knowledge from one Claude Code session. "
    "Return only the emit_units tool call. Be conservative — emit a unit only "
    "when the session clearly demonstrates the corresponding kind of insight. "
    "Empty arrays are valid."
)


async def extract_units_from_session(
    session_id: str,
    events: list[dict[str, Any]],
    cwd: str | None = None,
) -> UnitBundle:
    if len(events) > _MAX_EVENTS:
        events = events[-_MAX_EVENTS:]
    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())

    user_payload = {
        "session_id": session_id,
        "cwd": cwd,
        "events": events,
    }
    msg = await client.messages.create(
        model=settings.claude_code_extraction_model,
        max_tokens=8000,
        system=_SYSTEM,
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "tool", "name": "emit_units"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract structured units from this session.\n\n"
                    + orjson.dumps(user_payload).decode("utf-8")
                ),
            }
        ],
    )

    for block in msg.content:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == "emit_units":
            data = block.input
            return UnitBundle(
                qa=[QA(**x) for x in data.get("qa", [])],
                code_change=[CodeChange(**x) for x in data.get("code_change", [])],
                decision=[Decision(**x) for x in data.get("decision", [])],
                file_ref=[FileRef(**x) for x in data.get("file_ref", [])],
            )
    return UnitBundle()  # model declined to use the tool — return empty bundle
