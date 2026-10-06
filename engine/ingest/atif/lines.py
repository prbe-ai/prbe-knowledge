"""An ATIF trajectory built by `engine.ingest.atif.build` -> the `Line`s the
search index and the unit extractor read.

The rendered text of each source event is rebuilt from the structured fields,
using the provenance in `extra.probe` / `extra.probe_parts` (see build.py) and
the formatting helpers the event renderer itself uses
(engine/shared/transcript_render.py), so the two cannot word anything
differently. Every ingest pass compares the result with the event path before
trusting it (kb/handlers/claude_code.py).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from engine.ingest.atif.build import (
    FLAG_COMPACT_BOUNDARY,
    FLAG_COMPACT_SUMMARY,
    FLAG_USER_TURN,
    PART_CALL,
    PART_EVENT,
    PART_RESULT,
    PART_STOP,
    PART_SYSTEM,
    PART_TEXT,
    PART_THINKING,
    PART_USER,
    RENDER_FORMAT,
)
from engine.shared.transcript_render import (
    Line,
    _render_tool_use,
    format_assistant_text,
    format_other_event,
    format_stop,
    format_system,
    format_thinking,
    format_tool_result,
    format_user_text,
)


class UnrenderableTrajectory(ValueError):
    """The trajectory carries no provenance this module knows how to read."""


def lines_from_trajectory(trajectory: dict[str, Any]) -> list[Line]:
    probe = (trajectory.get("extra") or {}).get("probe") or {}
    if probe.get("format") != RENDER_FORMAT:
        raise UnrenderableTrajectory(f"render format {probe.get('format')!r}")

    parts_by_line: dict[int, list[tuple[int, dict[str, Any], list[Any]]]] = defaultdict(list)
    for step in trajectory.get("steps") or []:
        for part in (step.get("extra") or {}).get("probe_parts") or []:
            parts_by_line[part[0]].append((part[1], step, part))

    lines: list[Line] = []
    for index, (line_no, flags) in enumerate(probe.get("lines") or []):
        rendered: list[str] = []
        stop: Any = None
        for _seq, step, part in sorted(parts_by_line.get(index, ()), key=lambda p: p[0]):
            kind = part[2]
            if kind == PART_STOP:
                stop = part[3]
                continue
            text = _render_part(kind, step, part)
            if text:
                rendered.append(text)
        if stop is not None and rendered:
            rendered.append(format_stop(stop))
        lines.append(
            Line(
                line_no=line_no,
                text="\n".join(rendered),
                user_turn=bool(flags & FLAG_USER_TURN),
                compact_boundary=bool(flags & FLAG_COMPACT_BOUNDARY),
                compact_summary=bool(flags & FLAG_COMPACT_SUMMARY),
            )
        )
    return lines


def _render_part(kind: str, step: dict[str, Any], part: list[Any]) -> str:
    message = step.get("message")
    extra = step.get("extra") or {}
    if kind == PART_USER:
        index, speaker = part[3], part[4]
        text = message if index is None else message[index]["text"]
        return format_user_text(speaker, text)
    if kind == PART_TEXT:
        return format_assistant_text(message[part[3]]["text"])
    if kind == PART_THINKING:
        start, end = part[3], part[4]
        return format_thinking(step["reasoning_content"][start:end])
    if kind == PART_CALL:
        call = step["tool_calls"][part[3]]
        call_extra = call.get("extra") or {}
        return _render_tool_use(
            {
                "name": call["function_name"],
                "summary": call_extra.get("summary"),
                "stats": call_extra.get("stats"),
            }
        )
    if kind == PART_RESULT:
        result = step["observation"]["results"][part[3]]
        result_extra = result.get("extra") or {}
        tool_use_id = result_extra.get("tool_use_id", result.get("source_call_id"))
        return format_tool_result(
            tool_use_id, result_extra.get("is_error"), result_extra.get("result_bytes")
        )
    if kind == PART_SYSTEM:
        return format_system(extra.get("subtype"), message)
    if kind == PART_EVENT:
        return format_other_event(extra.get("event_type"), message)
    raise UnrenderableTrajectory(f"part kind {kind!r}")
