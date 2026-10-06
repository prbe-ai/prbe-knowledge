"""A captured session's events -> its ATIF trajectory: the call the ingest
pass, the scripts and the tests make.

INPUT is the merged event list `fetch_supplementary` returns: `probe-events/1`,
the shape every capture sanitizer emits (Claude Code's event shape, with the
other harnesses' own fields under `_codex_extras` / `_pi_extras` /
`_kimi_extras`). OUTPUT is a `BuildResult`: the document (format, render
provenance and what it never carries: fold.py), how many events could not be
mapped, and the first few of them for the caller to log.

TWO BUILDERS, ONE DOCUMENT
--------------------------
`reference`  build_reference.py: the stateful builder every stored trajectory
             was written by, frozen. The default.
`fold`       fold(map(fragment, events)): the same document from per-event
             fragments (fragment.py, the shape clients will upload) folded by
             fold.py, which treats them as untrusted.

`SESSION_ATIF_BUILDER` picks one for the ingest pass and the backfill; any
other value means `reference`, so a typo in a deploy cannot take ingestion
down. tests/test_atif_fragment_fold.py holds the two equal on every fixture,
and `scripts/atif_sessions.py replay --compare-builders` on stored sessions.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from engine.ingest.atif import build_reference
from engine.ingest.atif.fold import (
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
    SCHEMA_VERSION,
    BuildResult,
    fold,
)
from engine.ingest.atif.fragment import USER_SHELL, fragment

__all__ = [
    "FLAG_COMPACT_BOUNDARY",
    "FLAG_COMPACT_SUMMARY",
    "FLAG_USER_TURN",
    "PART_CALL",
    "PART_EVENT",
    "PART_RESULT",
    "PART_STOP",
    "PART_SYSTEM",
    "PART_TEXT",
    "PART_THINKING",
    "PART_USER",
    "RENDER_FORMAT",
    "SCHEMA_VERSION",
    "USER_SHELL",
    "BuildResult",
    "Builder",
    "build_and_render",
    "build_trajectory",
    "builder_for",
]


class Builder(StrEnum):
    #: The frozen stateful builder (build_reference.py).
    REFERENCE = "reference"
    #: fold(map(fragment, events)) (fragment.py + fold.py).
    FOLD = "fold"


def builder_for(value: Any) -> Builder:
    """The builder a setting names; anything unknown means `reference`."""
    try:
        return Builder(str(value).strip().lower())
    except ValueError:
        return Builder.REFERENCE


def build_trajectory(
    events: list[dict[str, Any]],
    *,
    session_id: str,
    agent_name: str,
    builder: Builder | str = Builder.REFERENCE,
) -> BuildResult:
    if builder_for(builder) is Builder.FOLD:
        # Non-object events are skipped, as the reference skips them.
        fragments = [fragment(ev) for ev in events if isinstance(ev, dict)]
        return fold(fragments, session_id=session_id, agent_name=agent_name)
    built = build_reference.build_trajectory(events, session_id=session_id, agent_name=agent_name)
    return BuildResult(built.trajectory, built.unparsed, list(built.unparsed_events))


def build_and_render(
    events: list[dict[str, Any]],
    session_id: str,
    agent_name: str,
    render: bool,
    builder: Builder | str = Builder.REFERENCE,
) -> tuple[BuildResult, list[Any] | None, str | None]:
    """`build_trajectory`, and when `render` its Lines too: one plain-data call
    the ingest pass can run in `cpu_pool`'s processes, so `builder` is read
    from the settings by the caller, not here. Returns (build, lines or None,
    the rendering error's class name or None)."""
    from engine.ingest.atif.lines import lines_from_trajectory

    built = build_trajectory(events, session_id=session_id, agent_name=agent_name, builder=builder)
    if not render:
        return built, None, None
    try:
        return built, lines_from_trajectory(built.trajectory), None
    except Exception as exc:
        return built, None, type(exc).__name__
