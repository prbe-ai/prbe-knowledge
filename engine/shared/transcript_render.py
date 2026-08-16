"""Rendering a merged transcript into the prose that gets read.

Lifted out of the claude_code connector when a SECOND consumer appeared: the
unit extractor. It used to send the model raw event JSON, which measured 3.7x
to 9.3x larger than this rendering of the same events -- 476k to 929k tokens
for a 2000-event window against a 200k-token context. Whatever the connector
shows the chunker is also the right thing to show the extractor, and there is
no reason for two representations of one conversation.

Shared by both agents: CodexConnector translates Codex rollouts into this same
shape before rendering.
"""
from __future__ import annotations

import re
from typing import Any


def _events_to_text(events: list[dict[str, Any]]) -> str:
    """Render merged Claude Code events into a chunkable text body.

    Output is human-readable prose — what the chunker + embedder consume,
    so it has to look like the conversation. NOT JSON dumps.

    Each turn becomes a block separated by blank lines:

        USER: how does auth work?

        ASSISTANT (thinking): the flow uses JWT in cookie X.
        ASSISTANT: we use JWT.

        TOOL_USE: Bash — git status
        TOOL_RESULT (toolu_xxx): ok

        USER: refactor it.

    Ordering matches the line_no-sorted merged stream so a chunker walking
    sequentially sees the session in transcript order.

    Events the plugin sanitizer already drops (file-history-snapshot,
    last-prompt, ai-title, permission-mode, stop_hook_summary, turn_duration)
    don't reach here. Anything else without a renderer is silently skipped
    rather than dumped as JSON — JSON noise in the embedded text was the
    original problem this rewrite solves.
    """
    return render_indexed(events)[0]


def render_indexed(
    events: list[dict[str, Any]],
) -> tuple[str, list[tuple[int, int, int]]]:
    """The same prose, plus a map from text offset back to transcript line.

    Exists so a quote can be traced to the event it came from. That is what
    turns an evidence string into a VERIFIED one — a quote that matches nothing
    was not in the transcript — and the matched span yields the unit's anchor
    line, which is the only way to order two units inside one segment.

    Returns (text, spans) where each span is (start, end, line_no) over the
    returned text. ANSI is stripped per-block so offsets stay valid; stripping
    it afterwards would shift every span left of the escape.
    """
    blocks: list[str] = []
    spans: list[tuple[int, int, int]] = []
    cursor = 0
    for ev in events:
        raw = ev.get("raw") if isinstance(ev, dict) else None
        if not isinstance(raw, dict):
            continue
        rendered = _ANSI_RE.sub("", _render_event(raw))
        if not rendered:
            continue
        if blocks:
            cursor += 2  # the "\n\n" join between blocks
        line_no = ev.get("line_no") if isinstance(ev, dict) else None
        spans.append(
            (cursor, cursor + len(rendered), line_no if isinstance(line_no, int) else -1)
        )
        cursor += len(rendered)
        blocks.append(rendered)
    return "\n\n".join(blocks), spans


def line_for_offset(spans: list[tuple[int, int, int]], offset: int) -> int | None:
    """The transcript line whose rendered text contains `offset`."""
    for start, end, line_no in spans:
        if start <= offset < end:
            return line_no if line_no >= 0 else None
    return None


def _render_event(raw: dict[str, Any]) -> str:
    ev_type = raw.get("type")
    if ev_type == "user":
        return _render_user(raw)
    if ev_type == "assistant":
        return _render_assistant(raw)
    if ev_type == "system":
        sub = raw.get("subtype") or ""
        content = raw.get("content")
        if isinstance(content, str) and content:
            return f"SYSTEM ({sub}): {content}" if sub else f"SYSTEM: {content}"
        # System event with no string content — note the subtype but skip
        # dumping the rest. Keeps the conversation flow readable.
        return f"SYSTEM ({sub})" if sub else ""

    # Top-level string `content` for unknown event types — preserve
    # forward-compat without leaking raw JSON into embeddings.
    content = raw.get("content")
    if isinstance(content, str) and content:
        label = (ev_type or "EVENT").upper()
        return f"{label}: {content}"
    return ""


# Terminal colour codes reach the transcript whenever a user pastes coloured
# output into a prompt, and they were being embedded verbatim: 1,200 sequences
# and 14,194 characters in one measured session. They tokenise into noise no
# query will ever match, and they render as mojibake anywhere the document is
# shown. Stripped once over the assembled body rather than per-renderer, so no
# future renderer can forget to do it.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


#: Machine-generated envelopes that arrive on the USER channel. They are not
#: speech — the harness injects them — but they render as `USER:` and are then
#: read as the researcher's own words by everything downstream, including the
#: extractor filling qa.prompt and any attempt to mine per-person habits.
#:
#: Measured over twelve real sessions (774 user turns carrying text): 23.6%
#: contained a task-notification, 6.2% slash-command plumbing, 5.2% the
#: local-command caveat, 4.8% an entire injected skill body.
_HARNESS_BLOCKS = [
    re.compile(r"<task-notification>.*?</task-notification>", re.S),
    re.compile(r"<system-reminder>.*?</system-reminder>", re.S),
    re.compile(r"<local-command-caveat>.*?</local-command-caveat>", re.S),
    re.compile(r"<local-command-(?:stdout|stderr)>.*?</local-command-(?:stdout|stderr)>", re.S),
    re.compile(r"</?command-(?:name|message|args)>", re.S),
    re.compile(r"<codex_internal_context>.*?</codex_internal_context>", re.S),
]

#: A whole injected document rather than an envelope around one: when a skill
#: fires, its body arrives as a user turn. There is no user text to preserve, so
#: the turn goes rather than being trimmed.
_INJECTED_DOCUMENT = re.compile(r"^\s*Base directory for this skill:", re.M)


def _strip_harness(text: str) -> str:
    """User text with the harness's own injections removed.

    Trimmed rather than dropped whole, because a turn routinely carries a real
    request AND an appended reminder; discarding the turn would lose the half
    that matters.
    """
    if _INJECTED_DOCUMENT.search(text):
        return ""
    for pattern in _HARNESS_BLOCKS:
        text = pattern.sub(" ", text)
    return text.strip()


def _render_tool_use(block: dict[str, Any]) -> str:
    """One line for a tool call: name, what it acted on, what it changed.

    The `stats` and `args` fields are the sanitizer's COMPACTION of inputs it
    would otherwise have deleted outright — counts, never content. Rendering
    them here is what makes them retrievable, and it is what lets the
    extraction model read the shape of a change instead of recalling it: an
    `Edit` used to reach the extractor as a bare file path, and
    `code_change.before` / `after` could only ever be the model's recollection
    of a diff nobody sent it.
    """
    name = block.get("name") or "tool"
    summary = block.get("summary") or ""
    line = f"TOOL_USE: {name} — {summary}" if summary else f"TOOL_USE: {name}"

    # VALIDATED, not trusted. The sanitizer that produces `stats` runs on the
    # researcher's machine, and the only thing between it and this renderer is
    # a device token — an old tap, a patched one, or a compromised device can
    # post whatever it likes. Rendering it verbatim would put arbitrary client
    # strings into the indexed body AND into the prompt sent to the model,
    # which is exactly what this module exists to prevent.
    stats = block.get("stats")
    if isinstance(stats, dict):
        added = stats.get("added_lines")
        removed = stats.get("removed_lines")
        if isinstance(added, int) or isinstance(removed, int):
            line += f" [+{_count(added)}/-{_count(removed)} lines]"
        if stats.get("replace_all") is True:
            line += " [replace_all]"
    return line


#: Upper bound on a rendered count. A line count cannot legitimately exceed
#: this, and clamping keeps a hostile or corrupted value from becoming a
#: multi-megabyte token in the index.
_MAX_RENDERED_COUNT = 10_000_000


def _count(value: Any) -> int:
    """A non-negative integer, or 0. Never a string, float, or bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, min(value, _MAX_RENDERED_COUNT))


def _render_user(raw: dict[str, Any]) -> str:
    msg = raw.get("message")
    if not isinstance(msg, dict):
        return ""

    # A compaction summary is NOT something the user said. When a session runs
    # out of context, Claude Code writes its own 18-25 KB summary of everything
    # so far and injects it as a `user` message flagged isCompactSummary. One
    # measured session carried four of them, 83 KB in total.
    #
    # Rendering that as "USER:" tells the index the researcher personally wrote
    # a structured account of their own intent, which is exactly the text an
    # extractor reaches for when filling qa.prompt — so the model's summary of
    # the human comes back out attributed to the human. Label it instead of
    # dropping it: it is the single densest statement of intent in a long
    # session, and for the early parts of a session that fall outside the
    # extractor's 2,000-event window it is the ONLY surviving record.
    speaker = "COMPACTION SUMMARY" if raw.get("isCompactSummary") else "USER"

    content = msg.get("content")
    if isinstance(content, str) and content:
        cleaned = _strip_harness(content)
        return f"{speaker}: {cleaned}" if cleaned else ""
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            text = _strip_harness(b.get("text") or "")
            if text:
                parts.append(f"{speaker}: {text}")
        elif bt == "tool_result":
            # SUCCESSFUL results are not rendered at all. Measured over a real
            # session, `TOOL_RESULT (toolu_x): ok` accounted for 1,660 lines and
            # 79,839 characters — 4.4% of the indexed document and the single
            # largest class of text in it that carries no information. In Codex
            # it is worse: those lines are most of a document whose human and
            # model content is ~3% of its lines, which is why a Codex session
            # matches search queries on tool plumbing instead of on what it
            # discussed. The tool_use line above already records that the call
            # happened; a bare "ok" only restates it.
            #
            # FAILURES still render — a failed call is a real event in the
            # session's story, and there are two orders of magnitude fewer of
            # them (53 of 1,660 here).
            if not b.get("is_error"):
                continue
            tool_id = b.get("tool_use_id") or ""
            label = f"TOOL_RESULT ({tool_id})" if tool_id else "TOOL_RESULT"
            size = b.get("result_bytes")
            if isinstance(size, int) and size > 0:
                parts.append(f"{label}: error ({size} bytes)")
            else:
                parts.append(f"{label}: error")
    return "\n".join(parts)


def _render_assistant(raw: dict[str, Any]) -> str:
    msg = raw.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str) and content:
        return f"ASSISTANT: {content}"
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            text = b.get("text") or ""
            if text:
                parts.append(f"ASSISTANT: {text}")
        elif bt == "thinking":
            text = b.get("thinking") or ""
            if text and text.strip():
                parts.append(f"ASSISTANT (thinking): {text}")
        elif bt == "tool_use":
            parts.append(_render_tool_use(b))

    # Note non-default stop_reasons (max_tokens, refusal, …); end_turn is
    # the boring case and noting it would just clutter every assistant turn.
    stop_reason = msg.get("stop_reason")
    if stop_reason and stop_reason not in ("end_turn", "tool_use") and parts:
        parts.append(f"[stop: {stop_reason}]")
    return "\n".join(parts)
