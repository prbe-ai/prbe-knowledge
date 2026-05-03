"""Triage + synthesis prompt builders.

Both stages call Anthropic via `messages.create(tools=[...], tool_choice=...)`
to force structured output. The shape returned by `build_*_prompt` matches
the kwargs `AsyncAnthropic.messages.create` expects, modulo `messages` /
`max_tokens` which the caller provides per call.

Caching: both prompts mark their system block + tool schema with
`cache_control: {"type": "ephemeral"}` (5-min TTL). Triage in particular
fires a Haiku call per ~50 events; reusing the system + tool schema across
those calls drops effective input cost by ~10x. Same pattern as
`services/retrieval/router.py:402-453`.

Tool-input schemas are derived from the Pydantic models in
`services/synthesis/models.py` so prompt + parser + type checker share one
source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from services.synthesis.models import SynthesisInput, TriageInput

# ---------------------------------------------------------------------------
# Triage — Haiku
# ---------------------------------------------------------------------------


_TRIAGE_TOOL_NAME = "record_triage"

_TRIAGE_TOOL: dict[str, Any] = {
    "name": _TRIAGE_TOOL_NAME,
    "description": (
        "For each queued event, decide whether it should change the wiki "
        "and which page(s) it belongs on."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "object",
                "description": (
                    "Map keyed by queue_id (string). One entry per event in "
                    "the input. Every input MUST appear here exactly once."
                ),
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "important": {
                            "type": "boolean",
                            "description": "Should this event change the wiki at all?",
                        },
                        "score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 10.0,
                            "description": (
                                "Importance score 0..10. Trivial chatter <= 3, "
                                "routine but useful 4-6, decision-bearing or "
                                "incident-level >= 7."
                            ),
                        },
                        "targets": {
                            "type": "array",
                            "description": (
                                "Wiki pages this event should land on. Empty "
                                "for unimportant events. Multiple entries are "
                                "allowed if the event genuinely affects more "
                                "than one page."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "wiki_type": {
                                        "type": "string",
                                        "enum": [
                                            "service_card",
                                            "decision",
                                            "feature",
                                            "runbook",
                                        ],
                                    },
                                    "slug": {
                                        "type": "string",
                                        "pattern": "^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$",
                                    },
                                    "action": {
                                        "type": "string",
                                        "enum": ["create", "update"],
                                    },
                                },
                                "required": ["wiki_type", "slug", "action"],
                            },
                        },
                        "reason": {
                            "type": "string",
                            "description": "One short sentence explaining the verdict.",
                        },
                    },
                    "required": ["important", "score", "targets"],
                },
            }
        },
        "required": ["verdicts"],
    },
}


def _triage_system(now: datetime) -> str:
    return (
        "You are the triage editor for a team-wide engineering wiki. "
        f"Today's date is {now.date().isoformat()}.\n\n"
        "You receive a batch of recently-ingested documents (Slack threads, "
        "GitHub PRs, Linear tickets, Notion pages, Sentry incidents, "
        "claude_code sessions, ...). For each one, decide:\n"
        "  1. Is it important enough to change the wiki at all?\n"
        "  2. If yes, which existing wiki page should it update, "
        "or what new page should be created?\n\n"
        "Wiki types:\n"
        "  - service_card: stable facts about a system or service "
        "(owner, runtime, deploy target, SLOs).\n"
        "  - decision: 'we decided to do X because Y' write-ups, "
        "ADRs, RFCs.\n"
        "  - feature: how a customer-visible capability is built and "
        "intended to behave.\n"
        "  - runbook: how to handle an operational situation "
        "(incident, oncall, recurring task).\n\n"
        "Scoring guide (0..10): trivial chatter <= 3, routine but useful "
        "4-6, decision-bearing or incident-level >= 7. Be conservative "
        "by default — false positives waste expensive synthesis calls.\n\n"
        "Slugs are lowercase a-z0-9 with single hyphens. If you propose "
        "a new page, choose a slug that matches the topic, not the "
        "current event (e.g. 'slack-backfill-stuck' not 'incident-2026-05-02')."
    )


def build_triage_prompt(
    events: list[TriageInput],
    *,
    now: datetime,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Construct the kwargs for `AsyncAnthropic.messages.create`.

    Caller adds `model=HAIKU_MODEL` and supplies `messages` from this dict.
    The system block + tool schema are marked for ephemeral prompt caching.
    """
    user = _format_triage_user_message(events)
    return {
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": _triage_system(now),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [_TRIAGE_TOOL],
        "tool_choice": {"type": "tool", "name": _TRIAGE_TOOL_NAME},
        "messages": [{"role": "user", "content": user}],
    }


def _format_triage_user_message(events: list[TriageInput]) -> str:
    parts: list[str] = ["Triage the following events. Return one verdict per queue_id."]
    for event in events:
        title = event.title or "(no title)"
        author = event.author_id or "(unknown)"
        parts.append(
            "\n---\n"
            f"queue_id: {event.queue_id}\n"
            f"doc_id: {event.doc_id}\n"
            f"doc_type: {event.doc_type}\n"
            f"source_system: {event.source_system}\n"
            f"title: {title}\n"
            f"author: {author}\n"
            "body:\n"
            f"<body>\n{event.body}\n</body>"
        )
    return "\n".join(parts)


def triage_tool_name() -> str:
    return _TRIAGE_TOOL_NAME


# ---------------------------------------------------------------------------
# Synthesize — Sonnet
# ---------------------------------------------------------------------------


_SYNTH_TOOL_NAME = "render_wiki_page"

_SYNTH_TOOL: dict[str, Any] = {
    "name": _SYNTH_TOOL_NAME,
    "description": (
        "Produce the new (or first) version of one wiki page, incorporating "
        "every event in the cluster. Preserve any existing human-authored "
        "content unless an event explicitly contradicts it. Output is "
        "plain GitHub-flavored markdown."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "minLength": 1,
                "maxLength": 200,
            },
            "body_markdown": {
                "type": "string",
                "description": (
                    "Full page body. Use [[Person: name]], [[Service: name]], "
                    "[[Repo: name]], [[Ticket: id]], [[Feature: name]], "
                    "[[Decision: slug]] for cross-references — they get "
                    "resolved into the knowledge graph."
                ),
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
                "description": (
                    "One-sentence summary used by the wiki.index table of "
                    "contents. Should answer 'what is this page about?'."
                ),
            },
            "frontmatter": {
                "type": "object",
                "description": "Optional frontmatter merged onto the page.",
            },
            "commit_message": {
                "type": "string",
                "minLength": 1,
                "maxLength": 240,
                "description": (
                    "Audit-log line describing what changed in this version "
                    "and why, in the style of a git commit subject."
                ),
            },
        },
        "required": ["title", "body_markdown", "summary", "commit_message"],
    },
}


def _synthesis_system(now: datetime) -> str:
    return (
        "You are the editor of a team-wide engineering wiki. "
        f"Today's date is {now.date().isoformat()}.\n\n"
        "You are given:\n"
        "  - the current version of a wiki page (or empty if creating).\n"
        "  - a cluster of recently-ingested raw documents that triage "
        "decided affect this page.\n\n"
        "Produce the new full body of the page in markdown. Rules:\n"
        "  - Keep any human-authored prose intact unless an event clearly "
        "contradicts it; in that case, note the contradiction inline rather "
        "than silently rewriting.\n"
        "  - Cite the source documents inline using [[Ticket: PRB-9]], "
        "[[Service: prbe-knowledge]], [[Person: mahit]], [[Decision: <slug>]] "
        "etc. Plain [[wiki page]] links are fine for refs to other wiki pages.\n"
        "  - Lead with the answer. Section headers organize the page; do not "
        "number them. Include code blocks where useful.\n"
        "  - Provide a one-sentence summary suitable for a table of contents.\n"
        "  - Provide a one-line commit_message describing what changed in "
        "this version and why."
    )


def build_synthesis_prompt(
    cluster: SynthesisInput,
    *,
    now: datetime,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    return {
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": _synthesis_system(now),
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "tools": [_SYNTH_TOOL],
        "tool_choice": {"type": "tool", "name": _SYNTH_TOOL_NAME},
        "messages": [{"role": "user", "content": _format_synthesis_user(cluster)}],
    }


def _format_synthesis_user(cluster: SynthesisInput) -> str:
    parts: list[str] = []
    parts.append(f"Wiki type: {cluster.wiki_type}")
    parts.append(f"Slug: {cluster.slug}")
    parts.append(f"Action: {cluster.action}")
    if cluster.action == "update":
        parts.append("\n--- CURRENT PAGE ---")
        parts.append(f"title: {cluster.current_title or '(none)'}")
        if cluster.current_summary:
            parts.append(f"summary: {cluster.current_summary}")
        parts.append("body:")
        parts.append("<current_body>")
        parts.append(cluster.current_body or "")
        parts.append("</current_body>")
    parts.append("\n--- EVENTS THAT AFFECT THIS PAGE ---")
    for event in cluster.events:
        parts.append("\n---")
        parts.append(f"doc_id: {event.doc_id}")
        parts.append(f"doc_type: {event.doc_type}")
        parts.append(f"source_system: {event.source_system}")
        if event.title:
            parts.append(f"title: {event.title}")
        if event.author_id:
            parts.append(f"author: {event.author_id}")
        parts.append("body:")
        parts.append("<body>")
        parts.append(event.body)
        parts.append("</body>")
    return "\n".join(parts)


def synthesis_tool_name() -> str:
    return _SYNTH_TOOL_NAME
