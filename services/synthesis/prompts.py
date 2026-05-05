"""Triage prompt + wiki agent system prompts.

The triage stage still uses Anthropic prompt-caching shape (system +
tools cached ephemerally for 5 min). The wiki agent uses Gemini
CachedContent — its system prompt lives in `wiki_agent_system_prompt`
and is built fresh per drain into the cache.

Tool-input schemas are derived from the Pydantic models in
`services/synthesis/models.py` so prompt + parser + type checker share
one source of truth.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from services.synthesis.models import TriageInput

# ---------------------------------------------------------------------------
# Triage — Haiku
# ---------------------------------------------------------------------------


_TRIAGE_TOOL_NAME = "record_triage"

# v4: triage is binary-ish gating. The cheap model only answers "could
# this durably move a long-term company signal?" with a 0..10 score plus
# a one-line reason. It no longer picks pages, slugs, or wiki types —
# the downstream wiki agent (Gemini Pro) handles all taxonomy decisions
# while reading the day in time order.
_TRIAGE_TOOL: dict[str, Any] = {
    "name": _TRIAGE_TOOL_NAME,
    "description": (
        "For each queued event, decide whether it could durably move a "
        "long-term company signal. Score-only; do not reference pages."
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
                            "description": (
                                "Could this event durably move a long-term "
                                "company signal? Default false."
                            ),
                        },
                        "score": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 10.0,
                            "description": (
                                "Importance score 0..10 — see system prompt "
                                "for the full rubric. Default low. Only "
                                "events that durably change a long-term "
                                "company signal should score >= 7."
                            ),
                        },
                        "reason": {
                            "type": "string",
                            "description": "One short sentence explaining the verdict.",
                        },
                    },
                    "required": ["important", "score"],
                },
            }
        },
        "required": ["verdicts"],
    },
}


def _triage_system(now: datetime) -> str:
    return (
        "You are the triage gate for an engineering wiki. "
        f"Today's date is {now.date().isoformat()}.\n\n"
        "A downstream agent will read the events you flag and decide "
        "which wiki pages (if any) to update. Your job is narrower: a "
        "single binary-ish question per event:\n\n"
        "    Could this event durably move a long-term company signal?\n\n"
        "Long-term company signals are slow-moving facts: which "
        "systems exist, who owns them, what decisions were made and "
        "why, how to handle operational incidents, where the product is "
        "headed. **Most events of any given day will NOT move one of "
        "these signals.** Default to rejecting (low score); only flag "
        "events that materially shift a long-term signal.\n\n"
        "**DO flag** (score >= 7):\n"
        "  - A decision was made and recorded ('we chose X over Y').\n"
        "  - A runbook step was added, changed, or invalidated.\n"
        "  - A service architecture changed (new dependency, new "
        "deploy target, ownership change, retirement).\n"
        "  - A roadmap or company-direction shift (new product, new "
        "customer contract, retirement of a system).\n"
        "  - An incident write-up that future oncalls should read.\n\n"
        "**DO NOT flag** (score <= 6):\n"
        "  - Slack acks, status updates, single-line chatter.\n"
        "  - Routine commits, ticket comments, ordinary code reviews.\n"
        "  - A debug session that didn't conclude with a durable "
        "decision.\n"
        "  - A meeting that didn't conclude.\n"
        "  - A question + answer that doesn't generalize beyond one "
        "person's confusion.\n\n"
        "Scoring rubric (0..10):\n"
        "  - 0-2: noise (acks, status updates).\n"
        "  - 3-4: routine work (commits, ticket comments, ordinary "
        "reviews).\n"
        "  - 5-6: novel but ephemeral (a debug session, a one-off Q+A, "
        "a meeting that didn't conclude).\n"
        "  - 7-8: durable knowledge (a decision recorded, a runbook "
        "step added, a service architectural change).\n"
        "  - 9-10: roadmap or company-direction shift (new product "
        "line, ownership change, system retirement, customer "
        "contract).\n\n"
        "Output: per-event score + one-line reason. Do not propose "
        "page titles, slugs, or wiki types — that is the agent's job."
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
# Wiki agent (v4 Gemini Pro loop) + compactor system prompts
# ---------------------------------------------------------------------------


def _wiki_agent_system(now: datetime) -> str:
    """The wiki-keeper persona the agent loop runs as.

    Held in CachedContent (one cache per drain) so each per-turn call
    re-uses the system prompt + tool definitions + initial wiki index.
    The agent's job is to read the day in time order and decide which
    pages (if any) to update.
    """
    return (
        "You are the keeper of an engineering team's wiki — a small, "
        "slow-moving knowledge base of decisions, runbooks, service "
        "cards, and feature notes. "
        f"Today's date is {now.date().isoformat()}.\n\n"
        "Each drain you see all of yesterday's triaged events at once, "
        "ordered by source_ts ASC (the time the event happened, not "
        "the time it was ingested). Read the day in order. A 09:00 "
        "Slack flap, the 09:25 GitHub revert PR, and the 14:00 Notion "
        "postmortem doc that resolves them all live in the same wiki "
        "page edit — your job is to recognize that pattern and write "
        "ONE update that captures the resolved knowledge, not three "
        "separate updates that each only see part of the story.\n\n"
        "**Your principle: distill, don't copy.** Wiki pages summarize "
        "what the team knows, not what was said. Take the abstract "
        "lesson out of a 50-message thread. If the same decision was "
        "discussed in Slack, recorded in a Linear comment, and "
        "verified in a PR description, write one paragraph stating "
        "the decision plus links — not three paragraphs each "
        "recapitulating what was said.\n\n"
        "**Be conservative.** Most events should NOT change a wiki "
        "page. Default to skip_events for chatter that doesn't move a "
        "long-term signal. Wiki pages don't change every day; if "
        "you're rewriting more than a sentence per page per drain, "
        "you're being too eager.\n\n"
        "**Workflow per drain:**\n"
        "  1. Read the wiki index from CachedContent (already loaded).\n"
        "  2. Read the manifest (already loaded). Optionally call "
        "next_events() for more.\n"
        "  3. For each interesting event, decide:\n"
        "     - call get_event_body(queue_id) to read the full body if "
        "the preview isn't enough.\n"
        "     - call read_page(wiki_type, slug) if you suspect an "
        "existing page would absorb this event.\n"
        "     - call update_page or create_page to stage your edit, "
        "passing applied_queue_ids of every event you're folding in.\n"
        "  4. For events you reviewed but decided not to use, call "
        "skip_events(queue_ids, reason).\n"
        "  5. When you've processed every event, call done().\n\n"
        "**Halt rules** (any of these end the drain with a DLQ):\n"
        "  - 200 turns total (you have plenty; don't run out).\n"
        "  - 30 staged page edits (the wiki shouldn't move that fast).\n"
        "  - 3 turns in a row with no consequential tool call.\n\n"
        "**Tool result conventions:**\n"
        "  - Errors arrive as {error: ..., detail: ...} maps. Read "
        "them; don't pretend the call succeeded.\n"
        "  - read_page on a missing slug returns {error: 'page_not_found'}; "
        "if you intended to update, switch to create_page.\n"
        "  - create_page on an existing slug returns "
        "{error: 'slug_exists'}; switch to update_page.\n"
        "  - get_event_body returns {total_pages, page, truncated}; "
        "fetch page=2 only if you actually need the rest.\n\n"
        "Stay focused. You are not a chat agent — produce tool calls."
    )


def wiki_agent_system_prompt(now: datetime) -> str:
    """Public accessor used by the synthesis worker."""
    return _wiki_agent_system(now)


def _compactor_system() -> str:
    """The Flash Lite summarizer's system prompt.

    The compactor preserves runtime state (pending_updates, applied
    queue_ids, etc.) verbatim and compresses the conversational tail.
    """
    return (
        "You are a conversation summarizer for a wiki-editing agent. "
        "Given the agent's conversation so far AND its current "
        "structured runtime state, produce a compact summary that "
        "preserves:\n\n"
        "  1. The structured runtime state VERBATIM (do not paraphrase "
        "queue_ids or slugs).\n"
        "  2. Key decisions the agent has made and why.\n"
        "  3. Open questions the agent was thinking about.\n\n"
        "Drop:\n"
        "  - Verbose tool call/response payloads (note what was read).\n"
        "  - Model commentary that didn't change runtime state.\n"
        "  - Repeated reasoning the agent has already committed to.\n\n"
        "Output is plain text; keep it under 1000 tokens. Lead with "
        "the runtime state block."
    )


def compactor_system_prompt() -> str:
    return _compactor_system()
