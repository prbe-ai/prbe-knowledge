"""Wiki agent tool palette — 8 tools.

Defines two parallel things:

  1. Gemini-shaped tool schema dicts the harness passes into
     `tools=[...]` for function-call mode.
  2. Pydantic input models the runtime uses for per-call validation.
     A failure in validation raises `ToolValidationError`, which the
     harness routes to a typed tool_result error so the model can
     re-decide on the next turn.

Tools (READ):
  - next_events(count: int = 200)           : fetch next manifest window
  - list_wiki_pages()                       : same as initial cache payload
  - read_page(wiki_type, slug)              : fetch live or staged page
  - get_event_body(queue_id, page=1)        : fetch event body, paginated

Tools (WRITE — staged, committed at done()):
  - update_page(wiki_type, slug, body, summary, commit_message,
                applied_queue_ids)
  - create_page(wiki_type, slug, title, body, summary, frontmatter,
                commit_message, applied_queue_ids)

Tools (BOOKKEEPING):
  - skip_events(queue_ids, reason)          : mark agent-reviewed,
                                              not page-changing
  - done()                                  : atomic commit + end of drain
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Tool schemas (Gemini function-call format)
# ---------------------------------------------------------------------------

# Reused enum literal for wiki types. Matches DocType.WIKI_*.
_WIKI_TYPES = ["service_card", "decision", "feature", "runbook"]


# Note on schema shape:
#   Gemini's function declarations accept JSON-schema-ish objects
#   under `parameters`. Mapping fields:
#     type=object, properties={...}, required=[...]
#   The shape mirrors Anthropic's input_schema so the dispatcher can
#   share validation models.

NEXT_EVENTS_TOOL: dict[str, Any] = {
    "name": "next_events",
    "description": (
        "Fetch the next batch of triaged events to read. Excludes events "
        "already applied or skipped in this drain. Ordered by source_ts ASC."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "description": "Number of events to return; default 200.",
            }
        },
    },
}

LIST_WIKI_PAGES_TOOL: dict[str, Any] = {
    "name": "list_wiki_pages",
    "description": (
        "Return the current wiki index (titles + slugs + summaries). "
        "Same payload as the initial CachedContent; use only if the "
        "cache view is suspect."
    ),
    "parameters": {"type": "object", "properties": {}},
}

READ_PAGE_TOOL: dict[str, Any] = {
    "name": "read_page",
    "description": (
        "Read the full body of a wiki page. Returns the staged version "
        "if this drain has staged an update or create; else the live "
        "DB version. Page not found -> typed PageNotFoundError result."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wiki_type": {"type": "string", "enum": _WIKI_TYPES},
            "slug": {"type": "string"},
        },
        "required": ["wiki_type", "slug"],
    },
}

GET_EVENT_BODY_TOOL: dict[str, Any] = {
    "name": "get_event_body",
    "description": (
        "Fetch the body of a triaged event. Pages are 6000 chars; "
        "total_pages == 1 means the whole body fits. Page numbers "
        "start at 1."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "queue_id": {"type": "integer"},
            "page": {"type": "integer", "minimum": 1, "default": 1},
        },
        "required": ["queue_id"],
    },
}

UPDATE_PAGE_TOOL: dict[str, Any] = {
    "name": "update_page",
    "description": (
        "Stage an update to an existing wiki page. Last-write-wins per "
        "slug; applied_queue_ids accumulate (union) across re-stages. "
        "Committed atomically at done()."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wiki_type": {"type": "string", "enum": _WIKI_TYPES},
            "slug": {"type": "string"},
            "body_markdown": {"type": "string"},
            "summary": {"type": "string"},
            "commit_message": {"type": "string"},
            "applied_queue_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": [
            "wiki_type",
            "slug",
            "body_markdown",
            "summary",
            "commit_message",
            "applied_queue_ids",
        ],
    },
}

CREATE_PAGE_TOOL: dict[str, Any] = {
    "name": "create_page",
    "description": (
        "Stage creation of a new wiki page. If the slug already exists "
        "on disk, returns an error result; the agent must call "
        "update_page instead. Committed atomically at done()."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "wiki_type": {"type": "string", "enum": _WIKI_TYPES},
            "slug": {"type": "string"},
            "title": {"type": "string"},
            "body_markdown": {"type": "string"},
            "summary": {"type": "string"},
            "frontmatter": {"type": "object"},
            "commit_message": {"type": "string"},
            "applied_queue_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
        },
        "required": [
            "wiki_type",
            "slug",
            "title",
            "body_markdown",
            "summary",
            "commit_message",
            "applied_queue_ids",
        ],
    },
}

SKIP_EVENTS_TOOL: dict[str, Any] = {
    "name": "skip_events",
    "description": (
        "Mark events as agent-reviewed-but-not-page-changing. "
        "Conflicts with applied_queue_ids -> skip wins (more conservative)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "queue_ids": {
                "type": "array",
                "items": {"type": "integer"},
            },
            "reason": {"type": "string"},
        },
        "required": ["queue_ids", "reason"],
    },
}

DONE_TOOL: dict[str, Any] = {
    "name": "done",
    "description": (
        "Commit all staged updates + creates atomically and end the "
        "drain. After done() the agent has no further turns."
    ),
    "parameters": {"type": "object", "properties": {}},
}


ALL_TOOLS: list[dict[str, Any]] = [
    NEXT_EVENTS_TOOL,
    LIST_WIKI_PAGES_TOOL,
    READ_PAGE_TOOL,
    GET_EVENT_BODY_TOOL,
    UPDATE_PAGE_TOOL,
    CREATE_PAGE_TOOL,
    SKIP_EVENTS_TOOL,
    DONE_TOOL,
]


# ---------------------------------------------------------------------------
# Pydantic input validators
# ---------------------------------------------------------------------------


WikiType = Literal["service_card", "decision", "feature", "runbook"]


class NextEventsArgs(BaseModel):
    count: int = Field(default=200, ge=1, le=500)


class ListWikiPagesArgs(BaseModel):
    pass


class ReadPageArgs(BaseModel):
    wiki_type: WikiType
    slug: str = Field(min_length=1, max_length=64)


class GetEventBodyArgs(BaseModel):
    queue_id: int
    page: int = Field(default=1, ge=1)


class UpdatePageArgs(BaseModel):
    wiki_type: WikiType
    slug: str = Field(min_length=1, max_length=64)
    body_markdown: str
    summary: str = Field(min_length=1, max_length=240)
    commit_message: str = Field(min_length=1, max_length=240)
    applied_queue_ids: list[int] = Field(default_factory=list)


class CreatePageArgs(BaseModel):
    wiki_type: WikiType
    slug: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    body_markdown: str
    summary: str = Field(min_length=1, max_length=240)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    commit_message: str = Field(min_length=1, max_length=240)
    applied_queue_ids: list[int] = Field(default_factory=list)


class SkipEventsArgs(BaseModel):
    queue_ids: list[int] = Field(min_length=1)
    reason: str = Field(min_length=1, max_length=240)


class DoneArgs(BaseModel):
    pass


# Map tool names to their Pydantic validators. Used by the runtime
# dispatch path: validate args before calling the actual handler.
TOOL_VALIDATORS: dict[str, type[BaseModel]] = {
    "next_events": NextEventsArgs,
    "list_wiki_pages": ListWikiPagesArgs,
    "read_page": ReadPageArgs,
    "get_event_body": GetEventBodyArgs,
    "update_page": UpdatePageArgs,
    "create_page": CreatePageArgs,
    "skip_events": SkipEventsArgs,
    "done": DoneArgs,
}


__all__ = [
    "ALL_TOOLS",
    "TOOL_VALIDATORS",
    "CreatePageArgs",
    "DoneArgs",
    "GetEventBodyArgs",
    "ListWikiPagesArgs",
    "NextEventsArgs",
    "ReadPageArgs",
    "SkipEventsArgs",
    "UpdatePageArgs",
]
