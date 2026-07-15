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

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from services.synthesis.models import WikiType

# Slug normalization: collapse `/`, whitespace, and dots into `-`, drop any
# remaining chars outside `[a-z0-9_-]`, collapse hyphen runs, trim leading/
# trailing hyphens/underscores. Mirrors the markdown extractor's allowed
# slug class in `services/synthesis/wiki_links.py` so a freshly-created
# page's slug is always linkable from another page's body.
_SLUG_SEP_RE = re.compile(r"[\s/.]+")
_SLUG_DISALLOWED_RE = re.compile(r"[^a-z0-9_-]+")
_SLUG_DASH_RUN_RE = re.compile(r"-{2,}")

# Typed title prefix shape (e.g. `Repo:`, `Service:`, `Person:`) used by
# the wiki agent when titling pages and by the index renderer when
# referencing them via `[[Type: Name]]`. Used to defensively strip a
# leading prefix from a summary when no authoritative title is in scope.
_TYPED_PREFIX_RE = re.compile(r"^\s*[A-Z][A-Za-z]+\s*:\s*[^:\n]{1,80}:\s+")


def _normalize_slug(value: str) -> str:
    """Coerce an LLM-supplied slug into the canonical `[a-z0-9_-]+` shape.

    Examples:
        `prbe-ai/kb`  -> `prbe-ai-kb`
        `Repo Name`   -> `repo-name`
        `Some.Thing`  -> `some-thing`
    """
    s = value.lower()
    s = _SLUG_SEP_RE.sub("-", s)
    s = _SLUG_DISALLOWED_RE.sub("", s)
    s = _SLUG_DASH_RUN_RE.sub("-", s)
    return s.strip("-_")


def _strip_title_prefix(summary: str, *, title: str | None = None) -> str:
    """Drop a leading `<title>:` (or generic `<Type>: <Name>:`) prefix.

    The wiki agent occasionally writes summaries like
    `Repo: prbe-ai/kb: Markdown knowledge base ...` which render in the
    index as duplicated chrome (`[[Repo: prbe-ai/kb]] - Repo: prbe-ai/kb:
    Markdown ...`). The summary field is meant to be the bare blurb after
    the title; strip the prefix at the validator boundary so persistence
    is the single source of truth.

    When `title` is supplied, only strip the exact match (case-insensitive,
    whitespace-tolerant). Otherwise apply the looser typed-prefix
    heuristic — caller carries the risk of stripping a legitimate
    leading bareword like `URL: ...`.
    """
    stripped = summary.lstrip()
    if title:
        prefix = title.strip().rstrip(":")
        # Match the literal title followed by `:` and optional whitespace.
        pattern = re.compile(
            rf"^{re.escape(prefix)}\s*:\s*",
            re.IGNORECASE,
        )
        rewritten = pattern.sub("", stripped, count=1)
        if rewritten != stripped:
            return rewritten.lstrip()
    rewritten = _TYPED_PREFIX_RE.sub("", stripped, count=1)
    return rewritten.lstrip() if rewritten != stripped else summary

# ---------------------------------------------------------------------------
# Tool schemas (Gemini function-call format)
# ---------------------------------------------------------------------------

# wiki_type is free-form — the agent picks slugs as it sees fit. The
# schema description below names common ones as guidance but the field
# itself accepts any string. URL-safety regex is enforced when the page
# is persisted, not at the tool boundary.
_WIKI_TYPE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "description": (
        "The page-kind discriminator. You typically pick from `repo`, "
        "`runbook`, `person`, `company`, `customer`, `project`, `event`, "
        "but you may invent new types if the corpus calls for it. Keep "
        "the slug lowercase, alphanumeric + underscore, <= 32 chars."
    ),
}


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
            "wiki_type": _WIKI_TYPE_SCHEMA,
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
            "wiki_type": _WIKI_TYPE_SCHEMA,
            "slug": {"type": "string"},
            "body_markdown": {"type": "string"},
            "summary": {
                "type": "string",
                "description": (
                    "One-sentence stable overview of what this page IS "
                    "(the repo's / runbook's / person's enduring "
                    "purpose). Shown as the wiki-index blurb; pass the "
                    "existing summary verbatim unless the page's "
                    "fundamental purpose has changed. Never rewrite it "
                    "to describe the latest change — the index would "
                    "read like a changelog."
                ),
            },
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
            "wiki_type": _WIKI_TYPE_SCHEMA,
            "slug": {"type": "string"},
            "title": {"type": "string"},
            "body_markdown": {"type": "string"},
            "summary": {
                "type": "string",
                "description": (
                    "One-sentence stable overview of what this page IS "
                    "(the repo's / runbook's / person's enduring "
                    "purpose). Shown as the wiki-index blurb; should "
                    "read the same in 6 months as today. Not what "
                    "triggered creating it."
                ),
            },
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

    @model_validator(mode="after")
    def _strip_summary_prefix(self) -> UpdatePageArgs:
        # No title in scope on update — apply the generic typed-prefix
        # heuristic so the index renderer doesn't double-print the
        # `[[Type: Name]] - Type: Name: ...` chrome.
        cleaned = _strip_title_prefix(self.summary)
        if cleaned != self.summary and cleaned:
            object.__setattr__(self, "summary", cleaned)
        return self


class CreatePageArgs(BaseModel):
    wiki_type: WikiType
    slug: str = Field(min_length=1, max_length=64)
    title: str = Field(min_length=1, max_length=200)
    body_markdown: str
    summary: str = Field(min_length=1, max_length=240)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    commit_message: str = Field(min_length=1, max_length=240)
    applied_queue_ids: list[int] = Field(default_factory=list)

    @field_validator("slug", mode="before")
    @classmethod
    def _normalize_slug_value(cls, value: object) -> object:
        # Normalize at create time only — updates need to use the slug
        # the page already lives under, which may pre-date this rule.
        if isinstance(value, str):
            return _normalize_slug(value) or value
        return value

    @model_validator(mode="after")
    def _strip_summary_prefix(self) -> CreatePageArgs:
        cleaned = _strip_title_prefix(self.summary, title=self.title)
        if cleaned != self.summary and cleaned:
            object.__setattr__(self, "summary", cleaned)
        return self


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
