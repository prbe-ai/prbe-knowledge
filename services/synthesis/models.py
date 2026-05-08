"""Pydantic models for wiki synthesis I/O.

Triage in/out shapes used both as type hints inside the worker AND as
Anthropic tool-use `input_schema` for forced structured output. Keeping
the schema in Python (rather than hand-rolled JSON) means the prompt +
the parser + the type checker share one source of truth.

v4 also defines the wiki agent's data shapes — `RouterEvent` (the
manifest entry the agent reads), `WikiIndexEntry` (one page in the
agent's CachedContent index), `PageUpdate` / `PageCreate` (staged
write intents), `AgentRunResult` (one drain's audit summary). These
do NOT correspond to v3's removed router stage; the name is the
agent-facing shape, not a router output.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# WikiType — the page-kind discriminator.
#
# Free-form string. The wiki agent picks page-type slugs as it sees fit
# (typically `repo`, `runbook`, `person`, but the LLM is free to invent
# new ones for a given customer's corpus). Validation is purely a
# URL-safety regex enforced at the ingestion route boundary
# (services/ingestion/handlers/wiki.is_valid_wiki_type) — the model
# layer treats it as opaque text.
# ---------------------------------------------------------------------------
WikiType = str


# ---------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------


class TriageInput(BaseModel):
    """One row from `wiki_synthesis_queue` joined to its `documents` row.

    The `body` field carries the FULL document body — not chunks, not the
    body_preview. Triage decides whether a document is wiki-worthy by
    reading the whole thing.
    """

    queue_id: int
    doc_id: str
    doc_type: str
    source_system: str
    title: str | None
    author_id: str | None
    body: str
    body_token_count: int


class TriageVerdict(BaseModel):
    """Per-event verdict produced by Flash Lite (or Haiku fallback).

    v4: score-only. The downstream wiki agent decides which page (if
    any) the event lands on after reading the day in time order; triage
    no longer picks (wiki_type, slug).

    `reason` is hard-capped at 240 chars in the schema so the model
    sees the constraint and doesn't write paragraph-length reasons
    that overflow the per-batch output budget. A 50-event batch with
    240-char reasons (~60 Anthropic tokens each + ~30 envelope) lands
    around 4500 output tokens — well under the 8000 max_tokens cap,
    leaving real headroom even when reasons run long.
    """

    important: bool
    score: float = Field(ge=0.0, le=10.0)
    reason: str | None = Field(
        default=None,
        max_length=240,
        description=(
            "One short sentence (<= 240 chars) explaining the decision for the audit log. Be terse."
        ),
    )

    @field_validator("reason", mode="before")
    @classmethod
    def _truncate_overlong_reason(cls, v: object) -> object:
        # Haiku occasionally writes longer reasons than the schema asks
        # for. Production hot bug (probe-founders, 2026-05-08): one
        # verdict's reason was 300+ chars, Pydantic raised
        # string_too_long on the batch-wide TriageOutput parse, the
        # provider wrapped it as TriageParseError, the split-retry
        # wrapper's overflow regexes didn't match, the batch was
        # marked triage_error on every row, and the worker's
        # "no verdicts this iteration" branch DLQ'd every pending row
        # for the customer.
        #
        # Truncate to the schema cap BEFORE the length validator runs;
        # `mode="before"` is the explicit Pydantic v2 spelling for
        # pre-constraint validators. The schema constraint still ships
        # to Haiku in the tool input_schema (nudging it toward terse
        # reasons), but enforcement no longer poisons sibling verdicts
        # when Haiku ignores the hint.
        if isinstance(v, str) and len(v) > 240:
            return v[:240]
        return v


class TriageOutput(BaseModel):
    """Top-level Haiku response: queue_id -> verdict."""

    verdicts: dict[str, TriageVerdict]


# ---------------------------------------------------------------------------
# Wiki agent (v4 Gemini Pro loop)
# ---------------------------------------------------------------------------


class RouterEvent(BaseModel):
    """One manifest entry the wiki agent reads via `next_events()`.

    The name "RouterEvent" is the agent-facing shape, not a router stage
    (v4 has no router; the agent does all routing itself). It carries
    just enough metadata for the agent to decide whether to read the
    event body in full via `get_event_body()`. Body is omitted from
    the manifest to keep CachedContent size bounded — the agent
    expands what it needs.
    """

    queue_id: int
    doc_id: str
    doc_type: str
    source_system: str
    title: str | None = None
    author_id: str | None = None
    source_ts: datetime
    body_preview: str = Field(
        default="",
        description=(
            "First few hundred chars of the body. Lets the agent skip "
            "noisy events without paying a get_event_body() call."
        ),
    )
    body_token_count: int = 0


class WikiIndexEntry(BaseModel):
    """One wiki page in the agent's CachedContent index.

    Built by `persistence.fetch_wiki_index(customer_id)` at drain start
    and embedded in the agent's CachedContent. Lets the agent see
    every COMPILED_WIKI page's title + slug + summary without paying
    a `read_page` call up front.
    """

    wiki_type: WikiType
    slug: str
    title: str
    summary: str | None = None
    last_updated: datetime
    version: int


class PageUpdate(BaseModel):
    """Staged update intent — written to runtime state, persisted at done()."""

    wiki_type: WikiType
    slug: str
    body_markdown: str
    summary: str = Field(min_length=1, max_length=240)
    commit_message: str = Field(min_length=1, max_length=240)
    applied_queue_ids: list[int] = Field(default_factory=list)


class PageCreate(BaseModel):
    """Staged create intent — same shape as PageUpdate plus title + frontmatter."""

    wiki_type: WikiType
    slug: str
    title: str = Field(min_length=1, max_length=200)
    body_markdown: str
    summary: str = Field(min_length=1, max_length=240)
    frontmatter: dict[str, Any] = Field(default_factory=dict)
    commit_message: str = Field(min_length=1, max_length=240)
    applied_queue_ids: list[int] = Field(default_factory=list)


class AgentRunResult(BaseModel):
    """One drain's audit summary returned to the synthesis worker."""

    agent_run_id: str
    pages_updated: int
    pages_created: int
    events_applied: int
    events_skipped: int
    halt_reason: str | None = None
    turns: int
    compaction_count: int = 0
    cache_hit_rate: float | None = None
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    gemini_call_count: int = 0
