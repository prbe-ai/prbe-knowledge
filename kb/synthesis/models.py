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
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# WikiType — the page-kind discriminator. A CLOSED set.
#
# It used to be `WikiType = str`, free-form, with the agent told it "may
# invent new types if the corpus calls for it". Two things went wrong with
# that, and only one of them is obvious.
#
# The obvious one: a wiki whose page kinds are invented per drain has no
# stable shape to navigate. `repo` and `repository` and `codebase` are three
# sections of the same wiki, and no reader or renderer can tell.
#
# The one that actually bit: the type is a PATH SEGMENT
# (`/api/wiki/pages/{wiki_type}/{slug}`) and a doc_id component
# (`wiki:{wiki_type}:{slug}`), so an invented type is a permanent identity.
# Nothing renames it afterwards -- there is no rename route -- so a typo the
# agent made once at 04:00 is a page kind forever.
#
# THE MEMBERS ARE THE PRODUCT DECISION, not a sample. `index` is included
# because it is a real stored page (the generated overview), and excluding it
# would mean the one type the system writes itself is not in the type it
# writes with. It stays reserved against human WRITES separately -- see
# `_validate_wiki_type` in kb/wiki_routes.py.
#
# Verified against production before closing: every wiki page in every tenant
# used one of `repo`, `project`, `runbook`, `person`, `index` (37 pages,
# 2026-08-12). Nothing is orphaned by this list. Adding a member later is a
# one-line change here that reaches the tool schema, the ingestion gate and
# the agent prompt at once -- which is the point of having one constant.
# ---------------------------------------------------------------------------
class WikiType(StrEnum):
    #: The auto-generated overview. Written only by the synthesis cron.
    INDEX = "index"
    #: A codebase.
    REPO = "repo"
    #: A stream of work with a goal and an end.
    PROJECT = "project"
    #: How to do a recurring operational thing.
    RUNBOOK = "runbook"
    #: A teammate: what they own, what they are working on.
    PERSON = "person"
    #: A research experiment -- hypothesis, setup, what it showed.
    EXPERIMENT = "experiment"
    #: A corpus: where it came from, its shape, its known problems.
    DATASET = "dataset"
    #: A trained or hosted model and what is known about its behaviour.
    MODEL = "model"
    #: A decision that was made, why, and what it ruled out.
    DECISION = "decision"


#: The types the AGENT may write. `index` is excluded: it is generated from
#: the other pages at the end of a drain, and an agent writing it directly
#: would be overwritten by that step within the same run.
AGENT_WIKI_TYPES: tuple[str, ...] = tuple(
    t.value for t in WikiType if t is not WikiType.INDEX
)


class AgentWikiType(StrEnum):
    """`WikiType` minus `index` — the type the agent's TOOL ARGS validate on.

    Separate from `WikiType` because the tool SCHEMA and the tool VALIDATOR
    have to agree, and they did not: the schema advertised `AGENT_WIKI_TYPES`
    (no `index`) while the Pydantic args used `WikiType` (with it). A model
    that emitted `index` would have passed validation against the one rule the
    schema said it could not break, and the write would then be silently
    overwritten by the index regeneration at the end of the same drain.

    Derived from the same members rather than restated, so adding a kind is
    still one edit in one place.
    """

    REPO = WikiType.REPO.value
    PROJECT = WikiType.PROJECT.value
    RUNBOOK = WikiType.RUNBOOK.value
    PERSON = WikiType.PERSON.value
    EXPERIMENT = WikiType.EXPERIMENT.value
    DATASET = WikiType.DATASET.value
    MODEL = WikiType.MODEL.value
    DECISION = WikiType.DECISION.value


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
        # for. Production hot bug (acme, 2026-05-08): one
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
