"""Pydantic models for wiki synthesis I/O.

Triage in/out and synthesis in/out shapes used both as type hints inside the
cron loop AND as Anthropic tool-use `input_schema` for forced structured
output. Keeping the schema in Python (rather than hand-rolled JSON) means
the prompt + the parser + the type checker share one source of truth.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

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


class TriageTarget(BaseModel):
    """One wiki page this event should land on, per the triage decision."""

    wiki_type: Literal["service_card", "decision", "feature", "runbook"]
    slug: str = Field(min_length=1, max_length=64)
    action: Literal["create", "update"]


class TriageVerdict(BaseModel):
    """Per-event verdict produced by Haiku."""

    important: bool
    score: float = Field(ge=0.0, le=10.0)
    targets: list[TriageTarget] = Field(default_factory=list)
    reason: str | None = Field(
        default=None,
        description="One sentence explaining the decision for the audit log.",
    )


class TriageOutput(BaseModel):
    """Top-level Haiku response: queue_id -> verdict."""

    verdicts: dict[str, TriageVerdict]


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


class SynthesisInput(BaseModel):
    """One cluster of triaged events that all target the same wiki page."""

    wiki_type: Literal["service_card", "decision", "feature", "runbook"]
    slug: str
    action: Literal["create", "update"]
    current_title: str | None = None
    current_body: str | None = None
    current_frontmatter: dict[str, object] = Field(default_factory=dict)
    current_summary: str | None = None
    events: list[TriageInput] = Field(min_length=1)


class SynthesisOutput(BaseModel):
    """Sonnet's tool-use return shape per page."""

    title: str = Field(min_length=1, max_length=200)
    body_markdown: str
    summary: str = Field(
        min_length=1,
        max_length=240,
        description="One-sentence summary used by the wiki.index page.",
    )
    frontmatter: dict[str, object] = Field(default_factory=dict)
    commit_message: str = Field(
        min_length=1,
        max_length=240,
        description="One-line audit message describing what changed and why.",
    )
