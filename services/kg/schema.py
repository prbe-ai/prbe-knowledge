"""Pydantic envelope for a debugging-KG class.

A class is conceptually `{ frontmatter, body }`:

- **Frontmatter** is structured JSON (signature, related, context_sources,
  evidence) and is what the agent reads / writes via JSON API endpoints. It
  is stored as JSONB end-to-end — there is no YAML round-trip step.
- **Body** is opaque markdown prose (the playbook). Its shape is not
  validated here; it is rendered to humans, parsed by the kg-check link
  resolver, and otherwise treated as a string by the schema.

Phase 1 is debugging-only: the only concrete `type` is `"bug-class"`. The
cross-domain Layer-1 / Layer-2 split (TaskClass vs BugClass) is deferred
to a future phase — the spec calls this out explicitly (§4).

Validation deliberately stays narrow:

- ``Frontmatter`` and ``BugClass`` use ``extra="forbid"`` so a typoed field
  name fails loudly instead of silently dropping data.
- ``Signature.must_match`` rejects all-blank rule lists (a signature with
  no real rule is a bug, not "match anything").
- ``Frontmatter.id`` is a slug (`^[a-z][a-z0-9-]{2,63}$`) — same shape
  used by the spec's example IDs and by the URL path the API will route
  on.
- ``ContextSource.priority`` is a ``Literal[1, 2, 3]`` (spec §4.1).
- ``ContextSource.params`` stays an opaque dict in Phase 1 — no
  per-tool discriminator yet.

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md
§4.1 (Layer 1 generic schema), §4.2 (debugging instantiation), §5.2 (class
shape with full JSON example).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Signature(BaseModel):
    """Rules + embedding seed used to match an incoming task to this class.

    `must_match` carries the structured predicates (e.g. ``"status_code == 401"``);
    the strings are opaque to this layer — interpretation is the classifier's
    job. `embedding_seed` is free-text concepts (e.g. ``"jwt refresh expired
    clock-skew"``) used to anchor semantic match in pgvector.
    """

    must_match: list[str]
    embedding_seed: str

    @field_validator("must_match")
    @classmethod
    def _must_match_has_real_rule(cls, value: list[str]) -> list[str]:
        # A signature with zero real rules effectively means "match anything",
        # which is never the intent — fail loudly.
        if not any(rule.strip() for rule in value):
            raise ValueError(
                "must_match requires at least one non-blank rule "
                "(all entries were empty after strip)"
            )
        return value

    @field_validator("embedding_seed")
    @classmethod
    def _embedding_seed_is_substantive(cls, value: str) -> str:
        if len(value.strip()) < 3:
            raise ValueError("embedding_seed must be at least 3 characters after strip")
        return value


class Related(BaseModel):
    """Outgoing edges to other classes by class_id.

    Spec §4.1 gives Layer 1 the generic edges (`analogous_to`, `overlaps_with`);
    spec §4.2 layers debugging-specific ones on top (`often_confused_with`,
    `regressed_by`). All four are flat `list[str]` of class_ids; resolution
    happens at read time via JSONB containment queries (no `kg_edges` table —
    see spec §5.1).
    """

    analogous_to: list[str] = Field(default_factory=list)
    overlaps_with: list[str] = Field(default_factory=list)
    often_confused_with: list[str] = Field(default_factory=list)
    regressed_by: list[str] = Field(default_factory=list)


class ContextSource(BaseModel):
    """A typed pointer the agent should consult when this class fires.

    `tool` names a callable surface (e.g. ``"github_diff"``, ``"file_read"``,
    ``"linear_search"``); `params` is the per-tool argument bag, kept as an
    opaque dict in Phase 1 (no per-tool discriminator yet — see spec §5.4).
    `priority` is the p1/p2/p3 rung from spec §4.1.
    """

    priority: Literal[1, 2, 3]
    name: str = Field(min_length=1, max_length=128)
    tool: str = Field(min_length=1, max_length=64)
    params: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """Episodic evidence summary kept on the class itself.

    The full episodic trail lives in the separate `kg_evidence` table
    (spec §5.1); this struct is the rolled-up view the agent sees as part
    of the frontmatter.
    """

    match_count: int = Field(default=0, ge=0)
    last_updated: datetime | None = None
    recent_refinements: list[str] = Field(default_factory=list)


class Frontmatter(BaseModel):
    """Structured JSON portion of a class — everything except the prose body.

    Stored as JSONB end-to-end (no YAML round-trip). Unknown fields are
    rejected: a typoed key would otherwise silently disappear on the next
    write, which would corrupt the agent's mental model of what it just
    wrote.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9-]{2,63}$")
    type: Literal["bug-class"]
    description: str = Field(min_length=1, max_length=512)
    signature: Signature
    related: Related = Field(default_factory=Related)
    context_sources: list[ContextSource] = Field(default_factory=list)
    evidence: Evidence = Field(default_factory=Evidence)


class BugClass(BaseModel):
    """The full envelope: structured frontmatter + opaque markdown body.

    The body is a free-form markdown playbook (spec §5.2 gives the section
    conventions: `## When this fires`, `## Playbook`, `## Common confusion`,
    `## Distilled fix patterns`). Inline `[[wiki-link]]` and
    `[[src/file.ts#fn]]` references are validated by `kg-check`, not by
    this schema.
    """

    model_config = ConfigDict(extra="forbid")

    frontmatter: Frontmatter
    body: str = ""
