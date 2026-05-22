"""Pydantic models the gatherer emits.

`GathererOutput` is the final-turn schema enforced via
`response_format=GathererOutput` -> LiteLLM JSON Schema conversion ->
Fireworks constrained decoding. The schema is the structural guarantee;
the prompt does NOT instruct the model to emit JSON like this.

The adapter (`services/retrieval/agent/adapter.py`) translates a
`GathererOutput` into the existing `RetrieveResponse` shape for MCP /
dashboard consumers — no breaking changes downstream.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Subset of channels we surface in matched_via. Mirrors the tool-name set
# the agent uses, plus the two derived-from-walker labels.
MatchedViaChannel = Literal[
    "vector",
    "bm25",
    "graph",
    "inferred_edge",
    "graph_walk",
    "inferred_neighbor",
    "id_lookup",
    "entity_cluster",
    "reissue",
]

ConfidenceLabel = Literal["high", "medium", "low"]

GathererStatus = Literal[
    "ok",
    "passthrough_harness_fallback",
    "loop_timeout",
    "schema_violation",
    "tool_budget_exceeded",
    # No LLM credentials configured (test env / bootstrap / self-host without
    # SEARCH_AGENT_INFERENCE_MODEL provider key). Loop short-circuits to an
    # empty GathererOutput rather than 503ing — mirrors the pre-cutover
    # router's `_call_haiku` short-circuit when neither Anthropic key nor
    # LLM gateway URL was set.
    "no_llm_configured",
    # LLMError raised mid-loop (provider 5xx, network, etc). The
    # harness re-raises as HTTPException(503), but we still capture the
    # trace blob with this status — the partial transcript is exactly
    # the artifact we need to debug provider-side regressions.
    "fatal_provider_error",
    # Grounding+extraction surfaced no entities AND all 4 pre-fan-out
    # channels returned 0 hits. The LLM cannot synthesize results from
    # nothing; the harness returns an empty GathererOutput without
    # entering the loop, saving 3-5s on truly hopeless queries.
    "zero_recall_short_circuit",
]


class GatheredEntity(BaseModel):
    """A graph node the agent chose to surface alongside the chunks.

    Schema tolerance: `extra="ignore"` so providers (e.g. Cerebras
    gpt-oss-120b) that emit non-schema fields like `title`/`source`/`url`
    by copying input shapes don't fail the whole emission. Required-field
    defaults absorb providers whose constrained decoding is laxer than
    Fireworks's — Fireworks still emits these fields, the defaults are
    only a fallback for non-strict providers.
    """

    model_config = ConfigDict(extra="ignore")

    canonical_id: str
    label: str = Field(
        default="",
        description="GraphLabel (Repo, Ticket, Person, etc.). Adapter falls "
        "back to canonical_id prefix when blank.",
    )
    properties: dict = Field(
        default_factory=dict,
        description="Node properties, trimmed at the tool layer to ~2KB.",
    )
    why_relevant: str = Field(
        default="",
        description="One-line, agent-written rationale for surfacing this entity.",
    )


class GatheredChunk(BaseModel):
    """A doc chunk the agent chose to surface.

    Schema tolerance: `extra="ignore"` for non-strict providers.
    `doc_id` stays required — without it the chunk has no resolvable
    citation. Pre-parse coercion in loop.py derives `doc_id` from
    `chunk_id` when the model omits it.

    Doc-level pass-through fields (source_system / title / source_url /
    created_at / updated_at / author_id) are optional on the agent's
    emission contract — they're filled from the pre-fan-out hit by
    `_coerce_lenient` in loop.py so the adapter doesn't have to
    fabricate them. Pre-extension, every QueryDocumentResult landed
    labelled `source_system="github"` because the adapter had no source
    field to read.
    """

    model_config = ConfigDict(extra="ignore")

    doc_id: str
    chunk_id: str
    content: str
    matched_via: list[MatchedViaChannel] = Field(
        default_factory=list,
        description="Channels that surfaced this chunk during the loop.",
    )
    why_relevant: str = Field(
        default="",
        description="One-line, agent-written rationale. For inferred-edge "
        "neighbors, quote the edge `why` string verbatim.",
    )
    source_system: str = Field(
        default="",
        description="Source system slug (github/slack/linear/notion/sentry/...). "
        "Harness fills from prefanout hit; falls back to doc_id prefix.",
    )
    title: str = Field(default="", description="Doc title; blank when not known.")
    source_url: str = Field(default="", description="Doc URL; blank when not known.")
    created_at: str | None = Field(default=None, description="ISO8601 timestamp from doc.")
    updated_at: str | None = Field(default=None, description="ISO8601 timestamp from doc.")
    author_id: str | None = Field(default=None, description="Author canonical_id from doc.")


class DroppedCandidate(BaseModel):
    """A candidate the agent saw but chose not to surface."""

    model_config = ConfigDict(extra="ignore")

    canonical_id: str
    reason: str = ""


class GathererNotes(BaseModel):
    """Self-reported metadata about the agent's loop.

    `turns_used` is harness-authoritative — the loop overwrites whatever
    the model emits from `state.turn_count` in `_parse_terminal_args`.
    Same pattern for `tools_called` (from `state.tools_fired`). These
    defaults are only relevant if those harness overwrites don't run.
    """

    model_config = ConfigDict(extra="ignore")

    turns_used: int = Field(default=0, ge=0)
    tools_called: list[str] = Field(
        default_factory=list,
        description="Tool names in call order. Sequence preserved for trace replay.",
    )
    confidence: ConfidenceLabel = "medium"
    dropped: list[DroppedCandidate] = Field(default_factory=list)


class GathererOutput(BaseModel):
    """Final emission from the gatherer agent.

    `response_format=GathererOutput` is constrained-decoded by Fireworks
    on the final turn. The harness re-parses for defence in depth; on
    parse failure it emits an empty GathererOutput with
    gatherer_status='passthrough_harness_fallback' (consumers see zero
    results + clear status). For non-strict providers (Cerebras et al.),
    pre-parse coercion in loop.py normalizes common field-name drift
    (title→ignored, source→ignored) and derives missing required fields
    from harness state before validation.
    """

    model_config = ConfigDict(extra="ignore")

    entities: list[GatheredEntity] = Field(default_factory=list)
    chunks: list[GatheredChunk] = Field(default_factory=list)
    gatherer_notes: GathererNotes = Field(default_factory=GathererNotes)


# ============================================================
# LLM-based entity extraction (parallel with deterministic grounding)
# ============================================================
# The Haiku router used to do LLM-based entity extraction before the
# cutover. Grounding's pg_trgm fuzzy + tsvector match recovers most
# bare-ID and prefix cases, but it misses paraphrased entities ("the
# new login flow" when the graph node is named "Authentication Phase 2").
# This shape is the recovery path: a tiny LLM call (same Fireworks model
# as the agent loop, parallel with grounding) reads the query and proposes
# entities. Results merge with grounding before pre-fan-out.

EntityType = Literal[
    "person",
    "repo",
    "service",
    "ticket",
    "pr",
    "feature",
    "decision",
    "error_group",
    "file_path",
    "channel",
    "session",
    "commit_sha",
]


class ExtractedEntity(BaseModel):
    """One LLM-proposed entity. canonical_id may be the LLM's best guess
    (synthesized) — `_reconcile_entities_with_bundle` swaps it for the
    grounded value when a fuzzy/bare-ID match exists in the bundle."""

    model_config = ConfigDict(extra="forbid")

    entity_type: EntityType
    canonical_id: str
    display_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)


# Allowed sort values for SearchOptions. Kept as a module-level tuple so the
# post-parse coercion helper in extractor.py can validate against the same
# source as the Pydantic Literal. Adding a value here requires updating the
# Literal too — drift caught by `tests/retrieval/agent/test_extractor.py`.
SortMode = Literal["recency", "relevance"]


class SearchOptions(BaseModel):
    """Deterministic options the extractor derives from the query shape.

    The harness threads these into every retriever channel of the
    pre-fan-out so the channels collectively honor the query's intent
    (recency-sorted vs relevance-sorted, narrowed to a class of docs vs
    unrestricted, etc.).

    Fields:
      - `sort`: "recency" when the query asks for the most recent /
        latest / etc.; "relevance" otherwise (default).
      - `doc_types`: when the query asks about a CLASS of entity
        ("show me the latest PRs", "what tickets are in progress") set
        the matching DocType strings here (e.g. ["github.pull_request"])
        so every channel filters `documents.doc_type = ANY(...)`.
        Combined with `sort=recency` this is how "latest PR" gets the
        actual latest PR — instead of the extractor over-grounding on
        one specific PR candidate it found in the corpus.

    Future expansions land here:
        - `temporal: TemporalSpecSymbolic | None` — date-window filters.
        - `sources: list[str]` — "search only Slack".

    `model_config["extra"] = "forbid"` ensures the schema can't drift
    without an explicit revision here.
    """

    model_config = ConfigDict(extra="forbid")

    sort: SortMode = "relevance"
    doc_types: list[str] | None = None


class EntityExtraction(BaseModel):
    """`response_format=EntityExtraction` is constrained-decoded on a tiny
    upfront call. The model returns entities plus search_options — the
    options modify HOW the harness queries each channel (sort by recency
    vs relevance, etc.), and default to today's behavior when unset."""

    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    search_options: SearchOptions = Field(default_factory=SearchOptions)
