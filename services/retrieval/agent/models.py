"""Pydantic models the gatherer emits.

`GathererOutput` is the final-turn schema enforced via
`response_format=GathererOutput` -> LiteLLM JSON Schema conversion ->
Fireworks constrained decoding. The schema is the structural guarantee;
the prompt does NOT instruct the model to emit JSON like this.

The adapter (`services/retrieval/agent/adapter.py`) translates a
`GathererOutput` into the existing `QueryResponse` shape for MCP /
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
]


class GatheredEntity(BaseModel):
    """A graph node the agent chose to surface alongside the chunks."""

    model_config = ConfigDict(extra="forbid")

    canonical_id: str
    label: str
    properties: dict = Field(
        default_factory=dict,
        description="Node properties, trimmed at the tool layer to ~2KB.",
    )
    why_relevant: str = Field(
        ...,
        description="One-line, agent-written rationale for surfacing this entity.",
    )


class GatheredChunk(BaseModel):
    """A doc chunk the agent chose to surface."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str
    chunk_id: str
    content: str
    matched_via: list[MatchedViaChannel] = Field(
        ...,
        description="Channels that surfaced this chunk during the loop.",
    )
    why_relevant: str = Field(
        ...,
        description="One-line, agent-written rationale. For inferred-edge "
        "neighbors, quote the edge `why` string verbatim.",
    )


class DroppedCandidate(BaseModel):
    """A candidate the agent saw but chose not to surface.

    `reason` is mandatory: the schema prevents the model from emitting a
    drop without a justification, so the dashboard's debug pane can
    always render the audit trail.
    """

    model_config = ConfigDict(extra="forbid")

    canonical_id: str
    reason: str


class GathererNotes(BaseModel):
    """Self-reported metadata about the agent's loop.

    Persisted into `query_traces` via migration 0078 columns and
    surfaced to debug consumers in the response.
    """

    model_config = ConfigDict(extra="forbid")

    turns_used: int = Field(..., ge=0)
    tools_called: list[str] = Field(
        default_factory=list,
        description="Tool names in call order. Sequence preserved for trace replay.",
    )
    confidence: ConfidenceLabel
    dropped: list[DroppedCandidate] = Field(default_factory=list)


class GathererOutput(BaseModel):
    """Final emission from the gatherer agent.

    `response_format=GathererOutput` is constrained-decoded by Fireworks
    on the final turn. The harness re-parses for defence in depth; on
    parse failure it emits an empty GathererOutput with
    gatherer_status='passthrough_harness_fallback' (consumers see zero
    results + clear status).
    """

    model_config = ConfigDict(extra="forbid")

    entities: list[GatheredEntity] = Field(default_factory=list)
    chunks: list[GatheredChunk] = Field(default_factory=list)
    gatherer_notes: GathererNotes


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


class EntityExtraction(BaseModel):
    """`response_format=EntityExtraction` is constrained-decoded on a tiny
    upfront call. The model returns only entities — no temporal/sort/mode
    classification (the agent infers those from query text + tool selection)."""

    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
