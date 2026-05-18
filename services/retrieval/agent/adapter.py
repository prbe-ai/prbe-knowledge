"""Adapter: GathererOutput -> existing QueryResponse shape.

The MCP consumer schema (`shared.models.QueryResponse`) is unchanged
by the cutover (per plan anti-scope #1). The gatherer emits its own
Pydantic shape; this adapter translates it into the existing response
so downstream consumers (Claude Code, Codex, dashboard, MCP server)
keep working without code changes.

The adapter is conservative — it surfaces what the gatherer emitted
and leaves optional fields empty rather than synthesizing data. The
new `gatherer_notes` field is passed through verbatim for debug clients.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from services.retrieval.agent.models import GathererOutput
from shared.constants import SourceSystem
from shared.models import (
    MatchProvenance,
    QueryChunk,
    QueryDocumentResult,
    QueryEntityResult,
    QueryResponse,
    QueryResult,
)


def _safe_source_system(value: str | None, doc_id: str | None = None) -> str:
    """Coerce a free-form source_system string to the SourceSystem enum value.

    `value` is whatever the gatherer surfaced (now propagated through
    GatheredChunk.source_system, filled from the prefanout hit by
    `_coerce_lenient`). `doc_id` is the namespaced doc_id — its prefix
    serves as a fallback (e.g. `slack:thread:T123` → `"slack"`) when
    `value` is missing. Final fallback is GitHub to match the
    pre-extension wire shape; this only fires when both the gatherer
    value AND the doc_id prefix are unrecognised, which should be rare.

    Pre-extension (before GatheredChunk grew a source_system field),
    every call site passed `None` and got `"github"` — every result was
    mislabelled as GitHub regardless of its true source.
    """
    if value:
        try:
            return SourceSystem(value).value
        except ValueError:
            return value  # unknown enum value — surface rather than drop
    if doc_id and ":" in doc_id:
        prefix = doc_id.split(":", 1)[0].strip().lower()
        try:
            return SourceSystem(prefix).value
        except ValueError:
            pass
    return SourceSystem.GITHUB.value


def _parse_iso(value: Any) -> datetime | None:
    """Parse a GatheredChunk ISO8601 timestamp string into a datetime, or
    return None when the input isn't a parseable string. The chunk model
    types these as `str | None`; the upstream channel hit had a real
    datetime that was stringified at the dict-conversion boundary."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _chunk_to_query_chunk(chunk: Any, doc_id: str, rank: int) -> QueryChunk:
    """Convert a GatheredChunk into the existing QueryChunk shape.

    `rank_in_doc` is 1-indexed (the existing shape). The gatherer surfaces
    chunks in the order it chose to emit them; we preserve that as the
    in-doc ranking.
    """
    return QueryChunk(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        score=1.0 - (0.01 * rank),  # decay so consumers' sort is stable; agent's
                                    # actual ranking signal is matched_via / why_relevant
        rank_in_doc=rank + 1,
        retriever_scores={},
        graph_evidence=[],
    )


def to_query_response(
    *,
    query: str,
    gathered: GathererOutput,
    trace_id: str,
    timing_ms: dict[str, float],
) -> QueryResponse:
    """Wrap a GathererOutput in the existing QueryResponse shape.

    Grouping: chunks with the same doc_id are merged into one
    QueryDocumentResult; the first chunk's doc_id determines metadata
    (source_system, title, created_at/updated_at are minimally filled).

    Entities surface as QueryEntityResult rows alongside Documents
    (existing pattern from list/search).
    """
    now = datetime.now(UTC)

    # Group chunks by doc_id.
    doc_groups: dict[str, list[Any]] = {}
    doc_order: list[str] = []
    for chunk in gathered.chunks:
        if chunk.doc_id not in doc_groups:
            doc_groups[chunk.doc_id] = []
            doc_order.append(chunk.doc_id)
        doc_groups[chunk.doc_id].append(chunk)

    results: list[QueryResult] = []
    rank_counter = 0

    for doc_id in doc_order:
        chunks = doc_groups[doc_id]
        rank_counter += 1
        provenance: list[MatchProvenance] = []
        for c in chunks:
            for ch in c.matched_via:
                # MatchProvenance Literal allows only a subset; coerce unknown
                # channels to "vector" (lowest-fidelity fallback). The gatherer
                # tracks the real channel name in matched_via separately.
                allowed = {
                    "vector", "bm25", "graph", "inferred_edge",
                    "id_lookup", "directed",
                }
                channel_value = ch if ch in allowed else "vector"
                provenance.append(
                    MatchProvenance(
                        channel=channel_value,  # type: ignore[arg-type]
                        rank=rank_counter,
                        score=1.0,  # gatherer chose to surface; treat as max-confidence within the curated set
                    )
                )

        # Doc-level metadata: take the first chunk's pass-through fields
        # (they should agree across chunks of the same doc — populated
        # from the prefanout hit by `_coerce_lenient`). Timestamps fall
        # back to request time only when the chunk doesn't carry one.
        first = chunks[0]
        created_at = _parse_iso(getattr(first, "created_at", None)) or now
        updated_at = _parse_iso(getattr(first, "updated_at", None)) or now
        title = getattr(first, "title", "") or None
        source_url = getattr(first, "source_url", "") or ""
        author_id = getattr(first, "author_id", None)

        results.append(
            QueryDocumentResult(
                canonical_id=doc_id,
                score=1.0 - (0.01 * (rank_counter - 1)),
                rank=rank_counter,
                matched_via=provenance,
                doc_id=doc_id,
                doc_version=1,
                source_system=_safe_source_system(  # type: ignore[arg-type]
                    getattr(first, "source_system", None),
                    doc_id=doc_id,
                ),
                source_url=source_url,
                title=title,
                author_id=author_id,
                created_at=created_at,
                updated_at=updated_at,
                chunks=[_chunk_to_query_chunk(c, doc_id=doc_id, rank=i) for i, c in enumerate(chunks)],
                chunk_count=len(chunks),
                retriever_scores={},
            )
        )

    for entity in gathered.entities:
        rank_counter += 1
        results.append(
            QueryEntityResult(
                canonical_id=entity.canonical_id,
                score=1.0 - (0.01 * (rank_counter - 1)),
                rank=rank_counter,
                matched_via=[],
                label=entity.label,
                display_name=str(entity.properties.get("name") or entity.properties.get("display_name") or entity.canonical_id),
                properties=entity.properties,
                attached_doc_ids=[],
                edge_types=[],
                doc_count=0,
            )
        )

    return QueryResponse(
        query=query,
        results=results,
        total_candidates=len(results),
        router_hit_cache=False,
        timing_ms=timing_ms,
        trace_id=trace_id,
        extracted_entities=[
            {
                "entity_type": e.label.lower(),
                "canonical_id": e.canonical_id,
                "display_name": str(
                    e.properties.get("name") or e.properties.get("display_name") or e.canonical_id
                ),
                "confidence": 1.0,
            }
            for e in gathered.entities
        ],
        gatherer_notes=gathered.gatherer_notes.model_dump(),
    )


__all__ = ["to_query_response"]
