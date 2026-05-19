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
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import (
    GraphEvidence,
    MatchProvenance,
    QueryChunk,
    QueryDocumentResult,
    QueryEntityResult,
    QueryResponse,
    QueryResult,
    RelatedEntity,
)

log = get_logger(__name__)


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


def _build_doc_to_graph_evidence(
    prefanout: dict[str, Any] | None,
) -> dict[str, list[GraphEvidence]]:
    """Index `inferred_edge` channel hits from the pre-fan-out by `doc_id`.

    Each inferred-edge hit carries `anchor_doc_id` (the originating doc
    of the LLM-asserted link) + `edge_type` + `confidence` + `why` (the
    rationale). The MCP consumer schema's `QueryChunk.graph_evidence`
    field is the projection point — it shows the chunk's graph
    provenance as `{edge_type, confidence, via_entity, reason}` entries.

    We dedup by `(anchor, edge_type)` per linked doc so the same chain
    hop doesn't render twice if fan-out anchors overlap across
    sub_queries. Linked-doc → list[GraphEvidence] is the return shape;
    the adapter joins per-chunk by parent doc_id.

    Returns an empty dict when there's no prefanout or no inferred-edge
    hits (the no-LLM / harness-passthrough fallback paths).
    """
    out: dict[str, list[GraphEvidence]] = {}
    seen_per_doc: dict[str, set[tuple[str, str]]] = {}
    if not prefanout:
        return out
    for sq in prefanout.get("sub_queries") or []:
        for hit in sq.get("inferred_edge") or []:
            doc_id = hit.get("doc_id")
            anchor = hit.get("anchor_doc_id")
            if not doc_id or not anchor:
                continue
            edge_type = hit.get("edge_type") or "INFERRED_EDGE"
            confidence = hit.get("confidence") or "INFERRED"
            seen = seen_per_doc.setdefault(doc_id, set())
            key = (anchor, edge_type)
            if key in seen:
                continue
            seen.add(key)
            out.setdefault(doc_id, []).append(
                GraphEvidence(
                    edge_type=edge_type,
                    confidence=confidence,
                    # The inferred-edge hit's "via" is the anchor doc the
                    # edge originates from — semantically the chain's
                    # other endpoint. GraphEvidence.via_entity is a
                    # free-form string in the existing schema.
                    via_entity=anchor,
                    reason=hit.get("why"),
                )
            )
    return out


# Per-anchor cap on injected neighbors. Surfacing graph_evidence for
# INFERRED edges where the result-set doc is one endpoint regardless of
# whether the OTHER endpoint is curated; cap so a high-degree doc
# (claude_code session with 126 neighbors, wiki:repo:prbe_knowledge with
# 83 neighbors) doesn't dump 100+ edges into the response and bloat
# graph_evidence beyond what the chain panel can render. Top 8 keeps
# the densest chains readable while letting low-degree docs surface
# all of their (typically 2-5) connections.
_ENRICH_TOP_K_PER_ANCHOR = 8


async def _enrich_graph_evidence_from_result_set(
    customer_id: str,
    doc_ids: list[str],
) -> dict[str, list[GraphEvidence]]:
    """Query `graph_edges` for INFERRED edges where AT LEAST ONE endpoint
    is in the curated result set; project each as a `GraphEvidence`
    entry on that endpoint's doc.

    Why ONE endpoint (not both): live-traced 2026-05-19. Same query
    twice 4 min apart, same grounding/extraction (entity
    `multi-granola-support`), but the Cerebras-backed gatherer emitted
    5 docs on run 1 (Linear PRB-18 + 3 Slack threads + entity) and only
    2 docs on run 2 (Linear PRB-18 + entity). Run 2's curation dropped
    the Slack threads that carry all the chain edges to PRB-18 →
    `confidence_breakdown.INFERRED` collapsed from 10 to 0 → chain
    panel renders empty even though PRB-18 has 47+ inferred-edge
    neighbors in the graph.

    Requiring BOTH endpoints in the result set means we only surface
    chain hops when the agent happened to curate matched pairs.
    Defaulting to ONE means a single result doc still gets its chain
    neighbors projected — the chain panel renders even under
    aggressive curation. The other endpoint's title is LEFT JOINed
    from `documents` so the chain viz can render the neighbor as a
    labeled node via `GraphEvidence.via_entity_title`.

    Per-anchor cap (`_ENRICH_TOP_K_PER_ANCHOR`) prevents a high-degree
    doc from dominating the payload — top-K by extraction recency.

    Returns empty dict when there are no doc_ids or the DB query
    fails. Best-effort — callers fall back to prefanout-only.
    """
    out: dict[str, list[GraphEvidence]] = {}
    if not doc_ids:
        return out
    try:
        async with with_tenant(customer_id) as conn:
            rows = await conn.fetch(
                """
                WITH result_endpoints AS (
                    -- All inferred edges touching at least one curated
                    -- doc, with the curated doc as `anchor_doc` and the
                    -- OTHER endpoint as `other_doc`. UNION ALL covers
                    -- both directions so the chain panel sees edges
                    -- regardless of from/to orientation.
                    SELECT
                        gn_from.canonical_id AS anchor_doc,
                        gn_to.canonical_id   AS other_doc,
                        ge.edge_type,
                        ge.confidence,
                        ge.properties->>'why' AS why,
                        ge.extracted_at
                    FROM graph_edges ge
                    JOIN graph_nodes gn_from ON gn_from.node_id = ge.from_node_id
                    JOIN graph_nodes gn_to   ON gn_to.node_id   = ge.to_node_id
                    WHERE ge.customer_id = $1
                      AND ge.confidence  = 'INFERRED'
                      AND gn_from.canonical_id = ANY($2::text[])
                      AND gn_from.canonical_id <> gn_to.canonical_id
                    UNION ALL
                    SELECT
                        gn_to.canonical_id   AS anchor_doc,
                        gn_from.canonical_id AS other_doc,
                        ge.edge_type,
                        ge.confidence,
                        ge.properties->>'why' AS why,
                        ge.extracted_at
                    FROM graph_edges ge
                    JOIN graph_nodes gn_from ON gn_from.node_id = ge.from_node_id
                    JOIN graph_nodes gn_to   ON gn_to.node_id   = ge.to_node_id
                    WHERE ge.customer_id = $1
                      AND ge.confidence  = 'INFERRED'
                      AND gn_to.canonical_id = ANY($2::text[])
                      AND gn_from.canonical_id <> gn_to.canonical_id
                ),
                ranked AS (
                    -- Dedup by (anchor, other, edge_type) so reciprocal
                    -- entries from the UNION ALL collapse, then top-K
                    -- per anchor by extraction recency.
                    SELECT
                        re.*,
                        d.title AS other_title,
                        row_number() OVER (
                            PARTITION BY re.anchor_doc, re.other_doc, re.edge_type
                            ORDER BY re.extracted_at DESC NULLS LAST
                        ) AS dedup_rn,
                        row_number() OVER (
                            PARTITION BY re.anchor_doc
                            ORDER BY re.extracted_at DESC NULLS LAST
                        ) AS anchor_rn
                    FROM result_endpoints re
                    LEFT JOIN documents d
                      ON d.customer_id = $1
                     AND d.doc_id = re.other_doc
                )
                SELECT anchor_doc, other_doc, edge_type, confidence,
                       why, other_title
                FROM ranked
                WHERE dedup_rn = 1
                  AND anchor_rn <= $3
                ORDER BY anchor_doc, anchor_rn
                """,
                customer_id,
                doc_ids,
                _ENRICH_TOP_K_PER_ANCHOR,
            )
    except Exception as exc:
        log.warning(
            "adapter.enrich_graph_evidence.failed",
            customer=customer_id,
            doc_id_count=len(doc_ids),
            error=str(exc),
        )
        return out
    for r in rows:
        ev = GraphEvidence(
            edge_type=r["edge_type"],
            confidence=r["confidence"],
            via_entity=r["other_doc"],
            reason=r["why"],
            via_entity_title=r["other_title"],
        )
        out.setdefault(r["anchor_doc"], []).append(ev)
    return out


def _chunk_to_query_chunk(
    chunk: Any,
    doc_id: str,
    rank: int,
    graph_evidence: list[GraphEvidence],
) -> QueryChunk:
    """Convert a GatheredChunk into the existing QueryChunk shape.

    `rank_in_doc` is 1-indexed (the existing shape). The gatherer surfaces
    chunks in the order it chose to emit them; we preserve that as the
    in-doc ranking. `graph_evidence` carries the inferred-edge chain
    metadata (edge_type / confidence / anchor doc / `why` rationale)
    when the parent doc was surfaced via the inferred-edge channel;
    empty list when the doc reached the agent via vector / bm25 / graph
    alone.
    """
    return QueryChunk(
        chunk_id=chunk.chunk_id,
        content=chunk.content,
        score=1.0 - (0.01 * rank),  # decay so consumers' sort is stable; agent's
                                    # actual ranking signal is matched_via / why_relevant
        rank_in_doc=rank + 1,
        retriever_scores={},
        graph_evidence=graph_evidence,
    )


async def to_query_response(
    *,
    query: str,
    gathered: GathererOutput,
    trace_id: str,
    timing_ms: dict[str, float],
    prefanout: dict[str, Any] | None = None,
    customer_id: str | None = None,
) -> QueryResponse:
    """Wrap a GathererOutput in the existing QueryResponse shape.

    Grouping: chunks with the same doc_id are merged into one
    QueryDocumentResult; the first chunk's doc_id determines metadata
    (source_system, title, created_at/updated_at are minimally filled).

    Entities surface as QueryEntityResult rows alongside Documents
    (existing pattern from list/search).

    `prefanout` (optional): the harness-captured `execute_search` result
    dict carrying the inferred-edge channel hits. When provided, each
    chunk's `graph_evidence` is populated from inferred-edge hits whose
    `doc_id` matches the chunk's parent doc — projecting the LLM-asserted
    edge metadata (`edge_type`, `confidence`, `anchor_doc_id`, `why`)
    onto the consumer-visible chain provenance. None on the no-LLM /
    harness-passthrough fallback paths (no chain data to project).

    `customer_id` (optional): when set, the adapter ALSO queries
    `graph_edges` for INFERRED edges between any pair of docs in the
    result set and merges them into `graph_evidence`. This catches the
    common case where the agent emits N cross-linked docs that came
    via vector/BM25/graph — the prefanout's inferred_edge channel
    missed them because none was the grounded anchor, but the graph
    *does* have edges between them. None preserves the pre-enrichment
    behaviour for tests / harness-passthrough.
    """
    now = datetime.now(UTC)

    doc_evidence = _build_doc_to_graph_evidence(prefanout)

    # Post-hoc enrichment from graph_edges. Merged into the prefanout-
    # derived doc_evidence below; dedup by (via_entity, edge_type) per
    # doc so a hop already surfaced via prefanout doesn't double-render.
    # Threshold dropped to 1 doc (was 2) — `_enrich_graph_evidence_from_
    # result_set` now surfaces edges where AT LEAST ONE endpoint is in
    # the curated set, so a single result doc still gets its chain
    # neighbors projected. Live trace 2026-05-19 showed Cerebras agent
    # curation collapsing 5 docs → 2 docs on identical-query reruns,
    # which under the old `>= 2` gate would have still failed to render
    # the chain. Under the new logic a single PRB-18 doc still gets its
    # 47 inferred-edge neighbors projected as graph_evidence.
    result_set_doc_ids = [c.doc_id for c in gathered.chunks if c.doc_id]
    if customer_id and result_set_doc_ids:
        graph_evidence_extra = await _enrich_graph_evidence_from_result_set(
            customer_id, list(set(result_set_doc_ids))
        )
        for doc_id, extras in graph_evidence_extra.items():
            existing = doc_evidence.setdefault(doc_id, [])
            seen: set[tuple[str, str]] = {(e.via_entity, e.edge_type) for e in existing}
            for ev in extras:
                key = (ev.via_entity, ev.edge_type)
                if key in seen:
                    continue
                seen.add(key)
                existing.append(ev)

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
    confidence_breakdown = {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0}

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

        # Per-doc graph_evidence is shared across this doc's chunks —
        # the edges connect docs, not chunks. Every chunk of a given doc
        # carries the same evidence (consumers may dedupe at the doc
        # level if they prefer).
        evidence = doc_evidence.get(doc_id, [])
        for ge in evidence:
            tier = ge.confidence if ge.confidence in confidence_breakdown else "AMBIGUOUS"
            confidence_breakdown[tier] += 1

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
                chunks=[
                    _chunk_to_query_chunk(
                        c, doc_id=doc_id, rank=i, graph_evidence=evidence
                    )
                    for i, c in enumerate(chunks)
                ],
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

    # `related_entities` is the dashboard / MCP-consumer crawl-candidate
    # list — non-Document graph nodes attached to the result docs that
    # callers can drop into the next search to BFS the graph. Pre-cutover
    # this was filled by `related_entities.py` (1-hop walker on result
    # docs). The gatherer doesn't run that retriever; instead we project
    # the agent's curated `entities[]` here so the dashboard's
    # related-entities panel stops rendering empty. Values are best-
    # effort: doc_count=1 (the agent kept it), score=1.0 (perfect within
    # curated set), max_confidence="EXTRACTED" (agent kept ≈ deterministic
    # grounding match). A future PR can swap this for the proper walker.
    related_entities = [
        RelatedEntity(
            canonical_id=e.canonical_id,
            label=e.label or _label_from_canonical_id(e.canonical_id),
            display_name=str(
                e.properties.get("name")
                or e.properties.get("display_name")
                or e.canonical_id
            ),
            edge_types=[],
            max_confidence="EXTRACTED",
            doc_count=1,
            score=1.0,
            associated_doc_ids=[],
            member_count=1,
            member_sources=[],
        )
        for e in gathered.entities
    ]

    return QueryResponse(
        query=query,
        results=results,
        total_candidates=len(results),
        router_hit_cache=False,
        timing_ms=timing_ms,
        trace_id=trace_id,
        confidence_breakdown=confidence_breakdown,
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
        related_entities=related_entities or None,
        gatherer_notes=gathered.gatherer_notes.model_dump(),
    )


def _label_from_canonical_id(canonical_id: str) -> str:
    """Derive a NodeLabel-style label from a canonical_id prefix when the
    agent emitted a `GatheredEntity` with an empty `label` field. The
    canonical_id namespace is the authoritative source-of-truth for the
    node's type (`feature:gh:...`, `pr:github:...`, `linear:...:issue:...`,
    etc.). Without this fallback the dashboard's RelatedEntity panel
    renders blank labels for any entity the agent didn't bother to
    label."""
    if not canonical_id or ":" not in canonical_id:
        return ""
    prefix = canonical_id.split(":", 1)[0].strip()
    return prefix.capitalize() if prefix else ""


__all__ = ["to_query_response"]
