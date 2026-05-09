"""Semantic search pipeline — vector + BM25 + graph + RRF + dedup + ACL.

This is the original retrieval pipeline (extracted from `main.py` during the
list-mode split). Two behavior changes vs. before:

1. `doc_types` is a SOFT signal here, not a filter. The list pipeline uses
   doc_type as a hard `WHERE` clause; on the search path, hard-filtering by
   Haiku's doc_type guess tanks recall when the answer lives in a related
   doc type (e.g. "the PR that broke login" — the answer might be in a
   linked commit). Instead we apply doc_type as an RRF score boost via
   helpers.apply_entity_filter — which already runs over fused hits when
   `entity_must_match` is on — and otherwise just pass the matching
   doc_types into the source-filter slot for retrievers that find them
   useful.

2. Sort intent does NOT trigger hard post-fusion sort here. When Haiku
   detects "most recent X about Y" (a search-path query because Y is a
   topic entity), we amplify the recency boost in RRF (shorter half-life)
   instead of the old `pool_multiplier=5 + post-fusion sort` behavior —
   that was the original bug we're fixing. Pure relevance ranking is
   preserved; new items get a stronger nudge.

Polymorphic output
──────────────────
The pipeline emits `QueryResponse.results: list[QueryResult]` where each
result is a discriminated union of `QueryDocumentResult` (body chunks
nested) and `QueryEntityResult` (graph entities the router asked about).
Each result carries a `matched_via: list[MatchProvenance]` trace so MCP
consumers can see which channel(s) surfaced it. A 4th channel
(`inferred_edge`) walks LLM-derived Doc-Doc edges from the top primary
results and surfaces linked docs as primary Document results with `why`
justifications attached.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any

from services.retrieval.acl import filter_by_acl
from services.retrieval.dedup import dedupe
from services.retrieval.fusion import FusedChunk, FusedDocument, fuse
from services.retrieval.helpers import (
    apply_entity_filter,
    embeddings_for_chunks,
)
from services.retrieval.retrievers.bm25 import BM25Hit, bm25_search, residualize_for_bm25
from services.retrieval.retrievers.directed import directed_search
from services.retrieval.retrievers.graph import GraphHit, graph_search
from services.retrieval.retrievers.id_lookup import id_lookup_search, is_lookup_candidate
from services.retrieval.retrievers.inferred_edges import (
    InferredEdgeHit,
    inferred_edge_search,
)
from services.retrieval.retrievers.related_entities import (
    build_exclude_node_keys,
    walk_result_doc_neighbors,
)
from services.retrieval.retrievers.vector import vector_search
from services.retrieval.router import RouterEntity, RouterOutput
from services.retrieval.temporal import build_predicate
from shared.constants import (
    INFERRED_EDGE_DAMPENING,
    INFERRED_EDGE_HYDRATION_CHUNKS,
    INFERRED_EDGE_TOP_K,
    ROUTER_ENTITY_TO_LABEL,
    SOURCE_SCORE_MULTIPLIERS,
    SourceSystem,
)
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import (
    GraphEvidence,
    MatchProvenance,
    QueryChunk,
    QueryDocumentResult,
    QueryEntityResult,
    QueryRequest,
    QueryResponse,
    QueryResult,
    RelatedEntity,
    TemporalSpec,
    normalize_author_id,
)

log = get_logger(__name__)

# When the router detects sort intent on a search-path query, drop the
# recency half-life by this factor (and clamp to a 7-day floor). Caller's
# explicit `recency_half_life_days` always wins.
_SORT_INTENT_HALF_LIFE_DIVISOR = 4
_SORT_INTENT_MIN_HALF_LIFE_DAYS = 7.0
_CODING_AGENT_SOURCES = {
    SourceSystem.CLAUDE_CODE.value,
    SourceSystem.CODEX.value,
}

# How many attached docs to surface on each QueryEntityResult. The cap keeps
# the response shape bounded for chatty entities; `doc_count` carries the
# uncapped total so consumers can audit completeness.
_ENTITY_ATTACHED_DOC_CAP = 5


async def _content_fallbacks_for_metadata_only_agent_hits(
    customer_id: str,
    ranked_lists: dict[str, list],
    spec: TemporalSpec,
) -> list[BM25Hit]:
    """Return displayable content chunks for coding-agent metadata-only hits.

    Metadata chunks carry searchable titles, names, emails, hostnames, and
    source URLs. Fusion must not return that synthetic text directly, but for
    coding-agent sessions the metadata is often the only place a user/person
    name appears. When a Codex/Claude Code doc matches only via metadata, fetch
    the first real content chunk so fusion can return the transcript with the
    metadata score.
    """
    metadata_doc_ids: set[str] = set()
    content_doc_ids: set[str] = set()

    for hits in ranked_lists.values():
        for hit in hits:
            kind = getattr(hit, "kind", "content")
            if kind == "metadata" and hit.source_system in _CODING_AGENT_SOURCES:
                metadata_doc_ids.add(hit.doc_id)
            elif kind != "metadata":
                content_doc_ids.add(hit.doc_id)

    doc_ids = sorted(metadata_doc_ids - content_doc_ids)
    if not doc_ids:
        return []

    async with with_tenant(customer_id) as conn:
        params: list = [customer_id, doc_ids]
        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)
        rows = await conn.fetch(
            f"""
            SELECT DISTINCT ON (c.doc_id)
                   c.chunk_id,
                   c.doc_id,
                   d.version AS doc_version,
                   d.source_system,
                   d.source_url,
                   d.title,
                   d.author_id,
                   c.content,
                   d.created_at,
                   d.updated_at
            FROM chunks c
            JOIN documents d
              ON c.doc_id = d.doc_id
             AND d.customer_id = c.customer_id
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE c.customer_id = $1
              AND c.doc_id = ANY($2::text[])
              AND COALESCE(c.kind, 'content') = 'content'
              {pred.chunk_sql}
              {pred.doc_sql}
            ORDER BY c.doc_id, c.chunk_index ASC
            """,
            *params,
        )

    return [
        BM25Hit(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            doc_version=r["doc_version"],
            source_system=r["source_system"],
            source_url=r["source_url"],
            title=r["title"],
            content=r["content"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            score=0.0,
            author_id=normalize_author_id(r["author_id"]),
            kind="content_fallback",
        )
        for r in rows
    ]


def _inject_id_lookup_hits(
    fused: list[FusedDocument], id_hits: list
) -> list[FusedDocument]:
    """Append synthetic FusedDocuments for any id_lookup-matched doc that
    didn't survive `fuse()`'s top_k cap.

    Why this exists: at MCP defaults (top_k=5), the fused pool is
    5 * pool_multiplier = 10. The fusion pipeline applies a per-source
    multiplier (claude_code/codex = 0.5x) and recency decay (7-day
    half-life), which routinely buries an exact-id match below 10 — so a
    pure pin-after-fusion would have nothing to pin. Injecting these docs
    here puts them into the dedupe/ACL/pin path with the rest, and the
    pin pass downstream guarantees they end up at rank 1.

    Synthetic docs carry `score=1.0` and one chunk with
    `retriever_scores={"id_lookup": 1.0}` so response telemetry still
    reflects which retriever surfaced them.
    """
    if not id_hits:
        return fused
    fused_doc_ids = {f.doc_id for f in fused}
    extras: list[FusedDocument] = []
    for h in id_hits:
        if h.doc_id in fused_doc_ids:
            continue
        fused_doc_ids.add(h.doc_id)
        extras.append(
            FusedDocument(
                doc_id=h.doc_id,
                doc_version=h.doc_version,
                source_system=h.source_system,
                source_url=h.source_url,
                title=h.title,
                created_at=h.created_at,
                updated_at=h.updated_at,
                score=1.0,
                author_id=h.author_id,
                retriever_scores={"id_lookup": 1.0},
                chunks=[
                    FusedChunk(
                        chunk_id=h.chunk_id,
                        content=h.content,
                        score=1.0,
                        retriever_scores={"id_lookup": 1.0},
                        rank_in_doc=1,
                    )
                ],
            )
        )
    return fused + extras


def _pin_id_lookup_matches(
    fused: list[FusedDocument], id_hits: list
) -> list[FusedDocument]:
    """Float docs whose source_id was an exact match by `id_lookup_search`
    to the top of the fused list, preserving id_lookup's original order.

    Why: a UUID-precise query like "session 3c325e11-..." should land the
    matching doc at rank 1. Fusion's per-source multiplier (claude_code /
    codex are 0.5x) and short half-life (7 days) can demote a session
    below code_graph BM25 hits even when `id_lookup` nailed an exact
    match — observed live: target session at rank 36 with id_lookup score
    1.0 still buried under code_graph chunks at BM25 ranks 1-35. This
    pin guarantees exact-id matches sit on top regardless of the discount
    factors, without changing the ranking of unmatched docs below.

    Pairs with `_inject_id_lookup_hits`: injection ensures the doc is in
    the candidate pool when top_k is small (MCP default 5); pin then
    floats it to position 0.
    """
    if not id_hits:
        return fused
    seen: set[str] = set()
    id_order: list[str] = []
    for h in id_hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            id_order.append(h.doc_id)
    by_doc = {f.doc_id: f for f in fused}
    pinned = [by_doc[did] for did in id_order if did in by_doc]
    rest = [f for f in fused if f.doc_id not in seen]
    return pinned + rest


def _per_channel_doc_ranks(
    ranked_lists: dict[str, list[Any]],
) -> dict[str, dict[str, tuple[int, float]]]:
    """For each channel, build {doc_id -> (best_rank, best_score)}.

    A channel can return multiple chunks of the same doc; we keep the
    BEST (lowest) rank and its score. Used downstream to populate
    `MatchProvenance` entries on the surviving QueryDocumentResults so
    the response carries a per-channel trace.
    """
    out: dict[str, dict[str, tuple[int, float]]] = {}
    for channel, hits in ranked_lists.items():
        per_doc: dict[str, tuple[int, float]] = {}
        for rank, hit in enumerate(hits, start=1):
            kind = getattr(hit, "kind", "content")
            if kind == "content_fallback":
                # Fallback hits don't represent a real channel match -- they
                # exist only to provide displayable content for metadata-only
                # agent hits. Don't claim provenance.
                continue
            doc_id = hit.doc_id
            score = float(getattr(hit, "score", 0.0))
            existing = per_doc.get(doc_id)
            if existing is None or rank < existing[0]:
                per_doc[doc_id] = (rank, score)
        out[channel] = per_doc
    return out


def _graph_evidence_by_doc(graph_hits: list[GraphHit]) -> dict[str, list[GraphEvidence]]:
    """Collapse GraphHits into per-doc lists of GraphEvidence entries.

    A single doc reachable via multiple seed entities accumulates one
    evidence entry per seed. Entries are deduped on (edge_type, via_entity)
    so a doc reached twice by the same seed-edge combo doesn't repeat.
    """
    out: dict[str, list[GraphEvidence]] = defaultdict(list)
    seen: dict[str, set[tuple[str | None, str]]] = defaultdict(set)
    for hit in graph_hits:
        key = (hit.edge_type, hit.via_entity)
        if key in seen[hit.doc_id]:
            continue
        seen[hit.doc_id].add(key)
        out[hit.doc_id].append(
            GraphEvidence(
                edge_type=hit.edge_type or "",
                confidence=hit.confidence or "EXTRACTED",
                via_entity=hit.via_entity,
                reason=None,
            )
        )
    return out


def _build_document_results_from_fused(
    fused_top: list[FusedDocument],
    ranked_lists: dict[str, list[Any]],
    graph_hits: list[GraphHit],
) -> list[QueryDocumentResult]:
    """Convert each FusedDocument to a QueryDocumentResult.

    Doc grouping + RRF_BREADTH_ALPHA scoring already happened in
    `fusion.fuse()`; FusedDocument arrives doc-keyed with `chunks` nested
    and `score` aggregated. This helper just wraps each into the
    polymorphic shape and populates per-channel `matched_via`.

    Within a doc, chunks are emitted in their FusedDocument order
    (fusion already sorted by score desc). `matched_via` is built from
    `ranked_lists` so every channel that surfaced the doc contributes a
    MatchProvenance entry.
    """
    per_channel = _per_channel_doc_ranks(ranked_lists)
    graph_evidence = _graph_evidence_by_doc(graph_hits)

    results: list[QueryDocumentResult] = []
    for doc in fused_top:
        chunk_models: list[QueryChunk] = [
            QueryChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                score=c.score,
                rank_in_doc=c.rank_in_doc or (i + 1),
                retriever_scores=dict(c.retriever_scores),
                graph_evidence=list(graph_evidence.get(doc.doc_id, [])),
            )
            for i, c in enumerate(doc.chunks)
        ]

        # Per-channel matched_via: only channels that surfaced this doc.
        # Score = the channel's best score for this doc; rank = the
        # channel's best (lowest) rank for this doc.
        provenances: list[MatchProvenance] = []
        for channel in ("vector", "bm25", "graph", "id_lookup"):
            entry = per_channel.get(channel, {}).get(doc.doc_id)
            if entry is None:
                continue
            rank, score = entry
            provenances.append(
                MatchProvenance(channel=channel, rank=rank, score=score)
            )

        results.append(
            QueryDocumentResult(
                canonical_id=doc.doc_id,
                doc_id=doc.doc_id,
                doc_version=doc.doc_version,
                source_system=SourceSystem(doc.source_system),
                source_url=doc.source_url,
                title=doc.title,
                author_id=doc.author_id,
                created_at=doc.created_at,
                updated_at=doc.updated_at,
                score=doc.score,
                rank=0,  # filled in by the caller after final sort
                matched_via=provenances,
                chunks=chunk_models,
                chunk_count=len(chunk_models),
                retriever_scores=dict(doc.retriever_scores),
            )
        )
    return results


def _inferred_edge_final_score(
    raw_score: float, source_system: str, linked_edge_count: int
) -> float:
    """Layer the cross-channel scoring policy on top of the retriever's
    raw dampened score.

    raw_score arrives as `dampening * 1/(1 + anchor_rank)`; this function
    multiplies in the per-source demotion (so an inferred-edge codex hit
    gets the same 0.5x as a vector codex hit) and divides by a fan-out
    penalty (so a high-degree hub doc gets crushed while a specific
    1-edge doc is barely affected).

    Fan-out divisor: 1 + ln(linked_edge_count). At 1 edge the divisor is
    1.0 (no penalty). At 30 edges it's ~4.4 (the codex-session-#1 case
    seen in production). linked_edge_count is clamped to >=1 in the
    retriever; the max() here is belt-and-suspenders against a stale
    dataclass instance.
    """
    src_mult = SOURCE_SCORE_MULTIPLIERS.get(SourceSystem(source_system), 1.0)
    fanout_div = 1.0 + math.log(max(1, linked_edge_count))
    return raw_score * src_mult / fanout_div


async def _hydrate_inferred_edge_chunks(
    customer_id: str, doc_ids: list[str], per_doc_cap: int
) -> dict[str, list[QueryChunk]]:
    """Fetch top-N body chunks for each inferred-edge-derived doc.

    Without this the chunks list is empty -- the dashboard renders
    "0 matched" and the synthesizer can't cite the doc. The caller
    attaches the returned chunks to each QueryDocumentResult so an
    inferred-edge result is first-class evidence, not a navigation stub.

    Ordering is `chunk_index ASC` -- the first chunks of a doc are
    usually the most identity-bearing (title + opening body for prose;
    metadata sentinel + symbol head for code-graph). The metadata
    sentinel chunk (chunk_index = -1) is excluded so we don't waste a
    slot on the synthetic header.
    """
    if not doc_ids or per_doc_cap <= 0:
        return {}
    sql = """
        WITH ranked AS (
            SELECT c.doc_id, c.chunk_id, c.content, c.chunk_index,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.doc_id ORDER BY c.chunk_index ASC
                   ) AS rn
            FROM chunks c
            WHERE c.customer_id = $1
              AND c.doc_id = ANY($2::text[])
              AND c.valid_to IS NULL
              AND c.chunk_index >= 0
        )
        SELECT doc_id, chunk_id, content, chunk_index, rn
        FROM ranked
        WHERE rn <= $3
        ORDER BY doc_id, rn
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, customer_id, doc_ids, per_doc_cap)

    out: dict[str, list[QueryChunk]] = defaultdict(list)
    for row in rows:
        out[row["doc_id"]].append(
            QueryChunk(
                chunk_id=row["chunk_id"],
                content=row["content"],
                # Inferred-edge chunks didn't participate in retrieval, so
                # there's no per-channel chunk score; tag them with the
                # parent doc's score so downstream consumers (synthesis,
                # dashboard rendering) see consistent ordering.
                score=0.0,  # caller overwrites with parent doc score
                rank_in_doc=int(row["rn"]),
                retriever_scores={"inferred_edge": 0.0},
                graph_evidence=[],
            )
        )
    return out


async def _build_inferred_edge_results(
    customer_id: str,
    document_results: list[QueryDocumentResult],
    requesting_user_id: str | None,
    timing: dict[str, float],
) -> list[QueryDocumentResult]:
    """Walk inferred Doc-Doc edges from the primary docs and wrap the hits
    as QueryDocumentResults with the inferred-edge channel populated.

    Returns documents that are NOT already in the primary set (the SQL
    excludes self-anchors). Each result carries a single MatchProvenance
    entry with channel='inferred_edge', the anchor info, and the LLM `why`.

    Score policy applied here (not in the retriever) so all cross-channel
    scoring lives in one place: `dampening * 1/(1+anchor_rank)` from the
    retriever, then per-source multiplier (codex/CC: 0.5x), then divided
    by `1 + ln(linked_edge_count)` to crush fan-out hubs.

    Body chunks are hydrated post-walk -- without this the dashboard
    renders "0 matched" on every inferred-edge result and the
    synthesizer can't cite from the doc.
    """
    if not document_results:
        return []

    top_doc_ids = [d.doc_id for d in document_results]
    t_inferred = time.perf_counter()
    try:
        hits = await inferred_edge_search(
            customer_id,
            top_doc_ids,
            top_k=INFERRED_EDGE_TOP_K,
            dampening=INFERRED_EDGE_DAMPENING,
        )
    except Exception as exc:
        # Enrichment channel: never break host search response. Log the
        # failure and return [] so the primary results still flow through.
        log.warning("inferred_edge_search.failed", exc_info=exc)
        timing["inferred_edge_ms"] = (time.perf_counter() - t_inferred) * 1000
        return []
    timing["inferred_edge_ms"] = (time.perf_counter() - t_inferred) * 1000

    if not hits:
        return []

    # ACL filter the inferred-edge hits the same way the primary path does.
    # filter_by_acl is a no-op until ENFORCE_ACL flips on, but the call
    # site keeps the contract honest.
    filtered: list[InferredEdgeHit] = await filter_by_acl(
        customer_id, requesting_user_id, hits
    )
    if not filtered:
        return []

    # Hydrate body chunks for the surviving (post-ACL) doc_ids only --
    # don't waste a SQL round-trip fetching content for docs the caller
    # can't see anyway.
    t_hydrate = time.perf_counter()
    chunks_by_doc = await _hydrate_inferred_edge_chunks(
        customer_id,
        [h.doc_id for h in filtered],
        INFERRED_EDGE_HYDRATION_CHUNKS,
    )
    timing["inferred_edge_hydrate_ms"] = (time.perf_counter() - t_hydrate) * 1000

    out: list[QueryDocumentResult] = []
    for h in filtered:
        final_score = _inferred_edge_final_score(
            raw_score=h.score,
            source_system=h.source_system,
            linked_edge_count=h.linked_edge_count,
        )
        prov = MatchProvenance(
            channel="inferred_edge",
            rank=h.anchor_rank,
            score=final_score,
            anchor_doc_id=h.anchor_doc_id,
            edge_type=h.edge_type,
            confidence=h.confidence,
            why=h.why or None,
        )
        # Stamp each hydrated chunk's score with the parent doc's final
        # score so per-chunk scoring stays consistent with the doc.
        doc_chunks = [
            QueryChunk(
                chunk_id=c.chunk_id,
                content=c.content,
                score=final_score,
                rank_in_doc=c.rank_in_doc,
                retriever_scores={"inferred_edge": final_score},
                graph_evidence=c.graph_evidence,
            )
            for c in chunks_by_doc.get(h.doc_id, [])
        ]
        out.append(
            QueryDocumentResult(
                canonical_id=h.doc_id,
                doc_id=h.doc_id,
                doc_version=h.doc_version,
                source_system=SourceSystem(h.source_system),
                source_url=h.source_url,
                title=h.title,
                author_id=h.author_id,
                created_at=h.created_at,
                updated_at=h.updated_at,
                score=final_score,
                rank=0,  # filled in by the caller after final sort
                matched_via=[prov],
                chunks=doc_chunks,
                chunk_count=len(doc_chunks),
                retriever_scores={"inferred_edge": final_score},
            )
        )
    return out


async def _build_entity_results(
    customer_id: str,
    routed_entities: list[RouterEntity],
    timing: dict[str, float],
) -> list[QueryEntityResult]:
    """Look up routed entities in graph_nodes and build QueryEntityResults.

    Each routed entity that resolves to a (label, canonical_id) graph node
    becomes one QueryEntityResult. We also collect 1-hop attached docs
    (capped at `_ENTITY_ATTACHED_DOC_CAP` ordered by recency) and the total
    1-hop Document count.

    Score is `confidence * log(1 + doc_count)` -- a high-confidence entity
    with many attached docs ranks above a low-confidence entity with few.
    The score scale is tuned to be comparable to QueryDocumentResult.score
    so the final concat-and-sort interleaves the two reasonably.
    """
    if not routed_entities:
        return []

    # Resolve each routed entity to a (label, canonical_id) tuple. Drop
    # entities whose entity_type doesn't map to a NodeLabel -- those have
    # no graph_nodes row to surface.
    resolved: list[tuple[str, str, RouterEntity]] = []
    for e in routed_entities:
        node_label = ROUTER_ENTITY_TO_LABEL.get(e.entity_type.lower())
        if node_label is None:
            continue
        # Skip 'session' entities -- they map to NodeLabel.DOCUMENT and
        # already surface as Document results via id_lookup.
        if node_label.value == "Document":
            continue
        resolved.append((node_label.value, e.canonical_id, e))

    if not resolved:
        return []

    labels = [r[0] for r in resolved]
    canonical_ids = [r[1] for r in resolved]

    t_entity = time.perf_counter()
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            WITH wanted AS (
                SELECT * FROM unnest($2::text[], $3::text[]) AS t(label, canonical_id)
            ),
            entity_nodes AS (
                SELECT gn.node_id, gn.label, gn.canonical_id, gn.properties
                FROM graph_nodes gn
                JOIN wanted w ON w.label = gn.label
                              AND w.canonical_id = gn.canonical_id
                WHERE gn.customer_id = $1
            ),
            attached_from AS (
                SELECT en.node_id AS entity_node_id,
                       ge.edge_type,
                       ge.to_node_id AS doc_node_id
                FROM entity_nodes en
                JOIN graph_edges ge
                  ON ge.customer_id = $1
                 AND ge.from_node_id = en.node_id
                 AND (ge.valid_to IS NULL OR ge.valid_to > now())
            ),
            attached_to AS (
                SELECT en.node_id AS entity_node_id,
                       ge.edge_type,
                       ge.from_node_id AS doc_node_id
                FROM entity_nodes en
                JOIN graph_edges ge
                  ON ge.customer_id = $1
                 AND ge.to_node_id = en.node_id
                 AND (ge.valid_to IS NULL OR ge.valid_to > now())
            ),
            attached_edges AS (
                SELECT * FROM attached_from
                UNION ALL
                SELECT * FROM attached_to
            ),
            entity_doc_attachments AS (
                SELECT ae.entity_node_id,
                       ae.edge_type,
                       gn.canonical_id AS doc_id,
                       d.updated_at
                FROM attached_edges ae
                JOIN graph_nodes gn
                  ON gn.customer_id = $1
                 AND gn.node_id = ae.doc_node_id
                 AND gn.label = 'Document'
                JOIN documents d
                  ON d.customer_id = $1
                 AND d.doc_id = gn.canonical_id
                 AND d.valid_to IS NULL
            ),
            ranked_attachments AS (
                SELECT entity_node_id, doc_id, updated_at,
                       ROW_NUMBER() OVER (
                           PARTITION BY entity_node_id
                           ORDER BY updated_at DESC, doc_id ASC
                       ) AS rn
                FROM (
                    SELECT DISTINCT entity_node_id, doc_id, updated_at
                    FROM entity_doc_attachments
                ) AS distinct_attachments
            )
            SELECT en.node_id, en.label, en.canonical_id, en.properties,
                   (SELECT array_agg(DISTINCT eda.edge_type)
                          FILTER (WHERE eda.edge_type IS NOT NULL)
                    FROM entity_doc_attachments eda
                    WHERE eda.entity_node_id = en.node_id) AS edge_types,
                   (SELECT COUNT(DISTINCT eda.doc_id)
                    FROM entity_doc_attachments eda
                    WHERE eda.entity_node_id = en.node_id) AS doc_count,
                   (SELECT array_agg(ra.doc_id ORDER BY ra.rn)
                    FROM ranked_attachments ra
                    WHERE ra.entity_node_id = en.node_id
                      AND ra.rn <= $4) AS attached_doc_pool
            FROM entity_nodes en
            ORDER BY en.label, en.canonical_id
            """,
            customer_id, labels, canonical_ids, _ENTITY_ATTACHED_DOC_CAP,
        )
    timing["entity_result_ms"] = (time.perf_counter() - t_entity) * 1000

    # Map (label, canonical_id) -> RouterEntity for confidence lookup.
    confidence_by_key: dict[tuple[str, str], float] = {}
    for label, cid, entity in resolved:
        key = (label, cid)
        # If the same (label, cid) shows up under multiple routed entities,
        # take the highest confidence.
        confidence_by_key[key] = max(
            confidence_by_key.get(key, 0.0), float(entity.confidence)
        )

    out: list[QueryEntityResult] = []
    for r in rows:
        label = r["label"]
        canonical_id = r["canonical_id"]
        key = (label, canonical_id)
        confidence = confidence_by_key.get(key, 1.0)
        properties = r["properties"]
        if isinstance(properties, str):
            # asyncpg sometimes returns JSONB as a string when no codec is
            # registered. Decode best-effort.
            import json
            try:
                properties = json.loads(properties)
            except (TypeError, ValueError):
                properties = {}
        if not isinstance(properties, dict):
            properties = {}
        display_name = properties.get("name") if isinstance(properties.get("name"), str) else None

        edge_types = list(r["edge_types"] or [])
        doc_count = int(r["doc_count"] or 0)
        attached = list(r["attached_doc_pool"] or [])

        # Score: confidence * log(1 + doc_count). The log keeps high-degree
        # entities from completely dominating; the confidence factor lets
        # the router signal trickle through.
        score = confidence * math.log1p(doc_count) if doc_count > 0 else confidence * 0.5

        out.append(
            QueryEntityResult(
                canonical_id=canonical_id,
                label=label,
                display_name=display_name,
                properties=properties,
                attached_doc_ids=attached,
                edge_types=edge_types,
                doc_count=doc_count,
                score=score,
                rank=0,  # filled in by the caller after final sort
                matched_via=[
                    MatchProvenance(channel="graph", rank=1, score=confidence)
                ],
            )
        )
    return out


def _final_rank(results: list[QueryResult]) -> list[QueryResult]:
    """Sort by score desc with tie-break, then assign 1-indexed `rank`."""

    def _sort_key(r: QueryResult) -> tuple[float, str]:
        return (-r.score, r.canonical_id)

    sorted_results = sorted(results, key=_sort_key)
    # Pydantic models: assigning to `.rank` mutates in place via Pydantic
    # v2's standard attribute setter. The discriminator fields stay intact.
    for i, r in enumerate(sorted_results, start=1):
        r.rank = i
    return sorted_results


async def run_search(
    req: QueryRequest,
    customer_id: str,
    routed: RouterOutput,
    spec: TemporalSpec,
    temporal_meta: dict[str, object],
    sort_meta: dict[str, object] | None,
    extracted_entities: list[dict[str, object]],
    doc_types: list[str] | None,
    trace_id: str,
    timing: dict[str, float],
) -> QueryResponse:
    """Run the existing semantic pipeline. Caller has already authenticated,
    resolved customer_id, and called the router."""
    sources = [s.value for s in req.sources] if req.sources else None
    pool_multiplier = 2  # No widening even when sort intent fires; recency
    # boost handles "give me recent X about Y" without needing to widen.
    queries = [req.query, *routed.expansions]

    # On the search path, doc_type is a SOFT signal. We pass None to the
    # retrievers (preserving recall) — the entity filter or RRF boost
    # picks it up downstream if the user toggled `entity_must_match`.
    retriever_doc_types: list[str] | None = None

    async def _vec_runner() -> list:
        return await vector_search(
            customer_id,
            req.query,
            top_k=req.top_k * pool_multiplier,
            sources=sources,
            doc_types=retriever_doc_types,
            temporal=spec,
        )

    async def _bm25_runner() -> list:
        # Fan out the query + router expansions in parallel — each is an
        # independent SELECT against the chunks GIN index, so wall time
        # collapses to the slowest single call instead of summing N round
        # trips. The per-chunk dedup keeps the highest-scoring hit when an
        # expansion surfaces a chunk the original query already matched.
        #
        # When the router extracts a stable identifier (UUID/ticket/PR ref)
        # id_lookup already pins the doc — strip identifier tokens and
        # identifier-frame descriptors from each BM25 query and skip the
        # SQL pass when nothing topical remains. See residualize_for_bm25
        # for the rationale (low-IDF descriptors otherwise drag 10k+
        # unrelated chunks through the heap recheck + ts_rank_cd).
        identifier_canonical_ids = [
            e.canonical_id
            for e in routed.entities
            if is_lookup_candidate(e.canonical_id)
        ]
        bm25_queries: list[str]
        if identifier_canonical_ids:
            bm25_queries = [
                r
                for r in (
                    residualize_for_bm25(q, identifier_canonical_ids) for q in queries
                )
                if r is not None
            ]
        else:
            bm25_queries = list(queries)

        if not bm25_queries:
            return []

        per_query_hits = await asyncio.gather(
            *(
                bm25_search(
                    customer_id,
                    q,
                    top_k=req.top_k * pool_multiplier,
                    sources=sources,
                    doc_types=retriever_doc_types,
                    temporal=spec,
                )
                for q in bm25_queries
            )
        )
        hits_by_chunk: dict = {}
        for hits in per_query_hits:
            for hit in hits:
                prior = hits_by_chunk.get(hit.chunk_id)
                if prior is None or hit.score > prior.score:
                    hits_by_chunk[hit.chunk_id] = hit
        return sorted(hits_by_chunk.values(), key=lambda h: h.score, reverse=True)

    async def _graph_runner() -> list:
        if not routed.entities:
            return []
        return await graph_search(
            customer_id,
            [(e.entity_type, e.canonical_id) for e in routed.entities],
            doc_types=retriever_doc_types,
            temporal=spec,
            min_confidence=req.min_confidence,
        )

    async def _id_lookup_runner() -> list:
        # Pin docs whose source_id/doc_id matches a router-extracted stable
        # identifier (UUID, ticket code, PR ref). Vector and BM25 both miss
        # exact-id queries — see retrievers/id_lookup.py for the rationale.
        ids = [
            e.canonical_id for e in routed.entities if is_lookup_candidate(e.canonical_id)
        ]
        if not ids:
            return []
        return await id_lookup_search(customer_id, ids, temporal=spec)

    async def _directed_runner() -> list:
        # Per-doc trigger phrases: surfaces wiki pages whose engineer-pinned
        # or LLM-generated phrase matches the user's query, even when the
        # page body itself doesn't. Doc-level booster (no chunk injected) —
        # see retrievers/directed.py + fusion.py. Always-on; the
        # DIRECTED_RETRIEVAL_WEIGHT constant (or an empty
        # directed_vectors table) is the kill switch.
        return await directed_search(
            customer_id,
            req.query,
            top_k=req.top_k * pool_multiplier,
            temporal=spec,
        )

    t_retrieve = time.perf_counter()
    vec_hits, bm25_hits, graph_hits, id_hits, directed_hits = await asyncio.gather(
        _vec_runner(),
        _bm25_runner(),
        _graph_runner(),
        _id_lookup_runner(),
        _directed_runner(),
    )
    ranked_lists = {
        "vector": vec_hits,
        "bm25": bm25_hits,
        "graph": graph_hits,
        "id_lookup": id_hits,
    }
    metadata_content_fallbacks = await _content_fallbacks_for_metadata_only_agent_hits(
        customer_id, ranked_lists, spec
    )
    timing["vector_ms"] = (time.perf_counter() - t_retrieve) * 1000
    # No separate directed_ms metric: all retrievers share one
    # asyncio.gather span, so any per-retriever number reported here
    # would be byte-identical to vector_ms. Aliasing is misleading
    # telemetry. If per-retriever timing becomes necessary, wrap each
    # runner in its own time.perf_counter and emit real spans.

    # Recency half-life: caller's explicit value always wins. Otherwise
    # amplify when Haiku detected sort intent (the "most recent X about Y"
    # case). When neither applies we hand fusion None and it falls back to
    # DEFAULT_RECENCY_HALF_LIFE_DAYS — the universal baseline keeps stale
    # backfilled content from surfacing at parity with fresh docs.
    if req.recency_half_life_days is not None:
        effective_half_life: float | None = req.recency_half_life_days
    elif routed.sort:
        # No caller default and Haiku saw sort intent — bias toward recent.
        effective_half_life = _SORT_INTENT_MIN_HALF_LIFE_DAYS
    else:
        effective_half_life = None

    t_fuse = time.perf_counter()
    fused = fuse(
        {
            **ranked_lists,
            "metadata_content_fallback": metadata_content_fallbacks,
        },
        top_k=req.top_k * pool_multiplier,
        recency_half_life_days=effective_half_life,
        now=datetime.now(UTC),
        sort=None,  # Hard post-fusion sort removed from search path — see
        # module docstring. Recency boost above is the right tool.
        discovery=req.discovery,
        directed_hits=directed_hits,
    )
    timing["fusion_ms"] = (time.perf_counter() - t_fuse) * 1000

    # Inject id_lookup-matched docs that didn't survive fuse()'s top_k cap
    # so they reach dedupe/ACL/pin even at MCP defaults (top_k=5 → pool 10).
    # Without this, the source-multiplier-discounted exact-id match can land
    # at rank 36+ pre-cap and never enter the candidate set the pin sees.
    fused = _inject_id_lookup_hits(fused, id_hits)

    applied_entity_filter: dict[str, object] | None = None
    if req.entity_must_match:
        pre_count = len(fused)
        fused, applied_entity_filter = apply_entity_filter(
            fused, routed.entities, threshold=req.entity_match_threshold
        )
        applied_entity_filter["candidates_before"] = pre_count
        applied_entity_filter["candidates_after"] = len(fused)

    t_dedup = time.perf_counter()
    # Dedup keys on the doc's representative chunk_id (best chunk by RRF).
    # Two docs whose top chunks are near-duplicate embeddings collapse to
    # the higher-ranked doc — same semantics as before, just at doc level.
    embeddings = await embeddings_for_chunks(
        customer_id, [d.chunk_id for d in fused if d.chunk_id]
    )
    deduped = dedupe(fused, embeddings)
    timing["dedup_ms"] = (time.perf_counter() - t_dedup) * 1000

    t_acl = time.perf_counter()
    filtered = await filter_by_acl(customer_id, req.requesting_user_id, deduped)
    timing["acl_ms"] = (time.perf_counter() - t_acl) * 1000

    # Pin exact-id matches on top of the ACL-filtered list so an extracted
    # canonical_id (UUID, ticket code, PR ref) always wins rank 1 — the
    # fusion source-multiplier + recency decay otherwise demote sessions
    # below code_graph noise. Must run AFTER ACL so we don't surface a
    # doc the requesting user can't see.
    filtered = _pin_id_lookup_matches(filtered, id_hits)

    top: list[FusedDocument] = filtered[: req.top_k]

    # Group fused chunks into per-doc results. The doc-grouping branch
    # (feat/doc-grouped-retrieval) enriches the chunk-list aggregation;
    # for now we group chunks already present in the fused output.
    document_results = _build_document_results_from_fused(
        top, ranked_lists, graph_hits
    )

    # Primary docs feed the inferred-edge channel anchors. document_results
    # is already <= top_k since `top` was capped above.
    primary_documents = document_results

    # Inferred-edge channel: walk Doc-Doc INFERRED edges from the top
    # primary docs. Returns Documents not already in primary_documents.
    inferred_documents = await _build_inferred_edge_results(
        customer_id, primary_documents, req.requesting_user_id, timing
    )

    # Entity results: surface routed entities as primary results so the
    # consumer can see "the user asked about Service:foo" alongside the
    # docs about it.
    entity_results = await _build_entity_results(
        customer_id, list(routed.entities), timing
    )

    # related_entities walk (post-fusion crawl candidates) -- separate
    # field from the primary results. Same shape as before.
    related: list[RelatedEntity] | None = None
    related_error: str | None = None
    if req.top_k_related > 0:
        exclude_keys = build_exclude_node_keys(
            routed.entities,
            entity_match_threshold=req.entity_match_threshold,
        )
        ranked_docs = [(d.doc_id, i) for i, d in enumerate(top, start=1)]
        t_related = time.perf_counter()
        try:
            related = await walk_result_doc_neighbors(
                customer_id,
                ranked_result_docs=ranked_docs,
                exclude_node_keys=exclude_keys,
                min_confidence=req.min_confidence,
                top_n=req.top_k_related,
            )
        except Exception as exc:
            # Enrichment field -- never break host search response. Log +
            # surface error on the response so MCP/LLM can distinguish
            # "broken" from "no neighbors" (codex-B4). related stays None;
            # error name flows back via the dedicated response field, NOT
            # via timing_ms (which dashboards read as durations).
            log.warning(
                "related_entities walk failed", exc_info=exc, trace_id=trace_id
            )
            related = None
            related_error = type(exc).__name__
        timing["related_entities_ms"] = (time.perf_counter() - t_related) * 1000

    # Concat primary docs + inferred-edge docs + entities, sort by score.
    all_results: list[QueryResult] = [
        *primary_documents,
        *inferred_documents,
        *entity_results,
    ]
    final_results = _final_rank(all_results)

    # Aggregate graph_evidence confidence tiers across every chunk in every
    # Document result. Entity results have no chunks so contribute nothing.
    confidence_breakdown = {"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0}
    for r in final_results:
        if not isinstance(r, QueryDocumentResult):
            continue
        for chunk in r.chunks:
            for ev in chunk.graph_evidence:
                tier = ev.confidence
                confidence_breakdown[tier] = confidence_breakdown.get(tier, 0) + 1

    return QueryResponse(
        query=req.query,
        results=final_results,
        total_candidates=len(fused),
        router_hit_cache=False,
        confidence_breakdown=confidence_breakdown,
        applied_temporal=temporal_meta,
        applied_sort=sort_meta,
        applied_entity_filter=applied_entity_filter,
        applied_mode="search",
        applied_doc_types=doc_types,
        applied_min_confidence=req.min_confidence,
        extracted_entities=extracted_entities,
        aggregation=None,
        timing_ms=timing,
        trace_id=trace_id,
        related_entities=related,
        related_entities_error=related_error,
    )
