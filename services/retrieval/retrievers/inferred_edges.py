"""Inferred-edges retriever: walk INFERRED Doc-Doc edges from anchor docs.

Lane B (`services/ingestion/inferred_edges/`) writes Doc-Doc edges with
LLM-derived `properties.why` justifications and `extractor_id =
'inferred_edges:v1'`. This retriever consumes that data: given the top-K
documents from the primary search, it walks their 1-hop INFERRED neighbors
and surfaces them as additional Document candidates.

The walk is bidirectional via UNION ALL on (from_node_id / to_node_id) so
each direction hits its dedicated edge index instead of betting on Postgres
BitmapOr (per memory `feedback_postgres_bidirectional_or_to_union.md`).

Edge-type ordering (DISCUSSES > RESOLVES > DOCUMENTS > MENTIONS_ENTITY >
RELATES_TO) reflects the prior we want for "linked doc relevance": a doc
that DISCUSSES the anchor is more likely to be useful than one that
loosely RELATES_TO it. Within a tier, the freshest doc wins.

Score is dampened RRF-style: `dampening * 1 / (1 + anchor_rank)`. The
caller passes 0.5 by default so an inferred-edge hit caps at half the
score of a top primary result. Multi-anchor hits keep the BEST anchor
(highest primary rank = lowest anchor_rank number) but record all
provenances on the resulting `MatchProvenance` list upstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from shared.constants import NodeLabel
from shared.db import with_tenant
from shared.models import normalize_author_id

# Tag the inferred-edges extractor writes onto each row. Lives here as a
# constant so this retriever doesn't need to import the producer module
# (cross-module-layering hygiene -- ingestion -> retrieval is one-way).
INFERRED_EDGES_EXTRACTOR_ID = "inferred_edges:v1"


@dataclass(slots=True)
class InferredEdgeHit:
    """One Document reached by walking an INFERRED Doc-Doc edge.

    `anchor_doc_id` / `anchor_rank` / `edge_type` / `confidence` / `why`
    flow into the resulting `QueryDocumentResult.matched_via` so the LLM
    consumer can see WHY the doc surfaced (the inferred-edge channel is
    the only one with a free-form justification).
    """

    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    author_id: str | None
    created_at: datetime
    updated_at: datetime
    anchor_doc_id: str
    anchor_rank: int  # 1-indexed rank of anchor in primary results
    edge_type: str
    confidence: str  # always 'INFERRED' in v1
    why: str
    score: float  # dampened RRF-style score


async def inferred_edge_search(
    customer_id: str,
    top_doc_ids: list[str],
    *,
    top_k: int = 5,
    dampening: float = 0.5,
) -> list[InferredEdgeHit]:
    """Walk INFERRED Doc-Doc edges from `top_doc_ids` and return up to
    `top_k` linked documents.

    `top_doc_ids` is ordered by primary rank (rank 1 first). Multi-anchor
    docs (a single neighbor reached from several anchors) collapse to the
    BEST anchor (lowest anchor_rank number) -- caller can layer additional
    provenance on top if it tracks all anchors.

    Excludes any `top_doc_ids` from the returned list (a doc surfacing in
    primary results doesn't need to surface again as its own neighbor).

    Empty `top_doc_ids` short-circuits without a SQL call.
    """
    if not top_doc_ids:
        return []

    document_label = NodeLabel.DOCUMENT.value
    sql = f"""
        WITH anchors AS (
            -- Resolve each top_doc_id to its Document graph_node, carrying
            -- the 1-indexed rank from the primary search forward.
            SELECT gn.node_id, gn.canonical_id AS doc_id,
                   array_position($2::text[], gn.canonical_id) AS anchor_rank
            FROM graph_nodes gn
            WHERE gn.customer_id = $1
              AND gn.label = '{document_label}'
              AND gn.canonical_id = ANY($2::text[])
        ),
        candidate_edges AS (
            -- Direction 1: anchor as from_node (uses idx_graph_edges_from).
            SELECT a.doc_id AS anchor_doc_id, a.anchor_rank,
                   ge.to_node_id AS neighbor_node_id,
                   ge.edge_type, ge.confidence, ge.properties->>'why' AS why
            FROM anchors a
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.from_node_id = a.node_id
             AND ge.extractor_id = '{INFERRED_EDGES_EXTRACTOR_ID}'
             AND ge.confidence = 'INFERRED'
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
            UNION ALL
            -- Direction 2: anchor as to_node (uses idx_graph_edges_to).
            SELECT a.doc_id AS anchor_doc_id, a.anchor_rank,
                   ge.from_node_id AS neighbor_node_id,
                   ge.edge_type, ge.confidence, ge.properties->>'why' AS why
            FROM anchors a
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.to_node_id = a.node_id
             AND ge.extractor_id = '{INFERRED_EDGES_EXTRACTOR_ID}'
             AND ge.confidence = 'INFERRED'
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
        ),
        neighbor_docs AS (
            -- Project neighbor_node_id back to a Document canonical_id;
            -- non-Document neighbors fall out (this retriever is Doc-Doc only).
            SELECT ce.anchor_doc_id, ce.anchor_rank, ce.edge_type,
                   ce.confidence, ce.why,
                   gn.canonical_id AS doc_id
            FROM candidate_edges ce
            JOIN graph_nodes gn
              ON gn.customer_id = $1
             AND gn.node_id = ce.neighbor_node_id
             AND gn.label = '{document_label}'
        )
        SELECT nd.doc_id, nd.anchor_doc_id, nd.anchor_rank,
               nd.edge_type, nd.confidence, nd.why,
               d.version AS doc_version, d.source_system, d.source_url,
               d.title, d.author_id,
               d.created_at, d.updated_at
        FROM neighbor_docs nd
        JOIN documents d
          ON d.customer_id = $1
         AND d.doc_id = nd.doc_id
         AND d.valid_to IS NULL
        WHERE nd.doc_id <> ALL($2::text[])  -- exclude top_doc_ids themselves
        ORDER BY
          CASE nd.edge_type
            WHEN 'DISCUSSES' THEN 1
            WHEN 'RESOLVES' THEN 2
            WHEN 'DOCUMENTS' THEN 3
            WHEN 'MENTIONS_ENTITY' THEN 4
            WHEN 'RELATES_TO' THEN 5
            ELSE 6
          END,
          d.updated_at DESC
        LIMIT $3
    """

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, customer_id, top_doc_ids, top_k)

    # Multi-anchor collapse: if the same neighbor doc surfaces from several
    # anchors, keep the BEST anchor (highest primary rank = lowest
    # anchor_rank number). Iteration order from the SQL is already
    # edge-type-then-recency stable; we reorder on a per-doc basis so the
    # downstream provenance uses the strongest anchor.
    by_doc: dict[str, InferredEdgeHit] = {}
    for r in rows:
        doc_id = r["doc_id"]
        anchor_rank = int(r["anchor_rank"])
        # Score: dampened reciprocal of anchor_rank.
        score = dampening * (1.0 / (1.0 + anchor_rank))
        candidate = InferredEdgeHit(
            doc_id=doc_id,
            doc_version=int(r["doc_version"]),
            source_system=r["source_system"],
            source_url=r["source_url"],
            title=r["title"],
            author_id=normalize_author_id(r["author_id"]),
            created_at=r["created_at"],
            updated_at=r["updated_at"],
            anchor_doc_id=r["anchor_doc_id"],
            anchor_rank=anchor_rank,
            edge_type=r["edge_type"],
            confidence=r["confidence"],
            why=r["why"] or "",
            score=score,
        )
        existing = by_doc.get(doc_id)
        if existing is None or anchor_rank < existing.anchor_rank:
            by_doc[doc_id] = candidate

    # Preserve the SQL's edge-type-priority + recency ordering by walking
    # the rows again in original order. `by_doc` already filtered to the
    # best anchor per doc; emit each doc's first appearance in the SQL
    # output's order.
    seen: set[str] = set()
    out: list[InferredEdgeHit] = []
    for r in rows:
        doc_id = r["doc_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        out.append(by_doc[doc_id])
    return out


__all__ = ["INFERRED_EDGES_EXTRACTOR_ID", "InferredEdgeHit", "inferred_edge_search"]
