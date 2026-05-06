"""Graph retriever: entity → 1-hop neighbor documents.

Takes router-extracted entities (typed by label + canonical_id) and returns
chunks from documents attached to nodes within 1 hop. Uses the relational
graph tables + RLS tenant isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from services.retrieval.temporal import build_predicate
from shared.constants import TOP_K_GRAPH
from shared.db import with_tenant
from shared.models import TemporalSpec, normalize_author_id


@dataclass(slots=True)
class GraphHit:
    chunk_id: str
    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    content: str
    created_at: datetime
    updated_at: datetime
    score: float
    via_entity: str  # canonical_id that anchored this hit
    author_id: str | None = None
    # Always 'content' for graph hits (graph_search filters synthetic
    # metadata chunks out — graph anchoring is about entity proximity,
    # not synthetic-text similarity).
    kind: str = "content"
    # Edge metadata threaded through from the join to graph_edges so
    # min_confidence filtering and downstream bundle construction
    # don't need a second SQL pass. (PR-A)
    edge_type: str | None = None
    confidence: str | None = None
    via_label: str | None = None


# Confidence tier ordering — higher rank = stronger. Mirrors graph_writer.
_CONFIDENCE_RANK: dict[str, int] = {"AMBIGUOUS": 0, "INFERRED": 1, "EXTRACTED": 2}


def passes_confidence_filter(confidence: str | None, min_confidence: str | None) -> bool:
    """Return True if `confidence` clears the `min_confidence` floor.

    `min_confidence` semantics:
        None        — accept all tiers (debug mode)
        EXTRACTED   — only deterministic edges
        INFERRED    — drops AMBIGUOUS (default for MCP consumers)
        AMBIGUOUS   — accepts everything (synonym for None at the floor)

    Edges whose source predates the confidence column come back as None
    from the DB; treat as EXTRACTED (the migration's default).
    """
    if min_confidence is None:
        return True
    effective = confidence or "EXTRACTED"
    return _CONFIDENCE_RANK.get(effective, 0) >= _CONFIDENCE_RANK.get(min_confidence, 0)


_ENTITY_TO_LABEL = {
    "service": "Service",
    "repo": "Repo",
    "person": "Person",
    "ticket": "Ticket",
    "pr": "PR",
    "error_group": "ErrorGroup",
    "channel": "Channel",
    "feature": "Feature",
    "decision": "Decision",
    "file_path": "Repo",  # fall-back: file paths live under a repo node
    # Code-graph PR-A: symbol entity types map to their respective NodeLabels.
    # Router can extract these directly from entity-bag queries that include
    # qualified names like `Normalizer.process_queue_row`.
    "function": "Function",
    "method": "Method",
    "class": "Class",
    "module": "Module",
    "symbol": "Symbol",
}

# Code-graph node labels — used by the bundle builder to detect when a
# query seeded on a symbol so it can group results under that seed.
CODE_GRAPH_LABELS = frozenset({"Function", "Method", "Class", "Module", "Symbol"})


async def graph_search(
    customer_id: str,
    entities: list[tuple[str, str]],  # (entity_type, canonical_id)
    top_k: int = TOP_K_GRAPH,
    doc_types: list[str] | None = None,
    temporal: TemporalSpec | None = None,
    min_confidence: str | None = "INFERRED",
) -> list[GraphHit]:
    """Return chunks from documents within 1 hop of any matching entity node.

    Scoring is flat (1.0) — callers normalize. The value this retriever adds
    is recall for entity-qualified queries where vector/BM25 miss the
    less-obviously-relevant docs (sibling tickets, co-owned services, etc.).

    `doc_types`, when set, hard-filters joined documents by `doc_type`.
    """
    if not entities:
        return []

    resolved = []
    for etype, cid in entities:
        label = _ENTITY_TO_LABEL.get(etype.lower())
        if label:
            resolved.append((label, cid))
    if not resolved:
        return []

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        labels = [r[0] for r in resolved]
        cids = [r[1] for r in resolved]
        params: list = [customer_id, labels, cids, top_k]

        doc_type_filter = ""
        if doc_types:
            params.append(doc_types)
            doc_type_filter = f"AND d.doc_type = ANY(${len(params)}::text[])"

        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

        rows = await conn.fetch(
            f"""
            WITH anchors AS (
                SELECT node_id, canonical_id, label
                FROM graph_nodes
                WHERE customer_id = $1
                  AND label = ANY($2::text[])
                  AND canonical_id = ANY($3::text[])
            ),
            neighbors AS (
                SELECT DISTINCT n.node_id,
                                a.canonical_id AS via,
                                a.label        AS via_label,
                                e.edge_type    AS edge_type,
                                e.confidence   AS confidence
                FROM anchors a
                JOIN graph_edges e
                  ON e.customer_id = $1
                 AND (e.from_node_id = a.node_id OR e.to_node_id = a.node_id)
                JOIN graph_nodes n
                  ON n.node_id = CASE WHEN e.from_node_id = a.node_id
                                      THEN e.to_node_id ELSE e.from_node_id END
                 AND n.label = 'Document'
                UNION
                SELECT node_id,
                       canonical_id AS via,
                       label        AS via_label,
                       NULL         AS edge_type,
                       NULL         AS confidence
                FROM anchors
                  WHERE EXISTS (SELECT 1 FROM graph_nodes gn
                                WHERE gn.node_id = anchors.node_id AND gn.label = 'Document')
            )
            SELECT c.chunk_id, c.doc_id, d.version AS doc_version,
                   d.source_system, d.source_url, d.title, d.author_id,
                   c.content, d.created_at, d.updated_at,
                   MIN(n.via)        AS via_entity,
                   MIN(n.via_label)  AS via_label,
                   MIN(n.edge_type)  AS edge_type,
                   MIN(n.confidence) AS confidence
            FROM neighbors n
            JOIN graph_nodes gn ON gn.node_id = n.node_id
            JOIN documents d
              ON d.doc_id = gn.canonical_id
             AND d.customer_id = $1
            JOIN chunks c
              ON c.doc_id = d.doc_id
             AND c.customer_id = $1
             AND c.kind = 'content'
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE 1 = 1
              {pred.chunk_sql}
              {pred.doc_sql}
              {doc_type_filter}
            GROUP BY c.chunk_id, c.doc_id, d.version,
                     d.source_system, d.source_url, d.title, d.author_id,
                     c.content, d.created_at, d.updated_at
            LIMIT $4
            """,
            *params,
        )

    hits: list[GraphHit] = []
    for r in rows:
        confidence = r["confidence"]
        if not passes_confidence_filter(confidence, min_confidence):
            continue
        hits.append(
            GraphHit(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                doc_version=r["doc_version"],
                source_system=r["source_system"],
                source_url=r["source_url"],
                title=r["title"],
                content=r["content"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                score=1.0,
                via_entity=r["via_entity"],
                author_id=normalize_author_id(r["author_id"]),
                edge_type=r["edge_type"],
                confidence=confidence,
                via_label=r["via_label"],
            )
        )
    return hits
