"""Graph retriever: entity → 1-hop neighbor documents.

Takes router-extracted entities (typed by label + canonical_id) and returns
chunks from documents attached to nodes within 1 hop. Uses the relational
graph tables + RLS tenant isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

from services.retrieval.temporal import build_predicate
from shared.constants import TOP_K_GRAPH
from shared.db import with_tenant
from shared.models import TemporalSpec


@dataclass(slots=True)
class GraphHit:
    chunk_id: str
    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    content: str
    score: float
    via_entity: str  # canonical_id that anchored this hit


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
}


async def graph_search(
    customer_id: str,
    entities: list[tuple[str, str]],  # (entity_type, canonical_id)
    top_k: int = TOP_K_GRAPH,
    temporal: TemporalSpec | None = None,
) -> list[GraphHit]:
    """Return chunks from documents within 1 hop of any matching entity node.

    Scoring is flat (1.0) — callers normalize. The value this retriever adds
    is recall for entity-qualified queries where vector/BM25 miss the
    less-obviously-relevant docs (sibling tickets, co-owned services, etc.).
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
        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

        rows = await conn.fetch(
            f"""
            WITH anchors AS (
                SELECT node_id, canonical_id
                FROM graph_nodes
                WHERE customer_id = $1
                  AND label = ANY($2::text[])
                  AND canonical_id = ANY($3::text[])
            ),
            neighbors AS (
                SELECT DISTINCT n.node_id, a.canonical_id AS via
                FROM anchors a
                JOIN graph_edges e
                  ON e.customer_id = $1
                 AND (e.from_node_id = a.node_id OR e.to_node_id = a.node_id)
                JOIN graph_nodes n
                  ON n.node_id = CASE WHEN e.from_node_id = a.node_id
                                      THEN e.to_node_id ELSE e.from_node_id END
                 AND n.label = 'Document'
                UNION
                SELECT node_id, canonical_id AS via FROM anchors
                  WHERE EXISTS (SELECT 1 FROM graph_nodes gn
                                WHERE gn.node_id = anchors.node_id AND gn.label = 'Document')
            )
            SELECT c.chunk_id, c.doc_id, d.version AS doc_version,
                   d.source_system, d.source_url, d.title, c.content,
                   MIN(n.via) AS via_entity
            FROM neighbors n
            JOIN graph_nodes gn ON gn.node_id = n.node_id
            JOIN documents d
              ON d.doc_id = gn.canonical_id
             AND d.customer_id = $1
            JOIN chunks c
              ON c.doc_id = d.doc_id
             AND c.customer_id = $1
             AND d.version BETWEEN c.first_seen_version AND c.last_seen_version
            WHERE 1 = 1
              {pred.chunk_sql}
              {pred.doc_sql}
            GROUP BY c.chunk_id, c.doc_id, d.version,
                     d.source_system, d.source_url, d.title, c.content
            LIMIT $4
            """,
            *params,
        )

    return [
        GraphHit(
            chunk_id=r["chunk_id"],
            doc_id=r["doc_id"],
            doc_version=r["doc_version"],
            source_system=r["source_system"],
            source_url=r["source_url"],
            title=r["title"],
            content=r["content"],
            score=1.0,
            via_entity=r["via_entity"],
        )
        for r in rows
    ]
