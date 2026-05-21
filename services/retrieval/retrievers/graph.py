"""Graph retriever: entity -> 1-hop neighbor documents.

Takes router-extracted entities (typed by label + canonical_id) and returns
chunks from documents attached to nodes within 1 hop. Uses the relational
graph tables + RLS tenant isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from services.retrieval.surprise import surprise_score
from services.retrieval.temporal import build_predicate
from shared.constants import ROUTER_ENTITY_TO_LABEL, TOP_K_GRAPH
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import TemporalSpec, normalize_author_id

log = get_logger(__name__)

# Surprise score on graph hits is now unconditional. Empirical 4-run A/B on
# probe-founders (3 flag-off baselines + 1 flag-on) showed flipping
# SURPRISE_SCORE_ENABLED produced zero ranking change beyond +/-0.001 jitter,
# because graph's RRF contribution is bounded (1/(60+rank)) and flat
# score=1.0 made the rank order arbitrary heap-scan order anyway.
# Always computing + sorting by surprise score is strictly better than
# heap-scan order regardless of how callers consume the score downstream.
# Discovery mode (QueryRequest.discovery) controls whether fusion uses the
# score in RRF math; this module just guarantees deterministic, signal-driven
# rank ordering for any consumer.


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
    # Surprise-score telemetry. Always set (even when flag=false) so we can
    # compare would-be scores via logs without flipping the flag in prod.
    retriever_scores: dict[str, float] | None = None


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


# Backwards-compat alias. `_ENTITY_TO_LABEL` was the historical name; the
# real source of truth now lives in shared.constants alongside NodeLabel.
# Keep the alias so any out-of-tree caller importing the old symbol still
# works, but every in-tree call uses ROUTER_ENTITY_TO_LABEL directly.
_ENTITY_TO_LABEL = {k: v.value for k, v in ROUTER_ENTITY_TO_LABEL.items()}

# Code-graph node labels — used by the bundle builder to detect when a
# query seeded on a symbol so it can group results under that seed.
# Post-migration 0091 the fine-grained Function/Method/Class/Module/Symbol
# labels all collapse to CodeSymbol; this set is now a singleton.
CODE_GRAPH_LABELS = frozenset({"CodeSymbol"})


async def graph_search(
    customer_id: str,
    entities: list[tuple[str, str]],  # (entity_type, canonical_id)
    top_k: int = TOP_K_GRAPH,
    doc_types: list[str] | None = None,
    temporal: TemporalSpec | None = None,
    min_confidence: str | None = "INFERRED",
    include_drafts: bool = False,
) -> list[GraphHit]:
    """Return chunks from documents within 1 hop of any matching entity node.

    Scoring is flat (1.0) — callers normalize. The value this retriever adds
    is recall for entity-qualified queries where vector/BM25 miss the
    less-obviously-relevant docs (sibling tickets, co-owned services, etc.).

    `doc_types`, when set, hard-filters joined documents by `doc_type`.
    """
    if not entities:
        return []

    # ---- Entity -> label resolution + unknown-type fallback ---------------
    # Two paths:
    #   1) Typed: router's entity_type IS in ROUTER_ENTITY_TO_LABEL ->
    #      look up by (label, canonical_id). Same shape as before.
    #   2) Fallback: entity_type is NOT in the map (router added a new
    #      type, dict drift) -> log a warning AND look up by canonical_id
    #      alone (any label) so the graph hit still surfaces. The first
    #      time an unknown type queries, the warning makes the drift
    #      visible in logs; previously the entity was silently dropped
    #      and the graph retriever returned 0 hits with no signal.
    resolved: list[tuple[str, str]] = []
    fallback_cids: list[str] = []
    unknown_types: set[str] = set()
    for etype, cid in entities:
        node_label = ROUTER_ENTITY_TO_LABEL.get(etype.lower())
        if node_label is not None:
            resolved.append((node_label.value, cid))
        else:
            fallback_cids.append(cid)
            unknown_types.add(etype)

    if unknown_types:
        log.warning(
            "graph.unknown_entity_types_fallback",
            customer=customer_id,
            unknown_types=sorted(unknown_types),
            fallback_cid_count=len(fallback_cids),
            fix_hint=(
                "Add the type to ROUTER_ENTITY_TO_LABEL in shared/constants.py"
            ),
        )

    if not resolved and not fallback_cids:
        return []

    spec = temporal or TemporalSpec()

    async with with_tenant(customer_id) as conn:
        labels = [r[0] for r in resolved]
        cids = [r[1] for r in resolved]
        # $2 = labels, $3 = typed cids, $4 = top_k, $5 = fallback cids.
        # Empty arrays are valid -- the SQL ANY filters degenerate to
        # "match nothing" for that branch, but the UNION's other branch
        # still runs.
        params: list = [customer_id, labels, cids, top_k, fallback_cids]

        doc_type_filter = ""
        if doc_types:
            params.append(doc_types)
            doc_type_filter = f"AND d.doc_type = ANY(${len(params)}::text[])"

        pred = build_predicate(
            spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1
        )
        params.extend(pred.params)

        # Hide drafts by default (Plan A Component 6); reviewer surfaces
        # opt in via include_drafts=True.
        visibility_filter = (
            ""
            if include_drafts
            else "AND c.visibility = 'approved' AND d.visibility = 'approved'"
        )

        rows = await conn.fetch(
            f"""
            WITH anchors AS (
                -- Typed lookup: router resolved entity_type -> NodeLabel via
                -- ROUTER_ENTITY_TO_LABEL. Match by (label, canonical_id).
                SELECT a.node_id,
                       a.canonical_id,
                       a.label,
                       a.degree        AS via_degree,
                       a.community_id  AS via_community,
                       -- Best-effort source_system for the anchor. Anchors are
                       -- non-document nodes (Service, Repo, Person, etc.) whose
                       -- provenance is tracked in graph_node_provenance. Take
                       -- MIN to get a deterministic value when a node has been
                       -- asserted by multiple connectors.
                       (SELECT MIN(p.source_system)
                        FROM graph_node_provenance p
                        WHERE p.node_id = a.node_id) AS via_source_system
                FROM graph_nodes a
                WHERE a.customer_id = $1
                  AND a.label = ANY($2::text[])
                  AND a.canonical_id = ANY($3::text[])
                UNION
                -- Fallback: entity_type was unknown to ROUTER_ENTITY_TO_LABEL
                -- (router added a new type the graph retriever hasn't been
                -- updated for). Match by canonical_id alone -- the DB tells
                -- us the label. Empty $5 (no unknowns) makes this branch
                -- match nothing. The UNION dedupes any node that matches
                -- both branches.
                SELECT a.node_id,
                       a.canonical_id,
                       a.label,
                       a.degree        AS via_degree,
                       a.community_id  AS via_community,
                       (SELECT MIN(p.source_system)
                        FROM graph_node_provenance p
                        WHERE p.node_id = a.node_id) AS via_source_system
                FROM graph_nodes a
                WHERE a.customer_id = $1
                  AND a.canonical_id = ANY($5::text[])
            ),
            neighbors AS (
                SELECT DISTINCT n.node_id,
                                a.canonical_id       AS via,
                                a.label              AS via_label,
                                e.edge_type          AS edge_type,
                                e.confidence         AS confidence,
                                a.via_degree         AS via_degree,
                                a.via_community      AS via_community,
                                a.via_source_system  AS via_source_system,
                                n.degree             AS degree,
                                n.community_id       AS community_id
                FROM anchors a
                JOIN graph_edges e
                  ON e.customer_id = $1
                 AND (e.from_node_id = a.node_id OR e.to_node_id = a.node_id)
                JOIN graph_nodes n
                  ON n.node_id = CASE WHEN e.from_node_id = a.node_id
                                      THEN e.to_node_id ELSE e.from_node_id END
                 AND n.label = 'Document'
                UNION
                -- When the anchor IS itself a Document node, the "neighbor"
                -- and the anchor are the same row. Both via_degree/community
                -- and degree/community_id come from the same node, so reuse
                -- the anchors CTE's pre-aliased values for both sides.
                SELECT node_id,
                       canonical_id    AS via,
                       label           AS via_label,
                       NULL            AS edge_type,
                       NULL            AS confidence,
                       via_degree,
                       via_community,
                       via_source_system,
                       via_degree      AS degree,
                       via_community   AS community_id
                FROM anchors
                  WHERE EXISTS (SELECT 1 FROM graph_nodes gn
                                WHERE gn.node_id = anchors.node_id AND gn.label = 'Document')
            )
            -- One row per neighbor doc, picking chunk_index=0 as the
            -- representative chunk. Without this cap, a giant anchor doc
            -- (e.g. a claude_code session with 200+ chunks) crowds out
            -- every other neighbor: the LIMIT $4 fills with chunks of one
            -- doc before a single chunk of any other neighbor makes it
            -- through. The graph retriever's natural unit is "doc reachable
            -- via this entity", not "best-matching chunk passage" -- that
            -- semantic mismatch is what lets one doc swallow the budget.
            -- ROW_NUMBER() PARTITION BY doc_id picks one chunk per doc;
            -- ORDER BY chunk_index makes the choice deterministic
            -- (chunk 0 is the doc's first content chunk, which carries the
            -- title / opening summary across every source_system shape).
            SELECT chunk_id, doc_id, doc_version,
                   source_system, source_url, title, author_id,
                   content, created_at, updated_at,
                   via_entity, via_label, edge_type, confidence,
                   via_degree, via_community, via_source_system,
                   degree, community_id
            FROM (
              SELECT c.chunk_id, c.doc_id, d.version AS doc_version,
                   d.source_system, d.source_url, d.title, d.author_id,
                   c.content, d.created_at, d.updated_at,
                   MIN(n.via)               AS via_entity,
                   MIN(n.via_label)         AS via_label,
                   MIN(n.edge_type)         AS edge_type,
                   MIN(n.confidence)        AS confidence,
                   MIN(n.via_degree)        AS via_degree,
                   MIN(n.via_community)     AS via_community,
                   MIN(n.via_source_system) AS via_source_system,
                   MIN(n.degree)            AS degree,
                   MIN(n.community_id)      AS community_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY c.doc_id
                       ORDER BY c.chunk_index ASC, c.chunk_id ASC
                   ) AS rn_in_doc
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
              {visibility_filter}
            GROUP BY c.chunk_id, c.doc_id, c.chunk_index, d.version,
                     d.source_system, d.source_url, d.title, d.author_id,
                     c.content, d.created_at, d.updated_at
            ) AS chunks_with_rn
            WHERE rn_in_doc = 1
            LIMIT $4
            """,
            *params,
        )

    hits: list[GraphHit] = []
    for r in rows:
        confidence = r["confidence"]
        if not passes_confidence_filter(confidence, min_confidence):
            continue

        # Surprise score is always computed and used as hit.score so the
        # rank order returned to fusion is deterministic + signal-driven
        # (highest-surprise edge at rank 1). Whether fusion AMPLIFIES this
        # contribution in RRF math is gated by QueryRequest.discovery; this
        # retriever just guarantees a consistent input order.
        surprise = surprise_score(
            edge_type=r["edge_type"],
            confidence=confidence,
            anchor_label=r["via_label"],
            target_label="Document",
            anchor_source=r["via_source_system"],
            target_source=r["source_system"],
            anchor_community=r["via_community"],
            target_community=r["community_id"],
            anchor_degree=r["via_degree"] or 0,
            target_degree=r["degree"] or 0,
        )

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
                score=surprise,
                via_entity=r["via_entity"],
                author_id=normalize_author_id(r["author_id"]),
                edge_type=r["edge_type"],
                confidence=confidence,
                via_label=r["via_label"],
                retriever_scores={"surprise": surprise},
            )
        )

    # Sort by surprise score so the highest-surprise edge lands at rank 1
    # and gets the biggest graph-side RRF contribution (1/61) in fusion.
    # Without this sort, hits would arrive in arbitrary heap-scan order.
    # Tie-break by chunk_id so the order is deterministic across runs and
    # MCP retriever_scores telemetry doesn't jitter on identical queries.
    hits.sort(key=lambda h: (-h.score, h.chunk_id))
    return hits
