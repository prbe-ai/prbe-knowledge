"""Related entities retriever: result-set docs -> 1-hop non-Document neighbors.

Takes the (doc_id, rank) tuples from the top chunks of a search/list response
and walks 1 hop through `graph_edges` to non-Document `graph_nodes`. Returns
ranked crawl candidates for an LLM doing knowledge-graph BFS: pick a high-
score entity, drop its `canonical_id` into the next `search_knowledge` query
bag, repeat.

Mirrors `retrievers/graph.py` -- same RLS/with_tenant pattern, same confidence-
tier ordering. The walk is bidirectional (UNION ALL on `from_node_id` /
`to_node_id`) so it hits the dedicated edge indexes
(idx_graph_edges_from / idx_graph_edges_to) instead of betting on BitmapOr.
"""

from __future__ import annotations

from services.retrieval.retrievers.graph import _CONFIDENCE_RANK, _ENTITY_TO_LABEL
from shared.constants import NodeLabel
from shared.db import with_tenant
from shared.models import RelatedEntity


def _confidence_case_sql(column: str) -> str:
    """Generate a SQL CASE expression mapping confidence tier -> int rank.

    Single-source from `_CONFIDENCE_RANK` so SQL/Python don't drift. The
    fragment is constant per call site; safe to interpolate via f-string.
    Edges whose source predates the confidence column come back as NULL;
    treat as EXTRACTED (the migration's default; mirrors
    `passes_confidence_filter`).
    """
    branches = "\n".join(
        f"        WHEN '{tier}' THEN {rank}"
        for tier, rank in _CONFIDENCE_RANK.items()
    )
    extracted_rank = _CONFIDENCE_RANK["EXTRACTED"]
    return (
        f"CASE {column}\n"
        f"{branches}\n"
        f"        ELSE {extracted_rank}  -- legacy NULL -> EXTRACTED\n"
        f"    END"
    )


# Inverse map: rank int -> tier name. Built once from the canonical
# `_CONFIDENCE_RANK` dict so the SQL CASE generator and the Python wrapper
# share the same source of truth (codex-A1).
_RANK_TO_CONFIDENCE: dict[int, str] = {rank: tier for tier, rank in _CONFIDENCE_RANK.items()}


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    """Return `values` with duplicates removed, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def build_exclude_node_keys(
    routed_entities,
    *,
    entity_match_threshold: float = 0.7,
) -> set[tuple[str, str]]:
    """Build a fuzzy-match exclusion key set from router-extracted entities.

    Mirrors `QueryRequest.entity_match_threshold` semantics: only entities
    above the threshold contribute. For each qualifying entity we emit up
    to four normalized variants per (label, string) under matching label:

      1. lower(canonical_id)
      2. lower(canonical_id) with namespace prefix stripped (regex `^.*/`)
      3. lower(display_name)            -- when display_name is set
      4. lower(display_name) with namespace stripped

    The SQL exclusion compares against the same normalized forms on the
    graph_node side, so router-vs-graph canonical_id mismatches like
    'prbe-backend' (router) vs 'prbe-ai/prbe-backend' (graph) match.

    `routed_entities` is a list of `RouterEntity`-like objects with
    `entity_type`, `canonical_id`, `confidence`, and optionally
    `display_name` attributes.
    """
    import re
    out: set[tuple[str, str]] = set()
    namespace_strip = re.compile(r"^.*/")
    for e in routed_entities:
        confidence = getattr(e, "confidence", 1.0) or 0.0
        if confidence < entity_match_threshold:
            continue
        label = _ENTITY_TO_LABEL.get(e.entity_type.lower())
        if not label:
            continue
        for raw in (
            getattr(e, "canonical_id", None),
            getattr(e, "display_name", None),
        ):
            if not raw:
                continue
            lowered = raw.lower()
            out.add((label, lowered))
            stripped = namespace_strip.sub("", lowered)
            if stripped and stripped != lowered:
                out.add((label, stripped))
    return out


async def walk_result_doc_neighbors(
    customer_id: str,
    # Top chunks' (doc_id, rank) tuples -- rank flows into associated_doc_ids
    # ordering so samples reflect strongest evidence first.
    ranked_result_docs: list[tuple[str, int]],
    # (label, canonical_id) tuples -- graph node identity is two-part per
    # db/schema.sql:488. Excluding by canonical_id alone over-excludes
    # when the same canonical_id appears under multiple labels (codex-A4).
    exclude_node_keys: set[tuple[str, str]],
    min_confidence: str | None = "INFERRED",
    top_n: int = 10,
) -> list[RelatedEntity]:
    """Return up to `top_n` non-Document graph nodes attached to the result docs.

    Scoring is IDF-style: doc_count_in_results / log(1 + global_doc_count_per_neighbor).
    Generic high-degree entities (Channel:#engineering attached to 10k tenant
    docs) get crushed; specific entities (PR:42 attached to 4 docs) surface.
    Sorted by `score DESC, doc_count DESC, max_confidence_rank DESC`.

    Returns `[]` (not None) when `ranked_result_docs` is empty -- the three-
    state contract on `QueryResponse.related_entities` distinguishes that
    legitimate empty from "not requested" / "walk failed" (codex-B4).

    SQL runs under `with_tenant(customer_id)` so the RLS GUC
    `app.current_customer_id` is set (per memory:
    `feedback_prbe_knowledge_rls_guc_name.md`).
    """
    if not ranked_result_docs:
        return []

    # Translate min_confidence tier name -> int rank for SQL HAVING.
    # When None, accept all tiers by gating on rank 0 (the floor).
    min_rank = _CONFIDENCE_RANK.get(min_confidence, 0) if min_confidence else 0

    doc_ids = [doc_id for doc_id, _ in ranked_result_docs]
    ranks = [rank for _, rank in ranked_result_docs]
    # Fuzzy exclusion: each tuple is (label, normalized_string) where the
    # caller has already lowercased the string and emitted variants per
    # entity (raw canonical_id, namespace-stripped, display_name, etc).
    # SQL normalizes the candidate side the same way.
    exclude_labels = [label for label, _ in exclude_node_keys]
    exclude_canonical_ids = [cid for _, cid in exclude_node_keys]

    confidence_case = _confidence_case_sql("ne.confidence")
    confidence_case_e2_from = _confidence_case_sql("e2.confidence")
    confidence_case_e2_to = _confidence_case_sql("e2.confidence")
    document_label = NodeLabel.DOCUMENT.value

    sql = f"""
        WITH doc_ranks AS (
            -- (doc_id, rank) tuples from the top chunks list
            SELECT * FROM unnest($2::text[], $3::int[]) AS t(doc_id, rank)
        ),
        doc_anchors AS (
            SELECT gn.node_id, dr.rank
            FROM doc_ranks dr
            JOIN graph_nodes gn
              ON gn.customer_id = $1
             AND gn.label = '{document_label}'
             AND gn.canonical_id = dr.doc_id
        ),
        exclude_keys AS (
            -- (label, canonical_id) tuples for routed-entity exclusion
            SELECT * FROM unnest($4::text[], $5::text[]) AS t(label, canonical_id)
        ),
        neighbor_edges AS (
            -- direction 1: doc is from_node -- uses idx_graph_edges_from
            SELECT e.edge_type, e.confidence,
                   e.to_node_id AS neighbor_node_id,
                   a.node_id    AS doc_node_id,
                   a.rank       AS doc_rank
            FROM doc_anchors a
            JOIN graph_edges e
              ON e.customer_id = $1
             AND e.from_node_id = a.node_id
             AND (e.valid_to IS NULL OR e.valid_to > now())
            UNION ALL
            -- direction 2: doc is to_node -- uses idx_graph_edges_to
            SELECT e.edge_type, e.confidence,
                   e.from_node_id AS neighbor_node_id,
                   a.node_id      AS doc_node_id,
                   a.rank         AS doc_rank
            FROM doc_anchors a
            JOIN graph_edges e
              ON e.customer_id = $1
             AND e.to_node_id = a.node_id
             AND (e.valid_to IS NULL OR e.valid_to > now())
        ),
        result_aggregates AS (
            SELECT
                gn.canonical_id,
                gn.label,
                COALESCE(NULLIF(ecm.display_name, ''), gn.properties->>'name') AS display_name,
                gn.node_id,
                array_agg(DISTINCT ne.edge_type) AS edge_types,
                max({confidence_case}) AS max_confidence_rank,
                COUNT(DISTINCT doc_gn.canonical_id) AS doc_count,
                -- Rank-ordered samples. Carries duplicates intentionally so
                -- the lowest-rank doc surfaces first; Python dedupes
                -- preserving first-seen order, then truncates to 3 (codex-A2).
                array_agg(doc_gn.canonical_id ORDER BY ne.doc_rank ASC) AS sample_pool,
                -- PHASE 2: cluster size = primary + alias count.
                (1 + COALESCE(ea_count.alias_count, 0))::int AS member_count,
                -- PHASE 2: distinct source_systems from consolidated provenance.
                COALESCE(gnp.sources, ARRAY[]::text[]) AS member_sources
            FROM neighbor_edges ne
            JOIN graph_nodes gn
              ON gn.node_id = ne.neighbor_node_id
             AND gn.customer_id = $1
            JOIN graph_nodes doc_gn
              ON doc_gn.node_id = ne.doc_node_id
             AND doc_gn.customer_id = $1
            -- PHASE 2: per-primary alias count (NULL when no merge happened).
            LEFT JOIN LATERAL (
                SELECT COUNT(*) AS alias_count
                FROM entity_aliases
                WHERE customer_id = $1
                  AND label = gn.label
                  AND primary_canonical_id = gn.canonical_id
            ) ea_count ON TRUE
            -- PHASE 2: distinct source_systems for the primary's node.
            -- LATERAL keyed on node_id (PK-ish) -> one index probe per neighbor.
            LEFT JOIN LATERAL (
                SELECT array_agg(DISTINCT source_system ORDER BY source_system) AS sources
                FROM graph_node_provenance
                WHERE customer_id = $1
                  AND node_id = gn.node_id
            ) gnp ON TRUE
            -- PHASE 2: optional curated display name override.
            LEFT JOIN entity_cluster_metadata ecm
              ON ecm.customer_id = $1
             AND ecm.label = gn.label
             AND ecm.primary_canonical_id = gn.canonical_id
            WHERE gn.label != '{document_label}'
              -- Fuzzy exclusion (codex-P2): the routed-entity canonical_id
              -- the LLM extracted may not exactly match the graph node's
              -- canonical_id (e.g. router emits 'prbe-backend' but graph
              -- stores 'prbe-ai/prbe-backend'). Compare normalized variants
              -- on the candidate side: lowered canonical_id, lowered
              -- namespace-stripped canonical_id, lowered display_name,
              -- and lowered namespace-stripped display_name. The caller
              -- emits the routed entity's normalized strings via
              -- exclude_node_keys so we can match either path.
              AND NOT EXISTS (
                  SELECT 1 FROM exclude_keys ek
                  WHERE ek.label = gn.label
                    AND ek.canonical_id IN (
                        lower(gn.canonical_id),
                        regexp_replace(lower(gn.canonical_id), '^.*/', ''),
                        lower(gn.properties->>'name'),
                        regexp_replace(lower(gn.properties->>'name'), '^.*/', '')
                    )
              )
            GROUP BY gn.canonical_id, gn.label, gn.properties->>'name',
                     gn.node_id, ea_count.alias_count, gnp.sources, ecm.display_name
            HAVING max({confidence_case}) >= $6  -- min_confidence floor
        ),
        neighbor_global_freq AS (
            -- IDF denominator: how many tenant docs is each neighbor
            -- attached to across the entire graph? Two halves UNION ALL'd
            -- so each hits a dedicated edge index instead of betting on
            -- BitmapOr. Apply the same min_confidence floor as the
            -- numerator so AMBIGUOUS edges don't inflate the denominator
            -- and bury specific entities (codex P2: min_confidence on
            -- IDF denom).
            SELECT node_id, COUNT(DISTINCT doc_node_id) AS global_doc_count
            FROM (
                -- direction 1: neighbor is from_node
                SELECT ra.node_id, e2.to_node_id AS doc_node_id
                FROM result_aggregates ra
                JOIN graph_edges e2
                  ON e2.customer_id = $1
                 AND e2.from_node_id = ra.node_id
                 AND (e2.valid_to IS NULL OR e2.valid_to > now())
                 AND {confidence_case_e2_from} >= $6
                JOIN graph_nodes doc_gn2
                  ON doc_gn2.customer_id = $1
                 AND doc_gn2.label = '{document_label}'
                 AND doc_gn2.node_id = e2.to_node_id
                UNION ALL
                -- direction 2: neighbor is to_node
                SELECT ra.node_id, e2.from_node_id AS doc_node_id
                FROM result_aggregates ra
                JOIN graph_edges e2
                  ON e2.customer_id = $1
                 AND e2.to_node_id = ra.node_id
                 AND (e2.valid_to IS NULL OR e2.valid_to > now())
                 AND {confidence_case_e2_to} >= $6
                JOIN graph_nodes doc_gn2
                  ON doc_gn2.customer_id = $1
                 AND doc_gn2.label = '{document_label}'
                 AND doc_gn2.node_id = e2.from_node_id
            ) bidirectional
            GROUP BY node_id
        )
        SELECT
            ra.canonical_id, ra.label, ra.display_name, ra.edge_types,
            ra.max_confidence_rank, ra.doc_count,
            -- IDF-adjusted score: more weight to specific (low-freq) entities
            (ra.doc_count::float / ln(1 + COALESCE(ngf.global_doc_count, 1)))
                AS score,
            ra.sample_pool,
            ra.member_count,
            ra.member_sources
        FROM result_aggregates ra
        LEFT JOIN neighbor_global_freq ngf USING (node_id)
        -- Final tiebreakers (label, canonical_id) make ordering deterministic
        -- across identical-score requests. Without them, LIMIT $7 silently
        -- flickers between calls, polluting LLM BFS heuristics.
        ORDER BY score DESC, ra.doc_count DESC, ra.max_confidence_rank DESC,
                 ra.label ASC, ra.canonical_id ASC
        LIMIT $7
    """

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            sql,
            customer_id,
            doc_ids,
            ranks,
            exclude_labels,
            exclude_canonical_ids,
            min_rank,
            top_n,
        )

    out: list[RelatedEntity] = []
    for r in rows:
        # Translate max_confidence_rank back to tier name (codex-A1).
        # Defensive fallback: an unknown rank means the SQL CASE returned
        # something we don't have a Python tier for; surface the highest
        # known tier rather than crashing.
        rank_int = int(r["max_confidence_rank"])
        max_confidence = _RANK_TO_CONFIDENCE.get(rank_int, "EXTRACTED")

        # Dedupe sample_pool preserving rank order, then cap at 3.
        sample_pool = list(r["sample_pool"] or [])
        associated_doc_ids = _dedupe_preserving_order(sample_pool)[:3]

        out.append(
            RelatedEntity(
                canonical_id=r["canonical_id"],
                label=r["label"],
                display_name=r["display_name"],
                edge_types=list(r["edge_types"] or []),
                max_confidence=max_confidence,
                doc_count=int(r["doc_count"]),
                score=float(r["score"]),
                associated_doc_ids=associated_doc_ids,
                member_count=int(r["member_count"]),
                member_sources=list(r["member_sources"] or []),
            )
        )
    return out
