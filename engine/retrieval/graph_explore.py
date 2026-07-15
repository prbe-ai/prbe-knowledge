"""Graph-explore queries for the dashboard graph visualization.

Two SQL-only paths used by POST /graph/explore:

  default_graph_query()  - top-N nodes by graph_nodes.degree DESC plus the
                           1-hop edges joined to that selected set. Powers
                           the "show me my whole graph" landing view.

  anchor_graph_query()   - tiered BFS centered on a single canonical_id.
                           Hop 1 caps at GRAPH_EXPLORE_HOP1_CAP neighbors;
                           if there's still budget, Hop 2 fills the rest
                           up to GRAPH_EXPLORE_NODE_CAP. Bidirectional via
                           UNION ALL on idx_graph_edges_from /
                           idx_graph_edges_to (per
                           feedback_postgres_bidirectional_or_to_union --
                           Postgres won't reliably BitmapOr two single-
                           column edge indexes).

  graph_search_query()   - lightweight prefix typeahead for the anchor
                           picker. Hits idx_graph_nodes_lower_canonical /
                           idx_graph_nodes_lower_props_name.

All queries run inside `with_tenant(customer_id)` so RLS enforces tenant
isolation at the DB level. The GUC is `app.current_customer_id` (per
feedback_prbe_knowledge_rls_guc_name -- the wrong name silently returns
empty results).

Edge cap: max GRAPH_EXPLORE_EDGE_CAP edges in the response regardless of
node count, ordered by (confidence_priority ASC, edge_type_priority ASC)
so the most-meaningful edges survive truncation.

Edge dedup: UNION ALL produces one row per direction; the serializer
collapses to one logical edge per (source, target, edge_type) triple.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from engine.shared.constants import (
    GRAPH_EXPLORE_EDGE_CAP,
    GRAPH_EXPLORE_HOP1_CAP,
    GRAPH_EXPLORE_HOP2_CAP,
    GRAPH_EXPLORE_NODE_CAP,
    GRAPH_EXPLORE_WHY_MAX_CHARS,
    GRAPH_SEARCH_MAX_LIMIT,
    EdgeType,
    NodeLabel,
)
from engine.shared.db import with_tenant

# Confidence -> priority (lower = more meaningful). Drives edge-cap
# truncation: when total edges exceed GRAPH_EXPLORE_EDGE_CAP, the highest-
# priority confidences survive. EXTRACTED is deterministic AST/connector
# extraction; INFERRED is LLM-promoted; AMBIGUOUS is unresolved.
CONFIDENCE_PRIORITY: dict[str, int] = {
    "EXTRACTED": 1,
    "INFERRED": 2,
    "AMBIGUOUS": 3,
}
# Fallback used when graph_edges.confidence is some unexpected legacy
# value (e.g. NULL on pre-Lane-B rows). Mirrors related_entities's
# "treat unknown as EXTRACTED" behavior so old rows don't drop out.
CONFIDENCE_PRIORITY_DEFAULT = CONFIDENCE_PRIORITY["EXTRACTED"]

# Edge-type priority for anchor BFS: lower = first in the result. Ties
# broken by neighbor.degree DESC, then neighbor.node_id (deterministic
# ordering -- repeated requests show stable graphs, which the frontend
# layout cache depends on).
EDGE_TYPE_PRIORITY: dict[str, int] = {
    EdgeType.DISCUSSES.value: 1,
    EdgeType.RESOLVES.value: 2,
    EdgeType.DOCUMENTS.value: 3,
    EdgeType.MENTIONS_ENTITY.value: 4,
    EdgeType.RELATES_TO.value: 5,
}
# Any edge_type not in the explicit map gets demoted below the priority
# tiers above. Rank > max(EDGE_TYPE_PRIORITY) so visualization-relevant
# inferred edges always lead.
EDGE_TYPE_PRIORITY_DEFAULT = 6

# Allowlist for the Pydantic request schema. Keep narrow: only the edge
# types meaningful for a knowledge-graph viz. Code-graph edges (CALLS /
# IMPORTS / etc.) live in the same tables but are too dense to render in
# the same view -- a future endpoint can expose them on demand.
EXPLORE_EDGE_TYPES: frozenset[str] = frozenset({
    EdgeType.DISCUSSES.value,
    EdgeType.RESOLVES.value,
    EdgeType.DOCUMENTS.value,
    EdgeType.MENTIONS_ENTITY.value,
    EdgeType.RELATES_TO.value,
})

EXPLORE_CONFIDENCES: frozenset[str] = frozenset({
    "EXTRACTED",
    "INFERRED",
    "AMBIGUOUS",
})


@dataclass(slots=True)
class ExploreFilters:
    """Filters applied to both default and anchor queries.

    All fields optional. None means "do not filter on this dimension"
    (i.e. include all values). Lists must be non-empty when set.
    """

    edge_types: list[str] | None = None
    confidences: list[str] | None = None
    source_systems: list[str] | None = None
    since: datetime | None = None


@dataclass(slots=True)
class GraphNodeRow:
    """Row shape produced by both default and anchor queries.

    `id` is graph_nodes.canonical_id (vendor-neutral; the Pydantic API
    schema also calls it `id`). `title` is the human-displayable label
    -- for Documents we surface the doc title via a join, for entities
    we surface properties->>'name' (or fall back to canonical_id at
    serialization time).
    """

    id: str
    label: str
    title: str | None
    source_system: str | None
    community_id: int | None
    degree: int


@dataclass(slots=True)
class GraphEdgeRow:
    """Row shape for one direction of a graph_edges row.

    Bidirectional walks emit two rows per logical edge (one with the
    anchor as from_node, one with anchor as to_node). The serializer
    collapses to one logical edge per (source, target, edge_type) triple.
    """

    source: str
    target: str
    edge_type: str
    confidence: str
    why: str | None


@dataclass(slots=True)
class GraphQueryResult:
    """Combined output of default_graph_query + anchor_graph_query.

    `total_nodes_available` and `total_edges_available` are the counts
    BEFORE the per-cap truncation, so the response can surface
    "showing X of Y" in the UI.
    """

    nodes: list[GraphNodeRow]
    edges: list[GraphEdgeRow]
    total_nodes_available: int
    total_edges_available: int


# ---------------------------------------------------------------------------
# Query builders -- helpers that produce SQL fragments + parameter lists.
# ---------------------------------------------------------------------------


def _build_edge_filter_sql(
    filters: ExploreFilters | None,
    *,
    edge_alias: str,
    next_param_idx: int,
) -> tuple[str, list[object]]:
    """Build a WHERE-clause fragment + params list for edge-side filters.

    Returns (sql_fragment, params). Fragment is empty string when no
    filters apply. Caller appends to existing WHERE.

    `next_param_idx` is the 1-based index of the FIRST new positional
    parameter -- the SQL fragment uses $next, $next+1, ...
    """
    parts: list[str] = []
    params: list[object] = []
    idx = next_param_idx

    if filters is None:
        return "", []

    if filters.edge_types:
        parts.append(f"AND {edge_alias}.edge_type = ANY(${idx}::text[])")
        params.append(list(filters.edge_types))
        idx += 1

    if filters.confidences:
        parts.append(f"AND {edge_alias}.confidence = ANY(${idx}::text[])")
        params.append(list(filters.confidences))
        idx += 1

    if filters.source_systems:
        # graph_edges.source_system is the per-edge provenance (which
        # connector wrote the edge). graph_nodes has no source_system
        # column -- node-side source data lives in properties->>'source_system'
        # for some labels and in graph_node_provenance for others. Filtering
        # at the edge level matches what the user actually means
        # ("show me edges that came from Linear") and keeps the query
        # planner-friendly.
        parts.append(f"AND {edge_alias}.source_system = ANY(${idx}::text[])")
        params.append(list(filters.source_systems))
        idx += 1

    if filters.since is not None:
        # graph_edges has no created_at column -- valid_from is the
        # timestamp at which the edge entered the graph (set at INSERT).
        # Use that as the closest semantic match to "edges since X".
        parts.append(f"AND {edge_alias}.valid_from >= ${idx}")
        params.append(filters.since)
        idx += 1

    return "\n            ".join(parts), params


def _node_title_expr(node_alias: str, doc_alias: str) -> str:
    """SQL expression for a node's display title.

    Documents: prefer the joined documents.title (more authoritative than
    properties stash). Entities: properties->>'name' (the canonical
    display name set by the upserter). Falls back to canonical_id at
    serialization time when both are NULL.
    """
    return (
        f"CASE WHEN {node_alias}.label = '{NodeLabel.DOCUMENT.value}'\n"
        f"            THEN COALESCE({doc_alias}.title, {node_alias}.properties->>'title')\n"
        f"            ELSE {node_alias}.properties->>'name'\n"
        f"       END"
    )


def _node_source_system_expr(node_alias: str, doc_alias: str) -> str:
    """SQL expression for a node's source system.

    Documents: documents.source_system (authoritative). Entities:
    properties->>'source_system'. Falls back to NULL when neither is
    available -- code_graph entity nodes intentionally don't carry one.
    """
    return (
        f"CASE WHEN {node_alias}.label = '{NodeLabel.DOCUMENT.value}'\n"
        f"            THEN {doc_alias}.source_system\n"
        f"            ELSE {node_alias}.properties->>'source_system'\n"
        f"       END"
    )


# ---------------------------------------------------------------------------
# Default mode: top-N nodes by degree, 1-hop edges among the selected set.
# ---------------------------------------------------------------------------


async def default_graph_query(
    *,
    customer_id: str,
    filters: ExploreFilters | None = None,
) -> GraphQueryResult:
    """Default graph view: top GRAPH_EXPLORE_NODE_CAP nodes by degree DESC.

    Edges are the subset of graph_edges where BOTH endpoints are in the
    selected node set. This produces a connected-by-construction subgraph
    suitable for force-directed layout in the dashboard.

    The node-side filter uses idx_graph_nodes_customer_degree (added in
    migration 0063). Edge-side filters apply on the join in WHERE before
    the LIMIT-equivalent (per feedback_filter_before_limit -- never
    filter in Python after a top-N).
    """
    edge_filter_sql, edge_filter_params = _build_edge_filter_sql(
        filters,
        edge_alias="ge",
        next_param_idx=4,  # $1=customer_id, $2=node_cap, $3=edge_cap
    )

    sql = f"""
        WITH selected_nodes AS (
            -- Top-N by degree. customer_id is enforced by RLS as well;
            -- repeated here so the planner uses
            -- idx_graph_nodes_customer_degree directly.
            SELECT node_id, canonical_id, label, properties, community_id, degree
            FROM graph_nodes
            WHERE customer_id = $1
            ORDER BY degree DESC, node_id ASC
            LIMIT $2
        ),
        node_count AS (
            -- Total available nodes (pre-cap) so the API can render
            -- "showing X of Y". Filter-aware: matches the same scope
            -- as selected_nodes' WHERE.
            SELECT COUNT(*) AS total FROM graph_nodes WHERE customer_id = $1
        ),
        candidate_edges AS (
            -- 1-hop edges among the selected set. Both endpoints must
            -- be in selected_nodes -- otherwise we'd surface dangling
            -- edges into nodes we're not rendering.
            SELECT ge.from_node_id, ge.to_node_id,
                   ge.edge_type, ge.confidence,
                   ge.properties->>'why' AS why,
                   ge.valid_from
            FROM graph_edges ge
            JOIN selected_nodes sn_from ON sn_from.node_id = ge.from_node_id
            JOIN selected_nodes sn_to   ON sn_to.node_id   = ge.to_node_id
            WHERE ge.customer_id = $1
              AND (ge.valid_to IS NULL OR ge.valid_to > now())
              {edge_filter_sql}
        ),
        edge_count AS (
            SELECT COUNT(*) AS total FROM candidate_edges
        ),
        capped_edges AS (
            -- Order-then-cap so the highest-priority edges survive
            -- truncation (per CONFIDENCE_PRIORITY / EDGE_TYPE_PRIORITY).
            SELECT ce.*
            FROM candidate_edges ce
            ORDER BY
                CASE ce.confidence
                    WHEN 'EXTRACTED' THEN 1
                    WHEN 'INFERRED' THEN 2
                    WHEN 'AMBIGUOUS' THEN 3
                    ELSE 1
                END ASC,
                CASE ce.edge_type
                    WHEN 'DISCUSSES' THEN 1
                    WHEN 'RESOLVES' THEN 2
                    WHEN 'DOCUMENTS' THEN 3
                    WHEN 'MENTIONS_ENTITY' THEN 4
                    WHEN 'RELATES_TO' THEN 5
                    ELSE 6
                END ASC,
                ce.from_node_id ASC, ce.to_node_id ASC
            LIMIT $3
        )
        SELECT
            'node' AS row_kind,
            sn.canonical_id AS source_canonical_id,
            NULL::text     AS target_canonical_id,
            sn.label, sn.community_id, sn.degree,
            {_node_title_expr("sn", "d")} AS title,
            {_node_source_system_expr("sn", "d")} AS source_system,
            NULL::text AS edge_type,
            NULL::text AS confidence,
            NULL::text AS why,
            (SELECT total FROM node_count) AS total_nodes_available,
            (SELECT total FROM edge_count) AS total_edges_available
        FROM selected_nodes sn
        LEFT JOIN documents d
          ON d.customer_id = $1
         AND d.doc_id = sn.canonical_id
         AND d.valid_to IS NULL
        UNION ALL
        SELECT
            'edge' AS row_kind,
            sn_from.canonical_id AS source_canonical_id,
            sn_to.canonical_id   AS target_canonical_id,
            NULL::text AS label,
            NULL::int  AS community_id,
            NULL::int  AS degree,
            NULL::text AS title,
            NULL::text AS source_system,
            ce.edge_type, ce.confidence, ce.why,
            (SELECT total FROM node_count) AS total_nodes_available,
            (SELECT total FROM edge_count) AS total_edges_available
        FROM capped_edges ce
        JOIN selected_nodes sn_from ON sn_from.node_id = ce.from_node_id
        JOIN selected_nodes sn_to   ON sn_to.node_id   = ce.to_node_id
    """

    params: list[object] = [
        customer_id,
        GRAPH_EXPLORE_NODE_CAP,
        GRAPH_EXPLORE_EDGE_CAP,
    ]
    params.extend(edge_filter_params)

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, *params)

    return _split_rows_to_result(rows)


# ---------------------------------------------------------------------------
# Anchor mode: tiered BFS centered on one canonical_id.
# ---------------------------------------------------------------------------


async def anchor_exists(*, customer_id: str, anchor_canonical_id: str) -> bool:
    """RLS-filtered check: does this canonical_id exist in this tenant?

    Used by the endpoint to differentiate 404 (anchor not found in
    tenant) from 200-with-empty-graph (anchor exists but has no edges).
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT 1
            FROM graph_nodes
            WHERE customer_id = $1 AND canonical_id = $2
            LIMIT 1
            """,
            customer_id,
            anchor_canonical_id,
        )
    return row is not None


async def anchor_graph_query(
    *,
    customer_id: str,
    anchor_canonical_id: str,
    filters: ExploreFilters | None = None,
) -> GraphQueryResult:
    """Tiered BFS centered on `anchor_canonical_id`.

    Hop 1: up to GRAPH_EXPLORE_HOP1_CAP neighbors of the anchor.
    Hop 2: up to GRAPH_EXPLORE_HOP2_CAP additional neighbors-of-neighbors,
           bounded by total = GRAPH_EXPLORE_NODE_CAP.

    Bidirectional via UNION ALL on idx_graph_edges_from /
    idx_graph_edges_to. Deterministic ordering by
    (edge_type_priority, neighbor.degree DESC, neighbor.node_id) so
    repeated requests show stable graphs.

    Caller MUST verify the anchor exists in this tenant before invoking
    (use anchor_exists()) -- this function returns an empty result for
    a missing anchor, indistinguishable from "anchor exists with 0
    edges". The endpoint translates the missing-anchor case to 404.
    """
    # The edge filter is identical across all six insertion points (hop1
    # both directions, hop2 both directions, all_edges both directions),
    # but each occurrence needs its own positional-param indices because
    # asyncpg has no named-param support. We append the same filter
    # params to the params list six times and bump next_param_idx by the
    # filter-param count between each call.
    #
    # $1=customer_id, $2=anchor_canonical_id, $3=hop1_cap, $4=hop2_cap,
    # $5=node_cap, $6=edge_cap, then six identical filter-param blocks.
    edge_filter_hop1_dir1, edge_filter_params_block = _build_edge_filter_sql(
        filters, edge_alias="ge", next_param_idx=7,
    )
    n_filter_params = len(edge_filter_params_block)
    edge_filter_hop1_dir2, _ = _build_edge_filter_sql(
        filters, edge_alias="ge", next_param_idx=7 + n_filter_params,
    )
    edge_filter_hop2_dir1, _ = _build_edge_filter_sql(
        filters, edge_alias="ge", next_param_idx=7 + 2 * n_filter_params,
    )
    edge_filter_hop2_dir2, _ = _build_edge_filter_sql(
        filters, edge_alias="ge", next_param_idx=7 + 3 * n_filter_params,
    )
    edge_filter_all_dir1, _ = _build_edge_filter_sql(
        filters, edge_alias="ge", next_param_idx=7 + 4 * n_filter_params,
    )
    edge_filter_all_dir2, _ = _build_edge_filter_sql(
        filters, edge_alias="ge", next_param_idx=7 + 5 * n_filter_params,
    )

    sql = f"""
        WITH anchor AS (
            -- Resolve the anchor canonical_id to its node_id.
            -- LIMIT 1 covers the (rare) case where two labels somehow
            -- share a canonical_id; the unique constraint on
            -- (customer_id, label, canonical_id) means there's at most
            -- one per label, but a per-tenant duplicate across labels
            -- is theoretically possible. Picking deterministically
            -- (lowest node_id) keeps repeated requests stable.
            SELECT node_id
            FROM graph_nodes
            WHERE customer_id = $1 AND canonical_id = $2
            ORDER BY node_id ASC
            LIMIT 1
        ),
        hop1_edges AS (
            -- Direction 1: anchor as from_node (uses idx_graph_edges_from).
            SELECT a.node_id AS anchor_node_id,
                   ge.to_node_id AS neighbor_node_id,
                   ge.edge_type, ge.confidence,
                   ge.properties->>'why' AS why,
                   ge.from_node_id, ge.to_node_id, ge.valid_from
            FROM anchor a
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.from_node_id = a.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
              {edge_filter_hop1_dir1}
            UNION ALL
            -- Direction 2: anchor as to_node (uses idx_graph_edges_to).
            SELECT a.node_id AS anchor_node_id,
                   ge.from_node_id AS neighbor_node_id,
                   ge.edge_type, ge.confidence,
                   ge.properties->>'why' AS why,
                   ge.from_node_id, ge.to_node_id, ge.valid_from
            FROM anchor a
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.to_node_id = a.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
              {edge_filter_hop1_dir2}
        ),
        hop1_neighbors_ranked AS (
            -- One row per neighbor (collapse direction-doubled edges).
            -- Preserve the highest-priority edge_type per neighbor for
            -- ordering -- a DISCUSSES edge wins over a RELATES_TO edge
            -- when picking which neighbors fit under HOP1_CAP.
            SELECT he.neighbor_node_id,
                   MIN(
                       CASE he.edge_type
                           WHEN 'DISCUSSES' THEN 1
                           WHEN 'RESOLVES' THEN 2
                           WHEN 'DOCUMENTS' THEN 3
                           WHEN 'MENTIONS_ENTITY' THEN 4
                           WHEN 'RELATES_TO' THEN 5
                           ELSE 6
                       END
                   ) AS edge_type_priority,
                   gn.degree, gn.node_id
            FROM hop1_edges he
            JOIN graph_nodes gn
              ON gn.customer_id = $1
             AND gn.node_id = he.neighbor_node_id
            GROUP BY he.neighbor_node_id, gn.degree, gn.node_id
        ),
        hop1_capped AS (
            -- Deterministic top-HOP1_CAP neighbors. Tie-break order:
            -- (edge_type_priority ASC, degree DESC, node_id ASC). The
            -- last tiebreaker matters -- without it, two neighbors with
            -- identical (priority, degree) flicker between calls and
            -- the layout cache thrashes.
            SELECT neighbor_node_id, edge_type_priority, degree, node_id
            FROM hop1_neighbors_ranked
            ORDER BY edge_type_priority ASC, degree DESC, node_id ASC
            LIMIT $3
        ),
        hop2_seed AS (
            -- Seeds for the second hop are the hop1 neighbors plus the
            -- anchor itself. Edges originating from any of these and
            -- terminating at a non-already-included node yield hop2
            -- candidates.
            SELECT neighbor_node_id AS node_id FROM hop1_capped
            UNION ALL
            SELECT node_id FROM anchor
        ),
        hop2_edges AS (
            -- Same UNION ALL pattern. Filter to edges NOT touching
            -- already-included nodes via NOT IN against hop2_seed in
            -- the hop2 select. Edge-type/confidence/source/since filters
            -- apply HERE so the BFS frontier respects the user's filter
            -- (otherwise hop2 nodes reachable only via filtered-out
            -- edges would still get included).
            SELECT s.node_id AS seed_node_id,
                   ge.to_node_id AS neighbor_node_id,
                   ge.edge_type, ge.confidence,
                   ge.properties->>'why' AS why,
                   ge.from_node_id, ge.to_node_id, ge.valid_from
            FROM hop2_seed s
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.from_node_id = s.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
              {edge_filter_hop2_dir1}
            UNION ALL
            SELECT s.node_id AS seed_node_id,
                   ge.from_node_id AS neighbor_node_id,
                   ge.edge_type, ge.confidence,
                   ge.properties->>'why' AS why,
                   ge.from_node_id, ge.to_node_id, ge.valid_from
            FROM hop2_seed s
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.to_node_id = s.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
              {edge_filter_hop2_dir2}
        ),
        hop2_neighbors_ranked AS (
            SELECT he.neighbor_node_id,
                   MIN(
                       CASE he.edge_type
                           WHEN 'DISCUSSES' THEN 1
                           WHEN 'RESOLVES' THEN 2
                           WHEN 'DOCUMENTS' THEN 3
                           WHEN 'MENTIONS_ENTITY' THEN 4
                           WHEN 'RELATES_TO' THEN 5
                           ELSE 6
                       END
                   ) AS edge_type_priority,
                   gn.degree, gn.node_id
            FROM hop2_edges he
            JOIN graph_nodes gn
              ON gn.customer_id = $1
             AND gn.node_id = he.neighbor_node_id
            WHERE he.neighbor_node_id NOT IN (SELECT node_id FROM hop2_seed)
            GROUP BY he.neighbor_node_id, gn.degree, gn.node_id
        ),
        hop2_capped AS (
            SELECT neighbor_node_id, edge_type_priority, degree, node_id
            FROM hop2_neighbors_ranked
            ORDER BY edge_type_priority ASC, degree DESC, node_id ASC
            LIMIT $4
        ),
        all_node_ids AS (
            -- Anchor + hop1 + hop2, deduped via UNION (set semantics).
            -- LIMIT $5 enforces the absolute node cap defensively even
            -- when HOP1_CAP + HOP2_CAP + 1 already fits.
            SELECT node_id FROM anchor
            UNION
            SELECT neighbor_node_id AS node_id FROM hop1_capped
            UNION
            SELECT neighbor_node_id AS node_id FROM hop2_capped
            LIMIT $5
        ),
        all_edges AS (
            -- Re-fetch ALL edges among the selected node set so the
            -- final response includes hop1<->hop2 cross edges (which
            -- the BFS proper missed). UNION ALL halves to use the
            -- single-column edge indexes. Edge filters apply HERE too
            -- so the response edges match what the user asked for --
            -- otherwise the BFS would respect filters but the response
            -- shape would still surface filtered-out edge types between
            -- the selected nodes.
            SELECT ge.from_node_id, ge.to_node_id,
                   ge.edge_type, ge.confidence,
                   ge.properties->>'why' AS why
            FROM all_node_ids a
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.from_node_id = a.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
              {edge_filter_all_dir1}
            JOIN all_node_ids a2 ON a2.node_id = ge.to_node_id
            UNION ALL
            SELECT ge.from_node_id, ge.to_node_id,
                   ge.edge_type, ge.confidence,
                   ge.properties->>'why' AS why
            FROM all_node_ids a
            JOIN graph_edges ge
              ON ge.customer_id = $1
             AND ge.to_node_id = a.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
              {edge_filter_all_dir2}
            JOIN all_node_ids a2 ON a2.node_id = ge.from_node_id
        ),
        edge_count AS (
            -- Pre-cap edge count for "showing X of Y". Note the UNION ALL
            -- in all_edges produces two rows per logical edge; the
            -- serializer dedups, so the user-facing count must mirror
            -- the de-duplicated total.
            SELECT COUNT(*) AS total FROM (
                SELECT DISTINCT from_node_id, to_node_id, edge_type
                FROM all_edges
            ) sub
        ),
        capped_edges AS (
            SELECT ae.*
            FROM all_edges ae
            ORDER BY
                CASE ae.confidence
                    WHEN 'EXTRACTED' THEN 1
                    WHEN 'INFERRED' THEN 2
                    WHEN 'AMBIGUOUS' THEN 3
                    ELSE 1
                END ASC,
                CASE ae.edge_type
                    WHEN 'DISCUSSES' THEN 1
                    WHEN 'RESOLVES' THEN 2
                    WHEN 'DOCUMENTS' THEN 3
                    WHEN 'MENTIONS_ENTITY' THEN 4
                    WHEN 'RELATES_TO' THEN 5
                    ELSE 6
                END ASC,
                ae.from_node_id ASC, ae.to_node_id ASC
            LIMIT $6
        ),
        node_count AS (
            SELECT COUNT(*) AS total FROM all_node_ids
        )
        SELECT
            'node' AS row_kind,
            gn.canonical_id AS source_canonical_id,
            NULL::text      AS target_canonical_id,
            gn.label, gn.community_id, gn.degree,
            {_node_title_expr("gn", "d")} AS title,
            {_node_source_system_expr("gn", "d")} AS source_system,
            NULL::text AS edge_type,
            NULL::text AS confidence,
            NULL::text AS why,
            (SELECT total FROM node_count) AS total_nodes_available,
            (SELECT total FROM edge_count) AS total_edges_available
        FROM all_node_ids ani
        JOIN graph_nodes gn
          ON gn.customer_id = $1
         AND gn.node_id = ani.node_id
        LEFT JOIN documents d
          ON d.customer_id = $1
         AND d.doc_id = gn.canonical_id
         AND d.valid_to IS NULL
        UNION ALL
        SELECT
            'edge' AS row_kind,
            gn_from.canonical_id AS source_canonical_id,
            gn_to.canonical_id   AS target_canonical_id,
            NULL::text AS label,
            NULL::int  AS community_id,
            NULL::int  AS degree,
            NULL::text AS title,
            NULL::text AS source_system,
            ce.edge_type, ce.confidence, ce.why,
            (SELECT total FROM node_count) AS total_nodes_available,
            (SELECT total FROM edge_count) AS total_edges_available
        FROM capped_edges ce
        JOIN graph_nodes gn_from
          ON gn_from.customer_id = $1
         AND gn_from.node_id = ce.from_node_id
        JOIN graph_nodes gn_to
          ON gn_to.customer_id = $1
         AND gn_to.node_id = ce.to_node_id
    """

    params: list[object] = [
        customer_id,
        anchor_canonical_id,
        GRAPH_EXPLORE_HOP1_CAP,
        GRAPH_EXPLORE_HOP2_CAP,
        GRAPH_EXPLORE_NODE_CAP,
        GRAPH_EXPLORE_EDGE_CAP,
    ]
    # Append the same filter param block six times -- once per insertion
    # point in the SQL above. asyncpg has no named-param support, so each
    # `${idx}` reference needs its own positional value; they all happen
    # to be the same value because the filter is identical at every site.
    for _ in range(6):
        params.extend(edge_filter_params_block)

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, *params)

    return _split_rows_to_result(rows)


# ---------------------------------------------------------------------------
# Row -> dataclass split + edge dedup.
# ---------------------------------------------------------------------------


def _split_rows_to_result(rows: list[object]) -> GraphQueryResult:
    """Split a single-query SELECT-with-UNION-ALL result into nodes + edges.

    Both queries return rows tagged with row_kind ('node' or 'edge') so
    a single round-trip returns both halves. Same totals on every row,
    so we read them off the first row.

    Edge dedup: collapse duplicate (source, target, edge_type) triples
    that the bidirectional UNION ALL produces. Mandatory -- UNION ALL
    is documented as not deduping.
    """
    nodes: list[GraphNodeRow] = []
    edge_seen: set[tuple[str, str, str]] = set()
    edges: list[GraphEdgeRow] = []
    total_nodes = 0
    total_edges = 0

    for row in rows:
        # asyncpg Record supports both __getitem__ and key access.
        kind = row["row_kind"]  # type: ignore[index]
        if total_nodes == 0 and total_edges == 0:
            total_nodes = int(row["total_nodes_available"] or 0)  # type: ignore[index]
            total_edges = int(row["total_edges_available"] or 0)  # type: ignore[index]

        if kind == "node":
            nodes.append(GraphNodeRow(
                id=row["source_canonical_id"],  # type: ignore[index]
                label=row["label"],  # type: ignore[index]
                title=row["title"],  # type: ignore[index]
                source_system=row["source_system"],  # type: ignore[index]
                community_id=row["community_id"],  # type: ignore[index]
                degree=int(row["degree"] or 0),  # type: ignore[index]
            ))
        else:
            source = row["source_canonical_id"]  # type: ignore[index]
            target = row["target_canonical_id"]  # type: ignore[index]
            edge_type = row["edge_type"]  # type: ignore[index]
            key = (source, target, edge_type)
            if key in edge_seen:
                continue
            edge_seen.add(key)
            confidence = row["confidence"] or "EXTRACTED"  # type: ignore[index]
            # `why` only carries meaning for INFERRED / AMBIGUOUS edges
            # (LLM-generated rationale). EXTRACTED edges are deterministic
            # AST/connector output, so any 'why' field on them is noise --
            # drop it.
            why_raw = row["why"] if confidence != "EXTRACTED" else None  # type: ignore[index]
            why = _truncate_why(why_raw)
            edges.append(GraphEdgeRow(
                source=source,
                target=target,
                edge_type=edge_type,
                confidence=confidence,
                why=why,
            ))

    return GraphQueryResult(
        nodes=nodes,
        edges=edges,
        total_nodes_available=total_nodes,
        total_edges_available=total_edges,
    )


def _truncate_why(why: str | None) -> str | None:
    """Cap `why` text at GRAPH_EXPLORE_WHY_MAX_CHARS.

    Returns None for None input. Truncated strings get a trailing ellipsis
    so the dashboard can render "..." cue without re-measuring text.
    """
    if why is None:
        return None
    if len(why) <= GRAPH_EXPLORE_WHY_MAX_CHARS:
        return why
    # Reserve 3 chars for the ellipsis so the total length stays at the cap.
    return why[: GRAPH_EXPLORE_WHY_MAX_CHARS - 3] + "..."


# ---------------------------------------------------------------------------
# /graph/search: prefix typeahead for the anchor picker.
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GraphSearchHit:
    """One typeahead match. Same shape across all node labels."""

    id: str
    label: str
    title: str | None
    source_system: str | None
    degree: int


async def graph_search_query(
    *,
    customer_id: str,
    q: str,
    limit: int,
) -> list[GraphSearchHit]:
    """Prefix typeahead over canonical_id and properties->>'name'.

    Empty `q` returns []. `limit` is capped at GRAPH_SEARCH_MAX_LIMIT
    by the endpoint before reaching this function.

    Uses prefix match (`q + '%'`) -- leading wildcards skip the
    idx_graph_nodes_lower_canonical / idx_graph_nodes_lower_props_name
    indexes. Search-by-substring is intentionally out of scope for the
    typeahead; callers wanting fuzzy-match should use the main /retrieve
    pipeline.

    Match surfaces: canonical_id (covers Document nodes -- doc titles
    are embedded in the canonical_id for our doc kinds, e.g.
    `github:org/repo:pr:123`, `linear:LIN-456`) and properties->>'name'
    (covers Person/Service/Concept entities). Document `properties.title`
    is not searched directly; there's no functional index on it and it
    would seq-scan. This matches the /retrieve precedent.
    """
    q_clean = q.strip().lower()
    if not q_clean:
        return []

    bounded_limit = max(1, min(limit, GRAPH_SEARCH_MAX_LIMIT))
    pattern = q_clean + "%"

    sql = """
        SELECT canonical_id, label,
               COALESCE(
                   properties->>'name',
                   properties->>'title'
               ) AS title,
               properties->>'source_system' AS source_system,
               degree
        FROM graph_nodes
        WHERE customer_id = $1
          AND (
              LOWER(canonical_id) LIKE $2
              OR LOWER(properties->>'name') LIKE $2
          )
        ORDER BY degree DESC, canonical_id ASC
        LIMIT $3
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, customer_id, pattern, bounded_limit)

    return [
        GraphSearchHit(
            id=row["canonical_id"],
            label=row["label"],
            title=row["title"],
            source_system=row["source_system"],
            degree=int(row["degree"] or 0),
        )
        for row in rows
    ]
