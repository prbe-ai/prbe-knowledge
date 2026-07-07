"""Fat tool surface for the gatherer agent.

Design principle: **FAT skills, thin harness.** Each tool is a complete
capability the agent invokes for a self-contained task — not a primitive
the agent has to chain together. The harness owns parallel fan-out and
multi-step plumbing internally.

Four fat tools + one terminal:

    search(queries[], entity_ids?, top_k?)
        Always fans out vector + bm25 + graph + inferred_edge in parallel
        per sub-query. Accepts 1+ queries (subsumes parallel_multi_query
        and the old reissue_query). Optionally accepts explicit
        entity_ids; otherwise the harness re-runs grounding per query
        for entity anchoring.

    subgraph(anchor_canonical_id, depth?, edge_types?, include_inferred?,
             include_aliases?, top_k_per_hop?)
        Multi-hop BFS in ONE call. Returns a tree of {nodes, edges,
        inferred_edges, alias_clusters}. depth 1-3. Subsumes graph_walk +
        expand_inferred_neighbors + expand_entity_cluster.

    fetch_doc(doc_id, max_chunks?, with_inferred_edges?, with_evidence?)
        Full doc detail. Subsumes fetch_doc_chunks +
        read_inferred_edge_evidence. Returns chunks + outbound inferred
        edges + the evidence chunks that produced each `why` string.

    need_deeper(reason)
        Soft budget extension (+10 tool calls, max 2). Handled directly
        by the loop, not via the registry — left here as a schema entry
        so the model can pick it.

    emit_gatherer_output(entities, chunks, gatherer_notes)
        TERMINAL — the agent calls this to end the loop. Its parameters
        ARE the GathererOutput schema; the loop reads the call's
        arguments as the final output. With tool_choice="required", the
        model MUST call something — either a retrieval tool or this
        terminal — so no prose path exists. This kills the prose-retry
        latency tax and eliminates the schema-violation failure mode.

Defense-in-depth from PR #282 inherited:
- `_escape_query_for_xml` wraps user-controlled text flowing into any
  downstream LLM call (the harness's own LLM extraction does this too).
- `top_k` is clamped per-tool so an agent loop can't DOS the DB.

Plan: docs/specs/agentic-search.md, section "Tool surface".
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from services.retrieval.agent.models import GathererOutput
from services.retrieval.grounding import GroundingBundle, build_bundle
from services.retrieval.helpers import expand_to_cluster_members
from services.retrieval.retrievers.bm25 import bm25_search as _bm25
from services.retrieval.retrievers.graph import graph_search as _graph
from services.retrieval.retrievers.inferred_edges import inferred_edge_search as _inferred
from services.retrieval.retrievers.vector import vector_search as _vector
from services.retrieval.router import (
    _escape_query_for_xml,  # noqa: F401 — re-exported for any caller
)
from shared.constants import (
    INFERRED_EDGE_HYDRATION_CHUNKS,
    SEARCH_AGENT_BM25_TOP_K,
    SEARCH_AGENT_CHUNK_WINDOW_DEFAULT,
    SEARCH_AGENT_CHUNK_WINDOW_MAX,
    SEARCH_AGENT_FETCH_CHUNKS_MAX,
    SEARCH_AGENT_GRAPH_TOP_K,
    SEARCH_AGENT_GRAPH_WALK_TOP_K,
    SEARCH_AGENT_INFERRED_EDGE_TOP_K,
    SEARCH_AGENT_PER_HIT_PROPERTIES_CAP,
    SEARCH_AGENT_VECTOR_TOP_K,
    NodeLabel,
)
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import TemporalSpec

log = get_logger(__name__)


# ============================================================
# Identifiers used outside this module
# ============================================================

# Name of the terminal tool. The loop checks `tool_call.function.name ==
# TERMINAL_TOOL_NAME` to detect end-of-loop and parse arguments as
# GathererOutput. Don't rename without updating loop.py.
TERMINAL_TOOL_NAME = "emit_gatherer_output"

# Name of the budget-extension tool. Like the terminal, this is handled
# by the loop directly (not via the registry) — listed here so the
# loop's dispatcher logic can short-circuit it cleanly.
NEED_DEEPER_TOOL_NAME = "need_deeper"

# Hard cap to keep an agent from asking for top_k=10000. Per-tool clamps
# enforce this — even if the prompt and JSON Schema both lie, this is
# the floor.
_HARD_TOP_K_CAP = 100

# Subgraph depth cap. depth=3 means up to 3 hops from the anchor.
# Iterative 1-hop walks (no AGE/Cypher), so each hop costs ~1 SQL round
# trip. Cap prevents pathological depth=10 BFS exhausting the DB pool.
_SUBGRAPH_MAX_DEPTH = 3

# Per-query sub-search cap (for execute_search). 5 mirrors MAX_INTENTS+headroom
# from PR #282; the agent uses this for 2-3 sub-queries typically.
_SEARCH_MAX_SUBQUERIES = 5


# ============================================================
# Helpers
# ============================================================

def _clamp_top_k(value: int | None, default: int) -> int:
    """Clamp `value` into [1, _HARD_TOP_K_CAP]; fall back to `default`."""
    if value is None:
        return default
    if value < 1:
        return 1
    return min(value, _HARD_TOP_K_CAP)


def _trim_properties(props: dict[str, Any]) -> dict[str, Any]:
    """Cap serialized properties at SEARCH_AGENT_PER_HIT_PROPERTIES_CAP bytes."""
    encoded = json.dumps(props, default=str)
    if len(encoded) <= SEARCH_AGENT_PER_HIT_PROPERTIES_CAP:
        return props
    out = dict(props)
    while len(json.dumps(out, default=str)) > SEARCH_AGENT_PER_HIT_PROPERTIES_CAP:
        candidates = [
            (k, len(str(v))) for k, v in out.items() if isinstance(v, str)
        ]
        if not candidates:
            break
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        biggest_key, biggest_len = candidates[0]
        if biggest_len <= 120:
            break
        truncated = str(out[biggest_key])[:120] + f"... <TRUNCATED {biggest_len} chars>"
        out[biggest_key] = truncated
    return out


def _hit_to_chunk_dict(hit: Any, channel: str) -> dict[str, Any]:
    """Normalize a channel hit (VectorHit / BM25Hit / GraphHit) to a
    chunk-shaped dict the agent reads."""
    return {
        "channel": channel,
        "chunk_id": getattr(hit, "chunk_id", None),
        "doc_id": hit.doc_id,
        "source_system": hit.source_system,
        "source_url": hit.source_url,
        "title": hit.title,
        "content": getattr(hit, "content", None),
        "score": float(hit.score),
        "created_at": hit.created_at.isoformat() if hit.created_at else None,
        "updated_at": hit.updated_at.isoformat() if hit.updated_at else None,
        "author_id": hit.author_id,
    }


def _inferred_hit_to_dict(hit: Any) -> dict[str, Any]:
    """Normalize an InferredEdgeHit for the agent. `why` is the moat."""
    return {
        "channel": "inferred_edge",
        "doc_id": hit.doc_id,
        "source_system": hit.source_system,
        "source_url": hit.source_url,
        "title": hit.title,
        "anchor_doc_id": hit.anchor_doc_id,
        "anchor_rank": hit.anchor_rank,
        "edge_type": hit.edge_type,
        "confidence": hit.confidence,
        "why": hit.why,
        "linked_edge_count": hit.linked_edge_count,
        "score": float(hit.score),
        "created_at": hit.created_at.isoformat() if hit.created_at else None,
        "updated_at": hit.updated_at.isoformat() if hit.updated_at else None,
        "author_id": hit.author_id,
    }


def _bundle_to_compact(bundle: GroundingBundle) -> dict[str, Any]:
    def _c(c: Any) -> dict[str, Any]:
        return {
            "entity_type": c.entity_type,
            "canonical_id": c.canonical_id,
            "display_name": c.display_name,
            "match_source": c.match_source,
        }
    return {
        "candidates": [_c(c) for c in bundle.candidates],
        "bare_id_matches": [_c(c) for c in bundle.bare_id_matches],
        "connected_sources": list(bundle.connected_sources),
        "timing_ms": float(bundle.timing_ms),
    }


async def _resolve_entities_to_anchor_docs(
    customer_id: str,
    *,
    entities: list[dict[str, str]],
    total_cap: int,
) -> list[str]:
    """Find Document graph_nodes attached to each entity (1-hop), return
    up to `total_cap` distinct doc canonical_ids ordered by recency."""
    if not entities:
        return []
    from services.retrieval.grounding import _LABEL_TO_ENTITY_TYPE
    entity_to_label = {v: k for k, v in _LABEL_TO_ENTITY_TYPE.items()}

    pairs: list[tuple[str, str]] = []
    for e in entities:
        et = (e.get("entity_type") or "").lower()
        cid = e.get("canonical_id")
        label = entity_to_label.get(et)
        if label and cid:
            pairs.append((label, cid))
    if not pairs:
        return []

    document_label = NodeLabel.DOCUMENT.value
    sql = f"""
        WITH anchor_entities AS (
            SELECT * FROM unnest($2::text[], $3::text[]) AS t(label, canonical_id)
        ),
        anchor_nodes AS (
            SELECT gn.node_id, gn.label
            FROM anchor_entities ae
            JOIN graph_nodes gn
              ON gn.customer_id = $1
             AND gn.label = ae.label
             AND gn.canonical_id = ae.canonical_id
        ),
        edges AS (
            SELECT ge.to_node_id AS neighbor_id
              FROM anchor_nodes an
              JOIN graph_edges ge
                ON ge.customer_id = $1 AND ge.from_node_id = an.node_id
            UNION ALL
            SELECT ge.from_node_id AS neighbor_id
              FROM anchor_nodes an
              JOIN graph_edges ge
                ON ge.customer_id = $1 AND ge.to_node_id = an.node_id
        )
        SELECT DISTINCT gn.canonical_id AS doc_id, gn.updated_at
        FROM edges e
        JOIN graph_nodes gn
          ON gn.customer_id = $1
         AND gn.node_id = e.neighbor_id
         AND gn.label = '{document_label}'
        ORDER BY gn.updated_at DESC NULLS LAST
        LIMIT $4
    """
    labels = [label for label, _ in pairs]
    cids = [cid for _, cid in pairs]
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, customer_id, labels, cids, total_cap)
    return [r["doc_id"] for r in rows]


# ============================================================
# 1. search — fat 4-channel fan-out per sub-query
# ============================================================

async def execute_search(
    customer_id: str,
    *,
    queries: list[str],
    entity_ids: list[dict[str, str]] | None = None,
    top_k: int | None = None,
    author_ids: list[str] | None = None,
    sort_by: Literal["relevance", "recency"] = "relevance",
    doc_types: list[str] | None = None,
) -> dict[str, Any]:
    """Fan out 1+ queries through the 4 channels (vector + bm25 + graph +
    inferred_edge) in parallel — same shape the harness runs on turn 0.

    If `entity_ids` is omitted and the agent provides multiple queries,
    each query gets its own grounding run; entity-anchored channels
    (graph, inferred_edge) anchor on the per-query grounding bundle.
    When `entity_ids` IS provided, those entities anchor every sub-query.

    `author_ids`, `sort_by`, and `doc_types` thread into every retriever
    call so all four channels apply the same author hard-filter,
    doc-type hard-filter, and ordering discipline. Defaults match today's
    behavior (no filters, relevance-ranked) — the harness only overrides
    when the LLM extractor's `search_options` say otherwise. See
    `services/retrieval/agent/extractor.py` and
    `services/retrieval/agent/models.py:SearchOptions`.

    `doc_types`, when set, hard-filters `documents.doc_type = ANY(...)`.
    Used for class-level queries ("the latest PR", "what tickets are in
    progress") so the channels return the top-K by recency/relevance
    from the matching class instead of being anchored on whatever
    specific entity the extractor picked from its candidates list.

    Use cases:
      - Reformulate the original query (one new sub-query).
      - Multi-intent decomposition ("X and Y about different things").
      - Recover from an entity-extraction misfire (pass explicit entity_ids).
      - Author-anchored / recency-sorted shots (pass author_ids + sort_by="recency").
      - Class-level queries ("latest PR", "tickets this week") — pass
        doc_types=["github.pull_request"] etc. instead of a specific
        entity anchor.

    Returns: {sub_queries: [{query, grounded_entities, vector[], bm25[],
                            graph[], inferred_edge[]}]}
    """
    queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()][:_SEARCH_MAX_SUBQUERIES]
    if not queries:
        return {"sub_queries": []}
    top_k_v = _clamp_top_k(top_k, SEARCH_AGENT_VECTOR_TOP_K)
    top_k_b = _clamp_top_k(top_k, SEARCH_AGENT_BM25_TOP_K)
    top_k_g = _clamp_top_k(top_k, SEARCH_AGENT_GRAPH_TOP_K)
    top_k_i = _clamp_top_k(top_k, SEARCH_AGENT_INFERRED_EDGE_TOP_K)

    async def _per_query(q: str) -> dict[str, Any]:
        # Resolve entities for this sub-query (use provided ones, else
        # per-query grounding).
        if entity_ids:
            ents = list(entity_ids)
            grounded_summary = {"source": "caller", "candidates": ents}
        else:
            bundle = await build_bundle(customer_id=customer_id, query=q)
            ents = [
                {"entity_type": c.entity_type, "canonical_id": c.canonical_id}
                for c in (list(bundle.candidates) + list(bundle.bare_id_matches))
            ]
            grounded_summary = _bundle_to_compact(bundle)

        # Channel-side anchor pairs for graph_search.
        graph_pairs: list[tuple[str, str]] = []
        if ents:
            from services.retrieval.retrievers.graph import _ENTITY_TO_LABEL
            for e in ents:
                et = (e.get("entity_type") or "").lower()
                cid = e.get("canonical_id")
                label = _ENTITY_TO_LABEL.get(et)
                if label and cid:
                    graph_pairs.append((label, cid))

        async def _vec_call() -> list[dict[str, Any]]:
            try:
                hits = await _vector(
                    customer_id=customer_id, query_text=q,
                    top_k=top_k_v, temporal=TemporalSpec(),
                    author_ids=author_ids,
                    sort_by=sort_by,
                    doc_types=doc_types,
                )
                return [_hit_to_chunk_dict(h, "vector") for h in hits]
            except Exception as exc:
                log.warning("agent.search_vector_failed", error=str(exc), query=q[:50])
                return []

        async def _bm25_call() -> list[dict[str, Any]]:
            try:
                hits = await _bm25(
                    customer_id=customer_id, query_text=q,
                    top_k=top_k_b, temporal=TemporalSpec(),
                    author_ids=author_ids,
                    sort_by=sort_by,
                    doc_types=doc_types,
                )
                return [_hit_to_chunk_dict(h, "bm25") for h in hits]
            except Exception as exc:
                log.warning("agent.search_bm25_failed", error=str(exc), query=q[:50])
                return []

        async def _graph_call() -> list[dict[str, Any]]:
            if not graph_pairs:
                return []
            try:
                hits = await _graph(
                    customer_id=customer_id, entities=graph_pairs,
                    top_k=top_k_g, temporal=TemporalSpec(),
                    author_ids=author_ids,
                    sort_by=sort_by,
                    doc_types=doc_types,
                )
                return [
                    {
                        **_hit_to_chunk_dict(h, "graph"),
                        "via_entity": h.via_entity,
                        "via_label": h.via_label,
                        "edge_type": h.edge_type,
                        "confidence": h.confidence,
                    }
                    for h in hits
                ]
            except Exception as exc:
                log.warning("agent.search_graph_failed", error=str(exc), query=q[:50])
                return []

        async def _inferred_call() -> list[dict[str, Any]]:
            if not ents:
                return []
            try:
                anchor_doc_ids = await _resolve_entities_to_anchor_docs(
                    customer_id=customer_id, entities=ents, total_cap=top_k_i,
                )
                if not anchor_doc_ids:
                    return []
                hits = await _inferred(
                    customer_id=customer_id, top_doc_ids=anchor_doc_ids,
                    top_k=top_k_i, dampening=1.0,
                    author_ids=author_ids,
                    sort_by=sort_by,
                    doc_types=doc_types,
                )
                return [_inferred_hit_to_dict(h) for h in hits]
            except Exception as exc:
                log.warning("agent.search_inferred_failed", error=str(exc), query=q[:50])
                return []

        v, b, g, i = await asyncio.gather(
            _vec_call(), _bm25_call(), _graph_call(), _inferred_call(),
        )
        return {
            "query": q,
            "grounded_entities": ents,
            "grounding_summary": grounded_summary,
            "vector": v,
            "bm25": b,
            "graph": g,
            "inferred_edge": i,
        }

    sub_queries = await asyncio.gather(*(_per_query(q) for q in queries))
    return {"sub_queries": list(sub_queries)}


# ============================================================
# 2. subgraph — multi-hop graph BFS in ONE call
# ============================================================

async def _graph_walk_one_hop(
    customer_id: str,
    *,
    anchor_canonical_id: str,
    edge_types: list[str] | None,
    top_k: int,
) -> dict[str, Any]:
    """1-hop bidirectional walk on graph_edges from a single anchor.
    Internal helper for execute_subgraph; same shape as the old
    execute_graph_walk."""
    from services.retrieval.graph_explore import anchor_exists
    from services.retrieval.main import _resolve_anchor_alias

    resolved = await _resolve_anchor_alias(
        customer_id=customer_id,
        anchor_canonical_id=anchor_canonical_id,
    )
    if not await anchor_exists(customer_id=customer_id, anchor_canonical_id=resolved):
        return {"anchor_canonical_id": resolved, "anchor_existed": False, "neighbors": []}

    edge_filter_sql = ""
    params: list[Any] = [customer_id, resolved, top_k]
    if edge_types:
        edge_types_clean = [
            t for t in edge_types if isinstance(t, str) and t.replace("_", "").isalnum()
        ]
        if edge_types_clean:
            edge_filter_sql = "AND ge.edge_type = ANY($4::text[])"
            params.append(edge_types_clean)

    sql = f"""
        WITH anchor AS (
            SELECT node_id FROM graph_nodes
            WHERE customer_id = $1 AND canonical_id = $2
            LIMIT 1
        ),
        neighbor_edges AS (
            SELECT ge.to_node_id AS neighbor_node_id, ge.edge_type, ge.confidence
            FROM anchor a
            JOIN graph_edges ge
              ON ge.customer_id = $1 AND ge.from_node_id = a.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
             {edge_filter_sql}
            UNION ALL
            SELECT ge.from_node_id AS neighbor_node_id, ge.edge_type, ge.confidence
            FROM anchor a
            JOIN graph_edges ge
              ON ge.customer_id = $1 AND ge.to_node_id = a.node_id
             AND (ge.valid_to IS NULL OR ge.valid_to > now())
             {edge_filter_sql}
        ),
        neighbor_rollup AS (
            SELECT neighbor_node_id,
                   COUNT(*) AS hit_count,
                   array_agg(DISTINCT edge_type) AS edge_types
            FROM neighbor_edges
            GROUP BY neighbor_node_id
        ),
        neighbor_degree AS (
            SELECT nr.*,
                   (SELECT COUNT(*) FROM graph_edges ge2
                       WHERE ge2.customer_id = $1
                         AND (ge2.from_node_id = nr.neighbor_node_id
                              OR ge2.to_node_id = nr.neighbor_node_id)
                   ) AS global_degree
            FROM neighbor_rollup nr
        )
        SELECT gn.canonical_id, gn.label,
               coalesce(gn.properties->>'name', gn.canonical_id) AS display_name,
               gn.properties,
               nd.hit_count, nd.edge_types, nd.global_degree,
               (nd.hit_count::float / ln(1 + GREATEST(1, nd.global_degree))) AS idf_score
        FROM neighbor_degree nd
        JOIN graph_nodes gn
          ON gn.customer_id = $1 AND gn.node_id = nd.neighbor_node_id
        ORDER BY idf_score DESC, nd.hit_count DESC
        LIMIT $3
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, *params)

    neighbors = []
    for r in rows:
        props = dict(r["properties"] or {})
        neighbors.append({
            "canonical_id": r["canonical_id"],
            "label": r["label"],
            "display_name": r["display_name"],
            "properties": _trim_properties(props),
            "edge_types": list(r["edge_types"] or []),
            "score": float(r["idf_score"]),
            "hit_count": int(r["hit_count"]),
            "global_degree": int(r["global_degree"]),
        })
    return {
        "anchor_canonical_id": resolved,
        "anchor_existed": True,
        "neighbors": neighbors,
    }


async def execute_subgraph(
    customer_id: str,
    *,
    anchor_canonical_id: str,
    depth: int | None = None,
    edge_types: list[str] | None = None,
    include_inferred: bool | None = None,
    include_aliases: bool | None = None,
    top_k_per_hop: int | None = None,
) -> dict[str, Any]:
    """Multi-hop BFS from an anchor node in ONE tool call.

    Walks the graph up to `depth` hops, deduping nodes, accumulating
    edges. Optionally enriches Document nodes with their outbound
    INFERRED Doc-Doc edges (`include_inferred=True`) so the LLM `why`
    strings surface inline. Optionally expands entity aliases
    (`include_aliases=True`) so Person/Repo clusters are visible.

    Returns: {anchor, depth, nodes[], inferred_edges[], alias_clusters{}}
    """
    depth = max(1, min(depth or 1, _SUBGRAPH_MAX_DEPTH))
    top_k_per_hop = _clamp_top_k(top_k_per_hop, SEARCH_AGENT_GRAPH_WALK_TOP_K)
    if include_inferred is None:
        include_inferred = True
    if include_aliases is None:
        include_aliases = True

    # Hop 1 — must succeed to even know the anchor exists.
    hop1 = await _graph_walk_one_hop(
        customer_id=customer_id,
        anchor_canonical_id=anchor_canonical_id,
        edge_types=edge_types,
        top_k=top_k_per_hop,
    )
    if not hop1.get("anchor_existed"):
        return {
            "anchor_canonical_id": hop1["anchor_canonical_id"],
            "anchor_existed": False,
            "depth": depth,
            "nodes": [],
            "inferred_edges": [],
            "alias_clusters": {},
        }
    resolved_anchor = hop1["anchor_canonical_id"]

    # Accumulate nodes (dedup by canonical_id); track depth-of-discovery.
    nodes: dict[str, dict[str, Any]] = {}
    for n in hop1["neighbors"]:
        nodes[n["canonical_id"]] = {**n, "hop": 1}
    frontier = [n["canonical_id"] for n in hop1["neighbors"]]

    # Hops 2..depth — sequential per hop level, but parallel within a hop.
    for hop_idx in range(2, depth + 1):
        if not frontier:
            break
        hop_results = await asyncio.gather(
            *(
                _graph_walk_one_hop(
                    customer_id=customer_id,
                    anchor_canonical_id=anchor,
                    edge_types=edge_types,
                    top_k=top_k_per_hop,
                )
                for anchor in frontier
            )
        )
        next_frontier: list[str] = []
        for parent, h in zip(frontier, hop_results, strict=True):
            for n in h.get("neighbors", []):
                cid = n["canonical_id"]
                if cid == resolved_anchor or cid in nodes:
                    continue
                nodes[cid] = {**n, "hop": hop_idx, "via_parent": parent}
                next_frontier.append(cid)
        frontier = next_frontier

    # Inferred-edge enrichment on Document nodes in the subgraph.
    inferred_edges: list[dict[str, Any]] = []
    if include_inferred:
        doc_label = NodeLabel.DOCUMENT.value
        doc_ids = [
            cid for cid, n in nodes.items()
            if n.get("label") == doc_label
        ][:10]
        if doc_ids:
            try:
                hits = await _inferred(
                    customer_id=customer_id,
                    top_doc_ids=doc_ids,
                    top_k=10,
                    dampening=1.0,
                )
                inferred_edges = [_inferred_hit_to_dict(h) for h in hits]
            except Exception as exc:
                log.warning("agent.subgraph_inferred_failed", error=str(exc))

    # Alias-cluster enrichment per non-Document label.
    alias_clusters: dict[str, dict[str, list[str]]] = {}
    if include_aliases:
        by_label: dict[str, list[str]] = {}
        for cid, n in nodes.items():
            lbl = n.get("label")
            if lbl and lbl != NodeLabel.DOCUMENT.value:
                by_label.setdefault(lbl, []).append(cid)
        async with with_tenant(customer_id) as conn:
            for lbl, cids in by_label.items():
                try:
                    mapping = await expand_to_cluster_members(
                        conn=conn,
                        customer_id=customer_id,
                        label=lbl,
                        canonical_ids=cids[:20],
                    )
                    if any(len(v) > 1 for v in mapping.values()):
                        alias_clusters[lbl] = mapping
                except Exception as exc:
                    log.warning(
                        "agent.subgraph_alias_failed",
                        error=str(exc),
                        label=lbl,
                    )

    return {
        "anchor_canonical_id": resolved_anchor,
        "anchor_existed": True,
        "depth": depth,
        "nodes": list(nodes.values()),
        "inferred_edges": inferred_edges,
        "alias_clusters": alias_clusters,
    }


# ============================================================
# 3. fetch_doc — full doc detail in ONE call
# ============================================================

async def execute_fetch_doc(
    customer_id: str,
    *,
    doc_id: str,
    max_chunks: int | None = None,
    offset: int | None = None,
    with_inferred_edges: bool | None = None,
    with_evidence: bool | None = None,
) -> dict[str, Any]:
    """Paginate a doc's chunks plus optional inferred-edge context in ONE call.

    Returns at most `max_chunks` (default `SEARCH_AGENT_FETCH_CHUNKS_MAX`)
    chunks starting at `offset` in chunk_index order, so a long doc is walked
    a page at a time rather than one call hauling the whole thing. `offset`
    also lets the agent reach a chunk past the first page — the prior
    `LIMIT`-only query could never return a matched chunk at index >= 10.

    Returns: {doc_id, chunks[], offset, limit, next_offset,
              outbound_inferred_edges[], evidence_by_edge_id{}}
    `next_offset` is the offset to pass for the next page, or None when this
    page reached the end of the doc.
    """
    n = _clamp_top_k(max_chunks, SEARCH_AGENT_FETCH_CHUNKS_MAX)
    off = max(0, int(offset)) if offset is not None else 0
    if with_inferred_edges is None:
        with_inferred_edges = False
    if with_evidence is None:
        with_evidence = False

    # Plan A Component 6: hide draft chunks from the agent fetch path.
    # The agent runs under an API key and never sees pre-approval content.
    chunks_sql = """
        SELECT chunk_id, doc_id, content, kind, chunk_index
        FROM chunks
        WHERE customer_id = $1 AND doc_id = $2
          AND visibility = 'approved'
        ORDER BY chunk_index ASC
        LIMIT $3 OFFSET $4
    """

    inferred_edges: list[dict[str, Any]] = []
    evidence_by_edge: dict[str, list[dict[str, Any]]] = {}

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(chunks_sql, customer_id, doc_id, n, off)
        chunks = [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "content": r["content"],
                "kind": r["kind"],
                "chunk_index": int(r["chunk_index"]),
            }
            for r in rows
        ]

        # Doc-level edges/evidence are independent of the page — only attach
        # them to the FIRST page so paging a long doc doesn't re-inject the
        # (potentially large) evidence blob into the message history on every
        # call, defeating the context budget this change protects.
        if with_inferred_edges and off == 0:
            try:
                hits = await _inferred(
                    customer_id=customer_id,
                    top_doc_ids=[doc_id],
                    top_k=10,
                    dampening=1.0,
                )
                inferred_edges = [_inferred_hit_to_dict(h) for h in hits]
            except Exception as exc:
                log.warning("agent.fetch_doc_inferred_failed", error=str(exc), doc_id=doc_id)

        if with_evidence and inferred_edges:
            # Pull the chunks the LLM was reasoning over for each edge.
            # graph_edges.properties->>'evidence_chunk_ids' is the producer's
            # stamp; hydrate from the chunks table.
            edge_sql = """
                SELECT edge_id, properties->>'why' AS why,
                       properties->'evidence_chunk_ids' AS evidence_chunk_ids
                FROM graph_edges
                WHERE customer_id = $1
                  AND extractor_id = 'inferred_edges:v1'
                  AND (from_node_id IN (
                       SELECT node_id FROM graph_nodes
                       WHERE customer_id = $1 AND canonical_id = $2
                       LIMIT 1
                  ) OR to_node_id IN (
                       SELECT node_id FROM graph_nodes
                       WHERE customer_id = $1 AND canonical_id = $2
                       LIMIT 1
                  ))
                LIMIT 20
            """
            edge_rows = await conn.fetch(edge_sql, customer_id, doc_id)
            all_chunk_ids: set[str] = set()
            edge_to_chunk_ids: dict[str, list[str]] = {}
            for r in edge_rows:
                raw = r["evidence_chunk_ids"]
                cids: list[str] = []
                if isinstance(raw, list):
                    cids = [str(x) for x in raw][:INFERRED_EDGE_HYDRATION_CHUNKS]
                edge_to_chunk_ids[str(r["edge_id"])] = cids
                all_chunk_ids.update(cids)
            if all_chunk_ids:
                # Hide drafts (Plan A Component 6): evidence chunks the
                # agent surfaces must be approved content.
                ev_rows = await conn.fetch(
                    """
                    SELECT chunk_id, doc_id, content, kind
                    FROM chunks
                    WHERE customer_id = $1 AND chunk_id = ANY($2::text[])
                      AND visibility = 'approved'
                    """,
                    customer_id,
                    list(all_chunk_ids),
                )
                ev_by_id = {
                    r["chunk_id"]: {
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "content": r["content"],
                        "kind": r["kind"],
                    }
                    for r in ev_rows
                }
                for edge_id, cids in edge_to_chunk_ids.items():
                    evidence_by_edge[edge_id] = [ev_by_id[c] for c in cids if c in ev_by_id]

    # A full page (len == n) may have more chunks after it; a short page is
    # the end of the doc. Deliberately avoids a COUNT(*) round-trip.
    next_offset = off + len(chunks) if len(chunks) == n else None
    return {
        "doc_id": doc_id,
        "chunks": chunks,
        "offset": off,
        "limit": n,
        "next_offset": next_offset,
        "outbound_inferred_edges": inferred_edges,
        "evidence_by_edge_id": evidence_by_edge,
    }


async def execute_fetch_chunk_window(
    customer_id: str,
    *,
    chunk_id: str,
    before: int | None = None,
    after: int | None = None,
) -> dict[str, Any]:
    """Return a matched chunk plus its immediate neighbours in the same doc.

    The pre-fan-out already surfaced the specific relevant chunk (vector /
    BM25 match). This pulls just enough adjacent context — `before` chunks
    before it and `after` chunks after it by chunk_index — to repair the
    fixed-window fragmentation of the 512-token chunker, WITHOUT hauling the
    whole doc. Total chunks returned <= before + 1 + after.

    Returns: {chunk_id, doc_id, chunks[], window: {before, after}}. When the
    chunk_id is unknown (or draft-only), `chunks` is empty and `doc_id` None.
    """
    b = SEARCH_AGENT_CHUNK_WINDOW_DEFAULT if before is None else max(0, int(before))
    a = SEARCH_AGENT_CHUNK_WINDOW_DEFAULT if after is None else max(0, int(after))
    b = min(b, SEARCH_AGENT_CHUNK_WINDOW_MAX)
    a = min(a, SEARCH_AGENT_CHUNK_WINDOW_MAX)

    # One query: resolve the target chunk's (doc_id, chunk_index) in a CTE,
    # then return the approved chunks in [idx - before, idx + after]. Draft
    # chunks stay hidden (Plan A Component 6), same as fetch_doc.
    window_sql = """
        WITH target AS (
            SELECT doc_id, chunk_index
            FROM chunks
            WHERE customer_id = $1 AND chunk_id = $2 AND visibility = 'approved'
            LIMIT 1
        )
        SELECT c.chunk_id, c.doc_id, c.content, c.kind, c.chunk_index
        FROM chunks c, target t
        WHERE c.customer_id = $1
          AND c.doc_id = t.doc_id
          AND c.visibility = 'approved'
          AND c.chunk_index BETWEEN t.chunk_index - $3 AND t.chunk_index + $4
        ORDER BY c.chunk_index ASC
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(window_sql, customer_id, chunk_id, b, a)

    chunks = [
        {
            "chunk_id": r["chunk_id"],
            "doc_id": r["doc_id"],
            "content": r["content"],
            "kind": r["kind"],
            "chunk_index": int(r["chunk_index"]),
        }
        for r in rows
    ]
    return {
        "chunk_id": chunk_id,
        "doc_id": chunks[0]["doc_id"] if chunks else None,
        "chunks": chunks,
        "window": {"before": b, "after": a},
    }


# ============================================================
# Tool schemas — OpenAI-style function defs forwarded to LiteLLM
# ============================================================

def tool_definitions() -> list[dict[str, Any]]:
    """Return the schemas LiteLLM forwards to Fireworks. Order matches
    the prompt's enumeration so the cached prefix stays stable."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": (
                    "Re-search with one or more reformulated queries. Always fans out "
                    "vector + bm25 + graph + inferred_edge in parallel per sub-query. "
                    "Use for: reformulating the original query, multi-intent splits, "
                    "recovering from a bad entity match by passing explicit entity_ids, "
                    "or refining the harness's pre-fan-out with author / recency filters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": _SEARCH_MAX_SUBQUERIES,
                        },
                        "entity_ids": {
                            "type": "array",
                            "description": (
                                "Optional: explicit entities to anchor graph + inferred_edge "
                                "channels on. When omitted, the harness re-runs grounding "
                                "per sub-query."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity_type": {"type": "string"},
                                    "canonical_id": {"type": "string"},
                                },
                                "required": ["entity_type", "canonical_id"],
                            },
                        },
                        "author_ids": {
                            "type": "array",
                            "description": (
                                "Optional: canonical_ids of `person` entities that "
                                "must be the authors of the returned docs. Hard "
                                "filter — channels narrow to "
                                "`documents.author_id = ANY(...)` before ranking. "
                                "Use when the query says 'PRs by X', 'commits from "
                                "Y', 'messages by Z', etc."
                            ),
                            "items": {"type": "string"},
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["relevance", "recency"],
                            "description": (
                                "Default `relevance` (semantic + lexical scoring). Pass "
                                "`recency` when you want the freshest hits first — e.g. "
                                "the harness's pre-fan-out was relevance-ordered but you "
                                "need to find the LATEST activity that matches. Works "
                                "with or without `author_ids`."
                            ),
                        },
                        "top_k": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                    "required": ["queries"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "subgraph",
                "description": (
                    "Multi-hop BFS from an anchor node. Returns a tree of "
                    "{nodes, inferred_edges, alias_clusters}. Subsumes the old "
                    "graph_walk + expand_inferred_neighbors + expand_entity_cluster."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor_canonical_id": {"type": "string"},
                        "depth": {"type": "integer", "minimum": 1, "maximum": _SUBGRAPH_MAX_DEPTH},
                        "edge_types": {"type": "array", "items": {"type": "string"}},
                        "include_inferred": {"type": "boolean"},
                        "include_aliases": {"type": "boolean"},
                        "top_k_per_hop": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                    "required": ["anchor_canonical_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_doc",
                "description": (
                    "Paginate a whole doc when you need MORE than the matched chunk you "
                    "already have — the answer spans several of its sections. Returns "
                    "`limit` chunks (default 10) starting at `offset` in document order; "
                    "pass the returned `next_offset` to read the next page. Optional "
                    "outbound inferred-edges + per-edge evidence. For just the context "
                    "AROUND a specific matched chunk, prefer `fetch_chunk_window`."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "max_chunks": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                        "offset": {
                            "type": "integer",
                            "minimum": 0,
                            "description": "Chunk index to start this page at. Default 0. Use next_offset to page.",
                        },
                        "with_inferred_edges": {"type": "boolean"},
                        "with_evidence": {"type": "boolean"},
                    },
                    "required": ["doc_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_chunk_window",
                "description": (
                    "Return a matched chunk plus its immediate neighbours in the same doc. "
                    "The pre-fan-out already gave you the specific relevant chunk; call this "
                    "only to pull a little surrounding context when that chunk reads like a "
                    "fragment (starts mid-thought, references something just before it). "
                    "Pass the chunk_id you already have. Cheaper than fetch_doc; does not "
                    "haul the whole document."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "before": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": SEARCH_AGENT_CHUNK_WINDOW_MAX,
                            "description": "Chunks to include before the matched one. Default 1.",
                        },
                        "after": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": SEARCH_AGENT_CHUNK_WINDOW_MAX,
                            "description": "Chunks to include after the matched one. Default 1.",
                        },
                    },
                    "required": ["chunk_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": NEED_DEEPER_TOOL_NAME,
                "description": (
                    "Soft-budget extension. +10 tool calls per extension, max 2. "
                    "Provide a `reason` string (logged for trace review)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"reason": {"type": "string"}},
                    "required": ["reason"],
                },
            },
        },
        # TERMINAL — the loop watches for this name and parses arguments as
        # GathererOutput. With tool_choice="required", the model MUST call
        # something — either a retrieval tool above or this terminal — so
        # prose-only output is impossible.
        {
            "type": "function",
            "function": {
                "name": TERMINAL_TOOL_NAME,
                "description": (
                    "TERMINAL. Call this when you've curated the answer. The "
                    "arguments ARE the final GathererOutput (entities, chunks, "
                    "gatherer_notes). Calling this ends the loop — do not call "
                    "any other tool in the same turn."
                ),
                "parameters": GathererOutput.model_json_schema(),
            },
        },
    ]


# ============================================================
# Registry + dispatcher
# ============================================================

TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "search": execute_search,
    "subgraph": execute_subgraph,
    "fetch_doc": execute_fetch_doc,
    "fetch_chunk_window": execute_fetch_chunk_window,
}


async def dispatch_tool_call(
    customer_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Route a model-emitted retrieval tool call to its executor.

    `need_deeper` and `emit_gatherer_output` are handled by the loop
    directly (need_deeper extends budget, emit_gatherer_output is
    terminal) and never reach this dispatcher.

    Catches exceptions so the agent loop never crashes on a single
    tool failure — agent sees `{"error": "..."}` and can recover.
    """
    executor = TOOL_REGISTRY.get(tool_name)
    if executor is None:
        return {"error": f"unknown tool: {tool_name}"}
    try:
        return await executor(customer_id=customer_id, **arguments)
    except TypeError as exc:
        log.warning("agent.tool_arg_mismatch", tool_name=tool_name, error=str(exc))
        return {"error": f"argument mismatch: {exc}"}
    except Exception as exc:
        log.warning(
            "agent.tool_execution_failed",
            tool_name=tool_name,
            error=str(exc),
            error_type=type(exc).__name__,
        )
        return {"error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "NEED_DEEPER_TOOL_NAME",
    "TERMINAL_TOOL_NAME",
    "TOOL_REGISTRY",
    "dispatch_tool_call",
    "execute_fetch_chunk_window",
    "execute_fetch_doc",
    "execute_search",
    "execute_subgraph",
    "tool_definitions",
]
