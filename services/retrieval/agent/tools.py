"""Tool surface for the gatherer agent.

Each tool wraps an existing retrieval primitive and returns a
JSON-serializable dict the model reads on the next turn. Tool defs are
emitted as OpenAI-style function schemas for LiteLLM forwarding to
Fireworks; the model can't emit a malformed tool call when the SDK
exposes the tools properly (`tool_use` schema enforcement).

Defense in depth from PR #282 inherited here:
- `_escape_query_for_xml` wraps every user-controllable text field that
  flows into a downstream LLM call (today only `reissue_query`, but
  any future tool that re-invokes LLM-touching code needs the same).
- `top_k` is clamped to a per-tool hard cap so an agent loop cannot DOS
  the DB by asking for 10k rows.

Plan: docs/specs/agentic-search.md, section "Tool surface".
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

from services.retrieval.grounding import GroundingBundle, build_bundle
from services.retrieval.helpers import expand_to_cluster_members
from services.retrieval.retrievers.bm25 import bm25_search as _bm25
from services.retrieval.retrievers.graph import graph_search as _graph
from services.retrieval.retrievers.inferred_edges import inferred_edge_search as _inferred
from services.retrieval.retrievers.related_entities import (
    walk_result_doc_neighbors as _walk_neighbors,
)
from services.retrieval.retrievers.vector import vector_search as _vector
from services.retrieval.router import _escape_query_for_xml
from shared.constants import (
    INFERRED_EDGE_HYDRATION_CHUNKS,
    SEARCH_AGENT_BM25_TOP_K,
    SEARCH_AGENT_EXPAND_NEIGHBORS_TOP_K,
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

# Hard cap to keep an agent from asking for top_k=10000. Any tool that
# accepts `top_k` enforces this — even if the prompt and JSON Schema
# both lie, this is the floor.
_HARD_TOP_K_CAP = 100


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
    """Cap serialized properties at SEARCH_AGENT_PER_HIT_PROPERTIES_CAP bytes.

    Drops the longest string fields one by one until the dict fits.
    Long fields are replaced with `"<TRUNCATED N chars>"`. Numeric and
    boolean fields are kept verbatim. Preserves common short fields
    (`name`, `display_name`, `summary`, `why`) first.
    """
    import json

    encoded = json.dumps(props, default=str)
    if len(encoded) <= SEARCH_AGENT_PER_HIT_PROPERTIES_CAP:
        return props

    out = dict(props)
    # Truncate the longest str-valued field repeatedly until under cap.
    while len(json.dumps(out, default=str)) > SEARCH_AGENT_PER_HIT_PROPERTIES_CAP:
        candidates = [
            (k, len(str(v))) for k, v in out.items() if isinstance(v, str)
        ]
        if not candidates:
            break
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        biggest_key, biggest_len = candidates[0]
        if biggest_len <= 120:
            # Everything left is short; give up to avoid stripping all fields.
            break
        truncated = str(out[biggest_key])[:120] + f"... <TRUNCATED {biggest_len} chars>"
        out[biggest_key] = truncated
    return out


def _hit_to_chunk_dict(hit: Any, channel: str) -> dict[str, Any]:
    """Normalize a channel hit (VectorHit / BM25Hit / GraphHit) to a
    chunk-shaped dict the agent reads."""
    return {
        "channel": channel,
        "chunk_id": hit.chunk_id,
        "doc_id": hit.doc_id,
        "source_system": hit.source_system,
        "source_url": hit.source_url,
        "title": hit.title,
        "content": hit.content,
        "score": float(hit.score),
        "created_at": hit.created_at.isoformat() if hit.created_at else None,
        "updated_at": hit.updated_at.isoformat() if hit.updated_at else None,
        "author_id": hit.author_id,
    }


def _inferred_hit_to_dict(hit: Any) -> dict[str, Any]:
    """Normalize an InferredEdgeHit for the agent.

    Surfaces `why` prominently — it's the moat. Also includes the
    `linked_edge_count` so the agent can self-filter hub-anchored edges.
    """
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


def _related_to_dict(ent: Any) -> dict[str, Any]:
    """Normalize a RelatedEntity (shared.models) for the agent.

    Drops the heavyweight `associated_doc_ids` past 3 (it's already
    capped server-side, but be explicit) and trims member_sources to a
    set-style list.
    """
    base = {
        "canonical_id": ent.canonical_id,
        "label": ent.label,
        "display_name": ent.display_name,
        "edge_types": list(ent.edge_types or []),
        "max_confidence": ent.max_confidence,
        "doc_count": ent.doc_count,
        "score": float(ent.score),
        "associated_doc_ids": list(ent.associated_doc_ids or [])[:3],
        "member_count": ent.member_count,
        "member_sources": list(ent.member_sources or []),
    }
    return base


# ============================================================
# Tool implementations
# ============================================================

async def execute_vector_search(
    customer_id: str,
    *,
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    top_k = _clamp_top_k(top_k, SEARCH_AGENT_VECTOR_TOP_K)
    hits = await _vector(
        customer_id=customer_id,
        query_text=query,
        top_k=top_k,
        temporal=TemporalSpec(),
    )
    return {"hits": [_hit_to_chunk_dict(h, "vector") for h in hits]}


async def execute_bm25_search(
    customer_id: str,
    *,
    query: str,
    top_k: int | None = None,
) -> dict[str, Any]:
    top_k = _clamp_top_k(top_k, SEARCH_AGENT_BM25_TOP_K)
    hits = await _bm25(
        customer_id=customer_id,
        query_text=query,
        top_k=top_k,
        temporal=TemporalSpec(),
    )
    return {"hits": [_hit_to_chunk_dict(h, "bm25") for h in hits]}


async def execute_graph_search(
    customer_id: str,
    *,
    entities: list[dict[str, str]],
    top_k: int | None = None,
) -> dict[str, Any]:
    """Wraps graph_search. `entities` is a list of {entity_type, canonical_id} dicts;
    we cast to tuples for the underlying call."""
    top_k = _clamp_top_k(top_k, SEARCH_AGENT_GRAPH_TOP_K)
    if not entities:
        return {"hits": []}
    pairs: list[tuple[str, str]] = [
        (e["entity_type"], e["canonical_id"])
        for e in entities
        if e.get("entity_type") and e.get("canonical_id")
    ]
    if not pairs:
        return {"hits": []}
    hits = await _graph(
        customer_id=customer_id,
        entities=pairs,
        top_k=top_k,
        temporal=TemporalSpec(),
    )
    return {
        "hits": [
            {
                **_hit_to_chunk_dict(h, "graph"),
                "via_entity": h.via_entity,
                "via_label": h.via_label,
                "edge_type": h.edge_type,
                "confidence": h.confidence,
            }
            for h in hits
        ]
    }


async def execute_inferred_edge_search(
    customer_id: str,
    *,
    entities: list[dict[str, str]] | None = None,
    doc_ids: list[str] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Walks INFERRED Doc-Doc edges.

    The plan asks for `entity_ids` on turn 1 (the agent doesn't have docs
    yet), so this tool accepts EITHER:
      - `doc_ids` directly (the existing primitive — used by
        `expand_inferred_neighbors`), OR
      - `entities` — we first resolve to attached Docs via a 1-hop graph
        walk, then run the inferred-edge walk from those Docs.

    Returns linked docs with their `why` strings attached.
    """
    top_k = _clamp_top_k(top_k, SEARCH_AGENT_INFERRED_EDGE_TOP_K)

    anchor_doc_ids: list[str]
    if doc_ids:
        anchor_doc_ids = list(doc_ids)
    elif entities:
        anchor_doc_ids = await _resolve_entities_to_anchor_docs(
            customer_id=customer_id,
            entities=entities,
            per_entity_cap=5,
            total_cap=top_k,
        )
    else:
        return {"hits": []}

    if not anchor_doc_ids:
        return {"hits": []}

    # Dampening=1.0 here: the gatherer reads inferred-edge hits as a
    # first-class tool return, no fusion with other channels. The old
    # 0.2 dampening was a fusion-time score adjustment that doesn't
    # apply when the agent reads each channel separately (see plan
    # "Dampening is deleted" + decision-log entry on INFERRED_EDGE_DAMPENING).
    hits = await _inferred(
        customer_id=customer_id,
        top_doc_ids=anchor_doc_ids,
        top_k=top_k,
        dampening=1.0,
    )
    return {"hits": [_inferred_hit_to_dict(h) for h in hits]}


async def _resolve_entities_to_anchor_docs(
    customer_id: str,
    *,
    entities: list[dict[str, str]],
    per_entity_cap: int,
    total_cap: int,
) -> list[str]:
    """Find Document graph_nodes attached to each entity (1-hop), return
    up to `total_cap` distinct doc canonical_ids."""
    if not entities:
        return []

    # Map RouterEntity entity_type -> graph_node label via the inverse of
    # grounding._LABEL_TO_ENTITY_TYPE. Build inline to avoid importing
    # a private symbol.
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


async def execute_parallel_multi_query(
    customer_id: str,
    *,
    queries: list[str],
    top_k: int | None = None,
) -> dict[str, Any]:
    """Fan out N sub-queries through vector+bm25 in parallel. Returns
    per-query merged candidates so the agent can pick which sub-query
    each hit came from.

    Caps at 5 sub-queries to mirror MAX_INTENTS + headroom; the agent
    is encouraged to use this for 2-3 sub-queries, not as a search-everything.
    """
    queries = [q for q in (queries or []) if isinstance(q, str) and q.strip()][:5]
    if not queries:
        return {"sub_queries": []}
    top_k = _clamp_top_k(top_k, SEARCH_AGENT_VECTOR_TOP_K)

    async def _one(q: str) -> dict[str, Any]:
        v, b = await asyncio.gather(
            _vector(customer_id=customer_id, query_text=q, top_k=top_k, temporal=TemporalSpec()),
            _bm25(customer_id=customer_id, query_text=q, top_k=top_k, temporal=TemporalSpec()),
            return_exceptions=True,
        )
        v_hits = [_hit_to_chunk_dict(h, "vector") for h in v] if not isinstance(v, BaseException) else []
        b_hits = [_hit_to_chunk_dict(h, "bm25") for h in b] if not isinstance(b, BaseException) else []
        return {"query": q, "vector": v_hits, "bm25": b_hits}

    results = await asyncio.gather(*(_one(q) for q in queries))
    return {"sub_queries": list(results)}


async def execute_expand_inferred_neighbors(
    customer_id: str,
    *,
    doc_id: str,
    max: int | None = None,
) -> dict[str, Any]:
    top_k = _clamp_top_k(max, SEARCH_AGENT_EXPAND_NEIGHBORS_TOP_K)
    hits = await _inferred(
        customer_id=customer_id,
        top_doc_ids=[doc_id],
        top_k=top_k,
        dampening=1.0,
    )
    return {"hits": [_inferred_hit_to_dict(h) for h in hits]}


async def execute_expand_entity_cluster(
    customer_id: str,
    *,
    canonical_ids: list[str],
    label: str,
) -> dict[str, Any]:
    if not canonical_ids or not label:
        return {"clusters": {}}
    async with with_tenant(customer_id) as conn:
        mapping = await expand_to_cluster_members(
            conn=conn,
            customer_id=customer_id,
            label=label,
            canonical_ids=canonical_ids,
        )
    return {"clusters": mapping}


async def execute_fetch_doc_chunks(
    customer_id: str,
    *,
    doc_id: str,
    max: int | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Hydrate up to `max` chunks of `doc_id` from the chunks table.

    For now selects chunks by `chunk_index ASC` (mirror's INFERRED_EDGE_HYDRATION
    behavior). The `query` kwarg is reserved for future per-chunk re-rank
    but is not used in v1 — the agent already saw the chunks via
    vector/bm25 channels.
    """
    n = _clamp_top_k(max, SEARCH_AGENT_FETCH_CHUNKS_MAX)
    sql = """
        SELECT chunk_id, doc_id, content, kind, chunk_index
        FROM chunks
        WHERE customer_id = $1 AND doc_id = $2
        ORDER BY chunk_index ASC
        LIMIT $3
    """
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(sql, customer_id, doc_id, n)
    return {
        "chunks": [
            {
                "chunk_id": r["chunk_id"],
                "doc_id": r["doc_id"],
                "content": r["content"],
                "kind": r["kind"],
                "chunk_index": int(r["chunk_index"]),
            }
            for r in rows
        ]
    }


async def execute_graph_walk(
    customer_id: str,
    *,
    anchor_canonical_id: str,
    edge_types: list[str] | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """1-hop bidirectional walk on graph_edges from `anchor_canonical_id`.

    Modeled after `related_entities.walk_result_doc_neighbors` (the IDF
    pattern from line 168+), reuses the alias-resolution + existence-check
    plumbing (`_resolve_anchor_alias` from main.py, `anchor_exists` from
    graph_explore.py). New SQL because that walker takes (doc_id, rank)
    tuples; we want a single-anchor input.

    Returns up to `top_k` neighbor nodes (any label) sorted by IDF score
    descending. Neighbor properties are trimmed to ~2KB each.
    """
    top_k = _clamp_top_k(top_k, SEARCH_AGENT_GRAPH_WALK_TOP_K)
    from services.retrieval.graph_explore import anchor_exists
    from services.retrieval.main import _resolve_anchor_alias

    resolved_anchor = await _resolve_anchor_alias(
        customer_id=customer_id,
        anchor_canonical_id=anchor_canonical_id,
    )
    if not await anchor_exists(customer_id=customer_id, anchor_canonical_id=resolved_anchor):
        return {"neighbors": [], "anchor_canonical_id": resolved_anchor, "anchor_existed": False}

    edge_filter_sql = ""
    params: list[Any] = [customer_id, resolved_anchor, top_k]
    if edge_types:
        # Pin to known edge types only; ignore unknown to avoid SQL surprise.
        edge_types_clean = [
            t for t in edge_types if isinstance(t, str) and t.replace("_", "").isalnum()
        ]
        if edge_types_clean:
            edge_filter_sql = "AND ge.edge_type = ANY($4::text[])"
            params.append(edge_types_clean)

    # 1-hop walk: bidirectional UNION ALL (hits both edge indexes), then
    # IDF score per neighbor based on global degree.
    sql = f"""
        WITH anchor AS (
            SELECT node_id
            FROM graph_nodes
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
        "anchor_canonical_id": resolved_anchor,
        "anchor_existed": True,
        "neighbors": neighbors,
    }


async def execute_reissue_query(
    customer_id: str,
    *,
    reformulated_query: str,
) -> dict[str, Any]:
    """Re-run grounding + the 4-channel turn-1 fan-out on a reformulated query.

    Returns the bundle + per-channel hits as a single tool return so the
    agent can pivot without burning 5 follow-up turns.
    """
    safe = _escape_query_for_xml(reformulated_query)  # noqa: F841 — placeholder if downstream LLMs are added
    bundle = await build_bundle(customer_id=customer_id, query=reformulated_query)

    # Re-fire the 4 channels in parallel, anchored on the new grounding.
    entity_dicts = [
        {"entity_type": c.entity_type, "canonical_id": c.canonical_id}
        for c in (list(bundle.candidates) + list(bundle.bare_id_matches))
    ]

    vector_t, bm25_t, graph_t, inferred_t = await asyncio.gather(
        execute_vector_search(customer_id, query=reformulated_query),
        execute_bm25_search(customer_id, query=reformulated_query),
        execute_graph_search(customer_id, entities=entity_dicts),
        execute_inferred_edge_search(customer_id, entities=entity_dicts),
        return_exceptions=True,
    )

    def _ok(r: Any) -> dict[str, Any]:
        return r if not isinstance(r, BaseException) else {"hits": [], "error": str(r)}

    return {
        "grounding": _bundle_to_compact(bundle),
        "channels": {
            "vector": _ok(vector_t),
            "bm25": _ok(bm25_t),
            "graph": _ok(graph_t),
            "inferred_edge": _ok(inferred_t),
        },
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


async def execute_read_inferred_edge_evidence(
    customer_id: str,
    *,
    edge_id: str,
) -> dict[str, Any]:
    """Fetch the chunks the LLM was reasoning over when it wrote a
    given inferred edge's `why`.

    The producer (`services/ingestion/inferred_edges/`) stages bundles
    in the `inferred_edges_queue` table; finalized rows materialize on
    `graph_edges.properties->>'why'` + `properties->>'evidence_chunk_ids'`.
    We surface the evidence chunk IDs and hydrate their content.
    """
    sql_edge = """
        SELECT properties->>'why' AS why,
               properties->'evidence_chunk_ids' AS evidence_chunk_ids,
               properties->>'producer_model' AS producer_model
        FROM graph_edges
        WHERE customer_id = $1 AND edge_id = $2
        LIMIT 1
    """
    async with with_tenant(customer_id) as conn:
        edge_row = await conn.fetchrow(sql_edge, customer_id, edge_id)
        if not edge_row:
            return {"edge_id": edge_id, "found": False}

        chunk_ids: list[str] = []
        raw_ids = edge_row["evidence_chunk_ids"]
        if isinstance(raw_ids, list):
            chunk_ids = [str(x) for x in raw_ids]

        chunks: list[dict[str, Any]] = []
        if chunk_ids:
            chunk_rows = await conn.fetch(
                """
                SELECT chunk_id, doc_id, content, kind
                FROM chunks
                WHERE customer_id = $1 AND chunk_id = ANY($2::text[])
                LIMIT $3
                """,
                customer_id,
                chunk_ids,
                INFERRED_EDGE_HYDRATION_CHUNKS,
            )
            chunks = [
                {
                    "chunk_id": r["chunk_id"],
                    "doc_id": r["doc_id"],
                    "content": r["content"],
                    "kind": r["kind"],
                }
                for r in chunk_rows
            ]

        return {
            "edge_id": edge_id,
            "found": True,
            "why": edge_row["why"],
            "producer_model": edge_row["producer_model"],
            "evidence_chunks": chunks,
        }


# ============================================================
# Tool definitions for LiteLLM (OpenAI-style schemas)
# ============================================================

def tool_definitions() -> list[dict[str, Any]]:
    """Return the tool schemas LiteLLM forwards to Fireworks.

    Order matches the prompt's tool list to preserve cache-friendliness
    (system prompt + tool defs are the cached prefix; reordering busts
    the cache).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": "vector_search",
                "description": "pgvector cosine search on chunks for `query`. Returns ranked chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Raw natural-language query."},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "bm25_search",
                "description": "pg_search Lucene-style BM25 on chunks for `query`. Returns ranked chunks.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_search",
                "description": (
                    "1-hop walk from explicit entity IDs to attached Documents. "
                    "Pass entities from the <grounding> block."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity_type": {"type": "string"},
                                    "canonical_id": {"type": "string"},
                                },
                                "required": ["entity_type", "canonical_id"],
                            },
                        },
                        "top_k": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                    "required": ["entities"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "inferred_edge_search",
                "description": (
                    "Walk INFERRED Doc-Doc edges. Accepts either `entities` (resolves to Docs internally) "
                    "or `doc_ids` (direct). Returns linked docs with their LLM-written `why` strings."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "entity_type": {"type": "string"},
                                    "canonical_id": {"type": "string"},
                                },
                                "required": ["entity_type", "canonical_id"],
                            },
                        },
                        "doc_ids": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "parallel_multi_query",
                "description": "Fan out 2-5 sub-queries through vector + bm25 in parallel.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
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
                "name": "expand_inferred_neighbors",
                "description": "Walk INFERRED edges out of a single doc. Returns its neighbors + `why` strings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "max": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                    "required": ["doc_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "expand_entity_cluster",
                "description": (
                    "Resolve a list of canonical_ids into their full cluster (aliases + primary). "
                    "Useful when grounding returned an alias but you want the whole entity-cluster."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "canonical_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1},
                        "label": {"type": "string"},
                    },
                    "required": ["canonical_ids", "label"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_doc_chunks",
                "description": "Pull more chunks from a doc you want to read fully (default ~3, up to ~10).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "max": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                        "query": {"type": "string"},
                    },
                    "required": ["doc_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "graph_walk",
                "description": (
                    "Thin 1-hop bidirectional walk from a single anchor canonical_id. "
                    "IDF-ranked top-20. Use to BFS the graph one neighbor at a time."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "anchor_canonical_id": {"type": "string"},
                        "edge_types": {"type": "array", "items": {"type": "string"}},
                        "top_k": {"type": "integer", "minimum": 1, "maximum": _HARD_TOP_K_CAP},
                    },
                    "required": ["anchor_canonical_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "reissue_query",
                "description": (
                    "Re-run grounding + the 4-channel turn-1 fan-out with a reformulated query. "
                    "Use ONLY when the original query was malformed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reformulated_query": {"type": "string"},
                    },
                    "required": ["reformulated_query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_inferred_edge_evidence",
                "description": (
                    "Fetch the chunks the LLM was reasoning over when an INFERRED edge's `why` "
                    "was produced. Use when the `why` is ambiguous and you need context."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "edge_id": {"type": "string"},
                    },
                    "required": ["edge_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "need_deeper",
                "description": (
                    "Soft-budget extension. Requests +10 tool calls. Provide a `reason` string; "
                    "max 2 extensions across the loop."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reason": {"type": "string"},
                    },
                    "required": ["reason"],
                },
            },
        },
    ]


# ============================================================
# Dispatcher
# ============================================================

# Map from JSON Schema tool name -> executor.
# `need_deeper` is handled out-of-band by the loop (it's a budget signal,
# not a retrieval call), so it deliberately routes to a sentinel.
TOOL_REGISTRY: dict[str, Callable[..., Awaitable[dict[str, Any]]]] = {
    "vector_search": execute_vector_search,
    "bm25_search": execute_bm25_search,
    "graph_search": execute_graph_search,
    "inferred_edge_search": execute_inferred_edge_search,
    "parallel_multi_query": execute_parallel_multi_query,
    "expand_inferred_neighbors": execute_expand_inferred_neighbors,
    "expand_entity_cluster": execute_expand_entity_cluster,
    "fetch_doc_chunks": execute_fetch_doc_chunks,
    "graph_walk": execute_graph_walk,
    "reissue_query": execute_reissue_query,
    "read_inferred_edge_evidence": execute_read_inferred_edge_evidence,
}


async def dispatch_tool_call(
    customer_id: str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Route a model-emitted tool call to its executor.

    Catches and packages exceptions so the agent loop never crashes on a
    single tool failure — instead the agent sees `{"error": "..."}` and
    can recover (skip, retry differently, surface as low-confidence).
    """
    executor = TOOL_REGISTRY.get(tool_name)
    if executor is None:
        return {"error": f"unknown tool: {tool_name}"}
    try:
        return await executor(customer_id=customer_id, **arguments)
    except TypeError as exc:
        # Wrong kwargs — the JSON Schema should have caught this, but
        # be defensive so a model misfire doesn't 500 the loop.
        log.warning(
            "agent.tool_arg_mismatch",
            tool_name=tool_name,
            error=str(exc),
        )
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
    "TOOL_REGISTRY",
    "dispatch_tool_call",
    "tool_definitions",
]


# Suppress unused-import warnings for symbols re-exported via the module
# but only referenced for docstring symmetry / future hookups.
_unused = (datetime, dataclasses, math, _walk_neighbors)
