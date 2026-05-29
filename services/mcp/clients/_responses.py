"""Response transforms that strip diagnostic fields agents don't reason
over, while preserving signals they actually use (top-level score,
recall hints, router decisions).

Compaction is on by default; tools expose `verbose=True` for the rare
case the caller needs the raw upstream payload (full retriever-score
breakdown, timing, trace ids).
"""

from __future__ import annotations

from typing import Any

# Top-level fields stripped by default. These are pure server
# instrumentation. We deliberately KEEP `total_candidates` (recall
# hint — agent needs it to decide whether to raise top_k),
# `extracted_entities` and `applied_temporal` (router-decision
# surface — agents need to see when the router misinterprets a
# query, otherwise they keep re-running broken phrasings).
_TOP_LEVEL_DROP = frozenset(
    {
        "timing_ms",
        "applied_sort",
        "applied_entity_filter",
        "applied_mode",
        "applied_doc_types",
        "aggregation",
        "router_hit_cache",
        "trace_id",
    }
)

# Per-Document fields stripped by default. We deliberately KEEP `score`
# (top-line confidence), `node_type` (so the agent can branch
# Document/Entity), and `matched_via` (provenance — carries the LLM's
# `why` when an inferred-edge result surfaced via Doc-Doc walk). The
# per-retriever breakdown in `retriever_scores` stays dropped — it's
# noise unless debugging.
_DOC_RESULT_DROP = frozenset(
    {
        "doc_version",
        "rank",
        "retriever_scores",
    }
)

# Per-Entity fields stripped by default. `rank` is implied by position
# in `results[]`; everything else is signal the agent uses to navigate
# (label, display_name, attached_doc_ids, edge_types, doc_count) or to
# weigh relevance (score, matched_via, properties).
_ENTITY_RESULT_DROP = frozenset(
    {
        "rank",
    }
)

# Per-chunk fields stripped by default. We keep `score`, `content`, and
# `graph_evidence` (a list of {edge_type, confidence, via_entity, reason}
# entries — the agent's only evidence the chunk actually grounds the
# query against the knowledge graph). `chunk_id` and `rank_in_doc` are
# pure server bookkeeping.
_CHUNK_DROP = frozenset(
    {
        "chunk_id",
        "rank_in_doc",
    }
)

# Fields stripped from a single-doc get_source response.
# `metadata` is dropped because source-system internals (Notion block
# trees, Slack channel IDs, etc.) usually duplicate `content` and
# rarely help the caller.
_SOURCE_DROP = frozenset(
    {
        "doc_version",
        "source_id",
        "chunk_count",
        "body_size_bytes",
        "entities",
        "ingested_at",
        "deleted_at",
        "metadata",
    }
)


def _strip(payload: dict[str, Any], drop: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in drop}


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Compact one polymorphic QueryResult.

    Branches on `node_type`:
      - "Entity" -> drop only `rank`; pass everything else (label,
        canonical_id, display_name, properties, attached_doc_ids,
        edge_types, doc_count, score, matched_via).
      - "Document" (or missing/unknown -> default): drop per-doc
        diagnostics and compact each nested chunk.
    """
    if result.get("node_type") == "Entity":
        return _strip(result, _ENTITY_RESULT_DROP)
    compacted = _strip(result, _DOC_RESULT_DROP)
    chunks = result.get("chunks") or []
    compacted["chunks"] = [_strip(c, _CHUNK_DROP) for c in chunks]
    return compacted


def compact_search(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip top-level + per-result + per-chunk diagnostics.

    Operates on the polymorphic shape: `results[*]` is a discriminated
    union of Document (with `chunks[*]` nested) and Entity. The top-level
    `confidence_breakdown` aggregate passes through (router signal — agents
    need to see it to decide whether to widen the search).
    """
    out = _strip(payload, _TOP_LEVEL_DROP)
    results = payload.get("results") or []
    out["results"] = [_compact_result(r) for r in results]
    return out


def compact_query(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip top-level + per-result + per-chunk diagnostics from a
    synthesized-answer response.

    The synthesized `answer`, `citations`, `insufficient_context`, and
    `model` fields pass through unchanged. Underlying polymorphic results
    (`results[*]`) are compacted the same way as `compact_search` so the
    two endpoints expose the same shape to callers; only debug/telemetry
    fields are removed.
    """
    out = _strip(payload, _TOP_LEVEL_DROP)
    results = payload.get("results")
    if results is not None:
        out["results"] = [_compact_result(r) for r in results]
    return out


def compact_source(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip source-system internals from a single-doc response."""
    return _strip(payload, _SOURCE_DROP)


def compact_source_view(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip internals from a bounded source view.

    Bounded views intentionally keep navigation/cost-safety metadata
    (`chunk_count`, `body_size_bytes`, `max_bytes`, `limit_lines`,
    `truncated`, cursors, and sections), because agents need those fields
    to decide whether and how to drill down further.
    """
    return _strip(
        payload,
        frozenset(
            {
                "source_id",
                "metadata",
                "entities",
                "ingested_at",
                "deleted_at",
            }
        ),
    )
