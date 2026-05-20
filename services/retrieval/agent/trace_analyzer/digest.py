"""Per-trace summarization — pure function, no I/O.

Reads one blob (with its `_db` stitch from the loader) and emits a flat
dict the nightly orchestrator can cluster on without re-reading the
full blob. The full blob is fetched on-demand by sub-agents via the
`bucket_name`+`blob_key` fields below.

Why this is its own layer: the orchestrator works at the population
level (cluster traces by failure shape, score by impact) and shouldn't
need to walk per-turn message arrays. Sub-agents work at the trace
level and DO need the full blob. The digest is the routing index.
"""

from __future__ import annotations

from typing import Any

# Mandatory turn-1 channels per the agent prompt. Anything missing from
# `turn_1_tools_fired` is logged as `turn_1_missed_channels` so the
# nightly orchestrator can cluster on "agent skipped the recall
# guarantee" failures. (Note: with the prefanout cutover, this may now
# fire on every trace because the channels run pre-loop; review and
# retire the signal if so.)
_REQUIRED_TURN_1_CHANNELS = frozenset({
    "vector_search",
    "bm25_search",
    "graph_search",
    "inferred_edge_search",
})


def _mean_or_none(values: list[float | None]) -> float | None:
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _has_tool(tools_fired: list[str], name: str) -> bool:
    return name in tools_fired


def summarize_trace(blob: dict[str, Any]) -> dict[str, Any]:
    """Flatten one trace blob into the orchestrator's digest shape.

    Pure. No I/O. Tolerant of missing fields (e.g. blobs from earlier
    schema versions) — uses `.get(...)` with defaults throughout.
    """
    db = blob.get("_db") or {}
    timing = blob.get("timing_ms") or {}
    gathered = blob.get("gathered") or {}
    notes = (gathered.get("gatherer_notes") if isinstance(gathered, dict) else {}) or {}

    tools_fired: list[str] = list(blob.get("tools_fired") or [])
    turn_1_tools: list[str] = list(blob.get("turn_1_tools_fired") or [])
    cache_rates: list[float | None] = list(blob.get("cache_hit_rates") or [])
    turn_latencies: list[float] = list(blob.get("turn_latencies_ms") or [])
    reasoning_per_turn = list(blob.get("reasoning_per_turn") or [])
    fingerprints_per_turn: list[str | None] = list(
        blob.get("system_fingerprints_per_turn") or []
    )

    missed = sorted(_REQUIRED_TURN_1_CHANNELS - set(turn_1_tools))

    return {
        # Identity + routing
        "trace_id": blob.get("trace_id") or db.get("request_id"),
        "request_id": db.get("request_id"),
        "customer_id": blob.get("customer_id") or db.get("customer_id"),
        "occurred_at": db.get("occurred_at"),
        "event_type": db.get("event_type"),
        "model": blob.get("model"),
        "query": blob.get("query") or "",
        # Bucket+key let sub-agents fetch the full blob from R2 without
        # a DB round-trip. The orchestrator passes these through verbatim.
        # Authoritative bucket name comes from the loader's `_db` stitch
        # (uses `bucket_for(customer_id)`); the fallback formula is only
        # for blobs that lack the stitch (e.g. fixture data in tests).
        "bucket_name": db.get("bucket_name") or _bucket_name_for(
            blob.get("customer_id") or db.get("customer_id") or ""
        ),
        "blob_key": db.get("trace_blob_key"),
        # Status / outcome
        "status": blob.get("status") or db.get("gatherer_status"),
        "confidence": notes.get("confidence") or db.get("confidence"),
        "chunk_count": len(gathered.get("chunks") or []) if isinstance(gathered, dict) else 0,
        "entity_count": len(gathered.get("entities") or []) if isinstance(gathered, dict) else 0,
        "dropped_count": db.get("dropped_count") or 0,
        # Loop shape
        "turn_count": blob.get("turn_count") or 0,
        "tool_calls_count": blob.get("tool_calls_count") or db.get("tool_calls_count") or 0,
        "extensions_used": blob.get("extensions_used") or db.get("need_deeper_extensions") or 0,
        "prose_retries": blob.get("prose_retries") or 0,
        # Tool sequence (ordered) — the analyzer clusters on this shape
        "tool_call_sequence": tools_fired,
        "turn_1_tools_fired": turn_1_tools,
        "turn_1_missed_channels": missed,
        # Exploration signals — useful binary clusters
        "had_reissue_query": _has_tool(tools_fired, "reissue_query"),
        "had_expand_inferred_neighbors": _has_tool(
            tools_fired, "expand_inferred_neighbors"
        ),
        "had_need_deeper": blob.get("extensions_used", 0) > 0,
        "had_reasoning_per_turn": any(r for r in reasoning_per_turn),
        # Latency
        "grounding_ms": timing.get("grounding_ms"),
        "prefanout_ms": timing.get("prefanout_ms"),
        "agent_ms": timing.get("agent_ms"),
        "agent_loop_ms": timing.get("agent_loop_ms"),
        "agent_tools_ms": timing.get("agent_tools_ms"),
        "extraction_ms": timing.get("extraction_ms"),
        # Cache
        "cache_hit_rate_mean": _mean_or_none(cache_rates),
        "cache_hit_rate_per_turn": cache_rates,
        # Determinism telemetry — seed + per-turn provider fingerprint.
        # `seed_changed` flags a backend roll mid-query (same query, two
        # distinct fingerprints across turns); the strong signal for
        # within-query reproducibility breaking.
        "seed": blob.get("seed"),
        "system_fingerprints_per_turn": fingerprints_per_turn,
        "system_fingerprint_changed_mid_query": (
            len({f for f in fingerprints_per_turn if f}) > 1
        ),
        # Per-turn LLM latency — slow turns surface model-side issues
        "turn_latencies_ms": turn_latencies,
        # Prefanout coverage
        "prefanout_hit_counts": dict(blob.get("prefanout_hit_counts") or {}),
        # Response sizing (from DB summary)
        "response_size_bytes": db.get("response_size_bytes"),
    }


def _bucket_name_for(customer_id: str) -> str:
    """Customer's R2 bucket name. Mirrors the naming convention from
    `customers.r2_bucket` for new tenants: `prbe-<customer_id>`. The
    canonical value comes from the DB row, but sub-agents are given the
    digest's `bucket_name` field — when present, that's authoritative.
    """
    if not customer_id:
        return ""
    return f"prbe-{customer_id}"


__all__ = ["summarize_trace"]
