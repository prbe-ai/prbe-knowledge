"""Per-query agent trace blob — full transcript persisted to R2.

The query_traces row carries the summary (status, tool_calls_count, confidence,
cache_hit_rate, …). The blob this module produces carries the per-turn detail
nobody wants in Postgres: full state.messages, per-turn latency, the final
GathererOutput, the pre-fan-out hit counts, etc.

Flow:
    run_gatherer (loop.py)
        ├── finishes the loop OR hits an exit path (timeout/503/no-llm)
        └── stashes raw refs onto request.state in a try/finally:
              search_agent_loop_state, search_agent_gathered,
              search_agent_status, search_agent_timing,
              search_agent_query, search_agent_should_persist
    middleware._persist_trace_blob_r2 (BackgroundTask, post-flush)
        ├── reads the stash
        ├── build_trace_blob(...)        ← assembles JSON dict here, NOT in
        │                                  the request path (CPU is post-flush)
        ├── persist_trace_blob_to_r2(...)  ← gzip + PUT
        └── sets request.state.trace_blob_key on success
    middleware._build_and_write_trace (next BackgroundTask in the chain)
        └── reads trace_blob_key, writes the query_traces row pointing at it

All functions in this module return-or-log on failure; nothing here is
allowed to escape into the BackgroundTask chain. Observability that 500s
the user request defeats the whole purpose.
"""

from __future__ import annotations

import gzip
import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from shared.exceptions import StorageUnavailable
from shared.storage import get_store

if TYPE_CHECKING:
    from services.retrieval.agent.loop import LoopState
    from services.retrieval.agent.models import GathererOutput, GathererStatus

log = logging.getLogger(__name__)

# Bumped when the blob JSON shape changes in a way that breaks the
# nightly trace-analyzer (renamed/restructured fields). Old blobs stay
# readable indefinitely; the analyzer filters by version when needed.
TRACE_BLOB_SCHEMA_VERSION = 1


def build_trace_blob(
    *,
    state: LoopState | None,
    gathered: GathererOutput | None,
    status: GathererStatus | None,
    timing: dict[str, float],
    query: str,
    customer_id: str,
    trace_id: str,
    model: str,
) -> dict[str, Any]:
    """Assemble the JSON-serializable trace payload.

    Pure function. No I/O. Tolerates None for `state`, `gathered`, and
    `status` so the 503 / pre-loop failure paths still produce a
    structured blob. The nightly analyzer keys on `status` to bucket
    failure shapes.
    """
    blob: dict[str, Any] = {
        "schema_version": TRACE_BLOB_SCHEMA_VERSION,
        "trace_id": trace_id,
        "customer_id": customer_id,
        "query": query,
        "model": model,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "timing_ms": dict(timing),
    }

    if state is not None:
        blob["messages"] = state.messages
        blob["tools_fired"] = list(state.tools_fired)
        blob["turn_1_tools_fired"] = list(state.turn_1_tools_fired)
        blob["turn_count"] = state.turn_count
        blob["tool_calls_count"] = state.tool_calls_count
        blob["extensions_used"] = state.extensions_used
        blob["cache_hit_rates"] = list(state.cache_hit_rates)
        blob["turn_latencies_ms"] = list(state.turn_latencies_ms)
        blob["tool_latencies_ms"] = list(state.tool_latencies_ms)
        blob["prose_retries"] = state.prose_retries
        # Pre-fan-out captured on LoopState by run_gatherer so the
        # analyzer can correlate channel coverage with curated outcomes.
        blob["prefanout"] = state.prefanout
        blob["prefanout_hit_counts"] = dict(state.prefanout_hit_counts)
        # Per-turn chain-of-thought from message.reasoning_content (gpt-oss
        # harmony `analysis` block, normalized by LiteLLM). NOT echoed
        # back into the next turn (OpenAI chat-completion contract
        # round-trips only role/content/tool_calls), so without this field
        # the agent's "why" per tool call is unrecoverable. May be all-
        # None when the provider doesn't emit reasoning for this model.
        blob["reasoning_per_turn"] = list(state.reasoning_per_turn)
    else:
        # Pre-loop failure (e.g. grounding raised before state was constructed
        # in a future refactor). Keep the keys present so analyzer schema
        # stays stable.
        blob["messages"] = []
        blob["tools_fired"] = []
        blob["turn_1_tools_fired"] = []
        blob["turn_count"] = 0
        blob["tool_calls_count"] = 0
        blob["extensions_used"] = 0
        blob["cache_hit_rates"] = []
        blob["turn_latencies_ms"] = []
        blob["tool_latencies_ms"] = []
        blob["prose_retries"] = 0
        blob["prefanout"] = {}
        blob["prefanout_hit_counts"] = {}
        blob["reasoning_per_turn"] = []

    if gathered is not None:
        blob["gathered"] = gathered.model_dump(mode="json")
    else:
        blob["gathered"] = None

    return blob


def compute_blob_key(trace_id: str, now: datetime) -> str:
    """Return the R2 object key for this trace.

    Customer isolation is implicit in the per-tenant bucket name — no
    customer_id in the key. Date prefix lets the nightly trace-analyzer
    list-by-prefix instead of scanning the whole bucket.
    """
    return f"search-traces/{now:%Y-%m-%d}/{trace_id}.json.gz"


async def persist_trace_blob_to_r2(
    customer_id: str,
    key: str,
    payload: dict[str, Any],
) -> str | None:
    """gzip + PUT the trace blob to the tenant's R2 bucket.

    Returns the key on success, None on any failure. Wrapped in a broad
    try/except so a misshapen payload, R2 outage, or boto3 quirk cannot
    raise into the BackgroundTask chain — losing a trace must never
    propagate into other telemetry (usage_event, query_traces row).
    """
    try:
        body = gzip.compress(json.dumps(payload, default=str).encode("utf-8"))
        store = get_store()
        bucket = await store.bucket_for(customer_id)
        await store.put(bucket, key, body, content_type="application/json")
        return key
    except StorageUnavailable as exc:
        log.warning(
            "trace_blob.r2_unavailable",
            extra={"customer_id": customer_id, "key": key, "error": str(exc)},
        )
        return None
    except Exception as exc:
        log.warning(
            "trace_blob.persist_failed",
            extra={
                "customer_id": customer_id,
                "key": key,
                "error": str(exc),
                "error_class": type(exc).__name__,
            },
        )
        return None


__all__ = [
    "TRACE_BLOB_SCHEMA_VERSION",
    "build_trace_blob",
    "compute_blob_key",
    "persist_trace_blob_to_r2",
]
