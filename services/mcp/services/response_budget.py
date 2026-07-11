"""Server-side byte budget for MCP tool responses.

Background
----------
MCP harnesses (Claude Code, Codex, etc.) cap individual tool-result
sizes around 25KB. Above the cap, the harness spills the result to a
file and replaces the tool result with a "read this file sequentially"
notice — which forces the agent into ad-hoc jq/Read work and defeats
the purpose of structured retrieval.

A real production trace: a single `search_knowledge` call with
`top_k≈20` returned 196KB / 12 docs. 90% of the bytes were
`chunks[].content`, dominated by 7 oversized code_graph chunks
(7-30KB each pre-symbol-rechunk). That request triggered the spill
fallback in Claude Code; the agent then had to dump the entire 196KB
back into context to read it.

This module is the safety net: enforce a byte cap on the assembled
response BEFORE returning it to FastMCP. The companion ingestion-side
fix (services/ingestion/code_graph/chunking.py in prbe-knowledge)
keeps individual chunks bounded going forward; this cap handles
adversarial cases (high top_k, dense `related_entities`, future
heavy fields) and the rollover window where existing 30KB chunks
still live in the index.

Truncation order
----------------
Cheap → expensive:

  1. Drop `graph_evidence` from tail chunks. It's near-zero bytes in
     practice (42 bytes / 196KB on the production trace) so this rarely
     helps, but it's free and may matter once graph retrieval lights
     up more frequently.
  2. Drop entire tail chunks. Documents are sorted by `score` desc and
     chunks within a doc are sorted by `rank_in_doc` asc, so dropping
     from the tail loses the lowest-relevance content first.
  3. As a last resort, truncate `chunks[].content` of the surviving
     tail chunk to ~500 chars + `content_truncated: true`. Avoids
     dropping the chunk entirely when even one tail chunk would put us
     under budget.
  4. Drop tail `related_entities` only after the primary evidence has
     been reduced as far as possible.

If an irreducible or future field still exceeds the hard cap, return a
small 413 diagnostic with the upstream trace id instead of handing an
oversized result to the client.

We stop trimming the moment the compact JSON's UTF-8 byte length is under
the target. Exact serialization is fast enough for response sizes <100KB;
no estimator is needed.

Truncation markers in the response
----------------------------------
  - `truncated: bool` — set True if any trimming occurred.
  - `dropped_chunk_count: int` — chunks dropped entirely (does not
     count graph_evidence drops or content truncation).
  - `dropped_result_count: int` — entity/document results dropped
     wholesale after their tail chunks were exhausted.
  - `cursor: null` — placeholder. Stateful continuation needs an
     offset parameter on the upstream retrieve call (cross-repo
     change in prbe-knowledge), deferred to a follow-up. Today the
     agent recovers by lowering `top_k` and re-querying, or by
     calling `get_source` on specific doc_ids.

`confidence_breakdown` is preserved during normal relevance-aware
trimming. It is omitted only from the emergency 413 envelope.
`related_entities` is trimmed after result/chunk trimming because the
caller can recover that graph context with a narrower follow-up query.
"""

from __future__ import annotations

import json
from typing import Any

# MCP harness spills around 25KB. Target with headroom for envelope +
# JSON encoding overhead in the wrapper, and for the conservative
# prefix-encoding sometimes used by stdio transports.
MAX_RESPONSE_BYTES_TARGET = 20_000
MAX_RESPONSE_BYTES_HARD = 24_000

# Last-resort content truncation length when even dropping the tail
# chunk wouldn't fit. Keeps enough text for the agent to recognize
# whether the chunk was relevant; full body is always available via
# `get_source`.
CONTENT_TRUNCATION_CHARS = 500


def _serialize_json(payload: Any) -> str:
    """Serialize a JSON value using the exact MCP wire settings."""
    return json.dumps(
        payload,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def serialize_tool_response(payload: dict[str, Any]) -> str:
    """Serialize one MCP tool payload exactly as the byte budget measures it."""
    return _serialize_json(payload)


def _encoded_size(payload: Any) -> int:
    """UTF-8 byte size of any value under the MCP JSON encoding."""
    return len(_serialize_json(payload).encode("utf-8"))


def _measure(payload: dict[str, Any]) -> int:
    """UTF-8 byte size of the compact JSON emitted on the MCP wire."""
    return _encoded_size(payload)


def _emergency_response(response: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded diagnostic when relevance-preserving trims cannot fit."""
    results = response.get("results") or []
    emergency: dict[str, Any] = {
        "error": "response exceeded MCP hard byte limit after trimming",
        "error_code": "response_too_large",
        "status": 413,
        "truncated": True,
        "dropped_chunk_count": sum(
            len(result.get("chunks") or [])
            for result in results
            if isinstance(result, dict)
        ),
        "dropped_result_count": len(results),
        "dropped_related_entity_count": len(response.get("related_entities") or []),
        "cursor": None,
    }
    if response.get("trace_id"):
        emergency["trace_id"] = str(response["trace_id"])[:256]
    return emergency


def fit_response_to_budget(
    response: dict[str, Any],
    *,
    target_bytes: int = MAX_RESPONSE_BYTES_TARGET,
    hard_bytes: int = MAX_RESPONSE_BYTES_HARD,
) -> dict[str, Any]:
    """Return response possibly trimmed to fit MCP tool-result cap.

    Mutates a shallow copy; original input is not modified.

    Adds five keys to result-bearing responses:
      - truncated (bool)
      - dropped_chunk_count (int)
      - dropped_result_count (int)
      - dropped_related_entity_count (int)
      - cursor (None — placeholder for future stateful continuation)

    No-op when:
      - response has no `results` key and fits under hard_bytes
      - response already fits under target_bytes

    Args:
        response: the upstream `client.retrieve` payload. Expected
            shape: `{results: [{chunks: [{content, graph_evidence,
            ...}, ...], ...}, ...], ...}`. The `results` array is
            polymorphic — Document items have `chunks`; Entity items
            have empty/no chunks but get popped wholesale when over
            budget. Anything else passes through untouched.
        target_bytes: trimming target. Below this, we stop.
        hard_bytes: emergency ceiling. If trimming hits this without
            getting under target, we keep going down to hard_bytes.
            The expected pathological case (one mega-chunk that
            exceeds the cap on its own) is covered by content
            truncation.

    Returns:
        A dict that's either the input unchanged (with markers added)
        or a trimmed copy (with markers reflecting the trim).
    """
    # Error/alternate envelopes do not need relevance-aware trimming, but a
    # future oversized field must still respect the hard wire limit.
    if not isinstance(response, dict):
        return response
    if "results" not in response:
        if _measure(response) <= hard_bytes:
            return response
        return _emergency_response(response)

    out = dict(response)  # shallow copy; we'll deep-copy chunks lists below
    out.setdefault("truncated", False)
    out.setdefault("dropped_chunk_count", 0)
    out.setdefault("dropped_result_count", 0)
    out.setdefault("dropped_related_entity_count", 0)
    out.setdefault("cursor", None)

    if _measure(out) <= target_bytes:
        return out

    # Materialize a working copy of results (and per-result chunks) so
    # we can mutate without affecting the caller's input. Entity items
    # keep their no-chunks wire shape and are candidates for whole-item
    # drops.
    docs: list[dict[str, Any]] = []
    for result in out.get("results") or []:
        copied = dict(result)
        if "chunks" in result or result.get("node_type") != "Entity":
            copied["chunks"] = [
                dict(chunk) for chunk in result.get("chunks") or []
            ]
        docs.append(copied)
    out["results"] = docs
    related_entities = out.get("related_entities")
    if isinstance(related_entities, list):
        related_entities = [
            dict(entity) if isinstance(entity, dict) else entity
            for entity in related_entities
        ]
        out["related_entities"] = related_entities

    current_size = _measure(out)

    # Stage 1: drop graph_evidence on tail chunks (rare savings, but
    # free when present). Replacing one JSON value with [] has an exact,
    # locally computable saving, so avoid serializing the whole payload once
    # per chunk.
    evidence_dropped = 0
    if current_size > target_bytes:
        for d in reversed(docs):
            for c in reversed(d.get("chunks") or []):
                evidence = c.get("graph_evidence")
                if evidence:
                    current_size -= _encoded_size(evidence) - _encoded_size([])
                    c["graph_evidence"] = []
                    evidence_dropped += 1
                    if current_size <= target_bytes:
                        break
            if current_size <= target_bytes:
                break

    # Stage 2: drop entire tail chunks. Walk docs from the tail (lowest
    # `score`); within each doc, drop chunks from the tail too (highest
    # `rank_in_doc`). When a doc has no chunks left, drop the doc.
    #
    # Floor: preserve one top-ranked result. For a Document, preserve one
    # chunk when possible; an Entity legitimately has no chunks. When the
    # floor is still over budget, stages 3-4 shrink recoverable fields before
    # the explicit emergency envelope is used.
    dropped = 0
    dropped_results = 0
    if current_size > target_bytes:
        total_chunks = sum(len(d.get("chunks") or []) for d in docs)
        while docs and current_size > target_bytes:
            chunks = docs[-1].get("chunks") or []
            if chunks and total_chunks > 1:
                if len(chunks) == 1 and len(docs) > 1:
                    removed_doc = docs.pop()
                    current_size -= _encoded_size(removed_doc) + 1
                    total_chunks -= 1
                    dropped += 1
                    dropped_results += 1
                else:
                    before_count = len(chunks)
                    removed_chunk = chunks.pop()
                    current_size -= _encoded_size(removed_chunk)
                    if before_count > 1:
                        current_size -= 1  # comma between array elements
                    total_chunks -= 1
                    dropped += 1
                continue
            if len(docs) > 1:
                removed_doc = docs.pop()
                current_size -= _encoded_size(removed_doc) + 1
                total_chunks -= len(chunks)
                dropped += len(chunks)
                dropped_results += 1
                continue
            # Preserve one top-ranked result. Stage 3 can shrink its final
            # chunk; entity-only results fall through to the hard-cap guard.
            break

    # Stage 3: last-resort content truncation on the surviving tail
    # chunk. Triggers in two cases:
    #   (a) one mega-chunk that's itself > hard_bytes (single oversized
    #       symbol surviving the rollover window from the chunker fix)
    #   (b) Stage 2 hit its 1-doc/1-chunk floor while still over target
    #
    # Either way, the last chunk's content gets clipped to a short
    # preview + `content_truncated: true` so the agent sees something
    # meaningful and can drill in via `get_source` for the full body.
    out["dropped_chunk_count"] = int(out["dropped_chunk_count"]) + dropped
    out["dropped_result_count"] = int(out["dropped_result_count"]) + dropped_results
    out["truncated"] = bool(
        out["truncated"]
        or dropped > 0
        or dropped_results > 0
        or evidence_dropped > 0
    )
    # Counter digit growth and false -> true can shift the estimate by a few
    # bytes. Re-measure once before the hard-limit stages.
    current_size = _measure(out)

    content_truncated = False
    if docs and current_size > hard_bytes:
        last_doc = docs[-1]
        last_chunks = last_doc.get("chunks") or []
        if last_chunks:
            last_chunk = last_chunks[-1]
            content = last_chunk.get("content")
            if isinstance(content, str) and len(content) > CONTENT_TRUNCATION_CHARS:
                last_chunk["content"] = content[:CONTENT_TRUNCATION_CHARS]
                last_chunk["content_truncated"] = True
                content_truncated = True
                out["truncated"] = True
                current_size = _measure(out)

    # Stage 4: related graph nodes are useful but recoverable. Trim them only
    # when the primary result has already been reduced as far as possible.
    dropped_related = 0
    if isinstance(related_entities, list) and related_entities and current_size > hard_bytes:
        out["truncated"] = True
        current_size = _measure(out)
        while related_entities and current_size > hard_bytes:
            before_count = len(related_entities)
            removed_entity = related_entities.pop()
            current_size -= _encoded_size(removed_entity)
            if before_count > 1:
                current_size -= 1  # comma between array elements
            old_count = int(out["dropped_related_entity_count"])
            new_count = old_count + 1
            current_size += _encoded_size(new_count) - _encoded_size(old_count)
            out["dropped_related_entity_count"] = new_count
            dropped_related += 1

    # Mark truncated if we actually changed anything observable. The related
    # count is updated inside its loop so the tracked byte size stays exact.
    out["truncated"] = bool(
        out["truncated"]
        or dropped_related > 0
        or content_truncated
    )

    # A future response field can still be unexpectedly huge. Never hand a
    # client an over-hard-limit payload: return a small, explicit diagnostic
    # with the upstream trace id instead of triggering opaque client spill or
    # truncation behavior.
    if _measure(out) > hard_bytes:
        return _emergency_response(response)

    return out
