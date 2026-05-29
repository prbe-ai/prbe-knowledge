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

We stop trimming the moment `len(json.dumps(payload))` is under the
target. `json.dumps` is fast enough for response sizes <100KB; no
fancy estimator needed.

Truncation markers in the response
----------------------------------
  - `truncated: bool` — set True if any trimming occurred.
  - `dropped_chunk_count: int` — chunks dropped entirely (does not
     count graph_evidence drops or content truncation).
  - `cursor: null` — placeholder. Stateful continuation needs an
     offset parameter on the upstream retrieve call (cross-repo
     change in prbe-knowledge), deferred to a follow-up. Today the
     agent recovers by lowering `top_k` and re-querying, or by
     calling `get_source` on specific doc_ids.

Caveat: `confidence_breakdown` and `related_entities` are NOT
truncated. They're bounded (10-20 entries each) and surface useful
overall-result-set signals. Touching them would corrupt the agent's
read of how trustworthy the truncated set is.
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


def _measure(payload: dict[str, Any]) -> int:
    """Compact JSON byte size of the payload.

    `separators=(',',':')` mirrors MCP wire encoding: no whitespace.
    Whitespace would over-estimate by 5-10% and cause unnecessary
    trimming.
    """
    return len(json.dumps(payload, separators=(",", ":"), default=str))


def fit_response_to_budget(
    response: dict[str, Any],
    *,
    target_bytes: int = MAX_RESPONSE_BYTES_TARGET,
    hard_bytes: int = MAX_RESPONSE_BYTES_HARD,
) -> dict[str, Any]:
    """Return response possibly trimmed to fit MCP tool-result cap.

    Mutates a shallow copy; original input is not modified.

    Adds three keys to the returned dict:
      - truncated (bool)
      - dropped_chunk_count (int)
      - cursor (None — placeholder for future stateful continuation)

    No-op when:
      - response is an error dict (no `results` key)
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
    # Pass-through: error envelopes, missing `results`, non-dict input.
    if not isinstance(response, dict) or "results" not in response:
        return response

    out = dict(response)  # shallow copy; we'll deep-copy chunks lists below
    out.setdefault("truncated", False)
    out.setdefault("dropped_chunk_count", 0)
    out.setdefault("cursor", None)

    if _measure(out) <= target_bytes:
        return out

    # Materialize a working copy of results (and per-result chunks) so
    # we can mutate without affecting the caller's input. Entity items
    # without chunks get a [] default so the trim loop can still walk
    # them as candidates for whole-item drops.
    docs: list[dict[str, Any]] = [
        {**d, "chunks": list(d.get("chunks") or [])}
        for d in out.get("results") or []
    ]
    out["results"] = docs

    # Stage 1: drop graph_evidence on tail chunks (rare savings, but
    # free when present).
    if _measure(out) > target_bytes:
        for d in reversed(docs):
            for c in reversed(d["chunks"]):
                if c.get("graph_evidence"):
                    c["graph_evidence"] = []
                    if _measure(out) <= target_bytes:
                        break
            if _measure(out) <= target_bytes:
                break

    # Stage 2: drop entire tail chunks. Walk docs from the tail (lowest
    # `score`); within each doc, drop chunks from the tail too (highest
    # `rank_in_doc`). When a doc has no chunks left, drop the doc.
    #
    # Floor: NEVER drop below 1 result / 1 chunk total. Returning an empty
    # `results` list is worse than returning a content-truncated
    # mega-chunk (the agent has no signal that the query was satisfied).
    # When we hit the floor while still over budget, fall through to
    # stage 3 for last-resort content truncation.
    dropped = 0
    if _measure(out) > target_bytes:
        doc_idx = len(docs) - 1
        while doc_idx >= 0 and _measure(out) > target_bytes:
            # Floor guard: stop if we'd reduce below 1 doc + 1 chunk.
            total_chunks = sum(len(d["chunks"]) for d in docs)
            if total_chunks <= 1:
                break
            chunks = docs[doc_idx]["chunks"]
            while chunks and _measure(out) > target_bytes:
                # Don't pop the very last chunk in the very last doc.
                total_chunks_now = sum(len(d["chunks"]) for d in docs)
                if total_chunks_now <= 1:
                    break
                chunks.pop()
                dropped += 1
            if not chunks and len(docs) > 1:
                docs.pop(doc_idx)
            doc_idx -= 1

    # Stage 3: last-resort content truncation on the surviving tail
    # chunk. Triggers in two cases:
    #   (a) one mega-chunk that's itself > hard_bytes (single oversized
    #       symbol surviving the rollover window from the chunker fix)
    #   (b) Stage 2 hit its 1-doc/1-chunk floor while still over target
    #
    # Either way, the last chunk's content gets clipped to a short
    # preview + `content_truncated: true` so the agent sees something
    # meaningful and can drill in via `get_source` for the full body.
    if docs and _measure(out) > hard_bytes:
        last_doc = docs[-1]
        if last_doc["chunks"]:
            last_chunk = last_doc["chunks"][-1]
            content = last_chunk.get("content")
            if isinstance(content, str) and len(content) > CONTENT_TRUNCATION_CHARS:
                last_chunk["content"] = content[:CONTENT_TRUNCATION_CHARS]
                last_chunk["content_truncated"] = True

    # Mark truncated if we actually changed anything observable.
    out["dropped_chunk_count"] = dropped
    out["truncated"] = bool(
        dropped > 0
        or any(
            c.get("content_truncated")
            for d in docs
            for c in d["chunks"]
        )
        or any(
            "graph_evidence" in c
            and c["graph_evidence"] == []
            and (
                # Only count graph_evidence-only changes if no chunks
                # were dropped — otherwise dropped count covers it.
                dropped == 0
            )
            for d in docs
            for c in d["chunks"]
        )
    )

    return out
