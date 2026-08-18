"""Response transforms that strip diagnostic fields agents don't reason
over, while preserving signals they actually use (top-level score,
recall hints, router decisions).

Compaction is on by default; tools expose `verbose=True` for the rare
case the caller needs the raw upstream payload (full retriever-score
breakdown and timing). The opaque `trace_id` stays in compact responses
so failed and successful calls can be correlated with upstream logs.
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
        "applied_sources",
        "aggregation",
        "router_hit_cache",
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
# populated `graph_evidence` (a list of {edge_type, confidence, via_entity,
# reason} entries — the agent's only evidence the chunk actually grounds the
# query against the knowledge graph). `chunk_id` and `rank_in_doc` are pure
# server bookkeeping; `retriever_scores` was already judged noise at document
# level and the chunk-level copy measured empty in 69/69 production chunks —
# a dict of nothing, shipped on every chunk of every response.
_CHUNK_DROP = frozenset(
    {
        "chunk_id",
        "rank_in_doc",
        "retriever_scores",
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


def _lean_matched_via(entries: Any) -> Any:
    """Strip null-valued keys from provenance entries.

    A typical vector-channel entry is {"channel": "vector", "rank": 1,
    "score": 1.0, "intent_idx": 0, "anchor_doc_id": null, "edge_type": null,
    "confidence": null, "why": null} — half the keys carry nothing. The
    non-null half stays untouched, so the inferred-edge case (`edge_type`,
    `why` populated — the reason provenance is kept at all, see #112) loses
    not one byte. Absent key == null here: no caller distinguishes them.
    """
    if not isinstance(entries, list):
        return entries
    return [
        {k: v for k, v in e.items() if v is not None} if isinstance(e, dict) else e
        for e in entries
    ]


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    """Compact one polymorphic QueryResult.

    Branches on `node_type`:
      - "Entity" -> drop only `rank`; pass everything else (label,
        canonical_id, display_name, properties, attached_doc_ids,
        edge_types, doc_count, score, matched_via).
      - "Document" (or missing/unknown -> default): drop per-doc
        diagnostics and compact each nested chunk.

    Beyond the deny-lists, four REPEATS are collapsed — each measured on live
    production responses before being touched, none of them information:

      * `canonical_id` goes when byte-identical to `doc_id` (69/83 measured
        docs). When they differ it is a real second identity and stays.
      * a chunk's `matched_via` goes when identical to its document's (the
        chunk inherited the doc-level provenance verbatim). A chunk that
        matched its own way keeps its own trail.
      * `graph_evidence: []` and `why_relevant: ""` go when empty — populated
        values always pass. "Nothing here" is what an absent key already says.
      * null-valued keys inside `matched_via` entries go (see
        `_lean_matched_via`).

    Under the response byte budget every one of these repeats was being paid
    for with dropped tail chunks — 9 of 10 measured responses were truncated —
    so collapsing them converts directly into evidence that survives the cap.
    """
    if result.get("node_type") == "Entity":
        return _strip(result, _ENTITY_RESULT_DROP)
    compacted = _strip(result, _DOC_RESULT_DROP)
    if compacted.get("canonical_id") == compacted.get("doc_id"):
        compacted.pop("canonical_id", None)
    doc_via = result.get("matched_via")
    if "matched_via" in compacted:
        compacted["matched_via"] = _lean_matched_via(compacted["matched_via"])
    chunks = result.get("chunks") or []
    lean_chunks: list[Any] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            lean_chunks.append(chunk)
            continue
        lean = _strip(chunk, _CHUNK_DROP)
        if lean.get("matched_via") == doc_via:
            lean.pop("matched_via", None)
        elif "matched_via" in lean:
            lean["matched_via"] = _lean_matched_via(lean["matched_via"])
        if not lean.get("graph_evidence"):
            lean.pop("graph_evidence", None)
        if not lean.get("why_relevant"):
            lean.pop("why_relevant", None)
        lean_chunks.append(lean)
    compacted["chunks"] = lean_chunks
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


# --------------------------------------------------------------------------
# detail profiles — the caller's altitude dial on search_knowledge.
#
# Three NAMED shapes, not a field-selection language. Measured against ten
# live production responses, three profiles capture ~85% of what a perfect
# per-call selection could save, and the remainder is only reachable by
# dropping the fields that let an agent cite a result or drill into it. A
# selection language would also hand the model a new failure mode — an
# under-selected response costs a retry worth more than the metadata it
# saved — so capability lives in this one parameter instead (the same seam
# probe-research's get_entity(view=...) uses).
#
# The ENVELOPE IS UNTOUCHABLE at every detail. `degraded`, `truncated`,
# `confidence_breakdown`, `total_candidates`, `extracted_entities` and their
# siblings say what the response could NOT cover or how it was produced;
# a profile that trimmed them would make a partial answer look complete,
# which is the exact bug the tripwire test on _TOP_LEVEL_DROP exists to
# prevent. Profiles therefore only ever rewrite rows inside `results` —
# envelope safety holds by construction, not by a second deny-list.

DETAIL_IDS = "ids"
DETAIL_EVIDENCE = "evidence"
DETAIL_FULL = "full"
_DETAILS = (DETAIL_IDS, DETAIL_EVIDENCE, DETAIL_FULL)

# Per-document metadata `evidence` does without: audit fields an agent reads
# past on the way to the content. All of it remains one detail="full" (or
# get_source) call away — and the compact response keeps `doc_id`, which is
# the handle that keeps that escape hatch reachable.
_EVIDENCE_DOC_DROP = frozenset({"created_at", "updated_at", "author_id"})

# What a document row keeps at detail="ids": enough to rank, cite, and fetch —
# nothing to read. The triage shape for "did the lab touch X at all?".
_IDS_DOC_KEEP = frozenset(
    {"doc_id", "canonical_id", "title", "score", "source_system", "node_type"}
)


def _has_inferred_edge(entries: Any) -> bool:
    """Did any provenance entry arrive over a knowledge-graph edge?

    That is the one case matched_via exists for (#112: it carries the LLM's
    `why` when an inferred-edge result surfaced via a Doc-Doc walk), so it is
    the one case `evidence` keeps it. Rank/score boilerplate on a plain
    vector hit tells an agent nothing `score` does not already say.
    """
    if not isinstance(entries, list):
        return False
    return any(
        isinstance(e, dict) and (e.get("edge_type") or e.get("why")) for e in entries
    )


def _evidence_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("node_type") == "Entity":
        return result
    lean = _strip(result, _EVIDENCE_DOC_DROP)
    if not _has_inferred_edge(lean.get("matched_via")):
        lean.pop("matched_via", None)
    chunks = lean.get("chunks")
    if isinstance(chunks, list):
        lean["chunks"] = [
            (
                {k: v for k, v in c.items() if not (k == "matched_via" and not _has_inferred_edge(v))}
                if isinstance(c, dict)
                else c
            )
            for c in chunks
        ]
    return lean


def _ids_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("node_type") == "Entity":
        return result
    return {k: v for k, v in result.items() if k in _IDS_DOC_KEEP}


def apply_detail(payload: dict[str, Any], detail: str) -> dict[str, Any]:
    """Project an already-compacted response down to the named profile.

    Runs BEFORE the byte budget on purpose: a leaner shape means the budget
    trims fewer tail chunks, so the profile buys recall, not just tokens.
    Entities pass through every profile untouched — they are small,
    self-describing, and have no chunk payload to spend.
    """
    if detail not in _DETAILS:
        raise ValueError(
            f"detail must be one of {', '.join(_DETAILS)}; got {detail!r}"
        )
    if detail == DETAIL_FULL:
        return payload
    project = _evidence_result if detail == DETAIL_EVIDENCE else _ids_result
    out = dict(payload)
    out["results"] = [
        project(r) if isinstance(r, dict) else r for r in payload.get("results") or []
    ]
    return out
