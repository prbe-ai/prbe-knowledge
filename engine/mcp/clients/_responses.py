"""Response transforms that strip diagnostic fields agents don't reason
over, while preserving signals they actually use (top-level score,
recall hints, router decisions).

Compaction is on by default; tools expose `verbose=True` for the rare
case the caller needs the raw upstream payload (full retriever-score
breakdown and timing). The opaque `trace_id` stays in compact responses
so failed and successful calls can be correlated with upstream logs.
"""

from __future__ import annotations

from typing import Any, Final

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

# Per-chunk fields stripped by default. We keep `content` and
# `why_relevant`. `chunk_id` and `rank_in_doc` are pure server bookkeeping;
# `retriever_scores` was already judged noise at document level and the
# chunk-level copy measured empty in 69/69 production chunks.
#
# `graph_evidence` is VERBOSE-ONLY as of 2026-08-31. When the budget's
# truncation order was written it measured 42 bytes on a 196KB response;
# after the KB graph channel went live it measured ~42% OF THE RESPONSE
# (33 entries x ~100 tokens on a live top_k=5 call), dominated by per-edge
# `via_entity_*` decoration re-shipped for the same entity. The envelope's
# `confidence_breakdown` still says evidence EXISTS (and how confident);
# an agent that wants the trails re-calls with `verbose=True`, which
# bypasses compaction entirely.
_CHUNK_DROP = frozenset(
    {
        "chunk_id",
        "rank_in_doc",
        "retriever_scores",
        "graph_evidence",
    }
)

# Chunk `score` is dropped when it carries the gatherer path's constant 1.0
# ("surfaced == max confidence within the curated set") — a real magnitude
# from any other path survives.
_CHUNK_CONSTANT_SCORE = 1.0

# `related_entities` rows keep only their identity (the crawl-candidate
# handle). The other seven fields are adapter constants on this path
# (edge_types:[], associated_doc_ids:[], member_sources:[], member_count:1,
# doc_count:1, score:1.0, max_confidence:"EXTRACTED") — measured boilerplate
# on every row of every response.
_RELATED_ENTITY_KEEP = frozenset({"canonical_id", "label", "display_name"})

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
        lean_entity = _strip(result, _ENTITY_RESULT_DROP)
        # Shed the near-always-empty navigation fields (matched_via: [],
        # properties: {}, attached_doc_ids: [], edge_types: [], doc_count: 0)
        # — absent already means "nothing here". Populated values pass.
        return {
            k: v
            for k, v in lean_entity.items()
            if v or k not in ("matched_via", "properties", "attached_doc_ids",
                              "edge_types", "doc_count")
        }
    compacted = _strip(result, _DOC_RESULT_DROP)
    if compacted.get("canonical_id") == compacted.get("doc_id"):
        compacted.pop("canonical_id", None)
    doc_via = result.get("matched_via")
    if "matched_via" in compacted:
        compacted["matched_via"] = _lean_matched_via(compacted["matched_via"])
    chunks = result.get("chunks")
    if not isinstance(chunks, list):
        # A string here would iterate as characters and hand the caller a
        # fabricated chunk list; upstream garbage earns an empty list, not
        # an invented one.
        chunks = []
    lean_chunks: list[Any] = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            # Dropped, not passed through: downstream (the byte budget's
            # dict(chunk) copy) would raise on it and destroy the whole
            # response; a chunk that is not an object carries no evidence
            # worth preserving.
            continue
        lean = _strip(chunk, _CHUNK_DROP)
        if lean.get("matched_via") == doc_via:
            lean.pop("matched_via", None)
        elif "matched_via" in lean:
            lean["matched_via"] = _lean_matched_via(lean["matched_via"])
        if lean.get("score") == _CHUNK_CONSTANT_SCORE:
            lean.pop("score", None)
        if not lean.get("why_relevant"):
            lean.pop("why_relevant", None)
        lean_chunks.append(lean)
    compacted["chunks"] = lean_chunks
    return compacted


def _lean_envelope(out: dict[str, Any]) -> dict[str, Any]:
    """Shed envelope fields that are dead weight when empty.

    `aggregations: []` (the populated case passes), `related_entities_error:
    null` (shown only when the walk actually failed), and the seven constant
    fields on each `related_entities` row. Every honesty field (`degraded`,
    `truncated`, `confidence_breakdown`, counts) passes untouched — the
    tripwire test on _TOP_LEVEL_DROP guards that contract.
    """
    if not out.get("aggregations"):
        out.pop("aggregations", None)
    if out.get("related_entities_error") is None:
        out.pop("related_entities_error", None)
    related = out.get("related_entities")
    if isinstance(related, list):
        out["related_entities"] = [
            {k: v for k, v in r.items() if k in _RELATED_ENTITY_KEEP}
            if isinstance(r, dict)
            else r
            for r in related
        ]
    return out


def compact_search(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip top-level + per-result + per-chunk diagnostics.

    Operates on the polymorphic shape: `results[*]` is a discriminated
    union of Document (with `chunks[*]` nested) and Entity. The top-level
    `confidence_breakdown` aggregate passes through (router signal — agents
    need to see it to decide whether to widen the search).
    """
    out = _strip(payload, _TOP_LEVEL_DROP)
    results = payload.get("results") or []
    out["results"] = [
        _compact_result(r) if isinstance(r, dict) else r for r in results
    ]
    return _lean_envelope(out)


def compact_query(payload: dict[str, Any]) -> dict[str, Any]:
    """Strip top-level + per-result + per-chunk diagnostics from a
    synthesized-answer response.

    The synthesized `answer`, `citations`, `insufficient_context`, and
    `model` fields pass through unchanged. Underlying polymorphic results
    (`results[*]`) are compacted the same way as `compact_search`, so the
    shared shape is search_knowledge's detail="full"; the search TOOL
    projects further via `apply_detail`, query_knowledge does not.
    """
    out = _strip(payload, _TOP_LEVEL_DROP)
    results = payload.get("results")
    if results is not None:
        out["results"] = [_compact_result(r) for r in results]
    return _lean_envelope(out)


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

DETAIL_IDS: Final = "ids"
DETAIL_EVIDENCE: Final = "evidence"
DETAIL_FULL: Final = "full"
#: Public on purpose: the tool layer validates against this same tuple, so a
#: fourth profile lands in exactly one place.
VALID_DETAILS = (DETAIL_IDS, DETAIL_EVIDENCE, DETAIL_FULL)


def detail_error(detail: object) -> str:
    """The one wording for an unknown detail, shared by the tool layer's 422
    and apply_detail's ValueError so the two can never drift.

    The echoed value is clipped: this error goes straight to the caller
    WITHOUT passing the response byte budget, so reflecting an arbitrarily
    long value verbatim would let the one unvalidated string in the call
    defeat the cap the budget exists to enforce (same discipline as
    KnowledgeError's body[:200])."""
    return f"detail must be one of {', '.join(VALID_DETAILS)}; got {str(detail)[:80]!r}"

# Per-document metadata `evidence` does without: audit fields an agent reads
# past on the way to the content. All of it remains one detail="full" (or
# get_source) call away — and the compact response keeps `doc_id`, which is
# the handle that keeps that escape hatch reachable.
_EVIDENCE_DOC_DROP = frozenset({"created_at", "updated_at", "author_id"})

# What a document row keeps at detail="ids": enough to rank and fetch
# (doc_id -> get_source) — nothing to read. The triage shape for "did the
# team touch X at all?". source_url is deliberately not here: linking a
# human to a source is an evidence-level act.
_IDS_DOC_KEEP = frozenset(
    {
        "doc_id",
        "canonical_id",
        "title",
        "score",
        "source_system",
        "node_type",
        # The weight signal: one matching span and twelve are different
        # triage answers, and the count costs single-digit tokens.
        "chunk_count",
    }
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


def _without_boiler_via(node: dict[str, Any]) -> dict[str, Any]:
    """Drop `matched_via` unless it carries an inferred-edge entry — the ONE
    rule of `evidence`-level provenance, applied identically at doc and chunk
    level so the two can never diverge."""
    if _has_inferred_edge(node.get("matched_via")):
        return node
    return {k: v for k, v in node.items() if k != "matched_via"}


def _evidence_result(result: dict[str, Any]) -> dict[str, Any]:
    if result.get("node_type") == "Entity":
        return result
    lean = _without_boiler_via(_strip(result, _EVIDENCE_DOC_DROP))
    chunks = lean.get("chunks")
    if isinstance(chunks, list):
        lean["chunks"] = [
            _without_boiler_via(c) if isinstance(c, dict) else c for c in chunks
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
    if detail not in VALID_DETAILS:
        raise ValueError(detail_error(detail))
    if detail == DETAIL_FULL:
        return payload
    project = _evidence_result if detail == DETAIL_EVIDENCE else _ids_result
    out = dict(payload)
    out["results"] = [
        project(r) if isinstance(r, dict) else r for r in payload.get("results") or []
    ]
    return out
