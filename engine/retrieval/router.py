"""Router — deterministic grounding only (Phase 2 cutover).

Pre-cutover this module called Haiku to extract structured `Intent` shapes
from the user's query. Post-cutover the gatherer agent reads channel
results directly; there is no LLM router step.

What remains here:
- `Intent`, `RouterEntity`, `RouterOutput` dataclasses — the gatherer
  doesn't consume them, but `pipeline.py`'s streaming compat shim still
  wraps grounding output in a single synthetic `Intent`, and
  `list_pipeline.py` (dead code post-cutover but kept for cleanup-PR
  ergonomics) reads them.
- `_build_bundle_with_token_fallback` — the deterministic grounding
  entry point used by `pipeline.run_router_phase` and the gatherer loop.
- `_reconcile_entities_with_bundle` — kept per plan, available as a
  helper if any tool emits ungrounded canonical_ids.
- `_escape_query_for_xml` — defence-in-depth wrapper inherited by the
  gatherer's tool surface; tools.py imports it for any future tool that
  re-invokes an LLM with user-controlled text.

Plan: docs/specs/agentic-search.md, section "Coordination with PR #282".
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from engine.retrieval.grounding import (
    GroundingBundle,
    _extract_tokens,
    _fuzzy_match_document_titles_multi,
    _fuzzy_match_entities_multi,
    build_bundle,
)
from engine.shared.logging import get_logger

log = get_logger(__name__)

# Hard cap inherited from PR #282 (defence in depth — `parallel_multi_query`
# clamps to the same value). Schema enforcement lives in the agent's tool
# definitions; this constant exists so callers without the tool schema
# (tests, list_pipeline) can still enforce the cap.
MAX_INTENTS = 3

# Entity-type buckets — kept for `list_pipeline.py` (dead code but still
# imported by tests) and for any tool that wants to classify entities.
NARROWING_ENTITY_TYPES: frozenset[str] = frozenset(
    {
        "service",
        "repo",
        "person",
        "ticket",
        "pr",
        "file_path",
        "channel",
        "source",
        "doc_type",
        "session",
    }
)

TOPIC_ENTITY_TYPES: frozenset[str] = frozenset({"feature", "decision", "error_group"})

DOC_TYPE_TOKENS: tuple[str, ...] = (
    "commit",
    "pr",
    "issue",
    "review",
    "release",
    "message",
    "thread",
    "page",
    "ticket",
    "comment",
    "session",
    "meeting",
)

OPERATIONS: tuple[str, ...] = ("list", "count", "group_by")

GROUP_BY_KEYS: tuple[str, ...] = ("source_system", "doc_type", "author_id")


# ---- Dataclasses ---------------------------------------------------------


@dataclass(slots=True)
class RouterEntity:
    entity_type: str
    canonical_id: str
    display_name: str
    confidence: float


@dataclass(slots=True)
class Intent:
    """A single extracted intent shape.

    Post-cutover the gatherer doesn't emit intents — it emits a
    `GathererOutput`. The compat shim in `pipeline.run_router_phase`
    synthesizes a single-Intent payload from the grounding bundle so the
    streaming endpoint's per-intent SSE event remains shape-stable.
    """

    query_text: str
    mode: str
    confidence: float
    entities: list[RouterEntity] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    temporal: dict[str, Any] | None = None
    sort: dict[str, Any] | None = None
    doc_type: str | None = None
    operation: str | None = None
    group_by_key: str | None = None


@dataclass(slots=True)
class RouterOutput:
    """Multi-intent container output from the router.

    Post-cutover always carries `intents=[fallback]` + the grounding
    bundle. `fallback_used=False` because the gatherer is the new
    pipeline — no Haiku call to "fall back from".
    """

    intents: list[Intent]
    grounding_bundle: GroundingBundle
    router_raw: dict[str, Any] = field(default_factory=dict)
    cache_tokens: dict[str, Any] | None = None
    fallback_used: bool = False


# ---- Private helpers (re-exported across the cutover boundary) -----------


def _fallback_intent(query: str) -> Intent:
    """Synthesise a single-search-mode Intent. Used as the gatherer-era
    placeholder in RouterOutput.intents."""
    return Intent(query_text=query, mode="search", confidence=0.0)


def _reconcile_entities_with_bundle(
    intents: list[Intent], bundle: GroundingBundle
) -> None:
    """Kept per plan as a helper. Swaps any intent entity's canonical_id
    for the grounded match when the bundle covered it.

    Originally a defence-in-depth backstop for Haiku synthesizing slugs
    like "backend" instead of copying the grounded "prbe-backend". The
    gatherer doesn't emit canonical_ids today, but any future tool that
    proposes ungrounded handles can call this to align them.
    """
    candidate_index: dict[str, list] = {}
    for c in bundle.candidates:
        candidate_index.setdefault(c.entity_type, []).append(c)

    known_canonical_ids: set[tuple[str, str]] = {
        (c.entity_type, c.canonical_id) for c in bundle.candidates
    } | {(m.entity_type, m.canonical_id) for m in bundle.bare_id_matches}

    for intent in intents:
        for entity in intent.entities:
            if (entity.entity_type, entity.canonical_id) in known_canonical_ids:
                continue
            emitted = entity.canonical_id.lower()
            emitted_kebab = emitted.replace("_", "-")
            replacement = None
            for c in candidate_index.get(entity.entity_type, []):
                cid_lower = c.canonical_id.lower()
                dname_lower = c.display_name.lower()
                if emitted in cid_lower:
                    replacement = c
                    break
                if emitted_kebab.replace("-", " ") in dname_lower:
                    replacement = c
                    break
            if replacement is not None:
                log.info(
                    "router.entity_reconciled",
                    emitted_canonical_id=entity.canonical_id,
                    grounded_canonical_id=replacement.canonical_id,
                    entity_type=entity.entity_type,
                )
                entity.canonical_id = replacement.canonical_id
                entity.display_name = replacement.display_name


def _escape_query_for_xml(query: str) -> str:
    """HTML-escape user input so it cannot break the <query> data boundary.

    Inherited from PR #282 (defence-in-depth #1). Any tool that passes
    user-controlled text into a downstream LLM call MUST apply this first.
    """
    return query.replace("&", "&amp;").replace("<", "&lt;")


# ---- Deterministic grounding entry point ---------------------------------

# Inherited from PR #282 — see grounding.py + pipeline.py for the spec.
_MIN_TOKEN_LEN_FOR_FALLBACK = 4
_MAX_TOKEN_FALLBACK_PROBES = 5


async def _build_bundle_with_token_fallback(
    customer_id: str, query: str
) -> GroundingBundle:
    """Build the grounding bundle, always merging per-token probes when
    the query has 2+ content tokens. See PR #282's commentary in
    `services/retrieval/grounding.py` for full rationale.
    """
    initial = await build_bundle(customer_id, query)

    tokens = [
        t for t in _extract_tokens(query) if len(t) >= _MIN_TOKEN_LEN_FOR_FALLBACK
    ]
    if len(tokens) < 2:
        return initial

    sorted_tokens = sorted(
        enumerate(tokens), key=lambda iv: (-len(iv[1]), iv[0])
    )
    probes = [t for _, t in sorted_tokens[:_MAX_TOKEN_FALLBACK_PROBES]]

    # BATCHED, not one build_bundle per probe. The per-probe bundles ran
    # every channel and kept only `.candidates` -- the bare-id and
    # connected-sources work was computed and discarded, and the two fuzzy
    # matchers were each a full tenant-wide scan per probe (the RLS-blocked
    # shape; see the multi variants' comment in grounding.py). One
    # statement per matcher now covers all probes; the reassembly below
    # mirrors build_bundle's own merge (entity candidates first, doc-title
    # appended, deduped by canonical_id within the probe), so each probe's
    # candidate list is what its bundle would have carried.
    matcher_results = await asyncio.gather(
        _fuzzy_match_entities_multi(
            customer_id, probes, per_type_cap=5, total_cap=20
        ),
        _fuzzy_match_document_titles_multi(customer_id, probes),
        return_exceptions=True,
    )
    empty: list[list] = [[] for _ in probes]
    entity_lists = (
        matcher_results[0]
        if not isinstance(matcher_results[0], BaseException)
        else empty
    )
    title_lists = (
        matcher_results[1]
        if not isinstance(matcher_results[1], BaseException)
        else empty
    )
    for label, r in zip(("entities_multi", "doc_titles_multi"), matcher_results, strict=True):
        if isinstance(r, BaseException):
            log.warning(
                "grounding.token_fallback_partial_failure",
                extra={"customer_id": customer_id, "subtask": label, "error": str(r)},
            )

    seen: set[tuple[str, str]] = {
        (c.entity_type, c.canonical_id) for c in initial.candidates
    }
    merged: list = list(initial.candidates)
    for ent, titles in zip(entity_lists, title_lists, strict=True):
        probe_seen = {c.canonical_id for c in ent}
        probe_candidates = list(ent)
        for c in titles:
            if c.canonical_id in probe_seen:
                continue
            probe_seen.add(c.canonical_id)
            probe_candidates.append(c)
        for c in probe_candidates:
            key = (c.entity_type, c.canonical_id)
            if key in seen:
                continue
            seen.add(key)
            merged.append(c)

    return GroundingBundle(
        candidates=merged,
        connected_sources=initial.connected_sources,
        bare_id_matches=initial.bare_id_matches,
        timing_ms=initial.timing_ms,
    )


__all__ = [
    "DOC_TYPE_TOKENS",
    "GROUP_BY_KEYS",
    "MAX_INTENTS",
    "NARROWING_ENTITY_TYPES",
    "OPERATIONS",
    "TOPIC_ENTITY_TYPES",
    "Intent",
    "RouterEntity",
    "RouterOutput",
    "_build_bundle_with_token_fallback",
    "_escape_query_for_xml",
    "_fallback_intent",
    "_reconcile_entities_with_bundle",
]
