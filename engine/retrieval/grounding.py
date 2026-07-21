"""Grounding bundle builder — what we hand Haiku before the tool-call.

Four concurrent SQL reads against the customer's knowledge graph plus
pure helpers for token extraction and bare-ID detection. The bundle is
uncached per query — system prompt + tool schema stay cached, only the
candidate list flows in the user message.

The four channels:
  1. _fuzzy_match_entities       -- pg_trgm + FTS on graph_nodes.properties->>'name'
                                    for GROUNDING_ENTITY_LABELS (Person, Service,
                                    Feature, Decision, ErrorGroup). EXCLUDES Document
                                    (matched by channel 4 instead) and CodeSymbol
                                    (see the cost note in shared/constants.py).
  2. _lookup_bare_id_matches     -- regex-extracted exact IDs ("PR #340", "PRB-18",
                                    7-char commit shas) looked up in graph_nodes by
                                    (label, properties['kind']) -- NOT by a bare
                                    label, which has matched nothing since 0091.
  3. _connected_sources          -- lists which sources this customer has wired up.
                                    Metadata only — informs fanout scoping.
  4. _fuzzy_match_document_titles -- pg_trgm + FTS on documents.title (NEW). Concept
                                    queries like "multi-granola" / "shared-managed
                                    pivot" anchor on the canonical doc's TITLE here,
                                    where channel 1 misses them (Document is not in
                                    its label list and titles live in `documents`,
                                    not `graph_nodes.properties`).

Pre-channel-4, concept queries got nothing from channels 1+2, the LLM extractor
invented a phantom entity from the raw query text, and fanout had no strong
anchor → BM25/vector lottery → non-deterministic primary doc curation across
runs. See 2026-05-20 investigation in feat/grounding-doc-title-channel.

Design: docs/superpowers/specs/2026-05-14-router-intelligence-design.md
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from engine.shared.constants import (
    GROUNDING_ENTITY_LABELS,
    entity_type_for_node,
    node_addressing_for_entity_type,
)
from engine.shared.db import with_tenant
from engine.shared.logging import get_logger

log = get_logger(__name__)


# ---- Stopwords + regexes ------------------------------------------------

_STOPWORDS: Final[frozenset[str]] = frozenset({
    "a", "an", "and", "any", "are", "as", "at", "be", "but", "by",
    "for", "from", "had", "has", "have", "i", "if", "in", "is", "it",
    "its", "me", "no", "not", "of", "on", "or", "our", "so", "than",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "those", "to", "us", "was", "we", "were", "what", "when",
    "where", "which", "while", "who", "why", "will", "with", "you",
    "show", "find", "list", "get", "give", "tell", "fetch", "search",
    "look", "want", "need", "please", "can",
})

_RE_LINEAR_JIRA: Final[re.Pattern[str]] = re.compile(r"\b([A-Z]{2,10}-\d+)\b")

# PR number detection. We want to catch every natural English phrasing
# that names a specific GitHub PR/issue:
#
#   #328                 -- the legacy GitHub-prefix form
#   PR 328 / pr 328      -- the conversational form (NL queries hit this)
#   PR#328 / pr#328      -- compact
#   PR-328 / pr-328      -- hyphenated
#   PR.328 / pr.328      -- defensive (rare but seen in NL output)
#
# The original regex was `r"#(\d{1,6})\b"` — `#`-prefix only. That broke
# the user-visible "Why was PR 328 created" query: bare-id detection
# missed it, grounding fell back to fuzzy match on the slug "prbe-knowledge"
# (which only hit Repo nodes), and the agent never grounded on the PR doc
# entity, so PR 328's body chunks never made the prefanout. Live-traced
# 2026-05-18.
#
# Issues use the same numbering namespace and the same conversational
# phrasings, so the alternation also catches "issue 77" / "issue#77" / etc.
# We tag both as ("pr", ...) here — the grounding pipeline doesn't
# distinguish the two for entity-lookup purposes; it'll match whichever
# canonical_id (pr: or issue:) the customer has indexed.
_RE_GITHUB_PR: Final[re.Pattern[str]] = re.compile(
    r"(?:#|\b(?:pr|issue)\s*[-#.]?\s*)(\d{1,6})\b",
    re.IGNORECASE,
)
_RE_GIT_SHA: Final[re.Pattern[str]] = re.compile(r"\b([0-9a-f]{7,40})\b")


# ---- Dataclasses --------------------------------------------------------

@dataclass(slots=True)
class GroundingCandidate:
    entity_type: str
    canonical_id: str
    display_name: str
    last_seen_at: datetime | None
    match_source: str  # "trgm" | "fts" | "bare_id_exact" | "doc_title"


@dataclass(slots=True)
class GroundingBundle:
    candidates: list[GroundingCandidate] = field(default_factory=list)
    connected_sources: list[str] = field(default_factory=list)
    bare_id_matches: list[GroundingCandidate] = field(default_factory=list)
    timing_ms: float = 0.0


# ---- Pure helpers -------------------------------------------------------

def _extract_tokens(query: str) -> list[str]:
    """Lowercase, drop stopwords, keep len >= 2. Preserves identifiers
    like `auth.py` by treating '.' inside an alphanumeric run as part
    of the token.
    """
    if not query:
        return []
    pieces: list[str] = []
    for raw in query.split():
        cleaned = re.sub(r"^[^\w.]+|[^\w.]+$", "", raw).lower()
        if cleaned:
            pieces.append(cleaned)
    return [t for t in pieces if len(t) >= 2 and t not in _STOPWORDS]


def _detect_bare_ids(query: str) -> list[tuple[str, str]]:
    """Return [(kind, canonical_id), ...] from regex-detected bare IDs."""
    out: list[tuple[str, str]] = []
    for m in _RE_LINEAR_JIRA.finditer(query):
        out.append(("ticket", m.group(1)))
    for m in _RE_GITHUB_PR.finditer(query):
        out.append(("pr", m.group(1)))
    for m in _RE_GIT_SHA.finditer(query):
        sha = m.group(1)
        if not any(sha in existing for _, existing in out):
            out.append(("commit_sha", sha))
    return out


# ---- Label mapping ------------------------------------------------------

# NOTE: the former private `_LABEL_TO_ENTITY_TYPE` map lived here. It keyed on
# the PRE-migration-0091 labels (Repo / Ticket / PR / File / Channel / Session /
# Commit), 7 of which stopped existing when 0091 collapsed them into Document
# and 0052 folded File in -- so they matched zero rows and this channel had
# silently shrunk to Person / Service / Feature / Decision / ErrorGroup.
# The vocabulary now derives from ENTITY_TYPE_REGISTRY in engine/shared/
# constants.py, which discriminates on (label, properties['kind']) and is the
# single source shared with ROUTER_ENTITY_TO_LABEL and the EntityType Literal.


# ---- SQL helpers --------------------------------------------------------

async def _fuzzy_match_entities(
    customer_id: str,
    tokens: list[str],
    per_type_cap: int = 5,
    total_cap: int = 20,
) -> list[GroundingCandidate]:
    """Top-K graph_nodes per (entity_type, similarity) via pg_trgm + tsvector.

    Names live at properties->>'name'. We coalesce on NULL so the
    similarity / tsvector ops never NPE on nodes missing a name.
    """
    if not tokens:
        return []

    trgm_probe = " ".join(tokens)
    labels = list(GROUNDING_ENTITY_LABELS)

    sql = """
    WITH ranked AS (
        SELECT
            label, canonical_id,
            properties->>'kind' AS kind,
            coalesce(properties->>'name', canonical_id) AS display_name,
            properties->>'last_seen_at' AS last_seen_at_raw,
            GREATEST(
                similarity(coalesce(properties->>'name',''), $2),
                CASE
                    WHEN to_tsvector('english', coalesce(properties->>'name', ''))
                         @@ plainto_tsquery('english', $3) THEN 0.5
                    ELSE 0.0
                END
            ) AS rel,
            ROW_NUMBER() OVER (
                PARTITION BY label
                ORDER BY GREATEST(
                    similarity(coalesce(properties->>'name',''), $2),
                    CASE
                        WHEN to_tsvector('english', coalesce(properties->>'name', ''))
                             @@ plainto_tsquery('english', $3) THEN 0.5
                        ELSE 0.0
                    END
                ) DESC,
                (properties->>'last_seen_at')::timestamptz DESC NULLS LAST
            ) AS rn
        FROM graph_nodes
        WHERE customer_id = $1
          AND label = ANY($4::text[])
          AND (
              coalesce(properties->>'name','') % $2
              OR to_tsvector('english', coalesce(properties->>'name', ''))
                 @@ plainto_tsquery('english', $3)
          )
    )
    SELECT label, canonical_id, kind, display_name, last_seen_at_raw, rel
    FROM ranked
    WHERE rn <= $5
    ORDER BY rel DESC, last_seen_at_raw DESC NULLS LAST
    LIMIT $6
    """

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            sql, customer_id, trgm_probe, trgm_probe, labels, per_type_cap, total_cap
        )

    out: list[GroundingCandidate] = []
    for r in rows:
        entity_type = entity_type_for_node(r["label"], r["kind"])
        if not entity_type:
            continue
        last_seen = None
        if r["last_seen_at_raw"]:
            try:
                last_seen = datetime.fromisoformat(r["last_seen_at_raw"])
            except ValueError:
                last_seen = None
        out.append(GroundingCandidate(
            entity_type=entity_type,
            canonical_id=r["canonical_id"],
            display_name=r["display_name"],
            last_seen_at=last_seen,
            match_source="trgm" if r["rel"] != 0.5 else "fts",
        ))
    return out


async def _lookup_bare_id_matches(
    customer_id: str,
    bare_ids: list[tuple[str, str]],
) -> list[GroundingCandidate]:
    if not bare_ids:
        return []

    # Address (label, properties['kind']), NOT a bare label. Migration 0091
    # collapsed PR / Issue / Ticket / Channel / Repo into label='Document' and
    # moved the distinction into properties['kind']; the previous
    # {"ticket": "Ticket", "pr": "PR", "commit_sha": "Commit"} map therefore
    # matched ZERO rows, silently breaking every bare-id lookup ("PR #340",
    # "PRB-18", commit shas) since that migration landed.
    by_addressing: dict[tuple[str, str | None, str], list[str]] = {}
    for entity_type, val in bare_ids:
        addressing = node_addressing_for_entity_type(entity_type)
        if addressing is None:
            continue
        label, node_kind = addressing
        by_addressing.setdefault((label.value, node_kind, entity_type), []).append(val)

    if not by_addressing:
        return []

    out: list[GroundingCandidate] = []
    async with with_tenant(customer_id) as conn:
        for (label, node_kind, entity_type), ids in by_addressing.items():
            rows = await conn.fetch(
                """
                SELECT canonical_id,
                       coalesce(properties->>'name', canonical_id) AS display_name,
                       properties->>'last_seen_at' AS last_seen_at_raw
                FROM graph_nodes
                WHERE customer_id = $1
                  AND label = $2
                  AND canonical_id = ANY($3::text[])
                  AND ($4::text IS NULL OR properties->>'kind' = $4)
                """,
                customer_id, label, ids, node_kind,
            )
            for r in rows:
                last_seen = None
                if r["last_seen_at_raw"]:
                    try:
                        last_seen = datetime.fromisoformat(r["last_seen_at_raw"])
                    except ValueError:
                        last_seen = None
                out.append(GroundingCandidate(
                    entity_type=entity_type,
                    canonical_id=r["canonical_id"],
                    display_name=r["display_name"],
                    last_seen_at=last_seen,
                    match_source="bare_id_exact",
                ))
    return out


# source_system → grounding entity_type for doc-title matches. The downstream
# extractor / fanout already understands ticket / pr / channel / commit_sha
# from the ENTITY_TYPE_REGISTRY surface; notion/wiki/other get a
# generic `page` / `document` so the field carries provenance without
# requiring downstream code changes. `entity_type` is informational on
# GroundingCandidate (the strong signal is canonical_id) — this map only
# affects telemetry and prompt readability.
_SOURCE_SYSTEM_TO_ENTITY_TYPE: Final[dict[str, str]] = {
    "linear": "ticket",
    "slack": "channel",
    "github": "document",  # disambiguated below via doc_id prefix
    "notion": "page",
    "wiki": "document",
    "claude_code": "session",
    "codex": "session",
    "granola": "document",
    "sentry": "error_group",
    "pagerduty": "incident",
    "incident_io": "incident",
}


def _doc_id_to_entity_type(doc_id: str, source_system: str) -> str:
    """Refine source_system → entity_type using the doc_id's structural prefix.

    GitHub docs come in flavors (`...:pr:N`, `...:issue:N`, `...:commit:sha`,
    `...:review:R`). Map each to the right entity_type so downstream
    consumers see semantically meaningful types instead of a flat
    `document` for everything-github.
    """
    if source_system == "github":
        parts = doc_id.split(":")
        if len(parts) >= 3:
            kind = parts[2].lower()
            if kind == "pr":
                return "pr"
            if kind == "issue":
                return "ticket"
            if kind == "commit":
                return "commit_sha"
    return _SOURCE_SYSTEM_TO_ENTITY_TYPE.get(source_system, "document")


# Title fuzzy-match caps. Concept queries often touch 1-3 canonical docs
# (a Linear ticket + a Notion design + a wiki page); 10 is a generous upper
# bound that covers multi-aspect queries without flooding the prompt.
_DOC_TITLE_TOTAL_CAP: Final[int] = 10

# pg_trgm similarity floor. Below this, matches are usually noise (single
# common token coincidences). 0.15 was the empirical floor in
# _fuzzy_match_entities; reuse it here for consistency.
_DOC_TITLE_TRGM_FLOOR: Final[float] = 0.15


async def _fuzzy_match_document_titles(
    customer_id: str,
    tokens: list[str],
    cap: int = _DOC_TITLE_TOTAL_CAP,
) -> list[GroundingCandidate]:
    """Pg_trgm + FTS match against documents.title.

    Plugs the structural gap where channel 1 (`_fuzzy_match_entities`)
    excludes Document nodes — title content lives in `documents.title`,
    not `graph_nodes.properties->>'name'`. Without this channel, concept
    queries (multi-granola, shared-managed pivot, apps plane) return
    zero grounding candidates, the downstream LLM extractor invents a
    phantom entity, and fanout falls through to vector/BM25 lottery.

    The query combines pg_trgm `%` (catches partial/typo matches —
    backed by `idx_documents_title_trgm`) with tsvector FTS (backed by
    the pre-existing `idx_documents_fts_title_preview` which covers
    `title || ' ' || body_preview`). Either match qualifies; ranking is
    by tsvector-rank when present, falling back to trgm similarity.

    Customer scoping is enforced both by `with_tenant()` (RLS) AND an
    explicit `customer_id = $1` filter — defense in depth, matches the
    pattern of every other channel in this file.

    `valid_to IS NULL` filters out soft-deleted versions. Wiki source
    is included (unlike inferred-edges enqueue, which skips it):
    wiki-synthesized pages like 'Probe Bench Test' or
    `wiki:event:shared_managed_pivot` ARE the canonical concept docs
    for many architectural queries, and excluding them would re-create
    the same gap this channel exists to close.
    """
    if not tokens:
        return []

    trgm_probe = " ".join(tokens)

    sql = """
    WITH ranked AS (
        SELECT
            d.doc_id,
            d.source_system,
            coalesce(d.title, '') AS title,
            d.updated_at,
            similarity(coalesce(d.title, ''), $2) AS trgm_sim,
            CASE
                WHEN to_tsvector('english', coalesce(d.title, '') || ' '
                                 || coalesce(d.body_preview, ''))
                     @@ plainto_tsquery('english', $3) THEN 1
                ELSE 0
            END AS fts_hit
        FROM documents d
        WHERE d.customer_id = $1
          AND d.valid_to IS NULL
          AND d.title IS NOT NULL
          AND d.title <> ''
          AND (
              coalesce(d.title, '') % $2
              OR to_tsvector('english', coalesce(d.title, '') || ' '
                             || coalesce(d.body_preview, ''))
                 @@ plainto_tsquery('english', $3)
          )
    )
    SELECT doc_id, source_system, title, updated_at, trgm_sim, fts_hit
    FROM ranked
    WHERE trgm_sim >= $4 OR fts_hit = 1
    ORDER BY fts_hit DESC, trgm_sim DESC, updated_at DESC NULLS LAST
    LIMIT $5
    """

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            sql, customer_id, trgm_probe, trgm_probe, _DOC_TITLE_TRGM_FLOOR, cap,
        )

    out: list[GroundingCandidate] = []
    for r in rows:
        out.append(GroundingCandidate(
            entity_type=_doc_id_to_entity_type(r["doc_id"], r["source_system"]),
            canonical_id=r["doc_id"],
            display_name=r["title"],
            last_seen_at=r["updated_at"],
            match_source="doc_title",
        ))
    return out


async def _connected_sources(customer_id: str) -> list[str]:
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT DISTINCT source_system
            FROM customer_source_mapping
            WHERE customer_id = $1
            ORDER BY source_system
            """,
            customer_id,
        )
    return [r["source_system"] for r in rows]


# ---- Public API ---------------------------------------------------------

async def build_bundle(customer_id: str, query: str) -> GroundingBundle:
    """Build the grounding bundle shown to Haiku before tool-call extraction.

    Per-task fault isolation: each SQL is awaited via asyncio.gather with
    return_exceptions=True. A failure in one read produces an empty list
    in that field; the rest of the bundle populates. If all four fail,
    returns an empty bundle (Haiku still works without grounding; the
    request still serves; the failure is logged).

    Doc-title matches (channel 4) merge into the same `candidates` list
    as entity matches (channel 1) — both are "candidate anchors for the
    LLM extractor to pick from." Downstream consumers don't distinguish
    by source within `candidates` (they key on canonical_id); the
    `match_source` field on each GroundingCandidate carries the
    provenance for telemetry.
    """
    t0 = time.perf_counter()
    tokens = _extract_tokens(query)
    bare_ids = _detect_bare_ids(query)

    results = await asyncio.gather(
        _fuzzy_match_entities(customer_id, tokens, per_type_cap=5, total_cap=20),
        _lookup_bare_id_matches(customer_id, bare_ids),
        _connected_sources(customer_id),
        _fuzzy_match_document_titles(customer_id, tokens),
        return_exceptions=True,
    )

    entity_candidates = results[0] if not isinstance(results[0], BaseException) else []
    bare_id_matches = results[1] if not isinstance(results[1], BaseException) else []
    sources = results[2] if not isinstance(results[2], BaseException) else []
    doc_title_candidates = results[3] if not isinstance(results[3], BaseException) else []

    for label, r in zip(
        ("fuzzy", "bare_id", "sources", "doc_title"), results, strict=True,
    ):
        if isinstance(r, BaseException):
            log.warning(
                "grounding.partial_failure",
                extra={"customer_id": customer_id, "subtask": label, "error": str(r)},
            )

    # Merge doc-title hits into the candidate pool. Dedup by canonical_id
    # so a doc that surfaces via both channels (e.g. a Linear ticket with
    # a Ticket-labeled graph_node AND a matching documents.title) doesn't
    # render twice. Order matters: entity_candidates first (preserves
    # the existing fuzzy-match precision), doc_title appended after.
    seen_canonical_ids: set[str] = {c.canonical_id for c in entity_candidates}
    merged_candidates: list[GroundingCandidate] = list(entity_candidates)
    for cand in doc_title_candidates:
        if cand.canonical_id in seen_canonical_ids:
            continue
        seen_canonical_ids.add(cand.canonical_id)
        merged_candidates.append(cand)

    return GroundingBundle(
        candidates=merged_candidates,
        connected_sources=sources,
        bare_id_matches=bare_id_matches,
        timing_ms=(time.perf_counter() - t0) * 1000,
    )
