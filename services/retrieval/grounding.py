"""Grounding bundle builder — what we hand Haiku before the tool-call.

Three concurrent SQL reads against the customer's knowledge graph plus
pure helpers for token extraction and bare-ID detection. The bundle is
uncached per query — system prompt + tool schema stay cached, only the
candidate list flows in the user message.

Design: docs/superpowers/specs/2026-05-14-router-intelligence-design.md
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

from shared.db import with_tenant
from shared.logging import get_logger

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
    match_source: str  # "trgm" | "fts" | "bare_id_exact"


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

_LABEL_TO_ENTITY_TYPE: dict[str, str] = {
    "Person": "person",
    "Repo": "repo",
    "Service": "service",
    "Ticket": "ticket",
    "PR": "pr",
    "Feature": "feature",
    "Decision": "decision",
    "ErrorGroup": "error_group",
    "File": "file_path",
    "Channel": "channel",
    "Session": "session",
    "Commit": "commit_sha",
}


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
    labels = list(_LABEL_TO_ENTITY_TYPE.keys())

    sql = """
    WITH ranked AS (
        SELECT
            label, canonical_id,
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
    SELECT label, canonical_id, display_name, last_seen_at_raw, rel
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
        entity_type = _LABEL_TO_ENTITY_TYPE.get(r["label"])
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

    kind_to_label = {"ticket": "Ticket", "pr": "PR", "commit_sha": "Commit"}
    by_label: dict[str, list[str]] = {}
    for kind, val in bare_ids:
        label = kind_to_label.get(kind)
        if label:
            by_label.setdefault(label, []).append(val)

    if not by_label:
        return []

    out: list[GroundingCandidate] = []
    async with with_tenant(customer_id) as conn:
        for label, ids in by_label.items():
            rows = await conn.fetch(
                """
                SELECT canonical_id,
                       coalesce(properties->>'name', canonical_id) AS display_name,
                       properties->>'last_seen_at' AS last_seen_at_raw
                FROM graph_nodes
                WHERE customer_id = $1 AND label = $2 AND canonical_id = ANY($3::text[])
                """,
                customer_id, label, ids,
            )
            entity_type = _LABEL_TO_ENTITY_TYPE.get(label, "")
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
    in that field; the rest of the bundle populates. If all three fail,
    returns an empty bundle (Haiku still works without grounding; the
    request still serves; the failure is logged).
    """
    t0 = time.perf_counter()
    tokens = _extract_tokens(query)
    bare_ids = _detect_bare_ids(query)

    results = await asyncio.gather(
        _fuzzy_match_entities(customer_id, tokens, per_type_cap=5, total_cap=20),
        _lookup_bare_id_matches(customer_id, bare_ids),
        _connected_sources(customer_id),
        return_exceptions=True,
    )

    candidates = results[0] if not isinstance(results[0], BaseException) else []
    bare_id_matches = results[1] if not isinstance(results[1], BaseException) else []
    sources = results[2] if not isinstance(results[2], BaseException) else []

    for label, r in zip(("fuzzy", "bare_id", "sources"), results, strict=True):
        if isinstance(r, BaseException):
            log.warning(
                "grounding.partial_failure",
                extra={"customer_id": customer_id, "subtask": label, "error": str(r)},
            )

    return GroundingBundle(
        candidates=candidates,
        connected_sources=sources,
        bare_id_matches=bare_id_matches,
        timing_ms=(time.perf_counter() - t0) * 1000,
    )
