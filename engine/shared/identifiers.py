"""Typed-identifier detection — ONE parser, request-scoped, raw-query only.

The identifier story had two half-parsers that agreed on nothing:
`grounding._detect_bare_ids` detected tickets/PRs/shas (uppercase-only, no
UUIDs) for the graph-lookup flow, while `id_lookup.is_lookup_candidate`
qualified UUIDs/tickets/shas/issue-refs but only ever saw whatever callers
happened to feed it -- and since the gatherer cutover, nothing fed it at
all. The outside-voice review (2026-08-31, D5) pinned the consequences:
lowercase tickets were never detected anywhere, UUIDs were never bare-id
detected, and the entity bag mixed user-typed identifiers with
LLM-extracted GUESSES with no provenance -- so nothing downstream could
safely treat "an identifier" as certain.

This module is the single answer. `detect_identifiers` reads the RAW USER
QUERY ONLY -- never extractor output -- so every result is something the
user actually typed. That provenance is the load-bearing property: the
id-pins lane pins these at the top of results as exact matches, and a
hallucinated id pinned with certainty would be worse than any miss.

Canonicalization is part of detection, not a caller chore: tickets
uppercase (documents.source_id equality is case-sensitive and Linear ids
are stored uppercase), hex forms lowercase. Detection order matters --
UUIDs are masked out before the sha pattern runs, or every UUID's 12-hex
tail would double-report as a commit sha.

Detection is deliberately loose where RESOLUTION is exact: a false
positive here costs one indexed lookup that matches nothing and pins
nothing. False negatives are the expensive direction -- they demote a
precise query back to similarity search, which is the 0.47 recall this
lane exists to fix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order-sensitive: UUID first (masked before sha runs), then ticket,
# issue-ref, sha. All case-insensitive on DETECTION; canonical case is
# applied per kind below.
_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
# LETTERS-DIGITS ticket codes (PRB-17, prb-17). Hyphen-aware lookarounds,
# not plain \b: a word boundary sits between `tunneling-` and `sambar`, so
# `tunneling-sambar-254` would shed its tail as ticket SAMBAR-254 (caught
# by this module's own smoke test). Resolution-gating would make that a
# harmless no-pin, but a run-slug query mislabeled as an unresolved ticket
# also blocks the pure-lookup path for no reason.
_TICKET_RE = re.compile(r"(?<![\w-])([A-Za-z][A-Za-z0-9]{1,9}-\d{1,6})(?![\w-])")
# repo#123 / owner/repo#123.
_ISSUE_REF_RE = re.compile(r"\b([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)?#\d{1,6})\b")
# Bare commit shas: 12-40 hex. Runs AFTER UUID masking. The 12 floor keeps
# ordinary words and short hex fragments out.
_SHA_RE = re.compile(r"\b[0-9a-fA-F]{12,40}\b")

# Common English words that satisfy the ticket shape (WORD-DIGITS). The
# ticket regex is intentionally loose; this list catches the handful of
# real-language collisions observed in query logs (e.g. "top-10",
# "utf-8"-adjacent forms). Resolution-gating catches the rest.
_TICKET_STOPWORDS = frozenset({"TOP", "UTF", "SHA", "MD", "BASE", "GPT", "V"})


@dataclass(frozen=True, slots=True)
class DetectedIdentifier:
    kind: str  # "uuid" | "ticket" | "issue_ref" | "commit_sha"
    canonical_id: str


def detect_identifiers(query: str) -> list[DetectedIdentifier]:
    """Every typed identifier in `query`, canonicalized, in query order.

    Deduplicated on canonical_id (the same ticket typed twice is one
    identifier). Returns [] for identifier-free queries, which is the
    common case and must stay O(regex).
    """
    out: list[DetectedIdentifier] = []
    seen: set[str] = set()

    def _add(kind: str, canonical: str) -> None:
        if canonical not in seen:
            seen.add(canonical)
            out.append(DetectedIdentifier(kind=kind, canonical_id=canonical))

    masked = query
    for m in _UUID_RE.finditer(query):
        _add("uuid", m.group(0).lower())
        # Replace with same-length padding so later spans stay aligned
        # and the sha pattern cannot see the uuid's hex segments.
        masked = masked.replace(m.group(0), "\x00" * len(m.group(0)))

    for m in _TICKET_RE.finditer(masked):
        prefix = m.group(1).rsplit("-", 1)[0].upper()
        if prefix in _TICKET_STOPWORDS:
            continue
        _add("ticket", m.group(1).upper())

    for m in _ISSUE_REF_RE.finditer(masked):
        _add("issue_ref", m.group(1))

    for m in _SHA_RE.finditer(masked):
        _add("commit_sha", m.group(0).lower())

    return out
