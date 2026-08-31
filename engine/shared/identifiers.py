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
# Number refs: '#383', 'PR #232', 'pull request #536', and repo-qualified
# 'research-os PR #539'. Runs AFTER issue_ref masking so 'repo#123' (no
# space, exact kind) is never re-captured. The optional qualifier is
# whatever word precedes the frame word — it can be junk ('the PR #232');
# resolution treats it as a SOFT filter and falls back to the bare-number
# rule when it matches no repo, so a junk qualifier costs nothing.
_NUMBER_REF_RE = re.compile(
    r"(?:\b(?P<repo>[A-Za-z0-9_.-]+)\s+)?"
    r"(?:(?:PR|pull\s+request|issue)\s*)?"
    r"#(?P<num>\d{1,6})\b",
    re.IGNORECASE,
)
# PagerDuty incident ids: 'Q' + 13 uppercase alphanumerics. Uppercase-only
# on purpose — lowercase words can never collide, and PD never emits
# lowercase ids.
_PD_RE = re.compile(r"\bQ[A-Z0-9]{12,15}\b")
# Bare hex prefixes: 7-11 hex chars — a short commit sha ('ce09c43') or the
# first segment of a UUID ('5e0f3220'). Runs LAST, on a query with uuids,
# number refs and full shas already masked, so it only sees residue. The 7
# floor keeps short hex-looking words out; 12+ is the sha kind. Detection
# is loose BY DESIGN: resolution pins a prefix only when it matches exactly
# one document, so a false positive here costs one bounded lookup.
_HEX_PREFIX_RE = re.compile(r"\b[0-9a-fA-F]{7,11}\b")

# Common English words that satisfy the ticket shape (WORD-DIGITS). The
# ticket regex is intentionally loose; this list catches the handful of
# real-language collisions observed in query logs (e.g. "top-10",
# "utf-8"-adjacent forms). Resolution-gating catches the rest.
_TICKET_STOPWORDS = frozenset({"TOP", "UTF", "SHA", "MD", "BASE", "GPT", "V"})


@dataclass(frozen=True, slots=True)
class DetectedIdentifier:
    # "uuid" | "ticket" | "issue_ref" | "commit_sha" | "pd_incident"
    #   -> EXACT kinds: the user typed the whole identifier; equality/suffix
    #      arms resolve it, and whatever matches is right.
    # "number_ref" | "hex_prefix"
    #   -> INFERRED kinds: the lookup must EXPAND them (a number to a repo's
    #      PR, a prefix to a full sha/uuid) and may pin only when the
    #      expansion is unique — see id_lookup.lookup_identifiers.
    kind: str
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

    def _mask(text: str, span: tuple[int, int]) -> str:
        # Same-length padding so later spans stay aligned and later
        # patterns cannot see fragments of an already-claimed identifier.
        a, b = span
        return text[:a] + "\x00" * (b - a) + text[b:]

    masked = query
    for m in _UUID_RE.finditer(query):
        _add("uuid", m.group(0).lower())
        masked = masked.replace(m.group(0), "\x00" * len(m.group(0)))

    for m in _TICKET_RE.finditer(masked):
        prefix = m.group(1).rsplit("-", 1)[0].upper()
        if prefix in _TICKET_STOPWORDS:
            continue
        _add("ticket", m.group(1).upper())

    for m in list(_ISSUE_REF_RE.finditer(masked)):
        _add("issue_ref", m.group(1))
        masked = _mask(masked, m.span())

    for m in list(_NUMBER_REF_RE.finditer(masked)):
        repo = m.group("repo") or ""
        # A frame word captured as the qualifier ('PR #232' -> repo='PR')
        # is the regex doing its job on the optional branch; strip it.
        if repo.lower() in ("pr", "pull", "issue", "request"):
            repo = ""
        _add("number_ref", f"{repo}#{m.group('num')}" if repo else f"#{m.group('num')}")
        masked = _mask(masked, m.span())

    for m in list(_SHA_RE.finditer(masked)):
        _add("commit_sha", m.group(0).lower())
        masked = _mask(masked, m.span())

    for m in _PD_RE.finditer(masked):
        _add("pd_incident", m.group(0))

    for m in _HEX_PREFIX_RE.finditer(masked):
        _add("hex_prefix", m.group(0).lower())

    return out
