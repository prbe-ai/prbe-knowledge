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
from typing import Literal

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
# Frame words a number ref can wear ('PR #232'). ONE constant feeds both the
# regex alternation and the qualifier strip test below — two hand-kept lists
# drift, and a frame word missing from the strip test becomes a phantom repo
# qualifier (review: conventions).
_NUMBER_FRAME_WORDS = ("PR", "pull request", "issue")
_NUMBER_REF_RE = re.compile(
    r"(?:\b(?P<repo>[A-Za-z0-9_.-]+)\s+)?"
    r"(?:(?:" + "|".join(w.replace(" ", r"\s+") for w in _NUMBER_FRAME_WORDS) + r")\s*)?"
    # The lookbehind keeps a bare '#N' from gluing onto the tail of a word:
    # without it, 'owner/repo#29' (an issue_ref, not masked) would ALSO mint
    # a phantom bare '#29'.
    r"(?<![\w.-])#(?P<num>\d{1,6})\b",
    re.IGNORECASE,
)
_NUMBER_FRAME_STRIP = frozenset(
    w.lower() for phrase in _NUMBER_FRAME_WORDS for w in phrase.split()
)
# PagerDuty incident ids: 'Q' + 13-15 uppercase alphanumerics, at least one
# digit. Uppercase-only AND digit-required on purpose: real PD ids are
# random base-alnum and always carry digits, while ALL-CAPS English words
# (QUALIFICATIONS) never do (review: line-by-line).
_PD_RE = re.compile(r"\bQ(?=[A-Z0-9]*\d)[A-Z0-9]{12,15}\b")
# Bare hex prefixes: 7-11 hex chars with AT LEAST ONE letter — a short
# commit sha ('ce09c43') or a UUID first segment ('5e0f3220'). The letter
# requirement excludes every pure-decimal token (dates '20260831', epoch
# fragments, order numbers) — those are overwhelmingly quantities, and each
# false positive costs a tenant-scoped scan and can even mis-pin (review:
# efficiency/cross-file). ~4% of real short shas are all-digit and are
# deliberately given up to close that class. Hyphen-aware lookarounds keep
# segments of hyphenated chains (run slugs, DEADBEEF-12 tickets) out.
_HEX_PREFIX_RE = re.compile(
    r"(?<![\w-])(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{7,11}(?![\w-])"
)

# Common English words that satisfy the ticket shape (WORD-DIGITS). The
# ticket regex is intentionally loose; this list catches the handful of
# real-language collisions observed in query logs (e.g. "top-10",
# "utf-8"-adjacent forms). Resolution-gating catches the rest.
_TICKET_STOPWORDS = frozenset({"TOP", "UTF", "SHA", "MD", "BASE", "GPT", "V"})


# EXACT kinds: the user typed the whole identifier; equality/suffix arms
# resolve it, and whatever matches is right. INFERRED kinds: the lookup must
# EXPAND them (a number to a repo's PR, a prefix to a full sha/uuid) and may
# pin only when the expansion is unique — see id_lookup.lookup_identifiers.
# One declaration, imported by the lookup router — a kind in neither set is
# silently dropped there, so the vocabulary must not fork (review).
IdentifierKind = Literal[
    "uuid", "ticket", "issue_ref", "commit_sha", "pd_incident",
    "number_ref", "hex_prefix",
]
EXACT_KINDS: frozenset[IdentifierKind] = frozenset(
    {"uuid", "ticket", "issue_ref", "commit_sha", "pd_incident"}
)
INFERRED_KINDS: frozenset[IdentifierKind] = frozenset(
    {"number_ref", "hex_prefix"}
)


@dataclass(frozen=True, slots=True)
class DetectedIdentifier:
    kind: IdentifierKind
    canonical_id: str
    # number_ref only: the structured halves, so the lookup never has to
    # re-split canonical_id and re-derive invariants held in this file
    # (review: simplification). qualifier may be junk ('the PR #232') — the
    # lookup treats it as a SOFT filter.
    qualifier: str = ""
    number: str = ""


def detect_identifiers(query: str) -> list[DetectedIdentifier]:
    """Every typed identifier in `query`, canonicalized, in query order.

    Deduplicated on canonical_id (the same ticket typed twice is one
    identifier). Returns [] for identifier-free queries, which is the
    common case and must stay O(regex).
    """
    out: list[DetectedIdentifier] = []
    seen: set[str] = set()

    def _add(kind: IdentifierKind, canonical: str) -> None:
        if canonical not in seen:
            seen.add(canonical)
            out.append(DetectedIdentifier(kind=kind, canonical_id=canonical))

    # ONLY uuids are masked: their hex segments would re-report under the
    # sha and prefix kinds. Nothing else needs it — issue_ref and number_ref
    # would collide only on an identical canonical string (deduped by _add),
    # and masking MORE was a verified regression: a number ref's qualifier
    # span swallowed an adjacent commit sha and the exact lane lost the one
    # identifier it resolves best (review: removed-behavior).
    masked = query
    for m in _UUID_RE.finditer(query):
        _add("uuid", m.group(0).lower())
        masked = masked.replace(m.group(0), "\x00" * len(m.group(0)))

    for m in _TICKET_RE.finditer(masked):
        prefix = m.group(1).rsplit("-", 1)[0].upper()
        if prefix in _TICKET_STOPWORDS:
            continue
        _add("ticket", m.group(1).upper())

    for m in _ISSUE_REF_RE.finditer(masked):
        _add("issue_ref", m.group(1))

    for m in _NUMBER_REF_RE.finditer(masked):
        repo = m.group("repo") or ""
        # A frame word captured as the qualifier ('PR #232' -> repo='PR')
        # is the regex doing its job on the optional branch; strip it.
        if repo.lower() in _NUMBER_FRAME_STRIP:
            repo = ""
        # A hex/uuid-shaped qualifier is an ADJACENT IDENTIFIER the greedy
        # capture swallowed ('commit 9027120fe3ab #383'), never a repo name;
        # drop it so the ref stays bare and the neighbor keeps its own kind.
        if re.fullmatch(r"[0-9a-fA-F-]{7,40}", repo):
            repo = ""
        num = m.group("num")
        canonical = f"{repo}#{num}" if repo else f"#{num}"
        if canonical not in seen:
            seen.add(canonical)
            out.append(
                DetectedIdentifier(
                    kind="number_ref",
                    canonical_id=canonical,
                    qualifier=repo,
                    number=num,
                )
            )

    for m in _SHA_RE.finditer(masked):
        _add("commit_sha", m.group(0).lower())

    for m in _PD_RE.finditer(masked):
        _add("pd_incident", m.group(0))

    for m in _HEX_PREFIX_RE.finditer(masked):
        _add("hex_prefix", m.group(0).lower())

    return out
