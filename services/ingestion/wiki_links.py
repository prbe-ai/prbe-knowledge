"""Pure parsing utilities for `[[wiki-link]]` syntax.

A wiki page body can reference other entities through `[[Target]]` markers.
Two shapes are supported:

    [[Plain target]]            kind="plain", target="Plain target"
    [[Person: Mahit]]           kind="person", target="Mahit"
    [[Service: prbe-knowledge]] kind="service", target="prbe-knowledge"
    [[Repo: prbe-knowledge]]    kind="repo", target="prbe-knowledge"
    [[Ticket: PRB-9]]           kind="ticket", target="PRB-9"
    [[Feature: auth]]           kind="feature", target="auth"
    [[Decision: pgvector]]      kind="decision", target="pgvector"

A token before the colon that we don't recognize collapses to `kind="plain"`
(target keeps the full inner text) — so unknown forms don't silently drop
references; the connector still emits an unresolved-link entry the future
lint job can flag.

This module is dependency-free below `pydantic` so the dashboard / a future
synthesize cron can reuse it without dragging in connector wiring.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Final

from pydantic import BaseModel

# Match `[[...]]`. Inner text excludes nested brackets to keep the regex
# linear; `[[a [b] c]]` parses as a plain link with target `a [b] c`.
_LINK_RE: Final = re.compile(r"\[\[([^\[\]\n]+?)\]\]")

# Recognized type prefixes. The token before the colon is lower-cased + spaces
# stripped before the lookup, so `[[ Person: x]]` matches.
_KNOWN_KINDS: Final[frozenset[str]] = frozenset(
    {"person", "service", "repo", "ticket", "feature", "decision"}
)


class WikiLink(BaseModel):
    """One `[[...]]` token parsed out of a wiki body."""

    raw: str
    kind: str
    target: str
    span: tuple[int, int]


def parse_wiki_links(body: str) -> list[WikiLink]:
    """Extract every `[[...]]` reference from `body` in source order.

    Returns an empty list for empty / link-free bodies. Order is preserved so
    callers that emit graph edges keep deterministic output across runs.
    """
    if not body:
        return []
    out: list[WikiLink] = []
    for match in _LINK_RE.finditer(body):
        inner = match.group(1).strip()
        if not inner:
            continue
        kind, target = _split_kind(inner)
        out.append(
            WikiLink(
                raw=match.group(0),
                kind=kind,
                target=target,
                span=(match.start(), match.end()),
            )
        )
    return out


def _split_kind(inner: str) -> tuple[str, str]:
    """`Person: X` -> ("person", "X"); `X` -> ("plain", "X")."""
    if ":" not in inner:
        return "plain", inner
    head, _, rest = inner.partition(":")
    head_norm = head.strip().lower()
    target = rest.strip()
    if not target:
        # Colon present but no target — treat the whole thing as plain so the
        # raw form survives into dangling-link surfacing.
        return "plain", inner
    if head_norm in _KNOWN_KINDS:
        return head_norm, target
    return "plain", inner


_SLUG_STRIP_RE: Final = re.compile(r"[^a-z0-9]+")
_SLUG_TRIM_RE: Final = re.compile(r"-{2,}")


def slugify(title: str) -> str:
    """Lowercase, ASCII-fold, collapse non-alphanumerics to single dashes.

    Empty or all-non-alpha titles return "" — callers must reject that before
    using the slug as a `source_id` component, since a stable identifier is
    required for upserts.
    """
    if not title:
        return ""
    folded = unicodedata.normalize("NFKD", title)
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_only.lower()
    dashed = _SLUG_STRIP_RE.sub("-", lowered)
    collapsed = _SLUG_TRIM_RE.sub("-", dashed)
    return collapsed.strip("-")
