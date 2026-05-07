"""Pure parsing utilities for `[[wiki-page-link]]` syntax.

Parses typed entity references inside wiki page bodies:

    [[Plain target]]            kind="plain", target="Plain target"
    [[Person: Mahit]]           kind="person", target="Mahit"
    [[Service: prbe-knowledge]] kind="service", target="prbe-knowledge"
    [[Repo: prbe-knowledge]]    kind="repo", target="prbe-knowledge"
    [[Anything: foo]]           kind="anything", target="foo"

The token before the colon is free-form (the LLM picks page kinds),
matched against a URL-safe regex (lowercase letters/digits/underscore,
1-32 chars). A token that doesn't match the shape collapses to
``kind="plain"`` so unknown forms don't silently drop references — the
connector still emits an unresolved-link entry the future lint job can
flag.

Sibling parser: `services/kg/links.py` (PR #47, debugging knowledge graph)
parses a different `[[...]]` grammar — `[[ClassName/file/path]]` and
`[[#section-anchor]]` — for class refs with line+col coordinates. The two
modules share only the `[[...]]` surface syntax; their kind taxonomies and
output shapes are deliberately disjoint. Callers that need both should
import them under explicit aliases.

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

# Wiki types are free-form — anything matching this regex (URL-safe slug
# shape) becomes a typed link. Other prefixes collapse to kind="plain".
_KIND_RE: Final = re.compile(r"^[a-z][a-z0-9_]{0,31}$")


class WikiPageLink(BaseModel):
    """One `[[...]]` token parsed out of a wiki page body.

    Distinct from `services/kg/links.py:WikiLink` (PR #47) — that module
    parses class-graph refs with line/col coordinates, this one parses
    typed entity refs (`[[Person: X]]`) for graph-node emission.
    """

    raw: str
    kind: str
    target: str
    span: tuple[int, int]


def parse_page_links(body: str) -> list[WikiPageLink]:
    """Extract every `[[...]]` reference from `body` in source order.

    Returns an empty list for empty / link-free bodies. Order is preserved so
    callers that emit graph edges keep deterministic output across runs.
    """
    if not body:
        return []
    out: list[WikiPageLink] = []
    for match in _LINK_RE.finditer(body):
        inner = match.group(1).strip()
        if not inner:
            continue
        kind, target = _split_kind(inner)
        out.append(
            WikiPageLink(
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
    if _KIND_RE.match(head_norm):
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
