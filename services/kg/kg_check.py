"""Validate that a class's wiki-links + frontmatter refs all resolve.

The "universe" parameter is the set of all known class IDs in the same
tenant. A class can reference itself (the validator excludes its own id
when computing the missing set) and any other id that exists. References
to source code (kind="source" wiki-links) are NOT validated here — those
are out-of-scope for kg_check.

See spec §5.3, §7.2.
"""
from __future__ import annotations

from .schema import BugClass
from .wiki_links import parse_wiki_links


class KgCheckError(Exception):
    """Raised when a class references a class_id that doesn't exist in the
    given universe (the set of all known class IDs in the same tenant)."""


def _frontmatter_refs(cls: BugClass) -> set[str]:
    r = cls.frontmatter.related
    return (
        set(r.analogous_to)
        | set(r.overlaps_with)
        | set(r.often_confused_with)
        | set(r.regressed_by)
    )


def _body_class_refs(cls: BugClass) -> set[str]:
    return {wl.target for wl in parse_wiki_links(cls.body) if wl.kind == "class"}


def check_class(cls: BugClass, *, universe: set[str]) -> None:
    """Validate cls's class-id references against universe.

    Raises KgCheckError listing every unresolved target. The class's
    own id is excluded from the unresolved set so a class can reference
    itself (e.g. as a sanity check or in 'self-similar' edges).
    """
    referenced = _frontmatter_refs(cls) | _body_class_refs(cls)
    missing = sorted(referenced - universe - {cls.frontmatter.id})
    if missing:
        raise KgCheckError(
            f"class {cls.frontmatter.id!r} references unknown classes: {missing}"
        )
