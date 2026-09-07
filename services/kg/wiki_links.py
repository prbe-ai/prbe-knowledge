"""Wiki-link parser for class bodies.

Class bodies are markdown prose with two flavors of inline reference,
borrowed from lat.md (spec §5.3):

- ``[[class-id]]`` — a cross-class reference (e.g. ``[[auth-403-rbac]]``).
- ``[[src/path.ts#symbol]]`` — a source-code reference (path with optional
  ``#symbol`` anchor).

Disambiguation is purely syntactic at this layer: a target containing ``/``
or ``#`` is treated as a source-code ref; everything else is a class ref.
Resolution (does the class id exist? does the file path exist?) happens in
the ``kg-check`` link-validator (Task 8) — this parser only extracts the
references and where they appear in the body.

Parsing rules:

- Targets are stripped of surrounding whitespace; an empty target after
  strip is dropped.
- The pattern is greedy-friendly but bounded by ``[``, ``]``, and newline,
  so unclosed brackets and stray ``[single]`` tokens never match.
- ``line`` is 1-indexed (matches editor / lsp convention); ``col`` is
  0-indexed (chars from start of line to the opening ``[``).

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md §5.3.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

WIKI_LINK_RE = re.compile(r"\[\[([^\[\]\n]+?)\]\]")


@dataclass(frozen=True)
class WikiLink:
    """A single ``[[...]]`` reference extracted from a class body.

    ``kind`` is ``"source"`` if the target contains ``/`` or ``#``
    (path-like), otherwise ``"class"``. ``line`` is 1-indexed; ``col`` is
    0-indexed and points at the opening ``[`` of the ``[[``.
    """

    kind: Literal["class", "source"]
    target: str
    line: int
    col: int


def parse_wiki_links(body: str) -> list[WikiLink]:
    """Extract every ``[[target]]`` from ``body``.

    Empty targets and unclosed brackets are skipped. Targets are stripped
    of surrounding whitespace before classification and emission.
    """
    out: list[WikiLink] = []
    for m in WIKI_LINK_RE.finditer(body):
        target = m.group(1).strip()
        if not target:
            continue
        kind: Literal["class", "source"] = (
            "source" if "/" in target or "#" in target else "class"
        )
        prefix = body[: m.start()]
        line = prefix.count("\n") + 1
        col = m.start() - (prefix.rfind("\n") + 1)
        out.append(WikiLink(kind=kind, target=target, line=line, col=col))
    return out
