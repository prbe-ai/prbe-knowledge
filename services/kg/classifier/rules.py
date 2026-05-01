"""Cheap rule filter against `signature.must_match` (spec §6 step 2).

This is the first leg of the hybrid match pipeline: rules → embedding →
LLM tiebreaker. It runs in microseconds and exists purely to drop classes
whose structured predicates can't possibly match the incident before the
embedding step has to look at them.

Grammar — two operators only:

- ``<key> == <value>`` — string-coerced equality. The right-hand side may
  be wrapped in single or double quotes; quotes are stripped.
- ``<key> in [<v1>, <v2>, ...]`` — membership over a comma-separated list.
  Individual values may be quoted; quotes are stripped per-value.

That's the entire surface. There is deliberately no:

- Boolean composition (``AND`` / ``OR``) — multi-rule lists are implicit
  ``AND``; ``OR`` is expressed by adding another class.
- Comparison operators (``<``, ``>``, ``!=``) — fuzzier semantics belong
  in the embedding step or the LLM tiebreaker, not here.
- Regex match — same reason.

A rule the parser doesn't recognize fails closed (returns ``False``) so a
malformed rule drops the class rather than silently matching everything.

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md §6.
"""

from __future__ import annotations

import re

from ..schema import Frontmatter

_EQ = re.compile(r"^\s*(\w+)\s*==\s*(.+?)\s*$")
_IN = re.compile(r"^\s*(\w+)\s+in\s+\[(.+?)\]\s*$")


def _match_rule(rule: str, incident: dict[str, object]) -> bool:
    """Return True iff ``rule`` matches ``incident``.

    Unrecognized operators return False (fail-closed).
    """
    eq = _EQ.match(rule)
    if eq:
        key, expected = eq.group(1), eq.group(2).strip().strip("'\"")
        return str(incident.get(key)) == expected
    in_match = _IN.match(rule)
    if in_match:
        key = in_match.group(1)
        choices = [c.strip().strip("'\"") for c in in_match.group(2).split(",")]
        return str(incident.get(key)) in choices
    return False


def filter_by_rules(
    incident: dict[str, object], classes: list[Frontmatter]
) -> list[Frontmatter]:
    """Return the subset of ``classes`` whose every must_match rule matches."""
    return [
        c
        for c in classes
        if all(_match_rule(rule, incident) for rule in c.signature.must_match)
    ]
