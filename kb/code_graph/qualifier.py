"""Cross-file resolution pass — promotes single-match AMBIGUOUS edges.

Per-language extractors do per-file resolution (imports + locals + self.*).
What's left AMBIGUOUS is a candidate name they couldn't tie to a single
target. The qualifier walks the full per-batch symbol set and:

    - If a single Symbol matches an AMBIGUOUS edge's candidate by `qname`
      tail (last segment), promotes the edge to a resolved CALLS / REFERENCES
      with `ambiguous=False` and the resolved `to_qname`.
    - Otherwise leaves the edge AMBIGUOUS for PR-B's promoter to inspect
      with LLM context.

This is the LOW-ambition cross-file step: pure name match, no type
inference. Per spec §4.5, deeper resolvers (jedi for Python, ts-morph for
TS, etc.) are fast-follow PRs once PR-B's LLM cost is measured.
"""

from __future__ import annotations

from collections import defaultdict

from kb.code_graph.types import ExtractResult


def promote_single_match(results: list[ExtractResult]) -> list[ExtractResult]:
    """Mutate `results` in place: promote single-match AMBIGUOUS edges.

    Builds a `tail_name -> [qname]` index from every Symbol across the
    batch, then walks each AMBIGUOUS edge:
      - len(candidates_for_tail) == 1: promote to non-ambiguous, target =
        the unique qname.
      - len(candidates) > 1: leave AMBIGUOUS, refresh `target_candidates`
        with the full match list so PR-B's promoter has better context.
      - len(candidates) == 0: leave as-is.

    Returns the same list reference (in-place mutation), so callers can
    chain it with no copy cost.
    """
    by_tail: dict[str, list[str]] = defaultdict(list)
    for r in results:
        for s in r.symbols:
            tail = s.qualified_name.rsplit(".", 1)[-1]
            by_tail[tail].append(s.qualified_name)

    for r in results:
        for edge in r.edges:
            if not edge.ambiguous:
                continue
            for raw_candidate in list(edge.target_candidates) or [edge.to_qname]:
                tail = raw_candidate.rsplit(".", 1)[-1]
                matches = by_tail.get(tail, [])
                if len(matches) == 1:
                    edge.to_qname = matches[0]
                    edge.ambiguous = False
                    edge.target_candidates = []
                    break
                if len(matches) > 1:
                    edge.target_candidates = matches
    return results


__all__ = ["promote_single_match"]
