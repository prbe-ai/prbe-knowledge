"""Low-confidence threshold for the classifier no-match fallback (spec §6 step 2).

When the best embedding match's score is below ``LOW_CONF_THRESHOLD``,
the classifier returns ``no_match`` and the debugging agent falls back
to freeform RAG over Probe MCP. The threshold lives here (rather than
inline in the orchestrator) so a future tuning is one constant edit
plus one test edit, not a hunt through the orchestrator code path.

Empty-result is low-confidence by definition: spec §6 step 2 says
``no_match`` is the fallback whenever the best score is below threshold.
An empty result has no best score at all, which is operationally below
any threshold — callers pass ``best_score=0.0`` to mean "nothing came
back from pgvector" and ``is_low_confidence`` returns True.

Refs: docs/superpowers/specs/2026-04-29-debugging-knowledge-graph-design.md §6.
"""

from __future__ import annotations

LOW_CONF_THRESHOLD = 0.7


def is_low_confidence(best_score: float) -> bool:
    """True iff the best embedding match's score is below the no-match threshold.

    A score of ``0.0`` (the convention for "no matches at all") is
    always low-confidence, since ``0.0 < LOW_CONF_THRESHOLD`` for any
    threshold in ``(0, 1]``.
    """
    return best_score < LOW_CONF_THRESHOLD
