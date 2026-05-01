"""Priority-ordered context-source walk (spec §6 step 4).

After the classifier picks an operative class, the traversal step loads
that class's ``context_sources`` in priority order:

- **P1 — always load.** These are the playbook's mandatory inputs (e.g.
  the relevant Sentry issue, the most-recent deploy diff). The agent
  cannot reason about the class without them.
- **P2 — load if any budget remains.** Useful augmentations (related
  Linear tickets, on-call runbooks). Skipped under tight budget so P1
  always survives.
- **P3 — load only if budget is generous.** Nice-to-haves the agent can
  decide it needs after reading P1+P2 (historical incidents, broad
  searches). Cheaper to defer than to crowd out P1 in a constrained
  context.

The ``5000`` threshold for P3 is a "generous budget" tunable. It's
inlined rather than extracted as a constant because there is no second
caller and no observability hook keying on it yet — when there is, lift
to a module-level constant or a settings field at that point. Keeping it
inline avoids a one-shot constant that pretends to be a contract.

Pure Python, sync, no DB. The function exists so the orchestrator (and
its unit tests) has a single place to assert priority semantics.
"""

from __future__ import annotations

from ..schema import ContextSource, Frontmatter


def walk_priority_edges(
    fm: Frontmatter, *, budget_remaining: int
) -> list[ContextSource]:
    """Return ``context_sources`` to load, ordered P1 -> P2 -> P3.

    Args:
        fm: The matched class's frontmatter. Its ``context_sources``
            list is partitioned by priority; input order within a tier
            is preserved.
        budget_remaining: Token budget the orchestrator still has to
            spend on context. P2 sources are included iff this is
            ``> 0``; P3 sources iff this is ``> 5000`` (the "generous
            budget" threshold).

    Returns:
        The selected sources in P1 -> P2 -> P3 order. Empty list if no
        sources match (e.g. ``fm.context_sources`` is empty or all
        sources fall behind the budget gate).
    """
    p1 = [s for s in fm.context_sources if s.priority == 1]
    p2 = [s for s in fm.context_sources if s.priority == 2]
    p3 = [s for s in fm.context_sources if s.priority == 3]
    out: list[ContextSource] = list(p1)
    if budget_remaining > 0:
        out.extend(p2)
    if budget_remaining > 5000:
        out.extend(p3)
    return out
