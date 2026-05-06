"""``BackfillFanout`` protocol — per-source Phase 2 target discoverer.

Each source that wants per-target deep-dive subtasks (currently just
GitHub; Slack/Linear when their crawlers land) registers a concrete
implementation. The orchestrator's post-Phase-1 hook calls
``discover_targets()`` and inserts one ``wiki_synthesis_runs`` row per
returned target with ``target=X``.
"""

from __future__ import annotations

from typing import ClassVar, Protocol, runtime_checkable

import httpx


@runtime_checkable
class BackfillFanout(Protocol):
    """Per-source post-Phase-1 fan-out discoverer.

    Implementations MUST NOT raise. Failures (auth, network, etc.)
    return an empty list — the orchestrator treats "0 targets" as
    "no Phase 2" and logs a warning at the call site. This keeps the
    fan-out hook a best-effort step (D4 swallow-and-log semantics).
    """

    source: ClassVar[str]

    async def discover_targets(
        self,
        *,
        customer_id: str,
        bearer: str,
        http: httpx.AsyncClient,
    ) -> list[str]:
        """Return up to ``BACKFILL_MAX_TARGETS_PER_SOURCE`` targets,
        sorted by recency-of-activity descending so the orchestrator
        truncates stale tail items first.
        """
        ...


# Module-level registry. Concrete fanouts register themselves at import
# time in ``services/synthesis/fanout/__init__.py``.
REGISTRY: dict[str, BackfillFanout] = {}
