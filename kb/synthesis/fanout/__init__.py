"""Per-source post-Phase-1 fan-out protocol + registry.

After a Phase 1 backfill (target=NULL) completes, the orchestrator's
``_maybe_fanout_phase2`` hook in ``backfill_app.py`` calls the source's
discoverer to get a list of per-target identifiers (repos for GitHub,
channels for Slack later, ...). One Phase 2 row is then inserted per
target, picked up by the existing ``_claim_one()`` worker loop.

This module exists so each new crawler can register its fan-out shape
without the orchestrator needing to know source-specific details.
"""

from __future__ import annotations

from services.synthesis.fanout.base import REGISTRY, BackfillFanout

# Eager registration so importing this module wires up every source
# discoverer at process start. Mirrors ``services/synthesis/crawlers/__init__.py``.
from services.synthesis.fanout.github import GitHubBackfillFanout

REGISTRY[GitHubBackfillFanout.source] = GitHubBackfillFanout()


__all__ = ["REGISTRY", "BackfillFanout"]
