"""Gatherer agent — retrieval pipeline v2.

The gatherer is the retrieval pipeline. Deterministic grounding
(`services/retrieval/grounding.py`) extracts an entity bag from the raw
query; the agent then runs a tool-use loop on Fireworks gpt-oss-120B,
mandatorily fanning out the four channels on turn 1, exploring further
in parallel on subsequent turns, and emitting a `GathererOutput` Pydantic
payload (`response_format`-enforced).

Entry point: `services.retrieval.agent.loop.run_gatherer`.

Plan: docs/specs/agentic-search.md.
"""

from services.retrieval.agent.models import (
    DroppedCandidate,
    GatheredChunk,
    GatheredEntity,
    GathererNotes,
    GathererOutput,
)

__all__ = [
    "DroppedCandidate",
    "GatheredChunk",
    "GatheredEntity",
    "GathererNotes",
    "GathererOutput",
]
