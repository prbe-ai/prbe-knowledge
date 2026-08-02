"""`is_degraded` + its projection onto the response as `degraded`.

Why this file exists: the gatherer degrades to a 200 rather than a 503 by
design (#411), so a provider outage and a healthy answer were byte-identical
at the call site. The only signal was a reason string inside
`gatherer_notes.dropped[]`, which no consumer reads.

Two things are pinned here:

  1. `is_degraded` is the SINGLE predicate behind both
     `query_traces.failure_recovered` and the caller-visible `degraded`
     field. Before it existed the telemetry side open-coded
     `status != "ok"`, which counted an honestly-empty zero-recall query
     as a recovered failure.
  2. The classification is EXHAUSTIVE over `GathererStatus`. A new status
     added without a decision here fails `test_every_status_is_classified`
     rather than silently inheriting a default.
"""

from __future__ import annotations

from typing import get_args

from engine.retrieval.agent.adapter import to_query_response
from engine.retrieval.agent.models import (
    GatheredChunk,
    GathererNotes,
    GathererOutput,
    GathererStatus,
    is_degraded,
)

# Every status the loop can terminate with, split by whether the caller is
# holding degraded output. Kept as explicit literals (not derived from the
# implementation) so this is a real assertion, not a tautology.
_HEALTHY = {"ok", "zero_recall_short_circuit"}
_DEGRADED = {
    "passthrough_harness_fallback",
    "loop_timeout",
    "schema_violation",
    "tool_budget_exceeded",
    "no_llm_configured",
    "fatal_provider_error",
    "provider_error_prefanout_fallback",
    "context_overflow",
}


def test_every_status_is_classified() -> None:
    """Tripwire: adding an 11th GathererStatus must force a decision here.

    Without this, a new status silently inherits "degraded" from the
    exclusion list and nobody revisits whether that was right.
    """
    assert _HEALTHY | _DEGRADED == set(get_args(GathererStatus))
    assert not (_HEALTHY & _DEGRADED)


def test_healthy_statuses_are_not_degraded() -> None:
    for status in _HEALTHY:
        assert is_degraded(status) is False, status


def test_degraded_statuses_are_degraded() -> None:
    for status in _DEGRADED:
        assert is_degraded(status) is True, status


def test_zero_recall_is_honestly_empty_not_degraded() -> None:
    """The carve-out that distinguishes this from the old `status != "ok"`.

    A zero-recall short-circuit means grounding found no entities AND all
    four pre-fan-out channels returned nothing. That is a correct empty
    answer. Reporting it as degraded would train callers to ignore the flag.
    """
    assert is_degraded("zero_recall_short_circuit") is False


def test_none_status_is_not_degraded() -> None:
    """List-only / router paths never run the gatherer."""
    assert is_degraded(None) is False


def _output() -> GathererOutput:
    return GathererOutput(
        entities=[],
        chunks=[
            GatheredChunk(
                doc_id="github:acme/repo:pr:1",
                chunk_id="github:acme/repo:pr:1:c0",
                content="...",
                matched_via=["bm25"],
                why_relevant="test",
            )
        ],
        gatherer_notes=GathererNotes(confidence="low"),
    )


async def test_adapter_projects_degraded_and_reason() -> None:
    resp = await to_query_response(
        query="q",
        gathered=_output(),
        trace_id="t-1",
        timing_ms={},
        status="provider_error_prefanout_fallback",
    )
    assert resp.degraded is True
    assert resp.degraded_reason == "provider_error_prefanout_fallback"


async def test_adapter_healthy_run_carries_no_reason() -> None:
    """A reason string on a healthy response invites callers to branch on it."""
    resp = await to_query_response(
        query="q", gathered=_output(), trace_id="t-1", timing_ms={}, status="ok"
    )
    assert resp.degraded is False
    assert resp.degraded_reason is None


async def test_adapter_defaults_to_not_degraded_without_status() -> None:
    """Non-gatherer callers (and existing tests) omit `status` entirely."""
    resp = await to_query_response(
        query="q", gathered=_output(), trace_id="t-1", timing_ms={}
    )
    assert resp.degraded is False
    assert resp.degraded_reason is None
