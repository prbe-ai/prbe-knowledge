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
from engine.shared.models import RetrieveResponse

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


async def test_explicit_none_status_reports_not_degraded() -> None:
    """Non-gatherer callers pass None EXPLICITLY — there is no default.

    `status` is a required keyword precisely so that forgetting it raises
    rather than quietly reporting a healthy run. Passing None is the
    supported way to say "no gatherer outcome to report".
    """
    resp = await to_query_response(
        query="q", gathered=_output(), trace_id="t-1", timing_ms={}, status=None
    )
    assert resp.degraded is False
    assert resp.degraded_reason is None


async def test_omitting_status_is_a_type_error() -> None:
    """The fail-open default is gone. Omission must be loud.

    Two of three gatherer call sites forgot this parameter when it had a
    default, shipping degraded=False on degraded responses. This pins the
    guard that makes that impossible.
    """
    import pytest

    with pytest.raises(TypeError, match="status"):
        await to_query_response(  # type: ignore[call-arg]
            query="q", gathered=_output(), trace_id="t-1", timing_ms={}
        )


def test_healthy_response_cannot_carry_a_reason() -> None:
    """`degraded=False, degraded_reason="..."` must be unrepresentable.

    The adapter never builds that pair, but mocks and future endpoints can
    construct the model directly. Normalizing at the model means a caller
    can branch on `degraded` alone without cross-checking the other field.
    """
    resp = RetrieveResponse(
        query="q",
        total_candidates=0,
        router_hit_cache=False,
        trace_id="t-1",
        degraded=False,
        degraded_reason="context_overflow",
    )
    assert resp.degraded_reason is None


def test_degraded_without_a_reason_stays_legal() -> None:
    """"Degraded, cause unknown" is a real state — do not normalize it away."""
    resp = RetrieveResponse(
        query="q",
        total_candidates=0,
        router_hit_cache=False,
        trace_id="t-1",
        degraded=True,
    )
    assert resp.degraded is True
    assert resp.degraded_reason is None


def test_unknown_status_string_fails_safe_to_degraded() -> None:
    """The exclusion list must fail SAFE for values outside the Literal.

    `is_degraded` accepts plain `str` (a status read back off a stored trace
    row never round-trips through the Literal, which is erased at runtime).
    Anything unrecognized has to land on the conservative side, or a future
    refactor to an inclusion list silently inverts the default.
    """
    assert is_degraded("some_future_status") is True
    assert is_degraded("") is True
    assert is_degraded("OK") is True  # case-sensitive; not the "ok" literal


def test_every_loop_call_site_forwards_status() -> None:
    """Every `to_query_response` call in the loop must pass `status=`.

    This is the tripwire for the bug class that shipped in review: the
    original change wired ONLY the terminal return and missed both early
    returns, so `no_llm_configured` — a genuinely degraded, empty response —
    reported `degraded=False`. Every adapter-level test still passed, because
    they call `to_query_response` directly and supply `status` themselves.

    Nothing about the type system catches this: `status` is optional by
    design (non-gatherer callers legitimately omit it), so an omission at a
    gatherer call site is silently valid and produces exactly the invisible
    failure this whole feature exists to eliminate. A static check on the
    call sites is the only thing that fails loudly.
    """
    import ast
    import inspect

    from engine.retrieval.agent import loop as loop_module

    tree = ast.parse(inspect.getsource(loop_module))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "to_query_response"
    ]
    assert calls, "no to_query_response calls found — did the symbol get renamed?"
    missing = [c.lineno for c in calls if not any(kw.arg == "status" for kw in c.keywords)]
    assert not missing, (
        f"to_query_response at line(s) {missing} omit status= — those responses "
        "will report degraded=False no matter how the gatherer actually ended."
    )
