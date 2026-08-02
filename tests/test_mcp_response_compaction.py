"""The MCP response-compaction layer must not strip caller-critical signals.

`compact_search` runs on every `search_knowledge` response and drops a
deny-list of diagnostic fields before the caller sees the payload. Two
top-level fields are deliberately exempt because an agent needs them to
decide whether to trust or re-run:

  * `confidence_breakdown` — pre-existing router signal.
  * `degraded`            — added with the degradation-visibility work.

The failure mode this guards is silent and total: if either name lands in
`_TOP_LEVEL_DROP`, the field vanishes from every response and a degraded
search becomes byte-identical to a healthy one again. There is no error, no
log, and no other test that would notice — which is the exact bug the
`degraded` flag exists to fix, reintroduced one deny-list entry later.
"""

from __future__ import annotations

from engine.mcp.clients._responses import _TOP_LEVEL_DROP, compact_search

# Fields that must survive compaction. Add to this list, never remove.
_MUST_SURVIVE = ("degraded", "degraded_reason", "confidence_breakdown")


def _payload() -> dict:
    return {
        "query": "outage",
        "results": [],
        "total_candidates": 0,
        "router_hit_cache": False,
        "trace_id": "t-1",
        "degraded": True,
        "degraded_reason": "provider_error_prefanout_fallback",
        "confidence_breakdown": {"EXTRACTED": 0, "INFERRED": 3, "AMBIGUOUS": 0},
        # A field that SHOULD be stripped, so this test also proves
        # compaction is actually running rather than passing everything.
        "timing_ms": {"total": 1.0},
    }


def test_critical_fields_are_not_in_the_deny_list() -> None:
    """Tripwire on the deny-list itself, independent of any payload."""
    for field in _MUST_SURVIVE:
        assert field not in _TOP_LEVEL_DROP, (
            f"{field!r} was added to _TOP_LEVEL_DROP — this silently removes it "
            "from every MCP response. See this module's docstring."
        )


def test_degraded_survives_compaction() -> None:
    out = compact_search(_payload())
    assert out["degraded"] is True
    assert out["degraded_reason"] == "provider_error_prefanout_fallback"


def test_healthy_response_still_carries_the_flag() -> None:
    """`degraded: false` must be present, not omitted.

    An absent key and a false key are the same thing to a careless caller,
    but only one of them is distinguishable from an old server that predates
    the field.
    """
    payload = _payload() | {"degraded": False, "degraded_reason": None}
    out = compact_search(payload)
    assert out["degraded"] is False
    assert out["degraded_reason"] is None


def test_compaction_still_strips_diagnostics() -> None:
    """Negative control: proves the deny-list is applied at all."""
    out = compact_search(_payload())
    assert "timing_ms" not in out
