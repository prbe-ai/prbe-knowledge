"""Unit tests for the inferred-edges extractor validation pipeline.

All tests mock the LLM call -- we test the validation logic, not the model.
Each drop reason is exercised: unknown_endpoint, unknown_type, self_edge,
bad_justification, unknown_confidence, forced_confidence_demoted,
bad_format.

The >50% unknown_endpoint kill-switch is also tested.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.ingestion.inferred_edges.bundle import Bundle, BundleDoc
from services.ingestion.inferred_edges.extractor import (
    _estimate_cost,
    extract_edges,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bundle(customer_id: str = "cust-test", anchor_doc_id: str = "doc1") -> Bundle:
    """Bundle with two docs and their nodes in the manifest."""
    bundle = Bundle(customer_id=customer_id, anchor_doc_id=anchor_doc_id)
    bundle.docs = [
        BundleDoc(
            doc_id="doc1",
            customer_id=customer_id,
            source_system="slack",
            title="Slack thread",
            content="Discussion about auth bug",
            token_count=50,
        ),
        BundleDoc(
            doc_id="doc2",
            customer_id=customer_id,
            source_system="linear",
            title="Linear issue",
            content="AUTH-123 fix the login timeout",
            token_count=40,
        ),
    ]
    bundle.total_tokens = 90
    return bundle


# Existing nodes in the "DB" for tests.
_EXISTING_NODES: set[tuple[str, str]] = {
    ("Document", "doc1"),
    ("Document", "doc2"),
    ("Ticket", "AUTH-123"),
}


def _make_mock_conn(existing_nodes: set[tuple[str, str]] | None = None) -> AsyncMock:
    """Mock asyncpg connection that returns a preset node set."""
    nodes = existing_nodes if existing_nodes is not None else _EXISTING_NODES
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[
            {"label": lbl, "canonical_id": cid}
            for (lbl, cid) in nodes
        ]
    )
    return conn


def _mock_llm_response(edges: list[dict]) -> MagicMock:
    """Create a mock Anthropic response simulating the prefilled response.

    The extractor sends an assistant prefill `[` so Haiku's response only
    contains what comes AFTER `[`. Real example for 1 edge:
        prefill: `[`
        response.content[0].text: `{"from": ..., "to": ...}]`

    So we strip the leading `[` from json.dumps(edges) to match real
    behavior. The empty-edges case (`[]`) becomes `]` after stripping --
    the extractor recognises that as the empty-array sentinel.
    """
    response = MagicMock()
    full_json = json.dumps(edges)
    body = full_json[1:] if full_json.startswith("[") else full_json
    response.content = [MagicMock(text=body)]
    response.usage = MagicMock(input_tokens=1000, output_tokens=200)
    return response


def _patch_llm(edges: list[dict]):
    """Context manager that patches the Anthropic client to return `edges`."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_mock_llm_response(edges))

    def _fake_client_class(api_key=None):
        return mock_client

    # _anthropic_module is the module-level reference set at import time.
    # We patch the AsyncAnthropic class on it so the extractor function
    # picks up our mock when it does: _anthropic_module.AsyncAnthropic(api_key=...)
    return patch(
        "services.ingestion.inferred_edges.extractor._anthropic_module.AsyncAnthropic",
        _fake_client_class,
    )


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_edge_parsed() -> None:
    """A well-formed INFERRED edge passes all validation."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Ticket", "canonical_id": "AUTH-123"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Slack thread explicitly mentions AUTH-123 as the root cause",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.edge_type == "DISCUSSES"
    assert edge.confidence == "INFERRED"
    assert edge.from_canonical_id == "doc1"
    assert edge.to_canonical_id == "AUTH-123"
    assert result.cost_usd > 0


# ---------------------------------------------------------------------------
# Tests: drop reasons
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_drop_unknown_endpoint() -> None:
    """Edges whose endpoint does not exist in graph_nodes are dropped."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Service", "canonical_id": "nonexistent-service"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Claims to reference a nonexistent service",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.dropped.get("unknown_endpoint", 0) == 1
    assert len(result.edges) == 0


@pytest.mark.asyncio
async def test_drop_unknown_type() -> None:
    """Edges with an unknown edge_type are dropped."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc2"},
            "edge_type": "INVENTED_TYPE",
            "confidence": "INFERRED",
            "why": "Some reason here",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.dropped.get("unknown_type", 0) == 1
    assert len(result.edges) == 0


@pytest.mark.asyncio
async def test_drop_self_edge() -> None:
    """Self-edges (from == to) are dropped."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc1"},
            "edge_type": "RELATES_TO",
            "confidence": "INFERRED",
            "why": "Doc references itself for some reason",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.dropped.get("self_edge", 0) == 1
    assert len(result.edges) == 0


@pytest.mark.asyncio
async def test_drop_bad_justification_empty() -> None:
    """Edges with an empty `why` field are dropped."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Ticket", "canonical_id": "AUTH-123"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.dropped.get("bad_justification", 0) == 1
    assert len(result.edges) == 0


@pytest.mark.asyncio
async def test_drop_bad_justification_too_long() -> None:
    """Edges with a `why` > 200 chars are dropped."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Ticket", "canonical_id": "AUTH-123"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "x" * 201,
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.dropped.get("bad_justification", 0) == 1
    assert len(result.edges) == 0


@pytest.mark.asyncio
async def test_drop_unknown_confidence() -> None:
    """Edges with an unrecognized confidence value are dropped."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Ticket", "canonical_id": "AUTH-123"},
            "edge_type": "DISCUSSES",
            "confidence": "MEDIUM",
            "why": "Something plausible",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.dropped.get("unknown_confidence", 0) == 1
    assert len(result.edges) == 0


@pytest.mark.asyncio
async def test_force_demote_extracted_to_ambiguous() -> None:
    """EXTRACTED confidence is force-demoted to AMBIGUOUS, edge is kept."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Ticket", "canonical_id": "AUTH-123"},
            "edge_type": "DISCUSSES",
            "confidence": "EXTRACTED",
            "why": "The LLM incorrectly claimed EXTRACTED confidence",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    # Edge kept but confidence demoted
    assert len(result.edges) == 1
    assert result.edges[0].confidence == "AMBIGUOUS"
    # Counter incremented for the demotion
    assert result.dropped.get("forced_confidence_demoted", 0) == 1


@pytest.mark.asyncio
async def test_drop_bad_format_non_dict() -> None:
    """Non-dict items in the array are dropped with bad_format."""
    edges = ["not a dict", 42, None]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.dropped.get("bad_format", 0) == 3
    assert len(result.edges) == 0


# ---------------------------------------------------------------------------
# Tests: kill-switch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bundle_kill_switch_unknown_endpoint_majority() -> None:
    """When >50% of edges have unknown_endpoint, the bundle is failed entirely."""
    # Build N edges with unknown endpoints to trigger the kill-switch.
    # 4 unknown + 1 valid = 80% unknown > 50% threshold.
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Service", "canonical_id": f"ghost-{i}"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Ghost reference that does not exist",
        }
        for i in range(4)
    ] + [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc2"},
            "edge_type": "RELATES_TO",
            "confidence": "INFERRED",
            "why": "Valid cross-source edge in the same bundle",
        }
    ]

    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.bundle_failed is True
    assert "unknown_endpoint" in result.bundle_fail_reason
    # All extracted edges should be wiped by the kill-switch
    assert result.edges == []


@pytest.mark.asyncio
async def test_bundle_kill_switch_not_triggered_below_threshold() -> None:
    """Kill-switch does NOT fire when unknown_endpoint <= 50% of total."""
    # 1 unknown + 1 valid = 50% unknown, which is not > 50%. Edge passes.
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Service", "canonical_id": "ghost-service"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Unknown service reference",
        },
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc2"},
            "edge_type": "RELATES_TO",
            "confidence": "INFERRED",
            "why": "Valid cross-source reference in bundle",
        },
    ]

    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert result.bundle_failed is False
    # 1 valid edge survives
    assert len(result.edges) == 1


# ---------------------------------------------------------------------------
# Tests: no API key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_api_key_returns_empty() -> None:
    """Without ANTHROPIC_API_KEY, return empty result (don't crash)."""
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": ""}, clear=False):
        result = await extract_edges(bundle, conn)

    assert result.edges == []
    assert result.bundle_failed is False  # Not a failure, just no-op
    assert result.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Tests: cost estimate
# ---------------------------------------------------------------------------


def test_estimate_cost_zero_tokens() -> None:
    cost = _estimate_cost(0, 0)
    assert cost == 0.0


def test_estimate_cost_positive() -> None:
    # 1M input + 1M output at standard Haiku pricing
    cost = _estimate_cost(1_000_000, 1_000_000)
    assert cost > 0.0
    # ~$4.80 total
    assert 4.0 < cost < 6.0


def test_estimate_cost_typical_call() -> None:
    # Typical call: ~60k input tokens, ~2k output tokens
    cost = _estimate_cost(60_000, 2_000)
    assert cost < 0.10  # Should be well under $0.10


# ---------------------------------------------------------------------------
# Tests: empty bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_bundle_returns_empty_result() -> None:
    """An empty bundle (no docs) returns an empty ExtractionResult."""
    bundle = Bundle(customer_id="cust-x", anchor_doc_id="doc1")
    # docs is empty by default
    conn = _make_mock_conn()

    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        result = await extract_edges(bundle, conn)

    assert result.edges == []
    assert result.cost_usd == 0.0
    assert not result.bundle_failed


# ---------------------------------------------------------------------------
# Tests: LLM returns empty array
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_llm_response_array() -> None:
    """When the LLM returns [], ExtractionResult has no edges."""
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm([]),
    ):
        result = await extract_edges(bundle, conn)

    assert result.edges == []
    assert not result.bundle_failed


# ---------------------------------------------------------------------------
# Tests: prefill empty-text behaviour (REGRESSION GUARD)
# ---------------------------------------------------------------------------
# In production we saw the LLM return an empty assistant message body
# (queue_ids 120-127, 100% bundle_fail rate). Without the empty-text
# fallback the extractor crashed with "json_parse_failed: Expecting
# value: line 1 column 1 (char 0)". With the fallback an empty body
# means "no edges" and the call succeeds with zero edges.
#
# Real Anthropic responses we have to handle gracefully (because of
# the assistant prefill `[`):
#   1. Empty body: "" -> []
#   2. Just the closing bracket: "]" -> []
#   3. Whitespace: "   \n" -> []
#   4. Whitespace + `]`: " ]" -> []


def _mock_response_with_text(text: str) -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(text=text)]
    response.usage = MagicMock(input_tokens=1000, output_tokens=0)
    return response


def _patch_llm_text(text: str):
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_mock_response_with_text(text))
    return patch(
        "services.ingestion.inferred_edges.extractor._anthropic_module.AsyncAnthropic",
        lambda api_key=None: mock_client,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("response_body", ["", " ", "\n", "]", " ]\n"])
async def test_empty_response_body_is_treated_as_zero_edges(response_body: str) -> None:
    """Regression: an empty-or-just-]-after-prefill response means no edges,
    not a JSON parse failure. Was the production crash on queue_ids 120-127.
    """
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm_text(response_body),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed, f"empty body {response_body!r} should not fail"
    assert result.edges == []
    assert not result.dropped


@pytest.mark.asyncio
async def test_truncated_response_falls_through_to_parse_failure() -> None:
    """If the model truncates mid-element (e.g. max_tokens hit), the
    salvage-by-appending-`]` doesn't produce valid JSON; the failure
    branch records it without crashing.
    """
    bundle = _make_bundle()
    conn = _make_mock_conn()
    truncated = '{"from": {"label": "Document", "canonical_id": "doc1"'

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm_text(truncated),
    ):
        result = await extract_edges(bundle, conn)

    assert result.bundle_failed
    assert "json_parse_failed" in (result.bundle_fail_reason or "")


# ---------------------------------------------------------------------------
# Tests: leading-`]` empty-array regression (post-PR175 follow-up)
# ---------------------------------------------------------------------------
# In production after PR #175 we still saw 142 json_parse_failed drops on
# the probe-founders backfill. Root cause: model emits `]\n\n(no edges
# found)` -- our `stripped == "]"` check was too strict. The candidate
# became `[]\n\n(no edges found)` which fails parse. Generalised to
# `stripped.startswith("]")`: any leading `]` after the prefill means
# the array is empty; trailing commentary is discarded.


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_body",
    [
        "]\n\n(no edges found)",
        "]\n\nNo significant cross-source relationships in this bundle.",
        "]\n",
        "] // empty",
        "]    ",
    ],
)
async def test_leading_close_bracket_with_garbage_is_empty_array(
    response_body: str,
) -> None:
    """Regression: any leading `]` after the prefill `[` means empty array.
    Trailing model commentary must be discarded, not parsed.
    """
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm_text(response_body),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed, f"body {response_body!r} should not fail"
    assert result.edges == []
    assert not result.dropped


# ---------------------------------------------------------------------------
# Tests: `why` justification flows through to ExtractionResult
# ---------------------------------------------------------------------------
# Worker-side bug found in production: 0 of 3,817 inferred edges had a
# `why` field persisted because worker.py built GraphEdgeSpec without
# properties. The worker fix is tested separately in
# test_inferred_edges_worker.py; this confirms the extractor itself
# carries the field through.


@pytest.mark.asyncio
async def test_why_field_preserved_on_extracted_edge() -> None:
    """Confirm the LLM's `why` justification ends up on InferredEdge.why
    so the worker has something to persist into graph_edges.properties.
    """
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Ticket", "canonical_id": "AUTH-123"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Slack thread debugging the AUTH-123 outage",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed
    assert len(result.edges) == 1
    assert result.edges[0].why == "Slack thread debugging the AUTH-123 outage"


# ---------------------------------------------------------------------------
# Tests: rate-limit backoff (PR-B follow-up)
# ---------------------------------------------------------------------------
# 1383 of 3257 bundles (42%) failed on `RateLimitError` during the
# probe-founders backfill burst. The fix wraps the LLM call in
# exponential backoff inside a single attempt -- a transient rate-limit
# window no longer kills the bundle. Non-rate-limit errors propagate
# immediately (no slow failure modes for genuine errors).


def _make_rate_limit_error() -> Exception:
    """Build something the extractor will recognise as a rate-limit error.

    The extractor matches by class name (`RateLimitError`) to avoid pulling
    the anthropic package into the type system. So a plain Exception
    subclass with that name does the job.
    """

    class RateLimitError(Exception):
        pass

    return RateLimitError("rate limited")


@pytest.mark.asyncio
async def test_rate_limit_then_success_does_not_fail_bundle() -> None:
    """First call raises RateLimitError; backoff fires; second call succeeds.
    Bundle is processed normally, not marked failed.
    """
    bundle = _make_bundle()
    conn = _make_mock_conn()

    successful_response = _mock_llm_response([])  # empty -> no edges
    rate_limit_then_success = AsyncMock(
        side_effect=[_make_rate_limit_error(), successful_response]
    )
    mock_client = MagicMock()
    mock_client.messages.create = rate_limit_then_success

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch(
            "services.ingestion.inferred_edges.extractor._anthropic_module.AsyncAnthropic",
            lambda api_key=None: mock_client,
        ),
        # Patch sleep to avoid wall-clock waits in tests.
        patch(
            "services.ingestion.inferred_edges.extractor.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed
    assert result.edges == []
    # Two attempts: one rate-limited, one success.
    assert rate_limit_then_success.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exhausted_marks_bundle_failed() -> None:
    """If every retry hits a rate limit, the bundle is marked failed
    with llm_call_failed: RateLimitError (so the queue worker can
    retry on next claim, when the rate window has likely passed).
    """
    bundle = _make_bundle()
    conn = _make_mock_conn()

    always_rate_limited = AsyncMock(side_effect=_make_rate_limit_error())
    mock_client = MagicMock()
    mock_client.messages.create = always_rate_limited

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch(
            "services.ingestion.inferred_edges.extractor._anthropic_module.AsyncAnthropic",
            lambda api_key=None: mock_client,
        ),
        patch(
            "services.ingestion.inferred_edges.extractor.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        result = await extract_edges(bundle, conn)

    assert result.bundle_failed
    assert "RateLimitError" in (result.bundle_fail_reason or "")
    # Per-attempt retries (matches _RATE_LIMIT_MAX_RETRIES = 4).
    assert always_rate_limited.call_count == 4


@pytest.mark.asyncio
async def test_non_rate_limit_error_does_not_retry() -> None:
    """A non-rate-limit error (e.g. APIConnectionError, AuthenticationError)
    propagates immediately. We do NOT want exponential backoff burning
    minutes on a genuine API/auth failure.
    """
    bundle = _make_bundle()
    conn = _make_mock_conn()

    class APIConnectionError(Exception):
        pass

    once_then_dies = AsyncMock(side_effect=APIConnectionError("connection refused"))
    mock_client = MagicMock()
    mock_client.messages.create = once_then_dies

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        patch(
            "services.ingestion.inferred_edges.extractor._anthropic_module.AsyncAnthropic",
            lambda api_key=None: mock_client,
        ),
        patch(
            "services.ingestion.inferred_edges.extractor.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        result = await extract_edges(bundle, conn)

    assert result.bundle_failed
    assert "APIConnectionError" in (result.bundle_fail_reason or "")
    # Exactly one call -- no retry on non-rate-limit errors.
    assert once_then_dies.call_count == 1
    # No sleep -- we did NOT enter the backoff loop.
    mock_sleep.assert_not_awaited()
