"""Unit tests for the inferred-edges extractor validation pipeline.

Phase-0b: Both Anthropic and Gemini paths now route through
`shared.llm.acompletion`. Tests patch the wrapper (referenced via
`services.ingestion.inferred_edges.extractor.acompletion`) rather than
the provider SDKs.

All tests mock the LLM call -- we test the validation logic, not the
model. Each drop reason is exercised: unknown_endpoint, unknown_type,
self_edge, bad_justification, unknown_confidence,
forced_confidence_demoted, bad_format.

The >50% unknown_endpoint kill-switch is also tested.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.ingestion.inferred_edges.bundle import Bundle, BundleDoc
from services.ingestion.inferred_edges.extractor import (
    _estimate_cost,
    extract_edges,
)
from shared.constants import HAIKU_MODEL


@pytest.fixture(autouse=True)
def _pin_to_haiku(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin every test in this file to the Haiku model so the Anthropic
    code path runs. Tests that need the Gemini path override this
    fixture's setattr with their own."""
    monkeypatch.setattr(
        "services.ingestion.inferred_edges.extractor.INFERRED_EDGES_MODEL",
        HAIKU_MODEL,
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


def _make_mock_conn(
    existing_nodes: set[tuple[str, str]] | None = None,
    *,
    titles: dict[tuple[str, str], str] | None = None,
) -> AsyncMock:
    """Mock asyncpg connection that returns a preset node set.

    `_load_existing_nodes` joins documents → graph_nodes and surfaces a
    `display_title` column. For legacy fixtures we default each title to
    the empty string — the topic-overlap sanity check (Rule 6) skips
    edges where both endpoints have empty titles (defers to the other
    validators + the kill-switch), so existing tests keep asserting on
    the validators they originally targeted. Tests exercising Rule 6
    pass an explicit `titles` map keyed by (label, canonical_id).
    """
    nodes = existing_nodes if existing_nodes is not None else _EXISTING_NODES
    titles = titles or {}
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[
            {
                "label": lbl,
                "canonical_id": cid,
                "display_title": titles.get((lbl, cid), ""),
            }
            for (lbl, cid) in nodes
        ]
    )
    return conn


def _litellm_text_response(text: str, *, in_tok: int = 1000, out_tok: int = 200) -> SimpleNamespace:
    """LiteLLM-shaped ChatCompletion response carrying plain text.

    The inferred-edges extractor reads
    `choices[0].message.content` as the model's body. Token counts
    surface via `usage.prompt_tokens` / `usage.completion_tokens`
    which `shared.llm_tools.usage_tokens` extracts.
    """
    message = SimpleNamespace(content=text, tool_calls=None)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    usage = SimpleNamespace(
        prompt_tokens=in_tok,
        completion_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        prompt_tokens_details=None,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


def _mock_anthropic_response(edges: list[dict]) -> SimpleNamespace:
    """Anthropic path: the assistant-prefill `[` trick means the
    response body is the CONTINUATION (starts with edges or `]`).
    Strip the leading `[` from json.dumps(edges) to match real
    behavior. Empty edges array -> `]` after stripping; the
    extractor recognises that as the empty-array sentinel.
    """
    full_json = json.dumps(edges)
    body = full_json[1:] if full_json.startswith("[") else full_json
    return _litellm_text_response(body)


def _patch_acompletion(side_effect):
    """Patch the `acompletion` symbol imported by the extractor module."""
    return patch(
        "services.ingestion.inferred_edges.extractor.acompletion",
        side_effect=side_effect,
    )


def _patch_llm(edges: list[dict]):
    """Patch the LLM wrapper to return a fixed Anthropic-shaped
    edges list (prefilled `[` already stripped)."""
    return _patch_acompletion(
        AsyncMock(return_value=_mock_anthropic_response(edges))
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
async def test_bundle_kill_switch_unknown_endpoint_near_total() -> None:
    """When unknown_endpoint ratio is at-or-near 100%, the bundle is
    failed entirely — the LLM clearly hallucinated the whole run. The
    one valid edge (if it happens to land) is also dropped to err on
    the side of not polluting the graph from a suspect run.

    Threshold is `_UNKNOWN_ENDPOINT_FAIL_RATIO = 0.9` (was 0.5 — see the
    constant's comment for why it was raised). To trip 0.9 we need
    almost all edges to be unknown; the test uses 9 ghosts + 1 valid =
    90.0% which (correctly) doesn't trip (`>` not `>=`); we use 10
    ghosts + 1 valid = ~90.9% to land just past the gate.
    """
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Service", "canonical_id": f"ghost-{i}"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Ghost reference that does not exist",
        }
        for i in range(10)
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
    assert result.edges == []


@pytest.mark.asyncio
async def test_bundle_kill_switch_not_triggered_mixed_unknown() -> None:
    """Kill-switch does NOT fire when unknown_endpoint stays below the
    threshold. 4 ghost + 1 valid = 80% — this is the live shape that
    auto-rationale-body PRs hit (PR #328 had ratios of 0.75-0.80).
    Under the old 0.5 threshold this whole batch was killed; under the
    new 0.9 threshold the 1 valid edge flows through."""
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

    assert result.bundle_failed is False
    assert len(result.edges) == 1
    assert result.dropped.get("unknown_endpoint", 0) == 4


@pytest.mark.asyncio
async def test_bundle_kill_switch_not_triggered_below_half() -> None:
    """Below-half ratio (1 ghost + 1 valid = 50%) is well clear of the
    new threshold — still emits the valid edge."""
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
    assert len(result.edges) == 1


# ---------------------------------------------------------------------------
# Tests: topic-overlap sanity check (Rule 6) — drops hallucinations
# where the `why` rationale and endpoint titles share zero topical
# content words. Catches "LLM picked a real canonical_id but the why
# describes topics that doc doesn't cover" failure mode that the
# raised 0.9 kill-switch threshold can no longer catch alone.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule6_drops_edge_with_zero_title_overlap() -> None:
    """Live shape: LLM cites PR #X in `why` describing topic Y, but
    PR #X's actual title is about topic Z. Rule 6 catches it because
    `why` shares zero content words with either endpoint's title."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc2"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Session discusses graph traversals which doc2 implements",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn(titles={
        ("Document", "doc1"): "Claude Code session 4c65a43c",
        ("Document", "doc2"): "fix(migrate-job): correct DSN routing to CNPG",
    })
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)
    assert result.edges == []
    assert result.dropped.get("unrelated_topic", 0) == 1


@pytest.mark.asyncio
async def test_rule6_keeps_edge_when_topic_words_overlap() -> None:
    """When the `why` and at least one endpoint title share a topical
    content word (case-insensitive, len>=3, identifier-prefixes
    stripped), Rule 6 passes."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc2"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "Session discusses why-chain entity emission rule",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn(titles={
        ("Document", "doc1"): "Claude Code session 4c65a43c",
        ("Document", "doc2"): "feat(gatherer): why-chain section + entity emission rule",
    })
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)
    assert len(result.edges) == 1
    assert result.dropped.get("unrelated_topic", 0) == 0


@pytest.mark.asyncio
async def test_rule6_skipped_when_both_endpoints_have_empty_titles() -> None:
    """Stub-upserted nodes (entity-cluster aliases, freshly-created
    canonical_ids before enrichment runs) often have no title. The
    sanity check can't catch hallucinations without title signal, so
    it defers — the kill-switch is the only backstop for these. This
    test pins the deferred behavior."""
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc2"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "completely unrelated rationale text",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()  # titles default to "" — Rule 6 defers
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)
    assert len(result.edges) == 1
    assert result.dropped.get("unrelated_topic", 0) == 0


@pytest.mark.asyncio
async def test_rule6_overlap_ignores_identifier_prefixes() -> None:
    """`pr`, `issue`, `the`, `discusses`, `session`, etc. are stripped
    from both sides of the content-word set — they're trivial-overlap
    risks where the LLM cites a PR/issue identifier and the endpoint
    title happens to lead with "PR" or "session".

    Setup: why and titles share only words that are in the strip list,
    plus content words that DON'T overlap (`rollout` vs `feature`).
    Without the strip list "session" / "PR" / "discusses" would create
    trivial overlap; with it, the actual content words diverge and Rule
    6 correctly rejects.
    """
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Document", "canonical_id": "doc2"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "PR session discusses the rollout strategy",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn(titles={
        # Strip-list words appear in both, real content words don't.
        ("Document", "doc1"): "PR feature implementation session",
        ("Document", "doc2"): "Discussion: PR migration the session",
    })
    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn)
    assert result.edges == []
    assert result.dropped.get("unrelated_topic", 0) == 1


# ---------------------------------------------------------------------------
# Tests: no API key
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_api_key_returns_empty() -> None:
    """Without ANTHROPIC_API_KEY (and no LLM_GATEWAY_URL), return empty
    result (don't crash)."""
    bundle = _make_bundle()
    conn = _make_mock_conn()

    # Wipe both — the gateway check is the new fall-through that
    # allows gateway-routed tenants to operate without provider keys.
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "", "LLM_GATEWAY_URL": ""}, clear=False):
        result = await extract_edges(bundle, conn)

    assert result.edges == []
    assert result.bundle_failed is False
    assert result.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Tests: cost estimate
# ---------------------------------------------------------------------------


def test_estimate_cost_zero_tokens() -> None:
    cost = _estimate_cost(HAIKU_MODEL, 0, 0)
    assert cost == 0.0


def test_estimate_cost_positive_haiku() -> None:
    # 1M input + 1M output at Haiku pricing ($1.00 in / $5.00 out = $6).
    cost = _estimate_cost(HAIKU_MODEL, 1_000_000, 1_000_000)
    assert cost == pytest.approx(6.0, abs=0.01)


def test_estimate_cost_positive_flash_lite() -> None:
    # 1M input + 1M output at Flash Lite pricing ($0.25 in / $1.50 out = $1.75).
    cost = _estimate_cost("gemini-3.1-flash-lite", 1_000_000, 1_000_000)
    assert cost == pytest.approx(1.75, abs=0.01)


def test_estimate_cost_typical_call() -> None:
    # Typical call at Haiku pricing: ~60k input + ~2k output tokens.
    cost = _estimate_cost(HAIKU_MODEL, 60_000, 2_000)
    assert cost < 0.10


def test_estimate_cost_unknown_model_returns_zero() -> None:
    cost = _estimate_cost("some-unknown-model", 1_000_000, 1_000_000)
    assert cost == 0.0


# ---------------------------------------------------------------------------
# Tests: empty bundle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_bundle_returns_empty_result() -> None:
    """An empty bundle (no docs) returns an empty ExtractionResult."""
    bundle = Bundle(customer_id="cust-x", anchor_doc_id="doc1")
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


def _patch_llm_text(text: str):
    """Patch acompletion to return a LiteLLM response with `text` content."""
    return _patch_acompletion(
        AsyncMock(return_value=_litellm_text_response(text, out_tok=0))
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("response_body", ["", " ", "\n", "]", " ]\n"])
async def test_empty_response_body_is_treated_as_zero_edges(response_body: str) -> None:
    """Regression: an empty-or-just-]-after-prefill response means no edges,
    not a JSON parse failure.
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


@pytest.mark.asyncio
async def test_why_field_preserved_on_extracted_edge() -> None:
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
# Tests: rate-limit backoff
# ---------------------------------------------------------------------------


def _make_rate_limit_error() -> Exception:
    """Build something the extractor will recognise as a rate-limit error
    by class name (legacy SDK shape). Used to exercise the backoff path."""

    class RateLimitError(Exception):
        pass

    return RateLimitError("rate limited")


@pytest.mark.asyncio
async def test_rate_limit_then_success_does_not_fail_bundle() -> None:
    """First call raises RateLimitError; backoff fires; second call succeeds."""
    bundle = _make_bundle()
    conn = _make_mock_conn()

    successful_response = _mock_anthropic_response([])  # empty -> no edges
    rate_limit_then_success = AsyncMock(
        side_effect=[_make_rate_limit_error(), successful_response]
    )

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_acompletion(rate_limit_then_success),
        patch(
            "services.ingestion.inferred_edges.extractor.asyncio.sleep",
            new_callable=AsyncMock,
        ),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed
    assert result.edges == []
    assert rate_limit_then_success.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exhausted_marks_bundle_failed() -> None:
    """If every retry hits a rate limit, the bundle is marked failed."""
    bundle = _make_bundle()
    conn = _make_mock_conn()

    always_rate_limited = AsyncMock(side_effect=_make_rate_limit_error())

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_acompletion(always_rate_limited),
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
    """A non-rate-limit error propagates immediately."""
    bundle = _make_bundle()
    conn = _make_mock_conn()

    class APIConnectionError(Exception):
        pass

    once_then_dies = AsyncMock(side_effect=APIConnectionError("connection refused"))

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_acompletion(once_then_dies),
        patch(
            "services.ingestion.inferred_edges.extractor.asyncio.sleep",
            new_callable=AsyncMock,
        ) as mock_sleep,
    ):
        result = await extract_edges(bundle, conn)

    assert result.bundle_failed
    assert "APIConnectionError" in (result.bundle_fail_reason or "")
    assert once_then_dies.call_count == 1
    mock_sleep.assert_not_awaited()


# ---------------------------------------------------------------------------
# Tests: Gemini dispatch path (model-based provider routing)
# ---------------------------------------------------------------------------


def _mock_gemini_response(edges: list[dict]) -> SimpleNamespace:
    """Gemini path: full JSON array on `choices[0].message.content`
    (no `[` prefill trick — structured-output mode returns a complete
    array)."""
    return _litellm_text_response(json.dumps(edges))


@pytest.mark.asyncio
async def test_gemini_path_extracts_valid_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the configured model has a `gemini-` prefix, the extractor
    routes through the LiteLLM Gemini provider and consumes its
    full-array JSON response (no `[` prefill). The validation pipeline
    runs the same way -- the model ends up tagged on the InferredEdge."""
    monkeypatch.setattr(
        "services.ingestion.inferred_edges.extractor.INFERRED_EDGES_MODEL",
        "gemini-3.1-flash-lite",
    )
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

    fake = AsyncMock(return_value=_mock_gemini_response(edges))
    with (
        patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=False),
        _patch_acompletion(fake),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed
    assert len(result.edges) == 1
    edge = result.edges[0]
    assert edge.edge_type == "DISCUSSES"
    assert edge.confidence == "INFERRED"
    assert edge.from_canonical_id == "doc1"
    assert edge.to_canonical_id == "AUTH-123"
    # `model` field tags every edge with its source LLM.
    assert edge.model == "gemini-3.1-flash-lite"
    # Sanity: the call routed through the gemini/ LiteLLM prefix.
    assert fake.await_count == 1
    kwargs = fake.await_args.kwargs
    assert kwargs["model"].startswith("gemini/")


@pytest.mark.asyncio
async def test_gemini_path_skips_when_google_api_key_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No GOOGLE_API_KEY AND no LLM_GATEWAY_URL -> empty result, no
    LLM call. Mirrors the Haiku-no-ANTHROPIC-key safety hatch so the
    worker doesn't crash in credential-less environments."""
    monkeypatch.setattr(
        "services.ingestion.inferred_edges.extractor.INFERRED_EDGES_MODEL",
        "gemini-3.1-flash-lite",
    )
    bundle = _make_bundle()
    conn = _make_mock_conn()

    fake = AsyncMock()
    with (
        patch.dict("os.environ", {}, clear=True),  # nuke all keys
        _patch_acompletion(fake),
    ):
        result = await extract_edges(bundle, conn)

    assert not result.bundle_failed
    assert result.edges == []
    # The wrapper was never called.
    fake.assert_not_awaited()


@pytest.mark.asyncio
async def test_explicit_model_override_wins_over_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Caller can pass `model=` explicitly to force a specific provider,
    overriding INFERRED_EDGES_MODEL."""
    monkeypatch.setattr(
        "services.ingestion.inferred_edges.extractor.INFERRED_EDGES_MODEL",
        "gemini-3.1-flash-lite",
    )
    edges = [
        {
            "from": {"label": "Document", "canonical_id": "doc1"},
            "to": {"label": "Ticket", "canonical_id": "AUTH-123"},
            "edge_type": "DISCUSSES",
            "confidence": "INFERRED",
            "why": "explicit model override test",
        }
    ]
    bundle = _make_bundle()
    conn = _make_mock_conn()

    with (
        patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}),
        _patch_llm(edges),
    ):
        result = await extract_edges(bundle, conn, model=HAIKU_MODEL)

    assert not result.bundle_failed
    assert len(result.edges) == 1
    # Edge tagged with the override, not the default.
    assert result.edges[0].model == HAIKU_MODEL


# Note: `_anthropic_module` (the legacy Anthropic SDK import) is no
# longer used by the production call path, but the module retains the
# import for the `RateLimitError` class-name match. The module-level
# `MagicMock` reference below is intentionally unused — it exists to
# silence flake-8 "unused import" should the import-time fallback path
# ever surface a None.
_ = MagicMock
