"""Gatherer agent loop tests — fat-tools + tool_choice=required + terminal-tool.

Mocked acompletion + mocked grounding/extraction/pre-fan-out. No live LLM,
no live DB. Covers:
- Happy path: terminal tool call on turn 1 → GathererOutput → QueryResponse adapter
- Multi-turn: exploration tool call → terminal on next turn
- Budget exhaustion: harness forces a final terminal turn
- Transient LLMError + citable pre-fan-out → low-confidence fallback
- Fatal/unknown LLMError → HTTPException(503)
- No tool calls (provider violated tool_choice=required) → schema_violation
- No-LLM-configured short-circuit (test env / bootstrap / self-host without keys)
- Per-stage latency telemetry: turn_latencies_ms, tool_latencies_ms
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from engine.retrieval.agent.loop import (
    LoopState,
    _affinity_key,
    _build_prefanout_doc_meta,
    _build_user_message,
    _count_tokens,
    _derive_source_system_from_doc_id,
    _empty_passthrough,
    _enforce_context_budget,
    _extract_cache_hit_rate,
    _format_inferred_chains,
    _has_citable_prefanout_evidence,
    _parse_terminal_args,
    _render_prefanout_budgeted,
    _seed_for_query,
    run_gatherer,
)
from engine.retrieval.agent.models import (
    EntityExtraction,
    ExtractedEntity,
    GathererOutput,
    SearchOptions,
)
from engine.retrieval.agent.tools import TERMINAL_TOOL_NAME
from engine.retrieval.grounding import GroundingBundle
from engine.shared.constants import SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS
from engine.shared.llm import LLMError
from engine.shared.llm_tools import is_context_overflow, is_transient_provider_error
from engine.shared.models import QueryRequest

# ============================================================
# Fixtures: fake LiteLLM response builder
# ============================================================

def _mk_resp(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    content: str | None = None,
    prompt_tokens: int = 100,
    cached_tokens: int = 0,
    reasoning_content: str | None = None,
    system_fingerprint: str | None = None,
) -> SimpleNamespace:
    """Build a SimpleNamespace mimicking a LiteLLM chat-completion response.

    `reasoning_content` simulates the gpt-oss harmony `analysis` block
    that LiteLLM surfaces as `message.reasoning_content`. Default None
    so existing tests don't shift; pass a string to assert the loop
    captures it onto state.

    `system_fingerprint` simulates the provider's backend identifier
    (Cerebras returns it on every response). Default None; pass a
    string to assert the loop captures it onto state.
    """
    tcs = []
    for tc in tool_calls or []:
        tcs.append(SimpleNamespace(
            id=tc.get("id", "call_x"),
            function=SimpleNamespace(
                name=tc["name"],
                arguments=json.dumps(tc.get("arguments", {})) if not isinstance(tc.get("arguments"), str) else tc["arguments"],
            ),
        ))
    msg = SimpleNamespace(
        content=content,
        tool_calls=tcs,
        reasoning_content=reasoning_content,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=50,
            prompt_tokens_details={"cached_tokens": cached_tokens},
        ),
        system_fingerprint=system_fingerprint,
    )


def _final_emission_args(*, confidence: str = "high", chunks: int = 2) -> dict[str, Any]:
    """Build the args the model would pass to emit_gatherer_output."""
    return {
        "entities": [],
        "chunks": [
            {
                "doc_id": f"doc-{i}",
                "chunk_id": f"chunk-{i}",
                "content": f"content body {i}",
                "matched_via": ["vector"],
                "why_relevant": f"surfaced via vector channel, rank {i+1}",
            }
            for i in range(chunks)
        ],
        "gatherer_notes": {
            "turns_used": 1,
            "tools_called": ["emit_gatherer_output"],
            "confidence": confidence,
            "dropped": [],
        },
    }


def _terminal_call(args: dict[str, Any] | None = None, *, id: str = "term_1") -> dict[str, Any]:
    return {
        "id": id,
        "name": TERMINAL_TOOL_NAME,
        "arguments": json.dumps(args or _final_emission_args()),
    }


@pytest.fixture
def fake_request() -> SimpleNamespace:
    return SimpleNamespace(state=SimpleNamespace())


@pytest.fixture
def fake_bundle() -> GroundingBundle:
    return GroundingBundle()


@pytest.fixture(autouse=True)
def _force_llm_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    """Most tests run as if LLM is configured. Tests that want the
    short-circuit override this fixture."""
    monkeypatch.setattr(
        "engine.retrieval.agent.loop._no_llm_configured", lambda: False
    )


@pytest.fixture(autouse=True)
def _stub_grounding_extraction_prefanout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the three upstream-of-loop calls so tests exercise only the
    agent loop itself. Individual tests can re-patch as needed.

    Pre-fan-out is stubbed to return one minimal vector hit so the
    `zero_recall_short_circuit` fast-path doesn't fire by default; tests
    that want to assert the short-circuit override this fixture with an
    empty `sub_queries`.
    """
    monkeypatch.setattr(
        "engine.retrieval.agent.loop._build_bundle_with_token_fallback",
        AsyncMock(return_value=GroundingBundle()),
    )
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.extract_entities_with_llm",
        AsyncMock(return_value=EntityExtraction()),
    )
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.execute_search",
        AsyncMock(return_value={"sub_queries": [{
            "query": "stub",
            "grounded_entities": [],
            "vector": [{"doc_id": "stub:0", "score": 0.5,
                        "source_system": "github", "title": "stub",
                        "content": "stub"}],
            "bm25": [], "graph": [], "inferred_edge": [],
        }]}),
    )


# ============================================================
# Pure helpers
# ============================================================

def test_affinity_key_is_customer_scoped_not_query_scoped() -> None:
    """Affinity hash routes by customer_id only, NOT by query.

    Rationale: the static system prompt + tool-defs prefix (~2.4K
    tokens) must cache-hit ACROSS queries, not just across the turns
    of a single query. Yesterday's digest (2026-05-19) measured a
    turn-0 mean cache_hit_rate of 0.13 across 126 traces because the
    old `(customer_id, query)` hash routed every new query to a fresh
    Cerebras replica — guaranteeing a cold KV-cache on turn 0.

    Multi-turn cache continuity is preserved because Cerebras's prefix
    cache is content-addressed; both turns route to the same replica
    so turn 1 still hits the warm prefix turn 0 wrote.

    See PRB-12 + the docstring on `loop._affinity_key`.
    """
    a = _affinity_key("cust-1", "what is PRB-17")
    b = _affinity_key("cust-1", "what is PRB-17")
    assert a == b
    # Different customers → different replicas (avoids one customer's
    # KV cache thrashing another's working set).
    assert _affinity_key("cust-2", "what is PRB-17") != a
    # Same customer + different query → SAME replica. This is the
    # behavior change: under the prior implementation these differed.
    assert _affinity_key("cust-1", "what is PRB-99") == a
    # Query value must not influence the hash at all — even an empty
    # query yields the same key for the same customer.
    assert _affinity_key("cust-1", "") == a
    assert len(a) == 32


def test_extract_cache_hit_rate_from_dict_details() -> None:
    resp = _mk_resp(prompt_tokens=100, cached_tokens=70)
    assert _extract_cache_hit_rate(resp) == pytest.approx(0.7)


def test_extract_cache_hit_rate_handles_missing() -> None:
    resp = SimpleNamespace(usage=None)
    assert _extract_cache_hit_rate(resp) is None
    resp = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=0))
    assert _extract_cache_hit_rate(resp) is None


def test_citable_prefanout_evidence_requires_doc_id_and_content() -> None:
    citable = {
        "sub_queries": [{
            "vector": [{"doc_id": "github:doc:1", "content": "evidence"}],
            "bm25": [],
            "graph": [],
            "inferred_edge": [],
        }]
    }
    missing_doc_id = {
        "sub_queries": [{
            "vector": [{"content": "orphaned evidence"}],
            "bm25": [],
            "graph": [],
            "inferred_edge": [],
        }]
    }
    blank_content = {
        "sub_queries": [{
            "vector": [{"doc_id": "github:doc:1", "content": "   "}],
            "bm25": [],
            "graph": [],
            "inferred_edge": [],
        }]
    }

    assert _has_citable_prefanout_evidence(citable) is True
    assert _has_citable_prefanout_evidence(missing_doc_id) is False
    assert _has_citable_prefanout_evidence(blank_content) is False


def test_parse_terminal_args_valid_dict() -> None:
    out = _parse_terminal_args(_final_emission_args(chunks=1))
    assert out is not None
    assert isinstance(out, GathererOutput)
    assert len(out.chunks) == 1
    assert out.gatherer_notes.confidence == "high"


def test_parse_terminal_args_valid_json_string() -> None:
    out = _parse_terminal_args(json.dumps(_final_emission_args(chunks=2)))
    assert out is not None
    assert len(out.chunks) == 2


def test_parse_terminal_args_invalid_returns_none() -> None:
    """Unrecoverable inputs (None, JSON parse failures) still return None.
    Schema-shape violations now degrade gracefully to an empty
    GathererOutput rather than None — see
    test_parse_terminal_args_lenient_accepts_unknown_fields."""
    assert _parse_terminal_args(None) is None
    assert _parse_terminal_args("{not valid json") is None


def test_parse_terminal_args_lenient_accepts_unknown_fields() -> None:
    """Cerebras gpt-oss-120b emits emit_gatherer_output args with input-
    shaped fields (title/source/url) instead of the strict schema. With
    extra='ignore' + optional fields with defaults, those become empty
    GathererOutput instead of None — same downstream effect as the old
    schema_violation status, but the loop trace records the model DID
    emit (just with wrong fields)."""
    out = _parse_terminal_args({"foo": "bar"})
    assert out is not None
    assert out.entities == []
    assert out.chunks == []
    # Defaults from model_config
    assert out.gatherer_notes.turns_used == 0
    assert out.gatherer_notes.confidence == "medium"


def test_parse_terminal_args_lenient_derives_doc_id_from_chunk_id() -> None:
    """Pre-parse coercion: when the model emits a chunk without doc_id
    but with a chunk_id that encodes the parent doc, derive it."""
    out = _parse_terminal_args({
        "chunks": [
            {
                "chunk_id": "github:owner/repo:pr:42:chunk:3",
                "content": "snippet",
            },
            {
                # No chunk_id either — can't derive, will be dropped
                "content": "orphan",
            },
        ],
    })
    assert out is not None
    assert len(out.chunks) == 1
    assert out.chunks[0].doc_id == "github:owner/repo:pr:42"
    assert out.chunks[0].chunk_id == "github:owner/repo:pr:42:chunk:3"


def test_parse_terminal_args_aliases_id_to_canonical_id() -> None:
    """Fourth Cerebras failure mode: entities emitted with `id` instead
    of `canonical_id` (common API convention)."""
    out = _parse_terminal_args({
        "entities": [
            {"id": "github:prbe-ai/prbe-backend:pr:256", "label": "PR"},
            {"canonical_id": "github:prbe-ai/prbe-backend:pr:299"},
            {"id": "notion:doc:abc", "title": "ignored"},
        ],
    })
    assert out is not None
    assert len(out.entities) == 3
    assert out.entities[0].canonical_id == "github:prbe-ai/prbe-backend:pr:256"
    assert out.entities[1].canonical_id == "github:prbe-ai/prbe-backend:pr:299"
    assert out.entities[2].canonical_id == "notion:doc:abc"


def test_parse_terminal_args_repairs_truncated_json() -> None:
    """Fifth Cerebras failure mode: model glitches mid-emission and the
    JSON is unterminated. Bracket-balancer salvages the valid prefix."""
    truncated = '''{"entities":[{"canonical_id":"a"},{"canonical_id":"b"}],"chunks":[{"doc_id":"x","chunk_id":"x","content":"first"},{"doc_id":"y","chunk_id":"y","content":"sec'''
    out = _parse_terminal_args(truncated)
    assert out is not None
    assert len(out.entities) == 2
    assert out.entities[0].canonical_id == "a"
    assert out.entities[1].canonical_id == "b"
    assert len(out.chunks) >= 1
    assert out.chunks[0].content == "first"


def test_parse_terminal_args_filters_non_dict_entities() -> None:
    """Third failure mode observed on Cerebras 2026-05-18: model emits
    malformed JSON fragments as bare strings inside the entities array
    (constrained-decoding partial failure inside the list). Drop the
    non-dict items so the valid neighbors survive."""
    out = _parse_terminal_args({
        "entities": [
            {"canonical_id": "github:owner/repo:pr:1", "label": "PR"},
            "{",  # malformed fragment
            {"canonical_id": "github:owner/repo:pr:2", "label": "PR"},
            '{canonical_id":...',  # mangled string
        ],
    })
    assert out is not None
    assert len(out.entities) == 2
    assert out.entities[0].canonical_id == "github:owner/repo:pr:1"
    assert out.entities[1].canonical_id == "github:owner/repo:pr:2"


def test_parse_terminal_args_lenient_doc_level_chunks() -> None:
    """Second failure mode observed on Cerebras 2026-05-18: model emits
    'doc-level chunks' where chunk_id is omitted (only doc_id present)
    and content lives under `description` or `summary` instead of
    `content`. Coercion: chunk_id ← doc_id; content ← description."""
    out = _parse_terminal_args({
        "chunks": [
            {
                "doc_id": "github:prbe-ai/prbe-backend:commit:abc",
                "description": "Refactored authentication flow.",
            },
            {
                "doc_id": "github:prbe-ai/prbe-backend:pr:42",
                "summary": "Wires up the dashboard self-host image.",
            },
            {
                # No doc_id, no chunk_id — unrecoverable, drops
                "content": "orphan content",
            },
        ],
    })
    assert out is not None
    assert len(out.chunks) == 2
    assert out.chunks[0].chunk_id == "github:prbe-ai/prbe-backend:commit:abc"
    assert out.chunks[0].content == "Refactored authentication flow."
    assert out.chunks[1].chunk_id == "github:prbe-ai/prbe-backend:pr:42"
    assert out.chunks[1].content == "Wires up the dashboard self-host image."


def test_parse_terminal_args_cerebras_real_failure_mode() -> None:
    """Reproduction of the exact failure observed live on 2026-05-18 when
    we flipped SEARCH_AGENT_INFERENCE_MODEL=cerebras/gpt-oss-120b.
    Cerebras emitted entities with input-shaped fields (title/source/url)
    and chunks missing doc_id + matched_via. Tolerant parser should
    produce a valid GathererOutput with the recoverable data."""
    cerebras_actual_payload = {
        "entities": [
            {
                "canonical_id": "github:prbe-ai/prbe-knowledge:pr:286",
                "title": "feat(id_lookup): match URL path segments",
                "source": "github",
                "url": "https://github.com/prbe-ai/prbe-knowledge/pull/286",
            }
        ],
        "chunks": [
            {
                "chunk_id": "github:prbe-ai/prbe-knowledge:pr:286:chunk:0",
                "content": "Implements the URL-path-segment match for id_lookup so PRB-17 hits rank 1.",
            },
        ],
        "gatherer_notes": {
            "confidence": "high",
            "dropped": [],
        },
    }
    out = _parse_terminal_args(cerebras_actual_payload)
    assert out is not None
    assert len(out.entities) == 1
    assert out.entities[0].canonical_id == "github:prbe-ai/prbe-knowledge:pr:286"
    # title/source/url silently dropped via extra="ignore"; label/why_relevant default to ""
    assert out.entities[0].label == ""
    assert out.entities[0].why_relevant == ""
    assert len(out.chunks) == 1
    assert out.chunks[0].doc_id == "github:prbe-ai/prbe-knowledge:pr:286"
    assert out.gatherer_notes.confidence == "high"


def test_empty_passthrough_constructs_low_confidence_dummy() -> None:
    out = _empty_passthrough("schema_violation")
    assert out.entities == []
    assert out.chunks == []
    assert out.gatherer_notes.confidence == "low"
    assert "schema_violation" in out.gatherer_notes.dropped[0].reason


# ============================================================
# Pre-fan-out compact formatter — input-token compression
# ============================================================
# The first-turn user message used to embed pre-fan-out as a JSON dump
# (~15K tokens worst-case). gpt-oss-120b is a reasoning model; that much
# cold-cache input deterministically hangs past the 90s loop timeout.
# Compact format preserves every doc_id + score + snippet (everything the
# model needs to pick its next tool call) while cutting per-hit overhead.

def _mk_hit(
    doc_id: str = "github:owner/repo:pr:42",
    score: float = 0.87,
    source_system: str = "github",
    title: str = "PR #42: add self-host docs",
    content: str = "This PR adds documentation for the self-hosting setup, including the Helm chart values and the data-plane secrets template.",
) -> dict[str, Any]:
    return {
        "channel": "vector",
        "chunk_id": "chunk-1",
        "doc_id": doc_id,
        "source_system": source_system,
        "source_url": "https://github.com/owner/repo/pull/42",
        "title": title,
        "content": content,
        "score": score,
        "created_at": "2026-05-17T12:00:00Z",
        "updated_at": "2026-05-17T12:00:00Z",
        "author_id": "richard",
    }


def test_prefanout_full_dump_includes_all_doc_fields() -> None:
    """Post-uncompress: every per-hit field reaches the LLM verbatim.
    The compact rendering used to strip chunk_id/created_at/updated_at/
    author_id/source_url and truncate content; under Cerebras the input
    throughput tax that motivated PR #307 doesn't apply, so we hand the
    model the full evidence."""
    prefanout = {"sub_queries": [{
        "query": "self-hosting features",
        "vector": [_mk_hit()],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    out = _render_prefanout_budgeted(prefanout)
    assert "github:owner/repo:pr:42" in out
    assert "self-host docs" in out
    assert "0.87" in out
    assert "github" in out
    # Pre-extension these were stripped:
    assert "richard" in out  # author_id
    assert "2026-05-17" in out  # created_at/updated_at
    # Content not truncated
    assert "documentation for the self-hosting" in out


def test_prefanout_full_dump_shows_every_hit_no_cap() -> None:
    """No per-channel display cap: every hit the retrievers returned
    must be visible to the LLM. The old top-10 cap silently hid
    non-GitHub chunks when bm25/vector's top-10 happened to all be
    GitHub — the "GitHub-only chunks" symptom."""
    n_hits = 20
    hits = [_mk_hit(doc_id=f"doc:{i}", score=1.0 - i * 0.01) for i in range(n_hits)]
    prefanout = {"sub_queries": [{
        "query": "q", "vector": hits, "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    out = _render_prefanout_budgeted(prefanout)
    for i in range(n_hits):
        assert f"doc:{i}" in out, f"hit {i}/{n_hits} not in LLM input"


def test_prefanout_full_dump_preserves_inferred_edge_why() -> None:
    """`why` is the inferred-edge moat — the rationale the agent uses to
    decide whether the edge is signal or noise. Always present, never
    truncated."""
    long_why = "PR #78 body explicitly references PR #71 as its prerequisite. " * 5
    inf_hit = {
        "channel": "inferred_edge",
        "doc_id": "github:owner/repo:pr:78",
        "source_system": "github",
        "title": "PR #78: dashboard auth flow",
        "score": 0.76,
        "edge_type": "references_pr",
        "why": long_why,
        "anchor_doc_id": "github:owner/repo:pr:71",
    }
    prefanout = {"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [], "inferred_edge": [inf_hit],
    }]}
    out = _render_prefanout_budgeted(prefanout)
    assert "references_pr" in out
    # Why is preserved in full — no 200-char truncate
    assert long_why.strip() in out


def test_prefanout_full_dump_handles_empty_prefanout() -> None:
    """Edge case: pre-fan-out returned nothing on any channel."""
    assert _render_prefanout_budgeted({"sub_queries": []}) == "(no pre-fan-out hits)"
    assert _render_prefanout_budgeted({}) == "(no pre-fan-out hits)"


def test_derive_source_system_from_doc_id_known_prefixes() -> None:
    """Doc-id prefix → SourceSystem enum value. Pre-extension every
    QueryDocumentResult was force-labelled `github` because the adapter
    had no field to read; this maps `slack:thread:T123` → `"slack"` etc."""
    cases = [
        ("github:owner/repo:pr:42", "github"),
        ("slack:thread:T123", "slack"),
        ("linear:ticket:PRB-17", "linear"),
        ("notion:doc:abc", "notion"),
        ("sentry:issue:E-1", "sentry"),
        ("code_graph:owner/repo:path/to.py", "code_graph"),
        ("wiki:page:foo", "wiki"),
    ]
    for doc_id, expected in cases:
        assert _derive_source_system_from_doc_id(doc_id) == expected


def test_derive_source_system_from_doc_id_unknown_returns_blank() -> None:
    """Unknown / unparseable inputs return empty string so the adapter
    can decide the GitHub fallback rather than us silently mislabelling."""
    assert _derive_source_system_from_doc_id(None) == ""
    assert _derive_source_system_from_doc_id("") == ""
    assert _derive_source_system_from_doc_id("nocolon") == ""
    assert _derive_source_system_from_doc_id("unknown_source:foo:bar") == ""


def test_build_prefanout_doc_meta_first_complete_wins() -> None:
    """When the same doc appears in multiple channels, first-non-blank
    write wins so the doc meta lookup is the most complete record."""
    hit_v = {
        "doc_id": "slack:thread:T123",
        "source_system": "slack",
        "title": "incident thread",
        "source_url": "https://slack.com/...",
        "created_at": "2026-05-10T00:00:00Z",
        "updated_at": "2026-05-11T00:00:00Z",
        "author_id": "user-1",
    }
    hit_b = {  # bm25 hit missing title — should not clobber the vector record
        "doc_id": "slack:thread:T123",
        "source_system": "slack",
        "title": "",
        "source_url": "",
    }
    meta = _build_prefanout_doc_meta({"sub_queries": [{
        "vector": [hit_v], "bm25": [hit_b], "graph": [], "inferred_edge": [],
    }]})
    assert meta["slack:thread:T123"]["title"] == "incident thread"
    assert meta["slack:thread:T123"]["source_url"] == "https://slack.com/..."
    assert meta["slack:thread:T123"]["author_id"] == "user-1"


def test_format_inferred_chains_dedupes_within_anchor_across_sub_queries() -> None:
    """The same `(anchor, doc_id, edge_type)` triple can appear under
    multiple sub_queries when fan-out anchors overlap — first occurrence
    wins, duplicates skip. Without dedup the agent sees the same chain
    hop N times and biases toward redundant emission."""
    anchor = "github:prbe-ai/prbe-backend:pr:72"
    hit = {
        "doc_id": "linear:org:issue:abc",
        "source_system": "linear",
        "title": "[Bug] enrichment 502s",
        "edge_type": "motivates_pr",
        "why": "ticket asks for the proxy that pr72 builds",
        "anchor_doc_id": anchor,
    }
    sibling = {
        "doc_id": "slack:T123:C456:1714.999",
        "source_system": "slack",
        "title": "discussion thread",
        "edge_type": "discussed_in",
        "why": "approach hashed out before pr72 landed",
        "anchor_doc_id": anchor,
    }
    # Two sub_queries, each carrying the same `hit` plus an extra unique
    # hit in the second sub_query.
    out = _format_inferred_chains({"sub_queries": [
        {"query": "q1", "vector": [], "bm25": [], "graph": [], "inferred_edge": [hit]},
        {"query": "q2", "vector": [], "bm25": [], "graph": [], "inferred_edge": [hit, sibling]},
    ]})
    # The duplicate linked-doc line appears exactly once.
    assert out.count("linear:org:issue:abc") == 1
    assert out.count("ticket asks for the proxy") == 1
    # The unique-to-q2 hit still renders.
    assert "slack:T123:C456:1714.999" in out
    assert "discussed_in" in out


def test_format_inferred_chains_respects_cap_with_truncation_note() -> None:
    """Worst-case 5-sub_query x 10-hit fanout would dump ~50 chain hits.
    The cap stops at `_PREFANOUT_INFERRED_CHAINS_CAP` and emits a
    "showing top N of M" footer so the agent knows the remainder is
    reachable via subgraph / fetch_doc — chain shape stays scannable
    even on hub-heavy queries."""
    from engine.retrieval.agent.loop import _PREFANOUT_INFERRED_CHAINS_CAP

    over = _PREFANOUT_INFERRED_CHAINS_CAP + 5
    hits = [
        {
            "doc_id": f"linear:issue:{i}",
            "source_system": "linear",
            "title": f"issue {i}",
            "edge_type": "motivates_pr",
            "why": f"reason {i}",
            "anchor_doc_id": "github:prbe-ai/prbe-backend:pr:72",
        }
        for i in range(over)
    ]
    out = _format_inferred_chains({"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [], "inferred_edge": hits,
    }]})
    # First N rendered, rest dropped.
    for i in range(_PREFANOUT_INFERRED_CHAINS_CAP):
        assert f"linear:issue:{i}" in out
    for i in range(_PREFANOUT_INFERRED_CHAINS_CAP, over):
        assert f"linear:issue:{i}" not in out
    # Truncation note tells the agent what to do next.
    assert (
        f"showing top {_PREFANOUT_INFERRED_CHAINS_CAP} of {over}" in out
    )
    assert "subgraph(anchor)" in out or "with_inferred_edges=true" in out


def test_format_inferred_chains_treats_empty_anchor_as_missing() -> None:
    """`anchor_doc_id=""` is falsy — same handling as missing key:
    the hit is silently skipped rather than grouped under a phantom
    empty-string anchor."""
    hits = [
        {"doc_id": "d1", "edge_type": "e1", "why": "w1", "anchor_doc_id": ""},
        {"doc_id": "d2", "edge_type": "e2", "why": "w2", "anchor_doc_id": "a-real"},
    ]
    out = _format_inferred_chains({"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [], "inferred_edge": hits,
    }]})
    assert "anchor: a-real" in out
    assert "d2" in out
    assert "d1" not in out
    assert "anchor: \n" not in out  # no phantom empty-anchor block


# ============================================================
# Inferred-chains re-grouping — chain shape for why-queries
# ============================================================
# The inferred_edge channel returns flat hits; `_format_inferred_chains`
# regroups them by `anchor_doc_id` so the agent sees the chain (one
# anchor → many linked docs with edge `why`s). This is the structural
# view why-chain queries need.

def test_format_inferred_chains_groups_by_anchor() -> None:
    """Two hits sharing an anchor render under a single `anchor:` line
    in call-order; a third hit with a different anchor opens its own
    block. Anchors get listed in first-seen order so the agent reads
    them top-down."""
    anchor_a = "github:prbe-ai/prbe-backend:pr:72"
    anchor_b = "github:prbe-ai/prbe-knowledge:pr:316"
    hits = [
        {
            "doc_id": "linear:org:issue:abc",
            "source_system": "linear",
            "title": "[Bug] enrichment 502s",
            "edge_type": "motivates_pr",
            "why": "ticket asks for the proxy that pr72 builds",
            "anchor_doc_id": anchor_a,
        },
        {
            "doc_id": "github:prbe-ai/prbe-backend:issue:76",
            "source_system": "github",
            "title": "Issue 76",
            "edge_type": "addresses",
            "why": "pr72 implements the github webhook fan-out called for by issue 76",
            "anchor_doc_id": anchor_a,
        },
        {
            "doc_id": "search-traces/2026-05-17/q-1779050066661.json.gz",
            "source_system": "trace",
            "title": "loop_timeout trace",
            "edge_type": "triggered_by",
            "why": "the pr316 soft-cap fix was triggered by this loop_timeout trace",
            "anchor_doc_id": anchor_b,
        },
    ]
    out = _format_inferred_chains({"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [], "inferred_edge": hits,
    }]})
    # Anchors render in first-seen order.
    pos_a = out.find(f"anchor: {anchor_a}")
    pos_b = out.find(f"anchor: {anchor_b}")
    assert 0 <= pos_a < pos_b, f"expected anchor_a before anchor_b — got {pos_a=} {pos_b=}"
    # Each anchor block contains its linked docs + edge + why.
    assert "linear:org:issue:abc" in out
    assert "edge=motivates_pr" in out
    assert "why: ticket asks for the proxy" in out
    assert "github:prbe-ai/prbe-backend:issue:76" in out
    assert "edge=addresses" in out
    assert "search-traces/2026-05-17" in out
    assert "edge=triggered_by" in out


def test_format_inferred_chains_returns_empty_with_no_inferred_hits() -> None:
    """No inferred-edge hits anywhere → empty string (NOT a header with
    nothing under it). `_build_user_message` checks this empty/non-empty
    to decide whether to render the `<inferred_chains>` section at all."""
    prefanout = {"sub_queries": [{
        "query": "q", "vector": [{"doc_id": "x"}], "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    assert _format_inferred_chains(prefanout) == ""


def test_format_inferred_chains_skips_hits_with_no_anchor() -> None:
    """Defensive: an inferred-edge hit missing `anchor_doc_id` can't be
    placed in a chain. Skip silently rather than crashing or grouping
    under a phantom 'None' anchor."""
    hits = [
        {"doc_id": "d1", "edge_type": "e1", "why": "w1"},  # no anchor
        {"doc_id": "d2", "edge_type": "e2", "why": "w2", "anchor_doc_id": "a-real"},
    ]
    out = _format_inferred_chains({"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [], "inferred_edge": hits,
    }]})
    assert "anchor: a-real" in out
    assert "d2" in out
    assert "d1" not in out  # the anchor-less hit was skipped
    assert "None" not in out
    assert "anchor: ?" not in out


# ============================================================
# _build_user_message — section composition
# ============================================================
# The user message is assembled from grounding + connected_sources +
# channel_results + inferred_chains + query. The chain section MUST
# only render when there are inferred-edge hits, otherwise it pollutes
# the prompt for all the vector/bm25-only queries.

def test_build_user_message_includes_inferred_chains_when_present() -> None:
    """When inferred-edge hits exist the `<inferred_chains>` section
    renders AFTER `<channel_results>` with the regrouped view + the
    instruction text. This is the structural cue for why-chain queries."""
    from engine.retrieval.grounding import GroundingBundle

    inf_hit = {
        "doc_id": "linear:org:issue:abc",
        "source_system": "linear",
        "title": "[Bug] enrichment 502s",
        "edge_type": "motivates_pr",
        "why": "ticket asks for the proxy that pr72 builds",
        "anchor_doc_id": "github:prbe-ai/prbe-backend:pr:72",
    }
    prefanout = {"sub_queries": [{
        "query": "q", "vector": [], "bm25": [], "graph": [], "inferred_edge": [inf_hit],
    }]}
    out = _build_user_message("why was PR 72 created", GroundingBundle(), prefanout)
    assert "<inferred_chains>" in out
    assert "anchor: github:prbe-ai/prbe-backend:pr:72" in out
    # Order: channel_results before inferred_chains, both before query.
    pos_chan = out.find("<channel_results>")
    pos_chain = out.find("<inferred_chains>")
    pos_query = out.find("<query>")
    assert 0 <= pos_chan < pos_chain < pos_query


def test_build_user_message_omits_inferred_chains_when_no_inferred_hits() -> None:
    """Vector/bm25-only queries don't get the empty chain section —
    that'd waste tokens + confuse the agent into looking for a chain
    that isn't there. The `<channel_results>` intro prose references
    `<inferred_chains>` by name (it tells the agent where inferred-edge
    data lives WHEN present), so a bare substring search isn't enough;
    we check the actual section opener appears on a fresh line."""
    from engine.retrieval.grounding import GroundingBundle

    prefanout = {"sub_queries": [{
        "query": "q",
        "vector": [_mk_hit()],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    out = _build_user_message("self-hosting features", GroundingBundle(), prefanout)
    assert "<channel_results>" in out
    assert "\n<inferred_chains>\n" not in out
    assert "</inferred_chains>" not in out


def test_build_user_message_includes_search_options_when_nondefault() -> None:
    """When the extractor flagged `sort=recency` AND/OR resolved author_ids,
    the `<search_options>` block renders between `<connected_sources>` and
    `<channel_results>`. The block tells the agent the channel ordering is
    authoritative — without it the agent will re-rank by intuition and
    scatter the result set."""
    from engine.retrieval.grounding import GroundingBundle

    out = _build_user_message(
        "what did mahit do last?",
        GroundingBundle(),
        prefanout=None,
        options=SearchOptions(sort="recency"),
        author_ids=["mahit@prbe.ai"],
    )
    assert "<search_options>" in out
    assert "sort=recency" in out
    assert "author_ids" in out and "mahit@prbe.ai" in out
    pos_sources = out.find("<connected_sources>")
    pos_options = out.find("<search_options>")
    pos_query = out.find("<query>")
    assert 0 <= pos_sources < pos_options < pos_query


def test_build_user_message_omits_sort_when_only_author_filter_applied() -> None:
    """A person-mentioning relevance-sorted query (sort=relevance,
    author_ids=[X]) used to render `sort=relevance` into the prompt —
    a string that never appeared pre-PR. That string would break
    prompt-cache prefix stability for every person-anchored query that
    wasn't asking for recency, blowing up the cache miss rate. The block
    must show ONLY `author_ids=[...]` in that case."""
    from engine.retrieval.grounding import GroundingBundle

    out = _build_user_message(
        "what does mahit work on",
        GroundingBundle(),
        prefanout=None,
        options=SearchOptions(sort="relevance"),
        author_ids=["mahit@prbe.ai"],
    )
    assert "<search_options>" in out
    assert "author_ids" in out
    assert "mahit@prbe.ai" in out
    # The critical assertion: don't emit the no-op `sort=relevance` token.
    assert "sort=relevance" not in out


def test_build_user_message_omits_search_options_for_default_query() -> None:
    """sort=relevance + no author filter → the tag is suppressed. This
    keeps the prompt-cache prefix bit-identical to pre-PR for the 90%
    case (non-deterministic queries), so cache hit rates don't drop."""
    from engine.retrieval.grounding import GroundingBundle

    out = _build_user_message(
        "how does auth work",
        GroundingBundle(),
        prefanout=None,
        options=SearchOptions(),  # defaults
        author_ids=[],
    )
    assert "<search_options>" not in out


def test_build_user_message_omits_inferred_chains_when_no_prefanout() -> None:
    """No pre-fan-out at all (LLM-extraction-only path) → neither
    channel_results nor inferred_chains render. Same opener-only check
    as the no-inferred-hits sibling: the prose intro doesn't render
    either since channel_results itself is skipped."""
    from engine.retrieval.grounding import GroundingBundle

    out = _build_user_message("q", GroundingBundle(), None)
    assert "<channel_results>" not in out
    assert "\n<inferred_chains>\n" not in out
    assert "</inferred_chains>" not in out


# ============================================================
# Loop integration (mocked everything)
# ============================================================

@pytest.mark.asyncio
async def test_extracted_search_options_flow_into_execute_search(
    monkeypatch: pytest.MonkeyPatch,
    fake_request: SimpleNamespace,
) -> None:
    """End-to-end loop wiring: extractor returns sort=recency + a person
    entity → run_gatherer must pass sort_by="recency" AND the resolved
    author_ids to execute_search. Regression guard for the
    "what did mahit do last" optimization — if the extracted options
    don't reach execute_search, the channels stay relevance-sorted and
    the deterministic queries scatter again."""
    req = QueryRequest(query="what did mahit do last?", customer_id="cust-1", top_k=5)

    monkeypatch.setattr(
        "engine.retrieval.agent.loop._build_bundle_with_token_fallback",
        AsyncMock(return_value=GroundingBundle()),
    )
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.extract_entities_with_llm",
        AsyncMock(return_value=EntityExtraction(
            entities=[
                ExtractedEntity(
                    entity_type="person",
                    canonical_id="mahit@prbe.ai",
                    display_name="Mahit",
                    confidence=1.0,
                ),
            ],
            search_options=SearchOptions(sort="recency"),
        )),
    )
    monkeypatch.setattr(
        "engine.retrieval.agent.loop._resolve_person_author_ids",
        AsyncMock(return_value=["mahit@prbe.ai", "mahitoburrito"]),
    )
    captured = AsyncMock(return_value={"sub_queries": [{
        "query": "what did mahit do last?",
        "grounded_entities": [],
        "vector": [{"doc_id": "stub:0", "score": 0.5,
                    "source_system": "github", "title": "stub",
                    "content": "stub"}],
        "bm25": [], "graph": [], "inferred_edge": [],
    }]})
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.execute_search",
        captured,
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(_final_emission_args(chunks=1, confidence="high"))],
        )),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert captured.await_count == 1
    kwargs = captured.await_args.kwargs
    assert kwargs.get("sort_by") == "recency"
    assert kwargs.get("author_ids") == ["mahit@prbe.ai", "mahitoburrito"]


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_enabled", [True, False])
async def test_terminal_on_turn_1_is_happy_path(
    fake_request: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    gateway_enabled: bool,
) -> None:
    """The model calls emit_gatherer_output on turn 1 → loop ends, args
    become the final GathererOutput, telemetry recorded."""
    req = QueryRequest(query="what is PRB-17", customer_id="cust-1", top_k=5)
    if gateway_enabled:
        monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.example")
    else:
        monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(_final_emission_args(chunks=2, confidence="high"))],
            cached_tokens=80,
        )),
    ) as mock_acomp:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # 2 emitted chunks + 1 recall-floor backfill doc from the stubbed
    # pre-fan-out pool (stub:0) the gatherer didn't emit.
    assert resp.total_candidates == 3
    assert resp.gatherer_notes is not None
    assert resp.gatherer_notes["confidence"] == "high"
    # Telemetry
    assert fake_request.state.gatherer_status == "ok"
    assert fake_request.state.confidence == "high"
    # The model emitted the terminal — that counts as 0 retrieval calls.
    assert fake_request.state.tool_calls_count == 0
    assert fake_request.state.cache_hit_rate == pytest.approx(0.8)
    # Tool surface sent to model
    call_kwargs = mock_acomp.call_args.kwargs
    assert call_kwargs.get("tool_choice") == "required"
    assert call_kwargs.get("custom_llm_provider") == "openai"
    assert call_kwargs.get("timeout") == SEARCH_AGENT_GATHERER_TIMEOUT_SECONDS
    if gateway_enabled:
        assert call_kwargs.get("max_retries") == 0
    else:
        assert "max_retries" not in call_kwargs
    assert "extra_headers" in call_kwargs
    assert "x-session-affinity" in call_kwargs["extra_headers"]


@pytest.mark.asyncio
async def test_run_gatherer_forwards_top_k_related_to_adapter(
    fake_request: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    req = QueryRequest(
        query="what is PRB-17",
        customer_id="cust-1",
        top_k=5,
        top_k_related=0,
    )
    sentinel = object()
    mock_adapter = AsyncMock(return_value=sentinel)
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.to_query_response",
        mock_adapter,
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(_final_emission_args())],
        )),
    ):
        response = await run_gatherer(
            req,
            customer_id="cust-1",
            request=fake_request,
        )

    assert response is sentinel
    assert mock_adapter.await_args.kwargs["top_k_related"] == 0


@pytest.mark.asyncio
async def test_exploration_then_terminal(
    fake_request: SimpleNamespace,
) -> None:
    """Turn 1: model calls search. Turn 2: model calls terminal. Loop ends."""
    req = QueryRequest(query="auth refactor", customer_id="cust-1", top_k=5)

    turn_1 = _mk_resp(tool_calls=[{
        "id": "s1",
        "name": "search",
        "arguments": {"queries": ["auth refactor design doc"]},
    }])
    turn_2 = _mk_resp(tool_calls=[_terminal_call(_final_emission_args(chunks=3))])

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ) as mock_dispatch:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # 3 emitted chunks + 1 recall-floor backfill doc from the pre-fan-out pool.
    assert resp.total_candidates == 4
    assert fake_request.state.tool_calls_count == 1  # one search call
    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args.kwargs["tool_name"] == "search"


@pytest.mark.asyncio
async def test_no_tool_calls_returns_schema_violation(
    fake_request: SimpleNamespace,
) -> None:
    """tool_choice=required SHOULD force a tool call. If the provider
    quirks and returns content-only, harness logs + returns
    schema_violation (no prose-retry path)."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    bad_turn = _mk_resp(content="I wasn't able to find anything.", tool_calls=[])

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=bad_turn),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.gatherer_status == "schema_violation"
    # Status stays schema_violation, but the recall-floor backfill now
    # surfaces the pre-fan-out pool doc the gatherer never emitted
    # (graceful degradation — see loop._backfill_recall_floor).
    assert resp.total_candidates == 1
    assert resp.gatherer_notes["confidence"] == "low"


@pytest.mark.asyncio
async def test_wrong_shape_terminal_args_recovers_via_recall_floor(
    fake_request: SimpleNamespace,
) -> None:
    """Model calls emit_gatherer_output with args that don't match the
    GathererOutput schema (e.g. Cerebras gpt-oss-120b emitting input-
    shaped fields like title/source/url). Tolerant parser absorbs the
    drift: parses as empty GathererOutput, status='ok'. The gatherer
    itself emitted nothing, but the recall-floor backfill then surfaces
    the pre-fan-out pool docs it never picked up, so the user gets
    candidates instead of an empty response (graceful degradation —
    see loop._backfill_recall_floor). The loop trace still records that
    the model DID emit, rather than treating it as a parse failure."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    bad_terminal = _terminal_call({"completely": "wrong shape"})

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(tool_calls=[bad_terminal])),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert fake_request.state.gatherer_status == "ok"
    # Gatherer emitted 0 chunks; recall-floor backfill surfaces the 1
    # stubbed pre-fan-out pool doc.
    assert resp.total_candidates == 1


@pytest.mark.asyncio
async def test_bad_confidence_literal_clamps_to_medium(
    fake_request: SimpleNamespace,
) -> None:
    """Model fabricates a `gatherer_notes.confidence` value outside the
    allowed Literal set ({"high","medium","low"}). The coercer clamps
    it to the schema default ("medium") so the emission parses and the
    user still gets the chunks — no extra LLM round-trip, no
    schema_violation. Replaces the retry path landed in PR #375.

    Covers Pattern A of the 2026-05-20 nightly digest: 12/16
    schema_violation traces were turn-1 emit_gatherer_output where the
    model nailed the SHAPE but drifted a single Literal value."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    bad_args = _final_emission_args(chunks=3)
    bad_args["gatherer_notes"]["confidence"] = "definitely_unknown_label"
    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(bad_args)],
        )),
    ) as mock_acomp:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)
    assert fake_request.state.gatherer_status == "ok"
    # 3 emitted + 1 recall-floor backfill doc from the pre-fan-out pool.
    assert resp.total_candidates == 4
    assert resp.gatherer_notes["confidence"] == "medium"
    # No retry: single LLM round-trip per the new design.
    assert mock_acomp.await_count == 1


@pytest.mark.asyncio
async def test_explicit_null_confidence_clamps_to_medium(
    fake_request: SimpleNamespace,
) -> None:
    """Model emits explicit JSON null for `confidence` (rare but real —
    Pydantic Literal rejects None). The coercer clamps to "medium"
    rather than letting the whole emission fail."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    args = _final_emission_args(chunks=1)
    args["gatherer_notes"]["confidence"] = None
    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(args)],
        )),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)
    assert fake_request.state.gatherer_status == "ok"
    # 1 emitted + 1 recall-floor backfill doc from the pre-fan-out pool.
    assert resp.total_candidates == 2
    assert resp.gatherer_notes["confidence"] == "medium"


@pytest.mark.asyncio
async def test_non_string_matched_via_member_does_not_crash(
    fake_request: SimpleNamespace,
) -> None:
    """Model emits a non-string member inside `matched_via` (dict from
    a constrained-decoding partial failure). The filter restricts to
    strings before the `in` check so it can't raise TypeError on an
    unhashable member."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    args = _final_emission_args(chunks=1)
    args["chunks"][0]["matched_via"] = ["vector", {"broken": "shape"}, 123]
    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(args)],
        )),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)
    assert fake_request.state.gatherer_status == "ok"
    # 1 emitted + 1 recall-floor backfill doc from the pre-fan-out pool.
    assert resp.total_candidates == 2


@pytest.mark.asyncio
async def test_bad_matched_via_values_are_filtered(
    fake_request: SimpleNamespace,
) -> None:
    """Model emits a chunk with `matched_via` containing one valid
    channel and one fabricated label ("telepathy"). The coercer drops
    the unknown member and keeps the chunk — the fabricated label was
    only telemetry, not part of the user-facing answer.

    If the entire list is invalid, it collapses to the empty-list
    default and the chunk still survives."""
    req = QueryRequest(query="x", customer_id="cust-1", top_k=5)
    args = _final_emission_args(chunks=2)
    args["chunks"][0]["matched_via"] = ["vector", "telepathy"]
    args["chunks"][1]["matched_via"] = ["fabricated_only"]
    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(return_value=_mk_resp(
            tool_calls=[_terminal_call(args)],
        )),
    ) as mock_acomp:
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)
    assert fake_request.state.gatherer_status == "ok"
    # 2 emitted + 1 recall-floor backfill doc from the pre-fan-out pool.
    assert resp.total_candidates == 3
    assert mock_acomp.await_count == 1


@pytest.mark.asyncio
async def test_unknown_llm_error_raises_503(
    fake_request: SimpleNamespace,
) -> None:
    """An untyped/unknown provider error stays fatal despite pre-fan-out."""
    from engine.shared.llm import LLMError
    req = QueryRequest(query="boom", customer_id="cust-1", top_k=5)

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=LLMError("fireworks down")),
    ), pytest.raises(HTTPException) as exc_info:
        await run_gatherer(req, customer_id="cust-1", request=fake_request)
    assert exc_info.value.status_code == 503
    assert fake_request.state.full_failure is True
    assert fake_request.state.gatherer_status == "fatal_provider_error"
    assert fake_request.state.tool_calls_count == 0
    assert fake_request.state.need_deeper_extensions == 0
    assert fake_request.state.confidence is None
    assert fake_request.state.dropped_count is None
    assert fake_request.state.cache_hit_rate is None
    assert fake_request.state.failure_recovered is False


@pytest.mark.asyncio
async def test_fatal_error_after_exploration_stashes_query_trace_summary(
    fake_request: SimpleNamespace,
) -> None:
    """Fatal 503s still expose partial loop metrics to QueryTrace middleware."""
    req = QueryRequest(query="fatal provider", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(
        tool_calls=[{
            "id": "s1",
            "name": "search",
            "arguments": {"queries": ["fatal provider evidence"]},
        }],
        prompt_tokens=100,
        cached_tokens=75,
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, LLMError("unknown provider failure")]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ), pytest.raises(HTTPException) as exc_info:
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert exc_info.value.status_code == 503
    assert fake_request.state.full_failure is True
    assert fake_request.state.gatherer_status == "fatal_provider_error"
    assert fake_request.state.tool_calls_count == 1
    assert fake_request.state.need_deeper_extensions == 0
    assert fake_request.state.confidence is None
    assert fake_request.state.dropped_count is None
    assert fake_request.state.cache_hit_rate == pytest.approx(0.75)
    assert fake_request.state.failure_recovered is False
    assert fake_request.state.search_agent_status == "fatal_provider_error"
    assert fake_request.state.search_agent_gathered is None
    assert len(
        fake_request.state.search_agent_loop_state.failed_turn_latencies_ms
    ) == 1
    state = fake_request.state.search_agent_loop_state
    timing = fake_request.state.search_agent_timing
    assert timing["agent_failed_llm_ms"] == pytest.approx(
        sum(state.failed_turn_latencies_ms)
    )
    assert timing["agent_loop_ms"] == pytest.approx(
        sum(state.turn_latencies_ms) + sum(state.failed_turn_latencies_ms)
    )
    assert timing["agent_tools_ms"] == pytest.approx(
        sum(state.tool_latencies_ms)
    )
    assert timing["agent_ms"] >= timing["agent_loop_ms"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [408, 429, 500, 503])
async def test_transient_http_error_degrades_to_citable_prefanout(
    fake_request: SimpleNamespace,
    status_code: int,
) -> None:
    """Exhausted transient provider chain returns the already-fetched docs."""
    req = QueryRequest(query="provider tail", customer_id="cust-1", top_k=5)
    error = LLMError(
        "gateway provider chain exhausted",
        status_code=status_code,
        provider="fireworks_ai",
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=error),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 1
    assert resp.results[0].doc_id == "stub:0"
    assert resp.results[0].chunks[0].content == "stub"
    assert resp.gatherer_notes["confidence"] == "low"
    assert fake_request.state.gatherer_status == "provider_error_prefanout_fallback"
    assert fake_request.state.failure_recovered is True
    assert getattr(fake_request.state, "full_failure", False) is False
    assert fake_request.state.search_agent_status == "provider_error_prefanout_fallback"
    assert fake_request.state.search_agent_gathered.chunks[0].doc_id == "stub:0"


@pytest.mark.asyncio
async def test_transient_error_after_exploration_preserves_loop_ledger_and_latency(
    fake_request: SimpleNamespace,
) -> None:
    """A failed second turn keeps completed work and records outage latency."""
    req = QueryRequest(query="provider tail", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(tool_calls=[{
        "id": "s1",
        "name": "search",
        "arguments": {"queries": ["provider tail evidence"]},
    }])
    provider_error = LLMError(
        "gateway provider chain exhausted",
        status_code=503,
        provider="fireworks_ai",
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, provider_error]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.gatherer_notes["confidence"] == "low"
    assert resp.gatherer_notes["turns_used"] == 1
    assert resp.gatherer_notes["tools_called"] == ["search"]
    state = fake_request.state.search_agent_loop_state
    assert state.turn_count == 1
    assert state.tools_fired == ["search"]
    assert len(state.turn_latencies_ms) == 1
    assert len(state.failed_turn_latencies_ms) == 1
    timing = fake_request.state.search_agent_timing
    assert timing["agent_failed_llm_ms"] == pytest.approx(
        sum(state.failed_turn_latencies_ms)
    )
    assert timing["agent_loop_ms"] == pytest.approx(
        sum(state.turn_latencies_ms) + sum(state.failed_turn_latencies_ms)
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("cause", [TimeoutError("late"), ConnectionError("reset")])
async def test_typed_transport_error_degrades_without_http_status(
    fake_request: SimpleNamespace,
    cause: BaseException,
) -> None:
    """Typed timeout/connection causes are transient without message matching."""
    req = QueryRequest(query="provider transport", customer_id="cust-1", top_k=5)
    error = LLMError("opaque transport failure", provider="cerebras")
    error.__cause__ = cause

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=error),
    ):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 1
    assert fake_request.state.gatherer_status == "provider_error_prefanout_fallback"
    assert fake_request.state.failure_recovered is True


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
async def test_non_transient_http_error_remains_fatal(
    fake_request: SimpleNamespace,
    status_code: int,
) -> None:
    """Auth/config/validation errors must never be hidden by pre-fan-out."""
    req = QueryRequest(query="bad configuration", customer_id="cust-1", top_k=5)
    error = LLMError("request rejected", status_code=status_code)
    if status_code == 401:
        # A hard status is authoritative even when the transport attaches a
        # connection-ish nested cause after receiving the rejection.
        error.__cause__ = ConnectionError("connection closed after response")

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=error),
    ), pytest.raises(HTTPException) as exc_info:
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert exc_info.value.status_code == 503
    assert fake_request.state.full_failure is True


@pytest.mark.asyncio
async def test_transient_error_without_citable_prefanout_remains_fatal(
    fake_request: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hit lacking content cannot support a citation, so preserve the 503."""
    req = QueryRequest(query="uncitable evidence", customer_id="cust-1", top_k=5)
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.execute_search",
        AsyncMock(return_value={
            "sub_queries": [{
                "query": "uncitable evidence",
                "vector": [{"doc_id": "github:doc:1", "content": "   "}],
                "bm25": [],
                "graph": [],
                "inferred_edge": [],
            }]
        }),
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=LLMError("busy", status_code=429)),
    ), pytest.raises(HTTPException) as exc_info:
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert exc_info.value.status_code == 503
    assert fake_request.state.full_failure is True


@pytest.mark.asyncio
async def test_no_llm_configured_short_circuits_to_empty(
    fake_request: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When no LLM provider is configured, gatherer returns empty 200
    instead of 503 — mirrors PR #282's _call_haiku graceful no-op."""
    req = QueryRequest(query="anything", customer_id="cust-1", top_k=5)
    monkeypatch.setattr(
        "engine.retrieval.agent.loop._no_llm_configured", lambda: True
    )
    boom = AsyncMock(side_effect=AssertionError("acompletion should NOT be called"))

    with patch("engine.retrieval.agent.loop.acompletion", new=boom):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 0
    assert resp.gatherer_notes["confidence"] == "low"
    assert fake_request.state.gatherer_status == "no_llm_configured"
    assert fake_request.state.tool_calls_count == 0
    assert fake_request.state.failure_recovered is True
    boom.assert_not_called()


@pytest.mark.asyncio
async def test_zero_recall_short_circuits_to_empty(
    fake_request: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every pre-fan-out channel (vector + BM25 + graph + inferred)
    returns 0 hits, the LLM has nothing to curate. Skip the loop, return
    empty — saves 67-90s on truly hopeless queries that would otherwise
    burn the wall-clock oscillating search/fetch_doc."""
    req = QueryRequest(query="zxyq-no-matches", customer_id="cust-1", top_k=5)
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.execute_search",
        AsyncMock(return_value={"sub_queries": []}),
    )
    boom = AsyncMock(side_effect=AssertionError("acompletion should NOT be called"))

    with patch("engine.retrieval.agent.loop.acompletion", new=boom):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 0
    assert fake_request.state.gatherer_status == "zero_recall_short_circuit"
    assert fake_request.state.tool_calls_count == 0
    boom.assert_not_called()


@pytest.mark.asyncio
async def test_zero_recall_short_circuits_even_with_extracted_entities(
    fake_request: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`entity_dicts` is fed into execute_search above as `entity_ids`,
    so the graph + inferred_edge channels already exercised anchor-driven
    exploration with these entities. A 0-hit prefanout result therefore
    means even entity-anchored paths found nothing; the loop has nothing
    new to try. Regression guard: prior gate required `not entity_dicts`,
    which kept "blue yeti microphones"-style queries in the 90s wall-clock
    death loop whenever extraction surfaced any entity at all."""
    req = QueryRequest(query="blue yeti microphones", customer_id="cust-1", top_k=5)
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.execute_search",
        AsyncMock(return_value={"sub_queries": []}),
    )
    monkeypatch.setattr(
        "engine.retrieval.agent.loop.extract_entities_with_llm",
        AsyncMock(return_value=EntityExtraction(entities=[
            ExtractedEntity(
                entity_type="service",
                canonical_id="blue_yeti",
                display_name="Blue Yeti",
                confidence=0.5,
            ),
        ])),
    )
    boom = AsyncMock(side_effect=AssertionError("acompletion should NOT be called"))

    with patch("engine.retrieval.agent.loop.acompletion", new=boom):
        resp = await run_gatherer(req, customer_id="cust-1", request=fake_request)

    assert resp.total_candidates == 0
    assert fake_request.state.gatherer_status == "zero_recall_short_circuit"
    assert fake_request.state.tool_calls_count == 0
    boom.assert_not_called()


@pytest.mark.asyncio
async def test_per_stage_latency_recorded_on_state(
    fake_request: SimpleNamespace,
) -> None:
    """LoopState accumulates per-turn LLM latencies + per-tool latencies."""
    req = QueryRequest(query="q", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(tool_calls=[{
        "id": "s1", "name": "search", "arguments": {"queries": ["q'"]},
    }])
    turn_2 = _mk_resp(tool_calls=[_terminal_call()])

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # cache_hit_rate is averaged across turns
    assert fake_request.state.cache_hit_rate is not None


@pytest.mark.asyncio
async def test_reasoning_content_captured_per_turn(
    fake_request: SimpleNamespace,
) -> None:
    """The gpt-oss harmony `analysis` block (surfaced by LiteLLM as
    `message.reasoning_content`) lands on `state.reasoning_per_turn`
    parallel to `turn_latencies_ms`. Without this capture the
    agent's "why" trail is lost because the OpenAI chat-completion
    round-trip only echoes role/content/tool_calls — not reasoning."""
    req = QueryRequest(query="why was PR 71 made", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(
        tool_calls=[{"id": "s1", "name": "search", "arguments": {"queries": ["q1"]}}],
        reasoning_content=(
            "User asks about PR #71 motivation. I'll start with the "
            "vector channel anchored on PR-71 ID."
        ),
    )
    turn_2 = _mk_resp(
        tool_calls=[_terminal_call()],
        reasoning_content=None,  # provider may emit reasoning on some turns and not others
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    # PR 1's stash exposes the LoopState ref on request.state.
    loop_state = fake_request.state.search_agent_loop_state
    assert loop_state is not None
    assert len(loop_state.reasoning_per_turn) == 2
    assert loop_state.reasoning_per_turn[0] is not None
    assert "PR #71" in loop_state.reasoning_per_turn[0]
    assert loop_state.reasoning_per_turn[1] is None  # provider didn't emit on turn 2


@pytest.mark.asyncio
async def test_reasoning_per_turn_starts_empty_and_grows(
    fake_request: SimpleNamespace,
) -> None:
    """No reasoning emitted = list of None entries (one per turn), NOT
    a missing key. The analyzer relies on len(reasoning_per_turn) ==
    turn_count for per-turn correlation."""
    req = QueryRequest(query="q", customer_id="cust-1", top_k=5)
    # Single turn — terminal immediately, no reasoning.
    turn_1 = _mk_resp(tool_calls=[_terminal_call()], reasoning_content=None)

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    loop_state = fake_request.state.search_agent_loop_state
    assert loop_state is not None
    assert len(loop_state.reasoning_per_turn) == loop_state.turn_count
    assert loop_state.reasoning_per_turn == [None]


# ============================================================
# Determinism telemetry: seed sent + system_fingerprint captured
# ============================================================


def test_seed_for_query_is_deterministic_and_query_scoped() -> None:
    """Same (customer_id, query) → same seed. Different query → different seed.
    Range fits 32-bit unsigned (0 to 2^32-1) for cross-provider safety."""
    a = _seed_for_query("cust-1", "what shipped this week")
    b = _seed_for_query("cust-1", "what shipped this week")
    c = _seed_for_query("cust-1", "what shipped last week")
    d = _seed_for_query("cust-2", "what shipped this week")
    assert a == b
    assert a != c
    assert a != d
    for s in (a, c, d):
        assert 0 <= s < 2**32


@pytest.mark.asyncio
async def test_seed_sent_and_system_fingerprint_captured_per_turn(
    fake_request: SimpleNamespace,
) -> None:
    """`seed` is forwarded to acompletion (same value all turns) and each
    response's `system_fingerprint` lands on state. The pair is the
    only on-rails way to detect Cerebras backend drift breaking
    reproducibility — live-traced 2026-05-19."""
    req = QueryRequest(query="why did multi-granola happen", customer_id="cust-7", top_k=5)
    turn_1 = _mk_resp(
        tool_calls=[{"id": "s1", "name": "search", "arguments": {"queries": ["q1"]}}],
        system_fingerprint="fp_alpha",
    )
    turn_2 = _mk_resp(
        tool_calls=[_terminal_call()],
        system_fingerprint="fp_beta",  # backend rolled between turns
    )

    fake_acompletion = AsyncMock(side_effect=[turn_1, turn_2])
    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=fake_acompletion,
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-7", request=fake_request)

    loop_state = fake_request.state.search_agent_loop_state
    assert loop_state is not None
    # Same seed sent on every turn — derived from (customer_id, query).
    expected_seed = _seed_for_query("cust-7", "why did multi-granola happen")
    assert loop_state.seed == expected_seed
    for call in fake_acompletion.await_args_list:
        assert call.kwargs["seed"] == expected_seed
    # Fingerprint stored per turn, in order.
    assert loop_state.system_fingerprints_per_turn == ["fp_alpha", "fp_beta"]


@pytest.mark.asyncio
async def test_per_turn_telemetry_cardinality_preserved_when_usage_missing(
    fake_request: SimpleNamespace,
) -> None:
    """Provider may omit `usage` on some responses (LiteLLM passes
    through whatever the gateway returned). cache_hit_rates must still
    grow per-turn so analyzers joining by turn index don't misalign
    cache vs fingerprint vs reasoning. None slots preserve cardinality."""
    req = QueryRequest(query="q", customer_id="cust-1", top_k=5)

    # Turn 1: usage present (cache hit measurable). Turn 2: simulate
    # missing usage by patching the helper to return None.
    turn_1 = _mk_resp(
        tool_calls=[{"id": "s1", "name": "search", "arguments": {"queries": ["q1"]}}],
        prompt_tokens=100,
        cached_tokens=80,
        system_fingerprint="fp_a",
    )
    turn_2 = _mk_resp(
        tool_calls=[_terminal_call()],
        system_fingerprint="fp_b",
    )

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1, turn_2]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ), patch(
        "engine.retrieval.agent.loop._extract_cache_hit_rate",
        side_effect=[0.8, None],  # turn 2's usage was unparseable
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    loop_state = fake_request.state.search_agent_loop_state
    assert loop_state is not None
    # All per-turn lists same length as turn_count.
    assert loop_state.turn_count == 2
    assert len(loop_state.cache_hit_rates) == 2
    assert len(loop_state.system_fingerprints_per_turn) == 2
    assert len(loop_state.turn_latencies_ms) == 2
    assert len(loop_state.reasoning_per_turn) == 2
    # None preserves the slot for turn 2.
    assert loop_state.cache_hit_rates == [0.8, None]
    # Aggregated mean filters None → 0.8.
    assert fake_request.state.cache_hit_rate == pytest.approx(0.8)


@pytest.mark.asyncio
async def test_system_fingerprint_none_when_provider_omits(
    fake_request: SimpleNamespace,
) -> None:
    """When the provider doesn't return system_fingerprint, the slot is
    None — not missing — so len(system_fingerprints_per_turn) ==
    turn_count for the analyzer's per-turn correlation."""
    req = QueryRequest(query="q", customer_id="cust-1", top_k=5)
    turn_1 = _mk_resp(tool_calls=[_terminal_call()], system_fingerprint=None)

    with patch(
        "engine.retrieval.agent.loop.acompletion",
        new=AsyncMock(side_effect=[turn_1]),
    ), patch(
        "engine.retrieval.agent.loop.dispatch_tool_call",
        new=AsyncMock(return_value={"sub_queries": []}),
    ):
        await run_gatherer(req, customer_id="cust-1", request=fake_request)

    loop_state = fake_request.state.search_agent_loop_state
    assert loop_state is not None
    assert loop_state.system_fingerprints_per_turn == [None]
    assert len(loop_state.system_fingerprints_per_turn) == loop_state.turn_count


# ============================================================
# Context-overflow protection (this PR): budgeted prefanout render,
# running context gate, and the 200-not-503 degrade classification.
# ============================================================

def _big_hit(
    doc_id: str,
    source_system: str = "github",
    content_reps: int = 400,
    chunk_id: str | None = None,
) -> dict[str, Any]:
    """A prefanout hit with a chunky ~content_reps-token body."""
    return {
        "channel": "vector",
        "chunk_id": chunk_id or f"{doc_id}:chunk:0",
        "doc_id": doc_id,
        "source_system": source_system,
        "source_url": f"https://example/{doc_id}",
        "title": f"title {doc_id}",
        "content": "lorem ipsum " * content_reps,
        "score": 0.9,
    }


def test_prefanout_budget_trims_when_over_budget() -> None:
    """Over-budget pre-fan-out is trimmed and the agent is told docs were
    omitted. This is the turn-1 half of the 131K overflow fix."""
    hits = [_big_hit(f"github:doc:{i}") for i in range(30)]
    prefanout = {"sub_queries": [{
        "query": "q", "vector": hits, "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    with patch("engine.retrieval.agent.loop.SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET", 2000):
        out = _render_prefanout_budgeted(prefanout)
    assert "trimmed to fit context" in out
    present = sum(1 for i in range(30) if f"github:doc:{i}" in out)
    assert 1 <= present < 30


def test_prefanout_budget_no_source_masking() -> None:
    """A minority-source doc that RANKS well survives the budget even when
    another source dominates by volume. Guards the PR#307/#328 'GitHub-only
    chunks' masking that got a per-channel cap reverted: this is global
    Top-N by fused rank, not per-channel."""
    # Smaller bodies so a handful of docs fit the budget (this tests ranking,
    # not raw capacity): 20 github hits fill the vector channel, one slack hit
    # sits at bm25 rank 0 so it fuses as high as the top github vector hit.
    github = [_big_hit(f"github:doc:{i}", content_reps=100) for i in range(20)]
    slack = _big_hit("slack:thread:T1", source_system="slack", content_reps=100)
    prefanout = {"sub_queries": [{
        "query": "q", "vector": github, "bm25": [slack], "graph": [], "inferred_edge": [],
    }]}
    with patch("engine.retrieval.agent.loop.SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET", 2500):
        out = _render_prefanout_budgeted(prefanout)
    assert "trimmed to fit context" in out  # budget did bite (github tail dropped)
    present_github = sum(1 for i in range(20) if f"github:doc:{i}" in out)
    assert present_github < 20  # some github docs were trimmed
    # ...but the high-fused minority source survived the trim (the reverted bug).
    assert "slack:thread:T1" in out, "minority source masked out — the reverted bug"


def test_prefanout_budget_deterministic_for_cache() -> None:
    """Same input -> byte-identical render, so the Cerebras prefix cache
    stays warm turn to turn."""
    hits = [_big_hit(f"github:doc:{i}") for i in range(15)]
    prefanout = {"sub_queries": [{
        "query": "q", "vector": hits, "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    with patch("engine.retrieval.agent.loop.SEARCH_AGENT_PREFANOUT_TOKEN_BUDGET", 1500):
        a = _render_prefanout_budgeted(prefanout)
        b = _render_prefanout_budgeted(prefanout)
    assert a == b


def test_prefanout_budget_small_input_unchanged() -> None:
    """Under-budget input is dumped in full (no trim, no note) — normal
    queries are unaffected."""
    prefanout = {"sub_queries": [{
        "query": "q", "vector": [_mk_hit()], "bm25": [], "graph": [], "inferred_edge": [],
    }]}
    out = _render_prefanout_budgeted(prefanout)
    assert "trimmed to fit context" not in out
    assert "github:owner/repo:pr:42" in out


def test_enforce_context_budget_stubs_oldest_tool_results() -> None:
    """Over-window history: oldest tool results get stubbed in place;
    system / user / assistant messages and tool_call_id pairing survive
    (an orphaned pair would 400)."""
    big = "x " * 100_000  # ~50K tokens each
    state = LoopState(
        customer_id="c", trace_id="t", query="q",
        messages=[
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": "the query evidence"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_1", "type": "function",
                             "function": {"name": "fetch_doc", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": big},
            {"role": "assistant", "content": "",
             "tool_calls": [{"id": "call_2", "type": "function",
                             "function": {"name": "fetch_doc", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_2", "content": big},
        ],
    )
    with patch("engine.retrieval.agent.loop.SEARCH_AGENT_MAX_CONTEXT_TOKENS", 10_000):
        _enforce_context_budget(state)
    # System / user / assistant untouched; tool_call_id linkage intact.
    assert state.messages[0]["content"][0]["text"] == "sys"
    assert state.messages[1]["content"] == "the query evidence"
    assert state.messages[2]["tool_calls"][0]["id"] == "call_1"
    assert state.messages[3]["tool_call_id"] == "call_1"  # pairing preserved
    assert '"truncated": true' in state.messages[3]["content"]  # oldest stubbed
    total = sum(
        _count_tokens(m["content"]) for m in state.messages if isinstance(m.get("content"), str)
    )
    assert total < 50_000  # materially smaller than the ~100K we started with


def test_is_context_overflow_classifies_overflow_vs_outage() -> None:
    """The gatherer degrades to 200 when the shared `is_context_overflow`
    predicate (shared/llm_tools) is True, and 503s otherwise. Lock the two
    real shapes: the typed ContextWindowExceededError cause, and the actual
    Cerebras message string from the gatherer-503 investigation."""
    import litellm

    # Typed cause path (status 400 required).
    cwe = litellm.ContextWindowExceededError(
        message="reduce the length", model="cerebras/gpt-oss-120b", llm_provider="cerebras"
    )
    wrapped = LLMError("wrapped", status_code=400)
    wrapped.__cause__ = cwe
    assert is_context_overflow(wrapped) is True

    # The verbatim production error string (message-regex path), status 400.
    real = LLMError(
        "litellm.ContextWindowExceededError: CerebrasException - Please reduce "
        "the length of the messages or completion. Current length is 229718 "
        "while limit is 131000",
        status_code=400,
    )
    assert is_context_overflow(real) is True

    # A genuine provider outage is NOT a context overflow -> stays a 503.
    outage = LLMError("connection reset by peer", status_code=503)
    assert is_context_overflow(outage) is False


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(408, True), (429, True), (500, True), (503, True), (400, False), (401, False)],
)
def test_is_transient_provider_error_classifies_http_statuses(
    status_code: int,
    expected: bool,
) -> None:
    assert (
        is_transient_provider_error(LLMError("opaque", status_code=status_code))
        is expected
    )


def test_is_transient_provider_error_requires_typed_statusless_cause() -> None:
    message_only = LLMError("timeout connecting to unavailable provider")
    assert is_transient_provider_error(message_only) is False

    typed = LLMError("opaque")
    typed.__cause__ = TimeoutError("late")
    assert is_transient_provider_error(typed) is True


def test_non_transient_http_status_overrides_connection_cause() -> None:
    auth_error = LLMError("authentication rejected", status_code=401)
    auth_error.__cause__ = ConnectionError("connection closed after response")
    assert is_transient_provider_error(auth_error) is False


def test_transient_provider_error_cause_cycle_terminates() -> None:
    first = RuntimeError("first")
    second = RuntimeError("second")
    first.__cause__ = second
    second.__cause__ = first
    wrapped = LLMError("opaque")
    wrapped.__cause__ = first

    assert is_transient_provider_error(wrapped) is False
