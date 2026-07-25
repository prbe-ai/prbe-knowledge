"""Unit tests for `engine.retrieval.agent.extractor`.

The extractor's job is to turn the raw query + grounding bundle into an
`EntityExtraction` (entities + `search_options`). After the
"what did X do last" optimization landed, the extractor also carries the
sort directive — verify both happy paths AND the post-parse coercion
defense against non-strict-decoding providers (Cerebras gpt-oss-120b is
known to emit unconstrained free-form text in Literal slots; see
`feedback_fireworks_response_format_4_layer_gotcha`).
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from engine.retrieval.agent.extractor import (
    _coerce_search_options,
    extract_entities_with_llm,
)
from engine.retrieval.agent.models import (
    EntityExtraction,
    SearchOptions,
    assert_grounded_types_are_emittable,
)
from engine.retrieval.grounding import GroundingBundle
from engine.shared.constants import (
    GROUNDING_ADDRESSABLE_ENTITY_TYPES,
    LLM_EXTRACTABLE_ENTITY_TYPES,
    SEARCH_AGENT_EXTRACTOR_TIMEOUT_SECONDS,
)


def _fake_completion(content: str) -> SimpleNamespace:
    """Shape what LiteLLM's acompletion returns: a response with `choices`,
    each choice carrying a `message.content` string."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
    )


def test_coerce_search_options_accepts_known_sort() -> None:
    assert _coerce_search_options({"sort": "recency"}) == {"sort": "recency"}
    assert _coerce_search_options({"sort": "relevance"}) == {"sort": "relevance"}


def test_coerce_search_options_coerces_unknown_sort_to_relevance() -> None:
    # Cerebras gpt-oss-120b has been observed to emit free-form text in
    # Literal slots (e.g. `entity_type="mahit namburu"`). Same risk for
    # `sort`: the model might emit `"recent"`, `"latest"`, `"newest"`,
    # etc. Coerce to the safe default so EntityExtraction validation
    # doesn't reject the entire payload over one bogus enum value.
    assert _coerce_search_options({"sort": "recent"}) == {"sort": "relevance"}
    assert _coerce_search_options({"sort": "latest"}) == {"sort": "relevance"}
    assert _coerce_search_options({"sort": "newest"}) == {"sort": "relevance"}


def test_coerce_search_options_passes_through_non_dict() -> None:
    # Defensive: a provider that emits search_options as a string or null
    # should fall through to the default SearchOptions, not crash.
    assert _coerce_search_options(None) == {}
    assert _coerce_search_options("recency") == {}
    assert _coerce_search_options([{"sort": "recency"}]) == {}
    # Numbers, booleans — defense-in-depth against pathological providers.
    assert _coerce_search_options(42) == {}
    assert _coerce_search_options(True) == {}


def test_coerce_search_options_preserves_unknown_keys() -> None:
    # Forward-compat: when SearchOptions gains a new field via a future PR,
    # an in-flight payload from a model that doesn't know about it should
    # NOT have the unknown key stripped here — Pydantic's `extra="forbid"`
    # on SearchOptions catches that at validation, and we want the parse
    # failure path to fire there (which logs preview), not silently here.
    out = _coerce_search_options({"sort": "recency", "doc_types": ["pr"]})
    assert out == {"sort": "recency", "doc_types": ["pr"]}


def test_coerce_search_options_handles_missing_sort_key() -> None:
    # Partial payloads (only some fields emitted) pass through untouched.
    # `SearchOptions` will fill in the default.
    assert _coerce_search_options({}) == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("gateway_enabled", [True, False])
async def test_extract_returns_entity_extraction_with_search_options(
    monkeypatch: pytest.MonkeyPatch,
    gateway_enabled: bool,
) -> None:
    """End-to-end: extractor returns the parsed EntityExtraction object
    carrying both entities and the sort directive."""
    payload = json.dumps({
        "entities": [
            {
                "entity_type": "person",
                "canonical_id": "mahit@prbe.ai",
                "display_name": "Mahit",
                "confidence": 1.0,
            }
        ],
        "search_options": {"sort": "recency"},
    })
    mock_acompletion = AsyncMock(return_value=_fake_completion(payload))
    if gateway_enabled:
        monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.example")
    else:
        monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.setattr(
        "engine.retrieval.agent.extractor.acompletion",
        mock_acompletion,
    )

    result = await extract_entities_with_llm(
        customer_id="cust-1",
        query="what did mahit do last?",
        bundle=GroundingBundle(),
    )

    assert isinstance(result, EntityExtraction)
    assert len(result.entities) == 1
    assert result.entities[0].canonical_id == "mahit@prbe.ai"
    assert result.search_options.sort == "recency"
    call_kwargs = mock_acompletion.await_args.kwargs
    assert call_kwargs["timeout"] == SEARCH_AGENT_EXTRACTOR_TIMEOUT_SECONDS
    if gateway_enabled:
        assert call_kwargs["max_retries"] == 0
    else:
        assert "max_retries" not in call_kwargs


@pytest.mark.asyncio
async def test_extract_coerces_bad_sort_before_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider emits `sort="latest"` (non-Literal). The two-stage parse
    must coerce it to `"relevance"` BEFORE Pydantic validates, otherwise
    the whole EntityExtraction is rejected and the entities are lost
    too."""
    payload = json.dumps({
        "entities": [
            {
                "entity_type": "person",
                "canonical_id": "mahit@prbe.ai",
                "display_name": "Mahit",
                "confidence": 1.0,
            }
        ],
        "search_options": {"sort": "latest"},
    })
    monkeypatch.setattr(
        "engine.retrieval.agent.extractor.acompletion",
        AsyncMock(return_value=_fake_completion(payload)),
    )

    result = await extract_entities_with_llm(
        customer_id="cust-1",
        query="what did mahit do last?",
        bundle=GroundingBundle(),
    )

    # Entities survive — coercion fixed the bad sort before validation.
    assert len(result.entities) == 1
    assert result.search_options.sort == "relevance"


@pytest.mark.asyncio
async def test_extract_returns_defaulted_on_json_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "engine.retrieval.agent.extractor.acompletion",
        AsyncMock(return_value=_fake_completion("not json at all")),
    )

    result = await extract_entities_with_llm(
        customer_id="cust-1",
        query="anything",
        bundle=GroundingBundle(),
    )

    assert isinstance(result, EntityExtraction)
    assert result.entities == []
    assert result.search_options == SearchOptions()


@pytest.mark.asyncio
async def test_extract_search_options_omitted_defaults_to_relevance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backwards compat: provider that doesn't emit search_options (old
    payload shape) still parses and gets the safe default."""
    payload = json.dumps({
        "entities": [
            {
                "entity_type": "feature",
                "canonical_id": "auth-refactor",
                "display_name": "auth refactor",
                "confidence": 0.9,
            }
        ],
    })
    monkeypatch.setattr(
        "engine.retrieval.agent.extractor.acompletion",
        AsyncMock(return_value=_fake_completion(payload)),
    )

    result = await extract_entities_with_llm(
        customer_id="cust-1",
        query="tell me about the auth refactor",
        bundle=GroundingBundle(),
    )

    assert result.search_options.sort == "relevance"
    assert len(result.entities) == 1


def test_grounded_entity_type_must_be_emittable() -> None:
    """The guard that would have caught the research-corpus outage.

    Grounding surfaced Experiment/Project candidates while the extractor's
    vocabulary omitted them, so the model named them, validation rejected the
    whole emission, and every search silently ran with zero entity anchors.
    Exercises the production callable, not a paraphrase of it — a guard that
    only runs on the day it matters is a guard nobody has verified.
    """
    # Real registry-derived tuples must already satisfy the invariant.
    assert_grounded_types_are_emittable(
        GROUNDING_ADDRESSABLE_ENTITY_TYPES, LLM_EXTRACTABLE_ENTITY_TYPES
    )

    # Drift in the direction that broke: grounded, but not emittable.
    with pytest.raises(RuntimeError, match="experiment"):
        assert_grounded_types_are_emittable(
            ["person", "experiment"], ["person"]
        )

    # The opposite direction is legitimate and must NOT raise: file_path,
    # session and commit_sha are emittable but never grounded (they reach the
    # graph via bare-ID detection).
    assert_grounded_types_are_emittable(["person"], ["person", "commit_sha"])


@pytest.mark.asyncio
async def test_extract_keeps_research_entities_and_drops_only_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REGRESSION: `experiment` / `project` used to fail validation and take
    every valid entity down with them, returning an empty extraction while the
    response still read state:"ok".

    Also pins that rejection is per-entity and not entity_type-specific: the
    bad `confidence` member is dropped on its own, without discarding the rest.
    """
    payload = json.dumps({
        "entities": [
            {
                "entity_type": "experiment",
                "canonical_id": "abag-leg3",
                "display_name": "AbAg Leg 3",
                "confidence": 0.9,
            },
            {
                "entity_type": "project",
                "canonical_id": "anthrogen",
                "display_name": "Anthrogen",
                "confidence": 0.8,
            },
            {
                "entity_type": "person",
                "canonical_id": "bad-confidence",
                "display_name": "Nope",
                "confidence": 1.5,
            },
        ],
        "search_options": {"sort": "relevance"},
    })
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    monkeypatch.setattr(
        "engine.retrieval.agent.extractor.acompletion",
        AsyncMock(return_value=_fake_completion(payload)),
    )

    result = await extract_entities_with_llm(
        customer_id="cust-1",
        query="anthrogen abag protenix",
        bundle=GroundingBundle(),
    )

    kept = {e.canonical_id: e.entity_type for e in result.entities}
    assert kept == {"abag-leg3": "experiment", "anthrogen": "project"}
