"""Unit tests for `services.retrieval.agent.extractor`.

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

from services.retrieval.agent.extractor import (
    _coerce_search_options,
    extract_entities_with_llm,
)
from services.retrieval.agent.models import EntityExtraction, SearchOptions
from services.retrieval.grounding import GroundingBundle


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
async def test_extract_returns_entity_extraction_with_search_options(
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch.setattr(
        "services.retrieval.agent.extractor.acompletion",
        AsyncMock(return_value=_fake_completion(payload)),
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
        "services.retrieval.agent.extractor.acompletion",
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
        "services.retrieval.agent.extractor.acompletion",
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
        "services.retrieval.agent.extractor.acompletion",
        AsyncMock(return_value=_fake_completion(payload)),
    )

    result = await extract_entities_with_llm(
        customer_id="cust-1",
        query="tell me about the auth refactor",
        bundle=GroundingBundle(),
    )

    assert result.search_options.sort == "relevance"
    assert len(result.entities) == 1
