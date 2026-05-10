"""Unit tests for the DirectedPhrasesProvider abstraction in
`services/synthesis/providers.py`.

Covers:
  - factory dispatch by model name (Anthropic vs Gemini, unknown raises)
  - _AnthropicDirectedPhrases parsing (happy + 3 error paths)
  - _GeminiDirectedPhrases parsing (delegates to _gemini_call_json)
  - _coerce_phrases (length cap, max-count cap, type coercion)
  - _thinking_budget_for (per-model rule)

DB and live LLM calls are NOT exercised here. End-to-end coverage lives
in tests/synthesis/test_directed_phrases.py (orchestrator) and
tests/retrieval/test_search_pipeline_directed.py (retrieval e2e).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services.synthesis.providers import (
    DirectedPhrasesParseError,
    _AnthropicDirectedPhrases,
    _coerce_phrases,
    _GeminiDirectedPhrases,
    _thinking_budget_for,
    get_directed_phrases_provider,
)
from shared.constants import (
    HAIKU_MODEL,
    MAX_DIRECTED_PHRASE_CHARS,
    MAX_DIRECTED_VECTORS_PER_DOC,
)

# ---- _coerce_phrases (rule-based, no LLM) --------------------------------


def test_coerce_phrases_happy_path() -> None:
    out = _coerce_phrases(["alpha", "beta", "  gamma  "])
    assert out == ["alpha", "beta", "gamma"]


def test_coerce_phrases_drops_empty_and_non_strings() -> None:
    out = _coerce_phrases(["alpha", "", "  ", 42, None, "beta"])
    assert out == ["alpha", "beta"]


def test_coerce_phrases_drops_oversize() -> None:
    long = "x" * (MAX_DIRECTED_PHRASE_CHARS + 10)
    out = _coerce_phrases(["short", long, "also short"])
    assert "short" in out
    assert "also short" in out
    assert long not in out


def test_coerce_phrases_caps_count() -> None:
    out = _coerce_phrases([f"phrase {i}" for i in range(MAX_DIRECTED_VECTORS_PER_DOC + 5)])
    assert len(out) == MAX_DIRECTED_VECTORS_PER_DOC


def test_coerce_phrases_rejects_non_list() -> None:
    with pytest.raises(DirectedPhrasesParseError):
        _coerce_phrases("not a list")  # type: ignore[arg-type]
    with pytest.raises(DirectedPhrasesParseError):
        _coerce_phrases({"phrases": ["alpha"]})  # type: ignore[arg-type]


# ---- _thinking_budget_for ------------------------------------------------


def test_thinking_budget_zero_for_flash_and_flash_lite() -> None:
    # The eval at scripts/eval_directed_phrases.py confirmed Flash and
    # Flash Lite both work with budget=0; pin that contract.
    assert _thinking_budget_for("gemini-3-flash-preview") == 0
    assert _thinking_budget_for("gemini-3.1-flash-lite-preview") == 0
    assert _thinking_budget_for("gemini-flash-lite") == 0


def test_thinking_budget_nonzero_for_pro() -> None:
    # Pro REJECTS budget=0 (NeonDbError-style 400 in the eval). Must
    # carry slack.
    assert _thinking_budget_for("gemini-3.1-pro-preview") > 0
    assert _thinking_budget_for("gemini-3-pro") > 0


# ---- _AnthropicDirectedPhrases ------------------------------------------


def _anthropic_response(phrases: list[str]) -> Any:
    """Mimic the AsyncAnthropic Message shape: .content is a list of blocks
    with .type and .name. _AnthropicDirectedPhrases.generate looks for
    `type == "tool_use"` and `name == "record_directed_phrases"`.
    """
    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_directed_phrases"
    block.input = {"phrases": phrases}
    response = MagicMock()
    response.content = [block]
    return response


def _anthropic_client_returning(response: Any) -> Any:
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


@pytest.mark.asyncio
async def test_anthropic_directed_happy_path() -> None:
    client = _anthropic_client_returning(
        _anthropic_response(["alpha", "beta", "gamma"])
    )
    provider = _AnthropicDirectedPhrases(client, model=HAIKU_MODEL)
    out = await provider.generate(page_title="t", page_body="b")
    assert out == ["alpha", "beta", "gamma"]


@pytest.mark.asyncio
async def test_anthropic_directed_no_tool_use_block_raises() -> None:
    response = MagicMock()
    response.content = []  # no tool_use block at all
    client = _anthropic_client_returning(response)
    provider = _AnthropicDirectedPhrases(client, model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError, match=r"no .* tool_use block"):
        await provider.generate(page_title="t", page_body="b")


@pytest.mark.asyncio
async def test_anthropic_directed_input_not_a_dict_raises() -> None:
    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_directed_phrases"
    block.input = "not a dict"  # provider returned a malformed payload
    response = MagicMock()
    response.content = [block]
    client = _anthropic_client_returning(response)
    provider = _AnthropicDirectedPhrases(client, model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError, match="not a dict"):
        await provider.generate(page_title="t", page_body="b")


@pytest.mark.asyncio
async def test_anthropic_directed_phrases_not_a_list_raises() -> None:
    # _coerce_phrases rejects non-list payloads with the same exception.
    response = _anthropic_response(["alpha"])
    response.content[0].input = {"phrases": "not a list"}
    client = _anthropic_client_returning(response)
    provider = _AnthropicDirectedPhrases(client, model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError, match="not a list"):
        await provider.generate(page_title="t", page_body="b")


# ---- _GeminiDirectedPhrases ----------------------------------------------


@pytest.mark.asyncio
async def test_gemini_directed_happy_path() -> None:
    """`_gemini_call_json` is the only thing that talks to the network.
    Patch it; verify _GeminiDirectedPhrases passes correct args + parses
    the returned dict.
    """
    with patch(
        "services.synthesis.providers._gemini_call_json",
        new=AsyncMock(return_value={"phrases": ["alpha", "beta"]}),
    ) as mock_call:
        provider = _GeminiDirectedPhrases(model="gemini-3-flash-preview")
        out = await provider.generate(page_title="t", page_body="b")
    assert out == ["alpha", "beta"]
    # Call args carry the model + flattened prompt.
    kwargs = mock_call.await_args.kwargs
    assert kwargs["model"] == "gemini-3-flash-preview"
    assert kwargs["schema"]["properties"]["phrases"]["type"] == "array"


@pytest.mark.asyncio
async def test_gemini_directed_underlying_failure_wraps_as_parse_error() -> None:
    """A network/SDK error from `_gemini_call_json` propagates as
    `DirectedPhrasesParseError` so callers can match a single exception
    type regardless of provider. Pins the 'silent partial state' fix
    from the directed_phrases.persist orchestrator.
    """
    with patch(
        "services.synthesis.providers._gemini_call_json",
        new=AsyncMock(side_effect=RuntimeError("gemini 5xx")),
    ):
        provider = _GeminiDirectedPhrases(model="gemini-3-flash-preview")
        with pytest.raises(DirectedPhrasesParseError, match=r"gemini.*call failed"):
            await provider.generate(page_title="t", page_body="b")


# ---- factory dispatch ----------------------------------------------------


def test_factory_dispatch_gemini_flash() -> None:
    provider = get_directed_phrases_provider(model_override="gemini-3-flash-preview")
    assert isinstance(provider, _GeminiDirectedPhrases)


def test_factory_dispatch_gemini_flash_lite_alias() -> None:
    # Explicit alias lets a future Flash -> Flash Lite flip be a one-line
    # constants.py change with no provider edit.
    provider = get_directed_phrases_provider(
        model_override="gemini-3.1-flash-lite-preview"
    )
    assert isinstance(provider, _GeminiDirectedPhrases)


def test_factory_dispatch_anthropic_requires_client() -> None:
    """When the configured model is Anthropic, the factory must require
    an AsyncAnthropic client. Passing none raises rather than building
    a half-configured provider."""
    with pytest.raises(ValueError, match="requires an AsyncAnthropic client"):
        get_directed_phrases_provider(model_override="haiku")


def test_factory_dispatch_anthropic_with_client() -> None:
    client = MagicMock()
    provider = get_directed_phrases_provider(
        anthropic_client=client, model_override="haiku"
    )
    assert isinstance(provider, _AnthropicDirectedPhrases)


def test_factory_dispatch_unknown_model_raises_loudly() -> None:
    """Loud failure at deploy time beats a silent fallback that ships
    the wrong provider for weeks unnoticed."""
    with pytest.raises(ValueError, match="unknown DIRECTED_PHRASES_MODEL"):
        get_directed_phrases_provider(model_override="some-unreleased-model")
