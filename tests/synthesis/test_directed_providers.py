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
    # The canonical id is what gets sent to the API. Pin it so a future
    # alias-table edit doesn't silently route flash traffic elsewhere.
    assert provider._model == "gemini-3-flash-preview"


def test_factory_resolves_short_aliases_to_canonical_ids() -> None:
    """REGRESSION for the dead-branch alias bug.
    Before fix: factory used `name if name.startswith("gemini") else <fallback>`
    which sent the alias string verbatim as the API model id (4xx in prod).
    """
    # Flash short aliases:
    for alias in ("gemini-flash", "gemini-3-flash"):
        provider = get_directed_phrases_provider(model_override=alias)
        assert isinstance(provider, _GeminiDirectedPhrases)
        assert provider._model == "gemini-3-flash-preview", (
            f"alias {alias!r} must resolve to canonical id"
        )
    # Flash-Lite short aliases:
    for alias in ("gemini-flash-lite",):
        provider = get_directed_phrases_provider(model_override=alias)
        assert isinstance(provider, _GeminiDirectedPhrases)
        assert provider._model == "gemini-flash-lite-preview"


def test_factory_dispatch_gemini_flash_lite_alias() -> None:
    # Explicit alias lets a future Flash -> Flash Lite flip be a one-line
    # constants.py change with no provider edit.
    provider = get_directed_phrases_provider(
        model_override="gemini-3.1-flash-lite-preview"
    )
    assert isinstance(provider, _GeminiDirectedPhrases)
    assert provider._model == "gemini-3.1-flash-lite-preview"


def test_factory_dispatch_anthropic_lazy_constructs_client(monkeypatch) -> None:
    """REGRESSION for the rollback footgun.
    Before fix: factory raised ValueError when Anthropic was selected
    without a caller-supplied client. The wiki_agent caller passes none,
    so flipping DIRECTED_PHRASES_MODEL='haiku' silently disabled the
    feature instead of switching providers. Factory now mirrors
    _gemini_client()'s lazy pattern.
    """
    # Stub out AsyncAnthropic so we don't actually hit Anthropic.
    fake_client_cls = MagicMock()
    monkeypatch.setattr(
        "services.synthesis.providers.AsyncAnthropic", fake_client_cls
    )
    # Stub get_settings so we don't depend on real env config.
    fake_settings = MagicMock()
    fake_settings.anthropic_api_key.get_secret_value.return_value = "sk-test"
    monkeypatch.setattr(
        "services.synthesis.providers.get_settings", lambda: fake_settings
    )
    provider = get_directed_phrases_provider(model_override="haiku")
    assert isinstance(provider, _AnthropicDirectedPhrases)
    fake_client_cls.assert_called_once_with(api_key="sk-test")


def test_factory_dispatch_anthropic_raises_when_no_api_key(monkeypatch) -> None:
    """If ANTHROPIC_API_KEY isn't configured, lazy construction must
    raise loudly. Better deploy-time crash than silent llm_failed=True
    on every page synthesis.
    """
    fake_settings = MagicMock()
    fake_settings.anthropic_api_key.get_secret_value.return_value = ""
    monkeypatch.setattr(
        "services.synthesis.providers.get_settings", lambda: fake_settings
    )
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY not configured"):
        get_directed_phrases_provider(model_override="haiku")


def test_factory_dispatch_anthropic_with_client_skips_lazy_construct() -> None:
    """When the caller supplies a client (e.g. tests, future workers
    that own one), use it directly rather than building a fresh one.
    """
    client = MagicMock()
    provider = get_directed_phrases_provider(
        anthropic_client=client, model_override="haiku"
    )
    assert isinstance(provider, _AnthropicDirectedPhrases)
    assert provider._client is client


def test_factory_dispatch_unknown_model_raises_loudly() -> None:
    """Loud failure at deploy time beats a silent fallback that ships
    the wrong provider for weeks unnoticed."""
    with pytest.raises(ValueError, match="unknown DIRECTED_PHRASES_MODEL"):
        get_directed_phrases_provider(model_override="some-unreleased-model")


# ---- Regression tests for missing-key + parse failures ------------------


@pytest.mark.asyncio
async def test_gemini_directed_missing_phrases_key_raises_not_silent_empty(
    monkeypatch,
) -> None:
    """REGRESSION for the silent-prior-row-purge bug.
    Before fix: payload.get("phrases", []) returned [] on key drift, the
    orchestrator treated this as "successful zero phrases" and called
    _reconcile_llm with keep=[], DELETING all prior LLM rows. A prompt
    drift to {"trigger_phrases": ...} would silently wipe every doc's
    directed_vectors with no warning.

    After fix: missing key raises DirectedPhrasesParseError, which the
    orchestrator catches and flips llm_failed=True, preserving prior rows.
    """
    # Simulate Gemini returning a different key (prompt/schema drift).
    with patch(
        "services.synthesis.providers._gemini_call_json",
        new=AsyncMock(return_value={"trigger_phrases": ["a", "b"]}),
    ):
        provider = _GeminiDirectedPhrases(model="gemini-3-flash-preview")
        with pytest.raises(DirectedPhrasesParseError, match=r"missing 'phrases' key"):
            await provider.generate(page_title="t", page_body="b")


@pytest.mark.asyncio
async def test_anthropic_directed_missing_phrases_key_raises_not_silent_empty() -> None:
    """Symmetric to the Gemini missing-key test. Anthropic providers go
    through the same _coerce_phrases path and should also raise rather
    than silently return [].
    """
    block = MagicMock()
    block.type = "tool_use"
    block.name = "record_directed_phrases"
    block.input = {"some_other_key": ["a"]}  # missing 'phrases'
    response = MagicMock()
    response.content = [block]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    provider = _AnthropicDirectedPhrases(client, model=HAIKU_MODEL)
    with pytest.raises(DirectedPhrasesParseError, match=r"missing 'phrases' key"):
        await provider.generate(page_title="t", page_body="b")
