"""Provider Protocol dispatch tests.

The triage / synthesis / verifier stages each select a provider via env
var. These tests verify:
  - default model names resolve to the Anthropic implementation.
  - explicit `model_override` flips the dispatch.
  - unknown model names raise.
  - the Protocol round-trip works for the Anthropic path with a mocked
    AsyncAnthropic client (Gemini path is exercised separately if a real
    GOOGLE_API_KEY is set; CI doesn't, so it stays a smoke test).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.synthesis.models import (
    SynthesisInput,
    TriageInput,
    VerifierInput,
)
from services.synthesis.providers import (
    SynthesisParseError,
    TriageParseError,
    VerifierParseError,
    _AnthropicSynthesis,
    _AnthropicTriage,
    _AnthropicVerifier,
    get_synthesis_provider,
    get_triage_provider,
    get_verifier_provider,
)


def _ev(qid: int, doc_id: str = "doc:1") -> TriageInput:
    return TriageInput(
        queue_id=qid,
        doc_id=doc_id,
        doc_type="github.commit",
        source_system="github",
        title="t",
        author_id="alice",
        body="body",
        body_token_count=10,
    )


def _tool_use(name: str, payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name=name, input=payload)
    return SimpleNamespace(content=[block])


def _mock_anthropic(payload_by_tool: dict[str, dict]) -> SimpleNamespace:
    async def create(*, model: str, **kwargs):
        tool_name = (kwargs.get("tools") or [{}])[0].get("name", "")
        return _tool_use(tool_name, payload_by_tool[tool_name])

    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=create))
    return client


# ---------------------------------------------------------------------------
# Factory dispatch
# ---------------------------------------------------------------------------


def test_default_triage_provider_is_anthropic() -> None:
    client = _mock_anthropic({})
    provider = get_triage_provider(client)
    assert isinstance(provider, _AnthropicTriage)


def test_default_synthesis_provider_is_anthropic() -> None:
    client = _mock_anthropic({})
    provider = get_synthesis_provider(client)
    assert isinstance(provider, _AnthropicSynthesis)


def test_default_verifier_provider_is_anthropic() -> None:
    client = _mock_anthropic({})
    provider = get_verifier_provider(client)
    assert isinstance(provider, _AnthropicVerifier)


def test_triage_provider_dispatches_to_gemini_on_override() -> None:
    from services.synthesis.providers import _GeminiTriage

    provider = get_triage_provider(model_override="gemini-flash-lite-preview")
    assert isinstance(provider, _GeminiTriage)


def test_synthesis_provider_dispatches_to_gemini_on_override() -> None:
    from services.synthesis.providers import _GeminiSynthesis

    provider = get_synthesis_provider(model_override="gemini-3.1-pro-preview")
    assert isinstance(provider, _GeminiSynthesis)


def test_verifier_provider_dispatches_to_gemini_on_override() -> None:
    from services.synthesis.providers import _GeminiVerifier

    provider = get_verifier_provider(model_override="gemini-3.1-pro-preview")
    assert isinstance(provider, _GeminiVerifier)


def test_unknown_triage_model_raises() -> None:
    with pytest.raises(ValueError, match="unknown WIKI_TRIAGE_MODEL"):
        get_triage_provider(model_override="not-a-real-model")


def test_anthropic_provider_requires_client() -> None:
    with pytest.raises(ValueError, match="requires an AsyncAnthropic client"):
        get_triage_provider(anthropic_client=None, model_override="haiku")


# ---------------------------------------------------------------------------
# Anthropic round-trips (verifier is the new path; triage/synthesis are
# already covered in test_triage / test_synthesize but re-asserted at the
# Protocol layer for symmetry)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_verifier_kept_subset_round_trip() -> None:
    cluster = VerifierInput(
        wiki_type="decision",
        slug="x",
        action="create",
        events=[_ev(1, "doc:keep"), _ev(2, "doc:drop")],
    )
    client = _mock_anthropic(
        {
            "record_verifier_verdict": {
                "kept_doc_ids": ["doc:keep"],
                "rationale_per_doc": {
                    "doc:keep": "states a decision",
                    "doc:drop": "unrelated",
                },
            }
        }
    )
    provider = get_verifier_provider(client)
    out = await provider.verify(cluster, now=datetime.now(UTC))
    assert out.kept_doc_ids == ["doc:keep"]
    assert out.rationale_per_doc["doc:drop"] == "unrelated"


@pytest.mark.asyncio
async def test_anthropic_verifier_empty_kept_round_trip() -> None:
    cluster = VerifierInput(
        wiki_type="runbook",
        slug="x",
        action="update",
        current_body="existing content",
        events=[_ev(1)],
    )
    client = _mock_anthropic(
        {
            "record_verifier_verdict": {
                "kept_doc_ids": [],
                "drop_reason": "restates existing content",
            }
        }
    )
    provider = get_verifier_provider(client)
    out = await provider.verify(cluster, now=datetime.now(UTC))
    assert out.kept_doc_ids == []
    assert out.drop_reason == "restates existing content"


@pytest.mark.asyncio
async def test_anthropic_verifier_raises_on_malformed_response() -> None:
    cluster = VerifierInput(wiki_type="decision", slug="x", action="create", events=[_ev(1)])

    # Response has a tool_use block but with the WRONG name — extractor
    # should not find the verifier block and raise.
    async def create(*, model: str, **kwargs):
        return _tool_use("not_the_verifier_tool", {})

    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(side_effect=create))
    provider = get_verifier_provider(client)
    with pytest.raises(VerifierParseError):
        await provider.verify(cluster, now=datetime.now(UTC))


@pytest.mark.asyncio
async def test_anthropic_synthesis_raises_on_validation_error() -> None:
    cluster = SynthesisInput(wiki_type="decision", slug="x", action="create", events=[_ev(1)])
    # body_markdown is required by SynthesisOutput; omit it.
    client = _mock_anthropic(
        {"render_wiki_page": {"title": "x", "summary": "x", "commit_message": "x"}}
    )
    provider = get_synthesis_provider(client)
    with pytest.raises(SynthesisParseError):
        await provider.synthesize(cluster, now=datetime.now(UTC))


@pytest.mark.asyncio
async def test_anthropic_triage_raises_on_validation_error() -> None:
    client = _mock_anthropic({"record_triage": {"verdicts": {"1": {"important": True}}}})
    provider = get_triage_provider(client)
    with pytest.raises(TriageParseError):
        await provider.triage([_ev(1)], now=datetime.now(UTC))
