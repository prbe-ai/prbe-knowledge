"""Unit tests for the Phase-0b chunk B migration of
`cross_repo_deps._call_classifier_llm`.

The production path now routes through `shared.llm.acompletion` so the
classifier honors `LLM_GATEWAY_URL` for gateway routing. The
test-injected `client` kwarg (mimicking the google-genai surface) is
preserved for fixtures that pre-date the migration.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kb.code_graph.cross_repo_deps import _call_classifier_llm


def _stub_acompletion_response(text: str):
    """Build a litellm-shaped ChatCompletion response."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


@pytest.mark.asyncio
async def test_classifier_returns_none_when_no_key_or_gateway(monkeypatch) -> None:
    """Without GOOGLE_API_KEY AND without LLM_GATEWAY_URL, the
    classifier returns None — the caller (classify_with_llm) treats
    that as ClassifierUnavailable."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    result = await _call_classifier_llm(
        source_repo="prbe-ai/example",
        user_prompt="match list",
    )
    assert result is None


@pytest.mark.asyncio
async def test_classifier_routes_through_shared_llm(monkeypatch) -> None:
    """With GOOGLE_API_KEY set the classifier calls
    `shared.llm.acompletion` with model `gemini/gemini-3.1-pro-preview`
    and returns the parsed `verdicts` list."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    captured: dict = {}

    async def fake_acompletion(*, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        # Free-text JSON: the classifier parses the returned content.
        return _stub_acompletion_response(
            '{"verdicts": [{"number": 1, "real": true, "reason": "import"}]}'
        )

    from engine.shared import llm as shared_llm

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    result = await _call_classifier_llm(
        source_repo="prbe-ai/example",
        user_prompt="candidates here",
    )

    # Pro pin must survive verbatim (project memory
    # project_gemini_dedupe_model_id forbids downgrades).
    assert captured["model"] == "gemini/gemini-3.1-pro-preview"
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert "static-analysis pre-filter" in captured["messages"][0]["content"].lower()
    assert captured["kwargs"].get("max_tokens") == 32768
    assert captured["kwargs"].get("response_format") == {"type": "json_object"}
    assert result == [{"number": 1, "real": True, "reason": "import"}]


@pytest.mark.asyncio
async def test_classifier_uses_gateway_without_google_key(monkeypatch) -> None:
    """Managed-isolated tenant: no GOOGLE_API_KEY but LLM_GATEWAY_URL
    is set — the classifier must still call shared.llm.acompletion."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.litellm.svc.cluster.local:4000")
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    async def fake_acompletion(*, model, messages, **kwargs):
        return _stub_acompletion_response('{"verdicts": []}')

    from engine.shared import llm as shared_llm

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    result = await _call_classifier_llm(
        source_repo="prbe-ai/example",
        user_prompt="x",
    )
    # Empty verdicts is a legitimate "no candidates were REAL" answer.
    assert result == []


@pytest.mark.asyncio
async def test_classifier_returns_none_on_llm_error(monkeypatch) -> None:
    """An LLMError from the wrapper translates to None (LLM failure)."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    from engine.shared import llm as shared_llm

    async def fake_acompletion(*, model, messages, **kwargs):
        raise shared_llm.LLMError(
            "rate limited", status_code=429, provider="google"
        )

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    result = await _call_classifier_llm(
        source_repo="prbe-ai/example",
        user_prompt="x",
    )
    assert result is None


@pytest.mark.asyncio
async def test_classifier_returns_none_on_malformed_json(monkeypatch) -> None:
    """Non-JSON content → caller-side parser returns None."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    async def fake_acompletion(*, model, messages, **kwargs):
        return _stub_acompletion_response("not actually json")

    from engine.shared import llm as shared_llm

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    result = await _call_classifier_llm(
        source_repo="prbe-ai/example",
        user_prompt="x",
    )
    assert result is None


@pytest.mark.asyncio
async def test_classifier_legacy_client_path_still_works() -> None:
    """A test-injected google-genai-shaped client bypasses the wrapper.
    Preserves backward-compat for older fixtures."""
    client = SimpleNamespace()
    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(
            text='{"verdicts": [{"number": 1, "real": false, "reason": "doc link"}]}'
        )
    )

    result = await _call_classifier_llm(
        source_repo="prbe-ai/example",
        user_prompt="x",
        client=client,
    )
    assert result == [{"number": 1, "real": False, "reason": "doc link"}]
