"""Unit tests for agent_compactor."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.synthesis.agent_compactor import call_summarizer, extract_state_for_summary
from shared.exceptions import AgentCompactionError


def _runtime_state(**overrides) -> dict:
    base = {
        "pending_updates": [
            {"wiki_type": "decision", "slug": "a", "applied_queue_ids": [1, 2]},
        ],
        "pending_creates": [
            {"wiki_type": "runbook", "slug": "b", "applied_queue_ids": [3]},
        ],
        "applied_queue_ids": [1, 2, 3],
        "skipped_queue_ids": [4, 5],
    }
    base.update(overrides)
    return base


def _stub_client(text: str = "summary text"):
    """Build a mock genai client whose generate_content returns `text`."""
    client = SimpleNamespace()
    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(text=text)
    )
    return client


@pytest.mark.asyncio
async def test_call_summarizer_happy_path() -> None:
    """Stubbed Gemini returns a summary; output contains the runtime
    state block + the LLM text."""
    state = _runtime_state()
    client = _stub_client("conversation summary here")
    out = await call_summarizer(
        [{"role": "user", "parts": [{"text": "hello"}]}],
        state,
        client=client,
    )
    assert "RUNTIME STATE:" in out
    # Either appended (because summary text didn't include it) OR the
    # summary includes it — either way, the runtime block must be in.
    assert "applied_queue_ids: [1, 2, 3]" in out


def test_extract_state_for_summary_correctness() -> None:
    """The state block round-trips structured fields verbatim."""
    state = _runtime_state()
    block = extract_state_for_summary(state)
    assert "pending_updates: 1 pages" in block
    assert "[decision/a]" in block
    assert "[runbook/b]" in block
    assert "applied_queue_ids: [1, 2, 3]" in block
    assert "skipped_queue_ids: [4, 5]" in block


@pytest.mark.asyncio
async def test_summarizer_fails_raises_AgentCompactionError() -> None:
    """An empty Gemini response should raise AgentCompactionError; the
    harness catches it and re-raises as AgentHaltError."""
    client = _stub_client("")
    with pytest.raises(AgentCompactionError):
        await call_summarizer(
            [{"role": "user", "parts": [{"text": "x"}]}],
            _runtime_state(),
            client=client,
        )


# ---- Phase-0b chunk B: shared.llm.acompletion path -----------------------


def _stub_acompletion_response(text: str):
    """Build a litellm-shaped ChatCompletion response stub."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


@pytest.mark.asyncio
async def test_call_summarizer_routes_through_shared_llm(monkeypatch) -> None:
    """When no client is injected, the summarizer calls
    `shared.llm.acompletion` with the gemini/<model> prefix and surfaces
    the message content. This is the production path that honors
    `LLM_GATEWAY_URL` for managed-isolated tenants."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)

    captured: dict = {}

    async def fake_acompletion(*, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return _stub_acompletion_response("conversation summary here")

    import services.synthesis.agent_compactor as compactor_mod
    from shared import llm as shared_llm

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)
    # The compactor imports `from shared.config import get_settings`
    # inside the function — patch the SecretStr roundtrip the easiest
    # way (via the existing settings instance).
    from shared.config import get_settings

    get_settings.cache_clear()

    out = await call_summarizer(
        [{"role": "user", "parts": [{"text": "hello"}]}],
        _runtime_state(),
    )

    assert captured["model"] == f"gemini/{compactor_mod.WIKI_AGENT_COMPACTOR_MODEL}"
    # System + user roles, in that order.
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert "conversation summarizer" in captured["messages"][0]["content"].lower()
    assert captured["kwargs"].get("max_tokens") == 2048
    assert "RUNTIME STATE:" in out
    assert "applied_queue_ids: [1, 2, 3]" in out


@pytest.mark.asyncio
async def test_call_summarizer_uses_gateway_when_no_google_key(monkeypatch) -> None:
    """Managed-isolated tenant: no GOOGLE_API_KEY but LLM_GATEWAY_URL is
    set — the compactor should still run (the wrapper auto-injects the
    gateway URL; api_key injection lands in chunk A)."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.litellm.svc.cluster.local:4000")

    async def fake_acompletion(*, model, messages, **kwargs):
        return _stub_acompletion_response("gateway summary text")

    from shared import llm as shared_llm
    from shared.config import get_settings

    get_settings.cache_clear()
    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    out = await call_summarizer(
        [{"role": "user", "parts": [{"text": "x"}]}],
        _runtime_state(),
    )
    assert "RUNTIME STATE:" in out


@pytest.mark.asyncio
async def test_call_summarizer_raises_when_neither_key_nor_gateway(monkeypatch) -> None:
    """Without GOOGLE_API_KEY and without LLM_GATEWAY_URL, the compactor
    raises AgentCompactionError (it has no way to reach a model)."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from shared.config import get_settings

    get_settings.cache_clear()

    with pytest.raises(AgentCompactionError):
        await call_summarizer(
            [{"role": "user", "parts": [{"text": "x"}]}],
            _runtime_state(),
        )


@pytest.mark.asyncio
async def test_call_summarizer_wraps_llm_error_as_compaction_error(
    monkeypatch,
) -> None:
    """An LLMError from the wrapper should become AgentCompactionError."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)

    from shared import llm as shared_llm
    from shared.config import get_settings

    get_settings.cache_clear()

    async def fake_acompletion(*, model, messages, **kwargs):
        raise shared_llm.LLMError("rate limited", status_code=429, provider="google")

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    with pytest.raises(AgentCompactionError):
        await call_summarizer(
            [{"role": "user", "parts": [{"text": "x"}]}],
            _runtime_state(),
        )
