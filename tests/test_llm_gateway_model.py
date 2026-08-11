"""A bare model id routed through the gateway must use the OpenAI transport.

Regression test for the 2026-08-11 wiki triage outage: every triage call failed
with `GeminiException - {"detail":"Method Not Allowed"}` and one batch failure
dead-lettered 13,147 queue rows. Root cause was not the model id and not the
gateway -- it was the pair. LiteLLM infers its transport from the id's prefix,
so a bare `gemini-3.5-flash` selects Gemini's NATIVE client, which looks for
Application Default Credentials and dies inside the pod before any request
reaches the proxy.
"""

from __future__ import annotations

import pytest

from engine.shared.llm import _gateway_model

GATEWAY = {"api_base": "http://litellm.svc:4000/v1"}


def test_bare_id_through_a_gateway_becomes_an_openai_call() -> None:
    # The exact pair that failed in production.
    assert _gateway_model("gemini-3.5-flash", dict(GATEWAY)) == "openai/gemini-3.5-flash"
    assert (
        _gateway_model("gemini-3.1-pro-preview", dict(GATEWAY))
        == "openai/gemini-3.1-pro-preview"
    )


def test_direct_provider_calls_are_untouched() -> None:
    """No gateway => the bare id is CORRECT and must not be rewritten.

    This is the half a blanket constant-rename would have broken: upstream runs
    with direct provider keys, where `gemini-3.5-flash` is exactly right.
    """
    assert _gateway_model("gemini-3.5-flash", {}) == "gemini-3.5-flash"


@pytest.mark.parametrize(
    "model",
    [
        "cerebras/gpt-oss-120b",       # already prefixed
        "accounts/fireworks/llama-v3", # provider path with slashes
        "openai/gemini-3.5-flash",     # already normalized -- must not double up
    ],
)
def test_ids_that_already_choose_a_transport_are_left_alone(model: str) -> None:
    assert _gateway_model(model, dict(GATEWAY)) == model


def test_an_explicit_provider_wins_over_the_normalization() -> None:
    kwargs = dict(GATEWAY) | {"custom_llm_provider": "vertex_ai"}
    assert _gateway_model("gemini-3.5-flash", kwargs) == "gemini-3.5-flash"


@pytest.mark.asyncio
async def test_acompletion_actually_applies_the_normalization(monkeypatch) -> None:
    """The WIRING, not just the helper.

    Testing `_gateway_model` alone is not enough and this test exists because
    that gap was demonstrated: deleting the `model = _gateway_model(...)` line
    from `acompletion` -- restoring the exact production bug -- left every other
    test in this file green. A unit test of a pure function cannot fail when the
    caller stops calling it, so assert on what LiteLLM actually receives.
    """
    import engine.shared.llm as llm

    seen: dict[str, object] = {}

    async def fake_acompletion(*, model, messages, **kwargs):
        seen["model"] = model
        seen["api_base"] = kwargs.get("api_base")
        return "resp"

    monkeypatch.setattr(llm.litellm, "acompletion", fake_acompletion)
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.svc:4000/v1")
    llm.gateway_url.cache_clear() if hasattr(llm.gateway_url, "cache_clear") else None

    await llm.acompletion(
        model="gemini-3.5-flash", messages=[{"role": "user", "content": "hi"}]
    )

    assert seen["api_base"] == "http://litellm.svc:4000/v1"
    assert seen["model"] == "openai/gemini-3.5-flash", (
        "acompletion must send the normalized id to LiteLLM"
    )
