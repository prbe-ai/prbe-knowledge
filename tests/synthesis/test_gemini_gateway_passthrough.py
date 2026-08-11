"""Gemini in gateway-routed mode: chat-completions for triage, passthrough for the agent.

Both fixes here undo the same wrong assumption -- that a LiteLLM gateway cannot
carry Gemini. It can, but the two call sites need OPPOSITE routes:

  * triage sends plain structured-output completions, which normalize to OpenAI
    shape cleanly -> `/chat/completions` with an `openai/` prefix;
  * the agent loop needs Gemini-NATIVE shape (CachedContent, thought_signature)
    -> the `/gemini/*` passthrough, which forwards requests untouched.

Sending either down the other's route fails, and both failures were observed in
production on 2026-08-11 (13,147 queue rows dead-lettered by the first).
"""

from __future__ import annotations

from kb.synthesis.providers import _gemini_litellm_model

GATEWAY = "http://litellm.svc:4000/v1"


def test_triage_uses_the_openai_route_when_a_gateway_is_set(monkeypatch) -> None:
    """`gemini/` selects Google's NATIVE transport, which then ignores the proxy
    and dies on Application Default Credentials inside the pod."""
    import engine.shared.llm as llm

    monkeypatch.setattr(llm, "gateway_url", lambda: GATEWAY)
    assert _gemini_litellm_model("gemini-3.5-flash") == "openai/gemini-3.5-flash"


def test_triage_keeps_the_gemini_prefix_without_a_gateway(monkeypatch) -> None:
    """Direct-provider mode is the deployment the bare convention was written for."""
    import engine.shared.llm as llm

    monkeypatch.setattr(llm, "gateway_url", lambda: None)
    assert _gemini_litellm_model("gemini-3.5-flash") == "gemini/gemini-3.5-flash"


def test_an_explicit_provider_path_is_never_rewritten(monkeypatch) -> None:
    import engine.shared.llm as llm

    monkeypatch.setattr(llm, "gateway_url", lambda: GATEWAY)
    assert _gemini_litellm_model("vertex_ai/gemini-3.5-flash") == (
        "vertex_ai/gemini-3.5-flash"
    )


def test_agent_client_constructs_against_the_passthrough(monkeypatch) -> None:
    """It must BUILD in gateway mode -- it used to raise on sight of a gateway.

    Asserts the base_url too: pointing google-genai at the OpenAI root (.../v1)
    instead of /gemini would 404 every native call, which is the same class of
    bug as the triage prefix and just as invisible until it runs.
    """
    from kb.synthesis import gemini_agent_client as gac

    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, *, api_key, http_options=None):
            captured["api_key"] = api_key
            captured["base_url"] = (http_options or {}).get("base_url")

    monkeypatch.setattr(gac.shared_llm, "gateway_url", lambda: GATEWAY)
    monkeypatch.setattr(gac.shared_llm, "gateway_key", lambda: "sk-gateway")
    monkeypatch.setitem(
        __import__("sys").modules, "google", type("m", (), {"genai": type("g", (), {"Client": _FakeClient})})()
    )

    client = gac.GeminiAgentClient()
    client._ensure_client()

    assert captured["api_key"] == "sk-gateway"
    assert captured["base_url"] == "http://litellm.svc:4000/gemini", (
        "must strip the OpenAI /v1 root and target the /gemini passthrough"
    )
