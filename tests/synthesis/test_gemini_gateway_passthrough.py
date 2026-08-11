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

import pytest

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


@pytest.mark.asyncio
async def test_triage_sends_openai_structured_output_in_gateway_mode(monkeypatch) -> None:
    """Gateway mode must ask for RAW json, not Gemini's native mime type.

    The proxy's /chat/completions route silently drops `response_mime_type`, so
    the model returns markdown-FENCED JSON and every verdict parses as missing.
    Nothing errors -- which is why this needs a test rather than a runbook.
    """
    import kb.synthesis.providers as prov

    seen: dict[str, object] = {}

    class _Captured(Exception):
        """Named so the assertion below cannot pass on an unrelated failure.

        `pytest.raises(Exception)` would swallow a TypeError from a changed
        signature and still let the response_format assertions run against a
        half-populated dict.
        """

    async def fake_acompletion(**kwargs):
        seen.update(kwargs)
        raise _Captured

    monkeypatch.setattr("engine.shared.llm.acompletion", fake_acompletion)
    monkeypatch.setattr("engine.shared.llm.gateway_url", lambda: GATEWAY)

    with pytest.raises(_Captured):
        await prov._gemini_call_json(
            model="gemini-3.5-flash",
            system="sys",
            user="score this",
            schema={"type": "object", "properties": {}},
            max_tokens=8000,
        )

    # json_schema, NOT json_object. Both unfence the JSON, but json_object
    # constrains only "some JSON" and Gemini answers with a top-level ARRAY,
    # which the parser rejects as "not a JSON object: list". The schema has to
    # ride along for the shape contract to match the native path.
    rf = seen.get("response_format")
    assert rf is not None and rf["type"] == "json_schema", rf
    assert rf["json_schema"]["schema"] == {"type": "object", "properties": {}}
    assert "response_mime_type" not in seen, "native mime type is dropped by the proxy"
    assert "response_schema" not in seen


def test_map_schema_becomes_an_array_for_the_openai_wire() -> None:
    """Triage's schema is a MAP keyed by queue_id; json_schema cannot express it.

    `additionalProperties` carries no generation semantics in OpenAI's
    json_schema, so the model emits `{}` -- which VALIDATES while containing
    nothing, and every batch reports "no verdict" with no error anywhere.
    Observed as `RAW RESULT: {"verdicts": {}}` in production.
    """
    from kb.synthesis.providers import _map_schema_to_array

    schema = {
        "type": "object",
        "properties": {
            "verdicts": {
                "type": "object",
                "description": "Map keyed by queue_id",
                "additionalProperties": {
                    "type": "object",
                    "properties": {"score": {"type": "number"}},
                    "required": ["score"],
                },
            }
        },
    }

    wire, converted = _map_schema_to_array(schema)

    assert converted == ["verdicts"]
    assert wire["properties"]["verdicts"]["type"] == "array"
    item = wire["properties"]["verdicts"]["items"]
    assert item["properties"]["_key"] == {"type": "string"}
    assert set(item["required"]) == {"_key", "score"}
    # The map's own description must survive -- it is what tells the model to
    # emit one entry per input event.
    assert wire["properties"]["verdicts"]["description"] == "Map keyed by queue_id"


def test_array_response_is_rekeyed_back_into_the_map() -> None:
    """The caller must see the SAME shape on both routes."""
    from kb.synthesis.providers import _array_to_map

    payload = {
        "verdicts": [
            {"_key": "1", "score": 8, "important": True},
            {"_key": "2", "score": 2, "important": False},
        ]
    }

    assert _array_to_map(payload, ["verdicts"]) == {
        "verdicts": {
            "1": {"score": 8, "important": True},
            "2": {"score": 2, "important": False},
        }
    }


def test_rekeying_is_a_no_op_on_the_native_route() -> None:
    """Direct-provider mode converts nothing, so nothing may be rewritten."""
    from kb.synthesis.providers import _array_to_map

    native = {"verdicts": {"1": {"score": 8}}}
    assert _array_to_map(native, []) == native


@pytest.mark.asyncio
async def test_gemini_call_json_converts_the_map_and_rekeys_the_response(
    monkeypatch,
) -> None:
    """The WIRING, end to end, not the two helpers in isolation.

    Written after both helper tests above passed against a build where
    `_map_schema_to_array` was never called AND where `_array_to_map` was never
    applied -- i.e. against the exact production bug, twice. A pure-function
    test cannot fail when its caller stops calling it, so this asserts what goes
    onto the wire and what comes back out.
    """
    import kb.synthesis.providers as prov

    sent: dict[str, object] = {}

    class _Msg:
        content = '{"verdicts":[{"_key":"7","score":9,"important":true}]}'

    class _Choice:
        message = _Msg()

    class _Resp:
        choices = [_Choice()]

    async def fake_acompletion(**kwargs):
        sent.update(kwargs)
        return _Resp()

    monkeypatch.setattr("engine.shared.llm.acompletion", fake_acompletion)
    monkeypatch.setattr("engine.shared.llm.gateway_url", lambda: GATEWAY)

    result = await prov._gemini_call_json(
        model="gemini-3.5-flash",
        system="sys",
        user="score",
        schema={
            "type": "object",
            "properties": {
                "verdicts": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"score": {"type": "number"}},
                    },
                }
            },
        },
        max_tokens=8000,
    )

    wire = sent["response_format"]["json_schema"]["schema"]
    assert wire["properties"]["verdicts"]["type"] == "array", "map must go out as an array"
    assert result == {"verdicts": {"7": {"score": 9, "important": True}}}, (
        "response must come back re-keyed as the map the parser expects"
    )
