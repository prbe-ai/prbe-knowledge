"""Unit tests for GeminiAgentClient — request-shape correctness.

The v4 wiki agent halted at turn 0 every run because
`generate_with_cache` set BOTH `cached_content=...` AND `tools=...`
on `GenerateContentConfig`. Gemini rejects that combination with a
400 ("CachedContent can not be used with GenerateContent request
setting system_instruction, tools or tool_config"); the SDK
surfaces the rejection as a ValueError, tenacity exhausts after
three attempts, and the harness halts with
`agent.gemini_persistent_error`.

These tests pin the contract so the regression can't recur:

  - With cache_name set: `tools` MUST be absent from the per-call
    config.
  - Without cache_name: `tools` MUST be present (we pay full input
    cost AND have to ship tool defs every call).
  - Empty `contents` on turn 0: replaced with a non-empty nudge so
    Gemini doesn't reject the request.
  - Schemas with `additionalProperties` / `$ref` get sanitized
    before being handed to FunctionDeclaration.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Fake google.genai SDK
# ---------------------------------------------------------------------------
#
# The production module imports `from google import genai` and
# `from google.genai.types import ...` lazily inside its methods, so
# we install a stub `google.genai` package in sys.modules before the
# class is exercised. The fake records every call so the tests can
# assert on request shape.


class _FakeFunctionDeclaration:
    def __init__(self, *, name, description=None, parameters=None) -> None:
        self.name = name
        self.description = description
        self.parameters = parameters


class _FakeTool:
    def __init__(self, *, function_declarations) -> None:
        self.function_declarations = list(function_declarations)


class _FakeAutomaticFunctionCallingConfig:
    def __init__(self, *, disable: bool = False) -> None:
        self.disable = disable


class _FakeGenerateContentConfig:
    def __init__(
        self,
        *,
        cached_content=None,
        tools=None,
        system_instruction=None,
        automatic_function_calling=None,
    ) -> None:
        # Accept the same kwargs the real type does. Storing them so
        # the test can assert on what actually went into the per-call
        # config — the bug was passing tools alongside cached_content,
        # PLUS leaving AFC at its default (on) which broke turn 2+ of
        # multi-turn agent runs.
        self.cached_content = cached_content
        self.tools = tools
        self.system_instruction = system_instruction
        self.automatic_function_calling = automatic_function_calling


class _FakeCreateCachedContentConfig:
    def __init__(self, *, contents=None, system_instruction=None, tools=None, ttl=None) -> None:
        self.contents = contents
        self.system_instruction = system_instruction
        self.tools = tools
        self.ttl = ttl


def _install_fake_genai(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install a fake `google.genai` package in sys.modules.

    Returns a SimpleNamespace with handles into the fake so tests
    can inspect the recorded calls.
    """
    recorder: SimpleNamespace = SimpleNamespace(
        generate_calls=[],
        cache_calls=[],
        cache_response=SimpleNamespace(name="cachedContents/fake-1"),
        generate_response=SimpleNamespace(
            text="hello",
            candidates=[],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                cached_content_token_count=900,
                candidates_token_count=20,
            ),
        ),
    )

    async def _generate_content(*, model, contents, config):
        recorder.generate_calls.append(
            {"model": model, "contents": contents, "config": config}
        )
        return recorder.generate_response

    async def _create_cache(*, model, config):
        recorder.cache_calls.append({"model": model, "config": config})
        return recorder.cache_response

    fake_aio_models = SimpleNamespace(generate_content=_generate_content)
    fake_aio_caches = SimpleNamespace(create=_create_cache)
    fake_aio = SimpleNamespace(models=fake_aio_models, caches=fake_aio_caches)
    fake_client = SimpleNamespace(aio=fake_aio)

    fake_genai_module = ModuleType("google.genai")
    fake_genai_module.Client = lambda **kwargs: fake_client  # type: ignore[attr-defined]

    fake_types_module = ModuleType("google.genai.types")
    fake_types_module.FunctionDeclaration = _FakeFunctionDeclaration  # type: ignore[attr-defined]
    fake_types_module.Tool = _FakeTool  # type: ignore[attr-defined]
    fake_types_module.GenerateContentConfig = _FakeGenerateContentConfig  # type: ignore[attr-defined]
    fake_types_module.CreateCachedContentConfig = _FakeCreateCachedContentConfig  # type: ignore[attr-defined]
    fake_types_module.AutomaticFunctionCallingConfig = _FakeAutomaticFunctionCallingConfig  # type: ignore[attr-defined]

    fake_google_module = ModuleType("google")
    fake_google_module.genai = fake_genai_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "google", fake_google_module)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types_module)
    return recorder


@pytest.fixture()
def fake_genai(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    return _install_fake_genai(monkeypatch)


@pytest.fixture()
def patched_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `get_settings` so `_ensure_client` finds an API key."""
    import services.synthesis.gemini_agent_client as mod

    fake_settings = SimpleNamespace(
        google_api_key=SimpleNamespace(get_secret_value=lambda: "fake-key")
    )
    monkeypatch.setattr(mod, "get_settings", lambda: fake_settings)


def _tool(
    name: str,
    *,
    description: str = "desc",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "parameters": parameters or {"type": "object", "properties": {}},
    }


# ---------------------------------------------------------------------------
# generate_with_cache: the bug-fix tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_with_cache_omits_tools_when_cache_set(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Regression for `agent.gemini_persistent_error` at turn 0.

    When cache_name is set, the per-call GenerateContentConfig must
    NOT carry tools (Gemini 400s the request otherwise).
    """
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    await client.generate_with_cache(
        cache_name="cachedContents/abc",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tools=[_tool("done")],
    )
    assert len(fake_genai.generate_calls) == 1
    config = fake_genai.generate_calls[0]["config"]
    assert config.cached_content == "cachedContents/abc"
    assert config.tools is None, (
        "tools must NOT be set when cached_content is set — Gemini rejects the "
        "combination with 400; that's the bug we're fixing."
    )
    assert config.system_instruction is None
    # AFC must be disabled — see test_afc_disabled_on_cache_path for why.
    assert config.automatic_function_calling is not None
    assert config.automatic_function_calling.disable is True


@pytest.mark.asyncio
async def test_afc_disabled_on_cache_path(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Regression for `agent.gemini_persistent_error` at turn 1+.

    The google-genai SDK defaults Automatic Function Calling ON
    whenever a request mentions functions — INCLUDING via
    cached_content. Our agent harness does manual dispatch (receive
    function_call -> run tool -> send function_response back). With
    AFC engaged, turn 2's request shape collides with the SDK's
    auto-handling and Gemini 400s every retry. Symptom in production:

        turn=1: HTTP 200 OK   (model returns function_call)
        turn=2: HTTP 400      (3x tenacity retries all 400)
        agent.halt reason=gemini_persistent_error turns=1

    The fix: pass AutomaticFunctionCallingConfig(disable=True) on
    EVERY generate_content call. This test pins the config carries
    that knob with disable=True.
    """
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    await client.generate_with_cache(
        cache_name="cachedContents/abc",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tools=[_tool("done")],
    )
    config = fake_genai.generate_calls[0]["config"]
    assert config.automatic_function_calling is not None, (
        "AFC config must be explicitly set; SDK default is enabled."
    )
    assert config.automatic_function_calling.disable is True


@pytest.mark.asyncio
async def test_afc_disabled_on_cache_miss_fallback(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """AFC must be disabled on the cache-miss fallback path too.

    When cache creation fails, the harness re-attaches tools+system
    on every call. AFC would still default to ON for those calls,
    re-introducing the same multi-turn 400 failure. Disable must
    apply to BOTH config branches.
    """
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    await client.generate_with_cache(
        cache_name="",  # cache miss
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tools=[_tool("done")],
    )
    config = fake_genai.generate_calls[0]["config"]
    assert config.tools is not None  # cache-miss path SHOULD attach tools
    assert config.automatic_function_calling is not None
    assert config.automatic_function_calling.disable is True


@pytest.mark.asyncio
async def test_generate_with_cache_attaches_tools_when_no_cache(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Cache-miss fallback path: tools must be on every call."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    await client.generate_with_cache(
        cache_name="",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tools=[_tool("done"), _tool("update_page")],
    )
    config = fake_genai.generate_calls[0]["config"]
    assert config.cached_content is None
    assert config.tools is not None
    assert len(config.tools) == 1
    fdecls = config.tools[0].function_declarations
    assert [d.name for d in fdecls] == ["done", "update_page"]


@pytest.mark.asyncio
async def test_generate_with_cache_empty_contents_uses_nudge(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Turn 0 has empty conversation tail; we must inject a non-empty
    user message so Gemini doesn't reject the request."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    await client.generate_with_cache(
        cache_name="cachedContents/abc",
        contents=[],
        tools=[_tool("done")],
    )
    contents = fake_genai.generate_calls[0]["contents"]
    assert len(contents) == 1
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"]  # non-empty


@pytest.mark.asyncio
async def test_generate_with_cache_passes_through_non_empty_contents(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Non-empty `contents` is sent as-is — no nudge prepended."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    msgs = [
        {"role": "model", "parts": [{"function_call": {"name": "done", "args": {}}}]},
        {"role": "user", "parts": [{"function_response": {"name": "done", "response": {}}}]},
    ]
    await client.generate_with_cache(
        cache_name="cachedContents/abc",
        contents=msgs,
        tools=[_tool("done")],
    )
    contents = fake_genai.generate_calls[0]["contents"]
    assert contents == msgs


@pytest.mark.asyncio
async def test_generate_with_cache_returns_normalized_dict(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Response shape: text + tool_calls + usage_metadata, all flat dicts."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    fake_genai.generate_response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="done", args={"why": "ok"}
                            )
                        )
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            cached_content_token_count=900,
            candidates_token_count=20,
        ),
    )
    client = GeminiAgentClient()
    out = await client.generate_with_cache(
        cache_name="cachedContents/abc",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tools=[_tool("done")],
    )
    assert out["text"] is None
    # Newer code carries thought_signature (None when SDK omitted it).
    assert out["tool_calls"] == [
        {"name": "done", "args": {"why": "ok"}, "thought_signature": None}
    ]
    assert out["usage_metadata"]["cached_content_token_count"] == 900


@pytest.mark.asyncio
async def test_thought_signature_extracted_from_part(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Regression for `agent.gemini_persistent_error` at turn 1+ on
    Gemini 3.x.

    The Gemini 3.x API emits an opaque `thought_signature` (bytes)
    on every part that holds a function_call. When the harness
    echoes that function_call back in the conversation history on
    turn 2+, the SAME signature must accompany it. Otherwise:

        400 INVALID_ARGUMENT: 'Function call is missing a
        thought_signature in functionCall parts. This is required
        for tools to work correctly...'

    `_extract_response` must capture the signature off the SDK's
    Part so the harness can round-trip it.
    """
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    sig_bytes = b"opaque-sdk-bytes-for-this-call"
    fake_genai.generate_response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="next_events", args={"count": 50}
                            ),
                            thought_signature=sig_bytes,
                        )
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=1, cached_content_token_count=0, candidates_token_count=1
        ),
    )
    client = GeminiAgentClient()
    out = await client.generate_with_cache(
        cache_name="cachedContents/abc",
        contents=[{"role": "user", "parts": [{"text": "hi"}]}],
        tools=[_tool("next_events")],
    )
    assert out["tool_calls"][0]["thought_signature"] == sig_bytes


# ---------------------------------------------------------------------------
# create_cache: tools/system_instruction live here, schemas sanitized
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_cache_attaches_tools_and_system_instruction(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Tools + system_instruction belong on the cache, not on per-call."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    name = await client.create_cache(
        system_instruction="you are an agent",
        tools=[_tool("done")],
        seed_contents=[{"role": "user", "parts": [{"text": "seed"}]}],
    )
    assert name == "cachedContents/fake-1"
    config = fake_genai.cache_calls[0]["config"]
    assert config.system_instruction == "you are an agent"
    assert config.tools is not None
    assert config.tools[0].function_declarations[0].name == "done"


@pytest.mark.asyncio
async def test_create_cache_strips_unsupported_schema_keys(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """`additionalProperties` and `$ref` would 400 the schema validator;
    sanitize them out before handing to FunctionDeclaration."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "x": {"type": "string"},
            "y": {"$ref": "#/components/schemas/Y"},
        },
    }
    client = GeminiAgentClient()
    await client.create_cache(
        system_instruction="sys",
        tools=[_tool("update_page", parameters=schema)],
        seed_contents=[{"role": "user", "parts": [{"text": "seed"}]}],
    )
    fdecls = fake_genai.cache_calls[0]["config"].tools[0].function_declarations
    sanitized = fdecls[0].parameters
    assert "additionalProperties" not in sanitized
    # $ref nested under properties.y — should also be gone.
    assert "$ref" not in sanitized["properties"]["y"]


@pytest.mark.asyncio
async def test_create_cache_empty_parameters_becomes_object_schema(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Tool with empty/None parameters produces a valid object schema."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    await client.create_cache(
        system_instruction="sys",
        tools=[{"name": "done", "description": "stop"}],
        seed_contents=[{"role": "user", "parts": [{"text": "seed"}]}],
    )
    fdecls = fake_genai.cache_calls[0]["config"].tools[0].function_declarations
    assert fdecls[0].parameters == {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# End-to-end flow: full first-turn shape (the production failure)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_first_turn_request_shape_is_well_formed(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Reproduce the v4 first-turn flow: create_cache, then
    generate_with_cache with empty contents (turn 0). Verify the
    final per-call config has cached_content, no tools, no
    system_instruction, and non-empty contents."""
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()
    cache_name = await client.create_cache(
        system_instruction="sys prompt",
        tools=[
            _tool("done"),
            _tool("update_page", parameters={
                "type": "object",
                "properties": {"slug": {"type": "string"}},
                "required": ["slug"],
            }),
        ],
        seed_contents=[{"role": "user", "parts": [{"text": "seed manifest"}]}],
    )
    await client.generate_with_cache(
        cache_name=cache_name,
        contents=[],
        tools=[_tool("done"), _tool("update_page")],
    )
    call = fake_genai.generate_calls[0]
    assert call["config"].cached_content == "cachedContents/fake-1"
    assert call["config"].tools is None
    assert call["config"].system_instruction is None
    assert len(call["contents"]) == 1
    assert call["contents"][0]["role"] == "user"
    assert call["contents"][0]["parts"][0]["text"]


# ---------------------------------------------------------------------------
# _sanitize_parameters / _strip_keys_recursive unit tests
# ---------------------------------------------------------------------------


def test_sanitize_parameters_handles_none() -> None:
    from services.synthesis.gemini_agent_client import _sanitize_parameters

    assert _sanitize_parameters(None) == {"type": "object", "properties": {}}
    assert _sanitize_parameters({}) == {"type": "object", "properties": {}}


def test_sanitize_parameters_preserves_supported_keys() -> None:
    """default, minimum, maximum, enum, required — all kept."""
    from services.synthesis.gemini_agent_client import _sanitize_parameters

    schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "minimum": 1,
                "maximum": 500,
                "default": 200,
            },
            "kind": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["kind"],
    }
    out = _sanitize_parameters(schema)
    assert out["properties"]["count"]["minimum"] == 1
    assert out["properties"]["count"]["maximum"] == 500
    assert out["properties"]["count"]["default"] == 200
    assert out["properties"]["kind"]["enum"] == ["a", "b"]
    assert out["required"] == ["kind"]


def test_sanitize_parameters_strips_rejected_keys_recursively() -> None:
    from services.synthesis.gemini_agent_client import _sanitize_parameters

    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "outer": {
                "type": "object",
                "additionalProperties": True,
                "properties": {"inner": {"$ref": "#/foo"}},
            }
        },
    }
    out = _sanitize_parameters(schema)
    assert "$schema" not in out
    assert "additionalProperties" not in out
    assert "additionalProperties" not in out["properties"]["outer"]
    assert "$ref" not in out["properties"]["outer"]["properties"]["inner"]


# ---------------------------------------------------------------------------
# Compatibility: the harness' AgentLoop must run end-to-end against
# this client (smoke test using AsyncMock to skip network).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_is_loop_compatible(
    fake_genai: SimpleNamespace, patched_settings: None
) -> None:
    """Surface check: AgentLoop.run() drives create_cache + generate_with_cache."""
    from services.synthesis.agent_harness import AgentLoop
    from services.synthesis.gemini_agent_client import GeminiAgentClient

    # Make the model return a `done` tool call so the loop terminates.
    fake_genai.generate_response = SimpleNamespace(
        text=None,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(name="done", args={})
                        )
                    ]
                )
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            cached_content_token_count=900,
            candidates_token_count=20,
        ),
    )

    class _StubRuntime:
        customer_id = "cust"
        agent_run_id = "run-1"
        is_done = False
        pending_update_count = 0

        async def dispatch_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
            if name == "done":
                self.is_done = True
            return {"committed": True}

        def state_snapshot_for_summary(self) -> dict[str, Any]:
            return {"pending_updates": [], "pending_creates": [], "applied_queue_ids": [], "skipped_queue_ids": []}

        async def initial_manifest(self, count: int) -> dict[str, Any]:
            return {"events": [], "remaining": 0, "drain_complete": True}

        async def wiki_index(self) -> list[dict[str, Any]]:
            return []

    runtime = _StubRuntime()
    llm = GeminiAgentClient()
    loop = AgentLoop(
        runtime=runtime,
        llm=llm,
        system_prompt="sys",
        tool_schemas=[_tool("done")],
    )
    metrics = await loop.run()
    assert metrics.turns == 1
    # Turn 0 generate call: cached_content set, tools NOT set.
    cfg = fake_genai.generate_calls[0]["config"]
    assert cfg.cached_content == "cachedContents/fake-1"
    assert cfg.tools is None


# ---------------------------------------------------------------------------
# Phase-0b carve-out: gateway-mode gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_gateway_mode_blocks_client_construction(
    fake_genai: SimpleNamespace,
    patched_settings: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In LLM_GATEWAY_URL (gateway-routed) mode the
    GeminiAgentClient must refuse to make a real call. LiteLLM doesn't
    expose Gemini CachedContent or thought_signature round-tripping, so
    routing this call site through the proxy is not viable today —
    surface the carve-out loudly rather than emit a confusing
    "GOOGLE_API_KEY not configured" that suggests a fixable config gap.

    Construction stays cheap; the gate fires on first `_ensure_client`
    (i.e. when create_cache / generate_with_cache is actually invoked).
    """
    monkeypatch.setenv("LLM_GATEWAY_URL", "https://litellm.example/v1")

    from services.synthesis.gemini_agent_client import GeminiAgentClient

    client = GeminiAgentClient()  # construction itself is fine
    with pytest.raises(RuntimeError) as exc_info:
        await client.create_cache(
            system_instruction="sys",
            tools=[_tool("done")],
            seed_contents=[],
        )
    msg = str(exc_info.value)
    assert "LLM_GATEWAY_URL" in msg
    assert "CachedContent" in msg or "carve-out" in msg.lower()
