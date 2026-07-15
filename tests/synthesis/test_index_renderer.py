"""Unit tests for the wiki index LLM renderer.

Covers the Phase-0b chunk B migration: `render_index_via_llm` now
routes the production path through `shared.llm.acompletion` so the call
honors `LLM_GATEWAY_URL` for gateway routing. The deterministic
fallback (no provider key AND no gateway → flat alphabetical list) and
the legacy injected-client path are preserved for backward compat.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from kb.synthesis.index_renderer import (
    _fallback_flat_list,
    _PageRow,
    render_index_via_llm,
)


def _row(source_id: str, title: str, summary: str = "", body_preview: str = "") -> dict:
    """Build a fake asyncpg.Record-like dict the renderer accepts.

    The renderer indexes rows by key (`row["title"]`, `row["metadata"]`,
    etc.); plain dicts work in tests.
    """
    return {
        "source_id": source_id,
        "title": title,
        "metadata": {"summary": summary} if summary else {},
        "body_preview": body_preview,
    }


def _stub_acompletion_response(text: str):
    """Build a litellm-shaped ChatCompletion response stub."""
    choice = SimpleNamespace(message=SimpleNamespace(content=text))
    return SimpleNamespace(choices=[choice])


@pytest.mark.asyncio
async def test_render_index_falls_back_when_no_pages() -> None:
    """Empty corpus returns the canned 'No pages yet.' placeholder; no
    LLM call needed."""
    out = await render_index_via_llm([], client=None)
    assert out == "# Wiki\n\nNo pages yet.\n"


@pytest.mark.asyncio
async def test_render_index_falls_back_without_key_or_gateway(monkeypatch) -> None:
    """No GOOGLE_API_KEY and no LLM_GATEWAY_URL → deterministic flat
    list (the index page must always render)."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    out = await render_index_via_llm(rows, client=None)
    assert "Title A" in out
    # Fallback is alphabetical bullets, not an LLM-shaped paragraph.
    assert out.startswith("# Wiki")


@pytest.mark.asyncio
async def test_render_index_via_shared_llm_happy_path(monkeypatch) -> None:
    """With GOOGLE_API_KEY set, the renderer calls
    `shared.llm.acompletion` (model id `gemini/<WIKI_AGENT_MODEL>`,
    system + user messages) and returns its content."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    captured: dict = {}

    async def fake_acompletion(*, model, messages, **kwargs):
        captured["model"] = model
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return _stub_acompletion_response(
            "# Probe\n\nIntro paragraph.\n\n## Pages\n\n- [[Title A]] — Summary A\n"
        )

    from engine.shared import llm as shared_llm

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    out = await render_index_via_llm(rows, client=None)

    from engine.shared.constants import WIKI_AGENT_MODEL

    assert captured["model"] == f"gemini/{WIKI_AGENT_MODEL}"
    assert [m["role"] for m in captured["messages"]] == ["system", "user"]
    assert "engineering wiki" in captured["messages"][0]["content"].lower()
    assert captured["kwargs"].get("max_tokens") == 16384
    assert "Title A" in out
    # The renderer trims the leading `# Wiki` heading; verify it didn't
    # touch the company H1.
    assert "# Probe" in out


@pytest.mark.asyncio
async def test_render_index_uses_gateway_without_google_key(monkeypatch) -> None:
    """Gateway-routed tenant: only LLM_GATEWAY_URL is set. The
    renderer must still call shared.llm.acompletion (the wrapper
    handles api_base/api_key injection)."""
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_GATEWAY_URL", "http://litellm.litellm.svc.cluster.local:4000")
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    async def fake_acompletion(*, model, messages, **kwargs):
        return _stub_acompletion_response("# Probe\n\nGateway intro.\n")

    from engine.shared import llm as shared_llm

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    out = await render_index_via_llm(rows, client=None)
    assert "# Probe" in out


@pytest.mark.asyncio
async def test_render_index_falls_back_on_llm_error(monkeypatch) -> None:
    """An LLMError from the wrapper should NOT crash the index render;
    the function falls back to the flat list."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    from engine.shared import llm as shared_llm

    async def fake_acompletion(*, model, messages, **kwargs):
        raise shared_llm.LLMError("upstream timeout", status_code=504, provider="google")

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    out = await render_index_via_llm(rows, client=None)
    # Fallback path → markdown bullets with the page title.
    assert "Title A" in out
    assert out.startswith("# Wiki")


@pytest.mark.asyncio
async def test_render_index_falls_back_on_empty_response(monkeypatch) -> None:
    """Empty LLM content → flat-list fallback."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    from engine.shared import llm as shared_llm

    async def fake_acompletion(*, model, messages, **kwargs):
        return _stub_acompletion_response("")

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    out = await render_index_via_llm(rows, client=None)
    assert "Title A" in out
    assert out.startswith("# Wiki")


@pytest.mark.asyncio
async def test_render_index_legacy_client_path_still_works() -> None:
    """A test-injected google-genai-shaped client bypasses the wrapper.
    The legacy path is preserved so existing fixtures keep working."""
    client = SimpleNamespace()
    client.aio = SimpleNamespace()
    client.aio.models = SimpleNamespace()
    client.aio.models.generate_content = AsyncMock(
        return_value=SimpleNamespace(text="# Probe\n\nLegacy intro.\n")
    )

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    out = await render_index_via_llm(rows, client=client)
    assert "# Probe" in out


def test_fallback_flat_list_renders_pages_alphabetically() -> None:
    """Sanity: the deterministic fallback emits the title list in
    alphabetical order."""
    pages = [
        _PageRow(wiki_type="decision", slug="b", title="Bravo", summary="b summary"),
        _PageRow(wiki_type="decision", slug="a", title="Alpha", summary="a summary"),
    ]
    out = _fallback_flat_list(pages)
    a_pos = out.index("Alpha")
    b_pos = out.index("Bravo")
    assert a_pos < b_pos
