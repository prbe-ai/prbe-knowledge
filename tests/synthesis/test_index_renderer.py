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
    _format_pages_for_prompt,
    _PageRow,
    render_index_via_llm,
)


def _row(
    source_id: str,
    title: str,
    summary: str = "",
    body_preview: str = "",
    body: str | None = None,
) -> dict:
    """Build a fake asyncpg.Record-like dict the renderer accepts.

    The renderer indexes rows by key (`row["title"]`, `row["metadata"]`,
    etc.); plain dicts work in tests. `body` is omitted entirely when not
    given, which is how an older caller's rows arrive -- the renderer must
    tolerate the key being absent rather than empty.
    """
    row = {
        "source_id": source_id,
        "title": title,
        "metadata": {"summary": summary} if summary else {},
        "body_preview": body_preview,
    }
    if body is not None:
        row["body"] = body
    return row


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
async def test_render_index_declines_without_key_or_gateway(monkeypatch) -> None:
    """No GOOGLE_API_KEY and no LLM_GATEWAY_URL → None, meaning "leave the
    published page alone".

    This used to return a flat alphabetical list of every page. Under the
    overview design that substitute IS the directory the front page stopped
    being, so a single unavailable LLM would silently undo the change.
    """
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    assert await render_index_via_llm(rows, client=None) is None


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
    assert captured["kwargs"].get("max_tokens") == 4096
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
async def test_render_index_declines_on_llm_error(monkeypatch) -> None:
    """An LLMError must not crash the render -- and must not substitute a
    page list either. `None` tells the caller to keep the published page."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    from engine.shared import llm as shared_llm

    async def fake_acompletion(*, model, messages, **kwargs):
        raise shared_llm.LLMError("upstream timeout", status_code=504, provider="google")

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    assert await render_index_via_llm(rows, client=None) is None


@pytest.mark.asyncio
async def test_render_index_declines_on_empty_response(monkeypatch) -> None:
    """Empty LLM content is a failed render, not an empty overview."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("LLM_GATEWAY_URL", raising=False)
    from engine.shared.config import get_settings

    get_settings.cache_clear()

    from engine.shared import llm as shared_llm

    async def fake_acompletion(*, model, messages, **kwargs):
        return _stub_acompletion_response("")

    monkeypatch.setattr(shared_llm, "acompletion", fake_acompletion)

    rows = [_row("decision:abc", "Title A", summary="Summary A")]
    assert await render_index_via_llm(rows, client=None) is None


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


def test_the_prompt_carries_page_text_not_just_titles() -> None:
    """THE POINT OF THE CHANGE. The renderer used to send titles and one-line
    summaries only, so the model writing the overview had never read a page it
    was describing -- it could re-group the list it was handed and nothing
    more. The bodies are what make an overview possible."""
    pages = [
        _PageRow(
            wiki_type="decision",
            slug="a",
            title="Alpha",
            summary="a summary",
            body="We chose pgvector because the alternative needed a second datastore.",
        ),
    ]
    corpus, stats = _format_pages_for_prompt(pages)

    assert "needed a second datastore" in corpus
    assert stats.missing_body == 0


def test_pages_with_no_body_are_counted_not_hidden() -> None:
    """A front page that quietly stopped seeing half the wiki would read as
    the wiki having gotten less interesting. The count is what makes that
    visible in the logs instead."""
    pages = [
        _PageRow(wiki_type="decision", slug="a", title="Alpha", summary="s", body=""),
        _PageRow(wiki_type="decision", slug="b", title="Bravo", summary="s", body="text"),
    ]
    _, stats = _format_pages_for_prompt(pages)
    assert stats.missing_body == 1


def test_a_long_body_is_trimmed_and_marked() -> None:
    """Trimming is signposted in the prompt so the model knows it is reading
    an excerpt, rather than treating a sentence cut mid-clause as the page's
    final word."""
    from kb.synthesis.index_renderer import PER_PAGE_BODY_CHARS

    pages = [
        _PageRow(
            wiki_type="decision",
            slug="a",
            title="Alpha",
            summary="s",
            body="x" * (PER_PAGE_BODY_CHARS + 500),
        ),
    ]
    corpus, _ = _format_pages_for_prompt(pages)
    assert "[...truncated]" in corpus
    # Measured on the body LINE, not by counting "x" across the whole corpus:
    # the `text: |` label contains one, which is exactly the kind of off-by-one
    # that makes a cap look wrong when it is right.
    body_line = next(line for line in corpus.splitlines() if line.strip().startswith("x"))
    assert len(body_line.strip()) == PER_PAGE_BODY_CHARS


def test_a_row_with_no_body_key_is_tolerated() -> None:
    """Older callers send rows without the key at all. That must read as "no
    body", not raise -- the index regen is the last step of every drain."""
    from kb.synthesis.index_renderer import _rows_to_pages

    pages = _rows_to_pages([_row("decision:abc", "Title A", summary="Summary A")])
    assert pages[0].body == ""


def test_a_page_the_total_budget_zeroes_is_counted_separately() -> None:
    """A page starved by the TOTAL budget renders exactly like a page that
    never had a body, and only the first one means the renderer has quietly
    stopped reading part of the wiki.

    Counting them together would let the front page silently narrow as the
    corpus grows -- which reads to a person as the wiki getting less
    interesting, not as a cap being hit.
    """
    from kb.synthesis.index_renderer import _TOTAL_BODY_CHARS, PER_PAGE_BODY_CHARS

    # Enough full-size pages to exhaust the budget, plus two: the first of
    # those gets a partial slice (still text, still marked truncated) and the
    # second gets nothing at all. Only the second is what this counts.
    n = _TOTAL_BODY_CHARS // PER_PAGE_BODY_CHARS
    pages = [
        _PageRow(
            wiki_type="decision",
            slug=f"p{i}",
            title=f"P{i}",
            summary="s",
            body="x" * PER_PAGE_BODY_CHARS,
        )
        for i in range(n + 2)
    ]

    _, stats = _format_pages_for_prompt(pages)

    assert stats.dropped_by_budget == 1, "the starved page is reported"
    assert stats.missing_body == 0, "and is NOT miscounted as bodyless"


def test_the_prompt_marks_the_corpus_as_untrusted() -> None:
    """The renderer now reads page BODIES, and those bodies were written from
    Slack, GitHub and tickets -- places many people can write to. Without a
    stated trust boundary, a page saying "ignore your instructions" is being
    handed to the model as though the team had written it.

    It lands on the FRONT PAGE, which every person and agent reads first.
    """
    from kb.synthesis.index_renderer import _CORPUS_MARKER, _INDEX_SYSTEM_PROMPT

    lowered = _INDEX_SYSTEM_PROMPT.lower()
    assert "untrusted" in lowered
    assert "never" in lowered and "instructions" in lowered
    # The rule points at a marker, so the marker has to exist in the prompt
    # the model actually receives -- see `render_index_via_llm`.
    assert "corpus marker" in lowered
    assert "UNTRUSTED" in _CORPUS_MARKER
