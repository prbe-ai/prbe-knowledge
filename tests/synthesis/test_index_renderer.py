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


def test_a_page_that_fits_is_read_in_full() -> None:
    """The whole point of the change that removed PER_PAGE_BODY_CHARS: a page
    under the budget reaches the model entire, not as a prefix.

    This is the regression guard. The old per-page cut was calibrated for pages
    of 4-8k chars, nothing enforced that size, and on this team's wiki it was
    discarding 46% of the corpus while the corpus was 35% of the real budget.
    """
    body = "x" * 30_000
    pages = [
        _PageRow(wiki_type="repo", slug="a", title="Alpha", summary="s", body=body),
    ]
    corpus, stats = _format_pages_for_prompt(pages)

    assert "[...truncated]" not in corpus, "a page inside the budget is not cut"
    assert stats.truncated_by_ceiling == 0
    assert stats.corpus_chars == len(body)
    body_line = next(line for line in corpus.splitlines() if line.strip().startswith("x"))
    assert len(body_line.strip()) == len(body)


def test_a_runaway_page_is_cut_by_the_starvation_ceiling_and_marked() -> None:
    """The guard that replaced the per-page cap. It exists so one enormous page
    cannot spend the whole corpus budget and zero every page behind it -- pages
    arrive newest-first, so a tail-drop would become a tail-wipe.

    Trimming stays signposted, so the model knows it is reading an excerpt
    rather than treating a clause cut mid-sentence as the page's final word.
    """
    from kb.synthesis.index_renderer import PAGE_STARVATION_CEILING

    pages = [
        _PageRow(
            wiki_type="repo",
            slug="a",
            title="Alpha",
            summary="s",
            body="x" * (PAGE_STARVATION_CEILING + 500),
        ),
    ]
    corpus, stats = _format_pages_for_prompt(pages)

    assert "[...truncated]" in corpus
    assert stats.truncated_by_ceiling == 1, "the cut is reported, never silent"
    # Measured on the body LINE, not by counting "x" across the whole corpus:
    # the `text: |` label contains one, which is exactly the kind of off-by-one
    # that makes a cap look wrong when it is right.
    body_line = next(line for line in corpus.splitlines() if line.strip().startswith("x"))
    assert len(body_line.strip()) == PAGE_STARVATION_CEILING


def test_an_over_budget_corpus_is_visible_in_the_stats() -> None:
    """`corpus_chars` is clipped to the budget by construction, so on its own it
    saturates: a wiki at 260k chars and one at 2.6M both report 250,000. This is
    the log's whole job -- seeing the budget coming -- so the raw total is carried
    separately and is the number the percentage is computed from.
    """
    from kb.synthesis.index_renderer import _TOTAL_BODY_CHARS

    page_chars = 30_000
    n = (_TOTAL_BODY_CHARS // page_chars) + 4  # comfortably over
    pages = [
        _PageRow(wiki_type="repo", slug=f"p{i}", title=f"P{i}", summary="s", body="x" * page_chars)
        for i in range(n)
    ]

    _, stats = _format_pages_for_prompt(pages)

    assert stats.corpus_chars == _TOTAL_BODY_CHARS, "emitted text is clipped"
    assert stats.raw_corpus_chars == n * page_chars, "the real size is not"
    assert stats.raw_corpus_chars > stats.budget_chars, "and so an overrun is visible"


def test_each_page_is_counted_under_exactly_one_outcome() -> None:
    """A page cut by the ceiling AND then zeroed by the budget used to increment
    both counters, reporting one page as two problems; a page given a partial
    slice at the budget boundary incremented neither and vanished from the log
    entirely. The docstring promises every kind of trimming is reported, which
    only holds if the branches are exclusive and the partial cut has its own
    counter.
    """
    from kb.synthesis.index_renderer import _TOTAL_BODY_CHARS, PAGE_STARVATION_CEILING

    # Big enough that the ceiling fires on every page and the budget runs out
    # partway through, so all four outcomes appear in one corpus.
    body = "x" * (PAGE_STARVATION_CEILING + 1_000)
    n = (_TOTAL_BODY_CHARS // PAGE_STARVATION_CEILING) + 3
    pages = [
        _PageRow(wiki_type="repo", slug=f"p{i}", title=f"P{i}", summary="s", body=body)
        for i in range(n)
    ]
    pages.append(_PageRow(wiki_type="repo", slug="empty", title="E", summary="s", body=""))

    _, stats = _format_pages_for_prompt(pages)

    accounted = (
        stats.truncated_by_ceiling
        + stats.truncated_by_budget
        + stats.dropped_by_budget
        + stats.missing_body
    )
    assert accounted <= len(pages), "no page is counted twice"
    assert stats.missing_body == 1
    assert stats.dropped_by_budget >= 1, "pages past the budget are reported"
    assert stats.truncated_by_ceiling >= 1, "pages the ceiling cut are reported"


def test_the_starvation_ceiling_is_not_lowered_under_known_page_sizes() -> None:
    """A ratchet on the CONSTANT, and only that. It compares two literals, so it
    cannot notice a real page growing past the ceiling -- if `research_os` reaches
    60k it gets cut and this still passes. That case is covered by the
    `truncated_by_ceiling` counter at runtime, not here.

    What it does catch: someone tuning the ceiling down to a number that would
    start truncating pages this team already has. That was the previous failure
    mode exactly -- a per-page cap of 4,000 chosen for pages of 4-8k, left in
    place while pages grew to 37,540 -- so it is worth a guard even a cheap one.
    """
    from kb.synthesis.index_renderer import PAGE_STARVATION_CEILING

    largest_page_seen_in_production = 37_540
    assert largest_page_seen_in_production < PAGE_STARVATION_CEILING


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
    from kb.synthesis.index_renderer import _TOTAL_BODY_CHARS

    # Enough pages to exhaust the budget, plus two: the first of those gets a
    # partial slice (still text, still marked truncated) and the second gets
    # nothing at all. Only the second is what this counts.
    #
    # Page size is a local constant now rather than the per-page cap, which no
    # longer exists. It must NOT divide the budget evenly: this test needs one
    # page to straddle the boundary (partial slice) and the NEXT one to be
    # zeroed, and an even divisor zeroes two pages instead, making
    # dropped_by_budget 2. 250,000 // 4,000 = 62 pages = 248,000 chars, so page
    # 63 takes the last 2,000 and page 64 is the single starved page below.
    page_chars = 4000
    assert _TOTAL_BODY_CHARS % page_chars != 0, "an even divisor breaks the setup"
    n = _TOTAL_BODY_CHARS // page_chars
    pages = [
        _PageRow(
            wiki_type="decision",
            slug=f"p{i}",
            title=f"P{i}",
            summary="s",
            body="x" * page_chars,
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


def test_the_prompt_demands_resolvable_links() -> None:
    """The front page is the one page every agent is told to read first, so a
    link it cannot follow is the most expensive kind to write.

    The prompt used to ask for `[[Title]]`. That form is unresolvable: `[[research-os]]`
    does not map to the page `repo/research_os` (hyphen vs underscore), and
    `[[Wiki Synthesis Pipeline (prbe-knowledge)]]` -> `feature/prbe_knowledge-wiki_synthesis`
    is not guessable at all. It is also recorded as a DANGLING reference by the
    link parser, so it connects nothing in the graph either.
    """
    from kb.synthesis.index_renderer import _INDEX_SYSTEM_PROMPT

    assert "[[type:slug|Title]]" in _INDEX_SYSTEM_PROMPT
    assert "never bare `[[Title]]`" in _INDEX_SYSTEM_PROMPT


def test_the_corpus_gives_the_model_the_type_and_slug_it_must_link_with() -> None:
    """Asking for `type:slug` only works if both are in front of the model. They
    are, on every page — this pins that they stay there."""
    pages = [
        _PageRow(wiki_type="repo", slug="research_os", title="research-os", summary="s", body="b")
    ]

    corpus, _ = _format_pages_for_prompt(pages)

    assert "type: repo" in corpus
    assert "slug: research_os" in corpus
