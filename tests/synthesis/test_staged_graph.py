"""Preflight over a drain's staged set.

The seam these tests pin exists because every rule that spans more than one page
previously had nowhere to live: `apply_edits` sees one body, `create_page`
bypasses it entirely, and link persistence runs after the page is already
published. See `kb/synthesis/staged_graph.py`.
"""

from __future__ import annotations

from kb.synthesis.staged_graph import StagedPage, publish_order, validate_batch


def _page(
    slug: str, *, is_new: bool = False, body: str = "text", wiki_type: str = "repo"
) -> StagedPage:
    return StagedPage(wiki_type=wiki_type, slug=slug, body=body, is_new=is_new)


def test_a_clean_batch_has_no_violations() -> None:
    """The common case. A preflight that fires on ordinary work is worse than
    none, because the drain then publishes nothing every night."""
    assert validate_batch([_page("a"), _page("b", is_new=True), _page("c")]) == []


def test_the_same_page_staged_twice_is_refused() -> None:
    """Staged as both a create and an update, the winner would be decided by
    publish order. Content must never depend on that, so the batch is refused
    rather than resolved."""
    violations = validate_batch([_page("dup", is_new=True), _page("dup")])

    assert len(violations) == 1
    assert violations[0].rule == "duplicate_staged_page"
    assert violations[0].ref == "repo/dup"
    assert "staged twice" in violations[0].detail


def test_the_same_slug_under_different_types_is_fine() -> None:
    """`(wiki_type, slug)` is the identity, not slug alone. `repo/probe` and
    `project/probe` are different pages and routinely coexist."""
    assert (
        validate_batch([_page("probe", wiki_type="repo"), _page("probe", wiki_type="project")])
        == []
    )


def test_every_violation_is_reported_not_just_the_first() -> None:
    """A batch with three problems must come back as three. Reporting one at a
    time means the agent fixes it, retries, and meets the next one a drain
    later -- and `concurrencyPolicy: Forbid` makes each of those a full cycle.
    """
    violations = validate_batch(
        [
            _page("a", is_new=True),
            _page("a"),
            _page("b", is_new=True),
            _page("b"),
        ]
    )

    assert {v.ref for v in violations} == {"repo/a", "repo/b"}


def test_publish_order_puts_creates_first() -> None:
    """A page created and linked from a page updated in the same batch has to
    exist by the time the linking page persists its links. The original order
    (updates, then creates) had this backwards."""
    pages = [_page("u1"), _page("c1", is_new=True), _page("u2"), _page("c2", is_new=True)]

    ordered = publish_order(pages)

    assert [p.slug for p in ordered] == ["c1", "c2", "u1", "u2"]


def test_publish_order_is_stable_within_each_group() -> None:
    """A batch that fails halfway must fail at the same page when replayed, or
    diagnosing a partial publish means guessing which pages landed."""
    pages = [_page(s, is_new=True) for s in ("x", "a", "m")]

    assert [p.slug for p in publish_order(pages)] == ["x", "a", "m"]
    assert [p.slug for p in publish_order(pages)] == ["x", "a", "m"]


def test_publish_order_preserves_the_whole_batch() -> None:
    """Ordering must not drop or duplicate. A silently shortened batch would
    publish some pages and mark every event applied."""
    pages = [_page("a"), _page("b", is_new=True), _page("c"), _page("d", is_new=True)]

    ordered = publish_order(pages)

    assert len(ordered) == len(pages)
    assert {p.key for p in ordered} == {p.key for p in pages}
