"""Preflight over a drain's staged set.

The seam these tests pin exists because every rule that spans more than one page
previously had nowhere to live: `apply_edits` sees one body, `create_page`
bypasses it entirely, and link persistence runs after the page is already
published. See `kb/synthesis/staged_graph.py`.
"""

from __future__ import annotations

from kb.synthesis.staged_graph import (
    MAX_CHILDREN,
    PAGE_CAP_BYTES,
    SPLIT_FLOOR_BYTES,
    StagedPage,
    publish_order,
    validate_batch,
)


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


# ---------------------------------------------------------------------------
# T2 — size: cap, split floor, fan-out
# ---------------------------------------------------------------------------


def _big(n: int) -> str:
    return "x" * n


def test_a_page_over_the_cap_is_refused_and_told_how_to_split() -> None:
    """The message is the feature. A refusal that only says "too big" leaves
    the model to guess between summarising (loses content), truncating (loses
    content) and splitting (the one we want)."""
    v = validate_batch([_page("big", body=_big(PAGE_CAP_BYTES + 1))])

    assert [x.rule for x in v] == ["page_over_cap"]
    assert "subpage" in v[0].detail
    assert "topic boundary" in v[0].detail


def test_the_cap_is_measured_in_utf8_bytes_not_characters() -> None:
    """`documents.body_size_bytes` is `len(body.encode("utf-8"))`, and this
    corpus has non-ASCII in it. A character-length cap would disagree with the
    number the database records for the same page."""
    body = "é" * (PAGE_CAP_BYTES - 100)  # 2 bytes each: well over, under in chars

    assert len(body) < PAGE_CAP_BYTES
    assert [x.rule for x in validate_batch([_page("accents", body=body)])] == ["page_over_cap"]


def test_the_index_page_is_exempt_from_the_cap() -> None:
    """The front page is generated whole from every other page. Splitting it is
    meaningless and capping it would just fail the render."""
    assert validate_batch([_page("contents", wiki_type="index", body=_big(50_000))]) == []


def test_a_split_child_under_the_floor_is_refused() -> None:
    """The rule that stops a cap turning the wiki into stubs. Splitting is
    always the cheapest way under a cap, so without a floor the equilibrium is
    dozens of tiny pages -- worse to read AND more expensive for the index,
    which pays per page regardless of size."""
    parent = _page("p", body="intro [[feature:p-detail|subpage|Detail]]")
    child = _page("p-detail", wiki_type="feature", is_new=True, body="tiny")

    v = validate_batch([parent, child])

    assert [x.rule for x in v] == ["split_below_floor"]
    assert "repo/p" in v[0].detail


def test_the_floor_does_not_apply_to_ordinary_small_pages() -> None:
    """`person/shi_dong` is 143 bytes and correct. The floor is about split
    PRODUCTS, not about pages being short -- a new page created on its own is a
    new subject, not a fragment of an old one."""
    assert validate_batch([_page("shi_dong", wiki_type="person", is_new=True, body="hi")]) == []


def test_more_than_six_children_is_refused() -> None:
    """Cap and floor bound ONE split; a page can reach the cap, split, grow and
    split again. Fan-out is what bounds accumulation across passes."""
    body = " ".join(f"[[feature:c{i}|subpage|C{i}]]" for i in range(MAX_CHILDREN + 1))
    pages = [_page("parent", body=body)] + [
        _page(f"c{i}", wiki_type="feature", body=_big(SPLIT_FLOOR_BYTES))
        for i in range(MAX_CHILDREN + 1)
    ]

    rules = [x.rule for x in validate_batch(pages)]

    assert "too_many_children" in rules


# ---------------------------------------------------------------------------
# T1 — graph invariants
# ---------------------------------------------------------------------------


def test_two_parents_claiming_one_child_is_refused() -> None:
    """ "Where does this live" has to have one answer."""
    pages = [
        _page("a", body="[[feature:shared|subpage|S]]"),
        _page("b", body="[[feature:shared|subpage|S]]"),
        _page("shared", wiki_type="feature", body=_big(SPLIT_FLOOR_BYTES)),
    ]

    v = [x for x in validate_batch(pages) if x.rule == "subpage_multiple_parents"]

    assert len(v) == 1
    assert "repo/a" in v[0].detail and "repo/b" in v[0].detail


def test_a_subpage_link_to_a_page_that_does_not_exist_is_refused() -> None:
    """The link parser drops an unknown type with a warning rather than
    raising, which is right for prose links and wrong for structural ones: a
    typo would silently orphan the child."""
    v = [
        x
        for x in validate_batch([_page("a", body="[[feature:ghost|subpage|G]]")])
        if x.rule == "orphan_subpage_target"
    ]

    assert len(v) == 1
    assert "feature/ghost" in v[0].ref


def test_a_cycle_in_subpage_links_is_refused() -> None:
    """`A subpage-of B subpage-of A` is representable today and nothing stops
    it. Any walker following those edges would not terminate."""
    pages = [
        _page("a", body="[[repo:b|subpage|B]]"),
        _page("b", body="[[repo:a|subpage|A]]"),
    ]

    assert "subpage_cycle" in {x.rule for x in validate_batch(pages)}


def test_depth_beyond_two_is_refused() -> None:
    """Both discovery surfaces are flat, so a third level exists only in the
    graph and no reader ever sees the shape."""
    pages = [
        _page("root", body="[[feature:mid|subpage|M]]"),
        _page("mid", wiki_type="feature", body="[[feature:leaf|subpage|L]]"),
        _page("leaf", wiki_type="feature", body="[[feature:deep|subpage|D]]"),
        _page("deep", wiki_type="feature", body=_big(SPLIT_FLOOR_BYTES)),
    ]

    assert "subpage_depth_exceeded" in {x.rule for x in validate_batch(pages)}


def test_depth_is_judged_against_the_live_graph_not_just_the_batch() -> None:
    """A drain that adds ONE level can breach depth through pages it never
    staged. `root -> mid -> leaf` is already published; this batch hangs a
    fourth page off `leaf`. Nothing in the batch alone looks deep."""
    live = {("feature", "mid"): ("repo", "root"), ("feature", "leaf"): ("feature", "mid")}
    pages = [
        _page("leaf", wiki_type="feature", body="[[feature:deep|subpage|D]]"),
        _page("deep", wiki_type="feature", body=_big(SPLIT_FLOOR_BYTES)),
    ]

    assert "subpage_depth_exceeded" in {x.rule for x in validate_batch(pages, live_parents=live)}


def test_a_cycle_closed_through_an_untouched_live_page_is_caught() -> None:
    """The case the live-edge read exists for. `a`'s parent is already `b`;
    this batch edits `a` to claim `b` as ITS child, closing a -> b -> a. The
    batch on its own is a single page with one link and looks acyclic."""
    live = {("repo", "a"): ("repo", "b")}
    pages = [_page("a", body="[[repo:b|subpage|B]]")]

    assert "subpage_cycle" in {x.rule for x in validate_batch(pages, live_parents=live)}


def test_a_clean_split_passes_every_rule() -> None:
    """The shape the whole change exists to produce: a parent under the cap
    holding the map, one child over the floor holding the detail."""
    parent = _page("research_os", body="Overview. [[feature:ros-deploy|subpage|Deployment]]")
    child = _page("ros-deploy", wiki_type="feature", is_new=True, body=_big(SPLIT_FLOOR_BYTES + 10))

    assert validate_batch([parent, child]) == []
