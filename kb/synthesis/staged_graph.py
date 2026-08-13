"""Preflight over a drain's STAGED SET, before anything is published.

WHY THIS EXISTS. `commit()` used to walk the staged pages and persist them one
at a time, each through its own transaction, having checked nothing about the
batch as a whole. Every rule that spans more than one page therefore had nowhere
to live:

  * `page_edits.apply_edits` sees ONE body. It cannot know the batch also
    creates the page that body links to, and the create path does not go
    through it at all (`create_page` carries a whole `body_markdown`).
  * `persist_links_for_page` runs AFTER the page is persisted, on its own
    connection, wrapped in `try:` -- so a rule like "no page may be its own
    ancestor" could only be evaluated once the offending page was already
    published. A check that cannot refuse is not a check.

So: one function that sees every staged page at once and runs before the first
write. Rules that need the whole batch go here. Rules about a single body stay
where they are.

WHAT THIS DOES NOT DO. It makes VALIDATION all-or-nothing, not persistence. If
the batch validates and then the third page fails to write, the first two are
already published -- `Normalizer._persist` owns its own transaction per document
and there is no batch transaction to enroll them in. Closing that needs a
reconciliation pass; the preflight only guarantees that a batch which was going
to be refused is refused before any of it lands.

PUBLISH ORDER IS DETERMINISTIC: creates first, then updates, stable within each
group. The reason is replay, not referential integrity. A batch that fails
halfway has published a prefix of itself, and diagnosing which pages landed
means being able to reproduce the same prefix -- dict insertion order across two
maps is not that.

It is NOT ordered so links resolve. `persist_links_for_page` writes
`dst_wiki_type`/`dst_slug` as plain text and `wiki_links` has no foreign key on
the destination, so an edge to a page that does not exist yet persists exactly
like one to a page that does. An earlier version of this docstring claimed
otherwise; it was wrong, and creates are unordered among themselves anyway, so a
create->create link inside one batch would break the claim even if the FK
existed. Ordering here buys reproducibility. If referential integrity is ever
wanted, it has to come from a constraint or a preflight rule, not from sequence.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from kb.synthesis.wiki_links import extract_links_from_markdown

#: The relation verb that makes one page a child of another:
#: `[[feature:research-os-deployment|subpage|Deployment]]`. Reuses the link
#: graph rather than adding a parent column -- the edges are already typed,
#: already persisted, and already have a parser.
SUBPAGE = "subpage"

#: Page bodies are measured in UTF-8 BYTES, matching `documents.body_size_bytes`
#: (`len(body.encode("utf-8"))`). Python character length, "KB", and Postgres
#: `length()` all disagree on non-ASCII, and this corpus has non-ASCII in it.
PAGE_CAP_BYTES = 8 * 1024

#: The anti-fragmentation control, and the one that actually does the work. A
#: cap alone creates constant pressure to split -- splitting is always the
#: cheapest way to get under it, and every pass adds content -- so without a
#: lower bound the equilibrium is dozens of tiny pages. Forty 900-byte pages are
#: worse to read than one long one AND cost the index more, since each page
#: contributes a full entry regardless of size.
#:
#: Applied ONLY to split products, never to pages generally: `person/shi_dong`
#: is 143 bytes and entirely legitimate. See `_split_children`.
SPLIT_FLOOR_BYTES = 2 * 1024

#: Cap and floor together bound ONE split, not accumulation across passes: a
#: page can reach the cap, split, grow, and split again. Fan-out is what bounds
#: the total.
MAX_CHILDREN = 6

#: root -> feature -> detail. Both discovery surfaces (`wiki list`, the MCP
#: `pages` view) are flat, so depth beyond this exists only in the graph and no
#: reader ever sees the shape.
MAX_DEPTH = 2

#: The front page is generated whole from every other page and has no business
#: being split. It is the one page the cap must not apply to.
CAP_EXEMPT_TYPES = frozenset({"index"})


@dataclass(frozen=True, slots=True)
class StagedPage:
    """One page a drain intends to write, create and update alike.

    `is_new` is what separates them. Rules care about it (a create cannot
    collide with a live page; an update cannot target one that does not exist),
    and publish order depends on it.
    """

    wiki_type: str
    slug: str
    body: str
    is_new: bool

    @property
    def key(self) -> tuple[str, str]:
        return (self.wiki_type, self.slug)

    @property
    def ref(self) -> str:
        return f"{self.wiki_type}/{self.slug}"

    @property
    def size_bytes(self) -> int:
        return len(self.body.encode("utf-8"))

    def subpage_targets(self) -> list[tuple[str, str]]:
        """The children this page claims, in body order.

        Order matters for the fan-out message: naming the seventh child is more
        useful than saying "too many".
        """
        return [
            (link.dst_wiki_type, link.dst_slug)
            for link in extract_links_from_markdown(self.body)
            if link.link_type == SUBPAGE
        ]


@dataclass(frozen=True, slots=True)
class Violation:
    """One reason a batch must not be published.

    `rule` is a stable identifier, not prose: it is what a log filter and a DLQ
    reason are keyed on. `detail` is for the human reading that DLQ row, and is
    allowed to be long.
    """

    rule: str
    ref: str
    detail: str


def page_over_cap(page: StagedPage) -> Violation | None:
    """The size rule, alone, so a TOOL can run it too.

    Checked at `update_page` / `create_page` as well as here, and that is the
    point rather than duplication. A refusal at commit time reaches the agent as
    a halted drain, after the conversation has ended, where nothing can act on
    it -- `apply_edits` already makes this argument for anchor failures ("what
    lets a failed anchor reach the model"). Checked at the tool, the model is
    told while it can still split the page. This copy is the backstop for
    everything that does not route through a tool.
    """
    if page.wiki_type in CAP_EXEMPT_TYPES or page.size_bytes <= PAGE_CAP_BYTES:
        return None
    return Violation(
        rule="page_over_cap",
        ref=page.ref,
        detail=(
            f"{page.ref} is {page.size_bytes} bytes, over the {PAGE_CAP_BYTES}-byte cap. "
            f"Move a whole section onto its own `feature` page and link it from here as "
            f"`[[feature:<slug>|{SUBPAGE}|<Title>]]`. Split on a topic boundary, not a byte "
            f"count, and leave both sides above {SPLIT_FLOOR_BYTES} bytes -- a page too small "
            f"to stand alone should stay where it is."
        ),
    )


def validate_batch(
    pages: Iterable[StagedPage],
    *,
    live_parents: Mapping[tuple[str, str], tuple[str, str]] | None = None,
) -> list[Violation]:
    """Every rule that needs to see the whole staged set. Pure, no I/O.

    `live_parents` maps an already-published child to its parent, so depth and
    cycles are checked against the REAL graph rather than the slice of it that
    happens to be in this batch. Omit it and only the staged subgraph is
    checked: still correct for anything created within one drain, blind to a
    batch that closes a loop through a page it did not touch. The caller reads
    it; keeping this function pure is what makes the rules testable without a
    database.

    Returns ALL violations rather than raising on the first. A batch with three
    problems should be reported as three, or the agent fixes one, retries, and
    discovers the next one a drain later -- and `concurrencyPolicy: Forbid`
    means each of those round trips costs a full cycle.
    """
    staged = list(pages)
    violations: list[Violation] = []

    seen: dict[tuple[str, str], StagedPage] = {}
    for page in staged:
        prior = seen.get(page.key)
        if prior is not None:
            # Staging the same page as both a create and an update leaves the
            # outcome dependent on publish order, which is precisely the kind of
            # thing that should never decide content. Refuse rather than pick.
            violations.append(
                Violation(
                    rule="duplicate_staged_page",
                    ref=page.ref,
                    detail=(
                        f"{page.ref} is staged twice in one drain "
                        f"(as {'create' if prior.is_new else 'update'} and "
                        f"{'create' if page.is_new else 'update'}). Which body wins "
                        "would depend on publish order. Stage it once."
                    ),
                )
            )
            continue
        seen[page.key] = page

    for page in seen.values():
        over = page_over_cap(page)
        if over is not None:
            violations.append(over)

    violations.extend(_graph_violations(seen, live_parents or {}))
    return violations


def _graph_violations(
    staged: Mapping[tuple[str, str], StagedPage],
    live_parents: Mapping[tuple[str, str], tuple[str, str]],
) -> list[Violation]:
    """Ownership, fan-out, depth, cycles, orphans, and the split floor."""
    violations: list[Violation] = []

    # parent -> children, and the inverse. Built from the staged bodies, then
    # overlaid on the live graph so a batch is judged against the tree that will
    # exist, not the fragment it contains.
    parents: dict[tuple[str, str], list[tuple[str, str]]] = dict(
        {child: [parent] for child, parent in live_parents.items()}
    )
    children: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key, page in staged.items():
        targets = page.subpage_targets()
        children[key] = targets
        if len(targets) > MAX_CHILDREN:
            violations.append(
                Violation(
                    rule="too_many_children",
                    ref=page.ref,
                    detail=(
                        f"{page.ref} claims {len(targets)} subpages, over the {MAX_CHILDREN} "
                        f"limit (the {MAX_CHILDREN + 1}th is "
                        f"{targets[MAX_CHILDREN][0]}/{targets[MAX_CHILDREN][1]}). Cap and floor "
                        "bound one split; this bounds accumulation across passes. Promote a "
                        "child to a top-level page of its own instead of adding another."
                    ),
                )
            )
        for target in targets:
            # A page re-staged this drain replaces its live parent claim rather
            # than adding to it: the body IS the edge set, so an edge the new
            # body does not carry no longer exists.
            if target in live_parents and live_parents[target] == key:
                continue
            parents.setdefault(target, [])
            if key not in parents[target]:
                parents[target].append(key)

    for child, claimants in sorted(parents.items()):
        if len(claimants) > 1:
            violations.append(
                Violation(
                    rule="subpage_multiple_parents",
                    ref=f"{child[0]}/{child[1]}",
                    detail=(
                        f"{child[0]}/{child[1]} is claimed as a subpage by "
                        + ", ".join(f"{t}/{s}" for t, s in claimants)
                        + ". A page has one parent; a second claim makes 'where does this "
                        "live' unanswerable. Use a plain [[type:slug]] mention for the "
                        "other reference."
                    ),
                )
            )
        if child not in staged and child not in live_parents:
            violations.append(
                Violation(
                    rule="orphan_subpage_target",
                    ref=f"{child[0]}/{child[1]}",
                    detail=(
                        f"{claimants[0][0]}/{claimants[0][1]} links {child[0]}/{child[1]} as a "
                        "subpage, but that page is neither staged in this drain nor already "
                        "published. Create it in the same batch, or make the reference a "
                        "plain [[type:slug]] mention. The link parser drops a bad type with "
                        "only a warning, so a typo here would silently orphan the child."
                    ),
                )
            )

    violations.extend(_depth_and_cycle_violations(staged, parents))
    violations.extend(_split_floor_violations(staged, parents))
    return violations


def _depth_and_cycle_violations(
    staged: Mapping[tuple[str, str], StagedPage],
    parents: Mapping[tuple[str, str], list[tuple[str, str]]],
) -> list[Violation]:
    """Walk each staged page upward. A cycle is a repeat; depth is the count.

    Walking UP rather than down because every page has at most one parent, so
    the ascent is a path and not a search -- and because a cycle anywhere above
    a page is exactly what makes that page's depth undefined. Nothing in the
    link layer prevents `A subpage-of B subpage-of A` today, and any walker
    following those edges would loop forever.
    """
    violations: list[Violation] = []
    for key, page in sorted(staged.items()):
        seen: list[tuple[str, str]] = [key]
        cursor = key
        while True:
            claimants = parents.get(cursor) or []
            if not claimants:
                break
            cursor = claimants[0]
            if cursor in seen:
                violations.append(
                    Violation(
                        rule="subpage_cycle",
                        ref=page.ref,
                        detail=(
                            "subpage links form a cycle: "
                            + " -> ".join(f"{t}/{s}" for t, s in [*seen, cursor])
                            + ". A page cannot be its own ancestor; any walker following "
                            "these edges would not terminate."
                        ),
                    )
                )
                break
            seen.append(cursor)
            if len(seen) - 1 > MAX_DEPTH:
                violations.append(
                    Violation(
                        rule="subpage_depth_exceeded",
                        ref=page.ref,
                        detail=(
                            f"{page.ref} sits {len(seen) - 1} levels deep, past {MAX_DEPTH} "
                            "(" + " -> ".join(f"{t}/{s}" for t, s in reversed(seen)) + "). "
                            "Both discovery surfaces are flat, so depth past this exists only "
                            "in the graph and no reader ever sees it. A topic this nested "
                            "usually wants its own top-level page."
                        ),
                    )
                )
                break
    return violations


def _split_floor_violations(
    staged: Mapping[tuple[str, str], StagedPage],
    parents: Mapping[tuple[str, str], list[tuple[str, str]]],
) -> list[Violation]:
    """The floor, applied ONLY to split products.

    A split product is a page created in this batch whose parent is also in it:
    that is what "the agent broke a page apart" looks like from here. The floor
    deliberately does NOT apply to pages generally -- `person/shi_dong` is 143
    bytes and correct -- nor to a new page created on its own, which is a new
    subject rather than a fragment of an old one.
    """
    violations: list[Violation] = []
    for key, page in sorted(staged.items()):
        if not page.is_new:
            continue
        claimants = parents.get(key) or []
        if not claimants or claimants[0] not in staged:
            continue
        if page.size_bytes >= SPLIT_FLOOR_BYTES:
            continue
        violations.append(
            Violation(
                rule="split_below_floor",
                ref=page.ref,
                detail=(
                    f"{page.ref} was split out of {claimants[0][0]}/{claimants[0][1]} at only "
                    f"{page.size_bytes} bytes, under the {SPLIT_FLOOR_BYTES}-byte floor. A cap "
                    "makes splitting the cheapest way to comply, so without a floor pages "
                    "fragment into pieces too small to be worth opening. Keep this section in "
                    "its parent, or move enough with it to stand on its own."
                ),
            )
        )
    return violations


def publish_order(pages: Iterable[StagedPage]) -> list[StagedPage]:
    """Creates first, then updates; stable within each group.

    See the module docstring for why this is about replay and not about links
    resolving. A batch that failed halfway must fail at the same page when
    replayed, or diagnosing a partial publish means guessing which pages made
    it.
    """
    staged = list(pages)
    return [p for p in staged if p.is_new] + [p for p in staged if not p.is_new]
