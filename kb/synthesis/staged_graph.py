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

PUBLISH ORDER IS PART OF THE CONTRACT. Creates go first. A page that is created
and linked from a page updated in the same batch must exist by the time the
linking page persists its links, or the edge resolves against nothing. The old
order (updates, then creates) got this backwards, which was invisible only
because nothing yet depends on the link graph being complete at publish time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


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


def validate_batch(pages: Iterable[StagedPage]) -> list[Violation]:
    """Every rule that needs to see the whole staged set. Pure, no I/O.

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

    return violations


def publish_order(pages: Iterable[StagedPage]) -> list[StagedPage]:
    """Creates first, then updates; stable within each group.

    See the module docstring. Stability matters for reproducing a partial
    publish: a batch that failed halfway should fail at the same page when
    replayed, or diagnosing it means guessing which pages made it.
    """
    staged = list(pages)
    return [p for p in staged if p.is_new] + [p for p in staged if not p.is_new]
