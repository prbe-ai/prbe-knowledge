"""Surgical edits to a wiki page body.

WHY THE AGENT DOES NOT SEND A WHOLE PAGE. `update_page` took
`body_markdown` -- the complete new text -- so every nightly pass asked the
model to reproduce everything it was NOT changing. It does not reproduce it
exactly. Measured on 2026-08-12 against the live corpus, comparing every
retired chunk's sentences against the live body:

    repo/research_os     467 of 5,534 sentences gone   8.4%
    repo/prbe_knowledge  246 of 1,907 sentences gone  12.9%

"Gone" there means present in an earlier version, absent from the live body,
and with no >=72%-similar sentence anywhere in it -- not reworded, dropped. 67
of them went in the last twenty versions alone, including facts that were still
true. The page feeds its own output back in as context for the next pass, so a
dropped fact stays dropped and nothing ever notices.

This is the mechanism the pre-migration single-document generator used, and its
contract stated the reason in one line: "A model asked to reproduce 20,000
characters it is not changing drifts on some of them every pass." The multi-page
rewrite kept the read-then-write shape and lost the edit ops. This restores them.

THE INVARIANT IS EXACT-ONCE, ALL-OR-NOTHING. `find` must match the current body
exactly once: zero matches means the model is editing text that is not there
(usually text it hallucinated or mis-copied), and two means it cannot know which
one it meant. Either way the whole call is refused and nothing is applied --
a partially applied edit list is a page in a state no one asked for, and the
model cannot see that it happened.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

#: How many near-miss lines to quote back when an anchor does not match.
#: Three is enough to cover "the model mis-copied one of a few similar
#: headings" without turning a refusal into a page dump.
_NEAR_MISS_COUNT = 3

#: Longest near-miss line quoted back. A pathological single-line page would
#: otherwise put the whole body in an error message.
_NEAR_MISS_MAX_CHARS = 200

#: Below this ratio the "closest" line is not close to anything and quoting it
#: would be noise the model has to reason past.
_NEAR_MISS_CUTOFF = 0.5


def _near_misses(body: str, find: str) -> list[str]:
    """Lines in `body` most similar to a `find` that matched nothing.

    WHY THE ERROR CARRIES TEXT. Telling the model "copy the anchor VERBATIM"
    without showing it what is actually there leaves it guessing from the same
    context that produced the wrong anchor, so it guesses wrong again. Each
    wrong guess costs a turn and is never consequential, and fifteen in a row
    halt the drain. Quoting the nearest real lines turns the retry into a copy.

    Matched per LINE rather than over the whole body: an anchor is nearly
    always one line or a fragment of one, and line-level matching is what makes
    the answer copyable.
    """
    needle = find.strip()
    if not needle:
        return []
    candidates = [ln for ln in body.splitlines() if ln.strip()]
    if not candidates:
        return []
    close = difflib.get_close_matches(
        needle, candidates, n=_NEAR_MISS_COUNT, cutoff=_NEAR_MISS_CUTOFF
    )
    return [ln[:_NEAR_MISS_MAX_CHARS] for ln in close]


def _not_found_message(index: int, op: str, body: str, find: str) -> str:
    """The zero-match refusal, with the nearest real lines when there are any."""
    base = (
        f"edit {index} ({op}): `find` does not appear in the page. "
        "Copy the anchor VERBATIM from the current body, including "
        "punctuation and capitalisation."
    )
    hits = _near_misses(body, find)
    if not hits:
        return base
    quoted = "\n".join(f"  {ln}" for ln in hits)
    return f"{base}\nClosest lines currently in the page:\n{quoted}"


class EditError(ValueError):
    """An edit could not be applied. Carries a message meant for the MODEL."""


@dataclass(frozen=True, slots=True)
class PageEdit:
    """One anchored edit. `op` is replace | append_after | delete."""

    op: str
    find: str
    text: str | None = None


def apply_edits(body: str, edits: list[PageEdit]) -> str:
    """Apply `edits` in order and return the new body. RAISES on any failure.

    Order matters and is the caller's: an edit's `find` is matched against the
    body as the PREVIOUS edits left it, so a model may anchor one edit on text
    an earlier one inserted. That is also why this cannot validate the whole
    list up front and then apply it -- match counts are only meaningful in
    sequence.

    Nothing is mutated on the way out. The body is rebuilt locally and returned,
    so a raise leaves the caller's copy untouched without needing a rollback.
    """
    if not edits:
        raise EditError("no edits given; send at least one")

    current = body
    for index, edit in enumerate(edits, start=1):
        if not edit.find:
            raise EditError(f"edit {index}: `find` is empty")

        occurrences = current.count(edit.find)
        if occurrences == 0:
            raise EditError(
                _not_found_message(index, edit.op, current, edit.find)
            )
        if occurrences > 1:
            raise EditError(
                f"edit {index} ({edit.op}): `find` appears {occurrences} times, "
                "so it does not say which one you mean. Extend the anchor with "
                "surrounding text until it is unique."
            )

        if edit.op == "delete":
            if edit.text:
                raise EditError(f"edit {index}: `delete` takes no `text`")
            current = current.replace(edit.find, "", 1)
        elif edit.op == "replace":
            if edit.text is None:
                raise EditError(f"edit {index}: `replace` needs `text`")
            current = current.replace(edit.find, edit.text, 1)
        elif edit.op == "append_after":
            if not edit.text:
                raise EditError(f"edit {index}: `append_after` needs `text`")
            current = current.replace(edit.find, edit.find + edit.text, 1)
        else:
            raise EditError(
                f"edit {index}: unknown op {edit.op!r}; "
                "use replace, append_after or delete"
            )

    return current
