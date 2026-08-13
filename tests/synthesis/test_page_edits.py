"""Anchored page edits — the mechanism that stops the agent losing text.

`update_page` took a whole `body_markdown`, so every nightly pass asked the
model to reproduce everything it was not changing. Measured against the live
corpus on 2026-08-12, comparing every retired chunk's sentences to the live
body: 467 of 5,534 sentences gone on repo/research_os (8.4%) and 246 of 1,907
on repo/prbe_knowledge (12.9%) -- present in an earlier version, absent now,
and not reworded anywhere. 67 of them went in the last twenty versions.

Every test here is about the two properties that make that impossible: an
anchor must be unambiguous, and a failed list changes nothing.
"""

from __future__ import annotations

import pytest

from kb.synthesis.page_edits import EditError, PageEdit, apply_edits

BODY = """# research-os

Probe's experiment-tracking platform.

## Storage

Blobs live in R2; the DB stores a pointer only.

## Auth

Sessions are opaque and server-side.
"""


def test_untouched_text_is_byte_identical() -> None:
    """THE WHOLE POINT. What the model does not name must not change.

    A full-body write cannot promise this -- it re-emits every character, and
    the drift measured above is what that costs. An edit list touches one span
    and the rest of the file is the same object it was.
    """
    out = apply_edits(BODY, [PageEdit("replace", "opaque and server-side", "opaque")])

    assert out.count("Blobs live in R2; the DB stores a pointer only.") == 1
    assert out.split("## Auth")[0] == BODY.split("## Auth")[0]


def test_an_anchor_matching_nothing_is_refused() -> None:
    """Zero matches means the model is editing text that is not there.

    Usually text it hallucinated or mis-copied -- which, applied loosely, is
    exactly how a page acquires a claim nobody made.
    """
    with pytest.raises(EditError, match="does not appear"):
        apply_edits(BODY, [PageEdit("replace", "Sessions are JWTs", "x")])


def test_an_ambiguous_anchor_is_refused() -> None:
    """Two matches means the model cannot know which one it meant.

    Silently taking the first is the failure that looks like a successful edit
    and lands in the wrong paragraph.
    """
    body = "alpha here.\n\nalpha here.\n"
    with pytest.raises(EditError, match="appears 2 times"):
        apply_edits(body, [PageEdit("replace", "alpha here.", "beta.")])


def test_a_failing_edit_applies_none_of_them() -> None:
    """ALL-OR-NOTHING. A partially applied list is a page in a state nobody
    asked for, and the model cannot see that it happened -- it is told the call
    failed, so it re-sends, and the good edits land twice."""
    edits = [
        PageEdit("replace", "Probe's experiment-tracking platform.", "Rewritten."),
        PageEdit("replace", "not in the page at all", "x"),
    ]

    with pytest.raises(EditError):
        apply_edits(BODY, edits)

    # The caller's copy is untouched: nothing is mutated in place, so there is
    # nothing to roll back.
    assert "Probe's experiment-tracking platform." in BODY
    assert "Rewritten." not in BODY


def test_edits_apply_in_order_and_may_anchor_on_earlier_ones() -> None:
    """Order is the caller's, and later edits see earlier results.

    Which is also why the whole list cannot be validated up front: a `find`
    that is ambiguous in the original body may be unique after edit 1, and one
    that is unique now may be duplicated by an insertion.
    """
    out = apply_edits(
        BODY,
        [
            PageEdit("append_after", "## Storage\n", "\nAdded first.\n"),
            PageEdit("replace", "Added first.", "Anchored on the insertion."),
        ],
    )

    assert "Anchored on the insertion." in out
    assert "Added first." not in out


def test_delete_removes_exactly_the_anchor() -> None:
    out = apply_edits(BODY, [PageEdit("delete", "\n## Auth\n\nSessions are opaque and server-side.\n")])

    assert "## Auth" not in out
    assert "## Storage" in out


def test_delete_refuses_replacement_text() -> None:
    """`delete` with `text` is a `replace` the model spelled wrong. Refusing it
    is cheaper than guessing which it meant."""
    with pytest.raises(EditError, match="takes no `text`"):
        apply_edits(BODY, [PageEdit("delete", "## Auth", "something")])


def test_replace_and_append_require_text() -> None:
    with pytest.raises(EditError, match="needs `text`"):
        apply_edits(BODY, [PageEdit("replace", "## Auth", None)])
    with pytest.raises(EditError, match="needs `text`"):
        apply_edits(BODY, [PageEdit("append_after", "## Auth", None)])


def test_an_unknown_op_is_refused_by_name() -> None:
    with pytest.raises(EditError, match="unknown op"):
        apply_edits(BODY, [PageEdit("rewrite", "## Auth", "x")])


def test_an_empty_edit_list_is_refused() -> None:
    """A staged update that changes nothing still writes a version, bumps
    `updated_at` and re-embeds the page. Silence is not an edit."""
    with pytest.raises(EditError, match="no edits"):
        apply_edits(BODY, [])


def test_an_empty_anchor_is_refused() -> None:
    """`"" in body` is always true and `str.count("")` is len+1, so an empty
    anchor would report an absurd match count and insert at position zero."""
    with pytest.raises(EditError, match="`find` is empty"):
        apply_edits(BODY, [PageEdit("replace", "", "x")])


def test_a_replacement_may_contain_its_own_anchor() -> None:
    """Extending a heading in place is the common shape of a real edit.

    The `, 1` on the replace is belt-and-braces, not the thing that makes this
    work: the exact-once check above already guarantees a single occurrence, so
    a global replace would behave identically here. What this pins is the
    OUTCOME -- no runaway expansion when the new text contains the old.
    """
    out = apply_edits(BODY, [PageEdit("replace", "## Auth", "## Auth and identity")])

    assert out.count("## Auth and identity") == 1
    assert "## Auth and identity and identity" not in out
