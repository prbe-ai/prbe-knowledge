"""The document-twin rule's id shapes (engine/ingest/auto_merge/twins.py)."""

from __future__ import annotations

import pytest

from engine.ingest.auto_merge.twins import documents_of, mention_of

U = "0f8fad5b-d9cb-469f-a165-70867728950e"


@pytest.mark.parametrize(
    ("document_id", "mention"),
    [
        ("github:acme/widgets:pr:12", "acme/widgets#12"),
        ("github:acme/widgets:issue:3", "acme/widgets#3"),
        ("github:acme/wid.gets-2:pr:7", "acme/wid.gets-2#7"),
        (f"linear:acme-ws:issue:{U}", U),
        ("github:acme/widgets:commit:deadbeef", None),
        ("github:acme/widgets:pr:12:comment", None),
        ("notion:page:" + U, None),
        (f"linear:acme-ws:project:{U}", None),
        ("acme/widgets#12", None),
    ],
)
def test_mention_of(document_id, mention):
    assert mention_of(document_id) == mention


def test_a_github_mention_can_stand_for_a_pr_or_an_issue():
    assert documents_of("acme/widgets#12") == ["github:acme/widgets:pr:12", "github:acme/widgets:issue:12"]


@pytest.mark.parametrize("mention", [U, "acme/widgets", "acme/widgets#", "#12", "acme#12"])
def test_other_ids_resolve_to_no_document(mention):
    assert documents_of(mention) == []
