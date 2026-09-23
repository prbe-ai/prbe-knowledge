"""A document's own graph node and the bare mention of the same thing.

Connectors write both for one real object:

    github:<owner>/<repo>:pr:<n>        <->  <owner>/<repo>#<n>
    github:<owner>/<repo>:issue:<n>     <->  <owner>/<repo>#<n>
    linear:<workspace>:issue:<uuid>     <->  <uuid>

The first id is a `documents.doc_id` -- retrieval reaches the document through
that node -- and the second is what other sources link to. They are the same
object by construction, so the analyzer folds the mention INTO the document's
node without asking a judge (the reverse, which is what a judge proposes,
detaches the document from the graph). 660 of the 661 documents detached by
auto-merge before 2026-09-23 were one of these pairs, folded the wrong way.
"""

from __future__ import annotations

import re

_UUID = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_REPO = r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"

_GITHUB_DOCUMENT_RE = re.compile(rf"^github:({_REPO}):(?:pr|issue):(\d+)$")
_LINEAR_DOCUMENT_RE = re.compile(rf"^linear:[^:]+:issue:({_UUID})$")
_GITHUB_MENTION_RE = re.compile(rf"^({_REPO})#(\d+)$")


def mention_of(document_id: str) -> str | None:
    """The bare mention id that names the same object as `document_id`."""
    m = _GITHUB_DOCUMENT_RE.match(document_id)
    if m:
        return f"{m.group(1)}#{m.group(2)}"
    m = _LINEAR_DOCUMENT_RE.match(document_id)
    if m:
        return m.group(1)
    return None


def documents_of(mention_id: str) -> list[str]:
    """The document ids a GitHub `owner/repo#n` mention can stand for.

    A PR and an issue share one number space per repo, so at most one of the
    two exists. A bare Linear uuid is not resolved here: its document id holds
    a workspace the mention does not carry, and the judge already folds such a
    uuid into its document (the mention is the alias, the right way round).
    """
    m = _GITHUB_MENTION_RE.match(mention_id)
    if not m:
        return []
    repo, number = m.group(1), m.group(2)
    return [f"github:{repo}:pr:{number}", f"github:{repo}:issue:{number}"]
