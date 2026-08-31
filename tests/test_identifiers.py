"""detect_identifiers: the provenance boundary of the id-pins lane.

Everything this parser emits gets pinned with certainty if it resolves, so
the tests pin BOTH directions: shapes that must detect (else precise
queries demote to similarity — the 0.47) and shapes that must NOT (else a
run slug or ordinary hyphenation blocks the pure-lookup path or wastes a
lookup).
"""

from __future__ import annotations

from engine.shared.identifiers import detect_identifiers


def _pairs(q: str) -> list[tuple[str, str]]:
    return [(d.kind, d.canonical_id) for d in detect_identifiers(q)]


def test_all_four_shapes_detect_and_canonicalize() -> None:
    q = (
        "why did prb-17 break after 3C325E11-2008-46A9-83F7-FC40D11EAF82 "
        "landed, see prbe-ai/prbe-knowledge#204 and DEADBEEFDEAD12"
    )
    assert _pairs(q) == [
        ("uuid", "3c325e11-2008-46a9-83f7-fc40d11eaf82"),
        ("ticket", "PRB-17"),
        ("issue_ref", "prbe-ai/prbe-knowledge#204"),
        ("commit_sha", "deadbeefdead12"),
    ]


def test_lowercase_ticket_detects_uppercase_canonical() -> None:
    """The outside-voice F2 case: lowercase tickets were never detected
    anywhere, while source_id equality is case-sensitive uppercase."""
    assert _pairs("what is prb-17") == [("ticket", "PRB-17")]


def test_uuid_segments_do_not_double_report_as_shas() -> None:
    ids = _pairs("session 3c325e11-2008-46a9-83f7-fc40d11eaf82 details")
    assert [k for k, _ in ids] == ["uuid"]


def test_run_slugs_and_hyphen_chains_do_not_detect() -> None:
    """Slugs are v2 (graph resolution); the ticket regex must not shed a
    hyphen chain's tail as a ticket — that would block the pure-lookup
    path on every slug query for no reason."""
    assert detect_identifiers("find tunneling-sambar-254") == []
    assert detect_identifiers("state-of-the-art-3 methods") == []


def test_identifier_free_query_is_empty_and_cheap() -> None:
    assert detect_identifiers("why is search slow today") == []


def test_duplicates_collapse_and_punctuation_boundaries_hold() -> None:
    assert _pairs("PRB-17 (PRB-17) prb-17.") == [("ticket", "PRB-17")]


def test_ticket_stopwords_do_not_detect() -> None:
    assert detect_identifiers("top-10 utf-8 gpt-5 results") == []
