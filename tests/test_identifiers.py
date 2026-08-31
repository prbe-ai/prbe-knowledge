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


# ---- inferred kinds: number refs, hex prefixes, PagerDuty ids ---------------


def _kinds(query: str) -> list[tuple[str, str]]:
    return [(d.kind, d.canonical_id) for d in detect_identifiers(query)]


def test_number_refs_bare_framed_and_qualified() -> None:
    assert _kinds("#383") == [("number_ref", "#383")]
    assert _kinds("PR #232") == [("number_ref", "#232")]
    assert _kinds("pull request #536") == [("number_ref", "#536")]
    assert _kinds("issue #42") == [("number_ref", "#42")]
    assert _kinds("research-os PR #539") == [("number_ref", "research-os#539")]
    # A junk qualifier is captured — resolution treats it as soft.
    assert _kinds("the PR #232") == [("number_ref", "the#232")]


def test_full_issue_ref_is_not_double_reported() -> None:
    out = _kinds("prbe-ai/prbe-knowledge#29")
    assert out == [("issue_ref", "prbe-ai/prbe-knowledge#29")]


def test_hex_prefixes_short_sha_and_uuid_segment() -> None:
    assert _kinds("ce09c43") == [("hex_prefix", "ce09c43")]
    assert _kinds("commit 0dd764a") == [("hex_prefix", "0dd764a")]
    assert _kinds("session 5e0f3220") == [("hex_prefix", "5e0f3220")]
    assert _kinds("CE09C43") == [("hex_prefix", "ce09c43")]


def test_hex_prefix_floors_and_guards() -> None:
    # 6 chars: below the floor; 12+: the sha kind, not a prefix.
    assert _kinds("abc123") == []
    assert _kinds("9027120fe3ab") == [("commit_sha", "9027120fe3ab")]
    # A full uuid is masked — its segments never re-report as prefixes.
    out = _kinds("3c325e11-2008-46a9-83f7-fc40d11eaf82")
    assert [k for k, _ in out] == ["uuid"]
    assert _kinds("#178599") == [("number_ref", "#178599")]


def test_hex_prefix_excludes_decimals_and_hyphen_chains() -> None:
    """Review regressions: pure-decimal tokens are quantities (dates,
    epochs, order numbers), never prefixes; segments of hyphenated tokens
    (tickets, run slugs) never re-report."""
    assert _kinds("what changed on 20260831") == []
    assert _kinds("revenue was 1000000 last quarter") == []
    assert _kinds("thread 1714000000.123") == []
    assert _kinds("DEADBEEF-12") == [("ticket", "DEADBEEF-12")]


def test_adjacent_identifiers_all_survive_a_number_ref() -> None:
    """Review regression: the number ref's greedy qualifier must not
    swallow (or mask) a neighboring identifier, and an identifier-shaped
    qualifier is dropped from the canonical."""
    out = _kinds("commit 9027120fe3ab #383")
    assert ("commit_sha", "9027120fe3ab") in out
    assert ("number_ref", "#383") in out
    out = _kinds("deadbeefcafe#42")
    assert ("issue_ref", "deadbeefcafe#42") in out
    assert ("commit_sha", "deadbeefcafe") in out
    out = _kinds("incident Q00CUSHZAE4OXF #12")
    assert ("pd_incident", "Q00CUSHZAE4OXF") in out


def test_number_ref_carries_structured_halves() -> None:
    d = detect_identifiers("research-os PR #539")[0]
    assert (d.qualifier, d.number) == ("research-os", "539")
    d = detect_identifiers("deadlock #383")[0]
    assert (d.qualifier, d.number) == ("deadlock", "383")


def test_pagerduty_incident_ids() -> None:
    assert _kinds("incident Q00CUSHZAE4OXF") == [("pd_incident", "Q00CUSHZAE4OXF")]
    assert _kinds("Q0RPQN0Z3INCYO") == [("pd_incident", "Q0RPQN0Z3INCYO")]
    # Lowercase can never be a PD id.
    assert _kinds("q00cushzae4oxf") == []
    # Digit-free ALL-CAPS words are English, not PD ids (review).
    assert _kinds("QUALIFICATIONS matrix") == []
