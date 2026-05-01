"""Tests for the classifier's cheap rule filter (spec §6 step 2).

The filter runs *before* embedding similarity to drop obvious non-matches
in microseconds. Two operators only: ``==`` and ``in [..]`` — anything
else is treated as non-matching (fail-closed).
"""

from __future__ import annotations

from services.kg.classifier.rules import filter_by_rules
from services.kg.schema import Evidence, Frontmatter, Related, Signature


def _fm(class_id: str, must_match: list[str]) -> Frontmatter:
    return Frontmatter(
        id=class_id,
        type="bug-class",
        description="x",
        signature=Signature(must_match=must_match, embedding_seed="rule-test-seed"),
        related=Related(),
        context_sources=[],
        evidence=Evidence(),
    )


def test_filter_keeps_matching_class() -> None:
    classes = [_fm("a-class", ["status_code == 401"])]
    incident: dict[str, object] = {"status_code": 401, "service": "auth"}
    result = filter_by_rules(incident, classes)
    assert [c.id for c in result] == ["a-class"]


def test_filter_drops_nonmatching_class() -> None:
    classes = [_fm("a-class", ["status_code == 500"])]
    incident: dict[str, object] = {"status_code": 401}
    assert filter_by_rules(incident, classes) == []


def test_in_operator_supported() -> None:
    classes = [_fm("a-class", ["service in [auth-svc, gateway]"])]
    assert filter_by_rules({"service": "auth-svc"}, classes)
    assert not filter_by_rules({"service": "billing"}, classes)


def test_multiple_rules_all_must_match() -> None:
    classes = [_fm("a-class", ["status_code == 401", "service == auth-svc"])]
    assert filter_by_rules({"status_code": 401, "service": "auth-svc"}, classes)
    assert not filter_by_rules({"status_code": 401, "service": "billing"}, classes)


def test_unknown_operator_fails_closed() -> None:
    # A malformed / unsupported operator must drop the class, not match it.
    # Fail-closed is the safer default for a security-adjacent matcher.
    classes = [_fm("a-class", ["status_code != 401"])]
    assert filter_by_rules({"status_code": 500}, classes) == []
    assert filter_by_rules({"status_code": 401}, classes) == []


def test_quoted_values_strip_quotes() -> None:
    # Single- or double-quoted RHS should compare as the unquoted string.
    classes_double = [_fm("a-class", ['service == "auth-svc"'])]
    classes_single = [_fm("b-class", ["service == 'auth-svc'"])]
    assert filter_by_rules({"service": "auth-svc"}, classes_double)
    assert filter_by_rules({"service": "auth-svc"}, classes_single)
