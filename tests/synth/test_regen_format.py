"""Tests for format_failure_context — converts validator violations into a
single human-readable block to inject into the regen prompt."""

from __future__ import annotations

from scripts.synth.llm.validator_pass2 import Pass2Result, Pass2Violation
from scripts.synth.regen import format_failure_context
from scripts.synth.validator import Violation


def test_format_pass1_only() -> None:
    pass1 = (
        Violation(doc_id="d1", out_of_world=("auto-scaling", "rate-limited")),
    )
    text = format_failure_context(
        pass1_violations=pass1,
        pass2_result=None,
        target_doc_id="d1",
    )
    assert "out-of-world tokens" in text
    assert "auto-scaling" in text
    assert "rate-limited" in text


def test_format_pass2_only() -> None:
    pass2 = Pass2Result(
        passed=False,
        violations=(Pass2Violation(doc_id="d1", issue="root_cause contradicts d0"),),
    )
    text = format_failure_context(
        pass1_violations=(),
        pass2_result=pass2,
        target_doc_id="d1",
    )
    assert "consistency issue" in text
    assert "root_cause contradicts d0" in text


def test_format_combined_pass1_and_pass2() -> None:
    pass1 = (Violation(doc_id="d1", out_of_world=("kubelet",)),)
    pass2 = Pass2Result(
        passed=False,
        violations=(Pass2Violation(doc_id="d1", issue="contradicts d0"),),
    )
    text = format_failure_context(
        pass1_violations=pass1,
        pass2_result=pass2,
        target_doc_id="d1",
    )
    assert "kubelet" in text
    assert "contradicts d0" in text


def test_format_filters_to_target_doc() -> None:
    pass1 = (
        Violation(doc_id="d0", out_of_world=("foo",)),
        Violation(doc_id="d1", out_of_world=("bar",)),
    )
    text = format_failure_context(
        pass1_violations=pass1,
        pass2_result=None,
        target_doc_id="d1",
    )
    assert "bar" in text
    assert "foo" not in text


def test_format_empty_when_no_violations() -> None:
    text = format_failure_context(
        pass1_violations=(),
        pass2_result=None,
        target_doc_id="d1",
    )
    assert text == ""
