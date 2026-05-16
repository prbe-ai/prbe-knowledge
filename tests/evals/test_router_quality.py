"""Router quality eval — manual run, scores Haiku against labeled fixtures.

Run:
    uv run pytest tests/evals/test_router_quality.py -v -s -m eval

Not in CI (costs Anthropic tokens, takes minutes). Re-run on every change
to services/retrieval/router.py, services/retrieval/grounding.py, or
services/retrieval/pipeline.py. Snapshot the aggregate score in the PR
description and update §Baseline in the design spec.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import yaml

from services.retrieval.router import route_query


@dataclass(slots=True)
class EvalCase:
    """One labeled query fixture.

    Optional per-intent fields (`expected_doc_type`, `forbidden_doc_type`,
    `expected_operation`) are set per intent index or left None for "don't
    check". When set, the per-intent value at index i is matched against
    the routed Intent at the same position. `forbidden_doc_type` accepts a
    list-of-lists so a case can forbid multiple values per intent.
    """

    name: str
    query: str
    customer_id: str
    expected_intents_count: int
    expected_modes: list[str]
    expected_canonical_ids: list[list[str]]
    notes: str = ""
    expected_doc_type: list[str | None] | None = None
    forbidden_doc_type: list[list[str]] | None = None
    expected_operation: list[str | None] | None = None


@dataclass(slots=True)
class EvalReport:
    case: EvalCase
    actual_intents_count: int
    actual_modes: list[str]
    actual_canonical_ids: list[list[str]]
    actual_doc_types: list[str | None]
    actual_operations: list[str | None]
    grounding_hit: bool
    passed: bool
    failure_reasons: list[str] = field(default_factory=list)


def _load_fixtures() -> list[EvalCase]:
    fp = Path(__file__).parent / "router_quality_fixtures.yaml"
    with fp.open() as f:
        raw = yaml.safe_load(f)
    return [EvalCase(**case) for case in raw["cases"]]


def _score_case(case: EvalCase, routed) -> EvalReport:
    actual_modes = [i.mode for i in routed.intents]
    actual_ids = [[e.canonical_id for e in i.entities] for i in routed.intents]
    actual_doc_types: list[str | None] = [i.doc_type for i in routed.intents]
    actual_operations: list[str | None] = [i.operation for i in routed.intents]
    failures: list[str] = []

    if len(routed.intents) != case.expected_intents_count:
        failures.append(
            f"intents_count: expected {case.expected_intents_count}, got {len(routed.intents)}"
        )
    if actual_modes != case.expected_modes:
        failures.append(f"modes: expected {case.expected_modes}, got {actual_modes}")
    for i, (expected_ids, actual) in enumerate(
        zip(case.expected_canonical_ids, actual_ids, strict=False)
    ):
        if not set(expected_ids).issubset(set(actual)):
            failures.append(
                f"intent[{i}] canonical_ids: missing {set(expected_ids) - set(actual)}"
            )

    # Optional per-intent doc_type checks.
    if case.expected_doc_type is not None:
        for i, expected in enumerate(case.expected_doc_type):
            if i >= len(actual_doc_types):
                continue
            actual_dt = actual_doc_types[i]
            if actual_dt != expected:
                failures.append(
                    f"intent[{i}] doc_type: expected {expected!r}, got {actual_dt!r}"
                )
    if case.forbidden_doc_type is not None:
        for i, forbidden in enumerate(case.forbidden_doc_type):
            if i >= len(actual_doc_types):
                continue
            actual_dt = actual_doc_types[i]
            if actual_dt in forbidden:
                failures.append(
                    f"intent[{i}] doc_type: {actual_dt!r} is in forbidden list {forbidden!r}"
                )
    if case.expected_operation is not None:
        for i, expected in enumerate(case.expected_operation):
            if i >= len(actual_operations):
                continue
            actual_op = actual_operations[i]
            if actual_op != expected:
                failures.append(
                    f"intent[{i}] operation: expected {expected!r}, got {actual_op!r}"
                )

    bundle_ids = {c.canonical_id for c in routed.grounding_bundle.candidates}
    grounding_hit = (
        any(any(e.canonical_id in bundle_ids for e in i.entities) for i in routed.intents)
        if bundle_ids
        else False
    )

    return EvalReport(
        case=case,
        actual_intents_count=len(routed.intents),
        actual_modes=actual_modes,
        actual_canonical_ids=actual_ids,
        actual_doc_types=actual_doc_types,
        actual_operations=actual_operations,
        grounding_hit=grounding_hit,
        passed=len(failures) == 0,
        failure_reasons=failures,
    )


@pytest.mark.eval
async def test_router_quality_eval(eval_seeded_customer):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set — eval skipped", allow_module_level=False)

    cases = _load_fixtures()
    reports: list[EvalReport] = []

    for case in cases:
        routed = await route_query(eval_seeded_customer.customer_id, case.query)
        reports.append(_score_case(case, routed))

    total = len(reports)
    passed = sum(1 for r in reports if r.passed)
    grounded = sum(1 for r in reports if r.grounding_hit) / max(total, 1) * 100

    print("\n=== ROUTER QUALITY EVAL ===")
    print(f"Total cases: {total}")
    print(f"Passed:      {passed}/{total} ({passed / total * 100:.0f}%)")
    print(f"Grounding hit rate: {grounded:.0f}%")
    print("\nFailures:")
    for r in reports:
        if not r.passed:
            print(f"  [{r.case.name}] {r.case.query!r}")
            print(
                f"     actual: modes={r.actual_modes} "
                f"ids={r.actual_canonical_ids} "
                f"doc_types={r.actual_doc_types} "
                f"operations={r.actual_operations}"
            )
            for reason in r.failure_reasons:
                print(f"     - {reason}")

    if passed / total < 0.80:
        pytest.skip(
            f"eval pass rate {passed / total * 100:.0f}% below 80% — review before shipping"
        )
