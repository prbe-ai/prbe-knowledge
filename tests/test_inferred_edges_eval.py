"""Eval harness for inferred-edges extractor.

Loads golden fixtures from tests/inferred_edges/golden/ and runs the
extractor against each. Passes if:
  - known-edge recall >= 80% (at least 4/5 expected edges found)
  - hallucination rate <= 5% (edges whose endpoints don't exist in bundle)

Run with: pytest tests/test_inferred_edges_eval.py -v -m eval

Skipped if ANTHROPIC_API_KEY is not set.

NOTE: 30 golden fixtures across 5 edge types. Recall/hallucination gate is
statistically stable at this size.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from engine.ingest.inferred_edges.bundle import Bundle, BundleDoc
from engine.ingest.inferred_edges.extractor import extract_edges
from engine.ingest.inferred_edges.prompts.v1 import PROMPT_VERSION

GOLDEN_DIR = Path(__file__).parent / "inferred_edges" / "golden"

pytestmark = pytest.mark.eval


def _load_fixtures() -> list[dict]:
    """Load all JSON fixtures from the golden directory."""
    fixtures = []
    for path in sorted(GOLDEN_DIR.glob("fixture_*.json")):
        with open(path) as f:
            data = json.load(f)
            data["_fixture_name"] = path.stem
            fixtures.append(data)
    return fixtures


def _bundle_from_fixture(fixture: dict) -> Bundle:
    """Reconstruct a Bundle from fixture JSON."""
    bundle_data = fixture["bundle"]
    bundle = Bundle(
        customer_id=bundle_data["customer_id"],
        anchor_doc_id=bundle_data["anchor_doc_id"],
    )
    for doc_data in bundle_data["docs"]:
        bundle.docs.append(
            BundleDoc(
                doc_id=doc_data["doc_id"],
                customer_id=doc_data["customer_id"],
                source_system=doc_data["source_system"],
                title=doc_data.get("title"),
                content=doc_data["content"],
                token_count=doc_data["token_count"],
            )
        )
    bundle.total_tokens = sum(d.token_count for d in bundle.docs)
    return bundle


def _mock_conn_from_fixture(fixture: dict) -> AsyncMock:
    """Mock DB connection returning the fixture's node set."""
    nodes = fixture.get("nodes", [])
    conn = AsyncMock()
    conn.fetch = AsyncMock(
        return_value=[
            {"label": n["label"], "canonical_id": n["canonical_id"]}
            for n in nodes
        ]
    )
    return conn


@pytest.mark.asyncio
async def test_eval_harness_recall_and_hallucination() -> None:
    """Eval CI gate: >=80% recall, <=5% hallucination on golden fixtures."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set -- skipping eval harness")

    fixtures = _load_fixtures()
    if not fixtures:
        pytest.skip("No golden fixtures found in tests/inferred_edges/golden/")

    total_expected = 0
    total_recalled = 0
    total_extracted = 0
    total_hallucinated = 0
    fixture_results: list[dict] = []

    for fixture in fixtures:
        bundle = _bundle_from_fixture(fixture)
        conn = _mock_conn_from_fixture(fixture)
        expected_edges = fixture.get("expected_edges", [])

        # Build the set of valid canonical_ids from the fixture nodes
        valid_cids = {n["canonical_id"] for n in fixture.get("nodes", [])}

        result = await extract_edges(bundle, conn)

        fixture_recalled = 0
        fixture_hallucinated = 0

        for edge in result.edges:
            total_extracted += 1
            # Hallucination check: endpoint not in fixture's node manifest
            if (
                edge.from_canonical_id not in valid_cids
                or edge.to_canonical_id not in valid_cids
            ):
                fixture_hallucinated += 1
                total_hallucinated += 1

        # Recall check: for each expected edge, was it found?
        for expected in expected_edges:
            total_expected += 1
            to_cid = expected["to_canonical_id"]
            edge_type = expected.get("edge_type")
            found = any(
                e.to_canonical_id == to_cid
                and (edge_type is None or e.edge_type == edge_type)
                for e in result.edges
            )
            if found:
                fixture_recalled += 1
                total_recalled += 1

        fixture_results.append(
            {
                "fixture": fixture["_fixture_name"],
                "expected": len(expected_edges),
                "recalled": fixture_recalled,
                "extracted": len(result.edges),
                "hallucinated": fixture_hallucinated,
                "bundle_failed": result.bundle_failed,
            }
        )

    # Report
    recall_rate = total_recalled / total_expected if total_expected > 0 else 1.0
    hallucination_rate = total_hallucinated / total_extracted if total_extracted > 0 else 0.0

    print(f"\n{'='*60}")
    print(f"Eval harness: PROMPT_VERSION={PROMPT_VERSION!r}")
    print(f"  Fixtures: {len(fixtures)}")
    print(f"  Expected edges: {total_expected}")
    print(f"  Recalled: {total_recalled} ({recall_rate:.1%})")
    print(f"  Total extracted: {total_extracted}")
    print(f"  Hallucinated: {total_hallucinated} ({hallucination_rate:.1%})")
    print(f"{'='*60}")
    for r in fixture_results:
        status = "PASS" if r["recalled"] == r["expected"] else "MISS"
        print(
            f"  [{status}] {r['fixture']}: "
            f"recall={r['recalled']}/{r['expected']} "
            f"halluc={r['hallucinated']}/{r['extracted']}"
            + (" [BUNDLE_FAILED]" if r["bundle_failed"] else "")
        )
    print(f"{'='*60}\n")

    # 30 golden fixtures present across 5 edge types; gate is statistically stable.
    assert recall_rate >= 0.80, (
        f"Recall {recall_rate:.1%} < 80% threshold. "
        f"Got {total_recalled}/{total_expected} expected edges."
    )
    assert hallucination_rate <= 0.05, (
        f"Hallucination rate {hallucination_rate:.1%} > 5% threshold. "
        f"Got {total_hallucinated}/{total_extracted} hallucinated edges."
    )


def test_golden_fixtures_exist() -> None:
    """Sanity: at least 1 golden fixture is present."""
    fixtures = _load_fixtures()
    assert len(fixtures) >= 1, "No golden fixtures found in tests/inferred_edges/golden/"


def test_golden_fixtures_schema_valid() -> None:
    """Each golden fixture has required fields."""
    for fixture in _load_fixtures():
        assert "bundle" in fixture, f"Missing 'bundle' in {fixture.get('_fixture_name')}"
        assert "expected_edges" in fixture, f"Missing 'expected_edges' in {fixture.get('_fixture_name')}"
        assert "nodes" in fixture, f"Missing 'nodes' in {fixture.get('_fixture_name')}"
        bundle = fixture["bundle"]
        assert "customer_id" in bundle
        assert "anchor_doc_id" in bundle
        assert "docs" in bundle
        for expected in fixture["expected_edges"]:
            assert "from_canonical_id" in expected or "from_label" in expected
            assert "to_canonical_id" in expected
