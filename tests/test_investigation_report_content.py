"""Unit tests for ``_row_to_report_content`` — the report-doc parser
that powers the bundled ``report`` payload on
GET /api/incident-investigations/{id}.

Pure-function tests so they run without a local Postgres. The DB-integration
tests in ``test_investigation_state.py`` cover the surrounding
``get_detail`` wiring.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from services.ingestion.investigation_state import _row_to_report_content


def _doc_row(
    *,
    doc_id: str = "pd:investigation:ABC:v1",
    title: str = "Investigation: db CPU spike",
    body: str = "# Investigation\n\nSomething happened.\n",
    metadata: dict[str, Any] | None = None,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Build the minimal asyncpg-style row dict the parser consumes."""
    return {
        "doc_id": doc_id,
        "title": title,
        "body": body,
        "metadata": metadata if metadata is not None else {
            "mode": "full",
            "version": 1,
            "evidence": [],
            "narrative": None,
        },
        "created_at": created_at or datetime(2026, 5, 24, tzinfo=UTC),
    }


def test_returns_none_when_row_is_none() -> None:
    assert _row_to_report_content(None) is None


def test_happy_path_full_mode() -> None:
    row = _doc_row(metadata={
        "mode": "full",
        "version": 3,
        "evidence": [
            {
                "source": "knowledge",
                "query": "service db",
                "result_summary": "Hit 1",
                "linked_doc_ids": ["a", "b"],
            },
        ],
        "narrative": "Root cause: slow query.",
    })
    out = _row_to_report_content(row)
    assert out is not None
    assert out.mode == "full"
    assert out.version == 3
    assert out.title == "Investigation: db CPU spike"
    assert out.body_markdown.startswith("# Investigation")
    assert out.narrative == "Root cause: slow query."
    assert len(out.evidence) == 1
    assert out.evidence[0].source == "knowledge"
    assert out.evidence[0].linked_doc_ids == ["a", "b"]


def test_metadata_accepts_json_string() -> None:
    """Some asyncpg setups surface jsonb as a string; parser must
    cope without forcing the caller to pre-decode."""
    row = _doc_row(metadata=None)
    row["metadata"] = (
        '{"mode":"full","version":2,"evidence":[],"narrative":"x"}'
    )
    out = _row_to_report_content(row)
    assert out is not None
    assert out.mode == "full"
    assert out.version == 2
    assert out.narrative == "x"


def test_returns_none_when_body_missing() -> None:
    row = _doc_row(body="")
    assert _row_to_report_content(row) is None


def test_returns_none_when_mode_unknown() -> None:
    row = _doc_row(metadata={"mode": "bogus", "version": 1})
    assert _row_to_report_content(row) is None


def test_returns_none_when_metadata_missing_mode() -> None:
    row = _doc_row(metadata={"version": 1})
    assert _row_to_report_content(row) is None


def test_malformed_evidence_entries_are_skipped() -> None:
    row = _doc_row(metadata={
        "mode": "playbook_only",
        "version": 1,
        "evidence": [
            {"source": "knowledge", "query": "q1", "result_summary": "ok"},
            "not-an-evidence-dict",
            {"source": "sentry"},  # missing required fields
            None,
        ],
    })
    out = _row_to_report_content(row)
    assert out is not None
    assert len(out.evidence) == 1
    assert out.evidence[0].source == "knowledge"


def test_default_version_when_metadata_misses_it() -> None:
    row = _doc_row(metadata={
        "mode": "stub",
        "evidence": [],
    })
    out = _row_to_report_content(row)
    assert out is not None
    assert out.version == 1


def test_falsy_title_defaults_to_investigation_string() -> None:
    row = _doc_row(title="")
    out = _row_to_report_content(row)
    assert out is not None
    assert out.title == "Investigation"


@pytest.mark.parametrize("mode", ["full", "playbook_only", "stub"])
def test_all_three_modes_round_trip(mode: str) -> None:
    row = _doc_row(metadata={"mode": mode, "version": 1, "evidence": []})
    out = _row_to_report_content(row)
    assert out is not None
    assert out.mode == mode
