"""Unit tests for incident_signals.extract_from_doc — converts a row
of `documents` (where doc_type='incident') into the orchestrator's
dispatch `incident_signals` block."""
from __future__ import annotations
from datetime import UTC, datetime

from services.investigation.incident_signals import extract_from_doc


def test_extract_pagerduty_signals() -> None:
    row = {
        "title": "Checkout service: database connection pool exhausted",
        "metadata": {
            "incident_id": "PD-INC-001",
            "current_status": "triggered",
            "urgency": "high",
            "priority": "P1",
            "service_id": "PSRV001",
            "service_url": "https://acme.pagerduty.com/services/PSRV001",
        },
        "source_system": "pagerduty",
        "created_at": datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
    }
    signals = extract_from_doc(row)
    assert signals["title"] == "Checkout service: database connection pool exhausted"
    assert signals["severity"] == "P1"
    assert signals["urgency"] == "high"
    assert signals["service"] == "PSRV001"
    assert signals["triggered_at"] == "2026-05-17T12:00:00+00:00"
    assert signals["raw_summary"] == ""
    assert signals["tags"] == []


def test_extract_incident_io_signals() -> None:
    row = {
        "title": "Checkout payments degraded",
        "metadata": {
            "severity": "Major",
            "current_status": "Investigating",
            "service_tag": "checkout-svc",
            "reference": "INC-42",
            "workspace_id": "ws_01XYZ",
        },
        "source_system": "incident_io",
        "created_at": "2026-05-17T12:00:00Z",
    }
    signals = extract_from_doc(row)
    assert signals["title"] == "Checkout payments degraded"
    assert signals["severity"] == "Major"
    assert signals["urgency"] is None
    assert signals["service"] == "checkout-svc"
    assert signals["triggered_at"] == "2026-05-17T12:00:00Z"


def test_extract_handles_missing_metadata_fields() -> None:
    row = {
        "title": "minimal",
        "metadata": {},
        "source_system": "pagerduty",
        "created_at": datetime(2026, 5, 17, tzinfo=UTC),
    }
    signals = extract_from_doc(row)
    assert signals["title"] == "minimal"
    assert signals["severity"] is None
    assert signals["service"] is None
    assert signals["urgency"] is None


def test_extract_accepts_datetime_or_isostr_for_created_at() -> None:
    """`created_at` from asyncpg is datetime; tests may also use raw strings."""
    row_dt = {
        "title": "x", "metadata": {}, "source_system": "pagerduty",
        "created_at": datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC),
    }
    row_str = {
        "title": "x", "metadata": {}, "source_system": "pagerduty",
        "created_at": "2026-05-17T12:00:00Z",
    }
    assert extract_from_doc(row_dt)["triggered_at"] == "2026-05-17T12:00:00+00:00"
    assert extract_from_doc(row_str)["triggered_at"] == "2026-05-17T12:00:00Z"


def test_extract_unknown_source_falls_back_gracefully() -> None:
    """An unrecognized source_system shouldn't crash; we use best-effort
    field lookups."""
    row = {
        "title": "x",
        "metadata": {"severity": "P3"},
        "source_system": "future-source",
        "created_at": datetime(2026, 5, 17, tzinfo=UTC),
    }
    signals = extract_from_doc(row)
    # No specific extraction rules; should not raise, returns title + None defaults.
    assert signals["title"] == "x"
    assert signals["service"] is None
