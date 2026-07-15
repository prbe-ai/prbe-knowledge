"""Unit tests for resolve_temporal — pure function, no DB."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from engine.retrieval.temporal import resolve_temporal
from engine.shared.models import TemporalMode

_NOW = datetime(2026, 4, 24, 12, 0, 0, tzinfo=UTC)


def test_resolve_returns_none_for_none_input() -> None:
    spec, err = resolve_temporal(None, _NOW)
    assert spec is None
    assert err is None


def test_resolve_returns_none_when_both_endpoints_null() -> None:
    spec, err = resolve_temporal(
        {"since": None, "until": None, "basis": "source", "raw_phrase": "irrelevant"},
        _NOW,
    )
    assert spec is None
    assert err is None


def test_resolve_relative_since() -> None:
    spec, err = resolve_temporal(
        {
            "since": {"kind": "rel", "offset_days": -30},
            "until": {"kind": "rel", "offset_days": 0},
            "basis": "source",
            "raw_phrase": "in the last month",
        },
        _NOW,
    )
    assert err is None
    assert spec is not None
    assert spec.mode == TemporalMode.CHANGED_BETWEEN
    assert spec.since == _NOW - timedelta(days=30)
    assert spec.until == _NOW
    assert spec.time_basis == "source"


def test_resolve_absolute_since_with_z_suffix() -> None:
    spec, _ = resolve_temporal(
        {
            "since": {"kind": "abs", "iso": "2024-03-15T00:00:00Z"},
            "until": None,
            "basis": "source",
            "raw_phrase": "since March 15",
        },
        _NOW,
    )
    assert spec is not None
    assert spec.since == datetime(2024, 3, 15, tzinfo=UTC)
    assert spec.until == _NOW  # open-ended → defaulted to now


def test_resolve_until_before_since_returns_none() -> None:
    spec, err = resolve_temporal(
        {
            "since": {"kind": "abs", "iso": "2024-04-01T00:00:00Z"},
            "until": {"kind": "abs", "iso": "2024-03-01T00:00:00Z"},
            "basis": "source",
            "raw_phrase": "weird",
        },
        _NOW,
    )
    assert spec is None
    assert err == "until is not after since"


def test_resolve_unresolvable_anchor_returns_error() -> None:
    spec, err = resolve_temporal(
        {
            "since": None,
            "until": None,
            "basis": "source",
            "raw_phrase": "since the auth refactor",
            "unresolvable_anchor": "the auth refactor",
        },
        _NOW,
    )
    assert spec is None
    assert err is not None
    assert "the auth refactor" in err


def test_resolve_invalid_kind_falls_through_to_none() -> None:
    spec, err = resolve_temporal(
        {
            "since": {"kind": "bogus", "value": 42},
            "until": None,
            "basis": "source",
            "raw_phrase": "garbage",
        },
        _NOW,
    )
    # Both endpoints unresolvable → both None → spec is None.
    assert spec is None
    assert err is None


def test_resolve_basis_defaults_to_source() -> None:
    spec, _ = resolve_temporal(
        {
            "since": {"kind": "rel", "offset_days": -1},
            "until": None,
            "raw_phrase": "yesterday",
            # no basis key
        },
        _NOW,
    )
    assert spec is not None
    assert spec.time_basis == "source"


def test_resolve_basis_ingest_when_specified() -> None:
    spec, _ = resolve_temporal(
        {
            "since": {"kind": "rel", "offset_days": -7},
            "until": None,
            "basis": "ingest",
            "raw_phrase": "indexed in the last week",
        },
        _NOW,
    )
    assert spec is not None
    assert spec.time_basis == "ingest"
