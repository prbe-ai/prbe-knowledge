"""Tests for _load_canonical_id — the manifest resolution helper used by
seed_tenant. Pure function, no DB / bucket dependency.

Pins the round-trip: `synth run --record-llm` writes a manifest.json
that `synth seed --canonical-dir <output>` can consume directly,
without the operator hand-editing the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.synth.seed import MissingCanonicalError, _load_canonical_id


def _write_manifest(dir_path: Path, name: str, payload: dict) -> Path:
    p = dir_path / name
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def test_uppercase_manifest_with_canonical_customer_id(tmp_path: Path) -> None:
    """Hand-authored corpus convention: MANIFEST.json + canonical_customer_id."""
    _write_manifest(tmp_path, "MANIFEST.json", {"canonical_customer_id": "cust-eval-canon-1"})
    assert _load_canonical_id(tmp_path) == "cust-eval-canon-1"


def test_lowercase_manifest_from_synth_run(tmp_path: Path) -> None:
    """`synth run --record-llm` writes manifest.json (lowercase) with both
    customer_id and canonical_customer_id (alias). The canonical alias wins."""
    _write_manifest(
        tmp_path,
        "manifest.json",
        {
            "customer_id": "cust-eval-run-1",
            "canonical_customer_id": "cust-eval-run-1",
            "run_id": "x",
        },
    )
    assert _load_canonical_id(tmp_path) == "cust-eval-run-1"


def test_lowercase_manifest_with_only_customer_id_falls_back(tmp_path: Path) -> None:
    """Older recordings (or third-party manifests) may have only customer_id.
    The seeder accepts that as a fallback so they remain usable."""
    _write_manifest(tmp_path, "manifest.json", {"customer_id": "cust-eval-legacy-1"})
    assert _load_canonical_id(tmp_path) == "cust-eval-legacy-1"


def test_canonical_alias_wins_over_customer_id(tmp_path: Path) -> None:
    """If both fields are present and disagree (shouldn't happen in
    practice), canonical_customer_id wins because it's the explicit
    seed-format declaration."""
    _write_manifest(
        tmp_path,
        "manifest.json",
        {"customer_id": "cust-eval-A", "canonical_customer_id": "cust-eval-B"},
    )
    assert _load_canonical_id(tmp_path) == "cust-eval-B"


def test_missing_manifest_raises(tmp_path: Path) -> None:
    """Empty dir with no manifest of either casing → MissingCanonicalError."""
    with pytest.raises(MissingCanonicalError, match="manifest not found"):
        _load_canonical_id(tmp_path)


def test_neither_field_present_raises(tmp_path: Path) -> None:
    """Manifest exists but lacks both canonical_customer_id and customer_id."""
    _write_manifest(tmp_path, "manifest.json", {"run_id": "x", "version": "v1"})
    with pytest.raises(MissingCanonicalError, match="missing required identifier"):
        _load_canonical_id(tmp_path)


def test_canonical_customer_id_must_be_non_empty(tmp_path: Path) -> None:
    """canonical_customer_id present but empty string → fall back to customer_id
    (or raise if customer_id is also missing/empty)."""
    _write_manifest(
        tmp_path,
        "manifest.json",
        {"canonical_customer_id": "", "customer_id": "cust-eval-fallback"},
    )
    assert _load_canonical_id(tmp_path) == "cust-eval-fallback"


def test_canonical_customer_id_must_be_string(tmp_path: Path) -> None:
    """Wrong type for both fields → raise."""
    _write_manifest(
        tmp_path,
        "manifest.json",
        {"canonical_customer_id": 42, "customer_id": ["not", "a", "string"]},
    )
    with pytest.raises(MissingCanonicalError, match="missing required identifier"):
        _load_canonical_id(tmp_path)


def test_invalid_json_raises(tmp_path: Path) -> None:
    """Malformed JSON in the manifest → MissingCanonicalError with parse detail."""
    (tmp_path / "manifest.json").write_text("{not valid json", encoding="utf-8")
    with pytest.raises(MissingCanonicalError, match="not valid JSON"):
        _load_canonical_id(tmp_path)
