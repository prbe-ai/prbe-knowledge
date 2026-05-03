"""Tests for the preset loader: apply_preset() and Profile.load_profile() preset integration."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts.synth.presets.loader import PresetNotFoundError, apply_preset
from scripts.synth.profile import load_profile


def test_apply_preset_tiny_test_populates_archetypes() -> None:
    """apply_preset({}, 'tiny_test') returns dict with archetypes block from preset."""
    result = apply_preset({}, "tiny_test")
    assert "archetypes" in result
    archetypes = result["archetypes"]
    assert "INCIDENT" in archetypes
    assert archetypes["INCIDENT"]["count"] >= 1


def test_profile_archetype_wins_over_preset() -> None:
    """Profile's INCIDENT count=5 overrides preset's INCIDENT count."""
    profile_raw = {"archetypes": {"INCIDENT": {"count": 5}}}
    result = apply_preset(profile_raw, "tiny_test")
    assert result["archetypes"]["INCIDENT"]["count"] == 5
    assert "LAUNCH" in result["archetypes"]


def test_profile_llm_wins_over_preset() -> None:
    """Profile's llm.planner_model=gemini-2.5-pro overrides preset's planner_model."""
    profile_raw = {"llm": {"planner_model": "gemini-2.5-pro"}}
    result = apply_preset(profile_raw, "tiny_test")
    assert result["llm"]["planner_model"] == "gemini-2.5-pro"
    assert "writer_model" in result["llm"]


def test_apply_preset_none_returns_unchanged() -> None:
    """apply_preset(raw, None) returns raw unchanged (no mutation, same content)."""
    raw = {"seed": 42, "archetypes": {"INCIDENT": {"count": 3}}}
    result = apply_preset(raw, None)
    assert result == raw
    assert raw["archetypes"]["INCIDENT"]["count"] == 3


def test_apply_preset_unknown_name_raises() -> None:
    """apply_preset(raw, 'nonexistent') raises PresetNotFoundError."""
    with pytest.raises(PresetNotFoundError, match="nonexistent"):
        apply_preset({}, "nonexistent")


def test_load_profile_with_preset_field(tmp_path: Path) -> None:
    """load_profile succeeds when YAML contains 'preset: tiny_test'."""
    profile_yaml = textwrap.dedent("""
        customer_id: cust-eval-prbe-01
        seed: 42
        repos:
          - url: file:///tmp/fake-repo
        preset: tiny_test
    """)
    profile_file = tmp_path / "profile.yaml"
    profile_file.write_text(profile_yaml)
    profile = load_profile(profile_file)
    assert profile.archetypes.get("INCIDENT") is not None
