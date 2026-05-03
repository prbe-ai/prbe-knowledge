"""Tests for the archetype library registry."""

from __future__ import annotations

from scripts.synth.archetypes.base import Archetype
from scripts.synth.archetypes.library import (
    ARCHETYPE_LIBRARY,
    BUILDERS,
    get_active,
)
from scripts.synth.archetypes.oncall import ON_CALL_HANDOFF, build_oncall_specs
from scripts.synth.archetypes.standup import STANDUP_UPDATE, build_standup_specs
from scripts.synth.profile import Profile


def _profile(raw: dict | None = None) -> Profile:
    """Construct a minimal Profile for tests, with overridable raw dict."""
    raw = raw or {}
    base = {
        "customer_id": "cust-eval-test-01",
        "repos": [{"url": "github.com/x/y", "local_path": "/tmp/y"}],
        "preset": "tiny_test",
        "seed": 42,
    }
    base.update(raw)
    return Profile(
        customer_id=base["customer_id"],
        repos=(),
        preset=base["preset"],
        seed=base["seed"],
        raw=base,
    )


def test_library_contains_both_archetypes() -> None:
    assert set(ARCHETYPE_LIBRARY.keys()) == {"STANDUP_UPDATE", "ON_CALL_HANDOFF"}
    assert isinstance(ARCHETYPE_LIBRARY["STANDUP_UPDATE"], Archetype)
    assert ARCHETYPE_LIBRARY["STANDUP_UPDATE"] is STANDUP_UPDATE
    assert ARCHETYPE_LIBRARY["ON_CALL_HANDOFF"] is ON_CALL_HANDOFF


def test_builders_resolve_to_correct_functions() -> None:
    assert BUILDERS["STANDUP_UPDATE"] is build_standup_specs
    assert BUILDERS["ON_CALL_HANDOFF"] is build_oncall_specs


def test_get_active_default_returns_full_library() -> None:
    p = _profile()
    active = get_active(p)
    assert set(active.keys()) == {"STANDUP_UPDATE", "ON_CALL_HANDOFF"}


def test_get_active_respects_count_zero_disable() -> None:
    p = _profile({"archetypes": {"STANDUP_UPDATE": {"count": 0}}})
    active = get_active(p)
    assert set(active.keys()) == {"ON_CALL_HANDOFF"}


def test_get_active_respects_archetype_filter() -> None:
    p = _profile()
    active = get_active(p, archetype_filter=("STANDUP_UPDATE",))
    assert set(active.keys()) == {"STANDUP_UPDATE"}


def test_get_active_filter_intersects_with_count_disable() -> None:
    p = _profile({"archetypes": {"STANDUP_UPDATE": {"count": 0}}})
    active = get_active(p, archetype_filter=("STANDUP_UPDATE", "ON_CALL_HANDOFF"))
    assert set(active.keys()) == {"ON_CALL_HANDOFF"}
