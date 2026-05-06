"""Profile YAML loader: must accept the minimal shape the spec describes
and reject obvious malformations early."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth.profile import (
    Profile,
    ProfileError,
    RepoSpec,
    load_profile,
)


def test_minimal_profile_loads(tmp_path: Path) -> None:
    p = tmp_path / "profile.yaml"
    p.write_text(
        """
customer_id: cust-eval-prbe-01
repos:
  - github.com/prbe-ai/prbe-knowledge
preset: tiny_test
seed: 42
""".strip()
    )
    profile = load_profile(p)
    assert isinstance(profile, Profile)
    assert profile.customer_id == "cust-eval-prbe-01"
    assert profile.preset == "tiny_test"
    assert profile.seed == 42
    assert profile.repos == (RepoSpec(url="github.com/prbe-ai/prbe-knowledge", local_path=None, branch=None),)


def test_repo_full_form_loads(tmp_path: Path) -> None:
    p = tmp_path / "profile.yaml"
    p.write_text(
        """
customer_id: cust-eval-prbe-02
repos:
  - url: github.com/prbe-ai/prbe-knowledge
    local_path: /tmp/clone
    branch: main
preset: tiny_test
seed: 7
""".strip()
    )
    profile = load_profile(p)
    assert profile.repos[0].local_path == Path("/tmp/clone")
    assert profile.repos[0].branch == "main"


def test_missing_required_field_errors(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("repos: []\nseed: 1\n")  # no customer_id, no preset
    with pytest.raises(ProfileError) as exc:
        load_profile(p)
    assert "customer_id" in str(exc.value)


def test_customer_id_prefix_enforced(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text(
        """
customer_id: cust-prod-real
repos:
  - github.com/x/y
preset: tiny_test
seed: 1
""".strip()
    )
    with pytest.raises(ProfileError) as exc:
        load_profile(p)
    assert "cust-eval-" in str(exc.value) or "cust-synth-" in str(exc.value)


def _write_profile(tmp_path: Path, extra: str = "") -> Path:
    body = (
        "customer_id: cust-eval-test\n"
        "preset: tiny_test\n"
        "seed: 1\n"
        "repos:\n"
        "  - https://github.com/acme/repo\n"
        f"{extra}"
    )
    p = tmp_path / "profile.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_load_profile_default_regen_max_rounds(tmp_path: Path) -> None:
    profile = load_profile(_write_profile(tmp_path))
    assert profile.regen_max_rounds == 3


def test_load_profile_explicit_regen_max_rounds(tmp_path: Path) -> None:
    profile = load_profile(_write_profile(tmp_path, "regen:\n  max_rounds: 5\n"))
    assert profile.regen_max_rounds == 5


def test_load_profile_regen_max_rounds_must_be_int(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="regen.max_rounds"):
        load_profile(_write_profile(tmp_path, "regen:\n  max_rounds: 'three'\n"))


def test_load_profile_regen_max_rounds_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ProfileError, match="regen.max_rounds"):
        load_profile(_write_profile(tmp_path, "regen:\n  max_rounds: 0\n"))
