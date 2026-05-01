"""Profile YAML loader: must accept the minimal shape the spec describes
and reject obvious malformations early."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth.profile import (
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
preset: flagship
seed: 42
""".strip()
    )
    profile = load_profile(p)
    assert profile.customer_id == "cust-eval-prbe-01"
    assert profile.preset == "flagship"
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
preset: tiny-test
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
preset: tiny-test
seed: 1
""".strip()
    )
    with pytest.raises(ProfileError) as exc:
        load_profile(p)
    assert "cust-eval-" in str(exc.value) or "cust-synth-" in str(exc.value)
