"""Manifest parsing. Used to discover service names, descriptions, and
dependencies from a repo's manifest files."""

from __future__ import annotations

from pathlib import Path

from scripts.synth.extractor.manifests import (
    Manifest,  # noqa: F401 - re-exported for callers importing from this test module
    ManifestKind,
    parse_manifests_in_repo,
)


def test_parses_pyproject(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """
[project]
name = "payments-api"
description = "Handles payment processing"
dependencies = ["fastapi", "shared-billing>=2.0"]
""".strip()
    )
    manifests = parse_manifests_in_repo(tmp_path)
    assert len(manifests) == 1
    m = manifests[0]
    assert m.kind == ManifestKind.PYPROJECT
    assert m.name == "payments-api"
    assert m.description == "Handles payment processing"
    assert "fastapi" in m.dependencies
    assert "shared-billing" in m.dependencies


def test_parses_package_json(tmp_path: Path) -> None:
    (tmp_path / "package.json").write_text(
        """{"name":"checkout-frontend","description":"Checkout UI","dependencies":{"react":"^18.0","payments-sdk":"^1.0"}}"""
    )
    manifests = parse_manifests_in_repo(tmp_path)
    [m] = manifests
    assert m.kind == ManifestKind.PACKAGE_JSON
    assert m.name == "checkout-frontend"
    assert "react" in m.dependencies
    assert "payments-sdk" in m.dependencies


def test_parses_fly_toml(tmp_path: Path) -> None:
    (tmp_path / "fly.api.toml").write_text(
        """
app = "prbe-knowledge-api"

[build]
image = "ghcr.io/prbe/api:latest"
""".strip()
    )
    [m] = parse_manifests_in_repo(tmp_path)
    assert m.kind == ManifestKind.FLY_TOML
    assert m.name == "prbe-knowledge-api"


def test_parses_docker_compose(tmp_path: Path) -> None:
    (tmp_path / "docker-compose.yml").write_text(
        """
services:
  api:
    image: ghcr.io/prbe/api
  worker:
    image: ghcr.io/prbe/worker
""".strip()
    )
    [m] = parse_manifests_in_repo(tmp_path)
    assert m.kind == ManifestKind.DOCKER_COMPOSE
    assert sorted(m.compose_service_names) == ["api", "worker"]


def test_walks_one_level_of_subdirs(tmp_path: Path) -> None:
    (tmp_path / "services").mkdir()
    (tmp_path / "services" / "payments").mkdir()
    (tmp_path / "services" / "payments" / "pyproject.toml").write_text(
        '[project]\nname = "payments"\n'
    )
    (tmp_path / "services" / "billing").mkdir()
    (tmp_path / "services" / "billing" / "package.json").write_text('{"name":"billing"}')

    manifests = parse_manifests_in_repo(tmp_path)
    names = {m.name for m in manifests}
    assert names == {"payments", "billing"}


def test_ignores_node_modules_and_venv(tmp_path: Path) -> None:
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / "package.json").write_text('{"name":"x"}')
    (tmp_path / ".venv" / "site-packages" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "site-packages" / "lib" / "pyproject.toml").write_text(
        '[project]\nname = "lib"\n'
    )
    (tmp_path / "package.json").write_text('{"name":"my-app"}')

    [m] = parse_manifests_in_repo(tmp_path)
    assert m.name == "my-app"
