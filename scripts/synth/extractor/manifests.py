"""Discover and parse manifest files in a repo.

Scope: top-level + first + second-level subdirs. Skips known noise dirs
(node_modules, venv, vendor, dist, build, target).
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import yaml


class ManifestKind(StrEnum):
    PYPROJECT = "pyproject"
    PACKAGE_JSON = "package_json"
    FLY_TOML = "fly_toml"
    DOCKER_COMPOSE = "docker_compose"


@dataclass(frozen=True)
class Manifest:
    kind: ManifestKind
    path: Path
    name: str | None
    description: str | None
    dependencies: tuple[str, ...] = ()
    compose_service_names: tuple[str, ...] = ()


_SKIP_DIRS = {
    "node_modules", ".venv", "venv", ".tox", "dist", "build", "target",
    "vendor", ".git", "__pycache__", ".pytest_cache", ".ruff_cache",
}


def _candidate_dirs(root: Path) -> list[Path]:
    """Root + every first-level subdir + their children (3 levels total).
    Filters _SKIP_DIRS and dotdirs. Silently skips unreadable directories
    (PermissionError, etc.) so a single chmod-0 dir doesn't break extraction.
    """
    dirs = [root]
    try:
        children = list(root.iterdir())
    except OSError:
        return dirs
    for child in children:
        if child.is_dir() and child.name not in _SKIP_DIRS and not child.name.startswith("."):
            dirs.append(child)
            try:
                grandchildren = list(child.iterdir())
            except OSError:
                continue
            for grandchild in grandchildren:
                if (
                    grandchild.is_dir()
                    and grandchild.name not in _SKIP_DIRS
                    and not grandchild.name.startswith(".")
                ):
                    dirs.append(grandchild)
    return dirs


def _parse_pyproject(path: Path) -> Manifest | None:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    project = data.get("project") or {}
    if not project:
        return None
    deps_raw = project.get("dependencies") or []
    deps = tuple(_dep_name(d) for d in deps_raw if isinstance(d, str))
    return Manifest(
        kind=ManifestKind.PYPROJECT,
        path=path,
        name=project.get("name"),
        description=project.get("description"),
        dependencies=deps,
    )


def _parse_package_json(path: Path) -> Manifest | None:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    deps = tuple((data.get("dependencies") or {}).keys())
    return Manifest(
        kind=ManifestKind.PACKAGE_JSON,
        path=path,
        name=data.get("name"),
        description=data.get("description"),
        dependencies=deps,
    )


def _parse_fly_toml(path: Path) -> Manifest | None:
    try:
        data = tomllib.loads(path.read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return None
    return Manifest(
        kind=ManifestKind.FLY_TOML,
        path=path,
        name=data.get("app"),
        description=None,
    )


def _parse_docker_compose(path: Path) -> Manifest | None:
    try:
        data = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(data, dict):
        return None
    services = data.get("services") or {}
    if not isinstance(services, dict):
        return None
    return Manifest(
        kind=ManifestKind.DOCKER_COMPOSE,
        path=path,
        name=None,
        description=None,
        compose_service_names=tuple(services.keys()),
    )


def _dep_name(spec: str) -> str:
    """Strip versions/markers: 'pkg>=1.0' → 'pkg'."""
    for sep in (" ", ";", "[", "==", ">=", "<=", "~=", ">", "<", "!=", "@"):
        if sep in spec:
            spec = spec.split(sep, 1)[0]
    return spec.strip()


def parse_manifests_in_repo(root: Path) -> list[Manifest]:
    found: list[Manifest] = []
    for d in _candidate_dirs(root):
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for entry in entries:
            if not entry.is_file():
                continue
            name = entry.name
            if name == "pyproject.toml":
                m = _parse_pyproject(entry)
            elif name == "package.json":
                m = _parse_package_json(entry)
            elif name.startswith("fly") and name.endswith(".toml"):
                m = _parse_fly_toml(entry)
            elif name in ("docker-compose.yml", "docker-compose.yaml", "compose.yaml", "compose.yml"):
                m = _parse_docker_compose(entry)
            else:
                m = None
            if m is not None:
                found.append(m)
    return found
