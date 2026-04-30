# Synth Tool — Plan 1: Deterministic Layer + `extract` CLI

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the deterministic foundation of the synthetic-corpus generator — a tool that takes a profile YAML pointing at one or more local-clone git repos (with optional GitHub API access), extracts repo signals, merges them into an immutable `WorldModel`, augments with `CompanyContext` (yaml-loaded or LLM-inferred), and exposes a `python -m scripts.synth extract` subcommand that dumps `world_model.json` for inspection. No DB writes, no narrative-layer code. Working & testable on its own.

**Architecture:** Single Python package at `scripts/synth/` with three concentric layers — `extractor/` (per-repo, no LLM), `world_model.py` + `company_context.py` (merge / augment), `cli.py` (argparse dispatch). All extraction is cacheable to `~/.cache/prbe-synth/` keyed by repo SHA. The LLM client is a thin wrapper around the existing `anthropic>=0.40.0` dep; mock-mode is a swappable client used in tests.

**Tech Stack:** Python 3.12, pydantic v2 / pydantic-settings, anthropic SDK, httpx (for GitHub API), orjson, tenacity, structlog, pytest + pytest-asyncio (already in `pyproject.toml`). Real `git` subprocess for git-log walking. Tiny on-disk git repos in pytest tmp dirs for integration tests.

---

## File structure

**New files (Plan 1):**

```
scripts/__init__.py                      # make scripts/ an explicit package
scripts/synth/__init__.py
scripts/synth/__main__.py                # CLI entry point: python -m scripts.synth ...
scripts/synth/cli.py                     # argparse dispatch (only `extract` in plan 1)
scripts/synth/profile.py                 # YAML loader + Profile dataclass
scripts/synth/cache.py                   # disk KV cache by SHA-keyed paths
scripts/synth/llm_client.py              # anthropic SDK wrapper (basic; full mock mode in plan 3)
scripts/synth/company_context.py         # YAML loader + auto-inferrer
scripts/synth/world_model.py             # WorldModel dataclasses + WorldModelMerger
scripts/synth/extractor/__init__.py
scripts/synth/extractor/repo.py          # RepoExtractor orchestrator + RepoSignals
scripts/synth/extractor/git_log.py       # git subprocess: commits, branches
scripts/synth/extractor/manifests.py     # pyproject/package.json/fly.toml/docker-compose
scripts/synth/extractor/codeowners.py    # CODEOWNERS parser
scripts/synth/extractor/github_api.py    # httpx client for issues/PRs/contributors

tests/synth/__init__.py
tests/synth/conftest.py                  # tmp git repo fixture
tests/synth/test_profile.py
tests/synth/test_cache.py
tests/synth/test_llm_client.py
tests/synth/test_extractor_manifests.py
tests/synth/test_extractor_codeowners.py
tests/synth/test_extractor_git_log.py
tests/synth/test_extractor_github_api.py
tests/synth/test_extractor_repo.py
tests/synth/test_world_model_merger.py
tests/synth/test_company_context.py
tests/synth/test_extract_cli.py
```

**Modified files: none** (scripts/ has no `__init__.py` today; the new file is what makes it a real package).

---

## Conventions

- Every dataclass that should be hashable / cached uses `frozen=True`. Mutable working buffers use plain dataclasses.
- All file paths in code are `pathlib.Path`, not strings.
- Logging via the existing `structlog` setup (`shared.logging.get_logger(__name__)`) — no `print` in library code.
- Async only where it pays for itself: GitHub API client (httpx), LLM client (anthropic AsyncAnthropic). Everything else (file walks, git subprocess, parsing) is sync.
- Subprocess calls use `subprocess.run(..., check=True, capture_output=True, text=True)` — no shells.
- Tests use `pytest-asyncio` auto mode (already configured in `pyproject.toml`).
- Commit per task. Conventional-commit style matching existing history (e.g., `feat(synth): add manifest parser`).

---

## Task 1: Bootstrap `scripts/synth/` package skeleton

**Files:**
- Create: `scripts/__init__.py`
- Create: `scripts/synth/__init__.py`
- Create: `scripts/synth/__main__.py`
- Create: `scripts/synth/cli.py`
- Create: `tests/synth/__init__.py`
- Create: `tests/synth/test_smoke.py`

- [ ] **Step 1: Write the failing smoke test**

Create `tests/synth/__init__.py` (empty) and `tests/synth/test_smoke.py`:

```python
"""Smoke test: the synth CLI is importable and prints help."""

from __future__ import annotations

import subprocess
import sys


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "scripts.synth", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "synth" in result.stdout.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synth/test_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.synth'` (or similar — the module doesn't exist yet).

- [ ] **Step 3: Create the package files**

Create `scripts/__init__.py` with content:

```python
"""prbe-knowledge ops scripts (CLI tools, seeders, sync helpers).

This file exists so submodules like `scripts.synth` are unambiguously
importable. Individual scripts (e.g. `scripts.seed_synthetic`) work either
way under Python 3.12's implicit namespace packages, but explicit beats
implicit when nesting subpackages.
"""
```

Create `scripts/synth/__init__.py`:

```python
"""Synthetic company-corpus generator.

See docs/superpowers/specs/2026-04-30-synthetic-company-eval-design.md
for the design.
"""
```

Create `scripts/synth/__main__.py`:

```python
"""Entry point: `python -m scripts.synth ...` dispatches to cli.main."""

from __future__ import annotations

from scripts.synth.cli import main

if __name__ == "__main__":
    main()
```

Create `scripts/synth/cli.py`:

```python
"""CLI dispatch for the synth tool. Subcommands grow over plans 1-3."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.synth",
        description="Synthetic company corpus generator for prbe-knowledge eval datasets.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser(
        "extract",
        help="Extract WorldModel from repos in a profile (no DB writes).",
    )
    extract.add_argument("--profile", required=True, type=str, help="Path to profile YAML.")
    extract.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write world_model.json (default: eval-datasets/<run-id>/).",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "extract":
        # Plan 1 task 26 wires this in. For now, surface a clear stub error.
        print("extract: not yet implemented", file=sys.stderr)
        return 2
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synth/test_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/synth/__init__.py scripts/synth/__main__.py scripts/synth/cli.py tests/synth/__init__.py tests/synth/test_smoke.py
git commit -m "$(cat <<'EOF'
feat(synth): scaffold scripts/synth/ package + extract subcommand stub

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Profile loader + dataclass

**Files:**
- Create: `scripts/synth/profile.py`
- Test: `tests/synth/test_profile.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_profile.py`:

```python
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
preset: flagship
seed: 42
""".strip()
    )
    profile = load_profile(p)
    assert profile.customer_id == "cust-eval-prbe-01"
    assert profile.preset == "flagship"
    assert profile.seed == 42
    assert profile.repos == [RepoSpec(url="github.com/prbe-ai/prbe-knowledge", local_path=None, branch=None)]


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_profile.py -v`
Expected: FAIL with ImportError on `scripts.synth.profile`.

- [ ] **Step 3: Create the profile module**

Create `scripts/synth/profile.py`:

```python
"""Profile YAML loader. The profile is the unit of "an eval dataset
configuration" — it points at repos, names the preset, sets the seed.

v1 surface is intentionally minimal: only fields we use this plan.
Plan 3 will extend this with archetype overrides, time_window, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml  # noqa: F401  # via pyyaml — already pulled in by pydantic; if not, will be added in plan 1 task 27.


class ProfileError(ValueError):
    """Raised when a profile YAML is missing required fields or otherwise
    malformed in a way the user can fix."""


@dataclass(frozen=True)
class RepoSpec:
    url: str
    local_path: Path | None
    branch: str | None


@dataclass(frozen=True)
class Profile:
    customer_id: str
    repos: tuple[RepoSpec, ...]
    preset: str
    seed: int
    company_context_path: Path | None = None
    raw: dict = field(default_factory=dict)  # full YAML for plan 3 to consume


_VALID_PREFIXES = ("cust-eval-", "cust-synth-")


def _normalize_repo(entry: object) -> RepoSpec:
    if isinstance(entry, str):
        return RepoSpec(url=entry, local_path=None, branch=None)
    if isinstance(entry, dict):
        url = entry.get("url")
        if not url or not isinstance(url, str):
            raise ProfileError(f"repo entry missing 'url': {entry!r}")
        lp = entry.get("local_path")
        return RepoSpec(
            url=url,
            local_path=Path(lp).expanduser() if lp else None,
            branch=entry.get("branch"),
        )
    raise ProfileError(f"repo entry must be a string or mapping, got {type(entry).__name__}")


def load_profile(path: Path) -> Profile:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ProfileError(f"profile must be a YAML mapping, got {type(raw).__name__}")

    missing = [k for k in ("customer_id", "repos", "preset", "seed") if k not in raw]
    if missing:
        raise ProfileError(f"profile missing required fields: {sorted(missing)}")

    customer_id = raw["customer_id"]
    if not isinstance(customer_id, str) or not customer_id.startswith(_VALID_PREFIXES):
        raise ProfileError(
            f"customer_id must start with one of {_VALID_PREFIXES} "
            f"(refusing to operate on production-shaped tenant): {customer_id!r}"
        )

    repos_raw = raw["repos"]
    if not isinstance(repos_raw, list) or not repos_raw:
        raise ProfileError("repos must be a non-empty list")
    repos = tuple(_normalize_repo(r) for r in repos_raw)

    seed = raw["seed"]
    if not isinstance(seed, int):
        raise ProfileError(f"seed must be an integer, got {type(seed).__name__}")

    cc = raw.get("company_context")
    cc_path = Path(cc).expanduser() if isinstance(cc, str) else None

    return Profile(
        customer_id=customer_id,
        repos=repos,
        preset=raw["preset"],
        seed=seed,
        company_context_path=cc_path,
        raw=raw,
    )
```

If `pyyaml` is not yet listed in `pyproject.toml`, add it:

```bash
# Verify dependency
grep -q "pyyaml\|PyYAML" pyproject.toml || echo "ADD pyyaml>=6.0 TO pyproject.toml [project] dependencies"
```

If the grep returns the "ADD" line, edit `pyproject.toml` to add `"pyyaml>=6.0",` to the `dependencies` list (alongside the other deps), then `.venv/bin/pip install -e ".[dev]"` to install.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_profile.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/profile.py tests/synth/test_profile.py pyproject.toml
git commit -m "$(cat <<'EOF'
feat(synth): profile YAML loader with customer_id prefix guard

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: On-disk cache scaffolding

**Files:**
- Create: `scripts/synth/cache.py`
- Test: `tests/synth/test_cache.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_cache.py`:

```python
"""Disk-backed KV cache for repo signals + worldmodels.

The cache key is a string; values are arbitrary JSON-serializable dicts.
Cache hits return the same value byte-for-byte; misses return None."""

from __future__ import annotations

from pathlib import Path

from scripts.synth.cache import DiskCache


def test_get_returns_none_on_miss(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.get("nope") is None


def test_put_then_get_roundtrips(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    value = {"a": 1, "b": [2, 3], "c": "ok"}
    cache.put("key1", value)
    assert cache.get("key1") == value


def test_keys_with_slashes_are_safe(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("github.com/x/y@abcd1234", {"sha": "abcd1234"})
    assert cache.get("github.com/x/y@abcd1234") == {"sha": "abcd1234"}


def test_invalidate_removes_entry(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.put("k", {"v": 1})
    cache.invalidate("k")
    assert cache.get("k") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_cache.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the cache**

Create `scripts/synth/cache.py`:

```python
"""Disk-backed key-value cache. Used to memoize repo extraction +
WorldModel merges + LLM responses across runs.

Keys are strings (any printable). Values are JSON-serializable.
Storage layout: each entry is a single .json file under the root,
with the key hashed to produce the filename so slashes / special chars
in keys don't matter.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import orjson


class DiskCache:
    """File-per-entry KV cache. Atomic on POSIX (rename is atomic).

    Not concurrency-safe across processes; we run synth single-process.
    """

    def __init__(self, root: Path) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return self._root / f"{digest}.json"

    def get(self, key: str) -> dict | None:
        p = self._path(key)
        if not p.exists():
            return None
        return orjson.loads(p.read_bytes())

    def put(self, key: str, value: dict) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_bytes(orjson.dumps(value))
        tmp.replace(p)

    def invalidate(self, key: str) -> None:
        p = self._path(key)
        p.unlink(missing_ok=True)


def default_cache_root(subdir: str) -> Path:
    """Return the canonical cache directory for a synth subsystem.

    `subdir` is one of {"repos", "worldmodel", "llm"}.
    """
    return Path.home() / ".cache" / "prbe-synth" / subdir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_cache.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/cache.py tests/synth/test_cache.py
git commit -m "$(cat <<'EOF'
feat(synth): disk-backed JSON cache for repo + worldmodel + LLM caches

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Manifest parser (pyproject.toml + package.json + fly.toml + docker-compose.yml)

**Files:**
- Create: `scripts/synth/extractor/__init__.py`
- Create: `scripts/synth/extractor/manifests.py`
- Test: `tests/synth/test_extractor_manifests.py`

- [ ] **Step 1: Write the failing tests**

Create `scripts/synth/extractor/__init__.py` (empty).

Create `tests/synth/test_extractor_manifests.py`:

```python
"""Manifest parsing. Used to discover service names, descriptions, and
dependencies from a repo's manifest files."""

from __future__ import annotations

from pathlib import Path

from scripts.synth.extractor.manifests import (
    Manifest,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_extractor_manifests.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the parser**

Create `scripts/synth/extractor/manifests.py`:

```python
"""Discover and parse manifest files in a repo.

Scope: top-level + first-level subdirs. Skips known noise dirs
(node_modules, venv, vendor, dist, build, target).
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
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
    """Root + every first-level subdir that isn't in _SKIP_DIRS."""
    dirs = [root]
    for child in root.iterdir():
        if child.is_dir() and child.name not in _SKIP_DIRS and not child.name.startswith("."):
            dirs.append(child)
            for grandchild in child.iterdir():
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
    for sep in (";", "[", "==", ">=", "<=", "~=", ">", "<", "!="):
        if sep in spec:
            spec = spec.split(sep, 1)[0]
    return spec.strip()


def parse_manifests_in_repo(root: Path) -> list[Manifest]:
    found: list[Manifest] = []
    for d in _candidate_dirs(root):
        for entry in d.iterdir():
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_extractor_manifests.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/extractor/__init__.py scripts/synth/extractor/manifests.py tests/synth/test_extractor_manifests.py
git commit -m "$(cat <<'EOF'
feat(synth): manifest parser (pyproject, package.json, fly.toml, compose)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: CODEOWNERS parser

**Files:**
- Create: `scripts/synth/extractor/codeowners.py`
- Test: `tests/synth/test_extractor_codeowners.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_extractor_codeowners.py`:

```python
"""CODEOWNERS — used to map service paths to canonical Person ids."""

from __future__ import annotations

from pathlib import Path

from scripts.synth.extractor.codeowners import (
    CodeownerRule,
    find_codeowners_file,
    parse_codeowners,
    resolve_owners,
)


def test_parses_basic_rules() -> None:
    text = """
# infra
/infra/  @alice @bob
/services/payments/   @payments-team
*.md  @docs
""".strip()
    rules = parse_codeowners(text)
    assert rules == (
        CodeownerRule(pattern="/infra/", owners=("@alice", "@bob")),
        CodeownerRule(pattern="/services/payments/", owners=("@payments-team",)),
        CodeownerRule(pattern="*.md", owners=("@docs",)),
    )


def test_skips_blank_and_comment_lines() -> None:
    text = """
# only comments
   # indented comment

   /a/  @x
""".strip()
    [rule] = parse_codeowners(text)
    assert rule == CodeownerRule(pattern="/a/", owners=("@x",))


def test_handles_no_owners_line() -> None:
    """A line with a pattern and no owners is treated as 'unowned' (zero owners)."""
    text = "/legacy/\n"
    [rule] = parse_codeowners(text)
    assert rule.pattern == "/legacy/"
    assert rule.owners == ()


def test_resolve_owners_picks_last_matching_rule() -> None:
    rules = (
        CodeownerRule(pattern="*", owners=("@everyone",)),
        CodeownerRule(pattern="/services/payments/", owners=("@payments-team",)),
    )
    assert resolve_owners("/services/payments/handler.py", rules) == ("@payments-team",)
    assert resolve_owners("/random.py", rules) == ("@everyone",)


def test_finds_codeowners_in_canonical_locations(tmp_path: Path) -> None:
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "CODEOWNERS").write_text("/x  @y\n")
    assert find_codeowners_file(tmp_path) == tmp_path / ".github" / "CODEOWNERS"


def test_finds_codeowners_at_root(tmp_path: Path) -> None:
    (tmp_path / "CODEOWNERS").write_text("/x  @y\n")
    assert find_codeowners_file(tmp_path) == tmp_path / "CODEOWNERS"


def test_returns_none_when_no_codeowners(tmp_path: Path) -> None:
    assert find_codeowners_file(tmp_path) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_extractor_codeowners.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the parser**

Create `scripts/synth/extractor/codeowners.py`:

```python
"""CODEOWNERS file parsing.

Format reference: https://docs.github.com/en/repositories/managing-your-repositories-settings-and-features/customizing-your-repository/about-code-owners
We don't need exhaustive correctness — just enough to map repo paths to
@-prefixed handles for service-owner resolution.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CodeownerRule:
    pattern: str
    owners: tuple[str, ...]


_CANONICAL_LOCATIONS = (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS")


def find_codeowners_file(repo_root: Path) -> Path | None:
    for loc in _CANONICAL_LOCATIONS:
        p = repo_root / loc
        if p.is_file():
            return p
    return None


def parse_codeowners(text: str) -> tuple[CodeownerRule, ...]:
    rules: list[CodeownerRule] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comment (rare but legal):
        if "#" in line:
            line = line.split("#", 1)[0].rstrip()
            if not line:
                continue
        parts = line.split()
        pattern = parts[0]
        owners = tuple(p for p in parts[1:] if p.startswith("@"))
        rules.append(CodeownerRule(pattern=pattern, owners=owners))
    return tuple(rules)


def resolve_owners(path: str, rules: tuple[CodeownerRule, ...]) -> tuple[str, ...]:
    """Return the owners of `path` per CODEOWNERS semantics: last matching rule wins."""
    matched: tuple[str, ...] = ()
    for rule in rules:
        if _matches(path, rule.pattern):
            matched = rule.owners
    return matched


def _matches(path: str, pattern: str) -> bool:
    # Anchored prefix:  /foo/  matches  /foo/anything
    if pattern.startswith("/"):
        return path.startswith(pattern)
    # Glob:  *.md  matches  any/path/x.md
    return fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(Path(path).name, pattern)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_extractor_codeowners.py -v`
Expected: PASS — 7 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/extractor/codeowners.py tests/synth/test_extractor_codeowners.py
git commit -m "$(cat <<'EOF'
feat(synth): CODEOWNERS parser + path-to-owners resolver

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Tmp git repo fixture

**Files:**
- Create: `tests/synth/conftest.py`

- [ ] **Step 1: Write the failing test**

Add a smoke test inside the (about to be created) conftest itself? No — write it in a separate test file so the fixture is exercised through the normal pytest path.

Create `tests/synth/test_conftest_fixture.py`:

```python
"""Smoke-test the tmp git repo fixture: it should produce a real
git repo with the expected commits and authors."""

from __future__ import annotations

import subprocess
from pathlib import Path


def test_tmp_repo_has_commits(tmp_repo: Path) -> None:
    log = subprocess.run(
        ["git", "-C", str(tmp_repo), "log", "--pretty=%H %ae"],
        check=True, capture_output=True, text=True,
    )
    lines = [ln for ln in log.stdout.splitlines() if ln.strip()]
    assert len(lines) >= 4  # multiple commits


def test_tmp_repo_has_distinct_authors(tmp_repo: Path) -> None:
    log = subprocess.run(
        ["git", "-C", str(tmp_repo), "log", "--pretty=%ae"],
        check=True, capture_output=True, text=True,
    )
    authors = {ln for ln in log.stdout.splitlines() if ln.strip()}
    assert len(authors) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synth/test_conftest_fixture.py -v`
Expected: FAIL — `fixture 'tmp_repo' not found`.

- [ ] **Step 3: Create the fixture**

Create `tests/synth/conftest.py`:

```python
"""Shared fixtures for synth tests.

`tmp_repo` builds a tiny but realistic git repo in a tmp dir:
    - 2 services in `services/payments/` and `services/billing/`
    - Manifests (pyproject)
    - CODEOWNERS file
    - README at root + per-service
    - 6 commits across 3 distinct authors
    - 1 feature branch
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str, env_extra: dict[str, str] | None = None) -> None:
    env = os.environ.copy()
    env.update({
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    })
    if env_extra:
        env.update(env_extra)
    subprocess.run(["git", "-C", str(repo), *args], check=True, env=env, capture_output=True)


def _commit(repo: Path, files: dict[str, str], message: str, author: str, email: str, date: str) -> None:
    for path, content in files.items():
        full = repo / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
    _git(repo, "add", "-A")
    _git(
        repo,
        "commit",
        "-m",
        message,
        env_extra={
            "GIT_AUTHOR_NAME": author,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": author,
            "GIT_COMMITTER_EMAIL": email,
            "GIT_AUTHOR_DATE": date,
            "GIT_COMMITTER_DATE": date,
        },
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "fake-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "commit.gpgsign", "false")

    _commit(
        repo,
        {
            "README.md": "# fake-repo\n\nA tiny test repo for synth.\n",
            "pyproject.toml": (
                '[project]\nname = "fake-repo"\ndescription = "Test repo"\n'
            ),
            ".github/CODEOWNERS": (
                "/services/payments/  @alice\n"
                "/services/billing/   @bob @alice\n"
            ),
        },
        "feat: initial commit",
        "Alice", "alice@example.com",
        "2026-01-15T09:00:00",
    )

    _commit(
        repo,
        {"services/payments/pyproject.toml": '[project]\nname = "payments"\n',
         "services/payments/main.py": "def handle(): pass\n"},
        "feat(payments): scaffold service",
        "Alice", "alice@example.com",
        "2026-02-01T10:00:00",
    )

    _commit(
        repo,
        {"services/billing/pyproject.toml": '[project]\nname = "billing"\n',
         "services/billing/main.py": "def invoice(): pass\n"},
        "feat(billing): scaffold service",
        "Bob", "bob@example.com",
        "2026-02-15T14:30:00",
    )

    _commit(
        repo,
        {"services/payments/main.py": "def handle():\n    return 'ok'\n"},
        "fix(payments): return ok",
        "Alice", "alice@example.com",
        "2026-03-01T11:00:00",
    )

    _commit(
        repo,
        {"services/billing/invoice.py": "def total(): return 0\n"},
        "feat(billing): add invoice totaling",
        "Carol", "carol@example.com",
        "2026-03-20T16:00:00",
    )

    _git(repo, "checkout", "-b", "feat/payments-refund")
    _commit(
        repo,
        {"services/payments/refund.py": "def refund(): pass\n"},
        "wip(payments): refund logic",
        "Alice", "alice@example.com",
        "2026-04-01T09:00:00",
    )
    _git(repo, "checkout", "main")

    return repo
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_conftest_fixture.py -v`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add tests/synth/conftest.py tests/synth/test_conftest_fixture.py
git commit -m "$(cat <<'EOF'
test(synth): tmp git repo fixture for extractor + worldmodel tests

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Git log / branch walker

**Files:**
- Create: `scripts/synth/extractor/git_log.py`
- Test: `tests/synth/test_extractor_git_log.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_extractor_git_log.py`:

```python
"""Walk git log + branches via subprocess. Returns Commits + Branches."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from scripts.synth.extractor.git_log import (
    walk_branches,
    walk_commits,
)


def test_walk_commits_returns_recent_commits(tmp_repo: Path) -> None:
    cutoff = datetime(2026, 1, 1, tzinfo=UTC)
    commits = walk_commits(tmp_repo, since=cutoff)
    assert len(commits) >= 5
    subjects = [c.subject for c in commits]
    assert any("scaffold service" in s for s in subjects)


def test_walk_commits_respects_since(tmp_repo: Path) -> None:
    # Cutoff after most fixture commits
    cutoff = datetime(2026, 4, 30, tzinfo=UTC)
    commits = walk_commits(tmp_repo, since=cutoff)
    assert commits == []


def test_walk_commits_captures_files_touched(tmp_repo: Path) -> None:
    commits = walk_commits(tmp_repo, since=datetime(2026, 1, 1, tzinfo=UTC))
    payments_commits = [c for c in commits if "payments" in c.subject.lower()]
    assert payments_commits
    for c in payments_commits:
        assert any("services/payments" in f for f in c.files_touched)


def test_walk_branches_lists_local_branches(tmp_repo: Path) -> None:
    branches = walk_branches(tmp_repo)
    names = [b.name for b in branches]
    assert "main" in names
    assert "feat/payments-refund" in names


def test_walk_branches_records_last_commit_ts(tmp_repo: Path) -> None:
    branches = walk_branches(tmp_repo)
    feat = next(b for b in branches if b.name == "feat/payments-refund")
    # Branch tip commit was authored 2026-04-01
    assert feat.last_commit_ts >= datetime(2026, 3, 31, tzinfo=UTC)
    assert feat.last_commit_ts <= datetime(2026, 4, 2, tzinfo=UTC) + timedelta(days=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_extractor_git_log.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement git_log walker**

Create `scripts/synth/extractor/git_log.py`:

```python
"""Walk git history via the `git` CLI.

We shell out rather than use a library: `git` is on every dev machine
and the output is stable and easy to parse with a custom format.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Commit:
    sha: str
    author_name: str
    author_email: str
    ts: datetime
    subject: str
    body: str
    files_touched: tuple[str, ...]


@dataclass(frozen=True)
class Branch:
    name: str
    last_commit_sha: str
    last_commit_ts: datetime


# Custom delimiter — \x1f is ASCII unit separator, won't show up in git output.
_FIELD = "\x1f"
_RECORD = "\x1e"
_FORMAT = (
    f"%H{_FIELD}%an{_FIELD}%ae{_FIELD}%aI{_FIELD}%s{_FIELD}%b{_RECORD}"
)


def _run(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def walk_commits(repo: Path, since: datetime, max_count: int = 1000) -> list[Commit]:
    """Return commits authored at-or-after `since`, newest first.

    Includes files-touched per commit (one extra `git show` per commit;
    cheap on small repos, acceptable for the recency-bounded list we use).
    """
    raw = _run(
        repo,
        "log",
        f"--since={since.isoformat()}",
        f"--max-count={max_count}",
        f"--pretty=format:{_FORMAT}",
    )
    commits: list[Commit] = []
    for record in raw.split(_RECORD):
        record = record.strip("\n")
        if not record:
            continue
        parts = record.split(_FIELD)
        if len(parts) < 6:
            continue
        sha, name, email, ts_iso, subject, body = parts[:6]
        files = _files_touched(repo, sha)
        commits.append(
            Commit(
                sha=sha,
                author_name=name,
                author_email=email,
                ts=datetime.fromisoformat(ts_iso),
                subject=subject,
                body=body.strip(),
                files_touched=files,
            )
        )
    return commits


def _files_touched(repo: Path, sha: str) -> tuple[str, ...]:
    raw = _run(repo, "show", "--no-patch", "--name-only", "--pretty=", sha)
    return tuple(p for p in raw.splitlines() if p.strip())


def walk_branches(repo: Path) -> list[Branch]:
    raw = _run(
        repo,
        "for-each-ref",
        "refs/heads/",
        "--format=%(refname:short)\x1f%(objectname)\x1f%(committerdate:iso-strict)",
    )
    out: list[Branch] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        name, sha, ts = line.split("\x1f")
        out.append(
            Branch(
                name=name,
                last_commit_sha=sha,
                last_commit_ts=datetime.fromisoformat(ts),
            )
        )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_extractor_git_log.py -v`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/extractor/git_log.py tests/synth/test_extractor_git_log.py
git commit -m "$(cat <<'EOF'
feat(synth): git log + branch walker via subprocess

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: GitHub API client (issues + PRs + contributors)

**Files:**
- Create: `scripts/synth/extractor/github_api.py`
- Test: `tests/synth/test_extractor_github_api.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_extractor_github_api.py`:

```python
"""GitHub API client. Tests use respx to mock httpx.AsyncClient."""

from __future__ import annotations

import httpx
import pytest
import respx

from scripts.synth.extractor.github_api import GithubClient, parse_repo_url


def test_parse_repo_url_https_form() -> None:
    assert parse_repo_url("github.com/prbe-ai/prbe-knowledge") == ("prbe-ai", "prbe-knowledge")
    assert parse_repo_url("https://github.com/prbe-ai/prbe-knowledge") == ("prbe-ai", "prbe-knowledge")
    assert parse_repo_url("https://github.com/prbe-ai/prbe-knowledge.git") == ("prbe-ai", "prbe-knowledge")


def test_parse_repo_url_rejects_non_github() -> None:
    with pytest.raises(ValueError):
        parse_repo_url("gitlab.com/x/y")


@pytest.mark.asyncio
async def test_fetch_contributors_returns_username_and_display() -> None:
    with respx.mock(base_url="https://api.github.com") as router:
        router.get("/repos/x/y/contributors").mock(
            return_value=httpx.Response(
                200,
                json=[
                    {"login": "alice", "id": 1, "contributions": 42},
                    {"login": "bob", "id": 2, "contributions": 17},
                ],
            )
        )
        router.get("/users/alice").mock(
            return_value=httpx.Response(200, json={"login": "alice", "name": "Alice X", "email": "alice@example.com"})
        )
        router.get("/users/bob").mock(
            return_value=httpx.Response(200, json={"login": "bob", "name": None, "email": None})
        )

        client = GithubClient(token="t")
        contributors = await client.fetch_contributors("x", "y")
        await client.close()

    assert {c.gh_username for c in contributors} == {"alice", "bob"}
    alice = next(c for c in contributors if c.gh_username == "alice")
    assert alice.display_name == "Alice X"
    assert alice.email_aliases == ("alice@example.com",)


@pytest.mark.asyncio
async def test_fetch_issues_paginates_and_strips_pull_requests() -> None:
    with respx.mock(base_url="https://api.github.com") as router:
        page1 = [
            {
                "number": 1, "title": "issue 1", "body": "b1",
                "state": "open", "labels": [{"name": "bug"}],
                "assignees": [{"login": "alice"}],
                "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-02T00:00:00Z",
            },
            {
                "number": 2, "title": "PR linked as issue", "body": "p",
                "state": "open", "labels": [],
                "assignees": [],
                "pull_request": {"url": "..."},
                "created_at": "2026-01-03T00:00:00Z", "updated_at": "2026-01-04T00:00:00Z",
            },
        ]
        router.get("/repos/x/y/issues").mock(
            return_value=httpx.Response(200, json=page1)
        )

        client = GithubClient(token="t")
        issues = await client.fetch_issues("x", "y", limit=200)
        await client.close()

    # PR-shaped entries dropped
    assert len(issues) == 1
    assert issues[0].number == 1
    assert issues[0].labels == ("bug",)


@pytest.mark.asyncio
async def test_fetch_handles_404_as_none() -> None:
    """If a repo doesn't exist or the token lacks access, fetch_* returns
    an empty list rather than crashing — the caller treats no-data the
    same way the no-token path does."""
    with respx.mock(base_url="https://api.github.com") as router:
        router.get("/repos/x/missing/contributors").mock(
            return_value=httpx.Response(404, json={"message": "Not Found"})
        )
        client = GithubClient(token="t")
        contributors = await client.fetch_contributors("x", "missing")
        await client.close()
    assert contributors == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_extractor_github_api.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the GitHub client**

Create `scripts/synth/extractor/github_api.py`:

```python
"""Async GitHub API client. Just enough to populate `RepoSignals.{issues,
prs, contributors, workflows}`. No external lib — httpx is already a dep.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx


@dataclass(frozen=True)
class Contributor:
    gh_username: str
    display_name: str | None
    email_aliases: tuple[str, ...]
    contributions: int


@dataclass(frozen=True)
class Issue:
    number: int
    title: str
    body: str
    state: str
    labels: tuple[str, ...]
    assignees: tuple[str, ...]
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class PullRequest:
    number: int
    title: str
    body: str
    state: str
    labels: tuple[str, ...]
    author: str | None
    reviewers: tuple[str, ...]
    files_changed: tuple[str, ...]
    created_at: datetime
    merged_at: datetime | None


@dataclass(frozen=True)
class Workflow:
    name: str
    last_run_status: str | None
    last_run_at: datetime | None


_REPO_RE = re.compile(r"(?:https?://)?github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?$")


def parse_repo_url(url: str) -> tuple[str, str]:
    """github.com/owner/repo (with or without https:// and .git) → (owner, repo)."""
    m = _REPO_RE.match(url.strip())
    if not m:
        raise ValueError(f"not a recognized github URL: {url!r}")
    return m.group(1), m.group(2)


class GithubClient:
    def __init__(self, token: str | None, base_url: str = "https://api.github.com") -> None:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any | None:
        try:
            r = await self._http.get(path, params=params)
        except httpx.HTTPError:
            return None
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    async def fetch_contributors(self, owner: str, repo: str, limit: int = 100) -> list[Contributor]:
        rows = await self._get(f"/repos/{owner}/{repo}/contributors", {"per_page": limit})
        if not rows:
            return []
        out: list[Contributor] = []
        for row in rows:
            login = row.get("login")
            if not login:
                continue
            user = await self._get(f"/users/{login}") or {}
            email = user.get("email")
            out.append(
                Contributor(
                    gh_username=login,
                    display_name=user.get("name"),
                    email_aliases=(email,) if email else (),
                    contributions=row.get("contributions") or 0,
                )
            )
        return out

    async def fetch_issues(self, owner: str, repo: str, limit: int = 200) -> list[Issue]:
        rows = await self._get(
            f"/repos/{owner}/{repo}/issues",
            {"state": "all", "per_page": min(limit, 100)},
        )
        if not rows:
            return []
        issues: list[Issue] = []
        for row in rows:
            if "pull_request" in row:  # GH lumps PRs into /issues
                continue
            issues.append(
                Issue(
                    number=row["number"],
                    title=row.get("title") or "",
                    body=row.get("body") or "",
                    state=row.get("state") or "open",
                    labels=tuple(lbl["name"] for lbl in row.get("labels", []) if lbl.get("name")),
                    assignees=tuple(a["login"] for a in row.get("assignees", []) if a.get("login")),
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(row["updated_at"].replace("Z", "+00:00")),
                )
            )
            if len(issues) >= limit:
                break
        return issues

    async def fetch_prs(self, owner: str, repo: str, limit: int = 200) -> list[PullRequest]:
        rows = await self._get(
            f"/repos/{owner}/{repo}/pulls",
            {"state": "all", "per_page": min(limit, 100)},
        )
        if not rows:
            return []
        prs: list[PullRequest] = []
        for row in rows:
            merged_at_raw = row.get("merged_at")
            prs.append(
                PullRequest(
                    number=row["number"],
                    title=row.get("title") or "",
                    body=row.get("body") or "",
                    state=row.get("state") or "open",
                    labels=tuple(lbl["name"] for lbl in row.get("labels", []) if lbl.get("name")),
                    author=(row.get("user") or {}).get("login"),
                    reviewers=tuple(
                        u["login"] for u in row.get("requested_reviewers", []) if u.get("login")
                    ),
                    files_changed=(),  # cost-tradeoff: separate /files call; skip in v1
                    created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
                    merged_at=(
                        datetime.fromisoformat(merged_at_raw.replace("Z", "+00:00"))
                        if merged_at_raw
                        else None
                    ),
                )
            )
            if len(prs) >= limit:
                break
        return prs

    async def fetch_workflows(self, owner: str, repo: str) -> list[Workflow]:
        rows = await self._get(f"/repos/{owner}/{repo}/actions/workflows")
        if not rows:
            return []
        out: list[Workflow] = []
        for row in rows.get("workflows", []) or []:
            out.append(
                Workflow(
                    name=row.get("name") or row.get("path") or "",
                    last_run_status=row.get("state"),
                    last_run_at=None,  # full run history is heavy; skip in v1
                )
            )
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_extractor_github_api.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/extractor/github_api.py tests/synth/test_extractor_github_api.py
git commit -m "$(cat <<'EOF'
feat(synth): GitHub API client (issues, PRs, contributors, workflows)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: RepoSignals + RepoExtractor orchestrator

**Files:**
- Create: `scripts/synth/extractor/repo.py`
- Test: `tests/synth/test_extractor_repo.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_extractor_repo.py`:

```python
"""End-to-end (local-only) RepoExtractor: walks a real tmp git repo,
returns RepoSignals."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.extractor.repo import RepoExtractor, RepoSignals


def test_extracts_local_signals(tmp_repo: Path) -> None:
    extractor = RepoExtractor(github_client=None)

    signals = extractor.extract_local(
        url=f"file://{tmp_repo}",
        clone_path=tmp_repo,
        since=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert isinstance(signals, RepoSignals)
    assert signals.url == f"file://{tmp_repo}"
    assert signals.latest_sha  # non-empty
    assert signals.default_branch == "main"

    # Manifests
    manifest_names = {m.name for m in signals.manifests if m.name}
    assert {"fake-repo", "payments", "billing"} <= manifest_names

    # CODEOWNERS
    assert signals.codeowners
    assert any("payments" in r.pattern for r in signals.codeowners)

    # Commits
    assert len(signals.commits) >= 5

    # Branches
    branch_names = {b.name for b in signals.branches}
    assert {"main", "feat/payments-refund"} <= branch_names

    # GitHub-only: None when no client
    assert signals.issues is None
    assert signals.prs is None
    assert signals.contributors is None


def test_latest_sha_matches_git(tmp_repo: Path) -> None:
    extractor = RepoExtractor(github_client=None)
    signals = extractor.extract_local(
        url=f"file://{tmp_repo}",
        clone_path=tmp_repo,
        since=datetime(2026, 1, 1, tzinfo=UTC),
    )
    expected = subprocess.run(
        ["git", "-C", str(tmp_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    assert signals.latest_sha == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_extractor_repo.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the orchestrator**

Create `scripts/synth/extractor/repo.py`:

```python
"""RepoExtractor — orchestrates per-repo signal extraction.

Local extraction (mandatory): file walk + manifests + CODEOWNERS +
git log + branches.

GitHub extraction (optional): issues + PRs + contributors + workflows.
Pass a `GithubClient` to enable; pass `None` to skip.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scripts.synth.extractor.codeowners import (
    CodeownerRule,
    find_codeowners_file,
    parse_codeowners,
)
from scripts.synth.extractor.git_log import Branch, Commit, walk_branches, walk_commits
from scripts.synth.extractor.github_api import (
    Contributor,
    GithubClient,
    Issue,
    PullRequest,
    Workflow,
    parse_repo_url,
)
from scripts.synth.extractor.manifests import Manifest, parse_manifests_in_repo


@dataclass(frozen=True)
class Readme:
    path: Path
    content: str


@dataclass(frozen=True)
class RepoSignals:
    url: str
    clone_path: Path
    default_branch: str
    latest_sha: str
    description: str | None
    manifests: tuple[Manifest, ...]
    readmes: tuple[Readme, ...]
    codeowners: tuple[CodeownerRule, ...]
    commits: tuple[Commit, ...]
    branches: tuple[Branch, ...]
    issues: tuple[Issue, ...] | None
    prs: tuple[PullRequest, ...] | None
    contributors: tuple[Contributor, ...] | None
    workflows: tuple[Workflow, ...] | None


class RepoExtractor:
    def __init__(self, github_client: GithubClient | None) -> None:
        self._gh = github_client

    def extract_local(self, url: str, clone_path: Path, since: datetime) -> RepoSignals:
        latest_sha = subprocess.run(
            ["git", "-C", str(clone_path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        default_branch = subprocess.run(
            ["git", "-C", str(clone_path), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        manifests = parse_manifests_in_repo(clone_path)
        readmes = _collect_readmes(clone_path)

        cof = find_codeowners_file(clone_path)
        codeowners = parse_codeowners(cof.read_text()) if cof else ()

        commits = walk_commits(clone_path, since=since)
        branches = walk_branches(clone_path)

        return RepoSignals(
            url=url,
            clone_path=clone_path,
            default_branch=default_branch,
            latest_sha=latest_sha,
            description=_top_level_description(manifests),
            manifests=tuple(manifests),
            readmes=readmes,
            codeowners=codeowners,
            commits=tuple(commits),
            branches=tuple(branches),
            issues=None,
            prs=None,
            contributors=None,
            workflows=None,
        )

    async def extract(self, url: str, clone_path: Path, since: datetime, *, fetch_github: bool) -> RepoSignals:
        local = self.extract_local(url, clone_path, since)
        if not fetch_github or self._gh is None:
            return local

        owner, repo = parse_repo_url(url)
        issues = await self._gh.fetch_issues(owner, repo)
        prs = await self._gh.fetch_prs(owner, repo)
        contributors = await self._gh.fetch_contributors(owner, repo)
        workflows = await self._gh.fetch_workflows(owner, repo)

        # Reconstruct with the GH fields filled in.
        return RepoSignals(
            url=local.url,
            clone_path=local.clone_path,
            default_branch=local.default_branch,
            latest_sha=local.latest_sha,
            description=local.description,
            manifests=local.manifests,
            readmes=local.readmes,
            codeowners=local.codeowners,
            commits=local.commits,
            branches=local.branches,
            issues=tuple(issues),
            prs=tuple(prs),
            contributors=tuple(contributors),
            workflows=tuple(workflows),
        )


def _collect_readmes(clone_path: Path) -> tuple[Readme, ...]:
    """Top-level README + first-level subdir READMEs."""
    found: list[Readme] = []
    for candidate in (
        clone_path / "README.md",
        clone_path / "README.rst",
        clone_path / "README",
    ):
        if candidate.is_file():
            found.append(Readme(path=candidate, content=candidate.read_text(errors="replace")))
            break
    for child in clone_path.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        for candidate in (child / "README.md", child / "README.rst"):
            if candidate.is_file():
                found.append(Readme(path=candidate, content=candidate.read_text(errors="replace")))
                break
    return tuple(found)


def _top_level_description(manifests: list[Manifest]) -> str | None:
    """Pick the description from a top-level manifest, if any."""
    for m in manifests:
        if m.description and len(m.path.parts) <= 2:  # repo root manifest
            return m.description
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_extractor_repo.py -v`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/extractor/repo.py tests/synth/test_extractor_repo.py
git commit -m "$(cat <<'EOF'
feat(synth): RepoExtractor orchestrator (local + optional GH)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 10: WorldModel + Person/Service dataclasses

**Files:**
- Create: `scripts/synth/world_model.py`
- Test: `tests/synth/test_world_model_dataclasses.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_world_model_dataclasses.py`:

```python
"""Smoke tests for the immutable WorldModel dataclasses + their helpers."""

from __future__ import annotations

from datetime import UTC, datetime

from scripts.synth.world_model import (
    DepEdge,
    Person,
    RepoSummary,
    Service,
    ServiceKind,
    Topic,
    TopicKind,
    WorldModel,
)


def test_worldmodel_is_frozen() -> None:
    wm = WorldModel(
        repos=(),
        people=(),
        services=(),
        topic_pool=(),
        channels=(),
        notion_sections=(),
        time_anchors=(),
        dep_graph=(),
        company_name="acme",
        seed=1,
        extracted_at=datetime.now(UTC),
        sha_set={},
    )
    try:
        wm.seed = 2  # type: ignore[misc]
    except AttributeError:
        pass
    else:
        raise AssertionError("WorldModel must be frozen")


def test_person_canonical_id_is_required() -> None:
    p = Person(
        canonical_id="gh:alice",
        gh_username="alice",
        display_name="Alice",
        email_aliases=("alice@example.com",),
        role_hint=None,
        repos_active_in=("github.com/x/y",),
        activity_score=12.0,
    )
    assert p.canonical_id == "gh:alice"


def test_service_qualified_name_collision() -> None:
    s = Service(
        name="payments",
        qualified="repo-a/payments",
        repo_url="github.com/x/repo-a",
        kind=ServiceKind.API,
        description=None,
        owners=("gh:alice",),
        recent_activity=1.0,
        deploy_target=None,
    )
    assert s.qualified == "repo-a/payments"


def test_dep_edge_directional() -> None:
    e = DepEdge(from_service="api", to_service="lib", source_repo="x")
    assert e.from_service == "api"


def test_topic_recency_weighted() -> None:
    t = Topic(
        text="auth refactor",
        kind=TopicKind.PR,
        repo_url="github.com/x/y",
        ts=datetime(2026, 4, 1, tzinfo=UTC),
        mentioned_services=("auth-svc",),
        mentioned_people=("gh:alice",),
        weight=0.85,
    )
    assert 0 < t.weight <= 1.0


def test_repo_summary_records_sha() -> None:
    r = RepoSummary(url="github.com/x/y", sha="abcd1234", default_branch="main")
    assert r.sha == "abcd1234"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_world_model_dataclasses.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the dataclasses**

Create `scripts/synth/world_model.py` (just the dataclasses for now; the merger comes in Task 11–16):

```python
"""WorldModel: immutable structure derived from input repos.

The deterministic layer's output. Every narrative-layer call (planner,
writer, validator) consumes this as cached prompt context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class ServiceKind(StrEnum):
    API = "api"
    WORKER = "worker"
    FRONTEND = "frontend"
    CLI = "cli"
    LIB = "lib"
    INFRA = "infra"
    UNKNOWN = "unknown"


class TopicKind(StrEnum):
    COMMIT = "commit"
    PR = "pr"
    ISSUE = "issue"
    README_SECTION = "readme_section"
    BRANCH = "branch"


@dataclass(frozen=True)
class RepoSummary:
    url: str
    sha: str
    default_branch: str


@dataclass(frozen=True)
class Person:
    canonical_id: str               # "gh:alice" if known, else hash-derived
    gh_username: str | None
    display_name: str
    email_aliases: tuple[str, ...]
    role_hint: str | None           # inferred from CODEOWNERS coverage
    repos_active_in: tuple[str, ...]
    activity_score: float


@dataclass(frozen=True)
class Service:
    name: str
    qualified: str                  # "repo/svc" if collision; else == name
    repo_url: str                   # primary owning repo
    kind: ServiceKind
    description: str | None
    owners: tuple[str, ...]         # canonical Person ids
    recent_activity: float
    deploy_target: str | None       # e.g. fly app name


@dataclass(frozen=True)
class Topic:
    text: str
    kind: TopicKind
    repo_url: str
    ts: datetime | None
    mentioned_services: tuple[str, ...]
    mentioned_people: tuple[str, ...]
    weight: float


@dataclass(frozen=True)
class ChannelHint:
    name: str                       # "#payments-deploys"
    suggested_topic: str | None
    related_services: tuple[str, ...]


@dataclass(frozen=True)
class SectionHint:
    title: str                      # "Engineering > Payments runbook"
    related_services: tuple[str, ...]


@dataclass(frozen=True)
class TimeAnchor:
    label: str                      # "active period 2026-W12"
    start: datetime
    end: datetime
    activity_score: float


@dataclass(frozen=True)
class DepEdge:
    from_service: str               # qualified name
    to_service: str                 # qualified name
    source_repo: str                # the repo whose manifest declared the dep


@dataclass(frozen=True)
class WorldModel:
    repos: tuple[RepoSummary, ...]
    people: tuple[Person, ...]
    services: tuple[Service, ...]
    topic_pool: tuple[Topic, ...]
    channels: tuple[ChannelHint, ...]
    notion_sections: tuple[SectionHint, ...]
    time_anchors: tuple[TimeAnchor, ...]
    dep_graph: tuple[DepEdge, ...]
    company_name: str
    seed: int
    extracted_at: datetime
    sha_set: dict[str, str] = field(default_factory=dict)  # repo_url → sha
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_world_model_dataclasses.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_dataclasses.py
git commit -m "$(cat <<'EOF'
feat(synth): WorldModel + Person/Service/Topic/DepEdge dataclasses

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: Person canonicalization

**Files:**
- Modify: `scripts/synth/world_model.py` (add merger functions)
- Test: `tests/synth/test_world_model_merger.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_world_model_merger.py`:

```python
"""WorldModelMerger — turns RepoSignals[] into a single WorldModel.

Person canonicalization is the highest-stakes step (treating two people
as one produces threads where someone replies to themselves)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.extractor.git_log import Commit
from scripts.synth.extractor.github_api import Contributor
from scripts.synth.extractor.repo import RepoSignals
from scripts.synth.world_model import canonicalize_people


def _commit(email: str, name: str, sha: str = "x") -> Commit:
    return Commit(
        sha=sha, author_name=name, author_email=email,
        ts=datetime(2026, 3, 1, tzinfo=UTC),
        subject="x", body="", files_touched=(),
    )


def _signals(commits, contributors=None) -> RepoSignals:
    return RepoSignals(
        url="github.com/x/y",
        clone_path=Path("/tmp/x"),
        default_branch="main",
        latest_sha="abcd",
        description=None,
        manifests=(),
        readmes=(),
        codeowners=(),
        commits=tuple(commits),
        branches=(),
        issues=None,
        prs=None,
        contributors=contributors,
        workflows=None,
    )


def test_canonicalize_uses_gh_username_when_available() -> None:
    sigs = [
        _signals(
            commits=[_commit("alice@work.com", "Alice"), _commit("alice@home.com", "Alice X")],
            contributors=(
                Contributor(gh_username="alice", display_name="Alice X",
                            email_aliases=("alice@work.com",), contributions=10),
            ),
        )
    ]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=10)
    assert len(people) == 1
    assert people[0].canonical_id == "gh:alice"
    assert "alice@work.com" in people[0].email_aliases
    assert "alice@home.com" in people[0].email_aliases


def test_canonicalize_falls_back_to_email_when_no_gh() -> None:
    sigs = [_signals(commits=[
        _commit("a@x.com", "A", sha="1"),
        _commit("a@x.com", "A", sha="2"),
        _commit("b@x.com", "B", sha="3"),
    ])]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=10)
    canon = sorted(p.canonical_id for p in people)
    assert canon == ["email:a@x.com", "email:b@x.com"]


def test_never_merges_by_display_name_alone() -> None:
    """Two 'John's at different companies must remain separate."""
    sigs = [_signals(commits=[
        _commit("john@a.com", "John"),
        _commit("john@b.com", "John"),
    ])]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=10)
    assert len(people) == 2


def test_min_threshold_drops_low_activity_personas() -> None:
    sigs = [_signals(commits=[
        _commit("alice@x.com", "Alice", sha="1"),
        _commit("alice@x.com", "Alice", sha="2"),
        _commit("alice@x.com", "Alice", sha="3"),
        _commit("once@x.com", "Once-Off", sha="4"),
    ])]
    people = canonicalize_people(sigs, min_threshold=2, max_personas=10)
    assert {p.display_name for p in people} == {"Alice"}


def test_max_personas_caps_pool() -> None:
    sigs = [_signals(commits=[
        _commit(f"u{i}@x.com", f"User {i}", sha=f"s{i}") for i in range(40)
    ])]
    people = canonicalize_people(sigs, min_threshold=1, max_personas=5)
    assert len(people) == 5


def test_repos_active_in_recorded_per_person() -> None:
    sig_a = _signals(commits=[_commit("alice@x.com", "Alice", sha="a")])
    sig_a = RepoSignals(
        url="github.com/org/A", clone_path=sig_a.clone_path, default_branch="main",
        latest_sha=sig_a.latest_sha, description=None, manifests=(), readmes=(),
        codeowners=(), commits=sig_a.commits, branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    sig_b = _signals(commits=[_commit("alice@x.com", "Alice", sha="b")])
    sig_b = RepoSignals(
        url="github.com/org/B", clone_path=sig_b.clone_path, default_branch="main",
        latest_sha=sig_b.latest_sha, description=None, manifests=(), readmes=(),
        codeowners=(), commits=sig_b.commits, branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )

    [p] = canonicalize_people([sig_a, sig_b], min_threshold=1, max_personas=10)
    assert sorted(p.repos_active_in) == ["github.com/org/A", "github.com/org/B"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_world_model_merger.py -v`
Expected: FAIL — `ImportError: cannot import name 'canonicalize_people'`.

- [ ] **Step 3: Implement person canonicalization**

Append to `scripts/synth/world_model.py`:

```python
# ---------------------------------------------------------------------------
# WorldModelMerger — combines RepoSignals[] into a single WorldModel.
#
# Implemented across tasks 11-16. Each function is independently testable
# so the merger pipeline (Task 17) can compose them confidently.
# ---------------------------------------------------------------------------

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.synth.extractor.repo import RepoSignals


def canonicalize_people(
    signals: list["RepoSignals"],
    *,
    min_threshold: int,
    max_personas: int,
) -> tuple[Person, ...]:
    """Merge committers + GH contributors into canonical Persons.

    Precedence for canonical_id:
      1. gh:<username> if a contributor entry mentions an email/name we see
      2. email:<lowercased> if no GH match
    Display name = the GH name if available, else the most-frequent commit name.
    Activity = total commit count across all repos.
    """
    # email -> gh_username (from contributor records)
    email_to_gh: dict[str, str] = {}
    # gh_username -> display_name + email_aliases
    gh_meta: dict[str, dict] = {}
    for sig in signals:
        for c in (sig.contributors or ()):
            for email in c.email_aliases:
                email_to_gh[email.lower()] = c.gh_username
            gh_meta.setdefault(
                c.gh_username,
                {"display_name": c.display_name, "emails": set()},
            )
            gh_meta[c.gh_username]["emails"].update(e.lower() for e in c.email_aliases)
            if c.display_name:
                gh_meta[c.gh_username]["display_name"] = c.display_name

    # Aggregate activity per canonical_id
    activity: dict[str, int] = defaultdict(int)
    repos_active_in: dict[str, set[str]] = defaultdict(set)
    aliases: dict[str, set[str]] = defaultdict(set)
    display_names: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sig in signals:
        for commit in sig.commits:
            email = commit.author_email.lower()
            gh = email_to_gh.get(email)
            if gh:
                cid = f"gh:{gh}"
            else:
                cid = f"email:{email}"
            activity[cid] += 1
            repos_active_in[cid].add(sig.url)
            aliases[cid].add(commit.author_email)
            display_names[cid][commit.author_name] += 1

    # Even contributors with zero recent commits should appear if they
    # show up in the GH contributor list — but only if the merger run
    # considers them above threshold. Per spec we drop low-activity, so
    # we leave the activity counter as is.

    rows: list[Person] = []
    for cid, count in activity.items():
        if count < min_threshold:
            continue
        gh_username: str | None = None
        if cid.startswith("gh:"):
            gh_username = cid.removeprefix("gh:")
            display = gh_meta.get(gh_username, {}).get("display_name") or gh_username
            aliases[cid].update(gh_meta.get(gh_username, {}).get("emails", set()))
        else:
            # most-frequent commit name
            display = max(display_names[cid].items(), key=lambda kv: kv[1])[0]

        rows.append(
            Person(
                canonical_id=cid,
                gh_username=gh_username,
                display_name=display,
                email_aliases=tuple(sorted(aliases[cid])),
                role_hint=None,                          # filled later by service-owner inference
                repos_active_in=tuple(sorted(repos_active_in[cid])),
                activity_score=float(count),
            )
        )

    rows.sort(key=lambda p: p.activity_score, reverse=True)
    return tuple(rows[:max_personas])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_world_model_merger.py -v`
Expected: PASS — 6 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_merger.py
git commit -m "$(cat <<'EOF'
feat(synth): person canonicalization across repos (gh-first, email-fallback)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Service inference + collision handling

**Files:**
- Modify: `scripts/synth/world_model.py` (add `infer_services`)
- Modify: `tests/synth/test_world_model_merger.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/synth/test_world_model_merger.py`:

```python
from scripts.synth.world_model import infer_services


def test_infer_service_from_top_level_pyproject() -> None:
    """A repo with one top-level pyproject is one service named after it."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind
    sig = _signals(commits=[])
    sig = RepoSignals(
        url="github.com/x/payments-api", clone_path=sig.clone_path,
        default_branch="main", latest_sha="abc", description="Pay svc",
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/x/pyproject.toml"),
                     name="payments-api", description="Pay svc", dependencies=()),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    services = infer_services([sig])
    assert {s.name for s in services} == {"payments-api"}


def test_infer_services_from_monorepo_subdirs() -> None:
    """Children of services/<name>/pyproject.toml each become a Service."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind
    sig = RepoSignals(
        url="github.com/x/mono", clone_path=Path("/tmp/mono"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/mono/services/payments/pyproject.toml"),
                     name="payments", description=None, dependencies=()),
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/mono/services/billing/pyproject.toml"),
                     name="billing", description=None, dependencies=()),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    services = infer_services([sig])
    assert {s.name for s in services} == {"payments", "billing"}


def test_infer_services_qualifies_on_collision() -> None:
    """Two repos define a service named 'payments' → qualified names."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    def _sig_with(url, manifest_name):
        return RepoSignals(
            url=url, clone_path=Path("/tmp/x"),
            default_branch="main", latest_sha="abc", description=None,
            manifests=(
                Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/x/pyproject.toml"),
                         name=manifest_name, description=None, dependencies=()),
            ),
            readmes=(), codeowners=(), commits=(), branches=(),
            issues=None, prs=None, contributors=None, workflows=None,
        )

    services = infer_services([_sig_with("github.com/o/A", "payments"), _sig_with("github.com/o/B", "payments")])
    qualified = sorted(s.qualified for s in services)
    assert qualified == ["A/payments", "B/payments"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_world_model_merger.py -v -k infer_service`
Expected: FAIL — `ImportError: cannot import name 'infer_services'`.

- [ ] **Step 3: Implement service inference**

Append to `scripts/synth/world_model.py`:

```python
def infer_services(signals: list["RepoSignals"]) -> tuple[Service, ...]:
    """Each repo contributes 1+ Service. Collisions on bare name get
    qualified by the repo's last URL segment (e.g. "A/payments")."""
    candidates: list[tuple[str, str, "Manifest"]] = []  # (svc_name, repo_url, manifest)
    for sig in signals:
        for m in sig.manifests:
            if m.name:
                candidates.append((m.name, sig.url, m))

    # Detect collisions across repos
    name_to_repos: dict[str, set[str]] = defaultdict(set)
    for name, repo_url, _ in candidates:
        name_to_repos[name].add(repo_url)

    services: list[Service] = []
    seen: set[tuple[str, str]] = set()  # (qualified, repo_url) — dedupe within a repo
    for name, repo_url, m in candidates:
        if (name, repo_url) in seen:
            continue
        seen.add((name, repo_url))
        if len(name_to_repos[name]) > 1:
            qualified = f"{repo_url.rsplit('/', 1)[-1]}/{name}"
        else:
            qualified = name
        services.append(
            Service(
                name=name,
                qualified=qualified,
                repo_url=repo_url,
                kind=_infer_kind(m),
                description=m.description,
                owners=(),
                recent_activity=0.0,
                deploy_target=None,
            )
        )
    return tuple(services)


def _infer_kind(manifest: "Manifest") -> ServiceKind:
    """Heuristic: kind from manifest type. Plan 1 keeps it simple; richer
    inference (Dockerfile presence, asyncpg.listen detection, etc.) lives
    in plan 3 if needed."""
    from scripts.synth.extractor.manifests import ManifestKind
    if manifest.kind == ManifestKind.PACKAGE_JSON:
        return ServiceKind.FRONTEND
    if manifest.kind == ManifestKind.FLY_TOML:
        return ServiceKind.API
    if manifest.kind == ManifestKind.DOCKER_COMPOSE:
        return ServiceKind.INFRA
    return ServiceKind.LIB
```

Also at the top of `scripts/synth/world_model.py`, add the import (the `TYPE_CHECKING` block already references it, but `infer_services` body uses `Manifest` at runtime so we need a real import inside the function, which the helper does correctly — verify).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_world_model_merger.py -v -k infer_service`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_merger.py
git commit -m "$(cat <<'EOF'
feat(synth): service inference with cross-repo collision qualification

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 13: Topic pool weighting

**Files:**
- Modify: `scripts/synth/world_model.py` (add `build_topic_pool`)
- Modify: `tests/synth/test_world_model_merger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/synth/test_world_model_merger.py`:

```python
from scripts.synth.world_model import build_topic_pool


def test_topic_pool_includes_recent_commits() -> None:
    sigs = [_signals(commits=[
        _commit("a@x.com", "A", sha="1"),
        _commit("b@x.com", "B", sha="2"),
    ])]
    sigs[0] = RepoSignals(  # noqa: this is a test, mutate is fine
        url="github.com/x/y", clone_path=sigs[0].clone_path,
        default_branch="main", latest_sha="abc", description=None,
        manifests=(), readmes=(), codeowners=(), commits=(
            Commit(sha="1", author_name="A", author_email="a@x.com",
                   ts=datetime(2026, 4, 25, tzinfo=UTC),
                   subject="fix(payments): null pointer in checkout",
                   body="", files_touched=("services/payments/checkout.py",)),
        ),
        branches=(), issues=None, prs=None, contributors=None, workflows=None,
    )
    pool = build_topic_pool(sigs, services=(), now=datetime(2026, 4, 30, tzinfo=UTC))
    assert any("checkout" in t.text for t in pool)


def test_topic_recency_weighted_higher_for_recent_commits() -> None:
    old = Commit(sha="o", author_name="A", author_email="a@x.com",
                 ts=datetime(2026, 1, 1, tzinfo=UTC),
                 subject="old commit", body="", files_touched=())
    new = Commit(sha="n", author_name="A", author_email="a@x.com",
                 ts=datetime(2026, 4, 28, tzinfo=UTC),
                 subject="new commit", body="", files_touched=())
    sig = _signals(commits=[])
    sig = RepoSignals(
        url=sig.url, clone_path=sig.clone_path, default_branch="main", latest_sha=sig.latest_sha,
        description=None, manifests=(), readmes=(), codeowners=(), commits=(old, new),
        branches=(), issues=None, prs=None, contributors=None, workflows=None,
    )
    pool = build_topic_pool([sig], services=(), now=datetime(2026, 4, 30, tzinfo=UTC))
    new_weight = next(t.weight for t in pool if t.text == "new commit")
    old_weight = next(t.weight for t in pool if t.text == "old commit")
    assert new_weight > old_weight
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_world_model_merger.py -v -k topic_pool`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the topic pool**

Append to `scripts/synth/world_model.py`:

```python
import math


def _recency_decay(ts: datetime, now: datetime, half_life_days: float = 30.0) -> float:
    delta_days = (now - ts).total_seconds() / 86400.0
    return 0.5 ** (delta_days / half_life_days)


_TOPIC_KIND_WEIGHT = {
    TopicKind.PR: 1.0,
    TopicKind.ISSUE: 0.8,
    TopicKind.COMMIT: 0.5,
    TopicKind.README_SECTION: 0.3,
    TopicKind.BRANCH: 0.4,
}


def build_topic_pool(
    signals: list["RepoSignals"],
    services: tuple[Service, ...],
    now: datetime,
) -> tuple[Topic, ...]:
    service_names = {s.name for s in services} | {s.qualified for s in services}

    topics: list[Topic] = []
    for sig in signals:
        for c in sig.commits:
            mentioned_services = tuple(
                n for n in service_names if n.lower() in c.subject.lower() or any(n in f for f in c.files_touched)
            )
            recency = _recency_decay(c.ts, now)
            weight = recency * _TOPIC_KIND_WEIGHT[TopicKind.COMMIT] * (
                1.0 + math.log1p(len(mentioned_services))
            )
            topics.append(
                Topic(
                    text=c.subject,
                    kind=TopicKind.COMMIT,
                    repo_url=sig.url,
                    ts=c.ts,
                    mentioned_services=mentioned_services,
                    mentioned_people=(),
                    weight=weight,
                )
            )
        for issue in sig.issues or ():
            mentioned = tuple(n for n in service_names if n.lower() in issue.title.lower())
            recency = _recency_decay(issue.updated_at, now)
            topics.append(
                Topic(
                    text=issue.title, kind=TopicKind.ISSUE,
                    repo_url=sig.url, ts=issue.updated_at,
                    mentioned_services=mentioned, mentioned_people=(),
                    weight=recency * _TOPIC_KIND_WEIGHT[TopicKind.ISSUE] * (
                        1.0 + math.log1p(len(mentioned))
                    ),
                )
            )
        for pr in sig.prs or ():
            mentioned = tuple(n for n in service_names if n.lower() in pr.title.lower())
            base_ts = pr.merged_at or pr.created_at
            recency = _recency_decay(base_ts, now)
            topics.append(
                Topic(
                    text=pr.title, kind=TopicKind.PR,
                    repo_url=sig.url, ts=base_ts,
                    mentioned_services=mentioned, mentioned_people=(),
                    weight=recency * _TOPIC_KIND_WEIGHT[TopicKind.PR] * (
                        1.0 + math.log1p(len(mentioned))
                    ),
                )
            )
        for branch in sig.branches:
            recency = _recency_decay(branch.last_commit_ts, now)
            topics.append(
                Topic(
                    text=branch.name, kind=TopicKind.BRANCH,
                    repo_url=sig.url, ts=branch.last_commit_ts,
                    mentioned_services=(), mentioned_people=(),
                    weight=recency * _TOPIC_KIND_WEIGHT[TopicKind.BRANCH],
                )
            )

    return tuple(topics)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_world_model_merger.py -v -k topic_pool`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_merger.py
git commit -m "$(cat <<'EOF'
feat(synth): topic pool weighting (recency × kind × service mentions)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Channel + Notion section synthesis

**Files:**
- Modify: `scripts/synth/world_model.py` (add `synthesize_channels`, `synthesize_sections`)
- Modify: `tests/synth/test_world_model_merger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/synth/test_world_model_merger.py`:

```python
from scripts.synth.world_model import (
    ServiceKind,
    synthesize_channels,
    synthesize_sections,
)


def _svc(name: str, kind: ServiceKind = ServiceKind.API, recent: float = 1.0) -> Service:
    return Service(
        name=name, qualified=name, repo_url="github.com/x/y", kind=kind,
        description=None, owners=(), recent_activity=recent, deploy_target=None,
    )


def test_synthesize_channels_includes_per_service_and_generic() -> None:
    services = (_svc("payments"), _svc("billing"))
    channels = synthesize_channels(services, codeowner_teams=set())
    names = {c.name for c in channels}
    assert "#payments" in names
    assert "#billing" in names
    assert "#general" in names
    assert "#incidents" in names


def test_synthesize_channels_adds_deploy_channels_for_top_active() -> None:
    services = tuple(_svc(f"svc{i}", recent=float(i)) for i in range(8))
    channels = synthesize_channels(services, codeowner_teams=set())
    deploy_channels = {c.name for c in channels if c.name.endswith("-deploys")}
    # top-5 by activity → svc7,6,5,4,3
    assert deploy_channels == {"#svc7-deploys", "#svc6-deploys", "#svc5-deploys", "#svc4-deploys", "#svc3-deploys"}


def test_synthesize_sections_fixed_set_plus_per_service_runbooks() -> None:
    services = (_svc("payments"), _svc("billing"))
    sections = synthesize_sections(services)
    titles = {s.title for s in sections}
    assert "Engineering" in titles
    assert "Postmortems" in titles
    assert "payments runbook" in titles
    assert "billing runbook" in titles
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_world_model_merger.py -v -k "synthesize_channels or synthesize_sections"`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement synthesis**

Append to `scripts/synth/world_model.py`:

```python
_GENERIC_CHANNELS = ("#general", "#random", "#incidents", "#engineering", "#announcements")

_FIXED_NOTION_SECTIONS = (
    "Engineering", "Runbooks", "Postmortems", "Architecture",
    "Onboarding", "Product", "People & Hiring",
)


def synthesize_channels(
    services: tuple[Service, ...],
    codeowner_teams: set[str],
) -> tuple[ChannelHint, ...]:
    out: list[ChannelHint] = []

    # Per-service channel for api/worker/frontend
    for svc in services:
        if svc.kind in (ServiceKind.API, ServiceKind.WORKER, ServiceKind.FRONTEND):
            out.append(
                ChannelHint(
                    name=f"#{svc.name}",
                    suggested_topic=svc.description,
                    related_services=(svc.qualified,),
                )
            )

    # Top-5 deploy channels
    top_active = sorted(services, key=lambda s: s.recent_activity, reverse=True)[:5]
    for svc in top_active:
        out.append(
            ChannelHint(
                name=f"#{svc.name}-deploys",
                suggested_topic=None,
                related_services=(svc.qualified,),
            )
        )

    # Team channels
    for team in sorted(codeowner_teams):
        out.append(ChannelHint(name=f"#team-{team}", suggested_topic=None, related_services=()))

    # Generic
    for g in _GENERIC_CHANNELS:
        out.append(ChannelHint(name=g, suggested_topic=None, related_services=()))

    # Dedupe by name (preserve first occurrence)
    seen: set[str] = set()
    deduped: list[ChannelHint] = []
    for c in out:
        if c.name in seen:
            continue
        seen.add(c.name)
        deduped.append(c)
    return tuple(deduped)


def synthesize_sections(services: tuple[Service, ...]) -> tuple[SectionHint, ...]:
    out: list[SectionHint] = []
    for title in _FIXED_NOTION_SECTIONS:
        out.append(SectionHint(title=title, related_services=()))

    top10 = sorted(services, key=lambda s: s.recent_activity, reverse=True)[:10]
    for svc in top10:
        out.append(
            SectionHint(
                title=f"{svc.name} runbook",
                related_services=(svc.qualified,),
            )
        )
    return tuple(out)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_world_model_merger.py -v -k "synthesize_channels or synthesize_sections"`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_merger.py
git commit -m "$(cat <<'EOF'
feat(synth): synthesize Slack channels + Notion sections from services

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 15: Cross-repo dependency edges

**Files:**
- Modify: `scripts/synth/world_model.py` (add `build_dep_graph`)
- Modify: `tests/synth/test_world_model_merger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/synth/test_world_model_merger.py`:

```python
from scripts.synth.world_model import build_dep_graph


def test_dep_edge_recorded_when_manifest_dep_matches_service() -> None:
    """A repo with manifest deps that name a Service produces a DepEdge
    from the manifest's owning service to the dependency service."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    services = (
        _svc("payments"),  # in repo A
        _svc("billing"),   # in repo B
    )
    services = (
        Service(name="payments", qualified="payments", repo_url="github.com/x/A",
                kind=ServiceKind.API, description=None, owners=(), recent_activity=1.0,
                deploy_target=None),
        Service(name="billing", qualified="billing", repo_url="github.com/x/B",
                kind=ServiceKind.LIB, description=None, owners=(), recent_activity=1.0,
                deploy_target=None),
    )
    sig_a = RepoSignals(
        url="github.com/x/A", clone_path=Path("/tmp/A"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/A/pyproject.toml"),
                     name="payments", description=None, dependencies=("billing", "fastapi")),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    sig_b = RepoSignals(
        url="github.com/x/B", clone_path=Path("/tmp/B"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/B/pyproject.toml"),
                     name="billing", description=None, dependencies=()),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )

    edges = build_dep_graph([sig_a, sig_b], services)
    assert len(edges) == 1
    assert edges[0].from_service == "payments"
    assert edges[0].to_service == "billing"
    assert edges[0].source_repo == "github.com/x/A"


def test_no_dep_edge_for_external_deps() -> None:
    """Manifest deps that don't match any Service are ignored."""
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    services = (_svc("payments"),)
    sig = RepoSignals(
        url="github.com/x/A", clone_path=Path("/tmp/A"),
        default_branch="main", latest_sha="abc", description=None,
        manifests=(
            Manifest(kind=ManifestKind.PYPROJECT, path=Path("/tmp/A/pyproject.toml"),
                     name="payments", description=None,
                     dependencies=("requests", "boto3", "stripe")),
        ),
        readmes=(), codeowners=(), commits=(), branches=(),
        issues=None, prs=None, contributors=None, workflows=None,
    )
    edges = build_dep_graph([sig], services)
    assert edges == ()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_world_model_merger.py -v -k dep_graph`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement the dep graph builder**

Append to `scripts/synth/world_model.py`:

```python
def build_dep_graph(
    signals: list["RepoSignals"],
    services: tuple[Service, ...],
) -> tuple[DepEdge, ...]:
    by_name = {s.name: s for s in services}

    edges: list[DepEdge] = []
    for sig in signals:
        for m in sig.manifests:
            if not m.name:
                continue
            from_svc = by_name.get(m.name)
            if not from_svc:
                continue
            for dep_name in m.dependencies:
                to_svc = by_name.get(dep_name)
                if not to_svc:
                    continue
                if to_svc.qualified == from_svc.qualified:
                    continue  # don't record self-edges
                edges.append(
                    DepEdge(
                        from_service=from_svc.qualified,
                        to_service=to_svc.qualified,
                        source_repo=sig.url,
                    )
                )
    return tuple(edges)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_world_model_merger.py -v -k dep_graph`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_merger.py
git commit -m "$(cat <<'EOF'
feat(synth): cross-repo dep graph from manifest dependencies

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 16: Time anchors (active-period clusters)

**Files:**
- Modify: `scripts/synth/world_model.py` (add `compute_time_anchors`)
- Modify: `tests/synth/test_world_model_merger.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/synth/test_world_model_merger.py`:

```python
from scripts.synth.world_model import compute_time_anchors


def test_time_anchors_groups_commits_by_iso_week() -> None:
    sigs = [_signals(commits=[
        Commit(sha=f"s{i}", author_name="A", author_email="a@x.com",
               ts=datetime(2026, 4, 1 + (i % 7), 10, tzinfo=UTC),
               subject=f"c{i}", body="", files_touched=())
        for i in range(20)
    ])]
    anchors = compute_time_anchors(sigs)
    # All 20 commits land in 1-2 ISO weeks; each anchor must have
    # nonzero activity_score.
    assert anchors
    assert all(a.activity_score > 0 for a in anchors)


def test_time_anchors_returned_in_chronological_order() -> None:
    sigs = [_signals(commits=[
        Commit(sha="a", author_name="A", author_email="a@x.com",
               ts=datetime(2026, 1, 1, tzinfo=UTC), subject="x", body="", files_touched=()),
        Commit(sha="b", author_name="A", author_email="a@x.com",
               ts=datetime(2026, 4, 1, tzinfo=UTC), subject="y", body="", files_touched=()),
    ])]
    anchors = compute_time_anchors(sigs)
    starts = [a.start for a in anchors]
    assert starts == sorted(starts)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_world_model_merger.py -v -k time_anchor`
Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement time anchors**

Append to `scripts/synth/world_model.py`:

```python
from datetime import timedelta


def compute_time_anchors(signals: list["RepoSignals"]) -> tuple[TimeAnchor, ...]:
    """Cluster commit timestamps by ISO week. Each non-empty week becomes
    a TimeAnchor with activity_score = number of commits that week."""
    week_counts: dict[tuple[int, int], int] = defaultdict(int)
    for sig in signals:
        for c in sig.commits:
            year, week, _ = c.ts.isocalendar()
            week_counts[(year, week)] += 1

    anchors: list[TimeAnchor] = []
    for (year, week), count in sorted(week_counts.items()):
        # ISO week → start (Monday) of that week
        # date.fromisocalendar exists in 3.8+
        from datetime import date as date_cls
        start_date = date_cls.fromisocalendar(year, week, 1)
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=UTC)
        anchors.append(
            TimeAnchor(
                label=f"active-{year}-W{week:02d}",
                start=start,
                end=start + timedelta(days=7),
                activity_score=float(count),
            )
        )
    return tuple(anchors)
```

Note: this also requires `from datetime import UTC` at the top of `world_model.py` if it isn't already there. Add that import to the module top if needed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_world_model_merger.py -v -k time_anchor`
Expected: PASS — 2 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_merger.py
git commit -m "$(cat <<'EOF'
feat(synth): time anchors from ISO-week commit clusters

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 17: WorldModelMerger composer (orchestrates 11–16)

**Files:**
- Modify: `scripts/synth/world_model.py` (add `merge_world_model`)
- Modify: `tests/synth/test_world_model_merger.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/synth/test_world_model_merger.py`:

```python
from scripts.synth.world_model import WorldModel, merge_world_model


def test_merge_world_model_end_to_end(tmp_repo: Path) -> None:
    from scripts.synth.extractor.repo import RepoExtractor
    extractor = RepoExtractor(github_client=None)
    sig = extractor.extract_local(
        url=f"file://{tmp_repo}", clone_path=tmp_repo,
        since=datetime(2026, 1, 1, tzinfo=UTC),
    )

    wm = merge_world_model(
        signals=[sig], company_name="acme", seed=42,
        min_threshold=1, max_personas=10,
        now=datetime(2026, 4, 30, tzinfo=UTC),
    )

    assert isinstance(wm, WorldModel)
    assert wm.company_name == "acme"
    assert wm.seed == 42
    assert wm.repos and wm.repos[0].url == f"file://{tmp_repo}"
    assert wm.sha_set[f"file://{tmp_repo}"] == sig.latest_sha
    # Personas: at least Alice, Bob, Carol (each has ≥ 1 commit)
    names = {p.display_name for p in wm.people}
    assert {"Alice", "Bob", "Carol"} <= names
    # Services: payments + billing + fake-repo
    svc_names = {s.name for s in wm.services}
    assert {"payments", "billing", "fake-repo"} <= svc_names
    # Channels: at least #general
    assert any(c.name == "#general" for c in wm.channels)
    # Time anchors: nonzero
    assert wm.time_anchors
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synth/test_world_model_merger.py::test_merge_world_model_end_to_end -v`
Expected: FAIL — `ImportError: cannot import name 'merge_world_model'`.

- [ ] **Step 3: Implement the composer**

Append to `scripts/synth/world_model.py`:

```python
def merge_world_model(
    signals: list["RepoSignals"],
    *,
    company_name: str,
    seed: int,
    min_threshold: int,
    max_personas: int,
    now: datetime,
) -> WorldModel:
    """Compose all merger steps into the immutable WorldModel."""
    people = canonicalize_people(signals, min_threshold=min_threshold, max_personas=max_personas)
    services = infer_services(signals)
    topic_pool = build_topic_pool(signals, services=services, now=now)

    # Codeowner team set: anything looking like @<team> with a slash (e.g. @org/team)
    codeowner_teams: set[str] = set()
    for sig in signals:
        for rule in sig.codeowners:
            for owner in rule.owners:
                if "/" in owner:  # @org/team
                    codeowner_teams.add(owner.split("/", 1)[1])

    channels = synthesize_channels(services, codeowner_teams=codeowner_teams)
    sections = synthesize_sections(services)
    dep_graph = build_dep_graph(signals, services)
    time_anchors = compute_time_anchors(signals)

    repos = tuple(
        RepoSummary(url=s.url, sha=s.latest_sha, default_branch=s.default_branch)
        for s in signals
    )
    sha_set = {s.url: s.latest_sha for s in signals}

    return WorldModel(
        repos=repos,
        people=people,
        services=services,
        topic_pool=topic_pool,
        channels=channels,
        notion_sections=sections,
        time_anchors=time_anchors,
        dep_graph=dep_graph,
        company_name=company_name,
        seed=seed,
        extracted_at=now,
        sha_set=sha_set,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synth/test_world_model_merger.py::test_merge_world_model_end_to_end -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/world_model.py tests/synth/test_world_model_merger.py
git commit -m "$(cat <<'EOF'
feat(synth): WorldModelMerger composer (canon + services + topics + edges)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 18: LLM client wrapper

**Files:**
- Create: `scripts/synth/llm_client.py`
- Test: `tests/synth/test_llm_client.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_llm_client.py`:

```python
"""Anthropic SDK wrapper. Plan 1 needs only basic generate(). Plan 3
adds prompt-caching block + mock-mode fixture loader."""

from __future__ import annotations

from typing import Any

import pytest

from scripts.synth.llm_client import (
    LlmClient,
    LlmRequest,
    LlmResponse,
    StaticLlmClient,
)


@pytest.mark.asyncio
async def test_static_client_returns_canned_response() -> None:
    client = StaticLlmClient({"hello": "hi there"})
    resp = await client.generate(LlmRequest(model="m", system="", prompt="hello"))
    assert resp.text == "hi there"


@pytest.mark.asyncio
async def test_static_client_raises_on_unmapped_prompt() -> None:
    client = StaticLlmClient({})
    with pytest.raises(KeyError):
        await client.generate(LlmRequest(model="m", system="", prompt="??"))


@pytest.mark.asyncio
async def test_real_client_passes_prompt_to_anthropic_sdk(monkeypatch) -> None:
    """Verify the real client wires prompt → anthropic.messages.create
    (we don't actually call the network)."""

    class FakeMessages:
        called_with: dict[str, Any] = {}

        async def create(self, **kwargs):
            FakeMessages.called_with = kwargs
            class _R:
                content = [type("B", (), {"text": "ok", "type": "text"})()]
            return _R()

    class FakeClient:
        def __init__(self, **_): self.messages = FakeMessages()
        async def aclose(self): pass

    import scripts.synth.llm_client as mod
    monkeypatch.setattr(mod, "AsyncAnthropic", FakeClient)

    client = LlmClient(api_key="test-key")
    resp = await client.generate(LlmRequest(model="claude-x", system="sys", prompt="hi"))
    await client.close()

    assert resp.text == "ok"
    assert FakeMessages.called_with["model"] == "claude-x"
    assert FakeMessages.called_with["system"] == "sys"
    assert FakeMessages.called_with["messages"][0]["content"] == "hi"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_llm_client.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement the wrapper**

Create `scripts/synth/llm_client.py`:

```python
"""LLM client. Plan 1 only needs basic single-shot generate() and a
StaticLlmClient used by tests / CompanyContext auto-inferrer.

Plan 3 will extend this with prompt-cache control blocks (Anthropic SDK
cache_control), retries via tenacity, and a fixture-keyed mock client
for the `--mock-llm` CLI flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from anthropic import AsyncAnthropic


@dataclass(frozen=True)
class LlmRequest:
    model: str
    system: str
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.4


@dataclass(frozen=True)
class LlmResponse:
    text: str


class LlmClientProtocol(Protocol):
    async def generate(self, req: LlmRequest) -> LlmResponse: ...
    async def close(self) -> None: ...


class LlmClient:
    """Real Anthropic-backed client."""

    def __init__(self, api_key: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(self, req: LlmRequest) -> LlmResponse:
        msg = await self._client.messages.create(
            model=req.model,
            system=req.system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            messages=[{"role": "user", "content": req.prompt}],
        )
        text_parts: list[str] = []
        for block in msg.content:
            if getattr(block, "type", None) == "text":
                text_parts.append(getattr(block, "text", ""))
        return LlmResponse(text="".join(text_parts))

    async def close(self) -> None:
        try:
            await self._client.aclose()  # type: ignore[attr-defined]
        except AttributeError:
            pass


class StaticLlmClient:
    """Test client: prompts → canned responses by exact match."""

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping

    async def generate(self, req: LlmRequest) -> LlmResponse:
        if req.prompt not in self._mapping:
            raise KeyError(f"no canned response for prompt: {req.prompt!r}")
        return LlmResponse(text=self._mapping[req.prompt])

    async def close(self) -> None:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_llm_client.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/llm_client.py tests/synth/test_llm_client.py
git commit -m "$(cat <<'EOF'
feat(synth): LLM client wrapper (Anthropic SDK + StaticLlmClient for tests)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 19: CompanyContext loader + auto-inferrer

**Files:**
- Create: `scripts/synth/company_context.py`
- Test: `tests/synth/test_company_context.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/synth/test_company_context.py`:

```python
"""CompanyContext: load from YAML, OR infer once from READMEs via LLM."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.synth.company_context import (
    CompanyContext,
    Customer,
    NonEngPerson,
    infer_company_context,
    load_company_context,
)
from scripts.synth.llm_client import StaticLlmClient


def test_load_minimal_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cc.yaml"
    p.write_text(
        """
name: prbe.ai
stage: seed
headcount: 8
""".strip()
    )
    cc = load_company_context(p)
    assert cc.name == "prbe.ai"
    assert cc.stage == "seed"
    assert cc.headcount == 8
    assert cc.customers == ()
    assert cc.non_eng_people == ()


def test_load_full_yaml(tmp_path: Path) -> None:
    p = tmp_path / "cc.yaml"
    p.write_text(
        """
name: acme
stage: series-a
headcount: 25
market: payment infra
competitors: [Stripe, Adyen]
customers:
  - {name: Globex, type: paying, plan: team}
non_eng_people:
  - {name: Sam Park, role: founding GTM}
recent_milestones: [Closed Series A]
ongoing_initiatives: [SOC2 Type 2]
""".strip()
    )
    cc = load_company_context(p)
    assert cc.competitors == ("Stripe", "Adyen")
    [cust] = cc.customers
    assert cust == Customer(name="Globex", type="paying", plan="team")
    [neng] = cc.non_eng_people
    assert neng == NonEngPerson(name="Sam Park", role="founding GTM")
    assert "SOC2 Type 2" in cc.ongoing_initiatives


@pytest.mark.asyncio
async def test_infer_company_context_uses_llm_and_returns_yaml(tmp_path: Path) -> None:
    canned = {
        "READMES_AND_REPOS_FOR_INFERENCE": """
name: inferred-co
stage: seed
headcount: 5
market: dev tools
""".strip()
    }
    static_llm = StaticLlmClient(canned)

    cc, raw_yaml = await infer_company_context(
        readme_blob="x",
        repo_descriptions=["y"],
        llm_client=static_llm,
        model="claude-opus",
    )
    # Inference returns both the dataclass and the raw YAML for inspection
    assert cc.name == "inferred-co"
    assert "inferred-co" in raw_yaml
```

The third test uses a tiny trick: the prompt sent to the LLM is the literal string `"READMES_AND_REPOS_FOR_INFERENCE"`. We arrange this by having the inferrer build the prompt deterministically — easier for testing than fuzzy matching.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/synth/test_company_context.py -v`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Implement CompanyContext**

Create `scripts/synth/company_context.py`:

```python
"""CompanyContext: business reality the repo doesn't expose.

Optional input. If not provided, an LLM call over aggregated READMEs +
repo descriptions produces a draft, written to disk for the user to inspect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scripts.synth.llm_client import LlmClientProtocol, LlmRequest


@dataclass(frozen=True)
class Customer:
    name: str
    type: str            # design_partner | paying | trial | ...
    plan: str | None = None
    stage: str | None = None


@dataclass(frozen=True)
class NonEngPerson:
    name: str
    role: str
    slack: str | None = None


@dataclass(frozen=True)
class CompanyContext:
    name: str
    stage: str
    headcount: int
    market: str | None = None
    competitors: tuple[str, ...] = ()
    customers: tuple[Customer, ...] = ()
    non_eng_people: tuple[NonEngPerson, ...] = ()
    recent_milestones: tuple[str, ...] = ()
    ongoing_initiatives: tuple[str, ...] = ()
    cadence: dict = field(default_factory=dict)
    inferred: bool = False  # True if produced by infer_company_context


def _to_tuple(seq: object) -> tuple:
    return tuple(seq) if isinstance(seq, list) else ()


def _from_dict(data: dict, *, inferred: bool) -> CompanyContext:
    return CompanyContext(
        name=data["name"],
        stage=data.get("stage", "unknown"),
        headcount=int(data.get("headcount", 0)),
        market=data.get("market"),
        competitors=_to_tuple(data.get("competitors")),
        customers=tuple(Customer(**c) for c in (data.get("customers") or [])),
        non_eng_people=tuple(NonEngPerson(**p) for p in (data.get("non_eng_people") or [])),
        recent_milestones=_to_tuple(data.get("recent_milestones")),
        ongoing_initiatives=_to_tuple(data.get("ongoing_initiatives")),
        cadence=data.get("cadence") or {},
        inferred=inferred,
    )


def load_company_context(path: Path) -> CompanyContext:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"company context must be a YAML mapping, got {type(raw).__name__}")
    return _from_dict(raw, inferred=False)


_SYSTEM = (
    "You produce minimal CompanyContext YAML for a synthetic-corpus eval tool. "
    "Output ONLY the YAML, no commentary, no fences. "
    "Required keys: name, stage, headcount, market. "
    "Optional: competitors (list), customers (list of {name,type,plan?}), "
    "non_eng_people (list of {name,role}), recent_milestones (list), "
    "ongoing_initiatives (list)."
)

_INFERENCE_PROMPT_KEY = "READMES_AND_REPOS_FOR_INFERENCE"


def _render_inference_prompt(readme_blob: str, repo_descriptions: list[str]) -> str:
    """We use a stable prompt-key for tests; in production this expands to
    the full readme + repo blob."""
    if not readme_blob and not repo_descriptions:
        return _INFERENCE_PROMPT_KEY
    body = "READMES:\n" + readme_blob + "\n\nREPO DESCRIPTIONS:\n" + "\n".join(repo_descriptions)
    return body


async def infer_company_context(
    *,
    readme_blob: str,
    repo_descriptions: list[str],
    llm_client: LlmClientProtocol,
    model: str,
) -> tuple[CompanyContext, str]:
    """One-shot LLM inference. Returns the CompanyContext + the raw YAML
    string (so the caller can write `inferred-company.yaml` for the user)."""
    prompt = _render_inference_prompt(readme_blob, repo_descriptions)
    resp = await llm_client.generate(
        LlmRequest(model=model, system=_SYSTEM, prompt=prompt, temperature=0.2)
    )
    raw_yaml = resp.text.strip()
    data = yaml.safe_load(raw_yaml)
    if not isinstance(data, dict):
        raise ValueError("LLM did not return a YAML mapping for CompanyContext")
    return _from_dict(data, inferred=True), raw_yaml
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/synth/test_company_context.py -v`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/company_context.py tests/synth/test_company_context.py
git commit -m "$(cat <<'EOF'
feat(synth): CompanyContext YAML loader + LLM auto-inferrer

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 20: Cache integration for repo extraction

**Files:**
- Modify: `scripts/synth/extractor/repo.py` (`RepoExtractor` accepts `DiskCache`)
- Modify: `tests/synth/test_extractor_repo.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/synth/test_extractor_repo.py`:

```python
from scripts.synth.cache import DiskCache


def test_extractor_uses_cache_when_sha_unchanged(tmp_repo: Path, tmp_path: Path) -> None:
    cache = DiskCache(tmp_path / "cache")
    extractor = RepoExtractor(github_client=None, cache=cache)

    s1 = extractor.extract_local(
        url="repo://x", clone_path=tmp_repo,
        since=datetime(2026, 1, 1, tzinfo=UTC),
    )
    # 2nd call with same SHA: must come from cache; we prove by mutating
    # the repo (new commit), running again, and observing identical sha.
    cached = cache.get(f"repo:repo://x@{s1.latest_sha}")
    assert cached is not None
    s2 = extractor.extract_local(
        url="repo://x", clone_path=tmp_repo,
        since=datetime(2026, 1, 1, tzinfo=UTC),
    )
    assert s1.latest_sha == s2.latest_sha
    assert len(s1.commits) == len(s2.commits)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synth/test_extractor_repo.py::test_extractor_uses_cache_when_sha_unchanged -v`
Expected: FAIL — `RepoExtractor.__init__()` doesn't accept `cache`.

- [ ] **Step 3: Wire the cache**

Modify `scripts/synth/extractor/repo.py`. At the top, add:

```python
from scripts.synth.cache import DiskCache
```

Replace the `RepoExtractor.__init__` and `extract_local` methods:

```python
class RepoExtractor:
    def __init__(
        self,
        github_client: GithubClient | None,
        cache: DiskCache | None = None,
    ) -> None:
        self._gh = github_client
        self._cache = cache

    def extract_local(self, url: str, clone_path: Path, since: datetime) -> RepoSignals:
        latest_sha = subprocess.run(
            ["git", "-C", str(clone_path), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        cache_key = f"repo:{url}@{latest_sha}"
        if self._cache is not None:
            hit = self._cache.get(cache_key)
            if hit is not None:
                return _signals_from_dict(hit)

        default_branch = subprocess.run(
            ["git", "-C", str(clone_path), "rev-parse", "--abbrev-ref", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()

        manifests = parse_manifests_in_repo(clone_path)
        readmes = _collect_readmes(clone_path)

        cof = find_codeowners_file(clone_path)
        codeowners = parse_codeowners(cof.read_text()) if cof else ()

        commits = walk_commits(clone_path, since=since)
        branches = walk_branches(clone_path)

        signals = RepoSignals(
            url=url,
            clone_path=clone_path,
            default_branch=default_branch,
            latest_sha=latest_sha,
            description=_top_level_description(manifests),
            manifests=tuple(manifests),
            readmes=readmes,
            codeowners=codeowners,
            commits=tuple(commits),
            branches=tuple(branches),
            issues=None,
            prs=None,
            contributors=None,
            workflows=None,
        )

        if self._cache is not None:
            self._cache.put(cache_key, _signals_to_dict(signals))

        return signals
```

Then add helpers near the bottom of the file:

```python
def _signals_to_dict(s: RepoSignals) -> dict:
    """Cache-serialize. We store enough to reconstruct: url, sha, default_branch,
    description, plus full commits + branches + manifests + codeowners + readmes.
    Datetimes are isoformatted; tuples become lists; Paths become strings."""
    import dataclasses

    def encode(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {k: encode(getattr(v, k)) for k in (f.name for f in dataclasses.fields(v))}
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple | list):
            return [encode(x) for x in v]
        if isinstance(v, dict):
            return {k: encode(val) for k, val in v.items()}
        return v

    out = encode(s)
    # Mark gh-only fields explicitly None so loader knows they're absent vs empty.
    return out


def _signals_from_dict(data: dict) -> RepoSignals:
    """Reconstruct RepoSignals from cached dict."""
    from scripts.synth.extractor.codeowners import CodeownerRule
    from scripts.synth.extractor.git_log import Branch, Commit
    from scripts.synth.extractor.manifests import Manifest, ManifestKind

    def commit(d):
        return Commit(
            sha=d["sha"], author_name=d["author_name"], author_email=d["author_email"],
            ts=datetime.fromisoformat(d["ts"]),
            subject=d["subject"], body=d["body"],
            files_touched=tuple(d["files_touched"]),
        )

    def branch(d):
        return Branch(name=d["name"], last_commit_sha=d["last_commit_sha"],
                      last_commit_ts=datetime.fromisoformat(d["last_commit_ts"]))

    def manifest(d):
        return Manifest(
            kind=ManifestKind(d["kind"]), path=Path(d["path"]),
            name=d["name"], description=d["description"],
            dependencies=tuple(d["dependencies"]),
            compose_service_names=tuple(d["compose_service_names"]),
        )

    def readme(d):
        return Readme(path=Path(d["path"]), content=d["content"])

    def rule(d):
        return CodeownerRule(pattern=d["pattern"], owners=tuple(d["owners"]))

    return RepoSignals(
        url=data["url"], clone_path=Path(data["clone_path"]),
        default_branch=data["default_branch"], latest_sha=data["latest_sha"],
        description=data["description"],
        manifests=tuple(manifest(m) for m in data["manifests"]),
        readmes=tuple(readme(r) for r in data["readmes"]),
        codeowners=tuple(rule(r) for r in data["codeowners"]),
        commits=tuple(commit(c) for c in data["commits"]),
        branches=tuple(branch(b) for b in data["branches"]),
        issues=None, prs=None, contributors=None, workflows=None,
        # GitHub-API caching is plan-3 work; for now, when cache_load is used
        # we skip those fields. Callers that need GH must call extract() not
        # extract_local().
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synth/test_extractor_repo.py -v`
Expected: PASS — all tests including the new one.

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/extractor/repo.py tests/synth/test_extractor_repo.py
git commit -m "$(cat <<'EOF'
feat(synth): cache RepoSignals by (url, sha) — first call slow, rest free

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 21: Wire the `extract` CLI subcommand

**Files:**
- Modify: `scripts/synth/cli.py` (implement `extract` end-to-end)
- Test: `tests/synth/test_extract_cli.py`

- [ ] **Step 1: Write the failing test**

Create `tests/synth/test_extract_cli.py`:

```python
"""End-to-end: run `python -m scripts.synth extract --profile <yaml>`
on a profile pointing at the tmp_repo fixture; assert world_model.json
is written and contains the expected high-level shape."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_extract_writes_world_model_json(tmp_repo: Path, tmp_path: Path) -> None:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        f"""
customer_id: cust-eval-fake-01
repos:
  - url: repo://fake
    local_path: {tmp_repo}
preset: tiny-test
seed: 7
world_model:
  min_commits_per_persona: 1
  topic_pool_lookback_days: 9999
""".strip()
    )

    out_dir = tmp_path / "out"
    env = os.environ.copy()
    env["ANTHROPIC_API_KEY"] = ""  # extract doesn't need a real key (no LLM unless company_context auto-infers)

    result = subprocess.run(
        [
            sys.executable, "-m", "scripts.synth", "extract",
            "--profile", str(profile),
            "--output-dir", str(out_dir),
        ],
        check=False, capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"

    world_model_path = out_dir / "world_model.json"
    assert world_model_path.exists()
    wm = json.loads(world_model_path.read_text())

    assert wm["company_name"]
    assert wm["seed"] == 7
    assert {p["display_name"] for p in wm["people"]} >= {"Alice", "Bob", "Carol"}
    assert {s["name"] for s in wm["services"]} >= {"payments", "billing", "fake-repo"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synth/test_extract_cli.py -v`
Expected: FAIL — exit 2 (`extract: not yet implemented`).

- [ ] **Step 3: Wire the CLI**

Replace `scripts/synth/cli.py`:

```python
"""CLI dispatch for the synth tool. Subcommands grow over plans 1-3."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from scripts.synth.cache import DiskCache, default_cache_root
from scripts.synth.company_context import (
    CompanyContext,
    infer_company_context,
    load_company_context,
)
from scripts.synth.extractor.github_api import GithubClient
from scripts.synth.extractor.repo import RepoExtractor, RepoSignals
from scripts.synth.llm_client import LlmClient, LlmClientProtocol
from scripts.synth.profile import Profile, load_profile
from scripts.synth.world_model import WorldModel, merge_world_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scripts.synth",
        description="Synthetic company corpus generator for prbe-knowledge eval datasets.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    extract = sub.add_parser(
        "extract",
        help="Extract WorldModel from repos in a profile (no DB writes).",
    )
    extract.add_argument("--profile", required=True, type=str, help="Path to profile YAML.")
    extract.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Where to write world_model.json (default: eval-datasets/<run-id>/).",
    )

    return parser


def _resolve_output_dir(profile: Profile, override: str | None) -> Path:
    if override:
        return Path(override)
    run_id = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ") + f"-{profile.preset}-seed{profile.seed}"
    return Path("eval-datasets") / run_id


async def _extract_async(profile: Profile, out: Path) -> int:
    cache = DiskCache(default_cache_root("repos"))
    gh_token = os.environ.get("GITHUB_TOKEN")
    gh_client = GithubClient(token=gh_token) if gh_token else None
    extractor = RepoExtractor(github_client=gh_client, cache=cache)

    out.mkdir(parents=True, exist_ok=True)

    # Profile world_model knobs override defaults from spec §12.3
    wm_cfg = profile.raw.get("world_model") or {}
    min_threshold = int(wm_cfg.get("min_commits_per_persona", 2))
    max_personas = int(wm_cfg.get("max_personas", 25))
    lookback_days = int(wm_cfg.get("topic_pool_lookback_days", 90))

    from datetime import timedelta
    since = datetime.now(UTC).replace(microsecond=0) - timedelta(days=lookback_days)

    signals: list[RepoSignals] = []
    for repo in profile.repos:
        if repo.local_path is None:
            print(f"warn: repo {repo.url!r} has no local_path; skipping", file=sys.stderr)
            continue
        if gh_client is not None:
            sig = await extractor.extract(repo.url, repo.local_path, since=since, fetch_github=True)
        else:
            sig = extractor.extract_local(repo.url, repo.local_path, since=since)
        signals.append(sig)

    if gh_client is not None:
        await gh_client.close()

    if not signals:
        print("error: no repos extracted; check profile.repos[*].local_path", file=sys.stderr)
        return 3

    cc = await _resolve_company_context(profile, signals, out)

    wm = merge_world_model(
        signals=signals,
        company_name=cc.name,
        seed=profile.seed,
        min_threshold=min_threshold,
        max_personas=max_personas,
        now=datetime.now(UTC),
    )

    (out / "world_model.json").write_text(_dumps(wm))
    (out / "company_context.json").write_text(_dumps(cc))
    print(f"wrote {out}/world_model.json", file=sys.stderr)
    return 0


async def _resolve_company_context(
    profile: Profile,
    signals: list[RepoSignals],
    out: Path,
) -> CompanyContext:
    if profile.company_context_path is not None:
        return load_company_context(profile.company_context_path)
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        # No LLM available; fall back to a minimal stub. The user can
        # add company_context: ./<file> later for richer context.
        return CompanyContext(
            name=_infer_company_name_from_repos(signals),
            stage="unknown",
            headcount=0,
            inferred=True,
        )
    llm: LlmClientProtocol = LlmClient(api_key=api_key)
    try:
        readme_blob = "\n\n".join(
            r.content for sig in signals for r in sig.readmes if r.content
        )[:20_000]
        repo_descs = [s.description or s.url for s in signals]
        cc, raw_yaml = await infer_company_context(
            readme_blob=readme_blob,
            repo_descriptions=repo_descs,
            llm_client=llm,
            model="claude-opus-4-7",
        )
        (out / "inferred-company.yaml").write_text(raw_yaml)
        return cc
    finally:
        await llm.close()


def _infer_company_name_from_repos(signals: list[RepoSignals]) -> str:
    """Best-effort name when no LLM available: longest-common-prefix
    of repo URL owners; else 'unknown'."""
    owners: set[str] = set()
    for sig in signals:
        # github.com/owner/repo  →  owner
        parts = sig.url.rstrip("/").split("/")
        if len(parts) >= 2:
            owners.add(parts[-2])
    if len(owners) == 1:
        return next(iter(owners))
    return "unknown"


def _dumps(obj) -> str:
    """Pretty JSON serializer that handles dataclasses + datetimes + Paths."""
    def encode(v):
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            return {k: encode(getattr(v, k)) for k in (f.name for f in dataclasses.fields(v))}
        if isinstance(v, datetime):
            return v.isoformat()
        if isinstance(v, Path):
            return str(v)
        if isinstance(v, tuple | list):
            return [encode(x) for x in v]
        if isinstance(v, dict):
            return {k: encode(val) for k, val in v.items()}
        return v
    return json.dumps(encode(obj), indent=2, sort_keys=False)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "extract":
        profile = load_profile(Path(args.profile))
        out = _resolve_output_dir(profile, args.output_dir)
        return asyncio.run(_extract_async(profile, out))
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run all synth tests**

Run: `pytest tests/synth/ -v`
Expected: PASS — all tests across all files (60+).

- [ ] **Step 5: Commit**

```bash
git add scripts/synth/cli.py tests/synth/test_extract_cli.py
git commit -m "$(cat <<'EOF'
feat(synth): wire extract CLI end-to-end (profile → WorldModel JSON)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Self-review checklist (run by writer; fix inline)

- **Spec coverage:** WorldModel ✓ (Tasks 10–17), CompanyContext ✓ (19), RepoExtractor ✓ (4–9, 20), caches ✓ (3, 20), LLM client basic ✓ (18), CLI extract ✓ (21). Skipped from spec but explicitly out of plan-1: SOURCE_WRAPPERS, IngestionWriter, tenant lifecycle, archetype library, ScenarioRunner, Planner, Writer, Validator, eval artifacts, presets, full mock-LLM mode, cost ceiling. All deferred to Plans 2–3 with clear handoff points.
- **Type consistency:** `Person.canonical_id`, `Service.qualified`, `Topic.weight`, `DepEdge.from_service/to_service` are referenced consistently across tasks 10–17. `RepoSignals.contributors` is `tuple[Contributor, ...] | None` everywhere (None means "no GitHub token"; empty tuple means "had token but repo had no contributors").
- **Cache-key shape:** `repo:{url}@{sha}` consistently in Task 20.
- **No placeholders:** every step has actual code or commands.
- **Customer-prefix guard:** Task 2 enforces `cust-eval-` / `cust-synth-` so `synth init` / `clean` (Plan 2) inherit the safety net.

## How to test the whole plan locally

After every task completes:

```bash
.venv/bin/pytest tests/synth/ -v
.venv/bin/ruff check scripts/synth/ tests/synth/
.venv/bin/mypy scripts/synth/ tests/synth/
```

The plan is complete when:

```bash
.venv/bin/python -m scripts.synth extract \
  --profile docs/superpowers/specs/2026-04-30-synthetic-company-eval-design.md.profile.yaml \
  --output-dir /tmp/synth-extract-test
```

(profile constructed manually pointing at `~/Desktop/prbe/prbe-knowledge` and `~/Desktop/prbe/prbe-backend`) writes a populated `world_model.json` with personas drawn from real CODEOWNERS, services drawn from real repo dirs, and a topic pool drawn from real recent commits.
