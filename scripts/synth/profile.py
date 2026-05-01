"""Profile YAML loader. The profile is the unit of "an eval dataset
configuration" — it points at repos, names the preset, sets the seed.

v1 surface is intentionally minimal: only fields we use this plan.
Plan 3 will extend this with archetype overrides, time_window, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


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
        branch = entry.get("branch")
        if branch is not None and not isinstance(branch, str):
            raise ProfileError(f"repo entry 'branch' must be a string, got {type(branch).__name__}: {branch!r}")
        return RepoSpec(
            url=url,
            local_path=Path(lp).expanduser() if lp else None,
            branch=branch,
        )
    raise ProfileError(f"repo entry must be a string or mapping, got {type(entry).__name__}")


def load_profile(path: Path) -> Profile:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ProfileError(f"profile file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ProfileError(f"profile YAML parse error: {exc}") from exc
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

    preset = raw["preset"]
    if not isinstance(preset, str) or not preset:
        raise ProfileError(f"preset must be a non-empty string, got {preset!r}")

    seed = raw["seed"]
    if isinstance(seed, bool):
        raise ProfileError(f"seed must be an integer, not bool (YAML 'true'/'false' are booleans, not integers): {seed!r}")
    if not isinstance(seed, int):
        raise ProfileError(f"seed must be an integer, got {type(seed).__name__}")

    cc = raw.get("company_context")
    cc_path = Path(cc).expanduser() if isinstance(cc, str) else None

    return Profile(
        customer_id=customer_id,
        repos=repos,
        preset=preset,
        seed=seed,
        company_context_path=cc_path,
        raw=raw,
    )
