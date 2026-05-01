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
    """Return the owners of `path` per CODEOWNERS semantics: last matching rule wins.

    Path must be repo-root-relative with a leading '/' (e.g., '/services/payments/handler.py').
    """
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
