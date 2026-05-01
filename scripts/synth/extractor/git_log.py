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
# Collision risk: a literal `\x1e` byte in commit body would split the record. Vanishingly rare in practice (no source code uses it).
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
    raw = _run(repo, "show", "--name-only", "--format=", sha)
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
