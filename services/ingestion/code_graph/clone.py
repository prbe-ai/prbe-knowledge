"""Initial-backfill scratch-dir manager: shallow clone + worktree walk.

The worker keeps cloned repos under `~/work/code_graph/<customer>/<repo_hash>/`
during a backfill and prunes them when the backfill completes (incremental
push events don't need the clone — they fetch via GitHub Contents API).

Steady-state worker tmpfs footprint trends to zero: a clone exists only
during an in-flight backfill and is removed at the final batch's yield.

Cloned bytes never leave the worker. Symbol Documents and graph rows are
the only persisted output of the extraction.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from shared.config import get_settings
from shared.exceptions import SourceAPIError
from shared.logging import get_logger

log = get_logger(__name__)


@dataclass(slots=True)
class FileEntry:
    """One file from a cloned worktree, ready for hashing + extraction."""

    rel_path: str           # repo-relative POSIX path
    content: bytes


def scratch_root() -> Path:
    """Per-worker scratch root. Override via PRBE_CODE_GRAPH_SCRATCH env var."""
    settings = get_settings()
    override = getattr(settings, "code_graph_scratch_root", None)
    if override:
        return Path(override)
    return Path.home() / "work" / "code_graph"


def repo_dir(customer_id: str, repo: str) -> Path:
    """Stable per-(customer, repo) clone directory.

    Hashes (customer_id, repo) so the path is opaque + filesystem-safe even
    when repo names contain characters worktree-locked filesystems hate.
    """
    digest = hashlib.sha256(f"{customer_id}::{repo}".encode()).hexdigest()[:16]
    return scratch_root() / digest


async def shallow_clone(
    repo: str,
    sha: str,
    token: str | None,
    target_dir: Path,
) -> None:
    """Shallow-clone `repo` at `sha` into `target_dir`.

    `git clone --filter=blob:none --depth=1` keeps disk pressure low — only
    the committed blobs at HEAD, no history. Re-uses the existing dir if
    one is already present (no-op in the steady state of a resumed
    backfill).

    Auth: HTTPS with `x-access-token:<token>` per GitHub App convention. If
    token is None we attempt anonymous clone (works for public repos in
    dev / fixture flows).
    """
    if target_dir.exists() and (target_dir / ".git").exists():
        log.info("code_graph.clone.reuse", repo=repo, dir=str(target_dir))
        return

    target_dir.parent.mkdir(parents=True, exist_ok=True)
    url = (
        f"https://x-access-token:{token}@github.com/{repo}.git"
        if token
        else f"https://github.com/{repo}.git"
    )

    cmd = [
        "git",
        "clone",
        "--filter=blob:none",
        "--depth=1",
        url,
        str(target_dir),
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        msg = stderr.decode("utf-8", errors="replace")[:500]
        raise SourceAPIError(f"git clone failed for {repo}: {msg}")

    # If sha differs from HEAD (rare on shallow), fetch + checkout. Skipped
    # by default; backfill always uses HEAD which is what we just cloned.
    if sha and sha != "HEAD":
        await _fetch_and_checkout(target_dir, sha)


async def _fetch_and_checkout(target_dir: Path, sha: str) -> None:
    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(target_dir),
        "fetch",
        "--depth=1",
        "origin",
        sha,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        log.warning(
            "code_graph.clone.fetch_failed",
            sha=sha,
            stderr=stderr.decode("utf-8", errors="replace")[:200],
        )
        return  # Soft-fail; backfill will continue with HEAD.

    proc = await asyncio.create_subprocess_exec(
        "git",
        "-C",
        str(target_dir),
        "checkout",
        sha,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    await proc.communicate()


def prune_scratch(customer_id: str, repo: str) -> None:
    """Remove the per-(customer, repo) clone dir after backfill completion."""
    target = repo_dir(customer_id, repo)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
        log.info("code_graph.clone.pruned", repo=repo, dir=str(target))


async def walk_files(
    target_dir: Path,
    extensions: tuple[str, ...],
) -> AsyncIterator[FileEntry]:
    """Yield FileEntry for every tracked file matching `extensions`.

    Skips:
      - .git directory
      - symlinks (avoid escape attacks)
      - files larger than the per-file size cap (default 1MiB) — pathological
        generated files don't pay for themselves.
    """
    settings = get_settings()
    max_bytes = getattr(settings, "code_graph_max_file_bytes", 1_048_576)

    def _walk_blocking() -> list[FileEntry]:
        entries: list[FileEntry] = []
        for path in target_dir.rglob("*"):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                if path.relative_to(target_dir).parts and path.relative_to(target_dir).parts[0] == ".git":
                    continue
            except ValueError:
                continue
            if not any(path.name.endswith(ext) for ext in extensions):
                continue
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > max_bytes:
                continue
            try:
                content = path.read_bytes()
            except OSError:
                continue
            rel = path.relative_to(target_dir).as_posix()
            entries.append(FileEntry(rel_path=rel, content=content))
        return entries

    # Walk the tree off the event loop — disk I/O on big repos can stall.
    entries = await asyncio.to_thread(_walk_blocking)
    for entry in entries:
        yield entry


__all__ = [
    "FileEntry",
    "prune_scratch",
    "repo_dir",
    "scratch_root",
    "shallow_clone",
    "walk_files",
]
