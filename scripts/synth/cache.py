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
