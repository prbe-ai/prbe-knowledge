"""The CPU pool never outgrows its container (review of #602, 2026-09-29).

The managed worker runs 1 CPU / 1Gi with memory.oom.group=1: two pool
processes there turned "a task OOMs, the pool is rebuilt" into "the container
is killed". The size is capped by the cgroup's own limits.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from engine.ingest.cpu_pool import _effective_size


def _cgroup(tmp_path: Path, cpu_max: str | None, memory_max: str | None) -> str:
    if cpu_max is not None:
        (tmp_path / "cpu.max").write_text(cpu_max + "\n")
    if memory_max is not None:
        (tmp_path / "memory.max").write_text(memory_max + "\n")
    return str(tmp_path)


@pytest.mark.parametrize(
    "cpu_max,memory_max,configured,expected",
    [
        ("100000 100000", str(1024**3), 2, 0),  # managed worker: 1 CPU / 1Gi
        ("200000 100000", str(4 * 1024**3), 2, 2),  # research worker: 2 CPU / 4Gi
        ("100000 100000", str(4 * 1024**3), 2, 1),  # 1 CPU caps to one process
        ("150000 100000", str(4 * 1024**3), 4, 1),  # a fractional CPU rounds down
        ("max 100000", "max", 2, 2),  # no limits: configured
        (None, None, 2, 2),  # unreadable cgroup caps nothing
        ("200000 100000", str(4 * 1024**3), 0, 0),  # 0 stays off
    ],
)
def test_effective_size(tmp_path, cpu_max, memory_max, configured, expected) -> None:
    assert _effective_size(configured, _cgroup(tmp_path, cpu_max, memory_max)) == expected
