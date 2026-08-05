"""The index-contract guard, surfaced to pytest as well as CI.

The real enforcement is `scripts/check_index_contracts.py`, wired into the lint
job -- this repo's CI does not run pytest, so a test-only guard would never
execute. This wrapper exists so the check also fires for anyone running the
suite locally, and so a failure shows up in the place engineers look first.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def test_index_contracts_hold() -> None:
    """Every declared retrieval predicate is still served by its index.

    A failure means either the query drifted (a coalesce() added, a sort key
    appended, a predicate moved across a join) or the index did (renamed,
    re-expressed, dropped from schema.sql). Both are the bug, and neither is
    visible at runtime: the results stay correct and only latency moves.

    Do NOT fix this by editing the contract to match the new code unless the
    index genuinely changed AND you re-measured the query with EXPLAIN.
    """
    proc = subprocess.run(
        [sys.executable, "scripts/check_index_contracts.py"],
        cwd=_REPO, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
