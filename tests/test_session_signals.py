"""The one rule for "has this session ended", and its SQL twin.

The worker applies the rule in Python (kb/handlers/claude_code.py); the sweep
and the backfill apply it in SQL. The daily re-mining bug was exactly two
places disagreeing about this question, so the two forms are pinned against
the SAME fixtures here, the SQL one on a real Postgres.
"""
from __future__ import annotations

import pytest

from engine.shared import session_signals as sig

SID = "0b1c2d3e-aaaa-bbbb-cccc-123456789abc"
C = "cust"
BATCH0 = f"raw/claude_code/{C}/2026/04/29/{SID}:0.json"
BATCH1 = f"raw/claude_code/{C}/2026/04/30/{SID}:1.json"
CLIENT_FIN = f"raw/claude_code/{C}/2026/04/30/{SID}.json"
MARKER = sig.cron_marker_key("claude_code", C, SID)
V2 = f"raw/claude_code/{C}/sessions-v2/{SID}/3-{'a' * 64}.json"

CASES = [
    ([], False),
    ([BATCH0], False),
    ([BATCH0, CLIENT_FIN], True),
    ([BATCH0, MARKER], True),
    ([BATCH0, CLIENT_FIN, BATCH1], False),
    ([BATCH0, MARKER, BATCH1], False),
    ([BATCH0, MARKER, BATCH1, MARKER], True),
    # A v2 batch key never looks like a v1 client finalize, whatever it holds.
    ([V2], False),
    ([V2, MARKER], True),
    # Another session's finalize on this row (impossible in practice) is not ours.
    ([BATCH0, f"raw/claude_code/{C}/2026/04/30/other-session.json"], False),
]


def test_the_marker_key_is_the_one_the_sweep_has_always_written() -> None:
    assert f"raw/claude_code/{C}/{SID}/finalize.marker" == MARKER
    assert sig.is_cron_marker_key(MARKER) and not sig.is_cron_marker_key(BATCH0)


@pytest.mark.parametrize(("keys", "ended"), CASES)
def test_last_key_rule(keys: list[str], ended: bool) -> None:
    assert sig.last_key_ends_v1_session(keys, SID) is ended


def test_completed_by_values_are_stable() -> None:
    """They are written to logs and read by dashboards and SQL."""
    assert {c.value for c in sig.CompletedBy} == {"v2_finalize", "v1_client_finalize", "cron_marker"}


@pytest.mark.asyncio
async def test_the_sql_twin_agrees_with_the_python_rule(live_db: None) -> None:
    from engine.shared.db import get_pool

    expr = sig.last_key_ends_v1_session_sql("k", "s")
    v2 = sig.has_v2_key_sql("k")
    async with get_pool().acquire() as conn:
        for keys, ended in CASES:
            got = await conn.fetchval(
                f"SELECT {expr} FROM (SELECT $1::text[] AS k, $2::text AS s) AS t", keys, SID
            )
            assert got is ended, (keys, got)
            has_v2 = await conn.fetchval(
                f"SELECT {v2} FROM (SELECT $1::text[] AS k) AS t", keys
            )
            assert has_v2 is any(sig.is_v2_key(k) for k in keys)
        # A session id with LIKE wildcards in it matches only itself.
        odd = "a_b%c"
        got = await conn.fetchval(
            f"SELECT {expr} FROM (SELECT $1::text[] AS k, $2::text AS s) AS t",
            ["raw/claude_code/c/2026/04/30/aXbYYc.json"],
            odd,
        )
        assert got is False
