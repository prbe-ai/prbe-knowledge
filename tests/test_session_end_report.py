"""The session-ending report classifies protocol-2 endings correctly.

Its `client_miss_rate` is the number someone will act on (a tap bug hunt), so
each of the four classes gets one row, and a late client finalize must not be
counted as a miss.
"""
from __future__ import annotations

import pytest

from engine.shared.db import get_pool
from scripts.session_end_report import report

C = "end-report-cust"


def _v2(sid: str, seq: int) -> str:
    return f"raw/claude_code/{C}/sessions-v2/{sid}/{seq}-{'a' * 8}.json"


def _marker(sid: str) -> str:
    return f"raw/claude_code/{C}/{sid}/finalize.marker"


@pytest.mark.asyncio
async def test_the_four_endings_and_the_miss_rate(live_db: None) -> None:
    cases = {
        # sid: (keys, stream finalized?)
        "client": ([_v2("client", 0), _v2("client", 1)], True),
        "late": ([_v2("late", 0), _marker("late"), _v2("late", 1)], True),
        "sweep": ([_v2("sweep", 0), _marker("sweep")], False),
        "open": ([_v2("open", 0)], False),
    }
    async with get_pool().acquire() as conn:
        await conn.execute(
            "INSERT INTO customers(customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'r', 'r-hash') ON CONFLICT DO NOTHING", C,
        )
        await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", C)
        await conn.execute("DELETE FROM session_streams WHERE customer_id = $1", C)
        for sid, (keys, finalized) in cases.items():
            await conn.execute(
                "INSERT INTO ingestion_queue (customer_id, source_system, source_event_id, "
                "payload_s3_key, payload_s3_keys, status, priority, version, first_enqueued_at) "
                "VALUES ($1, 'claude_code', $2, $3, $4::text[], 'done', 60, 1, NOW() - INTERVAL '10 days')",
                C, sid, keys[0], keys,
            )
            await conn.execute(
                "INSERT INTO session_streams (customer_id, source_system, session_id, stream_id, "
                "protocol_version, last_seq, source_byte_end, source_line_end, event_end, "
                "prefix_sha256, finalized) VALUES ($1, 'claude_code', $2, $2, 2, 1, 0, 0, 0, '', $3)",
                C, sid, finalized,
            )
    try:
        out = await report(days=7, grace_days=7)
    finally:
        async with get_pool().acquire() as conn:
            await conn.execute("DELETE FROM ingestion_queue WHERE customer_id = $1", C)
            await conn.execute("DELETE FROM session_streams WHERE customer_id = $1", C)
    got = out["v2_endings"]["claude_code"]
    assert {k: got[k] for k in ("client_finalized", "late_client", "sweep_only", "open")} == {
        "client_finalized": 1, "late_client": 1, "sweep_only": 1, "open": 1,
    }
    assert got["client_miss_rate"] == round(1 / 3, 4)
