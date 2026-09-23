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

#: A pre-0026 legacy row identity `<session>:<batch>`: its batch keys end in
#: `/<session>:<batch>.json`, which would otherwise read as a client finalize.
LEGACY_SID = f"{SID}:0"
LEGACY_KEY = f"raw/claude_code/{C}/2026/04/29/{SID}:0.json"


def test_the_marker_key_is_the_one_the_sweep_has_always_written() -> None:
    assert f"raw/claude_code/{C}/{SID}/finalize.marker" == MARKER
    assert sig.is_cron_marker_key(MARKER) and not sig.is_cron_marker_key(BATCH0)


@pytest.mark.parametrize(("keys", "ended"), CASES)
def test_last_key_rule(keys: list[str], ended: bool) -> None:
    assert sig.last_key_ends_v1_session(keys, SID) is ended


def test_a_slash_in_a_session_id_matches_its_stored_key() -> None:
    """kb/ingestion_app._payload_key stores `/` as `_`; both forms of the rule
    must compare against the stored spelling, or such a session never ends."""
    sid = "team/abc"
    key = "raw/claude_code/c/2026/04/30/team_abc.json"
    assert sig.signal_for_key(key, sid) == sig.CompletedBy.V1_CLIENT_FINALIZE


def test_a_legacy_row_identity_is_never_an_end_signal() -> None:
    assert sig.signal_for_key(LEGACY_KEY, LEGACY_SID) is None
    assert sig.last_key_ends_v1_session([LEGACY_KEY], LEGACY_SID) is False


def _body_for(key: str) -> bytes:
    """What each key's object really holds."""
    import orjson

    if sig.is_cron_marker_key(key):
        return sig.cron_marker_body(SID)
    if sig.is_v1_client_finalize_key(key, SID) or key.endswith("other-session.json"):
        return orjson.dumps({"payload": {"finalize": True, "session_id": SID}})
    if sig.is_v2_key(key):
        return orjson.dumps({"payload": {"protocol_version": 2, "batch_seq": 3, "session_id": SID,
                                         "events": [{"line_no": 9, "raw": {}}]}})
    return orjson.dumps({"payload": {"session_id": SID, "batch_seq": 0,
                                     "events": [{"line_no": 0, "raw": {}}]}})


@pytest.mark.parametrize(("keys", "ended"), [c for c in CASES if c[0]])
@pytest.mark.asyncio
async def test_the_worker_agrees_with_the_sweep_on_every_fixture(monkeypatch, keys, ended) -> None:
    """The bug this module exists for was the worker and the sweep answering
    this question differently. Run the WORKER's real code over the same rows."""
    from datetime import UTC, datetime

    from engine.ingest.handlers.base import make_default_context
    from engine.shared.constants import SourceSystem
    from engine.shared.models import WebhookEvent
    from kb.handlers import claude_code

    blobs = {k: _body_for(k) for k in keys}

    class _Store:
        async def bucket_for(self, customer):
            return "b"

        async def get(self, bucket, key):
            return blobs[key]

    monkeypatch.setattr(claude_code, "get_store", lambda: _Store())
    event = WebhookEvent(
        customer_id=C, source_system=SourceSystem.CLAUDE_CODE, source_event_id=SID,
        received_at=datetime.now(UTC), payload_s3_key=keys[0], payload_s3_keys=keys,
        raw_payload={"session_id": SID}, headers={},
    )
    connector = claude_code.ClaudeCodeConnector(make_default_context())
    hydrated = await connector.fetch_supplementary(event, None)
    # A v2 row is decided by its sequence in the worker; the key rule only
    # speaks for the marker on top of one.
    if any(sig.is_v2_key(k) for k in keys) and not sig.is_cron_marker_key(keys[-1]):
        assert hydrated["session_complete"] is False
    else:
        assert hydrated["session_complete"] is ended


def test_completed_by_values_are_stable() -> None:
    """They are written to logs and read by dashboards and SQL."""
    assert {c.value for c in sig.CompletedBy} == {"v2_finalize", "v1_client_finalize", "cron_marker"}


@pytest.mark.asyncio
async def test_the_sql_twin_agrees_with_the_python_rule(live_db: None) -> None:
    from engine.shared.db import get_pool

    expr = sig.ends_v1_session_sql(sig.last_key_sql("k"), "s")
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
        got = await conn.fetchval(
            f"SELECT {expr} FROM (SELECT $1::text[] AS k, $2::text AS s) AS t",
            ["raw/claude_code/c/2026/04/30/team_abc.json"],
            "team/abc",
        )
        assert got is True
        # The legacy identity guard holds in SQL too.
        got = await conn.fetchval(
            f"SELECT {expr} FROM (SELECT $1::text[] AS k, $2::text AS s) AS t",
            [LEGACY_KEY],
            LEGACY_SID,
        )
        assert got is False
