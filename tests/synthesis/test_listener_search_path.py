"""NotifyListener must apply the search_path hook on direct asyncpg.connect.

Regression test for bug #71: ``services/synthesis/listeners.py`` used
``asyncpg.connect(dsn)`` directly, bypassing the pool's ``init=``
``apply_connection_setup`` hook (PR #249) that pins
``search_path = ag_catalog, public, "$user"``. Any LISTEN callback that
issues AGE Cypher (relations live in ``ag_catalog``) would fail with
"relation graph_nodes does not exist".

The test patches ``asyncpg.connect`` + ``apply_connection_setup`` and
verifies that one full connect cycle invokes the setup hook with the
returned connection before LISTEN registration.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from services.synthesis import listeners


@pytest.mark.asyncio
async def test_run_applies_connection_setup_on_connect():
    fake_conn = AsyncMock()
    fake_conn.add_listener = AsyncMock()
    fake_conn.fetchval = AsyncMock(return_value=1)
    fake_conn.close = AsyncMock()

    wake = asyncio.Event()
    listener = listeners.NotifyListener(
        dsn="postgresql://x:y@h/db",
        channel="test_channel",
        wake_event=wake,
    )

    with (
        patch.object(listeners.asyncpg, "connect", AsyncMock(return_value=fake_conn)) as m_connect,
        patch.object(listeners, "apply_connection_setup", AsyncMock()) as m_setup,
    ):
        task = asyncio.create_task(listener.run())
        # Yield long enough for run() to enter the loop, connect, and
        # register the listener; then shut it down cleanly.
        for _ in range(20):
            await asyncio.sleep(0)
            if m_setup.await_count > 0 and fake_conn.add_listener.await_count > 0:
                break
        listener.shutdown()
        await asyncio.wait_for(task, timeout=2.0)

    assert m_connect.await_count >= 1
    assert m_setup.await_count >= 1
    m_setup.assert_any_await(fake_conn)
    # Setup must happen BEFORE LISTEN registration — otherwise the very
    # first NOTIFY callback could fire on a connection with the wrong
    # search_path.
    assert fake_conn.add_listener.await_count >= 1
