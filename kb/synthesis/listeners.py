"""Notify listeners for the wiki synthesis pipeline.

Two channels:
  - WIKI_PENDING_CHANNEL  — fired by the wiki-cron / manual trigger /
    catchup script. Wakes the wiki-worker app.
  - WIKI_TRIAGED_CHANNEL  — fired by the wiki-worker after a triage
    batch commits its UPDATE. Wakes the wiki-synthesis app.

Both share the same connection management pattern as
`services/ingestion/worker.py:GranolaNotifyListener`: dedicated asyncpg
connection, exponential reconnect backoff, periodic SELECT 1 to detect
half-open connections faster than waiting for the kernel timeout.
"""

from __future__ import annotations

import asyncio
import contextlib

import asyncpg

from engine.shared.db import apply_connection_setup
from engine.shared.logging import get_logger

log = get_logger(__name__)


class NotifyListener:
    """LISTEN on a single Postgres channel; set `wake_event` on every NOTIFY.

    Generic across channels. Used by the wiki-worker to listen on
    `wiki_synthesize_pending` and the wiki-synthesis app to listen on
    `wiki_synthesize_triaged`. The fired wake_event is informational —
    the worker still polls periodically as a safety net for missed
    notifies during a connection drop.
    """

    def __init__(
        self,
        dsn: str,
        channel: str,
        wake_event: asyncio.Event,
        *,
        log_prefix: str | None = None,
    ) -> None:
        self._dsn = dsn
        self._channel = channel
        self._wake = wake_event
        self._log_prefix = log_prefix or f"notify_listener[{channel}]"
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info(f"{self._log_prefix}.start", channel=self._channel)
        backoff = 1.0
        while not self._shutdown.is_set():
            try:
                conn = await asyncpg.connect(self._dsn)
                # Apply the same per-connection bootstrap the pool's
                # ``init=`` hook runs (pins ``search_path`` to include
                # ``ag_catalog``). Without this, any LISTEN callback that
                # touches AGE/Cypher fails with "relation graph_nodes
                # does not exist". See shared/db.py:apply_connection_setup.
                await apply_connection_setup(conn)
            except (asyncpg.PostgresError, OSError) as exc:
                log.warning(
                    f"{self._log_prefix}.connect_failed",
                    error=str(exc),
                    backoff_seconds=backoff,
                )
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._shutdown.wait(), timeout=backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            backoff = 1.0
            try:

                def _on_notify(_conn, _pid, _channel, payload) -> None:
                    log.info(
                        f"{self._log_prefix}.notified",
                        channel=_channel,
                        payload=payload,
                    )
                    self._wake.set()

                await conn.add_listener(self._channel, _on_notify)
                log.info(f"{self._log_prefix}.ready")
                while not self._shutdown.is_set():
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=30.0)
                    except TimeoutError:
                        try:
                            await conn.fetchval("SELECT 1")
                        except (asyncpg.PostgresError, OSError) as exc:
                            log.warning(f"{self._log_prefix}.lost", error=str(exc))
                            break
            finally:
                with contextlib.suppress(Exception):
                    await conn.close()

        log.info(f"{self._log_prefix}.stop")

    def shutdown(self) -> None:
        self._shutdown.set()
