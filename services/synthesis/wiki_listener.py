"""LISTEN wiki_synthesize on a dedicated asyncpg connection.

Mirrors `services.ingestion.worker.GranolaNotifyListener` verbatim — the
reconnect/backoff/keepalive shape is the same; only the channel and the
log keys differ. Sets a shared `asyncio.Event` whenever a NOTIFY arrives,
which `WikiSynthesisCron` uses to break its periodic-tick sleep early.
"""

from __future__ import annotations

import asyncio
import contextlib

import asyncpg

from shared.constants import WIKI_SYNTHESIZE_CHANNEL
from shared.logging import get_logger

log = get_logger(__name__)


class WikiSynthesisListener:
    def __init__(self, dsn: str, wake_event: asyncio.Event) -> None:
        self._dsn = dsn
        self._wake = wake_event
        self._shutdown = asyncio.Event()

    async def run(self) -> None:
        log.info("wiki_synthesis_listener.start", channel=WIKI_SYNTHESIZE_CHANNEL)
        backoff = 1.0
        while not self._shutdown.is_set():
            try:
                conn = await asyncpg.connect(self._dsn)
            except (asyncpg.PostgresError, OSError) as exc:
                log.warning(
                    "wiki_synthesis_listener.connect_failed",
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
                        "wiki_synthesis_listener.notified",
                        payload=payload,
                    )
                    self._wake.set()

                await conn.add_listener(WIKI_SYNTHESIZE_CHANNEL, _on_notify)
                log.info("wiki_synthesis_listener.ready")
                while not self._shutdown.is_set():
                    try:
                        await asyncio.wait_for(self._shutdown.wait(), timeout=30.0)
                    except TimeoutError:
                        try:
                            await conn.fetchval("SELECT 1")
                        except (asyncpg.PostgresError, OSError) as exc:
                            log.warning("wiki_synthesis_listener.lost", error=str(exc))
                            break
            finally:
                with contextlib.suppress(Exception):
                    await conn.close()

        log.info("wiki_synthesis_listener.stop")

    def shutdown(self) -> None:
        self._shutdown.set()
