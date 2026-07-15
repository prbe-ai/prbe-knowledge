"""Run periodically (e.g. every minute) to finalize idle Claude Code sessions.

The cron writes a one-byte placeholder R2 object before INSERTing the queue
row, so the worker's R2 read step doesn't fail. The connector's
fetch_supplementary recognizes the ':finalize' suffix and re-reads the
session's actual batches.
"""
from __future__ import annotations

import asyncio

from engine.shared.config import get_settings
from engine.shared.db import init_pool
from kb.session_completer import enqueue_idle_session_finalizers


async def _main() -> None:
    await init_pool()
    try:
        settings = get_settings()
        n = await enqueue_idle_session_finalizers(settings.claude_code_session_idle_minutes)
        print(f"enqueued {n} finalize events")
    finally:
        from engine.shared.db import close_pool
        await close_pool()


if __name__ == "__main__":
    asyncio.run(_main())
