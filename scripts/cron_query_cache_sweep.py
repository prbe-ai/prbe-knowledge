"""Delete expired router-cache rows. Hourly."""

from __future__ import annotations

import asyncio

from shared.db import close_pool, init_pool, raw_conn
from shared.logging import configure_logging, get_logger
from shared.metrics import counter

log = get_logger(__name__)


async def sweep() -> int:
    configure_logging()
    await init_pool()
    async with raw_conn() as conn:
        result = await conn.execute("DELETE FROM query_cache WHERE expires_at < NOW()")
    await close_pool()
    # asyncpg returns "DELETE <n>"
    deleted = int(result.split()[1]) if result and result.startswith("DELETE") else 0
    if deleted:
        log.info("query_cache.swept", deleted=deleted)
        counter("query_cache.swept", deleted)
    return deleted


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(sweep())
