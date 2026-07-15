"""Refresh OAuth tokens that expire within the next hour.

Runs every 15 minutes. For each (customer, source) with `expires_at <= NOW + 1h`,
calls the connector's refresh logic — currently implemented per-source via
`exchange_refresh_token` if the connector exposes it, otherwise records the
impending expiry in `integration_tokens.last_refresh_error` so Grafana alerts.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

from engine.ingest.handlers.base import make_default_context
from engine.ingest.handlers.registry import build_connector
from engine.shared.db import close_pool, init_pool
from engine.shared.logging import configure_logging, get_logger
from engine.shared.metrics import counter
from engine.shared.tokens import (
    list_tokens_expiring_within,
    load_token,
    mark_refresh_error,
    save_token,
)

log = get_logger(__name__)

REFRESH_WINDOW = timedelta(hours=1)


async def refresh_all() -> tuple[int, int]:
    configure_logging()
    await init_pool()

    cutoff = datetime.now(UTC) + REFRESH_WINDOW
    targets = await list_tokens_expiring_within(cutoff)

    ctx = make_default_context()
    refreshed = 0
    failed = 0
    for customer_id, source in targets:
        token = await load_token(customer_id, source)
        if token is None or token.refresh_token is None:
            continue
        connector = build_connector(source, ctx)
        refresher = getattr(connector, "exchange_refresh_token", None)
        if refresher is None:
            # Connector doesn't implement refresh — record the stale state.
            await mark_refresh_error(customer_id, source, "connector lacks refresh impl")
            failed += 1
            continue
        try:
            new_token = await refresher(token)
            new_token = new_token.model_copy(update={"customer_id": customer_id})
            await save_token(new_token)
            refreshed += 1
        except Exception as exc:
            await mark_refresh_error(customer_id, source, str(exc))
            failed += 1
            log.warning(
                "token.refresh_failed",
                customer=customer_id,
                source=source.value,
                error=str(exc),
            )

    await ctx.http.aclose()
    await close_pool()
    counter("token.refreshed", refreshed)
    counter("token.refresh_failed", failed)
    log.info("token.refresh_cycle", refreshed=refreshed, failed=failed)
    return refreshed, failed


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(refresh_all())
