"""Periodic token-health probe for OAuth integrations.

Runs every ~6 hours. For each (customer_id, source_system) with a singleton
token in `status='active'`, calls the connector's `verify_token_health()`
contract — currently implemented for Linear, easy to extend per source.

A False return value (definitive 401 / AUTHENTICATION_ERROR) flips the row
to `status='auth_failed'`, which causes `load_token` to stop returning the
dead credential. The dashboard surfaces the `auth_failed` state so the user
knows to re-OAuth.

A raised exception (5xx, network blip) is treated as inconclusive — the
row is left active so a transient outage doesn't cascade into a fleet of
false-positive disconnects. The next cron tick re-probes.

This loop ONLY downgrades. It never re-promotes `auth_failed` rows back to
`active`; that path is the OAuth callback's job after a fresh user-driven
re-install.

Why this exists: Linear's API silently started returning 401 to
probe-founders' token after 4 days of normal operation in May 2026. The
DB row stayed `active`, webhooks stopped flowing, and nothing alerted —
because no code path was checking the credential's actual liveness. This
cron is the missing observability hook.
"""

from __future__ import annotations

import asyncio

from services.ingestion.handlers.base import make_default_context
from services.ingestion.handlers.registry import build_connector
from shared.constants import IntegrationStatus, SourceSystem
from shared.db import close_pool, get_pool, init_pool
from shared.logging import configure_logging, get_logger
from shared.metrics import counter
from shared.tokens import load_token, mark_token_auth_failed

log = get_logger(__name__)


async def _list_active_tokens() -> list[tuple[str, SourceSystem]]:
    async with get_pool().acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT customer_id, source_system
            FROM integration_tokens
            WHERE status = $1
              AND device_id IS NULL
            ORDER BY customer_id, source_system
            """,
            IntegrationStatus.ACTIVE.value,
        )
    return [(r["customer_id"], SourceSystem(r["source_system"])) for r in rows]


async def health_check_all() -> tuple[int, int, int]:
    """Iterate every active singleton token and probe its liveness.

    Returns (probed, failed, skipped) — `failed` is the count of rows
    actually flipped to `auth_failed`, `skipped` is rows whose connector
    doesn't implement `verify_token_health` yet, `probed` is the total
    attempts (failed + healthy + transient-error).
    """
    configure_logging()
    await init_pool()

    targets = await _list_active_tokens()
    ctx = make_default_context()
    probed = 0
    failed = 0
    skipped = 0
    for customer_id, source in targets:
        token = await load_token(customer_id, source)
        if token is None:
            continue
        connector = build_connector(source, ctx)
        verify = getattr(connector, "verify_token_health", None)
        if verify is None:
            # Connector hasn't opted in. Not an error — many connectors
            # don't have a cheap liveness endpoint, or the failure mode
            # (e.g. webhook signing) doesn't manifest as a probeable 401.
            skipped += 1
            continue
        probed += 1
        try:
            healthy = await verify(token)
        except Exception as exc:
            # Transient — skip flagging, log so Grafana shows the rate.
            log.warning(
                "token.health_probe_transient",
                customer=customer_id,
                source=source.value,
                error=type(exc).__name__,
            )
            continue
        if healthy:
            continue
        flipped = await mark_token_auth_failed(
            customer_id,
            source,
            f"verify_token_health returned False at {asyncio.get_event_loop().time()}",
        )
        if flipped:
            failed += 1
            log.warning(
                "token.health_probe_auth_failed",
                customer=customer_id,
                source=source.value,
            )

    await ctx.http.aclose()
    await close_pool()
    counter("token.health_probed", probed)
    counter("token.health_auth_failed", failed)
    counter("token.health_skipped", skipped)
    log.info(
        "token.health_check_cycle",
        probed=probed,
        failed=failed,
        skipped=skipped,
    )
    return probed, failed, skipped


if __name__ == "__main__":  # pragma: no cover
    asyncio.run(health_check_all())
