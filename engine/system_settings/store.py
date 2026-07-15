"""In-process accessor for `system_settings` rows with a short TTL cache.

Webhook ingestion is the hot caller — every POST to /webhooks/{source}
checks the killswitch before doing any other work. The cache caps DB
load at ~2 reads/min/process regardless of webhook volume.

Cache semantics:
  - 30s TTL.
  - Per-process, not shared across the ingestion fleet — each pod has
    its own cache, so a flip propagates within 30s.
  - On DB error: bubble up. Caller (webhook handler) decides what to do.
    The webhook handler treats DB errors as fail-OPEN (assume enabled),
    matching the missing-row case below — better to keep ingesting on a
    transient DB blip than halt every customer's daemon.
  - Missing row: returns enabled=True (fail open). The seed migration
    inserts the row, so this only happens on a half-deployed state or
    if someone deletes the row by hand.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from shared.db import get_pool

log = logging.getLogger("prbe-knowledge.system_settings")

_CACHE_TTL_S = 30.0


@dataclass(frozen=True)
class IngestionKillswitch:
    enabled: bool
    reason: str | None
    fetched_at: float


_cache: IngestionKillswitch | None = None
_cache_lock = asyncio.Lock()


async def get_ingestion_killswitch(
    *, force_refresh: bool = False
) -> IngestionKillswitch:
    """Return the current killswitch state. Cached 30s.

    `force_refresh=True` bypasses the cache; use it from the
    /api/internal/ingestion-status endpoint so admin polling is never
    served stale.
    """
    global _cache
    now = time.monotonic()
    if (
        not force_refresh
        and _cache is not None
        and (now - _cache.fetched_at) < _CACHE_TTL_S
    ):
        return _cache

    async with _cache_lock:
        # Double-check inside the lock so a thundering-herd refresh only
        # hits the DB once.
        now = time.monotonic()
        if (
            not force_refresh
            and _cache is not None
            and (now - _cache.fetched_at) < _CACHE_TTL_S
        ):
            return _cache

        try:
            async with get_pool().acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT value FROM system_settings WHERE key = $1",
                    "ingestion_killswitch",
                )
        except Exception as exc:
            # Table missing (mid-migration) or DB unreachable. Fail OPEN —
            # we'd rather keep ingesting than halt every customer over a
            # transient blip. Log so this doesn't go unnoticed.
            log.warning(
                "ingestion_killswitch read failed; failing open (enabled=True): %s",
                exc,
            )
            fallback = IngestionKillswitch(
                enabled=True, reason=None, fetched_at=now
            )
            # Cache the fallback briefly so we don't hammer the DB while
            # it's down. Same TTL so recovery is automatic.
            _cache = fallback
            return fallback

        if row is None:
            log.warning(
                "system_settings.ingestion_killswitch row missing; failing open"
            )
            value: dict = {"enabled": True, "reason": None}
        else:
            raw = row["value"]
            # asyncpg returns JSONB as a string by default unless a codec
            # is registered. Be defensive on either shape.
            if isinstance(raw, str):
                import json as _json

                try:
                    value = _json.loads(raw)
                except _json.JSONDecodeError:
                    log.warning(
                        "ingestion_killswitch JSON decode failed; failing open"
                    )
                    value = {"enabled": True, "reason": None}
            elif isinstance(raw, dict):
                value = raw
            else:
                log.warning(
                    "ingestion_killswitch unexpected value type %s; failing open",
                    type(raw).__name__,
                )
                value = {"enabled": True, "reason": None}

        result = IngestionKillswitch(
            enabled=bool(value.get("enabled", True)),
            reason=value.get("reason"),
            fetched_at=now,
        )
        _cache = result
        return result


def invalidate_cache() -> None:
    """Force the next `get_ingestion_killswitch()` call to re-read from
    the DB. Intended for tests; production callers rely on the TTL."""
    global _cache
    _cache = None
