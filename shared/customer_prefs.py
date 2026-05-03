"""Per-tenant feature toggles read from `customers.preferences` (JSONB).

The column is added by alembic 0023; this module is the read-side. Each
key is a single boolean. Missing keys, missing customer rows, malformed
JSON, and any DB error all resolve to **False** — the policy is
fail-closed / opt-in. The dashboard PATCHes the column; this module
never writes.

Mirrors the per-(agent_kind, source) opt-in posture in
prbe-orchestrator's `is_enrichment_enabled`: a tenant who has not
explicitly opted in does not get the feature, even if the upstream
deploy temporarily can't read the row.
"""

from __future__ import annotations

import json

from shared.db import raw_conn
from shared.logging import get_logger

log = get_logger(__name__)

WIKI_GENERATION_ENABLED_KEY = "wiki_generation_enabled"


async def is_wiki_generation_enabled(customer_id: str) -> bool:
    """Return True iff the tenant has explicitly opted into wiki synthesis.

    Fail-closed on every error path: missing customer, missing key,
    JSON decode failure, unexpected value type, DB error. The wiki
    cron and the queue writer both call this; a False return must
    short-circuit before any LLM-driven work.
    """
    if not customer_id:
        return False
    try:
        async with raw_conn() as conn:
            raw = await conn.fetchval(
                "SELECT preferences FROM customers WHERE customer_id = $1",
                customer_id,
            )
    except Exception as exc:
        log.warning(
            "customer_prefs.read_failed",
            customer=customer_id,
            error=str(exc),
            error_class=type(exc).__name__,
        )
        return False
    return _coerce_bool(raw, WIKI_GENERATION_ENABLED_KEY)


def _coerce_bool(raw: object, key: str) -> bool:
    """Pull `key` out of a JSONB blob; return False unless the value is
    a real bool True. asyncpg may return JSONB as dict or str depending
    on driver setup — handle both.
    """
    if raw is None:
        return False
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return False
    if not isinstance(raw, dict):
        return False
    value = raw.get(key)
    return value is True
