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
# JSONB sub-key for per-repo code-graph branch overrides. Shape:
#     {"acme/api": "develop", "acme/worker": "release"}
# Missing repo → fall back to the push payload's `repository.default_branch`.
# Future: dashboard UI writes this; for now operators PATCH it via SQL.
CODE_GRAPH_BRANCH_OVERRIDES_KEY = "code_graph_branch_overrides"


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


async def code_graph_indexed_branch(
    customer_id: str,
    repo: str,
    default_branch: str,
) -> str:
    """Return the branch the code-graph extractor should track for `repo`.

    Resolution: per-repo override under `code_graph_branch_overrides[repo]`
    in `customers.preferences`, falling back to `default_branch` (which
    the caller pulls out of the push webhook payload). This is the
    extension point the dashboard UI will write to when per-repo branch
    selection ships; today it's empty for every tenant and the function
    is a pure pass-through to the default branch.

    Fail-soft: any DB error, malformed JSON, or unexpected value type
    falls back to `default_branch`. Indexing-by-default-branch is the
    safe behavior when prefs are unreadable.
    """
    if not customer_id or not repo:
        return default_branch
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
        return default_branch
    return _coerce_branch_override(raw, repo, default_branch)


def _coerce_branch_override(raw: object, repo: str, fallback: str) -> str:
    """Extract preferences[CODE_GRAPH_BRANCH_OVERRIDES_KEY][repo] or fallback.

    asyncpg may return JSONB as dict or str depending on driver setup;
    handle both, plus any decode/shape mismatch by returning fallback.
    """
    if raw is None:
        return fallback
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            raw = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return fallback
    if not isinstance(raw, dict):
        return fallback
    overrides = raw.get(CODE_GRAPH_BRANCH_OVERRIDES_KEY)
    if not isinstance(overrides, dict):
        return fallback
    branch = overrides.get(repo)
    return branch if isinstance(branch, str) and branch else fallback


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
