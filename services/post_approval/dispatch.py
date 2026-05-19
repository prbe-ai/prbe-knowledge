"""Single dispatch seam for the (approved ∧ resolved) post-approval trigger.

Whichever code path sets the SECOND of ``{approved_at, resolved_at}`` on
the ``incident_investigations`` row fires the orchestrator dispatch.

Exactly-once HTTP semantics under concurrency are achieved by claiming
the dispatch INSIDE a ``SELECT … FOR UPDATE`` transaction:

1. Inside FOR UPDATE: if both timestamps are set AND
   ``post_approval_dispatched_at IS NULL``, set
   ``post_approval_dispatched_at = now()`` immediately. Build the
   payload from the row. Commit.
2. OUT-OF-DB: POST to the orchestrator with bounded retry.
3. If the POST failed (4xx, 5xx-exhaustion, or transport error): clear
   ``post_approval_dispatched_at`` back to NULL and stamp
   ``metadata.post_approval_dispatch_failed=true``. The dashboard's
   "Re-trigger post-approval" button (``fire_post_approval_dispatch``)
   surfaces these for human recovery.
4. If the POST succeeded: leave ``post_approval_dispatched_at`` in
   place — the row is now in its terminal dispatched state.

The pessimistic claim (set timestamp BEFORE HTTP, clear on failure) is
deliberately preferred over an optimistic flip (HTTP first, set
timestamp on success). Optimistic ordering means two concurrent
callers both observe the guard NULL post-lock-release and both fire
the HTTP — the orchestrator receives the dispatch twice. The
pessimistic ordering serializes guard claims through the FOR UPDATE
lock, so the HTTP fires exactly once across concurrent callers. The
trade-off is a narrow crash-window: if the worker dies between step 1
commit and step 3 cleanup, the row stays "dispatched" but the
orchestrator never received the call. The recovery button is the
intended mitigation for that case.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx

from shared.config import get_settings
from shared.db import with_tenant
from shared.logging import get_logger

log = get_logger(__name__)

_DISPATCH_TIMEOUT_S = 30
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 1.0


def _source_from_incident_doc_id(incident_doc_id: str) -> str:
    """Derive the source-system tag from the Plan 1 incident_doc_id prefix.

    PD and incident.io connectors both write incident docs with stable
    prefixes (``pd:incident:...`` and ``iio:incident:...``); the
    orchestrator wants the source as a clean string for its branching.
    """
    if incident_doc_id.startswith("pd:"):
        return "pagerduty"
    if incident_doc_id.startswith("iio:"):
        return "incident_io"
    raise ValueError(f"unknown incident doc id prefix: {incident_doc_id}")


async def _post_dispatch(payload: dict[str, Any]) -> bool:
    """POST to orchestrator ``/internal/post-approval-actions`` with bounded retry.

    Returns True on a 2xx response, False on any of:
    - 4xx (no retry — caller bug; surface for dashboard)
    - 5xx persisting through ``_MAX_RETRIES``
    - ``httpx.HTTPError`` persisting through ``_MAX_RETRIES`` (timeout,
      connection refused, etc.)
    - misconfigured ``orchestrator_base_url`` (empty)

    Failures are logged at ``warning`` (transient retries) and ``error``
    (4xx / exhausted) levels with the attempt count + status so ops can
    correlate with orchestrator-side logs.
    """
    settings = get_settings()
    base_url = settings.orchestrator_base_url.rstrip("/")
    if not base_url:
        # No orchestrator configured (typical for dev). Leaving this as a
        # silent False would mask a real misconfig in prod — log loud.
        log.error(
            "post_approval.dispatch_no_orchestrator_url",
            payload_keys=list(payload.keys()),
        )
        return False
    url = f"{base_url}/internal/post-approval-actions"
    headers = {
        "x-internal-backend-key":
            settings.internal_backend_api_key.get_secret_value(),
        "content-type": "application/json",
    }

    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=_DISPATCH_TIMEOUT_S) as client:
                resp = await client.post(url, headers=headers, json=payload)
            if 200 <= resp.status_code < 300:
                return True
            if resp.status_code >= 500:
                log.warning(
                    "post_approval.dispatch_retry",
                    status=resp.status_code,
                    attempt=attempt + 1,
                )
                if attempt < _MAX_RETRIES - 1:
                    await asyncio.sleep(_BACKOFF_BASE_S * (3 ** attempt))
                continue
            # 4xx — orchestrator says "no", retrying won't help.
            log.error(
                "post_approval.dispatch_4xx",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return False
        except httpx.HTTPError as exc:
            log.warning(
                "post_approval.dispatch_http_error",
                error=str(exc),
                error_class=type(exc).__name__,
                attempt=attempt + 1,
            )
            # Exhaustion check: don't sleep after the last attempt — the
            # caller is about to bail.
            if attempt < _MAX_RETRIES - 1:
                await asyncio.sleep(_BACKOFF_BASE_S * (3 ** attempt))
            continue
        except Exception as exc:
            # DNS failure (OSError), JSON serialization bug, an
            # ``asyncio.CancelledError`` mid-flight, etc. must NOT
            # propagate up to ``_check_and_dispatch``: that would skip
            # the ``_mark_dispatch_failed`` rollback and leave the guard
            # stamped with no orchestrator action AND no dashboard
            # recovery flag. Convert to a definitive False so the
            # caller's failure path runs. CancelledError is a special
            # case — re-raising is technically cleaner, but a returned
            # False is correct enough (the next event loop step will
            # re-deliver the cancellation) and it preserves the
            # invariant that this function is total over its inputs.
            log.exception(
                "post_approval.dispatch_unexpected_error",
                error_class=type(exc).__name__,
                attempt=attempt + 1,
            )
            return False

    log.error("post_approval.dispatch_failed_exhausted")
    return False


async def _mark_dispatch_failed(
    customer_id: str, incident_doc_id: str, stamped_at: datetime,
) -> None:
    """Rollback step: clear the pessimistic guard back to NULL and
    stamp ``metadata.post_approval_dispatch_failed=true`` so the
    dashboard's recovery button can find this row.

    Runs in its own short transaction OUTSIDE the FOR UPDATE block so
    we don't hold any lock during the metadata mutation. By the time
    this runs, the orchestrator has rejected or timed out — the row's
    post_approval_dispatched_at was set by ``_check_and_dispatch`` and
    must be reverted to keep "NULL == not yet sent" the invariant
    the dashboard reads.

    The CAS predicate (``WHERE post_approval_dispatched_at = $3``,
    matching the stamp this dispatcher itself wrote) ensures that if
    a concurrent recovery dispatch has already cleared and re-stamped
    the guard between our HTTP failure and this rollback, our stale
    rollback is a no-op — we don't overwrite the recovery's state and
    we don't falsely raise ``post_approval_dispatch_failed`` on a row
    that just succeeded. Race this guards:

    1. Initial dispatch stamps at T1, HTTP slow.
    2. Operator clicks Re-trigger → ``fire_post_approval_dispatch``
       clears the guard.
    3. Recovery dispatch stamps at T2, HTTP succeeds at T3.
    4. Initial dispatch's HTTP finally fails at T4. Without CAS, the
       rollback would NULL the guard and re-set the failure flag,
       lying to the dashboard. With CAS, ``WHERE … = $T1`` matches
       zero rows and we skip cleanly.
    """
    async with with_tenant(customer_id) as conn:
        result = await conn.execute(
            """
            UPDATE incident_investigations
            SET post_approval_dispatched_at = NULL,
                metadata = COALESCE(metadata, '{}'::jsonb)
                          || jsonb_build_object(
                               'post_approval_dispatch_failed', true),
                updated_at = now()
            WHERE customer_id = $1 AND incident_doc_id = $2
              AND post_approval_dispatched_at = $3;
            """,
            customer_id, incident_doc_id, stamped_at,
        )
    # asyncpg returns "UPDATE <n>" as the command tag. n=0 means
    # someone else has moved the row's state — e.g. a concurrent
    # ``fire_post_approval_dispatch`` cleared our stamp and a recovery
    # dispatch re-stamped. The right move is to log + drop, not retry.
    if result.endswith(" 0"):
        log.info(
            "post_approval.dispatch_rollback_skipped_stale",
            customer_id=customer_id,
            incident_doc_id=incident_doc_id,
            stamped_at=stamped_at.isoformat(),
        )


async def _check_and_dispatch(
    customer_id: str, incident_doc_id: str,
) -> None:
    """Inside a FOR UPDATE row lock: if both timestamps are set and the
    one-shot guard is NULL, claim the dispatch (set guard, build
    payload), release the lock, then POST.

    The pessimistic claim is the key to exactly-once HTTP semantics:
    concurrent callers serialize through the FOR UPDATE lock and only
    one observes ``post_approval_dispatched_at IS NULL`` — that
    caller claims the dispatch. The other caller's SELECT FOR UPDATE
    waits, then sees the guard set, then returns without dispatching.

    The HTTP call happens OUTSIDE the transaction. Holding a row lock
    across a 60s orchestrator round-trip would funnel every concurrent
    (approved ∧ resolved) flip through serial waits and burn CPU on
    the pg_lock spin.
    """
    payload: dict[str, Any] | None = None
    stamped_at: datetime | None = None
    async with with_tenant(customer_id) as conn:
        # with_tenant already opens a transaction; the FOR UPDATE row
        # lock is held until that outer tx commits (i.e., until the
        # `async with` block exits). No nested conn.transaction() is
        # needed (or correct — asyncpg would treat that as a SAVEPOINT
        # and the FOR UPDATE lock would still scope to the outer tx).
        row = await conn.fetchrow(
            """
            SELECT customer_id, incident_doc_id, current_report_doc_id,
                   approved_at, resolved_at, post_approval_dispatched_at
            FROM incident_investigations
            WHERE customer_id = $1 AND incident_doc_id = $2
            FOR UPDATE;
            """,
            customer_id, incident_doc_id,
        )
        if row is None:
            return
        if row["approved_at"] is None or row["resolved_at"] is None:
            return
        if row["post_approval_dispatched_at"] is not None:
            return
        # Pessimistic claim: stamp the guard NOW, inside the lock. Any
        # concurrent caller blocked on FOR UPDATE will see the guard set
        # when their SELECT returns and return without dispatching.
        #
        # The stamp is computed in Python rather than via SQL ``now()``
        # so the exact value can be threaded through to
        # ``_mark_dispatch_failed`` as a CAS predicate (see I1 race).
        # Without that, a stale rollback after a concurrent recovery
        # dispatch could stomp the recovery's good state.
        stamped_at = datetime.now(UTC)
        await conn.execute(
            """
            UPDATE incident_investigations
            SET post_approval_dispatched_at = $3,
                updated_at = $3
            WHERE customer_id = $1 AND incident_doc_id = $2;
            """,
            customer_id, incident_doc_id, stamped_at,
        )
        payload = {
            "customer_id": customer_id,
            "incident_doc_id": incident_doc_id,
            "investigation_doc_id": row["current_report_doc_id"],
            "source": _source_from_incident_doc_id(incident_doc_id),
            "approved_at": row["approved_at"].isoformat(),
            "resolved_at": row["resolved_at"].isoformat(),
        }
    # Lock released. Now hit the orchestrator without holding it.
    if payload is None or stamped_at is None:
        return

    ok = await _post_dispatch(payload)
    if not ok:
        # Rollback the pessimistic claim so the recovery button can
        # re-dispatch. Crash-window caveat: if the process dies here,
        # the guard stays set with no orchestrator-side action — the
        # human operator hits the button. CAS predicate guards against
        # racing a concurrent recovery dispatch (see I1 docstring).
        await _mark_dispatch_failed(
            customer_id, incident_doc_id, stamped_at,
        )
        log.error(
            "post_approval.dispatch_failed",
            customer_id=customer_id,
            incident_doc_id=incident_doc_id,
        )
        return

    log.info(
        "post_approval.dispatched",
        customer_id=customer_id,
        incident_doc_id=incident_doc_id,
    )


async def on_resolution_event(
    customer_id: str,
    incident_doc_id: str,
    resolved_at: datetime | None = None,
) -> None:
    """Called by the ingest worker after a resolution event normalizes.

    Idempotent: the COALESCE preserves the FIRST observed ``resolved_at``
    so out-of-order webhook re-deliveries can't overwrite the canonical
    resolution time.

    Creates a partial row (state='pending_review', approved_at NULL) if
    no investigation row exists yet — the investigation pipeline will
    UPSERT into the same primary key when its created webhook lands.
    """
    resolved_at = resolved_at or datetime.now(UTC)
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            INSERT INTO incident_investigations
                (customer_id, incident_doc_id, state, resolved_at)
            VALUES ($1, $2, 'pending_review', $3)
            ON CONFLICT (customer_id, incident_doc_id) DO UPDATE
              SET resolved_at = COALESCE(
                    incident_investigations.resolved_at, EXCLUDED.resolved_at
                  ),
                  updated_at = now();
            """,
            customer_id, incident_doc_id, resolved_at,
        )
    await _check_and_dispatch(customer_id, incident_doc_id)


async def on_approval(
    customer_id: str,
    incident_doc_id: str,
    approved_at: datetime | None = None,
) -> None:
    """Called by ``mark_approved`` in ``services/ingestion/investigation_state.py``.

    Idempotent: COALESCE preserves the first observed approval time. A
    no-op when the row does not exist yet (e.g. approval was called for
    a never-investigated incident — should be impossible in practice
    since mark_approved itself requires a prior row, but we don't UPSERT
    here because approval-without-investigation has no sensible state).
    """
    approved_at = approved_at or datetime.now(UTC)
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE incident_investigations
            SET approved_at = COALESCE(approved_at, $3),
                updated_at = now()
            WHERE customer_id = $1 AND incident_doc_id = $2;
            """,
            customer_id, incident_doc_id, approved_at,
        )
    await _check_and_dispatch(customer_id, incident_doc_id)


async def fire_post_approval_dispatch(
    customer_id: str,
    incident_doc_id: str,
) -> None:
    """Manual re-trigger entrypoint (dashboard 'Re-trigger post-approval').

    Clears the dispatch guard + the ``post_approval_dispatch_failed``
    metadata flag, then re-runs ``_check_and_dispatch``. Intended for
    the failed-dispatch recovery flow only — calling this on a row that
    successfully dispatched already will re-dispatch (which is the
    correct behavior for the recovery semantic: the operator hit the
    button because they want it to fire again).
    """
    async with with_tenant(customer_id) as conn:
        await conn.execute(
            """
            UPDATE incident_investigations
            SET post_approval_dispatched_at = NULL,
                metadata = (COALESCE(metadata, '{}'::jsonb)
                            - 'post_approval_dispatch_failed'),
                updated_at = now()
            WHERE customer_id = $1 AND incident_doc_id = $2;
            """,
            customer_id, incident_doc_id,
        )
    await _check_and_dispatch(customer_id, incident_doc_id)


__all__ = [
    "fire_post_approval_dispatch",
    "on_approval",
    "on_resolution_event",
]
