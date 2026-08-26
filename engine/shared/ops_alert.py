"""Fire-and-forget operational alerts to PostHog.

WHY A DIRECT POST RATHER THAN THE POSTHOG SDK
---------------------------------------------
This repository has no PostHog dependency and does not need one for this. The
capture endpoint is a single unauthenticated-by-design POST with a project API
key in the body; adding an SDK to carry it would mean a new dependency, its
transitive tree, and a background flush thread inside a cron process that exits
seconds later. `httpx` is already here.

WHY FAILURES ARE SWALLOWED
--------------------------
The caller is the pg_search guardian, whose job is to make the database plan
again. An alert that raises and aborts the repair would trade an outage for a
notification. So every failure here is logged and swallowed, and the caller's
exit status reflects the DATABASE work only.

That leaves a real gap, stated rather than hidden: if PostHog is down at the
moment the guardian acts, the repair happens and nobody is told. The fallback
is the CronJob's own history -- the guardian exits nonzero on database failure,
so `kubectl get jobs` shows red -- plus the structured log line this always
writes locally, whatever the network did. A missing alert is recoverable; a
database that will not plan is not.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any

import httpx

from engine.shared.logging import get_logger

log = get_logger(__name__)

# Bounded hard. This runs inside a one-minute cron whose useful work is already
# done by the time it is called; a hung analytics endpoint must not hold the
# job open until its activeDeadlineSeconds.
CAPTURE_TIMEOUT_SECONDS = 5.0

DEFAULT_POSTHOG_HOST = "https://us.i.posthog.com"


def _distinct_id() -> str:
    """Stable per-instance identity so events group by cluster, not by pod.

    A CronJob pod name changes every tick, so using it would scatter one
    cluster's alerts across hundreds of distinct ids and make any PostHog
    breakdown useless.
    """
    return os.environ.get("PRBE_CLUSTER_NAME") or os.environ.get("ENVIRONMENT") or "unknown-cluster"


def capture(event: str, properties: dict[str, Any] | None = None) -> bool:
    """Send one PostHog event. Returns True if it was accepted.

    A False return is informational only -- no caller should branch on it to
    decide whether to do its real work.
    """
    # EVERYTHING is inside the try, including the local log.
    #
    # An earlier version logged before entering it, on the reasoning that a
    # local structlog call cannot fail. It could: the kwarg was named `event`,
    # which is the name structlog already binds from the positional message, so
    # every call raised TypeError straight through this function. The guardian
    # then repaired the database, died on the alert, exited nonzero, and never
    # recorded its timeline -- the exact coupling this module exists to
    # prevent, introduced by the one line placed outside the guard.
    #
    # So the guarantee is structural rather than argued: no statement here runs
    # outside the handler, and the contract is "never raises", not "raises only
    # if something unlikely happens".
    try:
        api_key = os.environ.get("POSTHOG_API_KEY")
        payload_props = dict(properties or {})
        # Always leave a local record, whatever the network does. This line is
        # what survives when the alerting path itself is the outage. `alert=`,
        # never `event=` -- see above.
        log.info("ops_alert", alert=event, **payload_props)

        if not api_key:
            # Not an error: self-hosted and local installs legitimately run
            # without analytics configured.
            log.info("ops_alert.skipped", alert=event, reason="POSTHOG_API_KEY unset")
            return False

        host = os.environ.get("POSTHOG_HOST", DEFAULT_POSTHOG_HOST).rstrip("/")
        payload_props.setdefault("cluster", _distinct_id())
        payload_props.setdefault("service", "prbe-knowledge")
        resp = httpx.post(
            f"{host}/capture/",
            json={
                "api_key": api_key,
                "event": event,
                "distinct_id": _distinct_id(),
                "properties": payload_props,
            },
            timeout=CAPTURE_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return True
    except Exception as exc:
        # Last resort, and itself guarded: if the logger is what is broken,
        # reporting that through the logger would re-raise from the handler.
        with contextlib.suppress(Exception):
            log.warning("ops_alert.capture_failed", alert=event, error=str(exc))
        return False
