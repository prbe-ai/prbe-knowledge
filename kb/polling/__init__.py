"""Per-source polling framework for self-host ingestion (Phase 2 PR B).

For self-host customers there are no public webhook endpoints — every
source has to be POLLED outbound from inside the customer's cluster.
This package is the framework the per-source pollers (one per
SourceSystem) register with.

Pieces:

  * ``BasePoller`` (base.py) — the per-source ABC. Each source poller
    implements ``fetch_since(cursor)``, ``extract_documents(raw)``, and
    ``advance_cursor(raw)``.
  * ``register_poller`` / ``get_poller`` (base.py) — registry the
    scheduler reads at tick time.
  * ``PollScheduler`` (scheduler.py) — tick loop that walks every
    enabled (customer x source x resource) cursor row and dispatches
    to the registered poller. Per-customer cursor reads/writes are
    wrapped in ``with_tenant`` so RLS gates everything.
  * Cursor helpers (cursors.py) — read/write/error-stamp helpers
    around the ``ingestion_cursors`` table.

Wiring up a new source is just:

  1. Add the source value to ``shared.constants.SourceSystem``.
  2. Subclass ``BasePoller`` in ``services/ingestion/polling/<source>.py``.
  3. Call ``register_poller(<source>, MyPoller)`` at module import.
  4. The scheduler picks it up on the next tick.

This package does NOT wire itself into any boot path — that lands in
the chart pair work (PR C). Per-source poller implementations land in
PRs E1-E5 (one per source).
"""

from __future__ import annotations

# Per-source pollers — importing each module triggers its
# ``register_poller(...)`` side effect so the scheduler's registry
# resolves SourceSystem -> poller-class without any extra wiring at
# the boot site.
from kb.polling import sentry as _sentry_poller  # noqa: F401
from kb.polling.base import BasePoller, get_poller, register_poller
from kb.polling.cursors import (
    advance_cursor,
    load_cursor,
    stamp_error,
)
from kb.polling.scheduler import PollScheduler

__all__ = [
    "BasePoller",
    "PollScheduler",
    "advance_cursor",
    "get_poller",
    "load_cursor",
    "register_poller",
    "stamp_error",
]
