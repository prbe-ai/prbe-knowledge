"""Connector base class — the shared contract every source integration implements.

Adding a new connector (Stripe, Jira, PagerDuty, ...) is a three-step job:

    1. Add a value to `SourceSystem` (shared/constants.py) and supporting enums.
    2. Subclass `Connector` below and implement the abstract methods.
    3. Decorate the class with `@register_connector(SourceSystem.XXX)`.

The normalizer/worker treats all connectors identically:

        webhook POST
             │
             ▼
    verify_signature(headers, body)          ← connector
             │
             ▼
    parse_webhook_event(...) → WebhookParseResult | None
             │            (None = ignore this event type)
             ▼
    store raw payload in R2, enqueue queue row   ← generic
             │
             ▼
    worker picks row up
             │
             ▼
    fetch_supplementary(event, token)         ← connector (optional)
             │
             ▼
    normalize(event, hydrated) → NormalizationResult   ← connector
             │
             ▼
    persist Document[] + chunks + graph + ACL snapshots   ← generic

Only `verify_signature`, `parse_webhook_event`, and `normalize` are required.
`fetch_supplementary` has a no-op default; `backfill` raises NotSupportedByConnector.
`oauth_install_url` / `exchange_oauth_code` are optional too — connectors
without OAuth can omit them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import httpx

from shared.config import Settings, get_settings
from shared.constants import DEFAULT_INGESTION_PRIORITY, BackfillStatus, SourceSystem
from shared.exceptions import NotSupportedByConnector
from shared.models import (
    ExternalWorkspaceRef,
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)
from shared.source_registry import DEFAULT_DOC_TYPE_PREFIX, DEFAULT_SCORE_MULTIPLIER


@dataclass(frozen=True, slots=True)
class PollConfig:
    """Declares that a Connector needs periodic backfill re-enqueueing.

    Set as a ClassVar on the Connector subclass. None (the default) means
    webhook-only — no polling. IntegrationPoller picks up every Connector
    whose poll_config is not None.

    Fields:
      interval_seconds: cadence at which the integration poller wakes for
        this source. Tick fires when NOW() - last_progress_at > interval.
      eligible_statuses: backfill_state.status values to re-enqueue.
        Typically (COMPLETE, FAILED) — PENDING/RUNNING are skipped because
        they're already in-flight. Including FAILED auto-retries transient
        upstream errors (R2 5xx, network blips) on the next tick.
      notify_channel: pg_notify channel that wakes BackfillWorker
        immediately after the row flips to PENDING. Without this, the worker
        only picks up on its own poll cycle.
    """

    interval_seconds: int
    eligible_statuses: tuple[BackfillStatus, ...]
    notify_channel: str


@dataclass(slots=True)
class ConnectorContext:
    """Shared dependencies handed to every connector instance.

    Keeping this a small dataclass lets tests stub specific pieces without
    needing a full-process bootstrap. All connectors receive the same shape
    so behavior stays predictable.
    """

    settings: Settings
    http: httpx.AsyncClient
    # Any other per-process helpers (storage, embedder) live on the normalizer,
    # not on the connector, because connectors should stay pure transformers.


class Connector(ABC):
    """Abstract base for all source integrations.

    Subclasses set `source_system` as a class variable and implement the
    abstract methods. See `handlers/slack.py` for a worked example.
    """

    # Set by every subclass. The registry uses this to dispatch webhooks.
    source_system: ClassVar[SourceSystem]

    # Display name used in logs / error messages. Defaults to the enum value.
    display_name: ClassVar[str] = ""

    # ---- source profile (registered into shared.source_registry) ----------
    # Generic engine code (queue priority at enqueue, fusion decay, the
    # retrieval doc-type resolver) reads these through the source registry
    # instead of hardcoded per-source dicts. @register_connector registers
    # them; defaults below match the registry's unregistered-key fallbacks.
    #
    # Dotted prefix of this source's DocType family ("slack.", "github.", ...).
    doc_type_prefix: ClassVar[str] = DEFAULT_DOC_TYPE_PREFIX
    # Queue priority at enqueue time (worker claims priority DESC): 100
    # interactive webhooks, 75 bursty agent/custom batches, 50 backfill-tier.
    ingestion_priority: ClassVar[int] = DEFAULT_INGESTION_PRIORITY
    # Post-RRF doc-score multiplier (< 1.0 demotes at equal relevance).
    score_multiplier: ClassVar[float] = DEFAULT_SCORE_MULTIPLIER
    # Recency half-life override (days) for fusion decay; None = baseline.
    half_life_days: ClassVar[float | None] = None

    # Set on poll-only connectors (Granola etc.) — IntegrationPoller reads
    # this to know how often to re-enqueue stale backfills. None means the
    # source is webhook-driven and needs no periodic poll.
    poll_config: ClassVar[PollConfig | None] = None

    def __init__(self, ctx: ConnectorContext) -> None:
        self.ctx = ctx
        self.settings = ctx.settings
        self.http = ctx.http

    # ---- 1. signature verification ----------------------------------------

    @abstractmethod
    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        """Verify the webhook's HMAC/JWT/whatever-this-source-uses signature.

        Return True for valid, False otherwise. Never raise on malformed
        signatures — the caller turns False into a 401.
        """

    # ---- 2. event parsing --------------------------------------------------

    @abstractmethod
    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        """Extract the source-stable event id + received_at from the payload.

        Returning None means: this webhook is valid but not something we want
        to persist (e.g. a heartbeat, a `user_typing` event, a deletion we
        handle differently). The fast path returns 200 without enqueueing.

        Returning a WebhookParseResult with a `source_event_id` makes the
        webhook idempotent: the UNIQUE (customer_id, source_system, source_event_id)
        constraint on ingestion_queue deduplicates redeliveries.
        """

    # ---- 3. hydration (optional) -------------------------------------------

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        """Hit the source API for data the webhook payload alone lacks.

        Examples: a Slack `message` event only carries the message itself —
        fetch thread siblings here. A GitHub `pull_request` event lacks the
        file list — call `/pulls/{n}/files` here. Default is a no-op.

        Must return a plain dict. The normalize() step receives this dict
        as its `hydrated` argument.
        """
        return {}

    # ---- 4. normalization --------------------------------------------------

    @abstractmethod
    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        """Produce the canonical Document[] + graph + ACL from the raw payload.

        This is where connector-specific mapping lives (Slack 'message' →
        DocType.SLACK_MESSAGE, thread root → parent_doc_id, etc.).

        Should NEVER directly touch the database, R2, or embeddings. Return
        a pure NormalizationResult and let the normalizer persist it.
        """

    # ---- 5. backfill (optional) --------------------------------------------

    def backfill(
        self,
        customer_id: str,
        token: IntegrationToken,
        cursor: str | None = None,
    ) -> AsyncIterator[WebhookEvent]:
        """Stream historical events for paginated initial sync.

        Implementations are an async generator that yields synthetic
        WebhookEvents — the worker replays them through normalize()
        just like live webhooks. Update `backfill_state.last_cursor`
        after each page so the run is resumable.
        """
        raise NotSupportedByConnector(
            f"{self.source_system.value} connector does not implement backfill"
        )

    # ---- 6. OAuth install (optional) ---------------------------------------

    def oauth_install_url(self, customer_id: str, redirect_uri: str) -> str:
        """Build the user-facing OAuth install URL for this source.

        Default raises; subclasses override if the source uses OAuth.
        """
        raise NotSupportedByConnector(
            f"{self.source_system.value} connector does not implement OAuth install"
        )

    async def exchange_oauth_code(
        self,
        code: str | None,
        redirect_uri: str,
        extra_params: dict[str, str] | None = None,
    ) -> IntegrationToken:
        """Exchange an OAuth authorization code for an access token.

        `extra_params` carries the full callback query string so providers
        that don't fit the standard `code` → token dance (e.g. GitHub Apps,
        which redirect with `installation_id` + `setup_action` instead of
        `code`) can reach the rest of the query. Standard OAuth2 connectors
        can ignore it.

        Default raises; subclasses override if the source uses OAuth.
        """
        raise NotSupportedByConnector(
            f"{self.source_system.value} connector does not implement OAuth exchange"
        )

    # ---- 7. workspace identification (webhook → customer routing) ----------

    async def identify_workspaces(self, token: IntegrationToken) -> list[ExternalWorkspaceRef]:
        """Return source-side workspace/team/org ids tied to this token.

        Called once at OAuth-callback time. Each returned ref is written
        to `customer_source_mapping` so future webhooks can resolve
        customer_id from the payload alone.

        Default returns []. Connectors that support webhooks MUST override —
        otherwise their webhooks will 400 unless X-Prbe-Customer is set manually.
        """
        return []

    def extract_external_id_from_payload(
        self,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> str | None:
        """Pull the workspace/team/org id out of an incoming webhook payload.

        Paired with `identify_workspaces`: the id this returns must match one
        of the ids `identify_workspaces` recorded at install time. The webhook
        handler uses this to look up the owning customer.

        Default returns None. Connectors that support webhooks MUST override.
        """
        return None

    # ---- housekeeping ------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Connector {self.source_system.value}>"


def make_default_context() -> ConnectorContext:
    """Build a ConnectorContext using the module-level Settings + a shared HTTP client.

    The worker holds one of these for its whole lifetime. Tests build ad-hoc
    contexts with mocked http clients.
    """
    settings = get_settings()
    client = httpx.AsyncClient(timeout=settings.http_timeout_seconds)
    return ConnectorContext(settings=settings, http=client)


__all__ = ["Connector", "ConnectorContext", "PollConfig", "make_default_context"]
