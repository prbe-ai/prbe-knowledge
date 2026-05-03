"""CodeGraph connector — symbol-level structural ingestion of git repos.

Reads synthetic events written by `services/ingestion/code_graph/bridge.py`,
dispatches by `kind`:

  initial_backfill  — shallow-clone + walk + extract; slices into ~200-file
                      batches via Connector.backfill async-iter (Lane B).
  incremental       — fetch changed files via Contents API; diff-extract
                      against code_repo_state cache.
  disconnect        — soft-delete code.symbol Documents for the affected
                      repos; close graph_node_provenance for code_graph.

verify_signature is a no-op return-True. Events arrive only from the
internal bridge, which is invoked from already-authenticated source
connector code (e.g., handlers/github.py after HMAC-verified push).
There is no public webhook surface for code_graph.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any, ClassVar

from services.ingestion.handlers.base import Connector
from services.ingestion.handlers.registry import register_connector
from shared.constants import SourceSystem
from shared.exceptions import InvalidWebhookPayload, UnsupportedEventType
from shared.logging import get_logger
from shared.models import (
    IntegrationToken,
    NormalizationResult,
    WebhookEvent,
    WebhookParseResult,
)

log = get_logger(__name__)

KIND_INITIAL_BACKFILL = "initial_backfill"
KIND_INCREMENTAL = "incremental"
KIND_DISCONNECT = "disconnect"
_KNOWN_KINDS = frozenset({KIND_INITIAL_BACKFILL, KIND_INCREMENTAL, KIND_DISCONNECT})


@register_connector(SourceSystem.CODE_GRAPH)
class CodeGraphConnector(Connector):
    source_system: ClassVar[SourceSystem] = SourceSystem.CODE_GRAPH
    display_name: ClassVar[str] = "code-graph"

    # ---- 1. signature verification --------------------------------------

    def verify_signature(
        self,
        headers: Mapping[str, str],
        raw_body: bytes,
    ) -> bool:
        # Internal-only event source. The bridge is invoked from already-
        # authenticated handler code; no external HMAC to check.
        return True

    # ---- 2. event parsing -----------------------------------------------

    def parse_webhook_event(
        self,
        customer_id: str,
        headers: Mapping[str, str],
        raw_payload: Mapping[str, Any],
    ) -> WebhookParseResult | None:
        kind = raw_payload.get("kind")
        if kind not in _KNOWN_KINDS:
            raise InvalidWebhookPayload(
                f"unknown code_graph payload kind: {kind!r}"
            )
        source_event_id = _recompute_event_id(raw_payload)
        return WebhookParseResult(
            source_event_id=source_event_id,
            received_at=datetime.now(UTC),
        )

    # ---- 3. hydration ---------------------------------------------------

    async def fetch_supplementary(
        self,
        event: WebhookEvent,
        token: IntegrationToken | None,
    ) -> dict[str, Any]:
        # Both clone and Contents API fetch live in normalize() / backfill()
        # where slicing decisions can keep Phase B small. fetch_supplementary
        # isn't async-iter-shaped, so it's not the right seam for code-graph.
        return {}

    # ---- 4. normalization -----------------------------------------------

    async def normalize(
        self,
        event: WebhookEvent,
        hydrated: Mapping[str, Any],
    ) -> NormalizationResult:
        kind = event.raw_payload.get("kind")
        if kind == KIND_INITIAL_BACKFILL:
            # Lane B: shallow clone, async-iter walk, ~200 files per batch.
            raise UnsupportedEventType(
                "code_graph initial_backfill not yet implemented",
                source=SourceSystem.CODE_GRAPH.value,
            )
        if kind == KIND_INCREMENTAL:
            # Lane B: Contents API fetch + diff-extract.
            raise UnsupportedEventType(
                "code_graph incremental not yet implemented",
                source=SourceSystem.CODE_GRAPH.value,
            )
        if kind == KIND_DISCONNECT:
            # Lane D: soft-delete cascade for the affected repos.
            raise UnsupportedEventType(
                "code_graph disconnect not yet implemented",
                source=SourceSystem.CODE_GRAPH.value,
            )
        raise InvalidWebhookPayload(
            f"unknown code_graph payload kind: {kind!r}"
        )


# ---- helpers --------------------------------------------------------------


def _recompute_event_id(payload: Mapping[str, Any]) -> str:
    """Mirror the bridge's source_event_id construction from the payload.

    Keeping this here (instead of stamping the id into the payload) lets the
    parse step verify the bridge's id-derivation by recomputing it from the
    semantic fields — a payload that was tampered with mid-flight surfaces
    as a UNIQUE-constraint mismatch on the queue row.
    """
    kind = payload["kind"]
    if kind == KIND_DISCONNECT:
        repos = payload.get("repos") or []
        repos_label = "+".join(sorted(str(r) for r in repos))[:200]
        ts = payload.get("enqueued_at", "")
        return f"code_graph:disconnect:{repos_label}:{ts}"
    repo = payload.get("repo", "")
    sha = payload.get("sha", "")
    if kind == KIND_INITIAL_BACKFILL:
        return f"code_graph:backfill:{repo}:{sha}"
    return f"code_graph:incremental:{repo}:{sha}"
