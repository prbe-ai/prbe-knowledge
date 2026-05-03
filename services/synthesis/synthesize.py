"""Synthesize + verifier stages — provider-dispatched calls.

Synthesize: one call per (target wiki page, cluster of triaged events).
Inputs are the current page body (if any) plus the FULL body of every
clustered event. The output (`SynthesisOutput`) is converted into a
synthetic `WebhookEvent` and persisted via Phase 1's
`build_normalization_result` + `Normalizer._persist`, so the cron writes
wiki pages through the exact same path the manual upload route uses.

Verifier: one call per cluster, between triage and synthesize. The
verifier returns `kept_doc_ids[]`; empty → mark queue rows
`verifier_rejected` (no synthesize). Non-empty → filter the cluster to
the kept docs, then synthesize.

Both stages dispatch to the configured provider (Anthropic by default,
Gemini if env var flips). The Anthropic `client` arg is kept for call-
site compatibility; unused when provider is Gemini.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from anthropic import AsyncAnthropic

from services.ingestion.handlers.wiki import (
    WIKI_PAYLOAD_KEY,
    build_normalization_result,
)
from services.synthesis.models import (
    SynthesisInput,
    SynthesisOutput,
    VerifierInput,
    VerifierOutput,
)
from services.synthesis.providers import (
    SynthesisParseError,
    VerifierParseError,
    get_synthesis_provider,
    get_verifier_provider,
)
from shared.constants import (
    CompileTrigger,
    DocClass,
    SourceSystem,
)
from shared.logging import get_logger
from shared.models import NormalizationResult, WebhookEvent

__all__ = [
    "SynthesisParseError",
    "VerifierParseError",
    "call_synthesize",
    "call_verifier",
    "render_synthesis_to_event",
    "synthesis_to_normalization",
]

log = get_logger(__name__)


async def call_synthesize(
    client: AsyncAnthropic,
    cluster: SynthesisInput,
    *,
    now: datetime,
) -> SynthesisOutput:
    provider = get_synthesis_provider(client)
    return await provider.synthesize(cluster, now=now)


async def call_verifier(
    client: AsyncAnthropic,
    cluster: VerifierInput,
    *,
    now: datetime,
) -> VerifierOutput:
    """Run the cluster through the verifier provider.

    Empty `kept_doc_ids` means the cluster is rejected (no synthesize
    call). Non-empty means filter `cluster.events` to the kept doc_ids
    before invoking `call_synthesize`.
    """
    provider = get_verifier_provider(client)
    return await provider.verify(cluster, now=now)


def render_synthesis_to_event(
    customer_id: str,
    cluster: SynthesisInput,
    output: SynthesisOutput,
    *,
    run_id: int,
    compile_trigger: CompileTrigger = CompileTrigger.SCHEDULED,
    received_at: datetime | None = None,
) -> WebhookEvent:
    """Build the synthetic WebhookEvent that drives `build_normalization_result`."""
    received_at = received_at or datetime.now(UTC)
    raw_payload: dict[str, Any] = {
        WIKI_PAYLOAD_KEY: {
            "wiki_type": cluster.wiki_type,
            "slug": cluster.slug,
            "title": output.title,
            "body": output.body_markdown,
            "frontmatter": dict(output.frontmatter),
            "doc_class": DocClass.COMPILED_WIKI.value,
            "compiled_from_doc_ids": [event.doc_id for event in cluster.events],
            "compile_trigger": compile_trigger.value,
            "is_delete": False,
            "updated_at": received_at.isoformat(),
            "summary": output.summary,
            "commit_message": output.commit_message,
            "commit_author": "agent:wiki-synthesis-cron",
            "commit_run_id": run_id,
            "author_id": "agent:wiki-synthesis-cron",
        }
    }
    return WebhookEvent(
        customer_id=customer_id,
        source_system=SourceSystem.WIKI,
        source_event_id=(f"{cluster.wiki_type}:{cluster.slug}:edit:{received_at.isoformat()}"),
        received_at=received_at,
        payload_s3_key="",
        payload_s3_keys=[],
        raw_payload=raw_payload,
        headers={},
    )


def synthesis_to_normalization(
    customer_id: str,
    cluster: SynthesisInput,
    output: SynthesisOutput,
    *,
    run_id: int,
    compile_trigger: CompileTrigger = CompileTrigger.SCHEDULED,
    received_at: datetime | None = None,
) -> NormalizationResult:
    """Convenience: WebhookEvent + build_normalization_result in one call."""
    event = render_synthesis_to_event(
        customer_id,
        cluster,
        output,
        run_id=run_id,
        compile_trigger=compile_trigger,
        received_at=received_at,
    )
    return build_normalization_result(event)
