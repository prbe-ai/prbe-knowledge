"""Synthesize stage — Sonnet call.

One call per (target wiki page, cluster of triaged events). Inputs are the
current page body (if any) plus the FULL body of every clustered event. The
output (`SynthesisOutput`) is converted into a synthetic `WebhookEvent` and
persisted via Phase 1's `build_normalization_result` + `Normalizer._persist`,
so the cron writes wiki pages through the exact same path the manual upload
route uses.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from anthropic import AsyncAnthropic

from services.ingestion.handlers.wiki import (
    WIKI_PAYLOAD_KEY,
    build_normalization_result,
)
from services.synthesis.models import SynthesisInput, SynthesisOutput
from services.synthesis.prompts import build_synthesis_prompt, synthesis_tool_name
from shared.constants import (
    SONNET_MODEL,
    CompileTrigger,
    DocClass,
    SourceSystem,
)
from shared.logging import get_logger
from shared.models import NormalizationResult, WebhookEvent

log = get_logger(__name__)


class SynthesisParseError(RuntimeError):
    """Sonnet returned a tool_use block we couldn't parse into SynthesisOutput."""


async def call_synthesize(
    client: AsyncAnthropic,
    cluster: SynthesisInput,
    *,
    now: datetime,
) -> SynthesisOutput:
    kwargs = build_synthesis_prompt(cluster, now=now)
    resp = await client.messages.create(model=SONNET_MODEL, **kwargs)
    payload = _extract_tool_input(resp.content, expected_name=synthesis_tool_name())
    try:
        return SynthesisOutput(**payload)
    except Exception as exc:
        raise SynthesisParseError(f"synthesis tool input failed validation: {exc}") from exc


def render_synthesis_to_event(
    customer_id: str,
    cluster: SynthesisInput,
    output: SynthesisOutput,
    *,
    run_id: int,
    compile_trigger: CompileTrigger = CompileTrigger.SCHEDULED,
    received_at: datetime | None = None,
) -> WebhookEvent:
    """Build the synthetic WebhookEvent that drives `build_normalization_result`.

    The cron calls this per cluster. `compile_trigger` defaults to SCHEDULED;
    pass SOURCE_UPDATE when the run was wake-driven, MANUAL for human-kicked.
    """
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


def _extract_tool_input(blocks: list[Any], *, expected_name: str) -> dict[str, Any]:
    for block in blocks:
        if getattr(block, "type", "") == "tool_use" and getattr(block, "name", "") == expected_name:
            payload = getattr(block, "input", None)
            if isinstance(payload, dict):
                return payload
            raise SynthesisParseError(f"tool_use input was not a dict: {type(payload).__name__}")
    raise SynthesisParseError(f"sonnet response had no {expected_name} tool_use block")
