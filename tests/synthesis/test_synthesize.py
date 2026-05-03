"""Unit tests for the synthesize stage.

- call_synthesize: round-trip via mocked AsyncAnthropic.
- render_synthesis_to_event + synthesis_to_normalization: produce the
  expected synthetic WebhookEvent and Document with COMPILED_WIKI shape.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from services.synthesis.models import SynthesisInput, SynthesisOutput, TriageInput
from services.synthesis.synthesize import (
    SynthesisParseError,
    call_synthesize,
    render_synthesis_to_event,
    synthesis_to_normalization,
)
from shared.constants import CompileTrigger, DocClass, DocType, SourceSystem


def _event(qid: int, body: str = "Auth flow refactored.") -> TriageInput:
    return TriageInput(
        queue_id=qid,
        doc_id=f"github:commit:abc{qid}",
        doc_type="github.commit",
        source_system="github",
        title=f"Commit {qid}",
        author_id="alice",
        body=body,
        body_token_count=10,
    )


def _cluster(action: str = "update") -> SynthesisInput:
    return SynthesisInput(
        wiki_type="runbook",
        slug="auth-failures",
        action=action,
        current_title="Auth failures runbook",
        current_body="Existing body.",
        current_frontmatter={"owner": "alice"},
        current_summary="When auth fails, do this.",
        events=[_event(1), _event(2)],
    )


def _tool_use_response(payload: dict) -> SimpleNamespace:
    block = SimpleNamespace(type="tool_use", name="render_wiki_page", input=payload)
    return SimpleNamespace(content=[block])


# ---------------------------------------------------------------------------
# call_synthesize
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_call_synthesize_round_trip() -> None:
    payload = {
        "title": "Auth failures runbook",
        "body_markdown": "When auth fails, ping [[Person: alice]].",
        "summary": "Procedure when auth flows go down.",
        "frontmatter": {"owner": "alice", "severity": "high"},
        "commit_message": "Incorporate two recent commits about auth refactor.",
    }
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(return_value=_tool_use_response(payload)))
    out = await call_synthesize(client, _cluster(), now=datetime(2026, 5, 2, tzinfo=UTC))
    assert isinstance(out, SynthesisOutput)
    assert out.title == "Auth failures runbook"
    assert "alice" in out.body_markdown
    assert out.commit_message.startswith("Incorporate")
    assert out.frontmatter["severity"] == "high"


@pytest.mark.asyncio
async def test_call_synthesize_validates_required_fields() -> None:
    payload = {
        "title": "X",
        "body_markdown": "ok",
        # missing summary + commit_message
    }
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(return_value=_tool_use_response(payload)))
    with pytest.raises(SynthesisParseError):
        await call_synthesize(client, _cluster(), now=datetime(2026, 5, 2, tzinfo=UTC))


@pytest.mark.asyncio
async def test_call_synthesize_no_tool_block_raises() -> None:
    text_only = SimpleNamespace(content=[SimpleNamespace(type="text", text="ok")])
    client = SimpleNamespace()
    client.messages = SimpleNamespace(create=AsyncMock(return_value=text_only))
    with pytest.raises(SynthesisParseError):
        await call_synthesize(client, _cluster(), now=datetime(2026, 5, 2, tzinfo=UTC))


# ---------------------------------------------------------------------------
# render_synthesis_to_event + synthesis_to_normalization
# ---------------------------------------------------------------------------


def _output() -> SynthesisOutput:
    return SynthesisOutput(
        title="Auth failures runbook",
        body_markdown="When auth fails, ping [[Person: alice]].",
        summary="Procedure when auth fails.",
        frontmatter={"owner": "alice"},
        commit_message="Initial compilation from 2 events.",
    )


def test_render_synthesis_to_event_carries_cron_metadata() -> None:
    received = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    event = render_synthesis_to_event(
        "cust-1",
        _cluster(),
        _output(),
        run_id=42,
        compile_trigger=CompileTrigger.SOURCE_UPDATE,
        received_at=received,
    )
    payload = event.raw_payload["wiki_page"]
    assert payload["wiki_type"] == "runbook"
    assert payload["slug"] == "auth-failures"
    assert payload["doc_class"] == DocClass.COMPILED_WIKI.value
    assert payload["compile_trigger"] == CompileTrigger.SOURCE_UPDATE.value
    assert payload["compiled_from_doc_ids"] == [
        "github:commit:abc1",
        "github:commit:abc2",
    ]
    assert payload["commit_run_id"] == 42
    assert payload["commit_message"].startswith("Initial compilation")
    assert payload["summary"].startswith("Procedure")
    assert payload["author_id"] == "agent:wiki-synthesis-cron"


def test_synthesis_to_normalization_emits_compiled_wiki_doc() -> None:
    received = datetime(2026, 5, 2, 12, 0, tzinfo=UTC)
    norm = synthesis_to_normalization(
        "cust-1",
        _cluster(action="create"),
        _output(),
        run_id=42,
        received_at=received,
    )
    assert len(norm.documents) == 1
    doc = norm.documents[0]
    assert doc.doc_id == "wiki:runbook:auth-failures"
    assert doc.source_system == SourceSystem.WIKI
    assert doc.doc_type == DocType.WIKI_RUNBOOK
    assert doc.doc_class == DocClass.COMPILED_WIKI
    assert doc.compiled_from_doc_ids == [
        "github:commit:abc1",
        "github:commit:abc2",
    ]
    assert doc.compile_trigger == CompileTrigger.SCHEDULED
    assert doc.metadata["body"] == "When auth fails, ping [[Person: alice]]."
    assert doc.metadata["summary"] == "Procedure when auth fails."
    assert doc.metadata["commit"]["message"] == "Initial compilation from 2 events."
    assert doc.metadata["commit"]["run_id"] == 42
    assert doc.metadata["commit"]["author"] == "agent:wiki-synthesis-cron"
