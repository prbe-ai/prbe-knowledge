"""Tests for IngestionWriter local mode (Plan 2 Task 11).

Integrate-mode tests are added in Task 14.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import orjson
import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import SynthDoc
from scripts.synth.output.writer import IngestionWriter


def _slack_doc(source_event_id: str = "doc-1") -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=Source.SLACK,
        source_event_id=source_event_id,
        text="hello",
        occurred_at=datetime(2026, 5, 1, 9, 0, tzinfo=UTC),
        channel="#standup",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-x",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments",),
    )


def _notion_doc(source_event_id: str = "page-1") -> SynthDoc:
    return SynthDoc(
        id=source_event_id,
        source=Source.NOTION,
        source_event_id=source_event_id,
        text="On-call handoff page body",
        occurred_at=datetime(2026, 5, 4, 10, 0, tzinfo=UTC),
        channel=None,
        page_id=source_event_id,
        thread_parent_id=None,
        scenario_id="scn-y",
        archetype="ON_CALL_HANDOFF",
        personas=("gh:alice", "gh:bob"),
        services_mentioned=("payments",),
    )


@pytest.mark.asyncio
async def test_local_writes_slack_envelope_to_disk(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_slack_doc("doc-1"))
    await writer.close()
    path = tmp_path / "raw" / "slack" / "doc-1.json"
    assert path.exists()
    payload = orjson.loads(path.read_bytes())
    assert payload["type"] == "event_callback"


@pytest.mark.asyncio
async def test_local_writes_notion_envelope_to_disk(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_notion_doc("page-1"))
    await writer.close()
    path = tmp_path / "raw" / "notion" / "page-1.json"
    assert path.exists()
    payload = orjson.loads(path.read_bytes())
    assert payload["type"] == "page.updated"


@pytest.mark.asyncio
async def test_local_overwrite_on_repeat_write(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    await writer.write(_slack_doc("doc-1"))
    await writer.write(_slack_doc("doc-1"))  # second write must not raise
    path = tmp_path / "raw" / "slack" / "doc-1.json"
    assert path.exists()


@pytest.mark.asyncio
async def test_local_unsupported_source_raises(tmp_path: Path) -> None:
    writer = IngestionWriter(out_dir=tmp_path)
    doc = SynthDoc(
        id="x",
        source=Source.GITHUB,  # GitHub wrapper deferred to Plan 3
        source_event_id="x",
        text="",
        occurred_at=datetime(2026, 5, 1, tzinfo=UTC),
        channel=None,
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-z",
        archetype="STANDUP_UPDATE",
        personas=(),
        services_mentioned=(),
    )
    with pytest.raises(ValueError, match="Plan 2 doesn't support source"):
        await writer.write(doc)
