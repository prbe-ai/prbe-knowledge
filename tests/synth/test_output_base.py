"""SynthDoc construction and Storage protocol satisfaction."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import pytest

from scripts.synth.archetypes.base import Source
from scripts.synth.output.base import Storage, SynthDoc


def _make_doc() -> SynthDoc:
    return SynthDoc(
        id="scn-standup-gh-alice-2026-05-01-slack-0",
        source=Source.SLACK,
        source_event_id="scn-standup-gh-alice-2026-05-01-slack-0",
        text="Yesterday: shipped payments. Today: auth-service - fix token refresh. Blockers: none.",
        occurred_at=datetime(2026, 5, 1, 9, 0, 0, tzinfo=UTC),
        channel="#standup",
        page_id=None,
        thread_parent_id=None,
        scenario_id="scn-standup-gh-alice-2026-05-01",
        archetype="STANDUP_UPDATE",
        personas=("gh:alice",),
        services_mentioned=("payments", "auth-service"),
        priority=100,
    )


def test_synthdoc_construct() -> None:
    doc = _make_doc()
    assert doc.source == Source.SLACK
    assert doc.priority == 100


def test_synthdoc_frozen() -> None:
    doc = _make_doc()
    with pytest.raises(FrozenInstanceError):
        doc.text = "mutated"


def test_storage_protocol_satisfied_by_stub() -> None:
    """A plain class with the right methods satisfies the Storage protocol at runtime."""

    class _FakeStore:
        async def put(self, bucket: str, key: str, data: bytes) -> None:
            pass

        async def list_keys(self, bucket: str, prefix: str) -> list[str]:
            return []

        async def delete(self, bucket: str, key: str) -> None:
            pass

        async def bucket_for(self, customer_id: str) -> str:
            return f"bucket-{customer_id}"

        async def ensure_bucket(self, bucket: str) -> None:
            pass

    store = _FakeStore()
    # Verify it satisfies the protocol structurally (runtime_checkable)
    assert isinstance(store, Storage)
