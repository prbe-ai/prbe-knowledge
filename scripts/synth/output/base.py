"""SynthDoc — the canonical output unit — and the Storage protocol.

SynthDoc is what every archetype builder ultimately produces (via DocSpec
materialization in ScenarioRunner). Source wrappers consume SynthDoc and
emit bytes. IngestionWriter consumes SynthDoc + bytes and writes files/R2.

Storage is a structural protocol matching shared.storage.ObjectStore
(after Task 13 adds the delete method). Tests can pass a fake stub;
production passes the real ObjectStore instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Protocol, runtime_checkable

from scripts.synth.archetypes.base import Source


@dataclass(frozen=True)
class SynthDoc:
    """One synthetic document ready for wrapping and writing."""
    id: str
    source: Source
    source_event_id: str
    text: str
    occurred_at: datetime
    channel: str | None          # Slack channel (e.g. "#standup"); None for Notion
    page_id: str | None          # Notion page id; None for Slack
    thread_parent_id: str | None # Slack thread parent source_event_id; None for root
    scenario_id: str
    archetype: str
    personas: tuple[str, ...]
    services_mentioned: tuple[str, ...]
    priority: int = field(default=100)


@runtime_checkable
class Storage(Protocol):
    """Structural protocol for object storage.

    Matches shared.storage.ObjectStore after Task 13 adds delete().
    Any class implementing these five methods satisfies the protocol.
    """

    async def put(self, bucket: str, key: str, data: bytes) -> None:
        ...

    async def list_keys(self, bucket: str, prefix: str) -> list[str]:
        ...

    async def delete(self, bucket: str, key: str) -> None:
        ...

    async def bucket_for(self, customer_id: str) -> str:
        ...

    async def ensure_bucket(self, bucket: str) -> None:
        ...
