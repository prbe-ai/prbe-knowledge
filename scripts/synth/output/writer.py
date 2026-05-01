"""IngestionWriter — writes SynthDocs to local files. Integrate mode (R2 +
ingestion_queue) is added in Task 14 by extending this class.

In both modes, every write produces a local file under <out_dir>/raw/<source>/
for human inspection. Integrate mode additionally calls bucket.put and
batches inserts into ingestion_queue.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from scripts.synth.archetypes.base import Source
from scripts.synth.output import notion as notion_wrapper
from scripts.synth.output import slack as slack_wrapper
from scripts.synth.output.base import SynthDoc


class IngestionWriter:
    """Plan 2 local-only writer. Task 14 extends with integrate mode."""

    def __init__(self, *, out_dir: Path, mode: Literal["local"] = "local") -> None:
        self.out_dir = out_dir
        self.mode = mode

    async def write(self, doc: SynthDoc) -> None:
        envelope = self._envelope(doc)
        path = self.out_dir / "raw" / doc.source.value / f"{doc.source_event_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(envelope)

    async def close(self) -> None:
        # Local mode has nothing to flush. Integrate mode (Task 14) overrides.
        return None

    def _envelope(self, doc: SynthDoc) -> bytes:
        if doc.source == Source.SLACK:
            return slack_wrapper.wrap(doc)
        if doc.source == Source.NOTION:
            return notion_wrapper.wrap(doc)
        raise ValueError(
            f"Plan 2 doesn't support source: {doc.source.value}. "
            "GitHub/Linear/Sentry/Granola wrappers land in Plan 3."
        )
