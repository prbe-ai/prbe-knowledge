"""IngestionWriter — writes SynthDocs to local files in both modes, and
additionally to R2 + ingestion_queue when mode='integrate'.

In local mode (default), each `write` produces one local JSON file under
`<out_dir>/raw/<source>/`. In integrate mode, the same local file is still
written (for inspection), AND the envelope is pushed to R2 at
`raw/<source>/<customer_id>/synth/<id>.json`, AND a row is batched into
`ingestion_queue` (flushed at BATCH_SIZE or on close).

Schema notes (real prbe-knowledge schema, NOT the spec's drift):
- column is `source_system`, not `source`
- payload column is `payload_s3_keys TEXT[]`, not `raw_key TEXT`
- conflict key is (source_system, source_event_id)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from scripts.synth.archetypes.base import Source
from scripts.synth.output import notion as notion_wrapper
from scripts.synth.output import slack as slack_wrapper
from scripts.synth.output.base import SynthDoc

BATCH_SIZE = 50


class IngestionWriter:
    """Plan 2 writer with two modes.

    local mode: writes to <out_dir>/raw/<source>/<id>.json. No DB or R2.
    integrate mode: also writes to bucket and inserts into ingestion_queue.

    integrate mode requires a prior `synth init` to have created the customer
    + bucket + integration_tokens stub rows.
    """

    def __init__(
        self,
        *,
        out_dir: Path,
        mode: Literal["local", "integrate"] = "local",
        customer_id: str | None = None,
        bucket=None,
        db=None,
    ) -> None:
        self.out_dir = out_dir
        self.mode = mode
        self.customer_id = customer_id
        self.bucket = bucket
        self.db = db
        self._batch: list[tuple[SynthDoc, str]] = []
        if mode == "integrate" and (customer_id is None or bucket is None or db is None):
            raise ValueError(
                "integrate mode requires customer_id, bucket, and db arguments"
            )

    async def write(self, doc: SynthDoc) -> None:
        envelope = self._envelope(doc)

        # Always write local file (for inspection in both modes).
        local_path = self.out_dir / "raw" / doc.source.value / f"{doc.source_event_id}.json"
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_bytes(envelope)

        if self.mode == "local":
            return

        # integrate mode: R2 put + queue batching
        bucket_name = self.bucket.bucket_for(self.customer_id)
        key = f"raw/{doc.source.value}/{self.customer_id}/synth/{doc.source_event_id}.json"
        await self.bucket.put(bucket_name, key, envelope)
        self._batch.append((doc, key))
        if len(self._batch) >= BATCH_SIZE:
            await self._flush_queue()

    async def close(self) -> None:
        if self.mode == "integrate" and self._batch:
            await self._flush_queue()

    async def _flush_queue(self) -> None:
        """Batch-INSERT to ingestion_queue using the actual prbe-knowledge schema."""
        rows = [
            (
                self.customer_id,
                doc.source.value,         # source_system
                doc.source_event_id,
                [key],                     # payload_s3_keys: TEXT[]
                doc.priority,
                doc.occurred_at,
            )
            for doc, key in self._batch
        ]
        await self.db.executemany(
            """
            INSERT INTO ingestion_queue
              (customer_id, source_system, source_event_id, payload_s3_keys,
               status, priority, occurred_at, enqueued_at)
            VALUES ($1, $2, $3, $4, 'pending', $5, $6, NOW())
            ON CONFLICT (source_system, source_event_id) DO NOTHING
            """,
            rows,
        )
        self._batch.clear()

    def _envelope(self, doc: SynthDoc) -> bytes:
        if doc.source == Source.SLACK:
            return slack_wrapper.wrap(doc)
        if doc.source == Source.NOTION:
            return notion_wrapper.wrap(doc)
        raise ValueError(
            f"Plan 2 doesn't support source: {doc.source.value}. "
            "GitHub/Linear/Sentry/Granola wrappers land in Plan 3."
        )
