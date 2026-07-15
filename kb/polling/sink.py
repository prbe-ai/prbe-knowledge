"""Real document sink for the polling scheduler.

Wires :class:`kb.polling.scheduler.PollScheduler` into the
existing webhook-ingestion path so polled documents land in the same
``ingestion_queue`` rows the inbound-webhook HTTP handlers produce. The
downstream normalizer treats poll-sourced and webhook-sourced events
identically because the envelope shape on disk is identical:

    {
        "_headers": {},          # poll path has no inbound headers
        "payload": <doc>,        # the dict each poller emits
        "received_at": "<iso>",
        "trace_id": "poll-<tick-uuid>",
    }

Each polled document is uploaded as one R2 object (keyed by
``_payload_key(source, customer_id, source_event_id)`` — the same helper
the FastAPI webhook handler uses, kept in sync) and one row is INSERTed
into ``ingestion_queue`` via :func:`kb.ingestion_app._enqueue`.

The sink lives here (not in :mod:`kb.ingestion_app`) because:

1. The worker process owns the polling loop; pulling main.py's FastAPI
   app into the worker would drag in route registration we don't need.
2. The sink contract (source-aware) is specific to the polling path —
   keeping it module-local makes the boundary explicit.

Source-event-id resolution per polled document, in order:

1. ``doc["source_event_id"]`` if the poller already set it (GitHub does).
2. A deterministic SHA-256 fingerprint of ``orjson.dumps(doc)`` —
   "<source>:poll:<hash[:32]>" — so re-polling the same payload UPSERTs
   onto the same queue row (the existing webhook-handler dedupe behaviour).
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

import orjson

from engine.shared.constants import SourceSystem
from engine.shared.logging import get_logger
from engine.shared.storage import ObjectStore, get_store
from kb.ingestion_app import _enqueue, _payload_key

log = get_logger(__name__)


class PollDocumentSink:
    """Production sink for :class:`PollScheduler`.

    Construct once per worker process; ``__call__`` is invoked by the
    scheduler each time a poller emits a non-empty document list.
    """

    def __init__(self, store: ObjectStore | None = None) -> None:
        # Default to the process-wide store singleton so the sink shares
        # the worker's R2 client + bucket cache. Tests inject a stub.
        self._store = store or get_store()

    async def __call__(
        self,
        customer_id: str,
        source: SourceSystem,
        documents: list[dict[str, Any]],
    ) -> None:
        if not documents:
            return

        # One trace-id per scheduler tick *batch* so all docs from the same
        # poll() call share a thread in the logs; individual events still
        # carry their own source_event_id.
        trace_id = f"poll-{uuid.uuid4().hex[:12]}"
        bucket = await self._store.bucket_for(customer_id)
        await self._store.ensure_bucket(bucket)

        accepted = 0
        for doc in documents:
            source_event_id = _resolve_source_event_id(source, doc)
            received_at_iso = _resolve_received_at(doc)
            envelope = orjson.dumps(
                {
                    "_headers": {},  # poll path has no inbound HTTP headers
                    "payload": doc,
                    "received_at": received_at_iso,
                    "trace_id": trace_id,
                }
            )
            key = _payload_key(source, customer_id, source_event_id)
            try:
                await self._store.put(bucket, key, envelope)
            except Exception:
                log.exception(
                    "polling.sink.r2_put_failed",
                    customer=customer_id,
                    source=source.value,
                    source_event_id=source_event_id,
                    trace_id=trace_id,
                )
                continue

            try:
                inserted = await _enqueue(
                    customer_id=customer_id,
                    source=source,
                    source_event_id=source_event_id,
                    payload_s3_key=key,
                )
            except Exception:
                log.exception(
                    "polling.sink.enqueue_failed",
                    customer=customer_id,
                    source=source.value,
                    source_event_id=source_event_id,
                    trace_id=trace_id,
                )
                continue

            if inserted:
                accepted += 1

        log.info(
            "polling.sink.flushed",
            customer=customer_id,
            source=source.value,
            documents=len(documents),
            accepted=accepted,
            trace_id=trace_id,
        )


def _resolve_source_event_id(source: SourceSystem, doc: dict[str, Any]) -> str:
    """Stable per-document id used for dedupe at the queue layer.

    Pollers that already set ``source_event_id`` (GitHub) get used verbatim;
    the rest get a deterministic fingerprint so re-polling the same
    upstream row collapses into the same queue row via the existing
    ``ON CONFLICT (customer_id, source_system, source_event_id) DO NOTHING``
    behaviour in :func:`_enqueue`.
    """
    existing = doc.get("source_event_id")
    if isinstance(existing, str) and existing:
        return existing
    canonical = orjson.dumps(doc, option=orjson.OPT_SORT_KEYS)
    digest = hashlib.sha256(canonical).hexdigest()[:32]
    return f"{source.value}:poll:{digest}"


def _resolve_received_at(doc: dict[str, Any]) -> str:
    """ISO timestamp for the envelope's ``received_at``.

    Pollers that surface an upstream timestamp (``updated_at``,
    ``last_edited_time``, ``dateCreated``, ``ts``, ``received_at``) get
    it threaded through; otherwise we stamp now-UTC so the normalizer
    still has a valid clock.
    """
    for key in ("received_at", "updated_at", "last_edited_time", "dateCreated", "ts"):
        value = doc.get(key)
        if isinstance(value, str) and value:
            return value
    return datetime.now(UTC).isoformat()


__all__ = ["PollDocumentSink"]
