"""The per-session `trajectory.json` in R2.

WHERE: the session's own folder, `raw/<source>/<customer>/<session>/`, beside
the idle sweep's marker and the extraction cache. Everything that removes a
session's objects already removes that folder's own keys
(engine/shared/session_suppression.is_own_folder_key): session deletion, the
worker's late sweep of a session deleted mid-pass, and the tenant purge.

WHEN: on a COMPLETING pass only. Writing it is not free: it goes through the
same credential scrub as everything else persisted (0.4-2.5 s per MB,
measured), and a live session is re-rendered on every batch. A live session's
readers fall back to its text until it ends.

It is derived data: rebuilt from the session's batches on every completing
pass, overwritten in place, never pinned by a receipt.
"""

from __future__ import annotations

import asyncio
from typing import Any

import orjson

from engine.ingest.atif.models import Trajectory
from engine.ingest.payload_redaction import redact_payload_async
from engine.shared.logging import get_logger
from engine.shared.session_suppression import TRAJECTORY_FILE, session_folder
from engine.shared.storage import StorageNotFound

log = get_logger(__name__)


def trajectory_key(source: str, customer_id: str, session_id: str) -> str:
    return f"{session_folder(source, customer_id, session_id)}{TRAJECTORY_FILE}"


def _validate(trajectory: dict[str, Any]) -> str | None:
    """The first validation error, or None. Never raises."""
    try:
        Trajectory.model_validate(trajectory)
    except Exception as exc:  # pydantic's ValidationError and anything odder
        return str(exc).splitlines()[0][:500] if str(exc) else type(exc).__name__
    return None


class TrajectoryTooLarge(Exception):
    def __init__(self, size: int, limit: int) -> None:
        super().__init__(f"trajectory is {size} bytes; limit {limit}")
        self.size, self.limit = size, limit


async def write_trajectory(
    store: Any,
    bucket: str,
    key: str,
    trajectory: dict[str, Any],
    *,
    max_bytes: int | None = None,
) -> tuple[int, str | None]:
    """Scrub, validate, write. Returns (bytes written, validation error or None).

    An invalid document is still written, with the error in
    `extra.validation_error`: a reader that gets something is better served
    than one that gets a 404, and the count of invalid ones is the signal.
    Raises TrajectoryTooLarge, before scrubbing anything, past `max_bytes`.
    """
    if max_bytes is not None:
        size = len(orjson.dumps(trajectory))
        if size > max_bytes:
            raise TrajectoryTooLarge(size, max_bytes)
    clean = await redact_payload_async(trajectory)
    error = await asyncio.to_thread(_validate, clean)
    if error is not None:
        clean.setdefault("extra", {})["validation_error"] = error
    body = orjson.dumps(clean)
    await store.put(bucket, key, body, content_type="application/json")
    return len(body), error


async def read_trajectory(store: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        body = await store.get(bucket, key)
    except StorageNotFound:
        return None
    value = orjson.loads(body)
    return value if isinstance(value, dict) else None
