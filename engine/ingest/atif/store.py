"""The per-session `trajectory.json` in R2.

WHERE: the session's own folder, `raw/<source>/<customer>/<session>/`, beside
the idle sweep's marker and the extraction cache. Everything that removes a
session's objects already removes that folder's own keys
(engine/shared/session_suppression.is_own_folder_key): session deletion, the
worker's late sweep of a session deleted mid-pass, and the tenant purge.

WHEN: on a COMPLETING pass only, and deleted again when the session resumes or a
later completing pass cannot replace it, so a reader never gets an old one.
Writing it is not free: it goes through the credential scrub (0.4-2.5 s per MB
measured), off the event loop as every large scrub is.

WHAT: the ATIF document without the engine's render provenance (`extra.probe`,
each step's `extra.probe_parts`), which only the ingest pass uses, in memory.
Free text -- messages, reasoning, tool-call summaries -- is scrubbed the way the
index is (a finding costs its line, never the field); everything else the way
uploaded payloads are. It is derived data, rebuilt from the session's batches
on every completing pass, overwritten in place, never pinned by a receipt.
"""

from __future__ import annotations

import asyncio
from typing import Any

import orjson

from engine.ingest.atif.models import Trajectory
from engine.ingest.payload_redaction import redact_payload_offloaded, redact_texts_async
from engine.shared.logging import get_logger
from engine.shared.session_suppression import TRAJECTORY_FILE, session_folder
from engine.shared.storage import StorageNotFound

log = get_logger(__name__)

#: Stands in for a free-text field while the rest of the document is scrubbed.
_FREE_TEXT_PREFIX = "probe-atif-free-text:"


def trajectory_key(source: str, customer_id: str, session_id: str) -> str:
    return f"{session_folder(source, customer_id, session_id)}{TRAJECTORY_FILE}"


def strip_provenance(trajectory: dict[str, Any]) -> dict[str, Any]:
    """The ATIF document a reader gets: `extra.probe` and every step's
    `extra.probe_parts` removed (engine/ingest/atif/build.py)."""
    out = dict(trajectory)
    extra = {k: v for k, v in (out.get("extra") or {}).items() if k != "probe"}
    if extra:
        out["extra"] = extra
    else:
        out.pop("extra", None)
    steps = []
    for step in out.get("steps") or []:
        step = dict(step)
        step_extra = {k: v for k, v in (step.get("extra") or {}).items() if k != "probe_parts"}
        if step_extra:
            step["extra"] = step_extra
        else:
            step.pop("extra", None)
        steps.append(step)
    out["steps"] = steps
    return out


#: Where a free-text field sits: (step index, field, part or call index).
_Slot = tuple[int, str, int | None]


def _free_text_slots(document: dict[str, Any]) -> list[_Slot]:
    slots: list[_Slot] = []
    for i, step in enumerate(document.get("steps") or []):
        message = step.get("message")
        if isinstance(message, str) and message:
            slots.append((i, "message", None))
        elif isinstance(message, list):
            for j, part in enumerate(message):
                if isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"]:
                    slots.append((i, "message", j))
        if isinstance(step.get("reasoning_content"), str) and step["reasoning_content"]:
            slots.append((i, "reasoning_content", None))
        for j, call in enumerate(step.get("tool_calls") or []):
            extra = call.get("extra")
            if isinstance(extra, dict) and isinstance(extra.get("summary"), str):
                slots.append((i, "summary", j))
    return slots


def _slot_get(document: dict[str, Any], slot: _Slot) -> Any:
    i, field, j = slot
    step = document["steps"][i]
    if field == "summary":
        return step["tool_calls"][j]["extra"]["summary"]
    if j is None:
        return step[field]
    return step[field][j]["text"]


def _slot_set(document: dict[str, Any], slot: _Slot, value: str) -> None:
    i, field, j = slot
    step = document["steps"][i]
    if field == "summary":
        step["tool_calls"][j]["extra"]["summary"] = value
    elif j is None:
        step[field] = value
    else:
        step[field][j]["text"] = value


async def scrub_trajectory(document: dict[str, Any]) -> dict[str, Any]:
    """Free text line by line (as the index is), the rest as uploaded payloads."""
    work = orjson.loads(orjson.dumps(document))
    slots = _free_text_slots(work)
    texts = [_slot_get(work, slot) for slot in slots]
    for n, slot in enumerate(slots):
        _slot_set(work, slot, f"{_FREE_TEXT_PREFIX}{n}")
    clean_texts = await redact_texts_async(texts) if texts else []
    structure = await redact_payload_offloaded(work, size=len(orjson.dumps(work)))
    for n, slot in enumerate(slots):
        try:
            intact = _slot_get(structure, slot) == f"{_FREE_TEXT_PREFIX}{n}"
        except (KeyError, IndexError, TypeError):
            intact = False
        if not intact:
            # The structural scrub moved or rewrote a stand-in: never store a
            # document whose text went missing without saying so.
            raise ValueError("free-text stand-ins were altered by the scrub")
        _slot_set(structure, slot, clean_texts[n])
    return structure


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
    """Strip, scrub, validate, write. Returns (bytes written, validation error or None).

    An invalid document is still written, with the error in
    `extra.validation_error`: a reader that gets something is better served
    than one that gets a 404, and the count of invalid ones is the signal.
    Raises TrajectoryTooLarge, before scrubbing anything, past `max_bytes`.
    """
    public = strip_provenance(trajectory)
    if max_bytes is not None:
        size = len(orjson.dumps(public))
        if size > max_bytes:
            raise TrajectoryTooLarge(size, max_bytes)
    clean = await scrub_trajectory(public)
    error = await asyncio.to_thread(_validate, clean)
    if error is not None:
        clean.setdefault("extra", {})["validation_error"] = error
    body = orjson.dumps(clean)
    await store.put(bucket, key, body, content_type="application/json")
    return len(body), error


async def read_trajectory(store: Any, bucket: str, key: str) -> dict[str, Any] | None:
    """The stored document, or None when there is none or it cannot be read."""
    try:
        body = await store.get(bucket, key)
    except StorageNotFound:
        return None
    try:
        value = orjson.loads(body)
    except orjson.JSONDecodeError:
        log.warning("trajectory.unreadable", key=key, bytes=len(body))
        return None
    return value if isinstance(value, dict) else None
