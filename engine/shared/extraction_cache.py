"""Reuse a segment's extraction when a session re-ends with that segment unchanged.

WHY. A coding-agent session is mined when it ends: one model call per segment
of its transcript, plus one over its decisions. When a session is resumed and
ended again, every segment used to be mined again, including the ones nothing
changed. On research, 2026-09-24, 159 of 322 segment calls (49 %) were such
repeats (docs/plans/extraction-spend-plan.md §5, §16).

WHAT IS CACHED: the model's ANSWER (tool-call arguments), never the units built
from it. A hit runs the same construction and grounding as a fresh answer, over
the current transcript, so a change to either code path applies to cached
answers too, and an answer that no longer builds is simply mined again.

    key         = sha256(sha256(segment text) : fingerprint)
    fingerprint = sha256(everything else the model was asked: model, revision,
                  system prompt, user template, tool name/description/schema,
                  max_tokens, cwd)

Anything that changes the question changes the key. The one change this process
cannot see is a gateway alias repointed to a different model; bump
`claude_code_extraction_cache_revision` for that.

WHERE. raw/<source>/<customer>/<session>/extraction-cache/<key>.json in the
tenant's bucket. The source purge deletes raw/<source>/<customer>/, so the
cache goes with the tenant's data. No migration and no eviction: an answer is a
few KB, and there is at most one per segment the session ever had.

NEVER A REASON A PASS FAILS OR STALLS. A dedicated store client (2 s connect,
3 s read, one attempt), a deadline over each operation, and the first storage
error turns the cache off for the rest of the pass. Every failure is a miss,
and a miss calls the model exactly as before.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from engine.shared.config import get_settings
from engine.shared.exceptions import StorageNotFound
from engine.shared.logging import get_logger
from engine.shared.storage import ObjectStore

log = get_logger(__name__)

#: Bumped only when the stored body's SHAPE changes. What the model was asked is
#: the fingerprint's job, not this.
BODY_VERSION = 1

_CONNECT_TIMEOUT_S = 2.0
_READ_TIMEOUT_S = 3.0
#: Over the client's own deadlines: the store runs boto3 in a thread, and this
#: bounds the pass even if a call ignores them.
_OP_DEADLINE_S = 6.0

_store: ObjectStore | None = None


def _cache_store() -> ObjectStore:
    global _store
    if _store is None:
        _store = ObjectStore(
            connect_timeout=_CONNECT_TIMEOUT_S,
            read_timeout=_READ_TIMEOUT_S,
            total_max_attempts=1,
        )
    return _store


def reset_store_for_tests() -> None:
    global _store
    _store = None


def fingerprint(**question: Any) -> str:
    """sha256 over everything the model was asked except the segment itself."""
    canonical = json.dumps(
        question, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def content_key(text: str, fp: str) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return hashlib.sha256(f"{text_hash}:{fp}".encode()).hexdigest()


class SegmentCache:
    """One mining pass's handle on one session's cached answers."""

    def __init__(self, store: Any, bucket: str, prefix: str) -> None:
        self._store = store
        self._bucket = bucket
        self._prefix = prefix
        #: Cleared by the first storage error: one slow or failing store costs a
        #: pass one deadline, not one per segment.
        self.enabled = True

    @classmethod
    async def for_session(
        cls, *, customer_id: str, source: str, session_id: str
    ) -> SegmentCache | None:
        """None when the cache is switched off or the tenant's bucket is unknown."""
        if not get_settings().claude_code_extraction_segment_cache:
            return None
        try:
            store = _cache_store()
            bucket = await asyncio.wait_for(store.bucket_for(customer_id), _OP_DEADLINE_S)
        except Exception as exc:  # the cache is optional, always
            log.warning(
                "extraction_cache.unavailable",
                customer=customer_id,
                error=type(exc).__name__,
            )
            return None
        return cls(store, bucket, f"raw/{source}/{customer_id}/{session_id}/extraction-cache/")

    def _path(self, key: str) -> str:
        return f"{self._prefix}{key}.json"

    async def load(self, key: str, fp: str) -> Any | None:
        """The cached answer, or None for any reason at all."""
        if not self.enabled:
            return None
        try:
            raw = await asyncio.wait_for(
                self._store.get(self._bucket, self._path(key)), _OP_DEADLINE_S
            )
        except StorageNotFound:
            return None
        except Exception as exc:
            self._trip("load", exc)
            return None
        try:
            body = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return None
        if (
            not isinstance(body, dict)
            or body.get("v") != BODY_VERSION
            or body.get("fingerprint") != fp
        ):
            return None
        return body.get("answer")

    async def save(self, key: str, fp: str, answer: Any) -> None:
        """Store an answer that has already been built and grounded successfully."""
        if not self.enabled:
            return
        body = json.dumps(
            {
                "v": BODY_VERSION,
                "fingerprint": fp,
                "answer": answer,
                "created_at": datetime.now(UTC).isoformat(),
            },
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
        try:
            await asyncio.wait_for(
                self._store.put(self._bucket, self._path(key), body), _OP_DEADLINE_S
            )
        except Exception as exc:
            self._trip("save", exc)

    def _trip(self, op: str, exc: BaseException) -> None:
        self.enabled = False
        log.warning(
            "extraction_cache.off_for_pass",
            op=op,
            error=type(exc).__name__,
        )
