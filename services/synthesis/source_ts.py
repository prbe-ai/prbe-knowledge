"""Source-event timestamp extraction for the wiki agent's day-in-order read.

The wiki agent reads triaged events ordered by `source_ts ASC` so a 09:00
Slack flap, the 09:25 GitHub revert PR, and the 14:00 Notion postmortem
that resolves them all flow into the same wiki page edit. The Normalizer
populates `wiki_synthesis_queue.source_ts` at insert by dispatching here.

Each connector exposes the source timestamp under a different key:

    slack       message.ts (epoch seconds string, e.g. '1717000000.123456')
    github      created_at / updated_at (RFC3339 string)
    linear      updatedAt (RFC3339 string; falls back to createdAt)
    granola     startedAt (RFC3339 string)
    notion      last_edited_time (RFC3339 string)
    claude_code session start (parsed in handler), fall back to
                documents.created_at if absent
    codex       same shape as claude_code
    manual_upload uploaded_at; fall back to documents.created_at

Anything missing or malformed falls back to `doc.created_at`. That's a
backstop, not a normal path — connectors should populate the metadata
key consistently.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from shared.constants import SourceSystem
from shared.logging import get_logger

log = get_logger(__name__)

__all__ = ["extract_source_ts"]


def extract_source_ts(doc: Any) -> datetime:
    """Return the best-effort source-side timestamp for `doc`.

    `doc` is the persisted shape from `services.ingestion.normalizer` — a
    pydantic Document model with `source_system`, `metadata`, and
    `created_at`. Falls back to `created_at` for any source we don't
    recognize, the metadata is missing the expected key, or the value
    can't be parsed.
    """
    source_system = _coerce_source_system(getattr(doc, "source_system", None))
    metadata = _coerce_mapping(getattr(doc, "metadata", None))
    fallback = _coerce_dt(getattr(doc, "created_at", None))

    parsed: datetime | None = None
    if source_system is SourceSystem.SLACK:
        parsed = _parse_slack(metadata)
    elif source_system is SourceSystem.GITHUB:
        parsed = _parse_iso(metadata.get("created_at") or metadata.get("updated_at"))
    elif source_system is SourceSystem.LINEAR:
        parsed = _parse_iso(metadata.get("updatedAt") or metadata.get("createdAt"))
    elif source_system is SourceSystem.GRANOLA:
        parsed = _parse_iso(metadata.get("startedAt") or metadata.get("started_at"))
    elif source_system is SourceSystem.NOTION:
        parsed = _parse_iso(
            metadata.get("last_edited_time") or metadata.get("last_edited_at")
        )

    if parsed is not None:
        return parsed
    if fallback is not None:
        return fallback
    # Nothing parseable. Stamp NOW() so the column stays NOT NULL.
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Per-source parsers
# ---------------------------------------------------------------------------


def _parse_slack(metadata: Mapping[str, Any]) -> datetime | None:
    """Parse Slack's `ts` field. Format: '1717000000.123456' (epoch seconds)."""
    ts = metadata.get("ts") or metadata.get("event_ts") or metadata.get("thread_ts")
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(float(ts), tz=UTC)
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(ts, str):
        return None
    try:
        return datetime.fromtimestamp(float(ts), tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_iso(value: Any) -> datetime | None:
    """Parse an RFC3339/ISO8601 string into a tz-aware UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if not isinstance(value, str):
        return None
    raw = value.strip()
    if not raw:
        return None
    # datetime.fromisoformat accepts the common 'Z' suffix only on 3.11+.
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# ---------------------------------------------------------------------------
# Coercion helpers
# ---------------------------------------------------------------------------


def _coerce_source_system(raw: Any) -> SourceSystem | None:
    if isinstance(raw, SourceSystem):
        return raw
    if isinstance(raw, str):
        try:
            return SourceSystem(raw)
        except ValueError:
            return None
    return None


def _coerce_mapping(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    return {}


def _coerce_dt(raw: Any) -> datetime | None:
    if isinstance(raw, datetime):
        return raw if raw.tzinfo is not None else raw.replace(tzinfo=UTC)
    return None
