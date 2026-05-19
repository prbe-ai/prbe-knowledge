"""Build the orchestrator dispatch `incident_signals` block from a
live INCIDENT document row.

The dispatch payload's `incident_signals` field is what the agent uses
to seed Phase 1 + Phase 2 prompts. The Documents table holds richer
metadata per source (PD has `service_id`/`priority`/`urgency`;
incident.io has `service_tag`/`severity`/`reference`). This helper
extracts a uniform shape per the orchestrator's `IncidentSignals`
schema.
"""
from __future__ import annotations

import json
from typing import Any


def extract_from_doc(doc_row: dict[str, Any]) -> dict[str, Any]:
    """Convert a row from the `documents` table (one INCIDENT doc) into
    the dispatch `incident_signals` block.

    Accepts both asyncpg Row-style mappings and plain dicts. The
    `metadata` field is the JSONB the connector stamped during normalize.
    asyncpg returns JSONB columns as raw JSON strings; this function
    decodes them transparently.
    """
    raw_metadata = doc_row.get("metadata") or {}
    if isinstance(raw_metadata, str):
        try:
            raw_metadata = json.loads(raw_metadata)
        except (ValueError, TypeError):
            raw_metadata = {}
    metadata: dict[str, Any] = raw_metadata
    source = doc_row.get("source_system") or ""
    title = doc_row.get("title") or ""

    created_at = doc_row.get("created_at")
    if hasattr(created_at, "isoformat"):
        triggered_at = created_at.isoformat()
    else:
        triggered_at = str(created_at or "")

    if source == "pagerduty":
        service = metadata.get("service_id")
        severity = metadata.get("priority")
        urgency = metadata.get("urgency")
    elif source == "incident_io":
        service = metadata.get("service_tag")
        severity = metadata.get("severity")
        urgency = None
    else:
        service = None
        severity = metadata.get("severity")
        urgency = None

    return {
        "title": title,
        "severity": severity,
        "urgency": urgency,
        "service": service,
        "triggered_at": triggered_at,
        "raw_summary": "",
        "tags": [],
    }
