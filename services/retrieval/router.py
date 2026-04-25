"""Query router — Haiku entity + temporal extraction.

Always calls Haiku (no DB cache). The output guides which indexes to hit
(if entity canonical_id matches a graph node, raise graph retriever weight)
and fans out the query into N expansions for BM25 recall. The raw query
always participates too so a bad expansion can't suppress a direct match.

Haiku also returns a symbolic `temporal` block when the query has time
scoping language ("last week", "since March", "after the auth refactor").
The caller resolves symbolic → absolute via `temporal.resolve_temporal()`
on every request, so relative phrases re-evaluate against `now`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from anthropic import APIError, AsyncAnthropic

from shared.config import get_settings
from shared.constants import HAIKU_MODEL
from shared.exceptions import RouterParseError, RouterTimeout
from shared.logging import get_logger

log = get_logger(__name__)

ROUTER_TIMEOUT_SECONDS = 5.0

_SYSTEM_PROMPT = """You are a retrieval router. Given a user query, extract structured entities,
propose 2-4 alternate phrasings, and capture any time scoping the query implies.

Return strict JSON:
{
  "entities": [
    {"entity_type": "service|repo|person|ticket|pr|error_group|feature|decision|file_path|channel",
     "canonical_id": "short-stable-id",
     "display_name": "human-readable",
     "confidence": 0.0-1.0}
  ],
  "expansions": ["phrasing 1", "phrasing 2"],
  "temporal": {
    "since": null | {"kind": "rel", "offset_days": -30} | {"kind": "abs", "iso": "2024-03-15T00:00:00Z"},
    "until": null | {"kind": "rel", "offset_days": 0}   | {"kind": "abs", "iso": "..."},
    "basis": "source",
    "raw_phrase": "in the last month",
    "unresolvable_anchor": null | "the auth refactor"
  } | null
}

Rules:
- Only extract entities you're confident are named concepts (not generic words).
- canonical_id should be the most likely stable identifier (service slug, repo name, user id, ticket code).
- Expansions should preserve intent but vary phrasing, synonyms, or level of specificity.
- Never invent facts. If no entities are clear, return an empty list.

Temporal rules:
- Resolve relative phrases (last week, yesterday, this month, last 30 days) to {"kind":"rel","offset_days":N}
  where N is negative for past, 0 for now.
- Absolute dates (since March 15, after 2024-Q1) to {"kind":"abs","iso":"YYYY-MM-DDTHH:MM:SSZ"} with UTC tz.
- "basis" is "source" unless the query explicitly says "ingested" or "indexed" (then "ingest").
- If the query references an event that requires lookup (since the auth refactor, after we shipped v2),
  set "unresolvable_anchor" to the anchor phrase and leave since/until null. Never set both.
- If the query has no time scoping at all, set "temporal": null.

Examples:
- "what shipped this week" → temporal:{"since":{"kind":"rel","offset_days":-7},"until":{"kind":"rel","offset_days":0},"basis":"source","raw_phrase":"this week","unresolvable_anchor":null}
- "PRs after 2024-03-15" → temporal:{"since":{"kind":"abs","iso":"2024-03-15T00:00:00Z"},"until":null,"basis":"source","raw_phrase":"after 2024-03-15","unresolvable_anchor":null}
- "since the auth refactor" → temporal:{"since":null,"until":null,"basis":"source","raw_phrase":"since the auth refactor","unresolvable_anchor":"the auth refactor"}
- "middleware bugs" → temporal: null
"""


@dataclass(slots=True)
class RouterEntity:
    entity_type: str
    canonical_id: str
    display_name: str
    confidence: float


@dataclass(slots=True)
class RouterOutput:
    entities: list[RouterEntity] = field(default_factory=list)
    expansions: list[str] = field(default_factory=list)
    temporal: dict[str, Any] | None = None


async def route_query(customer_id: str, query: str) -> RouterOutput:
    """Return entities + expansions + symbolic temporal for `query`.

    No cache — Haiku is on the path for every call. Symbolic temporal
    output stays query-stable, so callers can resolve it relative to a
    fresh `now` on each request.
    """
    try:
        parsed = await _call_haiku(query)
    except RouterTimeout:
        log.warning("router.timeout", query_len=len(query))
        return RouterOutput()
    except RouterParseError as exc:
        log.warning("router.parse_error", error=str(exc))
        return RouterOutput()

    return RouterOutput(
        entities=[RouterEntity(**e) for e in parsed.get("entities", [])],
        expansions=parsed.get("expansions", []),
        temporal=parsed.get("temporal"),
    )


# ---- Haiku call ---------------------------------------------------------


async def _call_haiku(query: str) -> dict:
    settings = get_settings()
    api_key = settings.anthropic_api_key.get_secret_value()
    if not api_key:
        # No Anthropic key configured — router returns empty (graceful no-op).
        return {"entities": [], "expansions": [], "temporal": None}

    client = AsyncAnthropic(api_key=api_key, timeout=ROUTER_TIMEOUT_SECONDS)
    try:
        resp = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
    except APIError as exc:
        raise RouterTimeout(str(exc)) from exc

    text = "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
    try:
        # Haiku may wrap in markdown fences; strip if present.
        if text.strip().startswith("```"):
            text = text.strip().strip("`")
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RouterParseError(f"haiku returned non-JSON: {text[:200]}") from exc
