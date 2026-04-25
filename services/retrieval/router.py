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
from datetime import UTC, datetime
from typing import Any

from anthropic import APIError, AsyncAnthropic

from shared.config import get_settings
from shared.constants import HAIKU_MODEL
from shared.exceptions import RouterParseError, RouterTimeout
from shared.logging import get_logger

log = get_logger(__name__)

ROUTER_TIMEOUT_SECONDS = 5.0


def _build_system_prompt(now: datetime) -> str:
    """Build the router system prompt with `now` baked in.

    Bare phrases like "April 15th" or "Q1" need a current-year anchor or
    Haiku will pick arbitrarily from training distribution. Passing today's
    date avoids that — relative resolution stays stable as the calendar moves.
    """
    today_iso = now.strftime("%Y-%m-%d")
    return f"""You are a retrieval router. Given a user query, extract structured entities,
propose 2-4 alternate phrasings, and capture any time scoping the query implies.

The user's current date (UTC) is: {today_iso}
Use this to resolve all relative and bare-month/bare-day phrases. When the user
says "April 15th" without a year, assume the most recent April 15th relative to
today. When the user says "this week", use today's calendar week. Never default
to a specific historical year.

Return strict JSON:
{{
  "entities": [
    {{"entity_type": "service|repo|person|ticket|pr|error_group|feature|decision|file_path|channel",
     "canonical_id": "short-stable-id",
     "display_name": "human-readable",
     "confidence": 0.0-1.0}}
  ],
  "expansions": ["phrasing 1", "phrasing 2"],
  "temporal": {{
    "since": null | {{"kind": "rel", "offset_days": -30}} | {{"kind": "abs", "iso": "YYYY-MM-DDTHH:MM:SSZ"}},
    "until": null | {{"kind": "rel", "offset_days": 0}}   | {{"kind": "abs", "iso": "..."}},
    "basis": "source",
    "raw_phrase": "in the last month",
    "unresolvable_anchor": null | "the auth refactor"
  }} | null,
  "sort": {{
    "field": "created_at" | "updated_at",
    "direction": "asc" | "desc",
    "trigger_phrase": "oldest"
  }} | null
}}

Rules:
- Only extract entities you're confident are named concepts (not generic words).
- canonical_id should be the most likely stable identifier (service slug, repo name, user id, ticket code).
- Expansions should preserve intent but vary phrasing, synonyms, or level of specificity.
- Never invent facts. If no entities are clear, return an empty list.

Temporal rules:
- Prefer "rel" with offset_days for any phrase that's relative to "now"
  (last week, yesterday, this month, last 30 days, today, since today).
  N is negative for past, 0 for now.
- Use "abs" only for fully-qualified dates like "since 2024-03-15" or
  "between 2025-Q1 and 2025-Q3" where the user gave you the year explicitly.
- Bare month/day phrases like "April 15th" or "since March" should be resolved
  to the most recent occurrence relative to {today_iso}. Output as "abs" with
  the resolved year in the ISO string. Do not pick years from prior knowledge.
- "basis" is "source" unless the query explicitly says "ingested" or "indexed"
  (then "ingest").
- If the query references an event that requires lookup (since the auth refactor,
  after we shipped v2), set "unresolvable_anchor" to the anchor phrase and leave
  since/until null. Never set both.
- If the query has no time scoping at all, set "temporal": null.

Sort rules:
- "oldest", "earliest", "first", "first ever" → {{"field":"created_at","direction":"asc","trigger_phrase":"oldest"}}
- "newest", "latest", "most recent", "last", "the last X" → {{"field":"updated_at","direction":"desc","trigger_phrase":"newest"}}
- "recently edited", "last touched", "most recently updated" → {{"field":"updated_at","direction":"desc","trigger_phrase":"recently edited"}}
- If the query asks for state-of-the-world without explicit sort intent
  ("what does the auth middleware do"), set "sort": null and let semantic
  relevance rank.
- A time filter and a sort can coexist: "the oldest fly.io change since
  April 15th" filters with `temporal` AND sorts with `sort`.

Examples (assume today is {today_iso}):
- "what shipped this week" → temporal:{{"since":{{"kind":"rel","offset_days":-7}},"until":{{"kind":"rel","offset_days":0}},"basis":"source","raw_phrase":"this week","unresolvable_anchor":null}}
- "yesterday" → temporal:{{"since":{{"kind":"rel","offset_days":-1}},"until":{{"kind":"rel","offset_days":0}},"basis":"source","raw_phrase":"yesterday","unresolvable_anchor":null}}
- "since April 15th" with bare month/day → resolve to the most recent April 15 at or before {today_iso}, output "abs" iso with that year.
- "since the auth refactor" → temporal:{{"since":null,"until":null,"basis":"source","raw_phrase":"since the auth refactor","unresolvable_anchor":"the auth refactor"}}
- "middleware bugs" → temporal: null
- "what was the oldest fly.io change since April 15th" →
    temporal: {{since: April 15th of the most recent year ≤ {today_iso}, ...}},
    sort: {{"field":"created_at","direction":"asc","trigger_phrase":"oldest"}}
- "the latest changes to billing" → sort:{{"field":"updated_at","direction":"desc","trigger_phrase":"latest"}}
- "what is the auth middleware" → temporal: null, sort: null
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
    sort: dict[str, Any] | None = None


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
        sort=parsed.get("sort"),
    )


# ---- Haiku call ---------------------------------------------------------


async def _call_haiku(query: str) -> dict:
    settings = get_settings()
    api_key = settings.anthropic_api_key.get_secret_value()
    if not api_key:
        # No Anthropic key configured — router returns empty (graceful no-op).
        return {"entities": [], "expansions": [], "temporal": None, "sort": None}

    system_prompt = _build_system_prompt(datetime.now(UTC))
    client = AsyncAnthropic(api_key=api_key, timeout=ROUTER_TIMEOUT_SECONDS)
    try:
        resp = await client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=512,
            system=system_prompt,
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
