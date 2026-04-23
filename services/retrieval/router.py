"""Query router — Haiku entity extraction + query expansion with a 1h Postgres cache.

Flow:
    query → hash → query_cache lookup
    miss  → Haiku prompt → parse JSON → persist → return
    hit   → return cached entities + expansions (no API call)

The router output is advisory: it guides which indexes to hit (if entity
canonical_id matches a graph node, raise graph retriever weight) and fans out
the query into N expansions for BM25 recall. The raw query always participates
too so a bad expansion can't suppress a direct match.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from anthropic import APIError, AsyncAnthropic

from shared.config import get_settings
from shared.constants import HAIKU_MODEL
from shared.db import get_pool
from shared.exceptions import RouterParseError, RouterTimeout
from shared.logging import get_logger

log = get_logger(__name__)

CACHE_TTL = timedelta(hours=1)
ROUTER_TIMEOUT_SECONDS = 5.0

_SYSTEM_PROMPT = """You are a retrieval router. Given a user query, extract structured entities
and propose 2-4 alternate phrasings that might surface different documents.

Return strict JSON:
{
  "entities": [
    {"entity_type": "service|repo|person|ticket|pr|error_group|feature|decision|file_path|channel",
     "canonical_id": "short-stable-id",
     "display_name": "human-readable",
     "confidence": 0.0-1.0}
  ],
  "expansions": ["phrasing 1", "phrasing 2"]
}

Rules:
- Only extract entities you're confident are named concepts (not generic words).
- canonical_id should be the most likely stable identifier (service slug, repo name, user id, ticket code).
- Expansions should preserve intent but vary phrasing, synonyms, or level of specificity.
- Never invent facts. If no entities are clear, return an empty list.
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
    hit_cache: bool = False


async def route_query(customer_id: str, query: str) -> RouterOutput:
    """Return entities + expansions for `query`, cached per customer."""
    cache_key = _cache_key(customer_id, query)
    cached = await _cache_get(cache_key)
    if cached is not None:
        return RouterOutput(
            entities=[RouterEntity(**e) for e in cached["entities"]],
            expansions=cached["expansions"],
            hit_cache=True,
        )

    try:
        parsed = await _call_haiku(query)
    except RouterTimeout:
        # Graceful degradation: query runs with no expansions / entities.
        log.warning("router.timeout", query_len=len(query))
        return RouterOutput()
    except RouterParseError as exc:
        log.warning("router.parse_error", error=str(exc))
        return RouterOutput()

    await _cache_put(cache_key, customer_id, query, parsed)
    return RouterOutput(
        entities=[RouterEntity(**e) for e in parsed["entities"]],
        expansions=parsed["expansions"],
        hit_cache=False,
    )


# ---- cache --------------------------------------------------------------


def _cache_key(customer_id: str, query: str) -> str:
    return hashlib.sha256(f"{customer_id}|{query.strip().lower()}".encode()).hexdigest()


async def _cache_get(key: str) -> dict | None:
    async with get_pool().acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT entities, expansions
            FROM query_cache
            WHERE cache_key = $1 AND expires_at > NOW()
            """,
            key,
        )
    if row is None:
        return None
    return {
        "entities": json.loads(row["entities"]) if isinstance(row["entities"], str) else row["entities"],
        "expansions": json.loads(row["expansions"]) if isinstance(row["expansions"], str) else row["expansions"],
    }


async def _cache_put(key: str, customer_id: str, query: str, parsed: dict) -> None:
    expires_at = datetime.now(UTC) + CACHE_TTL
    query_hash = hashlib.sha256(query.encode()).hexdigest()
    async with get_pool().acquire() as conn:
        await conn.execute(
            """
            INSERT INTO query_cache
                (cache_key, customer_id, query_text_hash, entities, expansions, expires_at)
            VALUES ($1, $2, $3, $4::jsonb, $5::jsonb, $6)
            ON CONFLICT (cache_key)
            DO UPDATE SET entities = EXCLUDED.entities,
                          expansions = EXCLUDED.expansions,
                          expires_at = EXCLUDED.expires_at
            """,
            key,
            customer_id,
            query_hash,
            json.dumps(parsed["entities"]),
            json.dumps(parsed["expansions"]),
            expires_at,
        )


# ---- Haiku call ---------------------------------------------------------


async def _call_haiku(query: str) -> dict:
    settings = get_settings()
    api_key = settings.anthropic_api_key.get_secret_value()
    if not api_key:
        # No Anthropic key configured — router returns empty (graceful no-op).
        return {"entities": [], "expansions": []}

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
