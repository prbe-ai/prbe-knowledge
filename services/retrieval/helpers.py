"""Cross-cutting helpers shared by the search + list pipelines."""

from __future__ import annotations

from shared.constants import SourceSystem
from shared.db import with_tenant

KNOWN_SOURCES: frozenset[str] = frozenset(s.value for s in SourceSystem)


def apply_entity_filter(
    fused: list,
    entities: list,
    threshold: float,
) -> tuple[list, dict[str, object]]:
    """Drop fused chunks whose content/title doesn't textually contain any
    extracted entity meeting the confidence threshold.

    Special case: when a needle is a known source platform name (slack,
    github, linear, notion, sentry, granola), a chunk also passes if its
    `source_system` matches. Slack messages themselves rarely contain the
    word "slack" in their content — they're conversation. Without this
    branch, "what happened in slack recently?" filters out every actual
    Slack chunk and returns only GitHub PRs that mention slack, which is
    the opposite of what the user wanted.

    Returns (filtered_hits, info). `info["needles"]` lists every needle;
    `info["source_needles"]` lists the subset that's a known platform.
    """
    qualifying = [e for e in entities if e.confidence >= threshold]
    info: dict[str, object] = {"enabled": True, "threshold": threshold}
    if not qualifying:
        info["skipped"] = f"no entities with confidence >= {threshold:.2f} extracted"
        info["needles"] = []
        info["source_needles"] = []
        return fused, info

    needles: list[str] = []
    seen: set[str] = set()
    for e in qualifying:
        for token in (e.canonical_id, e.display_name):
            if not token:
                continue
            tok = token.lower().strip()
            if tok and tok not in seen:
                seen.add(tok)
                needles.append(tok)
    source_needles: set[str] = {n for n in needles if n in KNOWN_SOURCES}
    info["needles"] = needles
    info["source_needles"] = sorted(source_needles)

    matched: list = []
    for hit in fused:
        haystack = ((hit.content or "") + " " + (hit.title or "")).lower()
        if any(n in haystack for n in needles):
            matched.append(hit)
            continue
        if source_needles and (hit.source_system or "").lower() in source_needles:
            matched.append(hit)
    return matched, info


async def embeddings_for_chunks(customer_id: str, chunk_ids: list[str]) -> dict[str, list[float]]:
    if not chunk_ids:
        return {}
    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            """
            SELECT chunk_id, embedding::text AS emb
            FROM chunks
            WHERE customer_id = $1 AND chunk_id = ANY($2::text[])
            """,
            customer_id,
            chunk_ids,
        )
    out: dict[str, list[float]] = {}
    for r in rows:
        raw = r["emb"].strip("[]")
        out[r["chunk_id"]] = [float(x) for x in raw.split(",")] if raw else []
    return out
