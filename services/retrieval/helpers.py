"""Cross-cutting helpers shared by the search + list pipelines."""

from __future__ import annotations

import asyncpg

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
            WHERE customer_id = $1
              AND chunk_id = ANY($2::text[])
              AND embedding IS NOT NULL
            """,
            customer_id,
            chunk_ids,
        )
    out: dict[str, list[float]] = {}
    for r in rows:
        raw = r["emb"].strip("[]")
        out[r["chunk_id"]] = [float(x) for x in raw.split(",")] if raw else []
    return out


async def resolve_aliases(
    conn: asyncpg.Connection,
    customer_id: str,
    refs: list[tuple[str, str]],
) -> dict[tuple[str, str], str]:
    """Bulk-resolve ``(label, alias_canonical_id) → primary_canonical_id``.

    Returns a dict with one entry per input ref that IS an alias. Refs that
    are not aliases (either unmerged nodes or primaries of clusters) are
    absent from the dict — callers should treat absence as "no rewrite
    needed" and use the original canonical_id.

    Mirrors ``services/ingestion/graph_writer.py:_fetch_aliases`` so the
    write-path and read-path share batching semantics. One bulk SELECT per
    call regardless of input size — ``entity_aliases`` is keyed on
    ``(customer_id, label, alias_canonical_id)`` and answers via index-only
    scan.
    """
    if not refs:
        return {}
    labels = [r[0] for r in refs]
    aliases = [r[1] for r in refs]
    rows = await conn.fetch(
        """
        SELECT label, alias_canonical_id, primary_canonical_id
        FROM entity_aliases
        WHERE customer_id = $1
          AND (label, alias_canonical_id) IN (
                SELECT * FROM UNNEST($2::text[], $3::text[])
              )
        """,
        customer_id, labels, aliases,
    )
    return {(r["label"], r["alias_canonical_id"]): r["primary_canonical_id"] for r in rows}


async def expand_to_cluster_members(
    conn: asyncpg.Connection,
    customer_id: str,
    label: str,
    canonical_ids: list[str],
) -> dict[str, list[str]]:
    """Return ``{input_id: [member_id, ...]}`` where each input maps to its
    cluster's full member list (primary + all aliases).

    Behavior per input id:
      * Unmerged id (not in entity_aliases) → singleton ``[id]``.
      * Alias id → ``[primary, alias_1, alias_2, ...]``.
      * Primary id → ``[primary, alias_1, alias_2, ...]``.

    Implementation: one SELECT joins entity_aliases twice to find each
    input's primary (or self if unmerged), then aggregates all aliases of
    that primary. Membership is label-scoped — ids of different labels
    don't collide.

    Note: duplicate input ids are coalesced to a single output key.
    """
    if not canonical_ids:
        return {}
    rows = await conn.fetch(
        """
        WITH inputs AS (
            SELECT canonical_id FROM UNNEST($3::text[]) AS t(canonical_id)
        ),
        primaries AS (
            -- For each input, find its primary. Three cases:
            --   (a) input IS an alias    -> ea_alias.primary_canonical_id
            --   (b) input IS a primary   -> input itself
            --   (c) input is unmerged    -> input itself
            SELECT
                i.canonical_id AS input_id,
                COALESCE(ea_alias.primary_canonical_id, i.canonical_id) AS primary_canonical_id
            FROM inputs i
            LEFT JOIN entity_aliases ea_alias
              ON ea_alias.customer_id = $1
             AND ea_alias.label = $2
             AND ea_alias.alias_canonical_id = i.canonical_id
        ),
        members AS (
            -- For each (input, primary), gather all aliases of that primary.
            SELECT
                p.input_id,
                p.primary_canonical_id,
                ARRAY(
                    SELECT alias_canonical_id
                    FROM entity_aliases
                    WHERE customer_id = $1
                      AND label = $2
                      AND primary_canonical_id = p.primary_canonical_id
                ) AS alias_list
            FROM primaries p
        )
        SELECT input_id, primary_canonical_id, alias_list
        FROM members
        """,
        customer_id, label, canonical_ids,
    )
    out: dict[str, list[str]] = {}
    for r in rows:
        cluster = [r["primary_canonical_id"]] + list(r["alias_list"] or [])
        out[r["input_id"]] = cluster
    return out
