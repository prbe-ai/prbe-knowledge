"""Cross-cutting helpers shared by the search + list pipelines."""

from __future__ import annotations

import asyncpg

from engine.shared.alias_sql import RESOLVE_ALIASES_SQL, cluster_members_cte, resolve_to_root_cte
from engine.shared.constants import SourceSystem
from engine.shared.db import with_tenant

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

    The primary is the END of the alias chain (a -> b -> c resolves a to c).
    Shares its SQL with ``engine/ingest/graph_writer.py:_fetch_aliases`` so
    the write path and the read path route identically. One bulk query per
    call regardless of input size.
    """
    if not refs:
        return {}
    labels = [r[0] for r in refs]
    aliases = [r[1] for r in refs]
    rows = await conn.fetch(RESOLVE_ALIASES_SQL, customer_id, labels, aliases)
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

    Chains are followed both ways: the primary is the end of the input's
    alias chain, and the members are every alias that routes to it through
    any chain. Membership is label-scoped — ids of different labels don't
    collide.

    Note: duplicate input ids are coalesced to a single output key.
    """
    if not canonical_ids:
        return {}
    rows = await conn.fetch(
        f"""
        WITH RECURSIVE inputs AS (
            SELECT DISTINCT canonical_id FROM UNNEST($3::text[]) AS t(canonical_id)
        ),
        {resolve_to_root_cte(customer_param="$1", label_param="$2", inputs="inputs")},
        primaries AS (
            -- Each input's primary: the end of its alias chain, or itself
            -- when it is a primary or unmerged.
            SELECT i.canonical_id AS input_id,
                   COALESCE(
                       (SELECT u.cur FROM up u WHERE u.input_id = i.canonical_id
                        ORDER BY u.depth DESC LIMIT 1),
                       i.canonical_id
                   ) AS root
            FROM inputs i
        ),
        roots AS (SELECT DISTINCT root FROM primaries),
        {cluster_members_cte(customer_param="$1", label_param="$2", roots="roots")}
        SELECT p.input_id, p.root AS primary_canonical_id,
               ARRAY(SELECT DISTINCT m.member FROM members m WHERE m.root = p.root) AS alias_list
        FROM primaries p
        """,
        customer_id, label, canonical_ids,
    )
    out: dict[str, list[str]] = {}
    for r in rows:
        cluster = [r["primary_canonical_id"], *(r["alias_list"] or [])]
        out[r["input_id"]] = cluster
    return out


async def expand_to_author_id_set(
    conn: asyncpg.Connection,
    customer_id: str,
    person_canonical_ids: list[str],
) -> list[str]:
    """Return the union of (cluster member canonical_ids) and (Lane E enrichment
    property values on cluster members) for the given Person canonical_ids.

    This is the right input for the SQL filter ``documents.author_id = ANY(...)``
    because ``documents.author_id`` is the raw connector-side identifier and is
    never rewritten on merge:

      * GitHub PR → ``author_id = 'mahitoburrito'`` (login)
      * Slack    → ``author_id = 'U0AUP7A7WCS'``    (uid)
      * Granola  → ``author_id = 'mahit@prbe.ai'``  (email)
      * Claude Code → ``author_id = '08578d48-...'`` (better-auth uuid, stored
        as a *property* on the Slack-rooted Person row by Lane E enrichment —
        never reified as its own graph_nodes row, so plain cluster expansion
        misses it)

    To match all of Mahit's documents under a single Person cluster, the
    SQL filter needs every (a) cluster member canonical_id AND (b) every
    ``properties->>'employee_id'``/``'login'``/``'email'`` value on each
    cluster member.

    Indexed by partial functional indexes added in migration 0091:
      idx_graph_nodes_person_{employee_id,login,email}.

    Args:
      person_canonical_ids: Person canonical_ids resolved from the extractor
        (e.g. ``['U0AUP7A7WCS', 'mahit@prbe.ai']``). Caller may pass aliases,
        primaries, or unmerged ids — all three shapes are handled.

    Returns:
      Flat deduplicated list of strings safe to substitute into
      ``WHERE documents.author_id = ANY($1::text[])``. Empty input → empty
      list.
    """
    if not person_canonical_ids:
        return []
    rows = await conn.fetch(
        f"""
        WITH RECURSIVE inputs AS (
            SELECT DISTINCT canonical_id FROM UNNEST($2::text[]) AS t(canonical_id)
        ),
        {resolve_to_root_cte(customer_param="$1", label_param="'Person'", inputs="inputs")},
        -- Resolve each input to the end of its alias chain (self if unmerged).
        primaries AS (
            SELECT COALESCE(
                       (SELECT u.cur FROM up u WHERE u.input_id = i.canonical_id
                        ORDER BY u.depth DESC LIMIT 1),
                       i.canonical_id
                   ) AS root
            FROM inputs i
        ),
        roots AS (SELECT DISTINCT root FROM primaries),
        {cluster_members_cte(customer_param="$1", label_param="'Person'", roots="roots")},
        -- Full cluster membership: each root plus every alias routing to it.
        cluster_members AS (
            SELECT root AS member FROM roots
            UNION
            SELECT member FROM members
        ),
        -- Lane E enrichment values: each member Person's enrichment properties
        -- become valid author_id matches.
        property_values AS (
            SELECT g.properties->>'employee_id' AS v
            FROM graph_nodes g
            JOIN cluster_members cm ON g.canonical_id = cm.member
            WHERE g.customer_id = $1
              AND g.label = 'Person'
              AND g.properties->>'employee_id' IS NOT NULL
            UNION
            SELECT g.properties->>'login'
            FROM graph_nodes g
            JOIN cluster_members cm ON g.canonical_id = cm.member
            WHERE g.customer_id = $1
              AND g.label = 'Person'
              AND g.properties->>'login' IS NOT NULL
            UNION
            SELECT g.properties->>'email'
            FROM graph_nodes g
            JOIN cluster_members cm ON g.canonical_id = cm.member
            WHERE g.customer_id = $1
              AND g.label = 'Person'
              AND g.properties->>'email' IS NOT NULL
        )
        SELECT DISTINCT v FROM (
            SELECT member AS v FROM cluster_members
            UNION ALL
            SELECT v FROM property_values
        ) all_ids
        WHERE v IS NOT NULL AND v <> ''
        """,
        customer_id, person_canonical_ids,
    )
    return [r["v"] for r in rows]


def source_key_predicate(
    params: list,
    source_keys: list[str] | None,
    *,
    alias: str,
    include_keyless: bool = False,
) -> str:
    """Build the `AND ...source_key...` predicate shared by every retriever.

    Appends the source_keys array to `params` (in place) and returns the SQL
    fragment. Returns '' when `source_keys` is falsy.

    Default is a HARD filter: `metadata->>'source_key' = ANY($n)`, so a doc with
    no source_key (connector-ingested: github, transcripts) is excluded. That
    is the long-standing behaviour and stays byte-identical.

    When `include_keyless` is True the predicate also admits keyless docs:
    `(metadata->>'source_key' = ANY($n) OR metadata->>'source_key' IS NULL)`.
    That lets ONE request mix keyed custom-ingest corpora with keyless connector
    corpora instead of a client fanning out one request per corpus.
    """
    if not source_keys:
        return ""
    params.append(source_keys)
    idx = len(params)
    keyed = f"{alias}.metadata->>'source_key' = ANY(${idx}::text[])"
    if include_keyless:
        return f"AND ({keyed} OR {alias}.metadata->>'source_key' IS NULL)"
    return f"AND {keyed}"


def project_scope_predicate(
    params: list,
    project_id: str | None,
    *,
    alias: str,
) -> str:
    """Build the `AND ...project_id...` predicate shared by every retriever.

    The pre-search half of `QueryRequest.scope`: a HARD filter on
    `metadata->>'project_id'`, the key research-os stamps on every document
    that belongs to a project. Appends `project_id` to `params` (in place)
    and returns the SQL fragment; '' when no project scope is set.

    Same shape as `source_key_predicate` on purpose -- the agent's in-loop
    gate (`tools._doc_scope_sql`) and the response gate
    (`adapter._enforce_scope_on_chunks`) restate exactly this predicate, and
    `tests/retrieval/agent/test_project_scope_threading.py` pins the three
    together. Documents with no project_id are OUT of a project scope.
    """
    if not project_id:
        return ""
    params.append(project_id)
    return f"AND {alias}.metadata->>'project_id' = ${len(params)}"


#: The values `documents.metadata->>'origin'` may carry. Anything else -- a
#: typo, a value from a newer writer, a hand-edited row -- is treated as
#: UNKNOWN and dropped. A value a reader cannot interpret is worse than an
#: absent one: it looks like an attribution and is not.
VALID_ORIGINS = frozenset({"human", "generated"})


def origin_of(row: object) -> str | None:
    """Read `origin` off a retriever row: "human", "generated", or None.

    Shared by every retriever so the three cannot disagree about what counts
    as an attribution. Tolerant of a row that has no `origin` column at all --
    an inner query that predates it still returns rows, and a missing key must
    read as "unknown" rather than raising mid-search.

    None NEVER means "human". Claiming a person wrote something we cannot
    attribute is the failure this field exists to prevent, and it is the
    reason the value is dropped rather than defaulted.
    """
    try:
        value = row["origin"]  # type: ignore[index]
    except (KeyError, IndexError, TypeError):
        return None
    return value if value in VALID_ORIGINS else None
