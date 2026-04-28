"""Deterministic SQL retriever — drives the list pipeline.

The semantic retrievers (vector, BM25, graph) all rank by relevance to the
query string. That's the wrong tool for queries like "3 most recent github
commits": their candidate pool is filtered by similarity to the literal
query text BEFORE sort runs, so the truly newest docs may never enter
ranking. This retriever skips relevance entirely and runs a parameterized
window/aggregate query against `documents` directly.

Three operations:
  - list:     SELECT * ORDER BY <field> <dir> LIMIT N    → returns one
              representative chunk per matching doc
  - count:    SELECT COUNT(*)                            → returns int
  - group_by: SELECT <key>, COUNT(*) GROUP BY <key>      → returns
              ranked groups (e.g. authors, sources)

All three respect the same filter set: customer_id (RLS), source_system,
doc_type, person (author_id), TemporalSpec, valid_to IS NULL.

Group-by is constrained to a small allowlist (`source_system`, `doc_type`,
`author_id`) so callers can't smuggle arbitrary SQL via the `key` field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from services.retrieval.temporal import build_predicate
from shared.db import with_tenant
from shared.models import TemporalSpec


@dataclass(slots=True)
class GraphEntityFilter:
    """One narrowing-entity-type filter, AND'd with other types.

    `label` is the graph_nodes label (NodeLabel.value) — Repo, Channel,
    Ticket, PR, Service. `values` is the set of acceptable matches; they
    are OR'd within the filter (a doc passes if it links to ANY of them).
    Across multiple GraphEntityFilters, the filters are AND'd (a doc must
    link to entities from EACH type).

    `values` accepts both the bare and full forms of an identifier
    (e.g. ['prbe-backend', 'prbe-ai/prbe-backend', 'Probe Backend']) — the
    SQL helper applies a loose match across canonical_id and
    properties->>'name' so any of these forms matches the same graph node.
    """

    label: str
    values: list[str] = field(default_factory=list)


def _entity_match_clause(
    label: str,
    values: list[str],
    next_param_index: int,
) -> tuple[str, list[object]]:
    """Build an EXISTS-clause SQL fragment + appended params for one
    GraphEntityFilter.

    Returns SQL like:

      AND EXISTS (
        SELECT 1 FROM graph_edges e
        JOIN graph_nodes gn ON gn.node_id = e.from_node_id
                            OR gn.node_id = e.to_node_id
        WHERE e.customer_id = $1
          AND gn.customer_id = $1
          AND gn.label = $LABEL
          AND (
            <loose match against any value>
          )
          AND <doc node match>
      )

    Loose match per value:
        LOWER(gn.canonical_id) = LOWER($X)
        OR LOWER(gn.canonical_id) LIKE '%/' || LOWER($X)
        OR LOWER(gn.properties->>'name') = LOWER($X)

    NOTE: the suffix-LIKE arm has a leading wildcard so it can't use a
    btree functional index; it falls back to a seq scan over rows
    pre-narrowed by (customer_id, label). At our scale that's fine.
    See migration 0019 for context.
    """
    if not values:
        return "", []

    params: list[object] = [label]
    label_param = next_param_index

    value_clauses: list[str] = []
    for value in values:
        next_param_index += 1
        params.append(value)
        i = next_param_index
        value_clauses.append(
            "(LOWER(gn.canonical_id) = LOWER($I) "
            "OR LOWER(gn.canonical_id) LIKE '%/' || LOWER($I) "
            "OR LOWER(gn.properties->>'name') = LOWER($I))".replace("$I", f"${i}")
        )

    # Anchor: the graph_nodes row representing the document itself.
    # We need a graph_edges row connecting the doc node to a node matching
    # the entity filter. Document nodes have label='Document' and
    # canonical_id = doc_id.
    #
    # `e.valid_to IS NULL` filters out soft-closed edges. No code path
    # writes valid_to on edges today, so this is prophylactic — when a
    # future feature ("user left channel" → close the Person→Channel
    # edge) starts soft-deleting, the entity filter won't silently pull
    # in stale relationships.
    sql = f"""
        AND EXISTS (
            SELECT 1
            FROM graph_nodes doc_gn
            JOIN graph_edges e
              ON e.customer_id = $1
             AND e.valid_to IS NULL
             AND (e.from_node_id = doc_gn.node_id OR e.to_node_id = doc_gn.node_id)
            JOIN graph_nodes gn
              ON gn.customer_id = $1
             AND gn.node_id = CASE
                   WHEN e.from_node_id = doc_gn.node_id THEN e.to_node_id
                   ELSE e.from_node_id END
             AND gn.label = ${label_param}
             AND ({" OR ".join(value_clauses)})
            WHERE doc_gn.customer_id = $1
              AND doc_gn.label = 'Document'
              AND doc_gn.canonical_id = d.doc_id
        )
    """
    return sql, params


@dataclass(slots=True)
class SQLListHit:
    """Same shape as VectorHit/BM25Hit so fusion code (or post-processing)
    can treat all retriever outputs uniformly. `score` is just rank-based
    (1.0 / rank) — callers reorder by created_at/updated_at directly."""

    chunk_id: str
    doc_id: str
    doc_version: int
    source_system: str
    source_url: str
    title: str | None
    content: str
    created_at: datetime
    updated_at: datetime
    score: float


GroupByKey = Literal["source_system", "doc_type", "author_id"]
SortField = Literal["created_at", "updated_at"]
SortDirection = Literal["asc", "desc"]


_ALLOWED_GROUP_KEYS: frozenset[str] = frozenset({"source_system", "doc_type", "author_id"})
_ALLOWED_SORT_FIELDS: frozenset[str] = frozenset({"created_at", "updated_at"})
_ALLOWED_SORT_DIRECTIONS: frozenset[str] = frozenset({"asc", "desc"})


def _validate_sort(field: str, direction: str) -> tuple[str, str]:
    if field not in _ALLOWED_SORT_FIELDS:
        raise ValueError(f"sort field must be one of {_ALLOWED_SORT_FIELDS}, got {field!r}")
    if direction not in _ALLOWED_SORT_DIRECTIONS:
        raise ValueError(
            f"sort direction must be one of {_ALLOWED_SORT_DIRECTIONS}, got {direction!r}"
        )
    return field, direction


def _validate_group_key(key: str) -> str:
    if key not in _ALLOWED_GROUP_KEYS:
        raise ValueError(f"group_by key must be one of {_ALLOWED_GROUP_KEYS}, got {key!r}")
    return key


async def sql_list(
    customer_id: str,
    top_k: int = 20,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    author_ids: list[str] | None = None,
    graph_entity_filters: list[GraphEntityFilter] | None = None,
    sort_field: SortField = "updated_at",
    sort_direction: SortDirection = "desc",
    temporal: TemporalSpec | None = None,
) -> list[SQLListHit]:
    """ORDER BY <sort_field> <sort_direction> LIMIT top_k.

    Returns one hit per matching document — the chunk with `chunk_index = 0`
    (the first chunk of the doc) when one exists, otherwise the chunk with
    the lowest `chunk_index`. That gives the dispatcher something to put in
    `QueryResponse.chunks` without having to fetch the entire doc.
    """
    field, direction = _validate_sort(sort_field, sort_direction)
    spec = temporal or TemporalSpec()

    params: list = [customer_id, top_k]
    source_filter = ""
    if sources:
        params.append(sources)
        source_filter = f"AND d.source_system = ANY(${len(params)}::text[])"

    doc_type_filter = ""
    if doc_types:
        params.append(doc_types)
        doc_type_filter = f"AND d.doc_type = ANY(${len(params)}::text[])"

    author_filter = ""
    if author_ids:
        params.append(author_ids)
        author_filter = f"AND d.author_id = ANY(${len(params)}::text[])"

    # Build entity-filter EXISTS clauses; AND'd across types, OR'd
    # within values for one type.
    entity_clauses_sql = ""
    if graph_entity_filters:
        for ef in graph_entity_filters:
            clause_sql, clause_params = _entity_match_clause(
                ef.label, ef.values, next_param_index=len(params) + 1
            )
            if clause_sql:
                params.extend(clause_params)
                entity_clauses_sql += clause_sql

    pred = build_predicate(spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1)
    params.extend(pred.params)

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            f"""
            WITH ranked_docs AS (
                SELECT d.doc_id, d.version, d.source_system, d.source_url,
                       d.title, d.created_at, d.updated_at
                FROM documents d
                WHERE d.customer_id = $1
                  {pred.doc_sql}
                  {source_filter}
                  {doc_type_filter}
                  {author_filter}
                  {entity_clauses_sql}
                ORDER BY d.{field} {direction.upper()}, d.doc_id
                LIMIT $2
            )
            SELECT c.chunk_id,
                   rd.doc_id,
                   rd.version AS doc_version,
                   rd.source_system,
                   rd.source_url,
                   rd.title,
                   c.content,
                   rd.created_at,
                   rd.updated_at,
                   c.chunk_index
            FROM ranked_docs rd
            JOIN LATERAL (
                SELECT chunk_id, content, chunk_index
                FROM chunks
                WHERE customer_id = $1
                  AND doc_id = rd.doc_id
                  AND valid_to IS NULL
                  AND kind = 'content'
                  AND rd.version BETWEEN first_seen_version AND last_seen_version
                ORDER BY chunk_index
                LIMIT 1
            ) c ON TRUE
            ORDER BY rd.{field} {direction.upper()}, rd.doc_id
            """,
            *params,
        )

    hits: list[SQLListHit] = []
    for i, r in enumerate(rows):
        # rank-derived score: 1.0 for the top hit, decaying. Callers that
        # care about pure recency reorder by `updated_at` directly; this
        # field exists only so the QueryResponse shape stays consistent.
        hits.append(
            SQLListHit(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                doc_version=r["doc_version"],
                source_system=r["source_system"],
                source_url=r["source_url"],
                title=r["title"],
                content=r["content"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
                score=1.0 / (1 + i),
            )
        )
    return hits


async def sql_count(
    customer_id: str,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    author_ids: list[str] | None = None,
    graph_entity_filters: list[GraphEntityFilter] | None = None,
    temporal: TemporalSpec | None = None,
) -> int:
    """Single SELECT COUNT(*) over documents matching the filter set."""
    spec = temporal or TemporalSpec()

    params: list = [customer_id]
    source_filter = ""
    if sources:
        params.append(sources)
        source_filter = f"AND d.source_system = ANY(${len(params)}::text[])"

    doc_type_filter = ""
    if doc_types:
        params.append(doc_types)
        doc_type_filter = f"AND d.doc_type = ANY(${len(params)}::text[])"

    author_filter = ""
    if author_ids:
        params.append(author_ids)
        author_filter = f"AND d.author_id = ANY(${len(params)}::text[])"

    entity_clauses_sql = ""
    if graph_entity_filters:
        for ef in graph_entity_filters:
            clause_sql, clause_params = _entity_match_clause(
                ef.label, ef.values, next_param_index=len(params) + 1
            )
            if clause_sql:
                params.extend(clause_params)
                entity_clauses_sql += clause_sql

    pred = build_predicate(spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1)
    params.extend(pred.params)

    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            f"""
            SELECT COUNT(*)::bigint AS n
            FROM documents d
            WHERE d.customer_id = $1
              {pred.doc_sql}
              {source_filter}
              {doc_type_filter}
              {author_filter}
              {entity_clauses_sql}
            """,
            *params,
        )
    return int(row["n"]) if row else 0


async def sql_group_by(
    customer_id: str,
    key: GroupByKey,
    top_k: int = 20,
    sources: list[str] | None = None,
    doc_types: list[str] | None = None,
    author_ids: list[str] | None = None,
    graph_entity_filters: list[GraphEntityFilter] | None = None,
    temporal: TemporalSpec | None = None,
) -> list[dict[str, object]]:
    """SELECT <key>, COUNT(*) GROUP BY <key> ORDER BY count DESC LIMIT top_k.

    `key` is restricted to {source_system, doc_type, author_id} to prevent
    arbitrary column injection — the value flows through unparameterized.
    Returns a list of {"key": <value>, "n": <count>} dicts.
    """
    col = _validate_group_key(key)
    spec = temporal or TemporalSpec()

    params: list = [customer_id, top_k]
    source_filter = ""
    if sources:
        params.append(sources)
        source_filter = f"AND d.source_system = ANY(${len(params)}::text[])"

    doc_type_filter = ""
    if doc_types:
        params.append(doc_types)
        doc_type_filter = f"AND d.doc_type = ANY(${len(params)}::text[])"

    author_filter = ""
    if author_ids:
        params.append(author_ids)
        author_filter = f"AND d.author_id = ANY(${len(params)}::text[])"

    entity_clauses_sql = ""
    if graph_entity_filters:
        for ef in graph_entity_filters:
            clause_sql, clause_params = _entity_match_clause(
                ef.label, ef.values, next_param_index=len(params) + 1
            )
            if clause_sql:
                params.extend(clause_params)
                entity_clauses_sql += clause_sql

    pred = build_predicate(spec, doc_alias="d", chunk_alias="c", next_param_index=len(params) + 1)
    params.extend(pred.params)

    async with with_tenant(customer_id) as conn:
        rows = await conn.fetch(
            f"""
            SELECT d.{col} AS key, COUNT(*)::bigint AS n
            FROM documents d
            WHERE d.customer_id = $1
              {pred.doc_sql}
              {source_filter}
              {doc_type_filter}
              {author_filter}
              {entity_clauses_sql}
            GROUP BY d.{col}
            ORDER BY n DESC, d.{col}
            LIMIT $2
            """,
            *params,
        )

    return [{"key": r["key"], "n": int(r["n"])} for r in rows]
