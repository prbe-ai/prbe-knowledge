"""Deterministic list pipeline — SQL window/aggregate against `documents`.

When the router classifies a query as `mode=list` (sort or temporal intent
present, no topic entity), retrieval bypasses the vector + BM25 + graph
fusion entirely and runs a parameterized SQL query.

Three operations:
  - list:     ORDER BY <field> <dir> LIMIT N           → chunks (one per doc)
  - count:    COUNT(*)                                  → aggregation={count: N}
  - group_by: <key>, COUNT(*) GROUP BY <key>            → aggregation={groups: [...]}

ACL filtering still runs (no-op until P4 flips ENFORCE_ACL on, but the path
is exercised so the Phase-1 flip doesn't surprise anyone). Embedding-based
dedup is skipped — SQL returns one chunk per distinct doc_id by construction,
so there are no near-duplicates to collapse.
"""

from __future__ import annotations

import time

from services.retrieval.acl import filter_by_acl
from services.retrieval.helpers import expand_to_author_id_set
from services.retrieval.retrievers.related_entities import (
    build_exclude_node_keys,
    expand_exclude_keys_with_aliases,
    walk_result_doc_neighbors,
)
from services.retrieval.retrievers.sql import sql_count, sql_group_by, sql_list
from services.retrieval.router import Intent
from shared.constants import SourceSystem
from shared.db import with_tenant
from shared.logging import get_logger
from shared.models import (
    MatchProvenance,
    QueryChunk,
    QueryDocumentResult,
    QueryRequest,
    QueryResult,
    RelatedEntity,
    RetrieveResponse,
    TemporalSpec,
)

log = get_logger(__name__)


def _pick_sort(intent: Intent) -> tuple[str, str]:
    """Default to (updated_at, desc). Honor router-extracted sort when present."""
    if intent.sort:
        field = intent.sort.get("field") or "updated_at"
        direction = intent.sort.get("direction") or "desc"
        if field not in ("created_at", "updated_at"):
            field = "updated_at"
        if direction not in ("asc", "desc"):
            direction = "desc"
        return field, direction
    return "updated_at", "desc"


def _author_ids_from_entities(intent: Intent) -> list[str] | None:
    """Pull person canonical_ids out of the extracted entities for the
    SQL `author_id = ANY(...)` filter. Confidence threshold matches what
    apply_entity_filter uses by default — entities below 0.7 are noise."""
    ids = [
        e.canonical_id
        for e in intent.entities
        if e.entity_type == "person" and e.confidence >= 0.7 and e.canonical_id
    ]
    return ids or None


# Narrowing-entity types that map to a graph_nodes label. `person` is
# excluded — it has its own direct documents.author_id filter, no need
# to round-trip through graph. `feature`/`decision`/`error_group` are
# TOPIC entities and never reach the list path (gated out upstream).
# `file_path` is intentionally omitted — file paths don't have a clean
# graph_node representation, deferred to a follow-up.
_NARROWING_TO_LABEL: dict[str, str] = {
    "service": "Service",
    # Post-0091, repo/ticket/pr/channel/issue all collapse to Document. The
    # router's entity_type stays fine-grained; the SQL filter just sees
    # graph_nodes labeled Document (with properties.kind = Repo/PR/etc. for
    # the entity-shape rows, or just doc_type for the content rows). The
    # broader loose-match in apply_entity_filter handles both shapes.
    "repo": "Document",
    "ticket": "Document",
    "pr": "Document",
    "issue": "Document",
    "channel": "Document",
}


def _graph_entity_filters_from_intent(
    intent: Intent, min_confidence: float = 0.7
) -> list:
    """Build GraphEntityFilters from narrowing entities in the router's
    output. Each entity becomes one filter (label, [canonical_id,
    display_name]). The SQL helper applies loose case-insensitive matching
    across canonical_id and properties->>'name' so both bare and full
    forms match the same graph node.

    Returns an empty list when there are no qualifying entities — caller
    treats it as "no entity filter" (existing behavior preserved).
    """
    from services.retrieval.retrievers.sql import GraphEntityFilter

    filters: list[GraphEntityFilter] = []
    for e in intent.entities:
        label = _NARROWING_TO_LABEL.get(e.entity_type)
        if label is None:
            continue
        if e.confidence < min_confidence:
            continue
        # Both forms — canonical_id might be the full owner/repo while
        # display_name is the bare name (or vice versa). Loose match in
        # SQL handles whichever form lives in graph_nodes.
        values = [v for v in (e.canonical_id, e.display_name) if v]
        if not values:
            continue
        # Dedupe (case-insensitive) — both fields often hold the same
        # string for non-repo entities.
        seen: set[str] = set()
        deduped: list[str] = []
        for v in values:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(v)
        filters.append(GraphEntityFilter(label=label, values=deduped))
    return filters


async def run_list(
    req: QueryRequest,
    customer_id: str,
    intent: Intent,
    spec: TemporalSpec,
    temporal_meta: dict[str, object],
    sort_meta: dict[str, object] | None,
    extracted_entities: list[dict[str, object]],
    doc_types: list[str] | None,
    trace_id: str,
    timing: dict[str, float],
    intent_idx: int = 0,
) -> RetrieveResponse:
    sources = [s.value for s in req.sources] if req.sources else None
    # Entity-based hard filters (author_id from `person` entities,
    # graph_nodes membership from narrowing entities) are gated on
    # `req.entity_must_match` — same flag that gates the search path's
    # post-fusion entity filter. When False (the MCP / default), the
    # list path skips entity-based narrowing and relies on sort +
    # temporal + source + doc_type only. This avoids zero-result SQL
    # when the router extracts an entity that doesn't have a matching
    # graph_nodes row or documents.author_id.
    if req.entity_must_match:
        author_ids = _author_ids_from_entities(intent)
        graph_entity_filters = _graph_entity_filters_from_intent(intent)
        # Phase 2 (post-0091): expand each Person canonical_id to the FULL
        # set of valid author_id values — cluster members + Lane E enrichment
        # property values (employee_id / login / email) on each member.
        # `documents.author_id` is historical raw text from whichever connector
        # wrote the row, and that raw text might be a Slack uid OR a GitHub
        # login OR a claude_code better-auth uuid that lives only as a Person
        # property. Without expanding through properties, recency queries miss
        # the claude_code sessions whose author_id is the uuid (see
        # helpers.expand_to_author_id_set docstring).
        if author_ids:
            async with with_tenant(customer_id) as conn:
                author_ids = await expand_to_author_id_set(
                    conn, customer_id, person_canonical_ids=author_ids,
                )
    else:
        author_ids = None
        graph_entity_filters = []
    operation = (intent.operation or "list").lower()
    if operation not in ("list", "count", "group_by"):
        operation = "list"

    aggregation: dict[str, object] | None = None
    document_results: list[QueryDocumentResult] = []
    total_candidates = 0

    t_sql = time.perf_counter()

    if operation == "count":
        n = await sql_count(
            customer_id,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            graph_entity_filters=graph_entity_filters or None,
            temporal=spec,
        )
        aggregation = {"count": n}
        total_candidates = n

    elif operation == "group_by":
        key = (intent.group_by_key or "source_system").lower()
        if key not in ("source_system", "doc_type", "author_id"):
            key = "source_system"
        groups = await sql_group_by(
            customer_id,
            key=key,  # type: ignore[arg-type]
            top_k=req.top_k,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            graph_entity_filters=graph_entity_filters or None,
            temporal=spec,
        )
        aggregation = {"key": key, "groups": groups}
        total_candidates = len(groups)

    else:  # operation == "list"
        sort_field, sort_dir = _pick_sort(intent)
        hits = await sql_list(
            customer_id,
            top_k=req.top_k,
            sources=sources,
            doc_types=doc_types,
            author_ids=author_ids,
            graph_entity_filters=graph_entity_filters or None,
            sort_field=sort_field,  # type: ignore[arg-type]
            sort_direction=sort_dir,  # type: ignore[arg-type]
            temporal=spec,
        )
        # ACL still runs for shape consistency (no-op until ENFORCE_ACL flips).
        t_acl = time.perf_counter()
        hits = await filter_by_acl(customer_id, req.requesting_user_id, hits)
        timing["acl_ms"] = (time.perf_counter() - t_acl) * 1000

        total_candidates = len(hits)
        # SQL list emits one chunk per doc by construction. Wrap each into
        # a QueryDocumentResult with a single QueryChunk under it -- the
        # per-doc shape matches the polymorphic search-path output.
        document_results = [
            QueryDocumentResult(
                canonical_id=h.doc_id,
                doc_id=h.doc_id,
                doc_version=h.doc_version,
                source_system=SourceSystem(h.source_system),
                source_url=h.source_url,
                title=h.title,
                author_id=h.author_id,
                created_at=h.created_at,
                updated_at=h.updated_at,
                score=h.score,
                rank=i + 1,
                matched_via=[
                    MatchProvenance(
                        channel="bm25",  # SQL list path is closest to BM25
                        rank=i + 1,
                        score=h.score,
                        intent_idx=intent_idx,
                    )
                ],
                chunk_count=1,
                retriever_scores={"sql": h.score},
                chunks=[
                    QueryChunk(
                        chunk_id=h.chunk_id,
                        content=h.content,
                        score=h.score,
                        rank_in_doc=1,
                        retriever_scores={"sql": h.score},
                        graph_evidence=[],
                    )
                ],
            )
            for i, h in enumerate(hits[: req.top_k])
        ]

    timing["sql_ms"] = (time.perf_counter() - t_sql) * 1000

    # `related_entities` walk: same shape as search_pipeline. Skip entirely
    # when the response is an aggregation (count / group_by) -- there are no
    # result docs to walk from (codex-B2). Three-state contract per codex-B4
    # is preserved: None = not requested OR walk failed OR aggregation mode;
    # [] = walked, no neighbors; [...] = walked, neighbors found.
    related: list[RelatedEntity] | None = None
    related_error: str | None = None
    if aggregation is None and req.top_k_related > 0:
        # Fuzzy exclusion (codex-P2): see search_pipeline.run_search for
        # rationale. Threshold + normalized variants.
        exclude_keys = build_exclude_node_keys(
            intent.entities,
            entity_match_threshold=req.entity_match_threshold,
        )
        # Phase 2: translate alias canonical_ids to primaries so the walker
        # doesn't recommend the cluster the user just typed.
        exclude_keys = await expand_exclude_keys_with_aliases(
            customer_id,
            intent.entities,
            exclude_keys,
            entity_match_threshold=req.entity_match_threshold,
        )
        # Dedupe doc_id, keep best (lowest) rank per doc -- list mode emits
        # one doc per result by construction, but the dedupe is cheap insurance.
        best_rank: dict[str, int] = {}
        for i, d in enumerate(document_results, start=1):
            best_rank.setdefault(d.doc_id, i)
        ranked_docs = sorted(best_rank.items(), key=lambda kv: kv[1])
        t_related = time.perf_counter()
        try:
            related = await walk_result_doc_neighbors(
                customer_id,
                ranked_result_docs=ranked_docs,
                exclude_node_keys=exclude_keys,
                min_confidence=req.min_confidence,
                top_n=req.top_k_related,
            )
        except Exception as exc:
            # Error name flows back via the dedicated `related_entities_error`
            # response field. Do NOT inject a sentinel into timing_ms --
            # dashboards parse that dict as stage durations.
            log.warning(
                "related_entities walk failed", exc_info=exc, trace_id=trace_id
            )
            related = None
            related_error = type(exc).__name__
        timing["related_entities_ms"] = (time.perf_counter() - t_related) * 1000

    # results: list[QueryResult] -- list pipeline only emits Documents
    # today. Entity surfacing is a search-path feature; list path stays
    # narrow on purpose (deterministic SQL window/aggregate semantics).
    results: list[QueryResult] = list(document_results)

    return RetrieveResponse(
        query=req.query,
        results=results,
        total_candidates=total_candidates,
        router_hit_cache=False,
        confidence_breakdown={"EXTRACTED": 0, "INFERRED": 0, "AMBIGUOUS": 0},
        applied_temporal=temporal_meta,
        applied_sort=sort_meta,
        applied_entity_filter=None,
        applied_mode="list",
        applied_doc_types=doc_types,
        extracted_entities=extracted_entities,
        aggregation=aggregation,
        timing_ms=timing,
        trace_id=trace_id,
        related_entities=related,
        related_entities_error=related_error,
    )
