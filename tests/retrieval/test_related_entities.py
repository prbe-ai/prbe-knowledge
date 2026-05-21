"""Integration tests for `walk_result_doc_neighbors`.

Real Postgres (no DB mocks per `feedback_no_real_cli_in_tests.md` and the
project's no-mock-DB rule for retrieval). Mirrors the seeding template
from `tests/retrieval/test_entity_filter.py:_seed_doc_with_repo_link`.

Cases covered (per locked plan section 6):
1. Two docs each linked to the same Repo -> doc_count=2
2. Routed-entity exclusion via (label, canonical_id) tuple
3. (label, canonical_id) collision -- single-key exclusion would over-exclude
4. min_confidence='EXTRACTED' drops INFERRED-only edges via SQL HAVING
5. top_n caps the response
6. Empty input returns [] without a SQL call
7. max_confidence_rank -> tier name round-trip across all tiers + NULL
8. associated_doc_ids ordering + DISTINCT
9. valid_to filter excludes closed edges
10. Non-ASCII canonical_id passes through unchanged
11. IDF ranking puts specific entity above generic
12. RLS isolation (tenant A walking with tenant B's doc IDs returns empty)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from services.retrieval.retrievers.related_entities import (
    _RANK_TO_CONFIDENCE,
    _confidence_case_sql,
    build_exclude_node_keys,
    walk_result_doc_neighbors,
)
from shared.config import Settings, get_settings
from shared.constants import EdgeType, NodeLabel
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---- seed helpers ---------------------------------------------------------


async def _seed_customer(customer_id: str) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )


async def _seed_doc_node(
    customer_id: str,
    *,
    doc_id: str,
    title: str = "doc",
    updated_at: datetime | None = None,
) -> None:
    """Seed a minimal documents row + matching Document graph_node.

    Source/doc_type: github.commit (matches an existing seeder pattern;
    semantics don't matter for the walk -- the retriever only joins on
    graph_nodes, not documents). `documents.source_id` follows the
    `<kind>:<uuid>` convention per memory `feedback_documents_source_id_format.md`.
    """
    if updated_at is None:
        updated_at = datetime(2026, 4, 28, 12, 0, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $1, 1, $2,
                'github', $3, 'https://example/' || $1,
                'raw_source', 'github.commit', 'text/plain',
                'h-' || $1, $4, 100, 0,
                $5, $5, $5, $5, '{}'::jsonb
            )
            """,
            doc_id, customer_id, f"commit:{doc_id}", title, updated_at,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1, $2, $3, 0, $4, $5, 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1
            )
            """,
            f"{doc_id}:c0", doc_id, customer_id,
            f"body of {doc_id}", f"chash-{doc_id}",
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, '{}'::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, NodeLabel.DOCUMENT.value, doc_id,
        )


async def _seed_neighbor_node(
    customer_id: str,
    *,
    label: str,
    canonical_id: str,
    name: str | None = None,
) -> None:
    properties_json = "{}" if name is None else f'{{"name": "{name}"}}'
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES ($1, $2, $3, $4::jsonb)
            ON CONFLICT (customer_id, label, canonical_id) DO NOTHING
            """,
            customer_id, label, canonical_id, properties_json,
        )


async def _seed_edge(
    customer_id: str,
    *,
    from_label: str,
    from_canonical_id: str,
    to_label: str,
    to_canonical_id: str,
    edge_type: str = EdgeType.MENTIONS.value,
    confidence: str = "EXTRACTED",
    valid_to: datetime | None = None,
) -> None:
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                confidence, valid_from, valid_to
            )
            SELECT $1, $2, f.node_id, t.node_id, $7, NOW(), $8
            FROM graph_nodes f, graph_nodes t
            WHERE f.customer_id = $1 AND f.label = $3 AND f.canonical_id = $4
              AND t.customer_id = $1 AND t.label = $5 AND t.canonical_id = $6
            ON CONFLICT DO NOTHING
            """,
            customer_id,
            edge_type,
            from_label,
            from_canonical_id,
            to_label,
            to_canonical_id,
            confidence,
            valid_to,
        )


# ---- pure-Python helpers (no DB) -----------------------------------------
# These two tests are sync; the module-level pytestmark applies the asyncio
# marker but pytest-asyncio's auto mode tolerates non-async functions
# (warns but doesn't fail). Splitting them into a separate file would
# be cleaner; keeping them here is intentional for cohesion.


def test_confidence_case_sql_generated_from_dict() -> None:
    """The CASE expression must include exactly the tiers in `_CONFIDENCE_RANK`,
    with the EXTRACTED rank as the legacy-NULL fallback. Tripwire for any
    drift between SQL/Python."""
    sql = _confidence_case_sql("e.confidence")
    assert "CASE e.confidence" in sql
    for tier in ("AMBIGUOUS", "INFERRED", "EXTRACTED"):
        assert f"WHEN '{tier}'" in sql
    # ELSE 2 (the rank for EXTRACTED) -- legacy NULL handling.
    assert "ELSE 2" in sql


def test_rank_to_confidence_inverse_complete() -> None:
    """Every rank int produced by the SQL CASE must map back to a tier
    name. Used by the Python wrapper after the fetch."""
    assert _RANK_TO_CONFIDENCE == {0: "AMBIGUOUS", 1: "INFERRED", 2: "EXTRACTED"}


# ---- build_exclude_node_keys (fuzzy match, codex-P2) ---------------------


class _FakeRouterEntity:
    """Stand-in for `services.retrieval.router.RouterEntity` -- tests don't
    need the dataclass machinery, just the duck-typed attrs.
    """
    def __init__(
        self,
        entity_type: str,
        canonical_id: str,
        display_name: str = "",
        confidence: float = 1.0,
    ) -> None:
        self.entity_type = entity_type
        self.canonical_id = canonical_id
        self.display_name = display_name
        self.confidence = confidence


def test_build_exclude_node_keys_below_threshold_drops_entity() -> None:
    """Entities below `entity_match_threshold` MUST NOT contribute to the
    exclude set -- a low-confidence router misfire should not suppress a
    real related entity (codex-P2)."""
    entities = [
        _FakeRouterEntity("service", "prbe-backend", confidence=0.4),
        _FakeRouterEntity("ticket", "PRB-17", confidence=0.95),
    ]
    keys = build_exclude_node_keys(entities, entity_match_threshold=0.7)
    # The ticket clears the threshold; the service does not.
    labels = {label for label, _ in keys}
    assert "Ticket" in labels
    assert "Service" not in labels


def test_build_exclude_node_keys_emits_namespace_stripped_variant() -> None:
    """Router emits `prbe-ai/prbe-backend`; exclude set must include both
    the full form AND the namespace-stripped form so the SQL can match
    graph nodes stored under either canonical_id shape (codex-P2)."""
    entities = [
        _FakeRouterEntity("repo", "prbe-ai/prbe-backend", confidence=0.95),
    ]
    keys = build_exclude_node_keys(entities)
    assert ("Repo", "prbe-ai/prbe-backend") in keys
    assert ("Repo", "prbe-backend") in keys  # namespace stripped


def test_build_exclude_node_keys_uses_display_name_fallback() -> None:
    """When the router emits a different canonical_id than the graph
    stores, the display_name is the fallback bridge. Both forms must
    appear in the exclude set."""
    entities = [
        _FakeRouterEntity(
            "service", "svc-canon", display_name="prbe-backend", confidence=0.95,
        ),
    ]
    keys = build_exclude_node_keys(entities)
    assert ("Service", "svc-canon") in keys  # raw canonical
    assert ("Service", "prbe-backend") in keys  # via display_name


def test_build_exclude_node_keys_lowercases_inputs() -> None:
    """Strings emitted into the exclude set are lowercased so the SQL's
    `lower(...)` comparison on the candidate side matches regardless of
    case differences between router and graph."""
    entities = [
        _FakeRouterEntity(
            "repo", "PRBE-AI/Prbe-Backend", display_name="Prbe-Backend",
            confidence=0.95,
        ),
    ]
    keys = build_exclude_node_keys(entities)
    # All emitted strings are lowercased.
    for _, val in keys:
        assert val == val.lower()
    assert ("Repo", "prbe-ai/prbe-backend") in keys
    assert ("Repo", "prbe-backend") in keys


def test_build_exclude_node_keys_unknown_entity_type_silently_dropped() -> None:
    """Unknown entity_types (not in `_ENTITY_TO_LABEL`) are skipped --
    we have no label to attach the exclusion to. Same behavior as the
    pre-fuzzy-match wire-in."""
    entities = [
        _FakeRouterEntity("alien", "weird", confidence=0.95),
    ]
    keys = build_exclude_node_keys(entities)
    assert keys == set()


# ---- behavior tests ------------------------------------------------------


async def test_empty_input_returns_empty_list_no_sql_call() -> None:
    """Case 6: ranked_result_docs=[] -> [] without any DB round trip.

    The three-state contract on QueryResponse.related_entities distinguishes
    [] (walked, no neighbors) from None (not requested OR walk failed).
    walk_result_doc_neighbors returns [] for the empty-input case so the
    pipeline can still expose `[]` to callers.
    """
    # No live_db fixture -- this path must short-circuit before any pool use.
    out = await walk_result_doc_neighbors(
        "any-tenant",
        ranked_result_docs=[],
        exclude_node_keys=set(),
    )
    assert out == []


async def test_two_docs_share_one_repo_doc_count_two(live_db) -> None:
    """Case 1: two result docs each linked to Repo:prbe-backend.

    Expected: one RelatedEntity for prbe-backend with doc_count=2,
    ranked above any singleton on score (IDF puts low-global-frequency
    entities up; we don't seed competing globals here so it's the only one).
    """
    cust = "cust-related-1"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_doc_node(cust, doc_id="doc:2")
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value,
        canonical_id="prbe-ai/prbe-backend", name="prbe-backend",
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="prbe-ai/prbe-backend",
        edge_type=EdgeType.TOUCHES.value,
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:2",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="prbe-ai/prbe-backend",
        edge_type=EdgeType.TOUCHES.value,
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1), ("doc:2", 2)],
        exclude_node_keys=set(),
    )
    assert len(out) == 1
    repo = out[0]
    assert repo.canonical_id == "prbe-ai/prbe-backend"
    assert repo.label == NodeLabel.DOCUMENT.value
    assert repo.display_name == "prbe-backend"
    assert repo.doc_count == 2
    assert EdgeType.TOUCHES.value in repo.edge_types
    assert repo.max_confidence == "EXTRACTED"
    assert set(repo.associated_doc_ids) == {"doc:1", "doc:2"}


async def test_routed_entity_excluded_via_label_canonical_id(live_db) -> None:
    """Case 2: an entity present in the result-set graph but also in
    `exclude_node_keys` (label, canonical_id) tuple does NOT surface."""
    cust = "cust-related-exclude"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_neighbor_node(
        cust, label=NodeLabel.SERVICE.value,
        canonical_id="prbe-backend", name="prbe-backend",
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.SERVICE.value, to_canonical_id="prbe-backend",
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys={(NodeLabel.SERVICE.value, "prbe-backend")},
    )
    assert out == []


async def test_fuzzy_namespace_match_excludes_namespaced_canonical_id(
    live_db,
) -> None:
    """codex-P2: router emits bare `prbe-backend` but graph stores
    `prbe-ai/prbe-backend`. The SQL exclusion compares against the
    namespace-stripped form on the candidate side so the exclusion still
    fires even though canonical_ids don't exact-match."""
    cust = "cust-related-fuzzy-ns"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    # Graph stores the namespaced form.
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value,
        canonical_id="prbe-ai/prbe-backend", name="prbe-backend",
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="prbe-ai/prbe-backend",
    )

    # Router emits the bare form.
    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys={(NodeLabel.DOCUMENT.value, "prbe-backend")},
    )
    assert out == []  # Excluded via namespace-stripped match.


async def test_fuzzy_match_excludes_via_display_name(live_db) -> None:
    """codex-P2: router emits a canonical_id that doesn't match the graph,
    but the graph node's display_name (properties->>'name') does. The SQL
    should compare against display_name on the candidate side."""
    cust = "cust-related-fuzzy-display"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_neighbor_node(
        cust, label=NodeLabel.SERVICE.value,
        canonical_id="svc-uuid-9f3c", name="prbe-backend",  # display_name
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.SERVICE.value, to_canonical_id="svc-uuid-9f3c",
    )

    # Exclude via display_name string. Caller would have built this from
    # the routed entity's display_name field via build_exclude_node_keys.
    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys={(NodeLabel.SERVICE.value, "prbe-backend")},
    )
    assert out == []  # Excluded via display_name match.


async def test_fuzzy_match_case_insensitive(live_db) -> None:
    """codex-P2: case mismatch between router and graph (e.g. graph stores
    `Prbe-Backend`, router emits lowercase) must not break exclusion."""
    cust = "cust-related-fuzzy-case"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_neighbor_node(
        cust, label=NodeLabel.SERVICE.value,
        canonical_id="Prbe-Backend", name="Prbe-Backend",
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.SERVICE.value, to_canonical_id="Prbe-Backend",
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys={(NodeLabel.SERVICE.value, "prbe-backend")},
    )
    assert out == []


async def test_label_canonical_id_collision_only_excludes_matching_label(
    live_db,
) -> None:
    """Case 3 (codex-A4): seed Person:mahit AND ServiceCard:mahit (same
    canonical_id, different labels). Excluding only Person:mahit must
    leave ServiceCard:mahit visible -- single-key exclusion would over-exclude.
    """
    cust = "cust-related-collision"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_neighbor_node(
        cust, label=NodeLabel.PERSON.value, canonical_id="mahit", name="Mahit P.",
    )
    await _seed_neighbor_node(
        cust, label=NodeLabel.SERVICE_CARD.value, canonical_id="mahit", name="Mahit's card",
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.PERSON.value, to_canonical_id="mahit",
        edge_type=EdgeType.AUTHORED.value,
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.SERVICE_CARD.value, to_canonical_id="mahit",
        edge_type=EdgeType.MENTIONS.value,
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys={(NodeLabel.PERSON.value, "mahit")},
    )
    labels = {(e.label, e.canonical_id) for e in out}
    assert (NodeLabel.SERVICE_CARD.value, "mahit") in labels
    assert (NodeLabel.PERSON.value, "mahit") not in labels


async def test_min_confidence_extracted_drops_inferred_only_neighbors(
    live_db,
) -> None:
    """Case 4: an entity reached only via INFERRED edges does NOT surface
    when min_confidence='EXTRACTED'. The HAVING filter must run BEFORE
    LIMIT so EXTRACTED-tier neighbors aren't displaced by INFERRED-tier
    ones in the top-n race.
    """
    cust = "cust-related-confidence"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id="repo-extracted", name="A",
    )
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id="repo-inferred", name="B",
    )
    # Doc -> A (EXTRACTED), Doc -> B (INFERRED only)
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="repo-extracted",
        confidence="EXTRACTED",
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="repo-inferred",
        confidence="INFERRED",
    )

    # Default INFERRED floor: both surface.
    out_default = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys=set(),
        min_confidence="INFERRED",
    )
    cids_default = {e.canonical_id for e in out_default}
    assert {"repo-extracted", "repo-inferred"} <= cids_default

    # EXTRACTED floor: only the EXTRACTED-edged neighbor.
    out_strict = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys=set(),
        min_confidence="EXTRACTED",
        top_n=10,
    )
    cids_strict = {e.canonical_id for e in out_strict}
    assert "repo-extracted" in cids_strict
    assert "repo-inferred" not in cids_strict


async def test_top_n_caps_response(live_db) -> None:
    """Case 5: top_n=2 returns at most 2 entities even when more match."""
    cust = "cust-related-cap"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    for i in range(5):
        canonical = f"repo-{i}"
        await _seed_neighbor_node(
            cust, label=NodeLabel.DOCUMENT.value, canonical_id=canonical, name=canonical,
        )
        await _seed_edge(
            cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
            to_label=NodeLabel.DOCUMENT.value, to_canonical_id=canonical,
        )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys=set(),
        top_n=2,
    )
    assert len(out) == 2


@pytest.mark.parametrize(
    ("seeded_confidence", "expected_tier"),
    [
        ("EXTRACTED", "EXTRACTED"),
        ("INFERRED", "INFERRED"),
        ("AMBIGUOUS", "AMBIGUOUS"),
    ],
)
async def test_max_confidence_rank_round_trip(
    live_db, seeded_confidence, expected_tier,
) -> None:
    """Case 7: Each seeded tier round-trips through the SQL CASE -> rank int
    -> Python wrapper -> tier name correctly.
    """
    cust = f"cust-confidence-{seeded_confidence.lower()}"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id="repo", name="r",
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="repo",
        confidence=seeded_confidence,
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys=set(),
        # `min_confidence=None` accepts every tier (debug mode in the docstring).
        min_confidence=None,
    )
    assert len(out) == 1
    assert out[0].max_confidence == expected_tier


async def test_associated_doc_ids_ordering_and_distinct(live_db) -> None:
    """Case 8 (codex-A2): an entity attached to docs at ranks [3, 1, 7]
    with TWO edges from doc-rank-1. samples must be ordered by rank with
    no duplicate doc_ids: [doc-rank-1, doc-rank-3, doc-rank-7].
    """
    cust = "cust-related-order"
    await _seed_customer(cust)
    # Doc IDs picked so lexicographic order != rank order, to make sure
    # ranking comes from the rank int, not from lex sort.
    await _seed_doc_node(cust, doc_id="doc-z")  # rank 1
    await _seed_doc_node(cust, doc_id="doc-m")  # rank 3
    await _seed_doc_node(cust, doc_id="doc-a")  # rank 7
    await _seed_neighbor_node(
        cust, label=NodeLabel.PERSON.value, canonical_id="mahit", name="Mahit",
    )
    # doc-z (rank 1) has TWO edges to mahit -- AUTHORED + MENTIONS.
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-z",
        to_label=NodeLabel.PERSON.value, to_canonical_id="mahit",
        edge_type=EdgeType.AUTHORED.value,
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-z",
        to_label=NodeLabel.PERSON.value, to_canonical_id="mahit",
        edge_type=EdgeType.MENTIONS.value,
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-m",
        to_label=NodeLabel.PERSON.value, to_canonical_id="mahit",
        edge_type=EdgeType.MENTIONS.value,
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc-a",
        to_label=NodeLabel.PERSON.value, to_canonical_id="mahit",
        edge_type=EdgeType.MENTIONS.value,
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc-z", 1), ("doc-m", 3), ("doc-a", 7)],
        exclude_node_keys=set(),
    )
    assert len(out) == 1
    person = out[0]
    assert person.canonical_id == "mahit"
    assert person.doc_count == 3
    # Cap=3, no duplicate doc-z, ordered by rank.
    assert person.associated_doc_ids == ["doc-z", "doc-m", "doc-a"]


async def test_valid_to_filter_excludes_closed_edges(live_db) -> None:
    """Case 9 (codex-A3): an edge with valid_to in the past is NOT walked.

    Without this filter, stale graph relationships would surface as crawl
    candidates. Mirrors the same filter on `services/retrieval/retrievers/sql.py`.
    """
    cust = "cust-related-validto"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id="#left", name="left",
    )
    yesterday = datetime.now(UTC) - timedelta(days=1)
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="#left",
        edge_type=EdgeType.MEMBER_OF.value,
        valid_to=yesterday,
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys=set(),
    )
    assert out == []


async def test_non_ascii_canonical_id_passes_through(live_db) -> None:
    """Case 10: UTF-8 canonical IDs (Slack/Notion display names with emoji /
    accented chars) survive the round trip unchanged.
    """
    cust = "cust-related-utf8"
    await _seed_customer(cust)
    await _seed_doc_node(cust, doc_id="doc:1")
    canonical = "channel-équipe-\U0001f680"  # accented + rocket emoji
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id=canonical, name=canonical,
    )
    await _seed_edge(
        cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id="doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id=canonical,
    )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[("doc:1", 1)],
        exclude_node_keys=set(),
    )
    assert len(out) == 1
    assert out[0].canonical_id == canonical
    assert out[0].display_name == canonical


async def test_idf_ranking_promotes_specific_over_generic(live_db) -> None:
    """Case 11 (codex-C1): neighbor A attached to ALL 5 result docs but to
    1000 other tenant docs globally; neighbor B attached to only 2 result
    docs but only 5 globally.

    score = doc_count_in_results / log(1 + global_doc_count)
    A: 5 / log(1 + 1005) ~= 5 / 6.9 ~= 0.72
    B: 2 / log(1 + 7)    ~= 2 / 2.08 ~= 0.96

    B must rank above A despite the lower flat doc_count -- the whole
    point of IDF.
    """
    cust = "cust-related-idf"
    await _seed_customer(cust)

    # Result-set docs: 5 of them.
    for i in range(5):
        await _seed_doc_node(cust, doc_id=f"result:{i}")
    # Global filler docs (NOT in the result set) -- 1000 of them.
    for i in range(1000):
        await _seed_doc_node(cust, doc_id=f"global:{i}")

    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id="#engineering", name="generic",
    )
    await _seed_neighbor_node(
        cust, label=NodeLabel.DOCUMENT.value, canonical_id="pr:42", name="specific",
    )

    # Generic A: attached to ALL 5 result docs + ALL 1000 global docs.
    for i in range(5):
        await _seed_edge(
            cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id=f"result:{i}",
            to_label=NodeLabel.DOCUMENT.value, to_canonical_id="#engineering",
        )
    for i in range(1000):
        await _seed_edge(
            cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id=f"global:{i}",
            to_label=NodeLabel.DOCUMENT.value, to_canonical_id="#engineering",
        )

    # Specific B: attached to only 2 result docs + 5 global docs.
    for i in range(2):
        await _seed_edge(
            cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id=f"result:{i}",
            to_label=NodeLabel.DOCUMENT.value, to_canonical_id="pr:42",
        )
    for i in range(5):
        await _seed_edge(
            cust, from_label=NodeLabel.DOCUMENT.value, from_canonical_id=f"global:{i}",
            to_label=NodeLabel.DOCUMENT.value, to_canonical_id="pr:42",
        )

    out = await walk_result_doc_neighbors(
        cust,
        ranked_result_docs=[(f"result:{i}", i + 1) for i in range(5)],
        exclude_node_keys=set(),
        top_n=10,
    )
    # Both surface; B (specific) must rank above A (generic).
    cids = [e.canonical_id for e in out]
    assert "pr:42" in cids
    assert "#engineering" in cids
    assert cids.index("pr:42") < cids.index("#engineering")
    # And the score field actually reflects what the LLM sees.
    pr_score = next(e.score for e in out if e.canonical_id == "pr:42")
    chan_score = next(e.score for e in out if e.canonical_id == "#engineering")
    assert pr_score > chan_score


async def test_rls_isolation_does_not_leak_other_tenant_neighbors(live_db) -> None:
    """Case 12 (CRITICAL): seed two tenants A and B, each with docs +
    graph_nodes + graph_edges. Run as tenant A, pass tenant B's doc_ids
    in `ranked_result_docs`. Response must be empty (no tenant B
    entities leak). Catches any accidental drop of `with_tenant()`.
    """
    await _seed_customer("tenant-A")
    await _seed_customer("tenant-B")

    # Tenant B has a doc with an attached entity.
    await _seed_doc_node("tenant-B", doc_id="b-doc:1")
    await _seed_neighbor_node(
        "tenant-B", label=NodeLabel.DOCUMENT.value,
        canonical_id="b-secret-repo", name="secret",
    )
    await _seed_edge(
        "tenant-B", from_label=NodeLabel.DOCUMENT.value, from_canonical_id="b-doc:1",
        to_label=NodeLabel.DOCUMENT.value, to_canonical_id="b-secret-repo",
    )

    # Tenant A walks. Even though we pass tenant B's doc_id, the RLS
    # policy on graph_nodes only lets us see customer_id='tenant-A' rows --
    # the doc_anchors CTE can't resolve b-doc:1 because no tenant-A row
    # has that canonical_id.
    out = await walk_result_doc_neighbors(
        "tenant-A",
        ranked_result_docs=[("b-doc:1", 1)],
        exclude_node_keys=set(),
    )
    assert out == []
