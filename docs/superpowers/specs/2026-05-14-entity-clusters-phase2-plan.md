# Entity Clusters — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `prbe-knowledge` retrieval cluster-aware so that post-Phase-1 merged entities (e.g. `richardwei6` ⇄ `mahit@prbe.ai` ⇄ `U07ABC123`) behave correctly when surfaced via `/graph/explore`, `/query` list-mode, and `/query` search-mode.

**Architecture:** Three retrieval entry points consult `entity_aliases` (populated by Phase 1's merge endpoint) to translate user-typed alias canonical_ids to their primary, expand author filters across cluster members, and stamp `RelatedEntity` results with `member_count` / `member_sources` for agent-facing metadata. A shared helper module hosts the alias-resolution primitives so all three sites share the same batching pattern. The B-promote design guarantees alias `graph_nodes` are hard-deleted at merge — so walkers and joins naturally see one node per cluster (the primary). We do **not** need anchor-expansion or per-row alias rewrites in walker SQL; we only need (a) translate inputs at the boundary, (b) enrich result rows from `entity_aliases` + `graph_node_provenance` + `entity_cluster_metadata`, (c) expand `author_ids` because `documents.author_id` is historical raw text.

**Tech Stack:** Python 3.12, FastAPI, asyncpg, Pydantic v2, Postgres with RLS, pytest + pytest-asyncio. All queries run under `with_tenant(customer_id)` which sets `app.current_customer_id` GUC.

**Branch:** `entity-clusters-phase2` (stacked on `entity-clusters-phase1`). Once Phase 1 PRs #265 and #266 merge to `main`, rebase this branch onto `main` before opening the Phase 2 PR.

---

## Scope (locked from design doc §"Read-side behavior (Phase 2 preview)")

1. **Graph anchor lookup translation** — at `/graph/explore?mode=anchor`, translate `anchor_node_id` via `entity_aliases` before `anchor_exists()` so users typing an alias canonical_id reach the cluster.
2. **Author filter expansion (list mode)** — at `run_list`, expand each `author_id` to the union of its cluster members (primary + all aliases) because `documents.author_id` is never rewritten.
3. **`RelatedEntity.member_count` + `member_sources`** — populated by the walker from `entity_aliases` (count) and `graph_node_provenance` of the primary (distinct source_systems).
4. **`entity_cluster_metadata.display_name` override** — when a primary has a curated display name set, prefer it over `graph_nodes.properties->>'name'`. Applies to both `RelatedEntity` (related-entities walker) and `QueryEntityResult` (search-pipeline entity hits).
5. **`exclude_node_keys` translation** — when the router extracts an entity the user typed as `mahit@prbe.ai`, the exclude key should also exclude the primary `richardwei6` so the walker doesn't crawl back to the typed entity.
6. **Routed-entity translation (search pipeline)** — when `routed.entities` contains an alias canonical_id, translate to primary so the `QueryEntityResult` hit lands on the cluster's primary node (which is the only node remaining post-merge).

**Anti-scope:**
- No changes to `services/ingestion/graph_writer.py` (Phase 1 locked alias-resolution there).
- No changes to `/api/entity-clusters/*` endpoints.
- No cluster-aware edge aggregation (e.g. summing `commit_count` across lanes) — that's Phase 2.5+ per design doc Open Items #4.
- No schema changes.

---

## Tech decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| `member_count` semantics | Total cluster size (primary + aliases). Unmerged node = 1. | UX-natural; reads as "this entity has 3 identities" rather than "2 aliases on top of the one you can see". |
| `member_sources` semantics | DISTINCT `source_system` from `graph_node_provenance` of primary's `node_id`. | Phase 1's merge consolidates alias provenance into the primary's row (`ON CONFLICT … DO UPDATE` min/max). Single query gets the whole cluster. |
| Author-filter scope | Person-label only. | `documents.author_id` is exclusively a person canonical_id (Slack user, GitHub login, Linear user, etc.). |
| Always-on, no feature flag | Yes. | Pre-merge `entity_aliases` is empty → behavior is identical to today. No regression risk; no need to gate. |
| Anchor translation label-scoping | None — match across all labels (mirror current `anchor_exists` semantics). | Canonical_ids are effectively label-unique per tenant; over-translating across labels is a non-issue in practice and a tiny risk vs. the cost of label inference at the endpoint. |
| Display-name override fallback | `COALESCE(NULLIF(ecm.display_name, ''), gn.properties->>'name')` | Empty-string override must fall through (treat as "no override"). |
| Helper module | `services/retrieval/helpers.py` (existing). | Already houses cross-cutting retrieval helpers. |
| Helper signature | `async def resolve_aliases(conn, customer_id, refs: list[tuple[str, str]]) -> dict[tuple[str, str], str]` mirroring `services/ingestion/graph_writer.py:_fetch_aliases`. | Established pattern; one bulk SELECT per call. |
| Expansion helper signature | `async def expand_to_cluster_members(conn, customer_id, label, canonical_ids: list[str]) -> dict[str, list[str]]` returning `{input_id: [member_id, ...]}` where each input maps to its cluster's full member list. | Caller can flatten + dedup; keeps the contract per-input clear. |

---

## File structure

| File | Responsibility | Phase 2 change |
|---|---|---|
| `services/retrieval/helpers.py` | Cross-cutting retrieval helpers (existing: `apply_entity_filter`, `embeddings_for_chunks`). | + `resolve_aliases()`, + `expand_to_cluster_members()` |
| `shared/models.py` | Pydantic models for the retrieval API. | + `member_count`, `member_sources` on `RelatedEntity` |
| `services/retrieval/main.py` | FastAPI endpoints. | Translate `anchor_node_id` before `anchor_exists()` (lines 596-600) |
| `services/retrieval/list_pipeline.py` | List-mode dispatcher. | Expand `author_ids` through cluster members after line 147 |
| `services/retrieval/retrievers/related_entities.py` | Walker: doc-neighbors aggregation SQL + Python builder. | Add `entity_aliases` / `graph_node_provenance` / `entity_cluster_metadata` LEFT JOINs; populate new model fields; translate `exclude_node_keys` |
| `services/retrieval/search_pipeline.py` | Search-mode dispatcher; `QueryEntityResult` builder (lines 640-790). | Translate `(label, canonical_id)` input pairs through aliases; add `entity_cluster_metadata` LEFT JOIN; use override at line 760 |
| `tests/retrieval/test_helpers_alias_resolution.py` | NEW. | Unit tests for `resolve_aliases` + `expand_to_cluster_members` |
| `tests/retrieval/test_graph_explore_alias_anchor.py` | NEW. | Endpoint-level test: alias anchor resolves to primary's graph |
| `tests/retrieval/test_list_pipeline_author_alias.py` | NEW. | Author filter expansion test through list pipeline |
| `tests/retrieval/test_related_entities_clusters.py` | NEW. | Walker `member_count` / `member_sources` / display-name override / exclude translation |
| `tests/retrieval/test_search_pipeline_entity_clusters.py` | NEW. | Routed-entity translation + display-name override in `QueryEntityResult` |

---

## Prerequisites

- Worktree: `/Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase2` on branch `entity-clusters-phase2`.
- Python 3.12 venv installed: `cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase2 && python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'`.
- Local Postgres + MinIO running via `docker compose up -d` from the worktree root.
- `INTERNAL_KNOWLEDGE_API_KEY=test-internal-key` in `.env` (also `KNOWLEDGE_BASE_URL` etc. — copy from worktree's `.env.example`).
- Database migrated to head: `.venv/bin/alembic -c db/alembic.ini upgrade head` (Phase 1 migration `20260514_0071_entity_clusters` already applied since this branch descends from `entity-clusters-phase1`).
- Pytest runs scoped to `tests/`: `.venv/bin/pytest tests/retrieval/<file> -v` (the `legacy/` directory has its own conftest that collides if you pass `pytest` without a scope).

---

## Execution chunks (subagent dispatch roadmap)

Tasks below are split into 12 logical chunks, each scoped to a single concern + single reviewable commit. The chunk index drives subagent dispatch; chunks within the same parent Task share file context but produce independent commits.

| # | Chunk | Files | Parent Task |
|---|---|---|---|
| 1 | `resolve_aliases` helper | `helpers.py` + 1 test file | Task 1 (helpers) |
| 2 | `expand_to_cluster_members` helper | `helpers.py` + 1 test file | Task 1 (helpers) |
| 3 | `RelatedEntity` model fields | `shared/models.py` | Task 2 |
| 4 | `/graph/explore` anchor translation | `services/retrieval/main.py` + endpoint test | Task 3 |
| 5 | List-pipeline author filter expansion | `services/retrieval/list_pipeline.py` + integration test | Task 4 |
| 6 | Walker cluster metadata (`member_count` + `member_sources`) | `related_entities.py` SQL + Python builder | Task 5 (walker) |
| 7 | Walker display-name override | `related_entities.py` SQL + Python builder | Task 5 (walker) |
| 8 | `exclude_node_keys` translation at caller | caller of walker (search_pipeline / pipeline) | Task 5 (walker) |
| 9 | Search-pipeline routed-entity translation | `search_pipeline.py` `_build_entity_results` translation block | Task 6 |
| 10 | Search-pipeline display-name override | `search_pipeline.py` SQL LEFT JOIN + Python builder | Task 6 |
| 11 | Final pass — full suite + lint/types + docs cross-link | (no code) | Task 7 |
| 12 | Live container smoke test | `scripts/smoke_phase2_clusters.py` | Task 8 |

**Dispatch protocol:** one implementer subagent per chunk, then spec-compliance reviewer → fix loop → code-quality reviewer → fix loop → mark complete → next chunk. After Chunk 11 passes, dispatch one final code-reviewer for the entire branch. Chunks 6, 7, 8 all modify the walker SQL — they will re-edit the same CTE block on consecutive commits; this is intentional and not churn (each commit isolates one logical concern). Chunks 9 and 10 likewise modify `_build_entity_results` on consecutive commits.

---

## Tasks

### Task 1: Alias-resolution helpers

**Files:**
- Modify: `services/retrieval/helpers.py` (append after `embeddings_for_chunks` at line 80)
- Create: `tests/retrieval/test_helpers_alias_resolution.py`

**Context:** Two helpers consumed by Tasks 3-6. `resolve_aliases` mirrors `services/ingestion/graph_writer.py:_fetch_aliases` (forward direction, alias → primary). `expand_to_cluster_members` does the inverse: for each input id, return its cluster's full member list (primary + all aliases). Both batch into one SELECT each.

- [ ] **Step 1: Write failing tests for `resolve_aliases`**

Create `tests/retrieval/test_helpers_alias_resolution.py`:

```python
"""Unit tests for retrieval-side alias helpers (Phase 2).

Real Postgres (no DB mocks for retrieval per project convention). The
``live_db`` fixture truncates between tests. Helpers run under
``with_tenant(customer_id)`` because they query RLS-protected tables.
"""

from __future__ import annotations

import pytest

from services.retrieval.helpers import (
    expand_to_cluster_members,
    resolve_aliases,
)
from shared.config import Settings, get_settings
from shared.db import raw_conn, with_tenant
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


CUSTOMER_ID = "alias-helpers-cust"
PRIMARY = "richardwei6"
ALIAS_A = "mahit@prbe.ai"
ALIAS_B = "U07ABC123"


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


async def _seed_audit(customer_id: str) -> str:
    """Create an audit row needed by entity_aliases FK. Returns merge_id."""
    async with raw_conn() as conn:
        row = await conn.fetchrow(
            """
            INSERT INTO entity_merge_audit (
                customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id,
                status
            )
            VALUES ($1, 'Person', $2, ARRAY[$3, $4]::text[],
                    '11111111-1111-1111-1111-111111111111', 'active')
            RETURNING merge_id
            """,
            customer_id, PRIMARY, ALIAS_A, ALIAS_B,
        )
    return str(row["merge_id"])


async def _seed_aliases(customer_id: str, merge_id: str) -> None:
    async with raw_conn() as conn:
        await conn.executemany(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            [
                (customer_id, ALIAS_A, PRIMARY, merge_id),
                (customer_id, ALIAS_B, PRIMARY, merge_id),
            ],
        )


async def test_resolve_aliases_returns_primary_for_known_aliases(live_db):
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await resolve_aliases(
            conn, CUSTOMER_ID,
            refs=[("Person", ALIAS_A), ("Person", ALIAS_B)],
        )
    assert out == {("Person", ALIAS_A): PRIMARY, ("Person", ALIAS_B): PRIMARY}


async def test_resolve_aliases_omits_non_aliases(live_db):
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await resolve_aliases(
            conn, CUSTOMER_ID,
            refs=[("Person", ALIAS_A), ("Person", "nobody"), ("Repo", "r1")],
        )
    # `nobody` and `Repo:r1` are not aliases — they're absent from the dict.
    assert out == {("Person", ALIAS_A): PRIMARY}


async def test_resolve_aliases_empty_input_returns_empty_dict(live_db):
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await resolve_aliases(conn, CUSTOMER_ID, refs=[])
    assert out == {}


async def test_resolve_aliases_is_tenant_scoped(live_db):
    """An alias in tenant A must NOT resolve when queried from tenant B."""
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    await _seed_customer("other-tenant")
    async with with_tenant("other-tenant") as conn:
        out = await resolve_aliases(
            conn, "other-tenant",
            refs=[("Person", ALIAS_A)],
        )
    assert out == {}


async def test_expand_to_cluster_members_returns_full_cluster(live_db):
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person",
            canonical_ids=[PRIMARY],
        )
    # Input was the primary -> cluster is {primary, alias_a, alias_b}.
    assert sorted(out[PRIMARY]) == sorted([PRIMARY, ALIAS_A, ALIAS_B])


async def test_expand_to_cluster_members_from_alias_input(live_db):
    """Querying with an alias id returns the same full cluster."""
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person",
            canonical_ids=[ALIAS_A],
        )
    assert sorted(out[ALIAS_A]) == sorted([PRIMARY, ALIAS_A, ALIAS_B])


async def test_expand_to_cluster_members_unmerged_id_returns_self(live_db):
    """An id that is neither a primary nor an alias maps to a singleton."""
    await _seed_customer(CUSTOMER_ID)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person",
            canonical_ids=["loner-id"],
        )
    assert out == {"loner-id": ["loner-id"]}


async def test_expand_to_cluster_members_mixed_input(live_db):
    """Mixed input: one alias, one primary, one unmerged. All collapse correctly."""
    await _seed_customer(CUSTOMER_ID)
    merge_id = await _seed_audit(CUSTOMER_ID)
    await _seed_aliases(CUSTOMER_ID, merge_id)
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person",
            canonical_ids=[ALIAS_A, PRIMARY, "loner-id"],
        )
    expected_cluster = sorted([PRIMARY, ALIAS_A, ALIAS_B])
    assert sorted(out[ALIAS_A]) == expected_cluster
    assert sorted(out[PRIMARY]) == expected_cluster
    assert out["loner-id"] == ["loner-id"]


async def test_expand_to_cluster_members_empty_input_returns_empty_dict(live_db):
    async with with_tenant(CUSTOMER_ID) as conn:
        out = await expand_to_cluster_members(
            conn, CUSTOMER_ID, "Person", canonical_ids=[],
        )
    assert out == {}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/retrieval/test_helpers_alias_resolution.py -v
```

Expected: ImportError on `from services.retrieval.helpers import (expand_to_cluster_members, resolve_aliases)`.

- [ ] **Step 3: Implement the helpers**

Append to `services/retrieval/helpers.py`:

```python
import asyncpg


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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/retrieval/test_helpers_alias_resolution.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add services/retrieval/helpers.py tests/retrieval/test_helpers_alias_resolution.py
git commit -m "feat(retrieval): alias resolution + cluster-member expansion helpers

Adds two helpers to services/retrieval/helpers.py:

- resolve_aliases: forward direction (alias → primary), mirrors the
  graph_writer._fetch_aliases pattern.
- expand_to_cluster_members: inverse + closure (any cluster member →
  list of all cluster members for that label).

Both are batch-friendly (one SELECT per call) and consumed by Phase 2's
anchor translation, author filter expansion, exclude-key translation,
and routed-entity translation."
```

---

### Task 2: `RelatedEntity` cluster fields on the shared model

**Files:**
- Modify: `shared/models.py:416-441` (the `RelatedEntity` Pydantic class)
- Test: covered by Tasks 5 + downstream — no isolated test file (the model class change has no behavior by itself; pydantic field additions are validated by Task 5's integration tests).

**Context:** Add `member_count: int = 1` and `member_sources: list[str] = Field(default_factory=list)` so the walker (Task 5) has somewhere to write the cluster metadata. Default `member_count=1` (unmerged node) ensures backward-compatible decoding of any pre-Phase-2 serialized payload.

- [ ] **Step 1: Add the fields**

Edit `shared/models.py` around line 425-441 — the `RelatedEntity` class. Change from:

```python
class RelatedEntity(BaseModel):
    """A non-Document graph node attached to >=1 doc in the result set.

    Surfaced to MCP consumers as crawl candidates: the LLM can drop the
    canonical_id into the next search_knowledge query bag to BFS the
    knowledge graph. Excludes any entity already in extracted_entities
    (the LLM has those handles already).
    """

    canonical_id: str
    label: str  # NodeLabel.value (Service, Repo, Person, Ticket, ...)
    display_name: str | None = None  # from properties->>'name'
    edge_types: list[str] = Field(default_factory=list)  # MENTIONS, AUTHORED, ...
    max_confidence: str  # EXTRACTED | INFERRED | AMBIGUOUS
    doc_count: int  # # of result-set docs adjacent to this entity (BFS priority)
    # IDF-adjusted score used for ranking. score = doc_count / log(1 +
    # global_doc_count). Generic high-degree entities (e.g.
    # Channel:#engineering attached to 10k docs) get crushed; specific
    # entities surface. Surfaced so LLMs can see the ranking signal.
    score: float
    # Up to 3 doc IDs the entity is attached to, ordered by result rank
    # (strongest first). Caps at 3 even when doc_count > 3 -- the LLM uses
    # these to ground/audit the doc_count claim against its visible chunks
    # list, not to enumerate every attached doc. DISTINCT -- multi-edge
    # docs do not duplicate.
    associated_doc_ids: list[str] = Field(default_factory=list)
```

To:

```python
class RelatedEntity(BaseModel):
    """A non-Document graph node attached to >=1 doc in the result set.

    Surfaced to MCP consumers as crawl candidates: the LLM can drop the
    canonical_id into the next search_knowledge query bag to BFS the
    knowledge graph. Excludes any entity already in extracted_entities
    (the LLM has those handles already).
    """

    canonical_id: str
    label: str  # NodeLabel.value (Service, Repo, Person, Ticket, ...)
    display_name: str | None = None  # from properties->>'name' or entity_cluster_metadata.display_name override
    edge_types: list[str] = Field(default_factory=list)  # MENTIONS, AUTHORED, ...
    max_confidence: str  # EXTRACTED | INFERRED | AMBIGUOUS
    doc_count: int  # # of result-set docs adjacent to this entity (BFS priority)
    # IDF-adjusted score used for ranking. score = doc_count / log(1 +
    # global_doc_count). Generic high-degree entities (e.g.
    # Channel:#engineering attached to 10k docs) get crushed; specific
    # entities surface. Surfaced so LLMs can see the ranking signal.
    score: float
    # Up to 3 doc IDs the entity is attached to, ordered by result rank
    # (strongest first). Caps at 3 even when doc_count > 3 -- the LLM uses
    # these to ground/audit the doc_count claim against its visible chunks
    # list, not to enumerate every attached doc. DISTINCT -- multi-edge
    # docs do not duplicate.
    associated_doc_ids: list[str] = Field(default_factory=list)
    # Total size of the entity cluster (primary + all merged aliases).
    # 1 for unmerged nodes. Lets agents prefer cluster-rich nodes when
    # picking BFS crawl candidates. Populated by the related-entities
    # walker from `entity_aliases` keyed on the primary.
    member_count: int = 1
    # Distinct source_systems across the cluster (from the primary's
    # consolidated `graph_node_provenance` — Phase 1 merges alias
    # provenance into the primary at merge time). [] for unmerged nodes
    # whose node hasn't been provenance-stamped yet (edge case; normal
    # ingest stamps it). Lets agents see "this person is GitHub +
    # Slack + Linear" without an extra round-trip.
    member_sources: list[str] = Field(default_factory=list)
```

- [ ] **Step 2: Run tests to verify the model still parses**

```bash
.venv/bin/pytest tests/retrieval/test_related_entities.py -v
```

Expected: all existing tests pass (defaults preserve backward compat; the walker doesn't populate the new fields yet so they take their defaults).

- [ ] **Step 3: Commit**

```bash
git add shared/models.py
git commit -m "feat(models): add member_count + member_sources to RelatedEntity

Cluster metadata fields surfaced alongside related-entities walker
output. member_count defaults to 1 so pre-Phase-2 payloads decode
unchanged. Populated by walker in a follow-up task."
```

---

### Task 3: Graph explore anchor translation

**Files:**
- Modify: `services/retrieval/main.py:587-605` (the `mode == "anchor"` branch in `graph_explore`)
- Create: `tests/retrieval/test_graph_explore_alias_anchor.py`

**Context:** Today: `/graph/explore?mode=anchor&anchor_node_id=mahit@prbe.ai` returns 404 if `mahit@prbe.ai` has been merged into `richardwei6` (because its `graph_nodes` row is hard-deleted). Phase 2 fix: translate `anchor_node_id` through `entity_aliases` before the existence check. The existing `anchor_exists()` and `anchor_graph_query()` then operate on the primary's canonical_id, which exists.

**Why no label scoping:** `anchor_exists()` doesn't filter by label (looks up by `canonical_id` only). We mirror that: a single `entity_aliases` SELECT by `alias_canonical_id` (without `label`) is consistent. In practice canonical_ids are label-unique per tenant; if a customer somehow has the same alias canonical_id under two different labels, the LIMIT-1 fallback picks one — same fuzziness as the existing anchor lookup.

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/test_graph_explore_alias_anchor.py`:

```python
"""End-to-end test for /graph/explore anchor-mode alias translation.

When a user types an alias canonical_id and that alias has been merged,
the endpoint must translate to the primary before resolving the anchor
and BFS — otherwise the response is 404 (alias node was hard-deleted at
merge time).

Uses raw HTTP via httpx.AsyncClient against the in-process app, the same
pattern as ``tests/test_entity_clusters_routes.py`` (TestClient clashes
with the live_db pool).
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from services.retrieval.main import app
from shared.config import Settings, get_settings
from shared.db import raw_conn, with_tenant
from shared.embeddings import reset_embedder
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio

CUSTOMER_ID = "graph-anchor-alias-cust"
API_KEY = "test-internal-key"  # query-side authentication header
PRIMARY = "richardwei6"
ALIAS = "mahit@prbe.ai"


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


async def _seed_cluster(customer_id: str) -> None:
    """Seed: customer + Person:PRIMARY graph_node (alias was hard-deleted
    at merge time per design doc) + entity_aliases row routing ALIAS to
    PRIMARY + a Repo:r1 node + AUTHORED edge from PRIMARY to a doc node so
    anchor_graph_query returns a 1-hop graph."""
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        # Two nodes: Person primary + Repo
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES
              ($1, 'Person', $2, '{"name":"Richard"}'::jsonb, 1),
              ($1, 'Repo',   'r1',  '{"name":"r1"}'::jsonb,    1)
            """,
            customer_id, PRIMARY,
        )
        # An edge so anchor_graph_query has something to return.
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type,
                from_node_id, to_node_id,
                confidence, properties
            )
            SELECT $1, 'TOUCHES',
                   p.node_id, r.node_id,
                   'EXTRACTED', '{}'::jsonb
            FROM graph_nodes p, graph_nodes r
            WHERE p.customer_id = $1 AND p.label = 'Person' AND p.canonical_id = $2
              AND r.customer_id = $1 AND r.label = 'Repo'   AND r.canonical_id = 'r1'
            """,
            customer_id, PRIMARY,
        )
        # Audit row (FK target for entity_aliases.merge_id).
        merge_row = await conn.fetchrow(
            """
            INSERT INTO entity_merge_audit (
                customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, 'Person', $2, ARRAY[$3]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            RETURNING merge_id
            """,
            customer_id, PRIMARY, ALIAS,
        )
        merge_id = merge_row["merge_id"]
        await conn.execute(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            customer_id, ALIAS, PRIMARY, merge_id,
        )


@pytest_asyncio.fixture
async def client(live_db) -> AsyncIterator[httpx.AsyncClient]:
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://test"
        ) as c:
            yield c


def _auth_headers(customer_id: str) -> dict[str, str]:
    """Build the query-side auth header that ``authenticate_query`` accepts.

    The retrieval service uses customer API-key auth. For tests, the
    ``live_db`` fixture seeds an ``api_key_hash`` of ``h-<customer_id>``,
    so the raw key sent on the wire is ``<customer_id>`` (the
    authenticate_query helper hashes and compares)."""
    return {"X-Prbe-Customer-Key": customer_id}


async def test_anchor_alias_resolves_to_primary_graph(client):
    """Typing the alias resolves to the primary's 1-hop graph (not 404)."""
    await _seed_cluster(CUSTOMER_ID)
    resp = await client.post(
        "/graph/explore",
        headers=_auth_headers(CUSTOMER_ID),
        json={"mode": "anchor", "anchor_node_id": ALIAS},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    node_ids = {n["id"] for n in body["nodes"]}
    # The primary's canonical_id is in the graph (the alias node was
    # hard-deleted at merge time so it's not).
    assert PRIMARY in node_ids
    assert ALIAS not in node_ids
    # 1-hop edge to Repo:r1 is present.
    assert "r1" in node_ids


async def test_anchor_unknown_returns_404(client):
    """An anchor that is neither a node nor an alias remains 404."""
    await _seed_cluster(CUSTOMER_ID)
    resp = await client.post(
        "/graph/explore",
        headers=_auth_headers(CUSTOMER_ID),
        json={"mode": "anchor", "anchor_node_id": "nobody-here"},
    )
    assert resp.status_code == 404


async def test_anchor_primary_id_unchanged(client):
    """Typing the primary's canonical_id directly works without translation."""
    await _seed_cluster(CUSTOMER_ID)
    resp = await client.post(
        "/graph/explore",
        headers=_auth_headers(CUSTOMER_ID),
        json={"mode": "anchor", "anchor_node_id": PRIMARY},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    node_ids = {n["id"] for n in body["nodes"]}
    assert PRIMARY in node_ids
```

> If the customer-key auth header above doesn't match the actual `authenticate_query` dependency (the existing graph_explore test suite is the source of truth — read `tests/retrieval/test_graph_*.py` for one), substitute the header that those tests use. Header shape is incidental to this task.

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/retrieval/test_graph_explore_alias_anchor.py -v
```

Expected: `test_anchor_alias_resolves_to_primary_graph` fails — anchor returns 404 because `mahit@prbe.ai`'s `graph_nodes` row was deleted at merge.

- [ ] **Step 3: Implement the translation**

Edit `services/retrieval/main.py:589-605`. Insert alias translation immediately after `assert req.anchor_node_id is not None` (line 596):

```python
    if req.mode == "anchor":
        # Cheap RLS-filtered existence check before the expensive BFS.
        # Translates a missing-anchor case to 404 (rather than returning
        # 200 with empty nodes/edges, which the frontend can't
        # distinguish from "exists but has no edges").
        # `anchor_node_id` is guaranteed non-None by the request
        # validator above; assert for type narrowing.
        assert req.anchor_node_id is not None

        # Phase 2: translate alias canonical_id to the cluster's primary
        # before the existence check. Without this, anchors typed as an
        # alias (e.g. mahit@prbe.ai post-merge) return 404 because their
        # graph_nodes row was hard-deleted at merge time. `entity_aliases`
        # is keyed on (customer_id, label, alias_canonical_id) but the
        # anchor endpoint doesn't carry a label — match across labels and
        # take the first hit (mirrors anchor_exists's label-less semantics).
        anchor_canonical_id = await _resolve_anchor_alias(
            customer_id=customer_id,
            anchor_canonical_id=req.anchor_node_id,
        )

        if not await anchor_exists(
            customer_id=customer_id, anchor_canonical_id=anchor_canonical_id
        ):
            raise HTTPException(status_code=404, detail="anchor_node_id not found")
        result = await anchor_graph_query(
            customer_id=customer_id,
            anchor_canonical_id=anchor_canonical_id,
            filters=req.filters.to_dataclass() if req.filters else None,
        )
    else:
```

Add the helper near the top of `services/retrieval/main.py` (or at the bottom — just above the endpoint is fine). Place it near the other graph_explore-related imports/helpers:

```python
from shared.db import with_tenant  # if not already imported at module scope


async def _resolve_anchor_alias(*, customer_id: str, anchor_canonical_id: str) -> str:
    """Translate a user-typed canonical_id through entity_aliases.

    If the input is an alias of a merged cluster, returns the primary's
    canonical_id. Otherwise returns the input unchanged (the lookup
    returns 0 rows for both unmerged nodes and primaries).

    Label-less by design — the anchor endpoint doesn't carry label
    context, and ``anchor_exists`` matches across labels too. The
    LIMIT 1 guards against the unlikely case where the same canonical_id
    is an alias under two different labels.
    """
    async with with_tenant(customer_id) as conn:
        row = await conn.fetchrow(
            """
            SELECT primary_canonical_id
            FROM entity_aliases
            WHERE customer_id = $1
              AND alias_canonical_id = $2
            LIMIT 1
            """,
            customer_id, anchor_canonical_id,
        )
    return row["primary_canonical_id"] if row else anchor_canonical_id
```

(If `with_tenant` is already imported at module scope — check the existing imports — skip the import line.)

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/retrieval/test_graph_explore_alias_anchor.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run existing graph_explore tests to verify no regression**

```bash
.venv/bin/pytest tests/retrieval/ -k graph -v
```

Expected: all existing graph tests pass (pre-merge: `entity_aliases` empty in their fixtures → translation is a no-op).

- [ ] **Step 6: Commit**

```bash
git add services/retrieval/main.py tests/retrieval/test_graph_explore_alias_anchor.py
git commit -m "feat(retrieval): translate alias anchors in /graph/explore

A user typing 'mahit@prbe.ai' as an anchor today returns 404 because
Phase 1's merge hard-deletes alias graph_nodes rows. Phase 2 translates
the alias to the primary canonical_id via entity_aliases before the
existence check, restoring the expected cluster-aware behavior.

Translation is label-less to mirror anchor_exists's own semantics. The
lookup is a single RLS-scoped SELECT and a no-op for non-aliased inputs."
```

---

### Task 4: Author filter expansion in list pipeline

**Files:**
- Modify: `services/retrieval/list_pipeline.py:145-152` (after `author_ids = _author_ids_from_entities(routed)`)
- Create: `tests/retrieval/test_list_pipeline_author_alias.py`

**Context:** `documents.author_id` is historical raw text written at ingest time — Phase 1 never rewrites it. Without expansion, asking "what did mahit@prbe.ai write" after merging mahit into richardwei6 misses every richardwei6-authored doc. Phase 2 expands each requested `author_id` to its full cluster (primary + aliases) before passing to `sql_list` / `sql_count` / `sql_group_by`. Label is fixed to `Person` (author_id is always a person canonical_id).

- [ ] **Step 1: Write failing test**

Create `tests/retrieval/test_list_pipeline_author_alias.py`:

```python
"""Test author filter expansion: a list-mode query with author_ids=[ALIAS]
must match documents whose author_id is PRIMARY (or any other alias).

documents.author_id is never rewritten on merge — Phase 2 expands at
filter time via entity_aliases.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval.list_pipeline import run_list
from services.retrieval.router import RouterOutput, RouterEntity
from shared.config import Settings, get_settings
from shared.constants import NodeLabel
from shared.db import raw_conn
from shared.embeddings import reset_embedder
from shared.models import QueryRequest, TemporalSpec
from shared.storage import reset_store

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch, settings: Settings) -> None:
    monkeypatch.setenv("TOKEN_ENCRYPTION_KEY", settings.token_encryption_key.get_secret_value())
    monkeypatch.setenv("ENVIRONMENT", "local")
    reset_embedder()
    reset_store()
    get_settings.cache_clear()  # type: ignore[attr-defined]


CUSTOMER_ID = "list-author-alias-cust"
PRIMARY = "richardwei6"
ALIAS = "mahit@prbe.ai"


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


async def _seed_doc_with_author(
    customer_id: str, *, doc_id: str, author_id: str, title: str
) -> None:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl,
                author_id
            ) VALUES (
                $1, 1, $2,
                'github', $3, 'https://example/' || $1,
                'raw_source', 'github.commit', 'text/plain',
                'h-' || $1, $4, 100, 0,
                $5, $5, $5, $5, '{}'::jsonb,
                $6
            )
            """,
            doc_id, customer_id, f"commit:{doc_id}", title, now, author_id,
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
            f"body {doc_id}", f"chash-{doc_id}",
        )


async def _seed_cluster(customer_id: str) -> None:
    async with raw_conn() as conn:
        merge_row = await conn.fetchrow(
            """
            INSERT INTO entity_merge_audit (
                customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, 'Person', $2, ARRAY[$3]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            RETURNING merge_id
            """,
            customer_id, PRIMARY, ALIAS,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            customer_id, ALIAS, PRIMARY, merge_row["merge_id"],
        )


def _routed_with_person(canonical_id: str) -> RouterOutput:
    """Build a minimal RouterOutput with one Person entity. Tweak the
    field names if RouterOutput / RouterEntity shape differs in your
    branch — read services/retrieval/router.py for the current type."""
    return RouterOutput(
        operation="list",
        entities=[
            RouterEntity(
                entity_type="person",
                canonical_id=canonical_id,
                display_name=canonical_id,
                confidence=0.9,
            )
        ],
    )


async def test_alias_author_expands_to_cluster(live_db):
    """Asking for documents authored by ALIAS returns docs authored
    by PRIMARY too (after the cluster expansion).
    """
    await _seed_customer(CUSTOMER_ID)
    await _seed_doc_with_author(CUSTOMER_ID, doc_id="doc-1", author_id=PRIMARY, title="primary doc")
    await _seed_doc_with_author(CUSTOMER_ID, doc_id="doc-2", author_id="someone-else", title="unrelated")
    await _seed_cluster(CUSTOMER_ID)

    req = QueryRequest(q="anything", top_k=10, entity_must_match=True)
    spec = TemporalSpec()
    routed = _routed_with_person(ALIAS)

    response = await run_list(
        req=req,
        customer_id=CUSTOMER_ID,
        routed=routed,
        spec=spec,
        temporal_meta={},
        sort_meta=None,
        extracted_entities=[{"canonical_id": ALIAS, "type": "person"}],
        doc_types=None,
        trace_id="t-1",
        timing={},
    )

    doc_ids = {d.doc_id for d in response.documents}
    assert "doc-1" in doc_ids, "Alias query should match primary-authored doc post-expansion"
    assert "doc-2" not in doc_ids


async def test_unmerged_author_passes_through(live_db):
    """Asking for documents by an unmerged author_id behaves as before."""
    await _seed_customer(CUSTOMER_ID)
    await _seed_doc_with_author(CUSTOMER_ID, doc_id="doc-3", author_id="loner-id", title="loner doc")

    req = QueryRequest(q="anything", top_k=10, entity_must_match=True)
    spec = TemporalSpec()
    routed = _routed_with_person("loner-id")

    response = await run_list(
        req=req,
        customer_id=CUSTOMER_ID,
        routed=routed,
        spec=spec,
        temporal_meta={},
        sort_meta=None,
        extracted_entities=[{"canonical_id": "loner-id", "type": "person"}],
        doc_types=None,
        trace_id="t-2",
        timing={},
    )
    doc_ids = {d.doc_id for d in response.documents}
    assert doc_ids == {"doc-3"}
```

> Note: the test must match your `RouterOutput`/`RouterEntity` constructor shape. Read `services/retrieval/router.py` for the current dataclass; the field names above are the most common shape but may differ. The test asserts at the response level so it tolerates downstream changes.

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest tests/retrieval/test_list_pipeline_author_alias.py -v
```

Expected: `test_alias_author_expands_to_cluster` fails — doc-1 is not in the result because the filter is `author_id = 'mahit@prbe.ai'` but the doc was authored by `richardwei6`.

- [ ] **Step 3: Implement the expansion**

Edit `services/retrieval/list_pipeline.py` around lines 137-152. Add a new import at the top of the file (alongside the existing `from services.retrieval.helpers import ...` if any, else add it):

```python
from services.retrieval.helpers import expand_to_cluster_members
from shared.db import with_tenant
```

Then in `run_list`, replace the author_ids computation:

```python
    if req.entity_must_match:
        author_ids = _author_ids_from_entities(routed)
        graph_entity_filters = _graph_entity_filters_from_routed(routed)
    else:
        author_ids = None
        graph_entity_filters = []
```

With:

```python
    if req.entity_must_match:
        author_ids = _author_ids_from_entities(routed)
        graph_entity_filters = _graph_entity_filters_from_routed(routed)
        # Phase 2: expand each author_id to its full Person cluster
        # (primary + aliases) so post-merge entities still match docs
        # written under their pre-merge author_id. `documents.author_id`
        # is historical raw text and is never rewritten on merge.
        if author_ids:
            async with with_tenant(customer_id) as conn:
                cluster_map = await expand_to_cluster_members(
                    conn, customer_id, label=NodeLabel.PERSON.value,
                    canonical_ids=author_ids,
                )
            expanded: list[str] = []
            seen: set[str] = set()
            for aid in author_ids:
                for member in cluster_map.get(aid, [aid]):
                    if member not in seen:
                        seen.add(member)
                        expanded.append(member)
            author_ids = expanded
    else:
        author_ids = None
        graph_entity_filters = []
```

Also confirm `NodeLabel` is imported at the top of `list_pipeline.py` — it likely already is (it's used for graph_entity_filters). If not, add:

```python
from shared.constants import NodeLabel
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest tests/retrieval/test_list_pipeline_author_alias.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Run existing list_pipeline tests to verify no regression**

```bash
.venv/bin/pytest tests/retrieval/test_list_pipeline_related.py tests/retrieval/test_list_path_metadata_filter.py tests/retrieval/test_list_pipeline_gating.py -v
```

Expected: all pass (pre-merge: `cluster_map[aid] == [aid]` → expanded list equals input → identical SQL).

- [ ] **Step 6: Commit**

```bash
git add services/retrieval/list_pipeline.py tests/retrieval/test_list_pipeline_author_alias.py
git commit -m "feat(retrieval): expand author_ids through entity clusters in list mode

documents.author_id is historical raw text and is never rewritten on
merge (Phase 1 only rewrites graph_edges endpoints). Without expansion,
querying author_ids=['mahit@prbe.ai'] post-merge misses every
richardwei6-authored doc. Expansion is Person-label-scoped (author_id
is always a person canonical_id) and a no-op pre-merge."
```

---

### Task 5: Related entities walker — cluster fields + display-name override + exclude translation

**Files:**
- Modify: `services/retrieval/retrievers/related_entities.py` (walker SQL ~lines 158-294 and Python builder ~lines 308-333; also `build_exclude_node_keys` near top)
- Create: `tests/retrieval/test_related_entities_clusters.py`

**Context:** The walker projects each non-Document neighbor node (which is always a primary because aliases were hard-deleted). Phase 2 enriches each row with:

1. `member_count` (primary + alias count from `entity_aliases`)
2. `member_sources` (DISTINCT `source_system` from `graph_node_provenance` of the primary's `node_id`)
3. Display name override from `entity_cluster_metadata.display_name` if set

Plus: when the caller passes `exclude_node_keys` containing an alias canonical_id (because the user typed an alias and the router preserved it), the SQL exclusion compares against `gn.canonical_id` — which is the primary, not the alias — so the exclusion silently misses. Fix: translate exclude keys through aliases at the caller boundary (where `exclude_node_keys` is constructed in `services/retrieval/pipeline.py` or wherever the typed routed-entities are first hardened into the exclude set).

**Identify the caller:** `grep -n "exclude_node_keys" services/retrieval/` to find where the keys are built. The most likely site is in `pipeline.py` or `search_pipeline.py` (`build_exclude_node_keys` from `related_entities.py` is just the in-walker variant helper). The implementer should grep first; if the call site is in `related_entities.py:build_exclude_node_keys`, translate there before returning.

- [ ] **Step 1: Locate exclude-key construction site**

```bash
grep -rn "exclude_node_keys\|build_exclude_node_keys" services/retrieval/ | head -20
```

Expected output identifies the function that constructs the exclude-key tuple set from `routed.entities`. Read it.

- [ ] **Step 2: Write failing tests**

Create `tests/retrieval/test_related_entities_clusters.py`:

```python
"""Phase 2 cluster awareness in the related-entities walker.

Covers:
1. member_count = primary + alias count (== 1 for unmerged)
2. member_sources = DISTINCT source_systems from graph_node_provenance
3. display_name override via entity_cluster_metadata
4. exclude_node_keys translates an alias to its primary so the typed
   alias doesn't recur as a related-entities suggestion.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval.retrievers.related_entities import (
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


CUSTOMER_ID = "rel-ents-cluster-cust"
PRIMARY = "richardwei6"
ALIAS_A = "mahit@prbe.ai"
ALIAS_B = "U07ABC123"
DOC_ID = "d-1"


async def _seed_full_cluster(customer_id: str) -> None:
    """Seed: customer, doc + chunk + Document node, Person:PRIMARY (alias
    rows hard-deleted at merge), AUTHORED edge from PRIMARY to doc,
    entity_aliases routing ALIAS_A and ALIAS_B to PRIMARY,
    graph_node_provenance for the cluster (consolidated to PRIMARY's node
    per Phase 1 merge logic), and a curated cluster display_name."""
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'test', 'h-' || $1)
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                $2, 1, $1,
                'github', 'commit:' || $2, 'https://example/' || $2,
                'raw_source', 'github.commit', 'text/plain',
                'h-' || $2, 'doc', 100, 0,
                $3, $3, $3, $3, '{}'::jsonb
            )
            """,
            customer_id, DOC_ID, now,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                $1, $2, $3, 0, 'body', 'chash', 5,
                array_fill(0::real, ARRAY[3072])::halfvec,
                1, 1
            )
            """,
            f"{DOC_ID}:c0", DOC_ID, customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES
              ($1, 'Document', $2, '{}'::jsonb, 1),
              ($1, 'Person',   $3, '{"name":"Richard"}'::jsonb, 1)
            """,
            customer_id, DOC_ID, PRIMARY,
        )
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type,
                from_node_id, to_node_id,
                confidence, properties
            )
            SELECT $1, 'AUTHORED',
                   p.node_id, d.node_id,
                   'EXTRACTED', '{}'::jsonb
            FROM graph_nodes p, graph_nodes d
            WHERE p.customer_id = $1 AND p.label = 'Person'   AND p.canonical_id = $3
              AND d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = $2
            """,
            customer_id, DOC_ID, PRIMARY,
        )
        # Provenance entries (post-merge state: alias provenance is
        # merged into the primary's node_id).
        await conn.execute(
            """
            INSERT INTO graph_node_provenance (
                customer_id, node_id, source_system,
                first_seen_at, last_seen_at
            )
            SELECT $1, p.node_id, src, $2, $2
            FROM graph_nodes p, UNNEST(ARRAY['github','slack','linear']) AS src
            WHERE p.customer_id = $1 AND p.label = 'Person' AND p.canonical_id = $3
            """,
            customer_id, now, PRIMARY,
        )
        merge_row = await conn.fetchrow(
            """
            INSERT INTO entity_merge_audit (
                customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, 'Person', $2, ARRAY[$3, $4]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            RETURNING merge_id
            """,
            customer_id, PRIMARY, ALIAS_A, ALIAS_B,
        )
        await conn.executemany(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            [
                (customer_id, ALIAS_A, PRIMARY, merge_row["merge_id"]),
                (customer_id, ALIAS_B, PRIMARY, merge_row["merge_id"]),
            ],
        )
        await conn.execute(
            """
            INSERT INTO entity_cluster_metadata (
                customer_id, label, primary_canonical_id, display_name
            ) VALUES ($1, 'Person', $2, 'Richard Wei (canonical)')
            """,
            customer_id, PRIMARY,
        )


async def test_walker_populates_member_count_and_sources(live_db):
    await _seed_full_cluster(CUSTOMER_ID)
    rels = await walk_result_doc_neighbors(
        customer_id=CUSTOMER_ID,
        ranked_result_docs=[(DOC_ID, 1)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    # Only one neighbor: the Person primary.
    [person] = [r for r in rels if r.label == "Person"]
    assert person.canonical_id == PRIMARY
    # member_count = primary (1) + 2 aliases = 3.
    assert person.member_count == 3
    # member_sources from consolidated provenance.
    assert sorted(person.member_sources) == ["github", "linear", "slack"]


async def test_walker_uses_display_name_override(live_db):
    await _seed_full_cluster(CUSTOMER_ID)
    rels = await walk_result_doc_neighbors(
        customer_id=CUSTOMER_ID,
        ranked_result_docs=[(DOC_ID, 1)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    [person] = [r for r in rels if r.label == "Person"]
    assert person.display_name == "Richard Wei (canonical)"


async def test_walker_falls_back_to_properties_name_when_no_override(live_db):
    """No entity_cluster_metadata row -> use graph_nodes.properties->>'name'."""
    await _seed_full_cluster(CUSTOMER_ID)
    async with raw_conn() as conn:
        await conn.execute(
            "DELETE FROM entity_cluster_metadata WHERE customer_id = $1",
            CUSTOMER_ID,
        )
    rels = await walk_result_doc_neighbors(
        customer_id=CUSTOMER_ID,
        ranked_result_docs=[(DOC_ID, 1)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    [person] = [r for r in rels if r.label == "Person"]
    assert person.display_name == "Richard"


async def test_walker_treats_empty_override_as_no_override(live_db):
    """Empty-string display_name override falls through to properties name."""
    await _seed_full_cluster(CUSTOMER_ID)
    async with raw_conn() as conn:
        await conn.execute(
            "UPDATE entity_cluster_metadata SET display_name = '' WHERE customer_id = $1",
            CUSTOMER_ID,
        )
    rels = await walk_result_doc_neighbors(
        customer_id=CUSTOMER_ID,
        ranked_result_docs=[(DOC_ID, 1)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    [person] = [r for r in rels if r.label == "Person"]
    assert person.display_name == "Richard"


async def test_walker_member_count_one_for_unmerged_node(live_db):
    """An unmerged neighbor reports member_count=1 + member_sources from its own provenance."""
    customer_id = "rel-ents-unmerged-cust"
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'test', 'h-' || $1) ON CONFLICT (customer_id) DO NOTHING",
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, ingested_at, acl
            ) VALUES (
                'd-loner', 1, $1,
                'github', 'commit:d-loner', 'https://example/d-loner',
                'raw_source', 'github.commit', 'text/plain',
                'h-d-loner', 'doc', 100, 0,
                $2, $2, $2, $2, '{}'::jsonb
            )
            """,
            customer_id, now,
        )
        await conn.execute(
            """
            INSERT INTO chunks (
                chunk_id, doc_id, customer_id,
                chunk_index, content, content_hash, token_count,
                embedding, first_seen_version, last_seen_version
            ) VALUES (
                'd-loner:c0', 'd-loner', $1, 0, 'body', 'chash', 5,
                array_fill(0::real, ARRAY[3072])::halfvec, 1, 1
            )
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES
              ($1, 'Document', 'd-loner', '{}'::jsonb, 1),
              ($1, 'Person',   'loner-id', '{"name":"Loner"}'::jsonb, 1)
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_edges (
                customer_id, edge_type, from_node_id, to_node_id,
                confidence, properties
            )
            SELECT $1, 'AUTHORED', p.node_id, d.node_id, 'EXTRACTED', '{}'::jsonb
            FROM graph_nodes p, graph_nodes d
            WHERE p.customer_id = $1 AND p.label = 'Person'   AND p.canonical_id = 'loner-id'
              AND d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = 'd-loner'
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_node_provenance (
                customer_id, node_id, source_system,
                first_seen_at, last_seen_at
            )
            SELECT $1, p.node_id, 'github', $2, $2
            FROM graph_nodes p
            WHERE p.customer_id = $1 AND p.label = 'Person' AND p.canonical_id = 'loner-id'
            """,
            customer_id, now,
        )
    rels = await walk_result_doc_neighbors(
        customer_id=customer_id,
        ranked_result_docs=[("d-loner", 1)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    [person] = [r for r in rels if r.label == "Person"]
    assert person.member_count == 1
    assert person.member_sources == ["github"]


async def test_alias_in_exclude_keys_translates_to_primary(live_db):
    """When the caller passes an alias canonical_id in exclude_node_keys,
    `build_exclude_node_keys` (or its caller) translates it to the primary
    so the walker actually excludes the primary node from results."""
    await _seed_full_cluster(CUSTOMER_ID)
    # Build exclude keys as if the router emitted Person:mahit@prbe.ai.
    # Pass the cluster-translated set into the walker.
    from services.retrieval.helpers import resolve_aliases
    from shared.db import with_tenant

    raw_keys = {("Person", ALIAS_A.lower())}
    async with with_tenant(CUSTOMER_ID) as conn:
        translated = await resolve_aliases(
            conn, CUSTOMER_ID, refs=[(lbl, cid) for (lbl, cid) in raw_keys]
        )
    # Build the final exclude set: original keys + translated primaries.
    exclude_keys = set(raw_keys)
    for (lbl, cid), primary in translated.items():
        exclude_keys.add((lbl, primary.lower()))

    rels = await walk_result_doc_neighbors(
        customer_id=CUSTOMER_ID,
        ranked_result_docs=[(DOC_ID, 1)],
        exclude_node_keys=exclude_keys,
        min_confidence=None,
        top_n=10,
    )
    # The Person primary must be excluded because the user typed its alias.
    assert all(r.label != "Person" for r in rels)
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/retrieval/test_related_entities_clusters.py -v
```

Expected failures:
- `member_count`/`member_sources` are default (1, []) on every row (not populated).
- Display name override not applied.
- Exclude-key translation test passes on its own (it's manual — verifies the helper output, doesn't depend on walker changes) but flags whether the walker uses the keys correctly.

- [ ] **Step 4: Implement the SQL + Python builder changes**

Edit `services/retrieval/retrievers/related_entities.py` around line 158-294. The `result_aggregates` CTE currently selects/joins:

```sql
        result_aggregates AS (
            SELECT
                gn.canonical_id,
                gn.label,
                gn.properties->>'name' AS display_name,
                gn.node_id,
                array_agg(DISTINCT ne.edge_type) AS edge_types,
                max({confidence_case}) AS max_confidence_rank,
                COUNT(DISTINCT doc_gn.canonical_id) AS doc_count,
                array_agg(doc_gn.canonical_id ORDER BY ne.doc_rank ASC) AS sample_pool
            FROM neighbor_edges ne
            JOIN graph_nodes gn
              ON gn.node_id = ne.neighbor_node_id
             AND gn.customer_id = $1
            JOIN graph_nodes doc_gn
              ON doc_gn.node_id = ne.doc_node_id
             AND doc_gn.customer_id = $1
            WHERE gn.label != '{document_label}'
              AND NOT EXISTS ( ... exclude logic ... )
            GROUP BY gn.canonical_id, gn.label, gn.properties->>'name', gn.node_id
            HAVING max({confidence_case}) >= $6
        ),
```

Change to (additions marked with `-- PHASE 2:`):

```sql
        result_aggregates AS (
            SELECT
                gn.canonical_id,
                gn.label,
                COALESCE(NULLIF(ecm.display_name, ''), gn.properties->>'name') AS display_name,
                gn.node_id,
                array_agg(DISTINCT ne.edge_type) AS edge_types,
                max({confidence_case}) AS max_confidence_rank,
                COUNT(DISTINCT doc_gn.canonical_id) AS doc_count,
                array_agg(doc_gn.canonical_id ORDER BY ne.doc_rank ASC) AS sample_pool,
                -- PHASE 2: cluster size = primary + alias count.
                (1 + COALESCE(ea_count.alias_count, 0))::int AS member_count,
                -- PHASE 2: distinct source_systems from consolidated provenance.
                COALESCE(gnp.sources, ARRAY[]::text[]) AS member_sources
            FROM neighbor_edges ne
            JOIN graph_nodes gn
              ON gn.node_id = ne.neighbor_node_id
             AND gn.customer_id = $1
            JOIN graph_nodes doc_gn
              ON doc_gn.node_id = ne.doc_node_id
             AND doc_gn.customer_id = $1
            -- PHASE 2: per-primary alias count (NULL when no merge happened).
            LEFT JOIN LATERAL (
                SELECT COUNT(*) AS alias_count
                FROM entity_aliases
                WHERE customer_id = $1
                  AND label = gn.label
                  AND primary_canonical_id = gn.canonical_id
            ) ea_count ON TRUE
            -- PHASE 2: distinct source_systems for the primary's node.
            -- LATERAL subquery is keyed on node_id (PK-ish) so this is
            -- one index probe per neighbor, no GROUP BY cardinality blowup.
            LEFT JOIN LATERAL (
                SELECT array_agg(DISTINCT source_system ORDER BY source_system) AS sources
                FROM graph_node_provenance
                WHERE customer_id = $1
                  AND node_id = gn.node_id
            ) gnp ON TRUE
            -- PHASE 2: optional curated display name override.
            LEFT JOIN entity_cluster_metadata ecm
              ON ecm.customer_id = $1
             AND ecm.label = gn.label
             AND ecm.primary_canonical_id = gn.canonical_id
            WHERE gn.label != '{document_label}'
              AND NOT EXISTS (
                  SELECT 1 FROM exclude_keys ek
                  WHERE ek.label = gn.label
                    AND ek.canonical_id IN (
                        lower(gn.canonical_id),
                        regexp_replace(lower(gn.canonical_id), '^.*/', ''),
                        lower(gn.properties->>'name'),
                        regexp_replace(lower(gn.properties->>'name'), '^.*/', '')
                    )
              )
            GROUP BY gn.canonical_id, gn.label, gn.properties->>'name',
                     gn.node_id, ea_count.alias_count, gnp.sources, ecm.display_name
            HAVING max({confidence_case}) >= $6
        ),
```

In the outer SELECT (~line 279-294), extend the projection:

```sql
        SELECT
            ra.canonical_id, ra.label, ra.display_name, ra.edge_types,
            ra.max_confidence_rank, ra.doc_count,
            (ra.doc_count::float / ln(1 + COALESCE(ngf.global_doc_count, 1)))
                AS score,
            ra.sample_pool,
            ra.member_count,
            ra.member_sources
        FROM result_aggregates ra
        LEFT JOIN neighbor_global_freq ngf USING (node_id)
        ORDER BY score DESC, ra.doc_count DESC, ra.max_confidence_rank DESC,
                 ra.label ASC, ra.canonical_id ASC
        LIMIT $7
```

In the Python builder (~lines 321-332), extend the `RelatedEntity` construction:

```python
        out.append(
            RelatedEntity(
                canonical_id=r["canonical_id"],
                label=r["label"],
                display_name=r["display_name"],
                edge_types=list(r["edge_types"] or []),
                max_confidence=max_confidence,
                doc_count=int(r["doc_count"]),
                score=float(r["score"]),
                associated_doc_ids=associated_doc_ids,
                member_count=int(r["member_count"]),
                member_sources=list(r["member_sources"] or []),
            )
        )
```

- [ ] **Step 5: Locate exclude-key construction site and translate**

Based on Step 1's grep output, modify the caller that builds `exclude_node_keys` from routed entities. The typical pattern is in `services/retrieval/pipeline.py` (or `search_pipeline.py` / wherever) — it iterates routed entities and adds `(label, normalized_canonical_id)` tuples. Phase 2 addition: after building the set, also add the primary's canonical_id for every alias.

Insert this pattern at the caller (replacing the equivalent block):

```python
from services.retrieval.helpers import resolve_aliases
from shared.db import with_tenant

# After building `exclude_keys: set[tuple[str, str]]` from routed entities,
# translate any alias canonical_ids to primaries so the walker's
# exclusion comparison (which compares against gn.canonical_id == primary)
# correctly excludes the cluster.
if exclude_keys:
    # Build the alias-lookup set: use the original-case canonical_id (not
    # the lowercased normalized variant) because entity_aliases stores
    # case-preserved canonical_ids.
    async with with_tenant(customer_id) as conn:
        translated = await resolve_aliases(
            conn, customer_id,
            refs=list({(lbl, cid) for (lbl, cid) in routed_entity_keys}),
        )
    for (lbl, _cid), primary in translated.items():
        exclude_keys.add((lbl, primary.lower()))
```

> Adapt the variable names to the actual call site. The point is: for every routed entity that's an alias of a merged cluster, ALSO add `(label, primary.lower())` to `exclude_keys`.

If the call site lives inside an already-async function with a `conn` already in scope (e.g., inside a `with_tenant` block), use `conn` directly instead of opening a new one.

- [ ] **Step 6: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/retrieval/test_related_entities_clusters.py tests/retrieval/test_related_entities.py -v
```

Expected: all new tests pass; existing tests still pass (pre-merge: LEFT JOINs no-op, member_count=1 default doesn't trip the existing assertions).

- [ ] **Step 7: Run the broader retrieval test suite**

```bash
.venv/bin/pytest tests/retrieval/ -v
```

Expected: all green. The `member_count` and `member_sources` additions are additive on `RelatedEntity` so existing tests that don't assert on them continue to pass.

- [ ] **Step 8: Commit**

```bash
git add services/retrieval/retrievers/related_entities.py \
        services/retrieval/pipeline.py \
        tests/retrieval/test_related_entities_clusters.py
git commit -m "feat(retrieval): cluster-aware related_entities walker

Walker SQL gains three LEFT JOINs to enrich each result row:
- entity_aliases for member_count (primary + aliases)
- graph_node_provenance for member_sources (distinct source_systems)
- entity_cluster_metadata for display_name override

Caller exclude_node_keys is also alias-translated so typing an alias
canonical_id still excludes the primary from the walker output.

Pre-merge: entity_aliases and entity_cluster_metadata are empty, so
the LEFT JOINs no-op and behavior is identical to today."
```

> If the exclude-key translation site lives in a different file than `services/retrieval/pipeline.py`, adjust the `git add` accordingly. Use `git status` to confirm.

---

### Task 6: Search pipeline — routed-entity translation + display-name override

**Files:**
- Modify: `services/retrieval/search_pipeline.py` (around lines 620-780, the entity result builder)
- Create: `tests/retrieval/test_search_pipeline_entity_clusters.py`

**Context:** Search pipeline's `QueryEntityResult` builder takes router-extracted entities, looks them up in `graph_nodes` via `(label, canonical_id) IN UNNEST(...)`, and emits one `QueryEntityResult` per hit. Post-Phase-1: if the router extracts `Person:mahit@prbe.ai`, the lookup misses (alias graph_node deleted) and the entity result is silently dropped. Phase 2 translates `(label, canonical_id)` through `entity_aliases` before the SQL lookup so the result lands on the primary's row. Also: apply the same `entity_cluster_metadata.display_name` override as Task 5 so curated names show up consistently.

**The confidence map quirk:** `confidence_by_key[(label, canonical_id)] = max(...)` is built from `resolved` (which uses the typed canonical_id). After translation, the SQL row will have `gn.canonical_id == primary`, so we need to key `confidence_by_key` by the post-translation primary too — otherwise the lookup at line 748 misses and confidence defaults to 1.0.

- [ ] **Step 1: Write failing tests**

Create `tests/retrieval/test_search_pipeline_entity_clusters.py`:

```python
"""Phase 2 search pipeline cluster awareness.

When the router extracts an alias canonical_id, the corresponding
QueryEntityResult must land on the cluster's primary (not be dropped),
and any entity_cluster_metadata display_name override must apply.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from services.retrieval.router import RouterEntity, RouterOutput
from services.retrieval.search_pipeline import _build_entity_results
from shared.config import Settings, get_settings
from shared.constants import NodeLabel
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


CUSTOMER_ID = "search-cluster-cust"
PRIMARY = "richardwei6"
ALIAS = "mahit@prbe.ai"


async def _seed(customer_id: str) -> None:
    now = datetime(2026, 4, 28, tzinfo=UTC)
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'test', 'h-' || $1) ON CONFLICT (customer_id) DO NOTHING",
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES ($1, 'Person', $2, '{"name":"Richard"}'::jsonb, 1)
            """,
            customer_id, PRIMARY,
        )
        merge_row = await conn.fetchrow(
            """
            INSERT INTO entity_merge_audit (
                customer_id, label, primary_canonical_id,
                merged_alias_canonical_ids, performed_by_user_id, status
            ) VALUES ($1, 'Person', $2, ARRAY[$3]::text[],
                      '11111111-1111-1111-1111-111111111111', 'active')
            RETURNING merge_id
            """,
            customer_id, PRIMARY, ALIAS,
        )
        await conn.execute(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            ) VALUES ($1, 'Person', $2, $3, $4)
            """,
            customer_id, ALIAS, PRIMARY, merge_row["merge_id"],
        )
        await conn.execute(
            """
            INSERT INTO entity_cluster_metadata (
                customer_id, label, primary_canonical_id, display_name
            ) VALUES ($1, 'Person', $2, 'Richard Wei (canonical)')
            """,
            customer_id, PRIMARY,
        )


async def test_alias_input_lands_on_primary(live_db):
    """Router extracts mahit@prbe.ai; result is for richardwei6."""
    await _seed(CUSTOMER_ID)
    routed = RouterOutput(
        entities=[
            RouterEntity(
                entity_type="person",
                canonical_id=ALIAS,
                display_name=ALIAS,
                confidence=0.9,
            )
        ],
    )
    results = await _build_entity_results(
        customer_id=CUSTOMER_ID,
        routed=routed,
        timing={},
    )
    assert len(results) == 1
    assert results[0].canonical_id == PRIMARY
    # Override applied.
    assert results[0].display_name == "Richard Wei (canonical)"


async def test_unmerged_input_unchanged(live_db):
    """Unmerged canonical_id flows through unchanged."""
    customer_id = "search-cluster-loner-cust"
    async with raw_conn() as conn:
        await conn.execute(
            "INSERT INTO customers (customer_id, display_name, api_key_hash) "
            "VALUES ($1, 'test', 'h-' || $1) ON CONFLICT (customer_id) DO NOTHING",
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
            VALUES ($1, 'Person', 'loner-id', '{"name":"Loner"}'::jsonb, 1)
            """,
            customer_id,
        )
    routed = RouterOutput(
        entities=[
            RouterEntity(
                entity_type="person",
                canonical_id="loner-id",
                display_name="Loner",
                confidence=0.9,
            )
        ],
    )
    results = await _build_entity_results(
        customer_id=customer_id,
        routed=routed,
        timing={},
    )
    assert len(results) == 1
    assert results[0].canonical_id == "loner-id"
    assert results[0].display_name == "Loner"  # properties->>'name'


async def test_two_aliases_of_same_primary_collapse(live_db):
    """Router extracts two aliases of the same primary; we emit ONE result."""
    await _seed(CUSTOMER_ID)
    # Add a second alias.
    async with raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO entity_aliases (
                customer_id, label, alias_canonical_id,
                primary_canonical_id, merge_id
            )
            SELECT $1, 'Person', 'U07ABC123', $2, merge_id
            FROM entity_merge_audit
            WHERE customer_id = $1 AND primary_canonical_id = $2
            LIMIT 1
            """,
            CUSTOMER_ID, PRIMARY,
        )
    routed = RouterOutput(
        entities=[
            RouterEntity(entity_type="person", canonical_id=ALIAS,        display_name=ALIAS,        confidence=0.9),
            RouterEntity(entity_type="person", canonical_id="U07ABC123",  display_name="U07ABC123",  confidence=0.8),
        ],
    )
    results = await _build_entity_results(
        customer_id=CUSTOMER_ID,
        routed=routed,
        timing={},
    )
    # Both aliases collapse to the primary -> one result row.
    assert len(results) == 1
    assert results[0].canonical_id == PRIMARY
```

> The test invokes `_build_entity_results` directly — this is the function around `search_pipeline.py:620` that constructs `QueryEntityResult`s from `routed.entities`. If the function name differs in your branch, grep for the function that issues the `WITH wanted AS (SELECT * FROM unnest...)` SQL block and use its name. The test is structured around the unit, not the full search pipeline.

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest tests/retrieval/test_search_pipeline_entity_clusters.py -v
```

Expected: `test_alias_input_lands_on_primary` fails (zero results — alias miss); `test_two_aliases_of_same_primary_collapse` fails (zero results); `test_unmerged_input_unchanged` passes.

- [ ] **Step 3: Implement the translation + override**

Edit `services/retrieval/search_pipeline.py`. Locate the `_build_entity_results` (or similarly-named) function around line 620. The current `resolved` loop (~lines 630-647) iterates `routed.entities`, validates labels, and pushes `(label, canonical_id, entity)` tuples. After that loop, before the SQL `labels = [r[0] for r in resolved]` line, add alias translation:

```python
    if not resolved:
        return []

    # Phase 2: translate any alias canonical_ids to their primaries so
    # the (label, canonical_id) lookup in graph_nodes hits the surviving
    # primary row. Without this, the lookup misses (alias graph_nodes
    # was hard-deleted at merge time) and the routed entity is silently
    # dropped.
    from services.retrieval.helpers import resolve_aliases
    from shared.db import with_tenant

    async with with_tenant(customer_id) as alias_conn:
        alias_map = await resolve_aliases(
            alias_conn, customer_id,
            refs=[(r[0], r[1]) for r in resolved],
        )
    if alias_map:
        translated: list[tuple[str, str, "RouterEntity"]] = []
        seen: set[tuple[str, str]] = set()
        for label, cid, entity in resolved:
            primary = alias_map.get((label, cid), cid)
            key = (label, primary)
            if key in seen:
                # Two aliases of the same primary — keep the higher-confidence one.
                # (The downstream confidence_by_key max() handles this too, but
                # de-duping the resolved list keeps the SQL UNNEST clean.)
                continue
            seen.add(key)
            translated.append((label, primary, entity))
        resolved = translated

    labels = [r[0] for r in resolved]
    canonical_ids = [r[1] for r in resolved]
```

Then in the SQL block (~lines 652-728), the SELECT around line 657 currently does:

```sql
            entity_nodes AS (
                SELECT gn.node_id, gn.label, gn.canonical_id, gn.properties
                FROM graph_nodes gn
                JOIN wanted w ON w.label = gn.label
                              AND w.canonical_id = gn.canonical_id
                WHERE gn.customer_id = $1
            ),
```

Extend to LEFT JOIN `entity_cluster_metadata`:

```sql
            entity_nodes AS (
                SELECT
                    gn.node_id, gn.label, gn.canonical_id, gn.properties,
                    -- PHASE 2: optional curated display name override.
                    NULLIF(ecm.display_name, '') AS override_display_name
                FROM graph_nodes gn
                JOIN wanted w ON w.label = gn.label
                              AND w.canonical_id = gn.canonical_id
                LEFT JOIN entity_cluster_metadata ecm
                  ON ecm.customer_id = gn.customer_id
                 AND ecm.label = gn.label
                 AND ecm.primary_canonical_id = gn.canonical_id
                WHERE gn.customer_id = $1
            ),
```

And extend the outer SELECT (~line 714) to project the new column:

```sql
            SELECT en.node_id, en.label, en.canonical_id, en.properties,
                   en.override_display_name,
                   (SELECT array_agg(DISTINCT eda.edge_type)
                          FILTER (WHERE eda.edge_type IS NOT NULL)
                    FROM entity_doc_attachments eda
                    WHERE eda.entity_node_id = en.node_id) AS edge_types,
                   (SELECT COUNT(DISTINCT eda.doc_id)
                    FROM entity_doc_attachments eda
                    WHERE eda.entity_node_id = en.node_id) AS doc_count,
                   (SELECT array_agg(ra.doc_id ORDER BY ra.rn)
                    FROM ranked_attachments ra
                    WHERE ra.entity_node_id = en.node_id
                      AND ra.rn <= $4) AS attached_doc_pool
```

In the Python builder, change the `display_name` line (~line 760) from:

```python
        display_name = properties.get("name") if isinstance(properties.get("name"), str) else None
```

To:

```python
        # Phase 2: prefer curated entity_cluster_metadata.display_name if set
        # (NULLIF coerces empty strings to NULL upstream).
        override = r["override_display_name"]
        if isinstance(override, str) and override:
            display_name = override
        else:
            display_name = properties.get("name") if isinstance(properties.get("name"), str) else None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest tests/retrieval/test_search_pipeline_entity_clusters.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run the broader search-pipeline test suite**

```bash
.venv/bin/pytest tests/retrieval/test_search_pipeline_polymorphic.py tests/retrieval/test_search_pipeline_directed.py tests/retrieval/test_search_pipeline_metadata_fallback.py tests/retrieval/test_search_pipeline_related.py tests/retrieval/test_entity_results_population.py -v
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add services/retrieval/search_pipeline.py tests/retrieval/test_search_pipeline_entity_clusters.py
git commit -m "feat(retrieval): cluster-aware routed-entity lookup in search pipeline

When the router extracts an alias canonical_id, the (label, canonical_id)
lookup in graph_nodes would miss post-merge because the alias row was
hard-deleted. Phase 2 translates aliases through entity_aliases before
the lookup so the QueryEntityResult lands on the primary's row.

Also: LEFT JOIN entity_cluster_metadata to apply display_name overrides,
matching the same treatment applied to RelatedEntity by the walker."
```

---

### Task 7: Final pass — full test suite + design-doc consistency

> **Pytest is not enough.** Task 8 (container smoke test) is REQUIRED before declaring Phase 2 done. Do not open the PR until both Tasks 7 and 8 pass.

**Files:**
- No code changes; sanity check + (optional) cross-reference note in design doc.

- [ ] **Step 1: Run the full retrieval test suite**

```bash
.venv/bin/pytest tests/retrieval/ -v
```

Expected: 100% green.

- [ ] **Step 2: Run the project-wide test suite (excludes legacy/)**

```bash
.venv/bin/pytest tests/ -v
```

Expected: 100% green (modulo pre-existing flakes in `tests/synthesis/test_reclaim.py` per Phase 1 notes; verify any failures are pre-existing).

- [ ] **Step 3: Verify type checks pass**

```bash
.venv/bin/mypy services/retrieval/ shared/models.py
```

Expected: 0 errors.

- [ ] **Step 4: Linter**

```bash
.venv/bin/ruff check services/retrieval/ shared/models.py tests/retrieval/
```

Expected: 0 issues.

- [ ] **Step 5: Append Phase 2 reference to the design doc (optional cross-link)**

Edit `docs/superpowers/specs/2026-05-13-entity-clusters-design.md` and add at the bottom of the §"Read-side behavior (Phase 2 preview)" section (line ~453):

```markdown

> Phase 2 implementation landed in commit range `entity-clusters-phase2`. See `docs/superpowers/specs/2026-05-14-entity-clusters-phase2-plan.md`.
```

- [ ] **Step 6: Commit + push**

```bash
git add docs/superpowers/specs/2026-05-13-entity-clusters-design.md
git commit -m "docs(entity-clusters): cross-link Phase 2 implementation to design doc"
git push -u origin entity-clusters-phase2
```

- [ ] **Step 7: Open PR — but only after Task 8 passes**

Stack the PR on `entity-clusters-phase1` (the parent branch). Use `gh pr create --base entity-clusters-phase1 ...` once Phase 1 PRs are merged you can rebase onto `main` and retarget.

```bash
gh pr create --base entity-clusters-phase1 \
  --title "feat(retrieval): Phase 2 — cluster-aware retrieval" \
  --body "$(cat <<'EOF'
## Summary
- Adds `resolve_aliases` + `expand_to_cluster_members` helpers (mirror `graph_writer._fetch_aliases`).
- `/graph/explore` anchor-mode translates alias canonical_ids to the primary before the existence check.
- `run_list` author filter expands `author_ids` through cluster members (Person-scoped).
- `RelatedEntity` gains `member_count` + `member_sources`; walker SQL enriches via LEFT JOINs on `entity_aliases`, `graph_node_provenance`, and `entity_cluster_metadata`.
- Search pipeline routed-entity lookup translates aliases so `QueryEntityResult` hits land on the primary; also applies `entity_cluster_metadata.display_name` override.
- `exclude_node_keys` is alias-translated at construction so the walker doesn't recommend a node the user explicitly named.

## Anti-scope
- No graph_writer changes.
- No `/api/entity-clusters/*` endpoint changes.
- No cluster-aware edge aggregation (Phase 2.5+).
- No schema changes.

## Test plan
- [x] `pytest tests/retrieval/` green
- [x] `pytest tests/` green (modulo pre-existing flakes)
- [x] `mypy services/retrieval/ shared/models.py` clean
- [x] `ruff check` clean
- [ ] Manual smoke (after PR 1a + 1b merge): seed merged cluster, hit `/graph/explore?anchor=mahit@prbe.ai`, verify primary's graph returns; hit `/query` list-mode with `author_ids=[alias]`, verify primary's docs match.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

---

### Task 8: Live container smoke test (REQUIRED before merge)

**Files:**
- Create: `scripts/smoke_phase2_clusters.py` (scratch seed script — committed to the branch but never imported by application code)
- No application code changes.

**Context:** Pytest exercises the modules in isolation but does not verify the full ASGI stack (lifespan context, RLS GUC inheritance across endpoints, header passthrough, asyncpg pool initialization). Phase 1 caught real semantic issues this way (e.g., 409-already-aliased path being unreachable because the existence check fires first). Phase 2 needs the same end-to-end verification: spin up real uvicorn against Docker Postgres, seed a cluster, hit BOTH the ingestion merge endpoint AND every retrieval surface this PR touches, and visually confirm the responses match the design doc's read-side claims.

**Auth scheme:** retrieval supports two paths:
1. `X-Prbe-Customer-Key: <raw_key>` (external) — hashed and compared to `customers.api_key_hash`.
2. `X-Internal-Knowledge-Key: <secret>` + `X-Prbe-Customer: <id>` (internal trust boundary) — used by the BFF.

We use scheme #2 for the smoke test because `INTERNAL_KNOWLEDGE_API_KEY` is already configured for the ingestion-side merge call; reusing it avoids computing a SHA hash for a raw external key.

**Ports:** Ingestion `services.ingestion.main:app` → 9817 (same as Phase 1 smoke). Retrieval `services.retrieval.main:app` → 8081 (the default in `services/retrieval/main.py:1276`). Run both concurrently in the background.

- [ ] **Step 1: Pre-flight checks**

```bash
cd /Users/mahitnamburu/Desktop/prbe/prbe-knowledge-worktrees/entity-clusters-phase2
docker compose ps                       # confirm postgres + minio up
.venv/bin/alembic -c db/alembic.ini current   # confirm head includes 20260514_0071
echo $INTERNAL_KNOWLEDGE_API_KEY          # confirm env var set (use test-internal-key)
```

If Docker isn't running: `docker compose up -d` and wait for healthy.
If migration isn't at head: `.venv/bin/alembic -c db/alembic.ini upgrade head`.

- [ ] **Step 2: Write the seed script**

Create `scripts/smoke_phase2_clusters.py`:

```python
"""One-off seed for the Phase 2 cluster-awareness smoke test.

Idempotent: drops the smoke customer first, then re-seeds. Safe to
re-run between iterations. Never imported by application code.

Seed shape:
  - Customer: smoke-phase2-cust
  - Person nodes:    richardwei6, mahit@prbe.ai, U07ABC123
  - Provenance:      richardwei6 -> github
                     mahit@prbe.ai -> slack
                     U07ABC123 -> linear
  - Document:        d-1 (authored by richardwei6) + d-2 (authored by mahit@prbe.ai)
  - Graph edges:     richardwei6 -AUTHORED-> Document:d-1
                     mahit@prbe.ai -AUTHORED-> Document:d-2
  - No entity_aliases rows yet — that's what /api/entity-clusters/merge writes.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from shared.db import close_pool, init_pool, raw_conn

CUSTOMER = "smoke-phase2-cust"
NOW = datetime(2026, 5, 14, 12, 0, tzinfo=UTC)


async def main() -> None:
    await init_pool()
    try:
        async with raw_conn() as conn:
            # Wipe prior smoke state.
            await conn.execute("DELETE FROM customers WHERE customer_id = $1", CUSTOMER)
            # Seed (FKs cascade from customers).
            await conn.execute(
                """
                INSERT INTO customers (customer_id, display_name, api_key_hash)
                VALUES ($1, 'phase2 smoke', 'h-' || $1)
                """,
                CUSTOMER,
            )
            # Documents + chunks.
            for doc_id, author in [("d-1", "richardwei6"), ("d-2", "mahit@prbe.ai")]:
                await conn.execute(
                    """
                    INSERT INTO documents (
                        doc_id, version, customer_id,
                        source_system, source_id, source_url,
                        doc_class, doc_type, content_type,
                        content_hash, title, body_size_bytes, body_token_count,
                        created_at, updated_at, valid_from, ingested_at, acl,
                        author_id
                    ) VALUES (
                        $1, 1, $2,
                        'github', $3, 'https://example/' || $1,
                        'raw_source', 'github.commit', 'text/plain',
                        'h-' || $1, 'doc-' || $1, 100, 0,
                        $4, $4, $4, $4, '{}'::jsonb,
                        $5
                    )
                    """,
                    doc_id, CUSTOMER, f"commit:{doc_id}", NOW, author,
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
                    f"{doc_id}:c0", doc_id, CUSTOMER,
                    f"body of {doc_id}", f"chash-{doc_id}",
                )
            # Graph nodes + provenance.
            for canonical, source in [
                ("richardwei6", "github"),
                ("mahit@prbe.ai", "slack"),
                ("U07ABC123", "linear"),
            ]:
                await conn.execute(
                    """
                    INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
                    VALUES ($1, 'Person', $2, jsonb_build_object('name', $2), 1)
                    """,
                    CUSTOMER, canonical,
                )
                await conn.execute(
                    """
                    INSERT INTO graph_node_provenance (
                        customer_id, node_id, source_system,
                        first_seen_at, last_seen_at
                    )
                    SELECT $1, gn.node_id, $3, $4, $4
                    FROM graph_nodes gn
                    WHERE gn.customer_id = $1 AND gn.label = 'Person' AND gn.canonical_id = $2
                    """,
                    CUSTOMER, canonical, source, NOW,
                )
            # Document graph_nodes.
            for doc_id in ("d-1", "d-2"):
                await conn.execute(
                    """
                    INSERT INTO graph_nodes (customer_id, label, canonical_id, properties, degree)
                    VALUES ($1, 'Document', $2, '{}'::jsonb, 1)
                    """,
                    CUSTOMER, doc_id,
                )
            # AUTHORED edges.
            for author, doc_id in [("richardwei6", "d-1"), ("mahit@prbe.ai", "d-2")]:
                await conn.execute(
                    """
                    INSERT INTO graph_edges (
                        customer_id, edge_type,
                        from_node_id, to_node_id,
                        confidence, properties
                    )
                    SELECT $1, 'AUTHORED',
                           p.node_id, d.node_id,
                           'EXTRACTED', '{}'::jsonb
                    FROM graph_nodes p, graph_nodes d
                    WHERE p.customer_id = $1 AND p.label = 'Person'   AND p.canonical_id = $2
                      AND d.customer_id = $1 AND d.label = 'Document' AND d.canonical_id = $3
                    """,
                    CUSTOMER, author, doc_id,
                )
        print(f"Seeded {CUSTOMER} with 3 Person nodes + 2 docs + AUTHORED edges + provenance.")
    finally:
        await close_pool()


if __name__ == "__main__":
    asyncio.run(main())
```

Run:

```bash
.venv/bin/python scripts/smoke_phase2_clusters.py
```

Expected: `Seeded smoke-phase2-cust with ...`.

- [ ] **Step 3: Launch both uvicorn services in the background**

```bash
# Ingestion (merge endpoint) on 9817
.venv/bin/uvicorn services.ingestion.main:app --host 127.0.0.1 --port 9817 > /tmp/phase2-ingest.log 2>&1 &
INGEST_PID=$!
# Retrieval (graph_explore + query) on 8081
.venv/bin/uvicorn services.retrieval.main:app --host 127.0.0.1 --port 8081 > /tmp/phase2-retrieval.log 2>&1 &
RETRIEVAL_PID=$!

sleep 2  # let lifespan init pools

# Sanity: both report healthy. Tail logs if either fails to start.
curl -s http://127.0.0.1:9817/healthz | head -5
curl -s http://127.0.0.1:8081/healthz | head -5
```

If health checks fail, check `tail -50 /tmp/phase2-ingest.log /tmp/phase2-retrieval.log`. Common causes: stale connections from a prior smoke test (kill old uvicorn first: `pkill -f "uvicorn services\."`); migration not at head; env vars missing.

- [ ] **Step 4: Baseline read — pre-merge state**

These should all behave the way they do today (no Phase 2 effect yet because `entity_aliases` is empty).

```bash
# 4a. Anchor on the actual node "mahit@prbe.ai" -> returns its graph.
curl -s -X POST http://127.0.0.1:8081/graph/explore \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: smoke-phase2-cust" \
  -H "Content-Type: application/json" \
  -d '{"mode":"anchor","anchor_node_id":"mahit@prbe.ai"}' | jq '.nodes | length'
# Expect: 2 (Person:mahit@prbe.ai + Document:d-2)
```

- [ ] **Step 5: Merge mahit@prbe.ai + U07ABC123 into richardwei6**

```bash
curl -s -X POST http://127.0.0.1:9817/api/entity-clusters/merge \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: smoke-phase2-cust" \
  -H "Content-Type: application/json" \
  -d '{
    "label":                "Person",
    "primary_canonical_id": "richardwei6",
    "alias_canonical_ids":  ["mahit@prbe.ai", "U07ABC123"],
    "customer_id":          "smoke-phase2-cust",
    "performed_by_user_id": "11111111-1111-1111-1111-111111111111",
    "reason":               "phase2 smoke"
  }' | jq .
```

Expected:

```json
{
  "merge_id": "...",
  "label": "Person",
  "primary_canonical_id": "richardwei6",
  "merged_alias_canonical_ids": ["mahit@prbe.ai", "U07ABC123"]
}
```

Verify alias nodes are gone:

```bash
docker compose exec -T postgres psql -U prbe -d prbe -c \
  "SELECT canonical_id FROM graph_nodes WHERE customer_id='smoke-phase2-cust' AND label='Person';"
# Expect: only 'richardwei6' (the two aliases were hard-deleted at merge time).
```

- [ ] **Step 6: Verify Phase 2 read-side behaviors**

**6a. `/graph/explore` anchor translation:**

```bash
# Hit anchor with the ALIAS canonical_id. Without Phase 2 this would 404.
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8081/graph/explore \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: smoke-phase2-cust" \
  -H "Content-Type: application/json" \
  -d '{"mode":"anchor","anchor_node_id":"mahit@prbe.ai"}'
# Expect: 200

# And the returned graph contains the primary's nodes.
curl -s -X POST http://127.0.0.1:8081/graph/explore \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: smoke-phase2-cust" \
  -H "Content-Type: application/json" \
  -d '{"mode":"anchor","anchor_node_id":"mahit@prbe.ai"}' \
  | jq '.nodes | map(.id)'
# Expect: ["richardwei6", "d-1", "d-2"] (primary + both documents now-attached after edge rewrite).
```

**6b. List-mode author filter expansion:**

```bash
# Build a /query request that filters by author_id=mahit@prbe.ai.
# The router would normally extract this from natural language; we
# pre-construct the routed entities by hitting an explicit endpoint or
# faking via the dispatcher. Easiest path: invoke /query with q that
# trivially extracts the person, and assert d-1 (authored by primary)
# AND d-2 (authored by alias) both appear.
#
# If routing extracts unreliably, fall back to a Python REPL one-liner:
.venv/bin/python -c "
import asyncio
from services.retrieval.list_pipeline import run_list
from services.retrieval.router import RouterEntity, RouterOutput
from shared.models import QueryRequest, TemporalSpec
async def main():
    req = QueryRequest(q='by mahit', top_k=20, entity_must_match=True)
    routed = RouterOutput(operation='list',
        entities=[RouterEntity(entity_type='person', canonical_id='mahit@prbe.ai',
                               display_name='Mahit', confidence=0.9)])
    resp = await run_list(req=req, customer_id='smoke-phase2-cust', routed=routed,
                          spec=TemporalSpec(), temporal_meta={}, sort_meta=None,
                          extracted_entities=[], doc_types=None,
                          trace_id='smoke', timing={})
    print(sorted(d.doc_id for d in resp.documents))
asyncio.run(main())
"
# Expect: ['d-1', 'd-2']
# (d-1 was authored by richardwei6, d-2 by mahit@prbe.ai. Without Phase 2
# expansion the author_id='mahit@prbe.ai' filter would only return d-2.)
```

**6c. `RelatedEntity` cluster fields:**

```bash
# Query that surfaces the Person primary as a related entity. Easiest:
# request docs and let the walker run.
.venv/bin/python -c "
import asyncio
from services.retrieval.retrievers.related_entities import walk_result_doc_neighbors
async def main():
    rels = await walk_result_doc_neighbors(
        customer_id='smoke-phase2-cust',
        ranked_result_docs=[('d-1', 1), ('d-2', 2)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    for r in rels:
        if r.label == 'Person':
            print(f'canonical={r.canonical_id} member_count={r.member_count} '
                  f'member_sources={sorted(r.member_sources)} display_name={r.display_name}')
asyncio.run(main())
"
# Expect: canonical=richardwei6 member_count=3 member_sources=['github','linear','slack'] display_name=richardwei6
```

**6d. Display-name override:**

```bash
docker compose exec -T postgres psql -U prbe -d prbe -c \
  "INSERT INTO entity_cluster_metadata (customer_id, label, primary_canonical_id, display_name)
   VALUES ('smoke-phase2-cust', 'Person', 'richardwei6', 'Richard Wei (canonical)');"

# Re-run the related_entities query — display_name now reflects the override.
.venv/bin/python -c "
import asyncio
from services.retrieval.retrievers.related_entities import walk_result_doc_neighbors
async def main():
    rels = await walk_result_doc_neighbors(
        customer_id='smoke-phase2-cust',
        ranked_result_docs=[('d-1', 1), ('d-2', 2)],
        exclude_node_keys=set(),
        min_confidence=None,
        top_n=10,
    )
    person = next(r for r in rels if r.label == 'Person')
    print(person.display_name)
asyncio.run(main())
"
# Expect: Richard Wei (canonical)
```

**6e. Routed-entity translation (search pipeline):**

```bash
.venv/bin/python -c "
import asyncio
from services.retrieval.router import RouterEntity, RouterOutput
from services.retrieval.search_pipeline import _build_entity_results
async def main():
    routed = RouterOutput(operation='search',
        entities=[RouterEntity(entity_type='person', canonical_id='mahit@prbe.ai',
                               display_name='Mahit', confidence=0.9)])
    results = await _build_entity_results(customer_id='smoke-phase2-cust',
                                          routed=routed, timing={})
    for r in results:
        print(f'canonical={r.canonical_id} display_name={r.display_name}')
asyncio.run(main())
"
# Expect: canonical=richardwei6 display_name=Richard Wei (canonical)
# (Without Phase 2 the alias miss would print nothing.)
```

- [ ] **Step 7: Verify post-partial-unmerge reverts to per-alias behavior for the unmerged member**

```bash
# Unmerge mahit@prbe.ai only — U07ABC123 stays in the cluster.
curl -s -o /dev/null -w "%{http_code}\n" -X DELETE \
  "http://127.0.0.1:9817/api/entity-clusters/Person/richardwei6/aliases/mahit@prbe.ai" \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: smoke-phase2-cust"
# Expect: 204

# Anchor on mahit@prbe.ai now returns mahit's OWN graph (not richardwei6's).
curl -s -X POST http://127.0.0.1:8081/graph/explore \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: smoke-phase2-cust" \
  -H "Content-Type: application/json" \
  -d '{"mode":"anchor","anchor_node_id":"mahit@prbe.ai"}' \
  | jq '.nodes | map(.id)'
# Expect: ["mahit@prbe.ai", "d-2"]  -- restored from snapshot.

# Anchor on U07ABC123 still translates to richardwei6 (still aliased).
curl -s -X POST http://127.0.0.1:8081/graph/explore \
  -H "X-Internal-Knowledge-Key: $INTERNAL_KNOWLEDGE_API_KEY" \
  -H "X-Prbe-Customer: smoke-phase2-cust" \
  -H "Content-Type: application/json" \
  -d '{"mode":"anchor","anchor_node_id":"U07ABC123"}' \
  | jq '.nodes | map(.id)'
# Expect: includes "richardwei6"; not "U07ABC123" (which is hard-deleted)
```

- [ ] **Step 8: Cleanup**

```bash
kill $INGEST_PID $RETRIEVAL_PID 2>/dev/null || true
wait $INGEST_PID $RETRIEVAL_PID 2>/dev/null || true

docker compose exec -T postgres psql -U prbe -d prbe -c \
  "DELETE FROM customers WHERE customer_id = 'smoke-phase2-cust';"
```

- [ ] **Step 9: Write up the smoke-test results**

Append a `<details>` block to the Phase 2 PR body listing:
- What was seeded
- Which endpoints/functions were exercised (6a, 6b, 6c, 6d, 6e, 7)
- The observed response for each (one line per check: "✅ /graph/explore alias anchor → 200 + ['richardwei6','d-1','d-2']")
- Any deviations from expected output (these become follow-up tickets, not silent passes)

- [ ] **Step 10: Commit the seed script**

```bash
git add scripts/smoke_phase2_clusters.py
git commit -m "test(scripts): phase2 smoke seed script

Reproduces the Phase 2 verification fixture (3 Person nodes + 2 docs +
provenance). Idempotent — wipes 'smoke-phase2-cust' before re-seeding."
```

---

## Self-review

**1. Spec coverage:**

| Spec requirement (from design doc §"Read-side behavior (Phase 2 preview)") | Task |
|---|---|
| Graph anchor lookup translation | Task 3 |
| Author filter expansion (list mode) | Task 4 |
| `RelatedEntity.member_count` / `member_sources` | Tasks 2 (model) + 5 (population) |
| `entity_cluster_metadata` display name override | Task 5 (RelatedEntity) + Task 6 (QueryEntityResult) |
| Shared `resolve_aliases()` helper | Task 1 |

| Implicit requirement | Task |
|---|---|
| Search-pipeline routed-entity translation (alias inputs must land on primary) | Task 6 |
| `exclude_node_keys` translation so typed aliases exclude the cluster | Task 5 |
| End-to-end live-service verification (per user feedback: "spinning up a docker instance and merging nodes then reading nodes actually worked") | Task 8 |

No gaps.

**2. Placeholder scan:**

- No "TBD"/"TODO"/"implement later" tokens in the plan.
- No "add appropriate error handling" — error paths are concrete (404 on missing anchor, etc).
- No "write tests for the above" — every task has explicit test code.
- No "similar to Task N" — code blocks repeat in full per task.
- Two callouts that say "if your branch differs, adapt" — these are about `RouterOutput`/`RouterEntity` shape and the exact name of the `_build_entity_results` function. These are necessary because:
  - `RouterOutput` is project-internal and may have evolved since the plan was drafted.
  - Function name in `search_pipeline.py` is `_build_entity_results` per the Phase 1 audit but the line range was inferred. The implementer should grep to confirm; this is normal codebase navigation, not a placeholder.

**3. Type consistency:**

- `resolve_aliases(conn, customer_id, refs: list[tuple[str, str]]) -> dict[tuple[str, str], str]` — used identically across Tasks 1, 5, 6.
- `expand_to_cluster_members(conn, customer_id, label, canonical_ids: list[str]) -> dict[str, list[str]]` — used identically across Tasks 1, 4.
- `RelatedEntity.member_count: int` (default 1) and `RelatedEntity.member_sources: list[str]` (default []) — populated by Task 5 SQL with `int` and `text[]` respectively. asyncpg returns native types for both.
- `entity_aliases` PK is `(customer_id, label, alias_canonical_id)` — every SELECT uses these three columns or label-less + LIMIT 1 (Task 3 anchor translation).
- `entity_cluster_metadata` PK is `(customer_id, label, primary_canonical_id)` — every LEFT JOIN uses all three.
- `graph_node_provenance` projection: `(customer_id, node_id, source_system, ...)` — joins on `(customer_id, node_id)`.

No type drift.

---

## Out of scope (deferred)

- **Cluster-aware edge aggregation** (e.g. summing `commit_count` across alias lanes in `RelatedEntity.edge_properties_summary`). Design doc Open Items #4. Phase 2.5+.
- **Anchor label-context propagation**: today the anchor endpoint is label-less. If a customer ever has two clusters with colliding alias canonical_ids under different labels, we pick one arbitrarily. Phase 2 keeps the current semantics; a future ticket can add label-aware anchor resolution if the collision becomes real.
- **`graph_search` typeahead alias merging**: `services/retrieval/main.py:644-659` (`/graph/search`) returns ranked prefix matches against `graph_nodes`. Post-merge, alias rows are gone so they won't appear in typeahead — this is acceptable (users won't search for canonical_ids they've already retired). Adding alias-as-typeahead would require querying `entity_aliases` UNION `graph_nodes` and is out of Phase 2 scope.
- **MCP tool description update** noting that merged entities show `member_count > 1`. Design doc Open Items #6. Phase 4.

---

**Plan complete. Ready for execution via `superpowers:subagent-driven-development`.**
