"""Retrieval-test fixtures: seeded customer with graph nodes for grounding tests."""

from __future__ import annotations

from dataclasses import dataclass

import pytest_asyncio

import shared.db as db_module


@dataclass
class SeededCustomer:
    customer_id: str


@pytest_asyncio.fixture
async def seeded_customer(live_db) -> SeededCustomer:
    """Customer with one Feature, one Repo, one Ticket, one PR; one github source mapping."""
    customer_id = "test-cust-grounding-1"
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'Test Grounding Customer', $2)
            """,
            customer_id,
            "hash-" + customer_id,
        )
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES
              ($1, 'Feature', 'auth-refactor', '{"name":"auth refactor"}'::jsonb),
              ($1, 'Repo', 'prbe-backend', '{"name":"prbe-backend"}'::jsonb),
              ($1, 'Ticket', 'ABC-123', '{"name":"Fix login flow"}'::jsonb),
              ($1, 'PR', '49', '{"name":"PR #49: refactor session handling"}'::jsonb)
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO customer_source_mapping (source_system, external_id, customer_id)
            VALUES ('github', 'test-ext-1', $1)
            """,
            customer_id,
        )
    return SeededCustomer(customer_id=customer_id)


@pytest_asyncio.fixture
async def seeded_customer_with_docs(live_db) -> SeededCustomer:
    """Customer with documents whose titles exercise _fuzzy_match_document_titles.

    Two live docs (PRB-18 Linear + Notion design rationale) anchor on the
    'multi-granola' concept. One soft-deleted doc (valid_to != NULL)
    verifies the live filter. One cross-tenant doc (different customer_id)
    verifies customer scoping. One wiki doc verifies wiki is included.
    """
    from datetime import UTC, datetime
    customer_id = "test-cust-doc-titles-1"
    other_customer_id = "test-cust-doc-titles-other"
    now = datetime.now(UTC)
    async with db_module.raw_conn() as conn:
        for cid in (customer_id, other_customer_id):
            await conn.execute(
                """
                INSERT INTO customers (customer_id, display_name, api_key_hash)
                VALUES ($1, 'Test Doc Titles ' || $1, $2)
                """,
                cid,
                "hash-" + cid,
            )
        # Doc 1: Linear PRB-18 (live) - canonical multi-granola plan
        # Doc 2: Notion design rationale (live)
        # Doc 3: Wiki page (live, NOT excluded)
        # Doc 4: Soft-deleted older version (valid_to set)
        # Doc 5: Cross-tenant doc with same title (different customer_id)
        await conn.executemany(
            """
            INSERT INTO documents (
                doc_id, version, customer_id,
                source_system, source_id, source_url,
                doc_class, doc_type, content_type,
                content_hash, title, body_preview, body_size_bytes, body_token_count,
                created_at, updated_at, valid_from, valid_to, ingested_at, acl
            ) VALUES (
                $1, 1, $2, $3, $4, $5,
                'raw_source', $6, 'text/markdown',
                'h-' || $1, $7, $8, 100, 0,
                $9, $9, $9, $10, $9, '{}'::jsonb
            )
            """,
            [
                ("linear:org:issue:prb-18", customer_id, "linear", "prb-18",
                 "https://linear.app/x/PRB-18", "linear.issue",
                 "Multi-Granola — End-to-End Implementation Plan",
                 "Probe Founders is a single customer_id; Granola Personal-tier keys are per-user.",
                 now, None),
                ("notion:page:design-mg", customer_id, "notion", "design-mg",
                 "https://notion.so/Multi-Granola-Design-Rationale", "notion.page",
                 "Multi-Granola — Design Rationale & Open Questions",
                 "Alternatives considered: shared team key, server-side rotation.",
                 now, None),
                ("wiki:project:multi_granola", customer_id, "wiki", "multi_granola",
                 "/wiki/project/multi_granola", "wiki.project",
                 "Multi-Granola feature index",
                 "Synthesized navigational hub for multi-granola work.",
                 now, None),
                ("linear:org:issue:prb-old", customer_id, "linear", "prb-old",
                 "https://linear.app/x/PRB-OLD", "linear.issue",
                 "Multi-Granola — Soft Deleted",
                 "This version was superseded; valid_to is set.",
                 now, now),  # valid_to=now → soft-deleted
                ("linear:org:issue:prb-cross", other_customer_id, "linear", "prb-cross",
                 "https://linear.app/x/PRB-CROSS", "linear.issue",
                 "Multi-Granola — Other Tenant Copy",
                 "Belongs to a different customer; must NOT leak.",
                 now, None),
                # Doc 6: title has no trigram overlap with "kubernetes",
                # but body_preview mentions it. Used to verify the
                # tsvector/FTS-only path (when trgm_sim < floor but
                # fts_hit=1). Title was deliberately chosen with no
                # overlapping trigrams to "kubernetes" so the trgm
                # similarity stays below the 0.15 floor.
                ("notion:page:onboarding", customer_id, "notion", "onboarding",
                 "https://notion.so/Onboarding-Runbook", "notion.page",
                 "Onboarding Runbook",
                 "First-week setup for new engineers; includes cluster access via kubernetes.",
                 now, None),
            ],
        )
    return SeededCustomer(customer_id=customer_id)


@pytest_asyncio.fixture
async def seeded_customer_many_repos(live_db) -> SeededCustomer:
    customer_id = "test-cust-grounding-many"
    async with db_module.raw_conn() as conn:
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'Test Grounding Many', $2)
            """,
            customer_id,
            "hash-" + customer_id,
        )
        for i in range(12):
            await conn.execute(
                """
                INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
                VALUES ($1, 'Repo', $2, $3::jsonb)
                """,
                customer_id, f"prbe-svc-{i}", f'{{"name":"prbe-svc-{i}"}}',
            )
        await conn.execute(
            """
            INSERT INTO customer_source_mapping (source_system, external_id, customer_id)
            VALUES ('github', 'test-ext-many', $1)
            """,
            customer_id,
        )
    return SeededCustomer(customer_id=customer_id)
