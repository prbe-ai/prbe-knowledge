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
