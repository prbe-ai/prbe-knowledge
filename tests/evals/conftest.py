"""Eval-only fixtures: a fixed-graph customer for the router quality harness.

Override the root conftest's force-empty ANTHROPIC_API_KEY at import time
so the eval can actually call Haiku. Root conftest sets it to "" to prevent
accidental API calls during ordinary `pytest`; eval is opt-in via `-m eval`
and explicitly needs a real key. We parse .env directly (python-dotenv is
not a dep) and restore the value if present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import pytest_asyncio


def _restore_anthropic_key_from_env_file() -> None:
    """Parse .env / .env.local for ANTHROPIC_API_KEY and overwrite the
    empty string the root conftest force-set. No-op if no .env or no key."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    here = Path(__file__).resolve()
    # Walk up to the repo root looking for .env / .env.local
    for candidate_dir in (here.parent.parent.parent,):  # tests/evals → tests → repo
        for fname in (".env.local", ".env"):
            fp = candidate_dir / fname
            if not fp.is_file():
                continue
            for line in fp.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    val = line.split("=", 1)[1].strip().strip('"').strip("'")
                    if val:
                        os.environ["ANTHROPIC_API_KEY"] = val
                        # The Settings cache was warmed by the root conftest
                        # with the empty key — invalidate so the next call
                        # to get_settings() picks up the restored value.
                        try:
                            from shared.config import get_settings
                            get_settings.cache_clear()  # type: ignore[attr-defined]
                        except Exception:
                            pass
                        return


_restore_anthropic_key_from_env_file()

import shared.db as db_module  # noqa: E402


@dataclass
class SeededCustomer:
    customer_id: str


@pytest_asyncio.fixture
async def eval_seeded_customer(live_db) -> SeededCustomer:
    """Customer with the fixed graph that router_quality_fixtures.yaml expects.

    Inserts: 1 Feature (auth-refactor), 1 Repo (prbe-backend), 1 Ticket (ABC-123),
    1 PR (49), 1 Person (mahit). Customer is mapped to github+linear+slack sources.
    """
    customer_id = "eval_demo"
    async with db_module.raw_conn() as conn:
        # customers table requires display_name + api_key_hash per Task 2 findings.
        await conn.execute(
            """
            INSERT INTO customers (customer_id, display_name, api_key_hash)
            VALUES ($1, 'eval-demo display', 'eval-demo-hash')
            ON CONFLICT (customer_id) DO NOTHING
            """,
            customer_id,
        )
        # graph_nodes: names live in properties->>'name', NOT a top-level column.
        await conn.execute(
            """
            INSERT INTO graph_nodes (customer_id, label, canonical_id, properties)
            VALUES
              ($1, 'Feature', 'auth-refactor', '{"name":"auth refactor"}'::jsonb),
              ($1, 'Repo', 'prbe-backend', '{"name":"prbe-backend"}'::jsonb),
              ($1, 'Ticket', 'ABC-123', '{"name":"Fix login flow"}'::jsonb),
              ($1, 'PR', '49', '{"name":"PR #49: refactor session handling"}'::jsonb),
              ($1, 'Person', 'mahit', '{"name":"Mahit"}'::jsonb)
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )
        await conn.execute(
            """
            INSERT INTO customer_source_mapping (source_system, external_id, customer_id)
            VALUES
              ('github', 'eval-gh', $1),
              ('linear', 'eval-linear', $1),
              ('slack', 'eval-slack', $1)
            ON CONFLICT DO NOTHING
            """,
            customer_id,
        )
    return SeededCustomer(customer_id=customer_id)
