"""Generic database migration for self-host / community edition.

Replaces the Neon/Fly-specific ``scripts/neon-migrate.sh`` for anyone running
the engine with plain ``DATABASE_URL``. Behaviour matches the canonical local
bootstrap:

  1. Provision a minimal ``neon_auth`` stand-in — ``customers.organization_id``
     has an FK to ``neon_auth.organization`` (a Neon Auth table that does not
     exist on a self-hosted Postgres). The shim gives the FK a target; the
     column stays nullable, so single-tenant mode never populates it.
  2. Fresh DB (no ``alembic_version``)  -> apply ``db/schema.sql`` (the
     canonical latest schema) and stamp the alembic head. We do NOT replay the
     migration chain on a fresh DB: migrations 0007+ duplicate state that
     ``schema.sql`` already creates, which ``alembic upgrade head`` chokes on.
  3. Existing DB -> ``alembic upgrade head`` (incremental).

Env:
  DATABASE_URL       asyncpg DSN, e.g. postgresql://user:pass@host:5432/db
  DATABASE_URL_SYNC  alembic DSN, e.g. postgresql+psycopg://user:pass@host:5432/db
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import asyncpg

_NEON_AUTH_SHIM = """
CREATE SCHEMA IF NOT EXISTS neon_auth;
CREATE TABLE IF NOT EXISTS neon_auth.organization (id UUID PRIMARY KEY);
CREATE TABLE IF NOT EXISTS neon_auth."user" (
    id UUID PRIMARY KEY,
    organization_id UUID REFERENCES neon_auth.organization(id),
    email TEXT,
    name TEXT
);
"""


async def _prepare_schema() -> bool:
    """Apply the neon_auth shim and, on a fresh DB, db/schema.sql.

    Returns True if the DB was fresh (schema.sql applied), False otherwise.
    """
    dsn = os.environ["DATABASE_URL"]
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(_NEON_AUTH_SHIM)
        has_alembic = await conn.fetchval(
            "SELECT to_regclass('public.alembic_version') IS NOT NULL"
        )
        if not has_alembic:
            schema = Path("db/schema.sql").read_text()
            await conn.execute(schema)
            return True
        return False
    finally:
        await conn.close()


def main() -> None:
    fresh = asyncio.run(_prepare_schema())

    # alembic's env.py reads DATABASE_URL_SYNC from the environment.
    from alembic import command
    from alembic.config import Config

    cfg = Config("alembic.ini")
    if fresh:
        command.stamp(cfg, "head")
        print("migrate: fresh DB — applied db/schema.sql + stamped alembic head")
    else:
        command.upgrade(cfg, "head")
        print("migrate: existing DB — ran alembic upgrade head")


if __name__ == "__main__":
    main()
