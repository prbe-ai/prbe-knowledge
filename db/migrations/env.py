"""Alembic environment. Reads DATABASE_URL_SYNC from the process env."""

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, text

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

database_url = os.environ.get(
    "DATABASE_URL_SYNC",
    "postgresql://prbe:prbe@localhost:5432/prbe_knowledge",
)
config.set_main_option("sqlalchemy.url", database_url)

target_metadata = None

# Pin the migrator session's search_path to public-first. AGE's
# ``CREATE EXTENSION age`` prepends ``ag_catalog`` for the session that
# runs it (and the migrate role's default may also have it first), so
# without this every unqualified ``CREATE TABLE`` lands in ag_catalog
# — that's how migrations 0001-0066 ended up creating ~30 tables in
# ag_catalog instead of public. Public-first means new tables land in
# public; ag_catalog stays in the list so AGE's own catalog access
# (e.g. ``ag_label``) still resolves unqualified within migration
# bodies. See migration ``0071_move_app_tables_to_public`` for the
# sweep that fixed the pre-existing drift.
_PIN_SEARCH_PATH_SQL = "SET search_path = public, ag_catalog"


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        if connection.dialect.name == "postgresql":
            connection.execute(text(_PIN_SEARCH_PATH_SQL))
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
