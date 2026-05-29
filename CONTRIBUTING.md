# Contributing

Thanks for contributing to the Probe knowledge engine. This is the open-source
single-tenant engine (ingestion → retrieval → knowledge-page synthesis).

## Dev setup

Requires Python 3.12+ and Docker.

```bash
# Dependencies (uv is supported — a uv.lock is checked in):
uv sync --extra dev
# or with a plain venv:
python -m venv .venv && .venv/bin/pip install -e ".[dev]"

# Bring up Postgres + MinIO + the services locally:
cp .env.example .env     # fill GOOGLE_API_KEY + one LLM key + KNOWLEDGE_API_TOKEN
make up
make health
```

On Apple Silicon, run the stack on `linux/amd64`
(`export DOCKER_DEFAULT_PLATFORM=linux/amd64`) — see
[docs/self-hosting.md](docs/self-hosting.md#prerequisites) for why.

## Running locally

```bash
make up          # build + start the full stack
make logs        # tail all services
make migrate     # re-run the DB migration (idempotent)
make query Q="what changed last week?"
make down        # stop (keeps volumes)
```

The schema lives in `db/schema.sql` (the source of truth); `scripts/migrate.py`
applies it on a fresh database and stamps the alembic head, or runs
`alembic upgrade head` on an existing one. If you change the schema, update both
`db/schema.sql` and add an alembic migration so fresh and existing databases stay
in sync.

## Gates

Two checks run in CI (`.github/workflows/tests.yml`); run them before opening a
PR:

```bash
ruff check .                 # lint — must pass
mypy shared services         # type-check (strict; currently soft-gated)
```

Config lives in `pyproject.toml` (`ruff` line-length 100, `mypy` strict).

## PR norms

- Keep changes focused and minimal — touch only what the change needs.
- Match existing patterns: connectors register via `@register_connector` in
  `services/ingestion/handlers/`; retrieval tuning constants live in
  `shared/constants.py`, not env vars.
- Make sure `ruff check .` is clean.
- Describe what changed and how you verified it. If it affects ingestion or
  retrieval behavior, note how you tested against the local Compose stack.
