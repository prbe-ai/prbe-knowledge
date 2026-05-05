# prbe-knowledge

Operational-memory data plane for Probe. This repo ingests Slack, GitHub,
Linear, Notion, Sentry, Granola, coding-agent transcripts, manual uploads, and
generated wiki pages into a tenant-scoped knowledge store, then serves retrieval
and synthesis to the dashboard, MCP server, and orchestrator.

**Status:** active production service. Public webhooks, OAuth install URLs, and
coding-agent device auth terminate in `prbe-backend`; this repo receives trusted
internal forwards plus per-tenant retrieval requests.

## Architecture

Runtime services:

| Service | Fly app | Responsibility |
|---|---|---|
| Ingestion API | `prbe-knowledge-ingestion` | Internal webhook receiver, manual uploads, OAuth token exchange, device registry, integration/admin/wiki APIs |
| Retrieval API | `prbe-knowledge-retrieval` | `/retrieve`, `/query`, `/query/stream`, source reads, usage logging |
| Worker | `prbe-knowledge-worker` | Drains `ingestion_queue`, persists raw payloads, normalizes docs/chunks/graph/ACLs, embeds content |
| Poller | `prbe-knowledge-poller` | Polling-only connector wakeups, currently Granola |
| Wiki triage | `prbe-knowledge-wiki-worker` | Nightly/manual triage of events into wiki synthesis candidates |
| Wiki synthesis | `prbe-knowledge-wiki-synthesis` | Agent-driven wiki page updates, verification, and index regeneration |
| Wiki cron | `prbe-knowledge-wiki-cron` | One-shot 02:00 UTC trigger for opted-in tenants with pending wiki events |

Data stores:

- Neon Postgres 16 + pgvector: documents, chunks, graph, ACL snapshots,
  integration tokens, queues, usage events, query traces, wiki pages, and
  customer preferences.
- Cloudflare R2 in production, MinIO locally: raw webhook archives and staged
  manual-upload originals.
- OpenAI embeddings (`text-embedding-3-large`), Anthropic routing/extraction,
  and Gemini-backed wiki synthesis where configured.

Primary flows:

1. `prbe-backend` verifies source signatures, OAuth state, or coding-agent
   bearer devices, resolves the tenant, and forwards to this service with
   `X-Internal-Knowledge-Key` and `X-Prbe-Customer`.
2. Ingestion validates the trusted headers, persists the raw event, and inserts
   an idempotent queue row.
3. Workers normalize source-specific payloads into documents, chunks, graph
   nodes/edges, ACL snapshots, and embeddings.
4. Retrieval routes each query through list/search modes, combines vector,
   BM25, graph, temporal, and entity filters, then optionally synthesizes a
   cited answer.
5. Wiki workers periodically triage slow-moving company knowledge and update
   generated service cards, decisions, features, runbooks, and the wiki index.

Design docs:

- [`docs/phase0-design.md`](docs/phase0-design.md)
- [`docs/storage-architecture.md`](docs/storage-architecture.md)
- [`docs/roadmap.md`](docs/roadmap.md)
- [`scripts/synth/README.md`](scripts/synth/README.md)
- [`tests/evals/README.md`](tests/evals/README.md)

## Supported sources

| Source | Ingestion shape |
|---|---|
| Slack | Events, messages, threads, users, channels |
| GitHub | Pull requests, issues, commits, reviews, CODEOWNERS |
| Linear | Issues and comments |
| Notion | Pages, databases, blocks |
| Sentry | Issues and event alerts |
| Granola | Polling/backfill of meeting notes |
| Claude Code | Paired-device transcript batches and extracted units |
| Codex | Paired-device transcript batches using the same agent transcript shape |
| Manual uploads | Dashboard-uploaded text, markdown, docx, and files |
| Wiki | Manual and generated service cards, decisions, features, runbooks, and index |

## Local development

Prerequisites:

- Python 3.12+
- Docker + Docker Compose
- `psql` client: `brew install libpq && brew link --force libpq`

```bash
# 1. Start local Postgres + MinIO
docker compose up -d

# 2. Create venv + install deps
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 3. Copy env template and fill provider/internal secrets as needed
cp .env.example .env.local

# 4. Apply the schema to local Postgres
scripts/neon-migrate.sh local
```

Local Postgres runs at `localhost:5432` with user `prbe`, password `prbe`, and
database `prbe_knowledge`. MinIO runs at `localhost:9000`; its console is
`localhost:9001` with `minioadmin` / `minioadmin`.

Run services locally:

```bash
# Ingestion API
.venv/bin/uvicorn services.ingestion.main:app --reload --port 8080

# Retrieval API
.venv/bin/uvicorn services.retrieval.main:app --reload --port 8081

# Queue worker
.venv/bin/python -m services.ingestion.worker

# Granola poller
.venv/bin/python -m services.ingestion.poller

# Wiki workers
.venv/bin/python -m services.synthesis.triage_app
.venv/bin/python -m services.synthesis.synthesis_app
```

## Common commands

```bash
# Run all tests
.venv/bin/pytest tests/ -q

# Lint
.venv/bin/ruff check .

# Type-check, currently soft-gated in CI
.venv/bin/mypy shared services

# Open psql for a branch
scripts/neon-psql.sh local
scripts/neon-psql.sh dev

# Run migrations
scripts/neon-migrate.sh local
scripts/neon-migrate.sh dev
scripts/neon-migrate.sh staging
scripts/neon-migrate.sh main
```

On a fresh local DB, `scripts/neon-migrate.sh local` provisions a minimal
`neon_auth` shim, applies `db/schema.sql` directly, and stamps Alembic head.
Existing DBs run incremental Alembic migrations.

## Deploying schema to Neon

The managed branches are `dev`, `staging`, and `main`. Their connection strings
are stored in the macOS Keychain so secrets do not land in shell history,
`.env` files, or git.

One-time setup per branch:

1. Neon console -> project `prbe-knowledge` -> pick the branch.
2. Click **Connect** and copy the full `postgresql://...` URL.
3. Run `scripts/neon-store.sh dev`, `scripts/neon-store.sh staging`, or
   `scripts/neon-store.sh main`.

Apply migrations:

```bash
scripts/neon-migrate.sh dev
scripts/neon-migrate.sh staging
scripts/neon-migrate.sh main
```

Interactive verification:

```bash
scripts/neon-psql.sh dev

\dx
\dt
SELECT version_num FROM alembic_version;
\q
```

### Reset a branch

```bash
scripts/neon-reset.sh dev
```

This is destructive. It drops Phase 0 tables and re-runs migrations. Use it only
for local/dev data you are willing to lose.

## Syncing Fly secrets

Local `.env` values live on your laptop; Fly apps need secrets set per app.

```bash
# Sync all seven apps from .env
scripts/fly-secrets-sync.sh

# Sync one app
scripts/fly-secrets-sync.sh ingestion
scripts/fly-secrets-sync.sh retrieval
scripts/fly-secrets-sync.sh worker
scripts/fly-secrets-sync.sh poller
scripts/fly-secrets-sync.sh wiki-worker
scripts/fly-secrets-sync.sh wiki-synthesis
scripts/fly-secrets-sync.sh wiki-cron

# Use a different env file
scripts/fly-secrets-sync.sh -f .env.staging
```

The script uses `flyctl secrets set`, decodes dotenv-style escaped multiline
values, and is safe to re-run after rotations.

## Deployment

GitHub Actions deploys changed services from `main` with path filters:

- `fly.ingestion.toml`
- `fly.retrieval.toml`
- `fly.worker.toml`
- `fly.poller.toml`
- `fly.wiki-worker.toml`
- `fly.wiki-synthesis.toml`
- `fly.wiki-cron.toml`

The ingestion deploy runs `alembic upgrade head` as the Fly release command.
The poller, wiki-worker, and wiki-synthesis jobs re-assert expected Fly machine
counts after deploy.

## Repo layout

```
prbe-knowledge/
|-- services/
|   |-- ingestion/      internal ingestion API, connector handlers, worker, poller
|   |-- retrieval/      retrieval, source reads, synthesis, usage logging
|   |-- synthesis/      wiki triage, agent loop, persistence, nightly trigger
|   `-- system_settings/
|-- shared/             config, models, db/storage helpers, constants, auth helpers
|-- db/
|   |-- schema.sql      canonical latest Postgres schema
|   `-- migrations/     Alembic migrations
|-- scripts/            Neon/Fly ops, backfills, synthetic data tools
|-- docs/               design, roadmap, storage, superpower specs
|-- fixtures/           sample source payloads
|-- tests/              unit, handler, retrieval, migration, and eval tests
|-- docker-compose.yml  local Postgres + MinIO
`-- fly.*.toml          Fly app configs
```
