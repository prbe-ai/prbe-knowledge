# prbe-knowledge

Phase 0 knowledge layer for PRBE. Multi-source ingestion (Slack, Linear, GitHub, Notion, Sentry) + retrieval service for agent orchestration.

**Status:** scaffold.

## Architecture

Two FastAPI services deployed to Fly.io:

- `services/ingestion/` — webhook fast path (R2 persist + queue insert) + async worker (parse → fetch → normalize → chunk → embed → upsert)
- `services/retrieval/` — `/query` endpoint: router (Haiku entity extraction) + parallel retrieval (pgvector HNSW + BM25 + relational graph) + RRF fusion + dedup

Storage:

- Neon Postgres 16 + pgvector (chunks, documents, graph, ACL snapshots, queue)
- Cloudflare R2 bucket-per-tenant (raw webhook payloads)

- Full design: [`docs/phase0-design.md`](docs/phase0-design.md)
- Build roadmap (Phase 0 tiers + Phase 1-4+): [`docs/roadmap.md`](docs/roadmap.md)

---

## Prerequisites

- Python 3.12+
- Docker + Docker Compose (for local dev)
- `psql` client — `brew install libpq && brew link --force libpq`

## Local dev setup

```bash
# 1. Start local Postgres + MinIO
docker compose up -d

# 2. Create venv + install deps
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"

# 3. Apply schema migration to local Postgres
scripts/neon-migrate.sh local
```

Local Postgres runs at `localhost:5432` (user `prbe`, pass `prbe`, db `prbe_knowledge`).
MinIO runs at `localhost:9000` (console `localhost:9001`, user `minioadmin`/`minioadmin`).

---

## Deploying the schema to Neon

Three branches: `dev`, `staging`, `main`. Each has its own compute endpoint and its own connection string.

Connection strings are stored in the **macOS Keychain** so they never land in shell history, `.env` files, or git.

### One-time setup per branch

For each branch:

1. Neon console → project `prbe-knowledge` → branch dropdown (top-left) → pick the branch
2. Click **Connect** → copy the full `postgresql://...` URL (Cmd+C)
3. Run:
   ```bash
   scripts/neon-store.sh dev        # or staging, or main
   ```

The script reads the URL from the macOS clipboard (pbpaste), validates it starts with `postgresql://`, strips any accidental line breaks from UI wrapping, and stores it in the macOS Keychain under service `neon-prbe-knowledge`, account `<branch>`.

It prints the parsed hostname and URL length so you can confirm the right branch landed:

```
Stored in Keychain.
  Service:    neon-prbe-knowledge
  Account:    dev
  Host:       ep-lively-star-am6vuivm-pooler.c-5.us-east-1.aws.neon.tech
  URL length: 147 chars
```

A URL length under 120 chars is flagged as suspicious — usually a truncated paste.

To overwrite after rotating the password, copy the new URL and re-run the same command. The script uses `-U` (update if exists) so there's no delete step.

### Apply the migration

```bash
scripts/neon-migrate.sh dev        # migrate dev branch
scripts/neon-migrate.sh staging    # migrate staging branch
scripts/neon-migrate.sh main       # migrate prod
scripts/neon-migrate.sh local      # migrate local docker-compose instead
```

### Reset a branch (pre-launch only, DESTRUCTIVE)

```bash
scripts/neon-reset.sh dev
```

Drops all Phase 0 tables on the target and re-runs the migration from scratch. Use this when the schema is still fluid and you'd rather rebuild clean than write an alter-table migration. Prompts for confirmation — you have to type the branch name to proceed.

**Do not run this after real customer data exists.** Once production data lands, changes go through forward-only Alembic migrations (new revision files), not a reset.

Under the hood it:

1. Pulls the connection string from Keychain
2. Swaps the scheme from `postgresql://` → `postgresql+psycopg://` (Alembic needs psycopg3)
3. Runs `.venv/bin/alembic upgrade head` with that URL as `DATABASE_URL_SYNC`

Expected output:

```
Migrating Neon branch: dev
INFO  [alembic.runtime.migration] Running upgrade  -> 0001_initial_schema, initial schema
Done.
```

### Interactive psql against a branch

```bash
scripts/neon-psql.sh dev
```

Opens a `psql` session connected to the named branch. Useful psql commands at the prompt:

```
\dt                                  list tables
\d chunks                            describe chunks table
\di idx_chunks_embedding_hnsw        describe HNSW index
\dx                                  list installed extensions
SELECT version_num FROM alembic_version;
\q                                   quit
```

Note: psql meta-commands start with **backslash** (`\`), not forward slash (`/`).

### Verification (expected state post-migration)

From `scripts/neon-psql.sh <branch>`:

| Check | Expected |
|---|---|
| `\dt` | 14 tables (13 Phase 0 tables + `alembic_version`) |
| `\dx` | `vector`, `pg_trgm`, `btree_gin` installed |
| RLS enabled | `graph_nodes`, `graph_edges` |
| HNSW index | `idx_chunks_embedding_hnsw` |
| FTS indexes | `idx_chunks_fts_content`, `idx_documents_fts_title_preview` |
| `SELECT version_num FROM alembic_version;` | `0001_initial_schema` |

### Rotating a Neon password

1. Neon console → branch → Roles → reset password on `neondb_owner`
2. Copy the new full connection string
3. `scripts/neon-store.sh <branch>` and paste the new string (overwrites the old entry)

No other file or config changes required.

### Removing a Keychain entry

```bash
security delete-generic-password -a "<branch>" -s "neon-prbe-knowledge"
```

---

## Syncing secrets to Fly

Local `.env` values live on your laptop; Fly apps need each secret set per app. Use `scripts/fly-secrets-sync.sh` to push everything from `.env` to all three Fly apps in one shot.

```bash
# Sync all three apps from .env
scripts/fly-secrets-sync.sh

# Sync a single app only
scripts/fly-secrets-sync.sh ingestion      # or: retrieval | worker

# Use a different env file (e.g. staging)
scripts/fly-secrets-sync.sh -f .env.staging
```

Under the hood it runs `flyctl secrets import -a <app> < <env-file>` on each of:

- `prbe-knowledge-ingestion`
- `prbe-knowledge-retrieval`
- `prbe-knowledge-worker`

Preflight checks bail early if `flyctl` is missing, you're not logged in, or the env file is empty. The underlying `flyctl secrets import` is idempotent — safe to re-run any time you rotate a key.

Verify after a run:

```bash
flyctl secrets list -a prbe-knowledge-ingestion
```

Secret names are shown, values are not. Re-run the script whenever a secret rotates.

---

## Repo layout

```
prbe-knowledge/
├── services/
│   ├── ingestion/      FastAPI webhook fast path + async worker
│   └── retrieval/      FastAPI /query endpoint
├── shared/             Pydantic models, enums, db/config/storage helpers
├── db/
│   ├── schema.sql      canonical Postgres schema (source of truth)
│   └── migrations/     Alembic migrations (execute schema.sql)
├── scripts/
│   ├── neon-store.sh          store a branch connection string in Keychain
│   ├── neon-psql.sh           open psql to a branch
│   ├── neon-migrate.sh        run alembic upgrade head against a branch
│   └── fly-secrets-sync.sh    push .env to Fly secrets on all three apps
├── fixtures/           real sample webhook payloads per source
├── tests/
├── docker-compose.yml  local Postgres + MinIO
└── alembic.ini
```
