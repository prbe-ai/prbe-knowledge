#!/usr/bin/env bash
# Boot the prbe-knowledge retrieval slice inside the product sandbox:
#   start Postgres -> apply canonical DDL -> seed corpus -> exec uvicorn.
#
#   ┌─ docker-entrypoint.sh postgres &   (initdb on first boot, then serve)
#   │     │
#   │     ▼  pg_isready
#   ├─ psql -f db/schema.sql             (CI-canonical DDL; NOT the migration chain)
#   │     │
#   │     ▼
#   ├─ corpus precedence:
#   │     1. /grade/corpus.sql.gz   GRADE   — held-out corpus the evaluator injected
#   │                                          (real gemini-embedding-2 vectors baked in);
#   │                                          never present in the agent's box.
#   │     2. sandbox/dev_corpus.sql IMPLEMENT— tiny BM25-able corpus baked into the image
#   │                                          so the agent's /retrieve returns something.
#   │     3. neither               EMPTY    — smoke still passes: /retrieve returns a valid
#   │                                          empty RetrieveResponse via the zero-recall
#   │                                          short-circuit (services/retrieval/agent/loop.py:1360).
#   │     ▼
#   └─ exec uvicorn services.retrieval.main:app :8081
set -euo pipefail

export POSTGRES_USER="${POSTGRES_USER:-prbe}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-prbe}"
export POSTGRES_DB="${POSTGRES_DB:-prbe_knowledge}"
: "${DATABASE_URL:=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}}"
export DATABASE_URL

# Postgres via the official entrypoint (handles initdb + role/db creation on first
# boot from POSTGRES_*), backgrounded so we can seed + serve in the same container.
docker-entrypoint.sh postgres &

until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" -h localhost >/dev/null 2>&1; do
  sleep 0.5
done

# Provision a minimal neon_auth stand-in: prod/CI get this schema from Neon Auth, but the bare
# pgvector image lacks it and db/schema.sql FKs customers.organization_id -> neon_auth.organization(id).
# The eval corpus never sets organization_id (stays NULL), so an empty stub satisfies the FK.
# Sandbox-only — sandbox/ ships in no prod image, and the prod bootstrap (scripts/migrate.py) keeps
# its own copy of this shim, so nothing here touches a prod code path.
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -c \
  "CREATE SCHEMA IF NOT EXISTS neon_auth; CREATE TABLE IF NOT EXISTS neon_auth.organization (id uuid PRIMARY KEY); CREATE TABLE IF NOT EXISTS neon_auth.\"user\" (id uuid PRIMARY KEY, organization_id uuid REFERENCES neon_auth.organization(id), email text, name text);"

# Fresh DB on first boot, so a one-shot apply is safe. `prbe` is a superuser here, so
# the RLS policies in schema.sql are bypassed and retrieval's explicit
# `WHERE customer_id = $1` does the tenant scoping.
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f /app/db/schema.sql

if [ -f /grade/corpus.sql.gz ]; then
  echo "[entrypoint] restoring held-out grade corpus"
  gunzip -c /grade/corpus.sql.gz | psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q
elif [ -f /app/sandbox/dev_corpus.sql ]; then
  echo "[entrypoint] seeding dev corpus (implement mode)"
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f /app/sandbox/dev_corpus.sql
fi

# --workers 1: one uvicorn + the embedded PG fit the sandbox resource box and keep
# logs legible (prod uses 4 — services/retrieval/Dockerfile:28). `::` = dual-stack.
exec uvicorn services.retrieval.main:app --host :: --port 8081 --workers 1
