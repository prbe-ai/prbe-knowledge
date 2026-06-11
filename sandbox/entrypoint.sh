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

# --- HELD-OUT eval infra: record/replay LLM proxy (sandbox/, agents cannot edit) ---
# REPLAY_ENABLED=1 routes every product LLM call through sandbox/replay_proxy.py, so the grader's
# replay pass returns cached responses at a TOKEN/BYTE-AWARE synthetic delay -> request wall-clock =
# deterministic code path, zero LLM serving variance, and a wider/padded candidate pays real latency.
# Gated + backward-compatible: unset -> the product talks to the gateway directly (unchanged).
if [ "${REPLAY_ENABLED:-0}" = "1" ]; then
  export REPLAY_UPSTREAM_URL="${LLM_GATEWAY_URL:?REPLAY_ENABLED requires LLM_GATEWAY_URL}"
  export REPLAY_PORT="${REPLAY_PORT:-8900}"
  export REPLAY_MODE="${REPLAY_MODE:-auto}"   # auto = record-on-miss / replay-on-hit; one mode all grade
  export LLM_GATEWAY_URL="http://127.0.0.1:${REPLAY_PORT}"
  echo "[entrypoint] replay proxy ON (mode=${REPLAY_MODE}): product -> ${LLM_GATEWAY_URL} -> ${REPLAY_UPSTREAM_URL}"
  python /app/sandbox/replay_proxy.py &
  _replay_up=0
  for _i in $(seq 1 120); do
    if (exec 3<>/dev/tcp/127.0.0.1/"${REPLAY_PORT}") 2>/dev/null; then exec 3>&-; _replay_up=1; echo "[entrypoint] replay proxy up"; break; fi
    sleep 0.5
  done
  # FAIL LOUD: LLM_GATEWAY_URL is already repointed at the proxy, so a never-started proxy would make
  # every product LLM call hit a dead port and SILENTLY score a broken grade. Refuse to boot instead.
  if [ "$_replay_up" != "1" ]; then
    echo "[entrypoint] FATAL: replay proxy never came up on 127.0.0.1:${REPLAY_PORT}" >&2
    exit 1
  fi
fi

# --workers 1: one uvicorn + the embedded PG fit the sandbox resource box and keep
# logs legible (prod uses 4 — services/retrieval/Dockerfile:28). `::` = dual-stack.
exec uvicorn services.retrieval.main:app --host :: --port 8081 --workers 1
