#!/usr/bin/env bash
# Drop everything and re-run the migration from scratch.
# Usage: scripts/neon-reset.sh <branch>
#   <branch>: local | dev | staging | main
#
# DESTRUCTIVE: drops all Phase 0 tables on the target. Use pre-launch only,
# when the schema is still fluid. Prompts for confirmation before dropping.
#
# Under the hood: alembic downgrade base && alembic upgrade head

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BRANCH="${1:-}"
if [ -z "$BRANCH" ]; then
    echo "Usage: $0 <branch>" >&2
    echo "  <branch>: local | dev | staging | main" >&2
    exit 1
fi

if [ ! -x ".venv/bin/alembic" ]; then
    echo "No .venv/bin/alembic found at $REPO_ROOT/.venv." >&2
    echo "Create the venv first:" >&2
    echo "  python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'" >&2
    exit 1
fi

SERVICE="neon-prbe-knowledge"

if [ "$BRANCH" = "local" ]; then
    URL="postgresql+psycopg://prbe:prbe@localhost:5432/prbe_knowledge"
    HOST="localhost:5432 (docker-compose)"
else
    if ! RAW_URL=$(security find-generic-password -a "$BRANCH" -s "$SERVICE" -w 2>/dev/null); then
        echo "No Keychain entry for branch '$BRANCH'." >&2
        echo "Run: scripts/neon-store.sh $BRANCH" >&2
        exit 1
    fi
    URL="postgresql+psycopg://${RAW_URL#postgresql://}"
    HOST=$(echo "$RAW_URL" | sed -E 's|^postgresql://[^@]+@([^/]+)/.*|\1|')
fi

# Loud, unambiguous warning.
echo "========================================================================"
echo "  DESTRUCTIVE RESET"
echo "========================================================================"
echo "  Target branch: $BRANCH"
echo "  Host:          $HOST"
echo
echo "  This will DROP all Phase 0 tables (documents, chunks, graph_nodes,"
echo "  graph_edges, acl_snapshots, ingestion_queue, etc.) and any data they"
echo "  contain, then re-create them from db/schema.sql."
echo "========================================================================"
echo

read -r -p "Type the branch name ('$BRANCH') to confirm: " CONFIRM
if [ "$CONFIRM" != "$BRANCH" ]; then
    echo "Aborted (got '$CONFIRM', expected '$BRANCH')." >&2
    exit 1
fi

echo
echo "Downgrading..."
DATABASE_URL_SYNC="$URL" .venv/bin/alembic downgrade base

echo
echo "Upgrading..."
DATABASE_URL_SYNC="$URL" .venv/bin/alembic upgrade head

echo
echo "Reset complete. Verify with: scripts/neon-psql.sh $BRANCH"
