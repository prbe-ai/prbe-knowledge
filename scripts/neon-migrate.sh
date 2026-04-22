#!/usr/bin/env bash
# Run `alembic upgrade head` against a Neon branch (or local docker-compose).
# Usage: scripts/neon-migrate.sh <branch>
#   where <branch> is: local | dev | staging | main
#
# `local`  -> docker-compose Postgres at localhost:5432
# others   -> connection string pulled from macOS Keychain (via neon-store.sh)

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
    echo "Migrating local docker-compose Postgres..."
else
    if ! RAW_URL=$(security find-generic-password -a "$BRANCH" -s "$SERVICE" -w 2>/dev/null); then
        echo "No Keychain entry for branch '$BRANCH'." >&2
        echo "Run: scripts/neon-store.sh $BRANCH" >&2
        exit 1
    fi
    # Alembic needs psycopg3; swap the scheme.
    URL="postgresql+psycopg://${RAW_URL#postgresql://}"
    echo "Migrating Neon branch: $BRANCH"
fi

DATABASE_URL_SYNC="$URL" .venv/bin/alembic upgrade head
echo "Done."
