#!/usr/bin/env bash
# Open an interactive psql session against a Neon branch.
# Usage: scripts/neon-psql.sh <branch>
#
# Pulls the connection string from the macOS Keychain (store it first via
# scripts/neon-store.sh). Requires `psql` on PATH (brew install libpq).

set -euo pipefail

BRANCH="${1:-}"
if [ -z "$BRANCH" ]; then
    echo "Usage: $0 <branch>" >&2
    exit 1
fi

SERVICE="neon-prbe-knowledge"

if ! URL=$(security find-generic-password -a "$BRANCH" -s "$SERVICE" -w 2>/dev/null); then
    echo "No Keychain entry for branch '$BRANCH'." >&2
    echo "Run: scripts/neon-store.sh $BRANCH" >&2
    exit 1
fi

if ! command -v psql >/dev/null 2>&1; then
    echo "psql not found on PATH." >&2
    echo "Install with: brew install libpq && brew link --force libpq" >&2
    exit 1
fi

echo "Connecting to Neon branch: $BRANCH"
exec psql "$URL"
