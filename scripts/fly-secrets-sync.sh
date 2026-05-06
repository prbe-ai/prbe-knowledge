#!/usr/bin/env bash
# Sync .env to Fly secrets on the eight prbe-knowledge apps.
#
# Usage:
#   scripts/fly-secrets-sync.sh                  # sync all eight apps
#   scripts/fly-secrets-sync.sh ingestion        # sync a single app
#   scripts/fly-secrets-sync.sh -f .env.staging  # use a different env file
#
# App shortcuts: ingestion | retrieval | worker | poller |
#                wiki-worker | wiki-synthesis | wiki-cron | wiki-bootstrap |
#                all (default)
#
# Requires: flyctl in PATH, logged in (`flyctl auth whoami`), and a .env file
# at the repo root (override with -f). The file format is the standard KEY=VALUE
# that `flyctl secrets import` accepts — comments with `#` and blank lines are ok.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="$REPO_ROOT/.env"
TARGET="all"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            ENV_FILE="$2"
            shift 2
            ;;
        -h|--help)
            awk '/^#!/{next} /^[^#]/{exit} {sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
        ingestion|retrieval|worker|poller|wiki-worker|wiki-synthesis|wiki-cron|wiki-bootstrap|all)
            TARGET="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [ingestion|retrieval|worker|poller|wiki-worker|wiki-synthesis|wiki-cron|wiki-bootstrap|all] [-f env-file]" >&2
            exit 1
            ;;
    esac
done

# -- preflight ---------------------------------------------------------------

if ! command -v flyctl >/dev/null 2>&1; then
    echo "flyctl not found in PATH — install from https://fly.io/docs/flyctl/" >&2
    exit 1
fi

if ! flyctl auth whoami >/dev/null 2>&1; then
    echo "Not logged in to Fly. Run: flyctl auth login" >&2
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Env file not found: $ENV_FILE" >&2
    echo "Copy .env.example to .env and fill in values." >&2
    exit 1
fi

# Count non-comment, non-blank KEY=VALUE lines so we can warn on obviously-empty files.
SECRET_COUNT=$(grep -cE '^[A-Z_][A-Z0-9_]*=' "$ENV_FILE" || true)
if [ "$SECRET_COUNT" -eq 0 ]; then
    echo "No KEY=VALUE lines found in $ENV_FILE — nothing to sync." >&2
    exit 1
fi

# -- app list ---------------------------------------------------------------

ALL_APPS=(
    "prbe-knowledge-ingestion"
    "prbe-knowledge-retrieval"
    "prbe-knowledge-worker"
    "prbe-knowledge-poller"
    "prbe-knowledge-wiki-worker"
    "prbe-knowledge-wiki-synthesis"
    "prbe-knowledge-wiki-cron"
    "prbe-knowledge-wiki-bootstrap"
)

case "$TARGET" in
    all)             APPS=("${ALL_APPS[@]}") ;;
    ingestion)       APPS=("prbe-knowledge-ingestion") ;;
    retrieval)       APPS=("prbe-knowledge-retrieval") ;;
    worker)          APPS=("prbe-knowledge-worker") ;;
    poller)          APPS=("prbe-knowledge-poller") ;;
    wiki-worker)     APPS=("prbe-knowledge-wiki-worker") ;;
    wiki-synthesis)  APPS=("prbe-knowledge-wiki-synthesis") ;;
    wiki-cron)       APPS=("prbe-knowledge-wiki-cron") ;;
    wiki-bootstrap)  APPS=("prbe-knowledge-wiki-bootstrap") ;;
esac

# -- parse -------------------------------------------------------------------
# Build a KEY=VALUE arg list, decoding dotenv-style escapes in double-quoted
# values so multiline secrets (e.g. PEM private keys stored as "...\n..." in
# .env) land in Fly with real newlines. `flyctl secrets import` can't handle
# multiline values; `flyctl secrets set` via argv can, because shell quoting
# preserves embedded newlines.

SECRET_ARGS=()
while IFS= read -r line || [ -n "$line" ]; do
    [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]] && continue
    [[ "$line" =~ ^([A-Z_][A-Z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[1]}"
    val="${BASH_REMATCH[2]}"
    # Strip surrounding double quotes and decode \n, \t, \", \\ escapes.
    if [[ "$val" == \"*\" ]]; then
        val="${val#\"}"
        val="${val%\"}"
        val="$(printf '%b' "$val")"
    fi
    SECRET_ARGS+=("$key=$val")
done < "$ENV_FILE"

# -- sync -------------------------------------------------------------------

echo "Syncing ${#SECRET_ARGS[@]} secret(s) from $ENV_FILE to: ${APPS[*]}"
echo

for app in "${APPS[@]}"; do
    echo "→ $app"
    if ! flyctl secrets set -a "$app" "${SECRET_ARGS[@]}"; then
        echo "  failed on $app — fix and re-run (sync is idempotent)" >&2
        exit 1
    fi
    echo
done

echo "Done. Verify with: flyctl secrets list -a prbe-knowledge-ingestion"
