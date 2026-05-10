#!/usr/bin/env bash
# Sync .env to Fly secrets on the prbe-knowledge apps. Auto-creates any
# target app that doesn't exist yet on Fly (so adding a new fly.<x>.toml
# + ALL_APPS entry is the only manual step before running this script).
#
# Usage:
#   scripts/fly-secrets-sync.sh                  # sync all apps (auto-create missing)
#   scripts/fly-secrets-sync.sh ingestion        # sync a single app
#   scripts/fly-secrets-sync.sh -f .env.staging  # use a different env file
#   scripts/fly-secrets-sync.sh --org acme-corp  # override fly org (default prbe-ai)
#   scripts/fly-secrets-sync.sh --no-create cron # fail (don't create) if app missing
#
# App shortcuts: ingestion | retrieval | worker | poller |
#                wiki-worker | wiki-synthesis | wiki-backfill |
#                cron | side-worker | all (default)
#
# Requires: flyctl in PATH, logged in (`flyctl auth whoami`), and a .env file
# at the repo root (override with -f). The file format is the standard KEY=VALUE
# that `flyctl secrets import` accepts — comments with `#` and blank lines are ok.
#
# App auto-creation:
#   For each target app, the script checks `flyctl status -a <app>` first.
#   If the app doesn't exist, it runs `flyctl apps create <app> --org $ORG`
#   before syncing secrets. Pass --no-create to disable this and fail
#   instead. Override the org with --org or the FLY_ORG env var.
#   Default org: prbe-ai.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_FILE="$REPO_ROOT/.env"
TARGET="all"
ORG="${FLY_ORG:-prbe-ai}"
AUTO_CREATE=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        -f|--file)
            ENV_FILE="$2"
            shift 2
            ;;
        --org)
            ORG="$2"
            shift 2
            ;;
        --no-create)
            AUTO_CREATE=0
            shift
            ;;
        -h|--help)
            awk '/^#!/{next} /^[^#]/{exit} {sub(/^# ?/,""); print}' "$0"
            exit 0
            ;;
        ingestion|retrieval|worker|poller|wiki-worker|wiki-synthesis|wiki-backfill|cron|side-worker|all)
            TARGET="$1"
            shift
            ;;
        *)
            echo "Unknown argument: $1" >&2
            echo "Usage: $0 [ingestion|retrieval|worker|poller|wiki-worker|wiki-synthesis|wiki-backfill|cron|side-worker|all] [-f env-file] [--org name] [--no-create]" >&2
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
    "prbe-knowledge-wiki-backfill"
    "prbe-knowledge-cron"
    "prbe-knowledge-side-worker"
)

case "$TARGET" in
    all)             APPS=("${ALL_APPS[@]}") ;;
    ingestion)       APPS=("prbe-knowledge-ingestion") ;;
    retrieval)       APPS=("prbe-knowledge-retrieval") ;;
    worker)          APPS=("prbe-knowledge-worker") ;;
    poller)          APPS=("prbe-knowledge-poller") ;;
    wiki-worker)     APPS=("prbe-knowledge-wiki-worker") ;;
    wiki-synthesis)  APPS=("prbe-knowledge-wiki-synthesis") ;;
    wiki-backfill)   APPS=("prbe-knowledge-wiki-backfill") ;;
    cron)            APPS=("prbe-knowledge-cron") ;;
    side-worker)     APPS=("prbe-knowledge-side-worker") ;;
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

# -- ensure apps exist -------------------------------------------------------
# `flyctl deploy` doesn't auto-create apps — a fresh fly.<x>.toml fails with
# "app not found" until someone runs `flyctl apps create` once. Doing that
# bootstrap here means adding a new fly.<x>.toml + ALL_APPS entry is the only
# manual step before the next sync/deploy cycle. Idempotent: existing apps
# skip the create call.

for app in "${APPS[@]}"; do
    if flyctl status -a "$app" >/dev/null 2>&1; then
        continue
    fi
    if [ "$AUTO_CREATE" -eq 0 ]; then
        echo "App $app does not exist on Fly. Re-run without --no-create, or:" >&2
        echo "  flyctl apps create $app --org $ORG" >&2
        exit 1
    fi
    echo "Creating Fly app: $app (org=$ORG)"
    if ! flyctl apps create "$app" --org "$ORG"; then
        echo "  failed to create $app — check org name and Fly auth" >&2
        exit 1
    fi
    echo
done

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
