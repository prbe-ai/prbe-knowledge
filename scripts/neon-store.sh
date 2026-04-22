#!/usr/bin/env bash
# Store a Neon connection string in the macOS Keychain.
# Usage: scripts/neon-store.sh <branch>
#   where <branch> is one of: dev | staging | main
#
# The script reads the connection string from the macOS clipboard (pbpaste).
# This avoids paste truncation issues when the Neon UI wraps the URL across lines.
#
# Workflow:
#   1. Neon console -> your branch -> Connect -> copy the full postgresql:// URL
#   2. scripts/neon-store.sh <branch>
#
# Re-running against the same branch overwrites the existing entry.

set -euo pipefail

BRANCH="${1:-}"
if [ -z "$BRANCH" ]; then
    echo "Usage: $0 <branch>" >&2
    echo "  <branch>: dev | staging | main (or any label you want)" >&2
    exit 1
fi

SERVICE="neon-prbe-knowledge"

# Read from macOS clipboard; strip any trailing/internal newlines/CRs that
# might have snuck in from a wrapped UI.
URL=$(pbpaste | tr -d '\r\n')

if [ -z "$URL" ]; then
    echo "Clipboard is empty. Copy the connection string from Neon first, then re-run." >&2
    exit 1
fi

if [[ "$URL" != postgresql://* ]]; then
    echo "Clipboard doesn't start with 'postgresql://'. Got:" >&2
    echo "  ${URL:0:60}..." >&2
    echo "Copy the full connection string from Neon -> Connect -> URL." >&2
    exit 1
fi

# Sanity-check: full Neon URL is typically 140-200 chars. Short strings mean
# a truncated paste (UI line-wrap, incomplete selection, etc).
LEN=${#URL}
if [ "$LEN" -lt 120 ]; then
    echo "URL looks suspiciously short ($LEN chars). Likely a truncated copy." >&2
    echo "Stored anyway, but verify with: scripts/neon-psql.sh $BRANCH" >&2
fi

# Extract the host portion for confirmation (no password shown).
HOST=$(echo "$URL" | sed -E 's|^postgresql://[^@]+@([^/]+)/.*|\1|')

security add-generic-password -a "$BRANCH" -s "$SERVICE" -U -w "$URL"

echo "Stored in Keychain."
echo "  Service:    $SERVICE"
echo "  Account:    $BRANCH"
echo "  Host:       $HOST"
echo "  URL length: $LEN chars"
echo
echo "Next: scripts/neon-migrate.sh $BRANCH"
