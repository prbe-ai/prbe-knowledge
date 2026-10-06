#!/usr/bin/env bash
# Re-vendor Harbor's ATIF pydantic models at a commit. See
# engine/ingest/atif/models/VENDORED.md.
set -euo pipefail
sha="${1:?usage: $0 <harbor commit sha>}"
dest="$(cd "$(dirname "$0")/.." && pwd)/engine/ingest/atif/models"
files=$(gh api "repos/laude-institute/harbor/contents/src/harbor/models/trajectories?ref=$sha" --jq '.[] | select(.name | endswith(".py")) | .name')
for f in $files; do
  gh api "repos/laude-institute/harbor/contents/src/harbor/models/trajectories/$f?ref=$sha" --jq .content \
    | base64 -d \
    | sed 's/from harbor\.models\.trajectories\./from engine.ingest.atif.models./' > "$dest/$f"
done
gh api "repos/laude-institute/harbor/contents/LICENSE?ref=$sha" --jq .content | base64 -d > "$dest/LICENSE"
sed -i "s/^Commit: .*/Commit: $sha ($(date -u +%F))/" "$dest/VENDORED.md"
echo "vendored $sha into $dest; now run: pytest tests/test_atif_build.py"
