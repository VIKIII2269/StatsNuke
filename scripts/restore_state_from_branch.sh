#!/usr/bin/env bash
# Copy small state/ files (e.g. which gameweeks were already pulled) from the data branch
# into the local lake before a collector run. No-op if the branch or files don't exist.
set -euo pipefail
LAKE_DIR=${1:-lake}
BRANCH=${2:-data-bronze}
git fetch --quiet --depth=1 --filter=blob:none origin "refs/heads/$BRANCH" 2>/dev/null || exit 0
git ls-tree -r --name-only FETCH_HEAD -- state/ | while IFS= read -r path; do
  mkdir -p "$LAKE_DIR/$(dirname "$path")"
  git show "FETCH_HEAD:$path" > "$LAKE_DIR/$path"
  echo "restored $path"
done
