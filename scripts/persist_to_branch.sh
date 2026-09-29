#!/usr/bin/env bash
# Commit the contents of a local lake directory to a data branch, without checking it out.
#
# Stopgap durability for .github/workflows/collect.yml when no bucket is configured
# (FPLH_LAKE_URI unset): runners are ephemeral, so new bronze objects are appended to the
# `data-bronze` branch with git plumbing. The branch is created (orphan) on first use.
# Existing paths are never removed; bronze keys are unique, so only state/ files change.
#
# usage: scripts/persist_to_branch.sh [lake_dir] [branch]
set -euo pipefail

LAKE_DIR=${1:-lake}
BRANCH=${2:-data-bronze}

if [ ! -d "$LAKE_DIR" ] || [ -z "$(find "$LAKE_DIR" -type f -print -quit)" ]; then
  echo "nothing to persist"
  exit 0
fi

git config user.name "fplh-collector"
git config user.email "41898282+github-actions[bot]@users.noreply.github.com"

for attempt in 1 2 3 4; do
  parent=""
  export GIT_INDEX_FILE
  GIT_INDEX_FILE=$(mktemp)
  rm -f "$GIT_INDEX_FILE"
  if git fetch --quiet --depth=1 --filter=blob:none origin "refs/heads/$BRANCH" 2>/dev/null; then
    parent=$(git rev-parse FETCH_HEAD)
    git read-tree "$parent"
  else
    git read-tree --empty
    readme=$(printf '# %s\n\nAppend-only raw data written by .github/workflows/collect.yml.\nSee docs/IMPLEMENTATION_PLAN.md (stopgap storage).\n' "$BRANCH" | git hash-object -w --stdin)
    git update-index --add --cacheinfo "100644,$readme,README.md"
  fi

  added=0
  while IFS= read -r -d '' file; do
    rel=${file#"$LAKE_DIR"/}
    case "$rel" in */.tmp-*|.tmp-*) continue ;; esac
    blob=$(git hash-object -w -- "$file")
    git update-index --add --cacheinfo "100644,$blob,$rel"
    added=$((added + 1))
  done < <(find "$LAKE_DIR" -type f -print0)

  tree=$(git write-tree)
  if [ -n "$parent" ] && [ "$tree" = "$(git rev-parse "$parent^{tree}")" ]; then
    echo "no changes"
    exit 0
  fi
  msg="collect: $added file(s) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [ -n "$parent" ]; then
    commit=$(git commit-tree "$tree" -p "$parent" -m "$msg")
  else
    commit=$(git commit-tree "$tree" -m "$msg")
  fi
  rm -f "$GIT_INDEX_FILE"
  unset GIT_INDEX_FILE

  if git push --quiet origin "$commit:refs/heads/$BRANCH"; then
    echo "pushed $added file(s) to $BRANCH ($commit)"
    exit 0
  fi
  echo "push rejected (attempt $attempt); refetching" >&2
  sleep $((2 ** attempt))
done
echo "failed to persist to $BRANCH" >&2
exit 1
