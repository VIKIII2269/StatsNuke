#!/usr/bin/env bash
# Copy the given top-level folders of a data branch (default: bronze/ and state/ of
# data-bronze) into a local lake directory. No-op if the branch does not exist.
#
# usage: scripts/restore_branch.sh [lake_dir] [branch] [folder ...]
set -euo pipefail
LAKE_DIR=${1:-lake}
BRANCH=${2:-data-bronze}
shift $(( $# > 2 ? 2 : $# ))
FOLDERS=("$@")
[ ${#FOLDERS[@]} -eq 0 ] && FOLDERS=(bronze state)
git fetch --quiet --depth=1 origin "refs/heads/$BRANCH" 2>/dev/null || { echo "no $BRANCH"; exit 0; }
mkdir -p "$LAKE_DIR"
present=()
for f in "${FOLDERS[@]}"; do
  if git cat-file -e "FETCH_HEAD:$f" 2>/dev/null; then present+=("$f"); fi
done
[ ${#present[@]} -eq 0 ] && { echo "nothing to restore"; exit 0; }
git archive FETCH_HEAD "${present[@]}" | tar -x -C "$LAKE_DIR"
echo "restored ${present[*]} from $BRANCH: $(find "$LAKE_DIR" -type f | wc -l) files in $LAKE_DIR"
