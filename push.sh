#!/usr/bin/env bash
#
# push.sh — stage, commit and push to every push URL configured on `origin`.
#
# Usage:
#   ./push.sh                       # commit with a timestamped message
#   ./push.sh "your commit message" # commit with your own message
#
set -euo pipefail

# Always operate from the repository root, no matter where the script is called from.
cd "$(git rev-parse --show-toplevel)"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
MESSAGE="${1:-"update: $(date '+%Y-%m-%d %H:%M:%S')"}"

echo "Repository : $(pwd)"
echo "Branch     : ${BRANCH}"
echo "Remotes    :"
git remote get-url --push --all origin | sed 's/^/             /'
echo

# ---- stage -----------------------------------------------------------------
git add -A

# ---- commit ----------------------------------------------------------------
if git diff --cached --quiet; then
  echo "No staged changes — skipping commit, pushing existing commits."
else
  echo "Committing: ${MESSAGE}"
  git commit -m "${MESSAGE}"
fi
echo

# ---- push ------------------------------------------------------------------
# `git push` fans out to every push URL on origin. It is NOT atomic across
# them: if a later URL is rejected, earlier ones have already been updated.
# We push each URL separately so one failure cannot hide the others.
FAILED=()
while read -r url; do
  [ -z "${url}" ] && continue
  echo "Pushing to ${url}"
  if git push "${url}" "${BRANCH}"; then
    echo "  ok"
  else
    echo "  FAILED"
    FAILED+=("${url}")
  fi
  echo
done < <(git remote get-url --push --all origin)

# ---- report ----------------------------------------------------------------
if [ ${#FAILED[@]} -eq 0 ]; then
  echo "Done — '${BRANCH}' pushed to all remotes."
else
  echo "Done with errors — these remotes did NOT receive the push:"
  printf '  %s\n' "${FAILED[@]}"
  exit 1
fi
