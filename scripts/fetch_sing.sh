#!/usr/bin/env bash
# Fetch SING's released code at the commit used in this audit.
# Fetches only that one commit (depth 1); a full clone of the repo can fail with "early EOF".
set -euo pipefail
COMMIT=e47b37db98dd09d6f8c79902b536f4e6f19dae59
URL=https://github.com/harel314/SING-analyzing-semantic-invariants-classifiers
DEST="${1:-${SING_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/_ext/sing}}"
if [ ! -d "$DEST/.git" ]; then
  mkdir -p "$DEST"
  git -C "$DEST" init --quiet
  git -C "$DEST" remote add origin "$URL"
fi
if ! git -C "$DEST" cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
  git -C "$DEST" fetch --depth 1 origin "$COMMIT"
fi
git -C "$DEST" checkout --quiet "$COMMIT"
HEAD_NOW=$(git -C "$DEST" rev-parse HEAD)
[ "$HEAD_NOW" = "$COMMIT" ] || { echo "fetch_sing: expected $COMMIT, got $HEAD_NOW" >&2; exit 1; }
echo "SING at $HEAD_NOW in $DEST"
