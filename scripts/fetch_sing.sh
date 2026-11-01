#!/usr/bin/env bash
# Clone SING's released code at the commit used in this audit.
set -euo pipefail
DEST="${1:-/tmp/sing}"
if [ ! -d "$DEST/.git" ]; then
  git clone https://github.com/harel314/SING-analyzing-semantic-invariants-classifiers "$DEST"
fi
git -C "$DEST" fetch --all --quiet || true
git -C "$DEST" checkout e47b37db98dd09d6f8c79902b536f4e6f19dae59
echo "SING at $(git -C "$DEST" rev-parse HEAD) in $DEST"
