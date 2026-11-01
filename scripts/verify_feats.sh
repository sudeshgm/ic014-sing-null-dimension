#!/usr/bin/env bash
# Check private feature files against the recorded hashes. Usage: scripts/verify_feats.sh $IC014_FEATS
set -euo pipefail
ROOT="${1:?usage: verify_feats.sh FEATS_ROOT}"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" && sha256sum -c "$HERE/checksums.sha256"
