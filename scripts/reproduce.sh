#!/usr/bin/env bash
# Recompute every number and figure that needs only committed results (CPU, no ImageNet).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
$PY r2/e1_ci.py > /dev/null
$PY r2/e3_symmetric.py > /dev/null
$PY r2/e2_sweep.py > /dev/null
$PY r2/e2_m1.py > /dev/null
$PY r2/holdout_tex.py > /dev/null
$PY r2/make_figs.py
echo "reproduce: done (r2/results, paper_workshop/figs)"
