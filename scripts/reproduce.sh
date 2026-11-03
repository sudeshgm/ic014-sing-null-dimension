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
$PY r3/paired_stats.py > /dev/null
$PY r2/make_figs.py
$PY r3/make_tables.py > /dev/null
$PY r4/make_tables_r4.py > /dev/null
$PY r5/calib_compare.py > /dev/null
$PY r5/make_tables_r5.py > /dev/null
echo "reproduce: done (r2/results, r3/results/F_paired_stats.json, paper_workshop/figs, paper_workshop/tables incl. round-4 and round-5 tables, paper_workshop/holdout_numbers.tex)"
