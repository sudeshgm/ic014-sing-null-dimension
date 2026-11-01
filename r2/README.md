# Round-2 analyses
Criteria: `../PREREG_R2.md`. Plan: `../ROUND2_PLAN.md`. CPU only; seed 0 unless noted.

| Script | Fix | Inputs | Output (`results/`) |
|---|---|---|---|
| `e2_controls.py` | 2, 5 | private features, SING | `fix2_models.csv`, `fix5_tolerance.csv` |
| `e2_sweep.py` | 3, 4, 5, KILL-R | committed CSVs | `fix2345_sweep.json`, `fix3_normalisers.csv` |
| `e1_ci.py` | 6 | `e1/results/e1_cells.csv` | `fix6_e1_ci.json` |
| `e3_symmetric.py` | 7 | `e3/results/e3_trials.csv`, `e3_table.csv` | `fix7_e3_symmetric.json` |
| `e2_claims_ci.py` | 8 | private `*_r2img.npz` | `fix8_claims_ci.{json,csv}` |
| `flatness_synth.py` | 9 | private features, E8/E9 results | `fix9_flatness.json` |
| `e9_holdout.py` | 10 | private features, SING | `fix10_e9_holdout.json` |
| `pc_bands.py` | 13 | private features, SING | `fix13_pc_bands.json` |
| `fix14_fidelity.py` | 14 | output of `gpu_jobs/fix14_extract_vitl.py` | `fix14_fidelity.json` (not run) |
| `make_figs.py` | 12 | committed results | `paper_workshop/figs/*` |
