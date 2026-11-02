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
| `e9_holdout.py` | 10; N10 (`--fast`) | private features, SING | `fix10_e9_holdout.json`; with `--fast`, `fix10_e9_holdout_fast.json` |
| `holdout_tex.py` | 10 | `results/fix10_e9_holdout.json` | `paper_workshop/holdout_numbers.tex` |
| `e2_m1.py` | 10 (E2 under M1, post hoc) | `results/fix2_models.csv` (optional cross-check: private `*_r2img.npz`) | `fix10_e2_m1.json` |
| `pc_bands.py` | 13 | private features, SING | `fix13_pc_bands.json` |
| `fix14_fidelity.py` | 14 | output of `gpu_jobs/fix14_extract_vitl.py` | `fix14_fidelity.json` |
| `make_figs.py` | 12; N11 | committed results (round 2 and `r3/results`) | `paper_workshop/figs/*` |

Round-3 changes (see `../PREREG_R3.md`):
* `e9_holdout.py --fast`: the reduced holdout declared for N10 (5 draws instead of 20, 2 random subspaces instead of 5, S9 dropped). It writes a separate cache (`$IC014_PRIVATE/e9_holdout_cache_fast.jsonl`) and output, and resumes from the cache if interrupted. The full run is unchanged.
* `holdout_tex.py` also writes three disclosure sentences: the S10 flip under the revised flatness rule; that the plant is written along the null projection M1 measures; and that "water" was also an E3 plant concept.
* `make_figs.py`: `fig_e2_ranking` adds ρ; new `fig_r3_tau` (τ forest plot for the round-3 scores); PDFs are written without a creation date, so reruns are byte-identical.
* `e2_controls.py` and `e2_claims_ci.py` write per-image arrays to `$IC014_PRIVATE` (default `_private/`, git-ignored), not to the feature store.
