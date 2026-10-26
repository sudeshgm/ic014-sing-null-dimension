# E3 results: planted leakage (raw, no interpretation)
Features: Kaggle hiddenslave/ic014-e3 v1 (2× T4, wall 9205 s). These are private and not in this repo. 8 models, N=10000, same subset as E1 (subset_files.txt identical). Concepts: red, striped, water. 20 random-subspace draws, 50 trials, seed 0.

Kendall τ(metric, α ranking) over 8 models. CI = 95% bootstrap CI of the mean over 50 trials:

| | AS_SING | AS_fair |
|---|---|---|
| red | 0.386 [0.327, 0.446] | 0.070 [0.001, 0.140] |
| striped | 0.403 [0.336, 0.469] | 0.061 [−0.011, 0.134] |
| water | 0.369 [0.303, 0.431] | 0.083 [0.014, 0.153] |
| concept-averaged | 0.386 [0.324, 0.447] | 0.071 [0.004, 0.138] |

The 2.5–97.5 percentiles of the per-trial τ (concept-averaged) are [−0.03, 0.77] for AS_SING and [−0.35, 0.56] for AS_fair.
Per-trial values are in `results/e3_trials.csv`, the per-(model, concept, α) table is in `results/e3_table.csv`, and the summary is in `results/e3_summary.json`.

Checks:
- Max |logit change| ≤ 2.3e-5 for every model at α=1.
- T(f) reconstruction max error is 0.0039, which is fp16 rounding of z16.
- Null coordinates match E1 for resnet and resnext (energy 131.6 vs 131.7; 2091 vs 2091).

Observations (raw):
- DeiT3-L AS_SING < 0 at every α. Its signed AS goes from negative toward positive, and |AS| shrinks.
- DeiT3-L AS_fair decreases with α.
- BiT AS_SING is 17–20° at α=1, while its AS_fair at α=0 is 0.045–0.063 (random-subspace |AS| ≈ 8–9.5°).
- DenseNet's null energy at α=0 is ρ=1e-4, which matches less than one random direction (k≈0.33).
- AS_fair is non-monotonic in α for several models: EVA02-L and ConvNeXt-B dip at α=1, and ResNeXt is nearly flat.
- The u null fraction (‖V_nV_nᵀA⁺t‖/‖A⁺t‖) is 0.73–0.79 for the null-dim-1048 models and 0.10–0.16 for the null-dim-24/27 models.
