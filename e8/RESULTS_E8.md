**E8 verdict (IC014 Idea Agent): KILL.** Mean Kendall is −0.33, below the 0.6 threshold, and NCP is flat on 0 of 4 eligible models. No reruns or tweaks.

Flags:
- **The z-score saturates.** Probe R² is close to 1 in the null and in the random subspaces alike, so NCP ends up set by a tiny random SD.
- **λ is chosen by approximate closed-form leave-one-out**, not the exact CV the prereg states.
- **λ sits at the grid floor (10^-3)** in 18 of 96 α-grid fits.
- **Per the plan, AS_fair (E2) is the proposed fix** instead of NCP.

# E8 results: NCP (raw; no verdict). Pre-registration: 5021aa9 (`PREREG_E8.md`)

Run: `analyze_e8.py`, CPU only, on the private E3 features. The log is at /tmp/e8_full.log. The run went from 22:43 to about 00:35 IST.

## Kendall τ(NCP, planted α) over 8 models, 50 trials (E3 protocol)
- **Concept-averaged mean τ:** −0.331, 95% bootstrap CI [−0.392, −0.273]. The per-trial 2.5–97.5 percentiles are [−0.759, 0.042].
- **Per concept:** red −0.337, striped −0.303, water −0.353.

## Flatness at α = 0.1 (NCP averaged over the 3 concepts; 5 nested sub-null draws per d)
| model | null dim | d=24 | 100 | 280 | 536 | full | max/min | per-concept max/min (red/striped/water) |
|---|---|---|---|---|---|---|---|---|
| resnet | 1048 | 12.47 | 6.84 | 6.36 | 7.91 | 5.52 | 2.26 | 2.67 / 2.17 / 1.96 |
| resnext | 1048 | 5.35 | 6.79 | 8.29 | 8.76 | 10.14 | 1.90 | 1.58 / 1.99 / 2.22 |
| dinov2 | 1048 | 4.33 | 7.15 | 12.18 | 15.35 | 16.87 | 3.89 | 3.54 / 4.04 / 4.13 |
| bitresnet | 1048 | 14.69 | 12.10 | 9.06 | 9.25 | 8.17 | 1.80 | 1.93 / 1.87 / 1.70 |
| deit | 24 | 7.17 | – | – | – | – | 1 (trivial) | |
| eva02_large | 24 | 17.68 | – | – | – | – | 1 (trivial) | |
| convnext | 24 | 18.79 | – | – | – | – | 1 (trivial) | |
| densenet | 27 | 10.70 | – | – | – | 10.41 | 1.03 | |

**Flat count:** 0 of 4 eligible models (null dim ≥ 1048).

## NCP vs α with the full null space (`e8_alpha_grid.csv`), concept-averaged
| model | α=0 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|
| resnet | −5.3 | 18.1 | 5.5 | 16.4 |
| resnext | −1.9 | 11.2 | 10.1 | 0.5 |
| dinov2 | −5.8 | 14.8 | 16.9 | 5.4 |
| bitresnet | 3.9 | 31.7 | 8.2 | 16.2 |
| deit | −6.8 | 10.3 | 7.2 | 3.5 |
| eva02_large | 2.5 | 65.8 | 17.7 | 14.7 |
| convnext | 1.2 | 22.6 | 18.8 | 9.8 |
| densenet | 8.3 | 244.8 | 10.4 | 2.0 |

## Observations (raw)
- NCP vs α: at α ≥ 0.1, R²_null and R²_rand both saturate near 1. The plant has energy α·σ in the full space, so energy-matched random subspaces pick up s_c as well, and the z-score is then set by a tiny SD over the random draws (for example, 1e-6 for resnet at α = 1).
- **R² at α = 0.** With no plant, R²_null is between 0.01 and 0.39, so CLIP concepts can be decoded linearly from the null coordinates. For large null spaces this R² is below that of the matched random subspaces, which gives NCP < 0.
- **DenseNet:** at α = 0.01 NCP is 175–376. Its random subspaces amount to about 0.3 directions (ρ ≈ 0), and their R² is about 0.
- **Small-null models** (null dim 24/27) only have one or two d values, so flatness is trivial for them.
- **Known ridge-implementation details:**
  - LOO-CV is computed in closed form on centred, standardised train data (the intercept is not refit per fold).
  - λ never reaches the top of the grid (10^5). It does sit at the bottom edge (10^-3) in 18 of 96 α-grid fits, so the grid may be too narrow at that end. The chosen λ is listed in `e8_alpha_grid.csv`.
  - As in E3's AS_fair, the random subspaces are matched on energy, not on d. Their k_rand is in the CSV.
