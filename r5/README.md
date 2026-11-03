# Round 5 (`PREREG_R5.md`)

Round 5 fixes the calibration worlds of round 4 and tests a pre-registered organisation score. The pre-registration was committed on its own before any confirmatory run. Each script writes only aggregate numbers (per model at most) to `r5/results/` or `r4/results/`; features and per-image outputs stay private.
Seeds: 99000 (residual permutation for the worlds), 99100+s (XCC Haar frames), 99200 (SO folds), 99300 (permutation nulls), 99400 / 99401 (world-S and inheritance plants), 99500 (G1 rotation), 99600 (XCC self-term folds), 94000+r (class-mean derangements, as in round 4).

| Script | Prereg | Output | What it does |
|---|---|---|---|
| `../r4/calibration.py --resid perm` | R5 step 1 | `r4/results/I2_calibration_v2.json` | Worlds C, M, D with residuals permuted across images. The round-4 file `I2_calibration.json` (in-sample residuals) is kept. |
| `calib_compare.py` | R5 step 1 | `R5_calibration_old_vs_new.json` | Every world value, old vs new, with a changed flag. |
| `organisation.py --part gates` | R5 §1–2 | `R5_gates.json` | SO_null and XCC_24 on 13 models and the two true DINO models; gates G1 (invariance), G2 (class-mean derangement), G3 (world S), G4 (corrected world C; worlds B, M, D descriptive), G5 (estimability); kill rules K1/K2. |
| `organisation.py --part confirm` | R5 §3–4 | `R5_confirm.json` | Kendall τ against SING's order on E12 / S11 / all 13 with bootstrap CIs, outcome labels, τ with ρ and η², partial given ρ, and the inheritance diagnostic. Stops if K1 fired. |
| `make_tables_r5.py` | — | `paper_workshop/tables/numbers_r5.tex`, `tab_{r5_gates,r5_worlds,r5_confirm,calib_insample,prereg_r5}.tex` | All round-5 macros and tables, from the committed JSON only. |

Run: `make reproduce-r5 IC014_FEATS=/path/to/feats` (CPU: about 20 min for the worlds and 10 min for the organisation scores on 4 cores), or `make reproduce` for the tables from committed aggregates.

Outcomes: no kill fired. SO_null INCONCLUSIVE (τ 0.33 E12, 0.45 S11), XCC_24 DOES-NOT-TRACK (−0.12, −0.05); inheritance NOT INHERITED; G4 not testable for SO_null. Every outcome is in the round-5 ledger (paper Appendix, `tab_prereg_r5.tex`) and in `RESULTS.md`.

## Post hoc: SO_null with exact centring (not pre-registered)

`organisation.py` `dcent()` subtracts the off-diagonal row means of a hollow (zero-diagonal) matrix. For an additive class effect a_i + a_j that leaves (a_i + a_j − 2ā)/(N − 1) in every entry, and the row sums of the output are not centred, so additive per-class effects are not removed exactly (the docstring overstates it). In a synthetic case with no content and large class offsets correlated with the text marginal, this gives a spurious, estimable SO_null. `so_exact_r5.py` reruns SO_null with an exact hollow double-centring (`dcent_exact`) on both the distance and the text side, with the same features, folds, text, permutation null, G2 derangements and G3 plants, next to the original values (its copy of the original path reproduces `results/R5_gates.json`). `organisation.py`, `PREREG_R5.md` and the registered results are unchanged. Synthetic check: `tests/test_so_exact.py`.

    python r5/so_exact_r5.py --feats-root $IC014_FEATS    # -> results/posthoc/SO_exact_models.json, SO_exact_summary.json

Outcome: see `results/posthoc/SO_exact_summary.json` and the paper (Appendix "Round 5", "Centring correction (post hoc)"). Per-model changes are at most 0.0013, the 13-model order is unchanged (τ 1.00) and the round-5 outcome is still INCONCLUSIVE.
