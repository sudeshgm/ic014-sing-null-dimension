# Round 6: post hoc analyses

Written after the round-6 results (`r6/results/R6_summary.json`) were known. Nothing here changes `PREREG_R6.md`, the
frozen round-6 code (`audit_r6.py`, `scores_r6.py`, `verdicts_r6.py`, `make_tables_r6.py`, `tests/test_r6.py`) or any
registered outcome. Every item is labelled post hoc in the paper.

## Why
The registered counting rule (k ≤ 1 of n scores ENERGY-DRIVEN → "largely specific to SING") was never checked on scores
that should be flagged. The NuSA reference, an energy score by construction, came out MIXED. These analyses check the rule
on positive controls, check the built-in normalisation of the audited scores, and check the sensitivity of the
V-information result.

## Analyses
1. **Positive controls** (`controls_r6.py`, `posthoc_r6.py`). SING's AS and IS (class-text target) go through exactly
   the round-6 interventions (real with the 200-resample image bootstrap, E1, E2, C1, C2, C3; same seeds), with the
   released translator (all 10k images) and the round-3 ridge translator (refit on the fixed training half, seed-0
   permutation, with the penalty in `r3/results/G_retrain_permodel.json`, scored on the other 5000 images). The verdict
   code is the unchanged `verdicts_r6.score_block`. ρ itself is also run through the energy part of the rule.
2. **ViM with α frozen** (`controls_r6.py`). α is fixed at its real-data value for each fit half under E1 and E2. The
   refit-α values reproduce the audit's values (consistency check stored in `R6_posthoc.json`).
3. **Retained cross-model range.** range(E1)/range(real) after E1, over the estimable E12
   (and S11) models, for every score and control.
4. **Power of the τ bar.** P(τ_EQ < 0.3) when the orders before and after E1 are independent (n = 12, 11; 20 000
   permutations, seed 0).
5. **V-information sensitivity.** Range and estimability without DenseNet121, R_B and τ_EQ on that basis,
   non-positive gains, C2 vs C1, penalty choices; and (`vinfo_repro_r6.py`) whether the inner-CV penalty choice for
   ResNet50 reproduces, with the real-data value for each penalty combination.
6. **C2 synthetic check** (`tests/test_posthoc_r6.py`). The within-class permutation keeps each class's null
   distribution, yet it can remove real image-level conditional information. So the ESTIMATOR-NOISY flag cannot
   separate a real loss of information from estimator noise.

## Readings withdrawn
* "Largely specific to SING" (the registered statement for k ≤ 1). The rule flags SING's IS but not SING's AS, the
  NuSA share or ρ, so k is uninformative.
* The content labels (PARTIAL, CONTENT-RESPONSIVE). They are kept as registered outcomes but not used as readings,
  because the C2 flag fires for every score and (item 6) C2 is not a clean noise control.

## Outcome
* AS, released: τ_EQ 0.39 / 0.45 (E12 / S11), median R_B 0.26 over the 13 estimable models (0.25 over E12), MIXED.
  AS, retrained: 0.85 / 0.85, R_B 0.67, MIXED. IS, released: 0.12 / 0.24, R_B 0.57, ENERGY-DRIVEN; retrained: 0.12 / 0.24,
  R_B 0.79, ENERGY-DRIVEN. ρ: τ_EQ undefined, MIXED. NuSA share (registered): MIXED. The released AS and IS values
  reproduce the committed round-2 values.
* ViM, α frozen: ResNet50 0.876 → 0.00025 under E1; τ_EQ −0.52 / −0.49, R_B 1.33, ENERGY-DRIVEN. Refit-α values equal
  the audit's (max difference 0).
* Retained cross-model range after E1 (E12): NuSA 0.005, IS 0.11 / 0.05, AS 0.11 / 0.15 (released / retrained), V-info 0.29,
  ActSub 0.78, ViM 0.90 (1.93 with α frozen), CORE 0.92. This is a descriptive comparison, not a validity gate and
  not the fraction of confounding removed: ViM with α frozen retains 1.93 of its range yet is ENERGY-DRIVEN.
* P(τ_EQ < 0.3) under independent orders: 0.90 (n = 12), 0.89 (n = 11).
* V-information: DenseNet121 is 86% of the 13-model range; without it 7 of E12 and 7 of S11 are estimable, so no verdict
  is possible. ResNet50 penalty: the inner CV chose λ_A/λ_B = 1/0.1 in 2 of 2 reruns, as in the audit (0.00158); a flip
  to 0.00103 was not reproduced, but the value is 0.00087–0.00158 over the four λ combinations in {0.1, 1}².
* C2 synthetic: a conditional gain of 1.18 nats falls to −0.01 after the within-class permutation.

A first version of `controls_r6.py` computed IS against the unnormalised row-space translation, which clipped every
angle to 0. It was corrected (IS as in `r2/e2_controls.py`) and the AS/IS part rerun (`--part sing`) before any IS
number was used.

Round 5: the separately labelled SO_null rerun with exact centring is in `r5/so_exact_r5.py` (see `r5/README.md`).

## Outputs
`r6/results/posthoc/controls_<model>.json`, `r6/results/posthoc/vinfo_repro_resnet.json`, `r6/results/R6_posthoc.json`,
`paper_workshop/tables/numbers_r6p.tex`, `paper_workshop/tables/tab_r6_compact.tex`.
