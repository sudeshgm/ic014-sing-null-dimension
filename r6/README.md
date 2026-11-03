# Round 6: audit of published null-space / residual scores

Pre-registration: `r6/PREREG_R6.md` (pushed on its own, commit a768051, before any real-model number). Scores: `scores_r6.py` (ViM, ActSub S→, CORE S_mem, conditional V-information, NuSA-style reference). Tests and driver: `audit_r6.py`. Statistics, verdicts and the counting rule: `verdicts_r6.py`. Tables and macros: `make_tables_r6.py`. Frozen synthetic unit tests: `tests/test_r6.py` (`python -m pytest -q r6/tests`; log at pre-registration in `results/unit_tests_prereg.log`).

Run (CPU; needs the private features):

    python r6/audit_r6.py --feats-root $IC014_FEATS --models vgg16,vgg19,... --worlds resnet,dinov2,resnext --threads 2
    python r6/verdicts_r6.py      # -> r6/results/R6_summary.json
    python r6/make_tables_r6.py   # -> paper_workshop/tables/{numbers_r6,tab_r6_*,tab_prereg_r6}.tex

The run used four processes of 2 threads on a shared 8-core machine: 3–74 min per model (the 4096-d VGG models are slowest) and about 1 h per calibration base.

## Outcome

* n = 4 scores with a verdict, k = 0 ENERGY-DRIVEN → **SING-SPECIFIC** by the counting rule (§4): the energy confound is largely specific to SING's AS/IS.
* ViM: NOT ENERGY-DRIVEN (τ_EQ 0.88 / 0.89, median R_B 0.06), content PARTIAL (C1 9/13, plant 5/13).
* ActSub S→: MIXED (τ_EQ 0.79 / 0.78, R_B 0.14), PARTIAL (11/13, 8/13).
* CORE S_mem: MIXED (τ_EQ 0.85 / 0.82, R_B 0.11), PARTIAL (11/13, 8/13).
* Conditional V-information: MIXED (τ_EQ 0.20 / 0.33, R_B 0.19), CONTENT-RESPONSIVE (11/13, 12/13); NA on ResNeXt101 (G5), still 11 of E12 and 10 of S11 estimable.
* NuSA-style reference (not counted): MIXED (τ_EQ 0.27 / 0.31, R_B 0.89), τ with SING 0.82 / 0.82.
* ESTIMATOR-NOISY (C2) is flagged for every score. Full numbers: `results/R6_summary.json`; per-model values: `results/models/*.json`; synthetic worlds: `results/worlds/*.json`. See `RESULTS.md` and the paper (§ "Do other null-space scores share the energy confound?", Appendix "Round 6").

Deviations from the pre-registration: the largest logit change on real data is 1.3e-5 (DenseNet121, whose null part is scaled up by 5e2–7e3; the 1e-8 tolerance in §1 refers to the synthetic tests); nothing else.
