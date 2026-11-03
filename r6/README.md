# Round 6: audit of published null-space / residual scores

Pre-registration: `r6/PREREG_R6.md` (pushed together with its code and synthetic unit tests, commit a768051, before any real-model number). Scores: `scores_r6.py` (ViM, ActSub S→, CORE S_mem, conditional V-information, NuSA-style reference). Tests and driver: `audit_r6.py`. Statistics, verdicts and the counting rule: `verdicts_r6.py`. Tables and macros: `make_tables_r6.py`. Frozen synthetic unit tests: `tests/test_r6.py` (`python -m pytest -q r6/tests`; log at pre-registration in `results/unit_tests_prereg.log`).

Run (CPU; needs the private features):

    python r6/audit_r6.py --feats-root $IC014_FEATS --models vgg16,vgg19,... --worlds resnet,dinov2,resnext --threads 2
    python r6/verdicts_r6.py      # -> r6/results/R6_summary.json
    python r6/make_tables_r6.py   # -> paper_workshop/tables/{numbers_r6,tab_r6_*,tab_prereg_r6}.tex

The run used four processes of 2 threads on a shared 8-core machine: 3–74 min per model (the 4096-d VGG models are slowest) and about 1 h per calibration base.

## Outcome

* n = 4 scores with a verdict, k = 0 ENERGY-DRIVEN → SING-SPECIFIC by the counting rule (§4). **Reading withdrawn (post hoc):** the rule flags SING's IS but misses SING's AS, ρ and the NuSA share (`POSTHOC_R6.md`), so k = 0 does not show that the confound is specific to SING.
* ViM: NOT ENERGY-DRIVEN (τ_EQ 0.88 / 0.89, median R_B 0.06), content PARTIAL (C1 9/13, plant 5/13).
* ActSub S→: MIXED (τ_EQ 0.79 / 0.78, R_B 0.14), PARTIAL (11/13, 8/13).
* CORE S_mem: MIXED (τ_EQ 0.85 / 0.82, R_B 0.11), PARTIAL (11/13, 8/13).
* Conditional V-information: MIXED (τ_EQ 0.20 / 0.33, R_B 0.19), CONTENT-RESPONSIVE (11/13, 12/13); NA on ResNeXt101 (G5), still 11 of E12 and 10 of S11 estimable.
* NuSA-style reference (not counted): MIXED (τ_EQ 0.27 / 0.31, R_B 0.89), τ with SING 0.82 / 0.82.
* ESTIMATOR-NOISY (C2) is flagged for every score. A synthetic check (`tests/test_posthoc_r6.py`) shows that C2 can remove real image-level information while keeping each class's null distribution, so the flag cannot separate that from estimator noise. The content labels (PARTIAL, CONTENT-RESPONSIVE) are registered outcomes, not readings. Full numbers: `results/R6_summary.json`; per-model values: `results/models/*.json`; synthetic worlds: `results/worlds/*.json`. See `RESULTS.md` and the paper (§ "Do other null-space scores share the energy confound?", Appendix "Round 6").

Deviations from the pre-registration: the largest logit change on real data is 1.3e-5 (DenseNet121, whose null part is scaled up by 5e2–7e3; the 1e-8 tolerance in §1 refers to the synthetic tests). There are no other deviations.

## Post hoc (`POSTHOC_R6.md`)

    python r6/controls_r6.py --feats-root $IC014_FEATS --force   # SING AS/IS and frozen-alpha ViM -> results/posthoc/controls_<model>.json
    python r6/vinfo_repro_r6.py --feats-root $IC014_FEATS --model resnet --reps 2
    python r6/posthoc_r6.py               # -> results/R6_posthoc.json
    python r6/make_tables_r6_posthoc.py   # -> paper_workshop/tables/{numbers_r6p,tab_r6_compact}.tex

Positive controls: AS released τ_EQ 0.39 / 0.45, R_B 0.26, MIXED; AS retrained 0.85 / 0.85, 0.67, MIXED; IS released 0.12 / 0.24, 0.57, ENERGY-DRIVEN; IS retrained 0.12 / 0.24, 0.79, ENERGY-DRIVEN; ρ MIXED. ViM with α frozen: ResNet50 0.876 → 2.5e-4, ENERGY-DRIVEN. Retained cross-model range after E1 (descriptive, not a validity gate): NuSA 0.005, AS 0.11, V-info 0.29, ActSub 0.78, ViM 0.90, CORE 0.92. V-information penalty rerun on ResNet50: same choice as the audit in 2 of 2 reruns. The value is 0.00087–0.00158 over neighbouring penalties. Full list in `POSTHOC_R6.md`.
