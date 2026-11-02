# Round 4 (`PREREG_R4.md`, `PREREG_R4B.md`, `PREREG_R4C.md`, `ROUND4_PLAN.md`)

Round-4 analyses respond to two reviews of the round-3 paper. Each script writes only aggregate numbers to `r4/results/` (per-model at most); per-image outputs and features stay private.
Registered seeds: 92000+ (surrogates), 93000+ (within-class permutation), 94000+ (class-mean derangement), 95000 (NCM shuffled labels), 96000+ (selectivity), 97000 (NZS control), 98000 (synthetic geometry), 70000 (Gaussian sanity check).

| Script | Prereg | Output | What it does |
|---|---|---|---|
| `surrogates.py` | R4 S | `S_surrogates{,_permodel}.json` | Content-free null surrogates (Gaussian, zero-mean, isotropic, class mean, across-image permutation) and one content-bearing one (within-class permutation), each rescaled to the model's null energy; τ against SING's order, the excess X = AS − AS_surr and the partial τ given AS_surr. |
| `retrained_a4.py` | R4 J | `J_retrained_a4{,_permodel}.json` | κ, γ and A4 under the retrained ridge translators. |
| `a1_stats.py` | R4 J | `J_a1_stats.json` | A1 as AS per unit null energy; jackknife and leave-two-out for both translator families; paired retrained-vs-released CIs. |
| `planted.py` | R4 P | `P_planted.json` | Planted content against the energy range r_E; α*; ratios at each model's own ρ (DINOv2-L: ×1.17 at α = 0.3). |
| `dino.py` | R4 DN | `DN_dino.json` | True DINO ViT-B/16 and ViT-S/16 (official linear heads) and a supervised AugReg ViT-B/16 control; ridge translators; ranks among 15 models. Needs GPU-extracted features. |
| `class_perm.py` | R4B idea 1 | `I1_class_perm.json` | Class-mean derangement (energy preserved exactly); AS_cperm and Δ_align. |
| `calibration.py` | R4B idea 2 | `I2_calibration.json` | Semi-synthetic worlds C, M, D for every estimator. |
| `selectivity.py` | R4B idea 3 | `I3_selectivity.json` | Class-discriminative vs energy-matched random null removal (R). |
| `ncm_null.py` | R4B idea 5 | `I5_ncm_null.json` | Translator-free nearest-class-mean decodability of the null coordinates. |
| `dino_addendum.py` | R4B idea 6 | `I6_dino_addendum.json` | Δ_align and NCM_null for the DINO models against prediction intervals. |
| `geometry.py` | R4C B1 | `B1_geometry.json` | In-plane/out-of-plane account of AS; G1 (13 models, exploratory), G2 (DINO, confirmatory), G3 (synthetic). |
| `nzs.py` | R4C B3 | `B3_nzs.json` | Null zero-shot accuracy with its validity gate. |
| `is_energy.py` | R4C B4 | `B4_is_energy.json` | IS against row-space energy 1 − ρ and κ_row. |
| `make_tables_r4.py` | — | `paper_workshop/tables/numbers_r4.tex`, `tab_*.tex` | All round-4 macros and tables, from the committed JSON only. |

`common4.py` and `estimators.py` are shared helpers.

Run: `make reproduce-r4 IC014_FEATS=/path/to/feats` (feature-level; CPU apart from the DINO extraction), or `make reproduce` for the tables from committed aggregates.
Outcomes: no pre-registered kill fired. Every outcome, pass or fail, is in the round-4 ledger (paper Appendix, `tab_prereg_r4.tex`) and in `RESULTS.md`.
N10 (holdout-fast under 30 min) was not re-timed in round 4 (as `PREREG_R4.md` states); it fails deterministically on accuracy (0.057 > 0.03), and the machine was shared with other jobs during round 4.
