# IC014 round-2 plan (response to review_20261101_v1.md, Weak Reject)

Criteria are in `PREREG_R2.md`, committed before any round-2 number is computed. Code goes in `r2/`, outputs in `r2/results/`.
Inputs: the cached private features under `$IC014_FEATS` (on our box `data/ic014`: E1 null features, E2/E3 `*_full.npz` for all 13 models, E3 null coordinates and CLIP concept scores) and SING at commit `e47b37d` (`scripts/fetch_sing.sh`).
Compute: CPU only (numpy/scipy/torch-cpu). Only fix 14 needs new feature extraction (spec in `gpu_jobs/`).

| Fix | What we run | Where | Needs GPU? |
|---|---|---|---|
| 1 | Table of all pre-registered experiments (Exp0, Exp1, E1, E2, E3, E8, E9, R2) with criterion and outcome; E8/E9 results and E8 method deviations in the paper and abstract. | paper | no |
| 2 | Two logit-preserving random controls next to the full-space Haar one: (B) energy-matched Haar subspaces inside the span of the lowest-σ row-space directions of W; (C) d-matched (d=24) random null sub-subspaces, normalised by family-B controls at the same energy. τ vs SING for each. | `r2/e2_controls.py` | no |
| 3 | Normaliser sweep AS/ρ^a (a ∈ {0.25, 0.43, 0.5, 1}), AS/d, AS_fair under families A/B/C; 2000× model bootstrap and leave-one-out ranges for τ vs SING. | `r2/e2_sweep.py` | no |
| 4 | Spearman(AS_fair, ρ) with model-bootstrap CI, all 13 and without DenseNet/BiT; fix the Fig. 8 caption/title. | `r2/e2_sweep.py` | no |
| 5 | Null-tolerance sweep (absolute 1e-6, 1e-5, 1e-4, 1e-3; relative 1e-5, 1e-4, 1e-3 × σ_max) of null dim, ρ, AS_SING, AS_fair for 13 models; pre-declared degeneracy screen; every headline number with and without screened models. | `r2/e2_controls.py --tol` | no |
| 6 | E1 cell-level bootstrap, cluster (model) bootstrap, leave-one-model-out, per-model a and b. | `r2/e1_ci.py` | no |
| 7 | E3 with α=0 subtraction for both scores, from the committed per-trial CSV plus the α=0 table. | `r2/e3_symmetric.py` | no |
| 8 | Signed effect + 95% image-bootstrap CI for every claim C1–C12; "reverses" only if the CI is strictly on the opposite side; C7 not applicable to AS_fair; C8/C10 flagged. | `r2/e2_claims_ci.py` | no |
| 9 | Synthetic oracle test of the flatness rule (pure numpy), plus a post-hoc re-adjudication of the E8/E9 flatness kills under a revised, oracle-relative rule. | `r2/flatness_synth.py` | no |
| 10 | E9 holdout (water, green, thin, knitted; seed 1; linear and nonlinear plant) for a declared candidate subset plus AS_SING and AS_fair, labelled post hoc. The plant is synthetic, so the cached E3 features and CLIP concept scores suffice; no new extraction. E2 ranking under M1 (descriptive). | `r2/e9_holdout.py` | no |
| 11 | Fix bib (SING authors, Rezaei, li2024null), pointers (Supp. Table 3, Supp. Fig. 12b), add Idnani et al. 2023, amnesic probing, INLP, control tasks, one CKA-confound paper; verify each entry against its arXiv/OpenReview/CVF page. | paper | no |
| 12 | `requirements-lock.txt`, `Makefile` (`make reproduce` from committed CSVs; `make reproduce-full` from cached features), `scripts/fetch_sing.sh` with commit check, `scripts/checksums.sha256` for cached features, image list, figure script. Features are not released (ImageNet terms); a fresh clone without features can rebuild all tables/figures from committed CSVs. | repo | no |
| 13 | Energy as confound vs signal: at d=24 (and 100) in the null of the 4 null-1048 models, AS/√ρ for top-variance, middle, bottom-variance and random null PC bands, image-bootstrap CIs. | `r2/pc_bands.py` | no |
| 14 | Translator fidelity cos(T(f), CLIP) for the other four ViT-L candidates on a 1k-image subset. Needs ImageNet-val images and ViT-L forward passes (not on this box). | `gpu_jobs/fix14_vitl_translator_fidelity.md` | yes |

Order: prereg commit → fixes 6, 7, 4 (seconds) → 2, 3, 5 (E2 recompute) → kill check → 8 → 9 → 13 → 10 → paper/docs → fix 12 → push.

## Status (2026-11-01)
1 done · 2 done (pass) · 3 done (D-weak pass, D-strong fail) · 4 done (trend not removed on screened set) · 5 failed (min adjacent τ < 0.9 for AS_SING and AS_fair) · 6 done · 7 done · 8 done · 9 done (registered rule ill-posed; revised rule) · 10 see `r2/results/fix10_e9_holdout.json` · 11 done · 12 done · 13 done · 14 needs GPU (`gpu_jobs/`). KILL-R did not fire.
