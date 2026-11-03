# IC014 round 6 pre-registration: auditing published null-space / residual content scores (committed and pushed before any real-model number is computed)

Date: 2026-11-03 (IST). Adds round 6 to `PREREG_R2.md`–`PREREG_R5.md`; none of them is edited. Model sets (all 13, E12 = the 12 models with ρ ≥ 0.005, S11 = E12 without DenseNet121 and BiT-R50x1), SING's order (the digitised IS/AS ratio; leakage = −ratio, as in round 5), ρ (null energy fraction) and Π (projector onto ker W) are as in `PREREG_R4.md` / `PREREG_R5.md`; `tau_ci`, `partial_ci` are the round-3/4 helpers. Code: `r6/scores_r6.py` (scores), `r6/audit_r6.py` (interventions and driver), `r6/verdicts_r6.py` (statistics, verdicts, counting rule), frozen synthetic unit tests `r6/tests/test_r6.py`. All are committed with this file.

## 0. What was known when this was written (plain disclosure)

- **Round 3, NuSA-style null share** (`r3/equal_energy.py`): τ with SING's order 0.85 (all 13), τ(NuSA, ρ) 0.97. NuSA is therefore a reference row and never counts in §4.
- **Idea P1 (conditional probing) scratch** (`_local/ideas/ideas_C_invariant_scores.md`): a whitened, fixed-k Haar variant on ResNet50 and DINOv2-L gave class-label conditional gains of 1.3e-3 and 4.9e-4 nats (about 8–14× a shuffled control), ≈0 for a concept target, and null-only class decoding of 1.1 and 3.0 nats. The raw (non-whitened) family of §1 D has not been run on any real model.
- **Ideas G/H scratch** (NIC, WC-CKA) on ResNet50, DINOv2-L and VGG16: different scores, not audited here.
- **Never computed on any real model before this commit:** ViM, ActSub S→, CORE S_mem, CANARY, the V-information of §1 D, and every round-6 test. Before the commit the code was run only on synthetic data (unit tests, and timing runs on synthetic matrices of the real sizes). From the real feature files only array shapes were read (d ∈ {1024 ×5, 1280, 1536, 2048 ×5, 4096 ×2}; d_null = d − 1000).
- **R4C** listed a whitened probe and a LEACE gap as backups; they were not run.
- **Prior expectations** (written before any data, `ideas_B_scope.md` and the round-6 planning note): norm scores were expected to track ρ, ActSub's cosine not to; CORE was expected to be energy-sensitive through mixing and content-responsive; V-information possibly not estimable.
- **Bars.** τ bars 0.6 / 0.3 and the 1.5 world-C ratio are reused from round 5. The new bars (R_B 0.5 / 0.1, 2 / 1 bootstrap SD, 10 of 13 models, the 0.1 estimability ratio) were set without any round-6 data.

## 1. Scores (faithfulness check against the official code, and deviations)

Common protocol: class-stratified random halves A / B, 5 of the 10 images per class each (seed 100100). Every fitted quantity is fitted on one half and evaluated on the other; the halves are swapped, so every image gets exactly one out-of-fit value; the model-level value is the mean over the 10 000 images. ŷ = argmax of the head's logits (unchanged by every intervention). **Deviation common to all scores:** the papers fit on the ImageNet training set; we have only the 10k validation features (10 per class), so all fits use 5 images per class from the other half.

**A. ViM** (Wang et al., CVPR 2022, arXiv 2203.10807; official `github.com/haoqiwang/vim`, `benchmark.py`, commit dabf9e5).
- Checked in the official code: origin u = −pinv(W)·b; `EmpiricalCovariance(assume_centered=True)` of (f − u); residual space = eigenvectors after the top D; **D = 1000 if d ≥ 2048, 512 if d ≥ 768, else d/2** (so D = 1000 for our 2048- and 4096-d models, 512 for 1024/1280/1536-d); α = mean max-logit / mean residual norm on the fit set; official per-image score s = logsumexp(logits) − α·R.
- Our model-level score: the mean of p_virtual = softmax([logits, αR])_virtual = sigmoid(αR − logsumexp), the paper's virtual-logit probability (a monotone function of the official s). The mean of s is descriptive (ViM_s).

**B. ActSub insignificant score S→** (Zöngür, Hesse, Roth, ICCV 2025, arXiv 2508.21695; official `github.com/visinf/actsub`, `actsub_standard/methods.py` and `actsub_openood/.../actsub_postprocessor.py`, commit 5b058e7).
- Checked in the official code: full SVD W = UΣVᵀ (d × d Vᵀ, so ker W is in the insignificant part); the split index k = argmin over idx of |mean‖a←‖ − mean‖a→‖| on the training embeddings (`find_c_idx`); reference set = a random 10 % of the training embeddings (`extract_train_samples.py`); **N = 10** nearest cosine neighbours (`f_cossim(..., k=10)`, the non-prototype default); S→ = −log(1 − mean top-10 cosine) (Eq. 7).
- Ours: k fitted on the fit half; reference = random 10 % of the fit half (500 images; seed 100150 for fit half A, 100151 for B). Only S→ is audited (S← is an energy of shaped decisive logits). Model level: mean S→.

**C. CORE membership score S_mem** (Yang, Kim, Bahk, arXiv 2603.18290, Eq. 1 and 4; no official code located).
- z⊥ = z − (z·w_ŷ/‖w_ŷ‖²) w_ŷ (residual w.r.t. the predicted class's weight); μ⊥^(c) = normalised mean of z⊥ over fit-half images with label c; S_mem = cos(z⊥, μ⊥^(ŷ)). Model level: the mean. The paper states that ≈13 samples per class suffice; we have 5 (deviation).
- Descriptive variant CORE_ker: the same with z⊥ replaced by Πz.

**D. Conditional V-information** (Hewitt, Ethayarajh, Liang, Manning, EMNLP 2021, arXiv 2109.09234).
- φ = coordinates of Πf on the top-min(64, d_null) principal directions of the fit half's Πf (centred on the fit-half mean, no standardisation). B = the head's 1000 logits. Y = class label (primary, `Vinfo`), or top-1 correctness (descriptive, `Vinfo_corr`, with B = [max logit, logsumexp, max softmax, top-2 margin], standardised on the fit half).
- Family V (**deviation, disclosed**): softmax(t·B + U φ + c) with a scalar t on the head's own logits (for correctness: an unpenalised linear map of the 4 summaries), class bias c and U penalised by λ‖U‖²; the baseline [B; 0] is the same family with U = 0. A full affine map of the 1000 logits is not identifiable from 5 images per class.
- Score = mean held-out NLL([B;0]) − NLL([B;φ]) in nats. λ = λ_rel / mean‖φ‖² on the real fit half; λ_rel ∈ {1e−4, 1e−3, 1e−2, 1e−1, 1} chosen once per model and fit half on real data by inner 2-fold class-stratified CV (seed 100700), then both λ_rel and mean‖φ‖² are **frozen** for every intervention (so the penalty is fixed in absolute terms, as for a fixed probe family). Fitting: torch L-BFGS (strong Wolfe, ≤ 200 iterations), float32; NLL evaluated in float64.
- **Fallback (pre-stated):** if V-information fails G5 (§2), the family count in §4 uses the 3 remaining scores.

**Reference. NuSA-style null share** (Cook, Zare, Gader, arXiv 2007.01263): mean ‖Πf‖/‖f‖. Never counted.

**Not run in round 6.** CANARY (Shin & Kim, arXiv 2610.33063): its per-class k = 100 neighbour rule cannot be implemented with 5 references per class. Rezaei & Sabokrou (arXiv 2305.19424), NECO, INLP/LEACE/amnesic probing: see the planning note; not scores of null content here.

**Implementation freeze.** `r6/tests/test_r6.py` (9 tests, synthetic data only) passes at this commit: ViM equals a line-by-line port of the official `benchmark.py` (and the D rule); ActSub's k equals the official `find_c_idx` loop and S→ equals the official `f_cossim` formula; CORE's residual is orthogonal to w_ŷ and S_mem = 1 for collinear residuals; V-information ≈ 0 (|gain| < 0.02 nats) when φ is independent of Y given B and > 0.3 nats when Y is planted in φ, for both targets; every intervention keeps the logits unchanged (≤ 1e−8 on synthetic), hits its ρ target exactly and keeps the null energy where required.

## 2. Tests (identical for every score, all 13 models unless stated)

All interventions change only Πf; logits are unchanged (max |Δlogit| reported). After each intervention the score is recomputed and every fitted quantity refitted (ViM's subspace and α, ActSub's k and references, CORE's means, V-info's PCA and probe), with the frozen halves and the frozen V-info λ.

- **E1, equal energy.** Πf rescaled so that ρ = ρ₀ = 0.005 exactly (Πf → s·Πf, s² = ρ₀E_row / ((1−ρ₀)E_null)). This *sets* the remaining null energy equal across models; it differs from round 3's A1, which removed an equal energy. Statistic τ_EQ = τ_b(S_E1, S_real) over the estimable models of E12 and of S11.
- **E2, world B.** Πf rescaled so that ρ equals each of the 12 E12 ρ values. R_B(m) = (max − min of S_m over the 12 targets) / (max − min of S_real over the 13 models); the statistic is the median over estimable models. Also τ_EO(base) = τ_b(S_m(real), S_base(ρ_m)) over the estimable E12 models, for the bases ResNet50, DINOv2-L, ResNeXt101 (median reported).
- **C1, content-destroying surrogate at fixed energy.** The rows of Πf permuted across images (5 draws, seeds 100200 + r). C_m = (S_real − mean S_perm) / SD_boot.
- **C2, content-preserving control.** Πf permuted within each class (5 draws, seeds 100250 + r). Flag ESTIMATOR-NOISY if |mean S_wc − S_real| > 1 SD_boot on more than 3 of 13 models (reported, not a kill).
- **C3, class-code plant at fixed energy.** Arbitrary Gaussian class code (1000 × d_null), and the semantic code (centred class texts × a Gaussian 768 × d_null matrix), both from seed 100400 (text matrix drawn first), each scaled to the null energy; Πz′ = Πz + α·u_y rescaled back to the original null energy; α ∈ {0.1, 0.3}. The verdict uses the arbitrary code at α = 0.3: Δ = (S_plant − S_real)/SD_boot. The semantic code is descriptive (none of the scores references class text).
- **SD_boot:** SD of the model-level mean over 200 image bootstraps of the per-image out-of-fit values (seed 100600), on real data.
- **W, synthetic worlds** (reported, not part of the verdicts). Worlds C, M, D of `r4/calibration.py --resid perm`, rebuilt with residual permutation seed 100500, on the 3 bases, 12 rows each (targets = the E12 ρ values; L = ½·min ρ in C; as in round 5). For every row the score's content-free floor is its value after one across-image permutation (seed 100200). World C (content fixed, ρ varies) passes on a base if max/min of (S − floor) ≤ 1.5 with all values > 0. τ(S, L) in M and D and τ(S, ρ) in C are reported.
- **G5, estimability.** A model is NA for a score if SD_boot > 0.1 × (max − min of S_real over the 13 models). A score gets no verdict if fewer than 9 E12 or 8 S11 models are estimable.

## 3. Verdicts (per score, symmetric)

- **Energy verdict.** ENERGY-DRIVEN if τ_EQ < 0.3 on both E12 and S11 AND median R_B ≥ 0.5. NOT ENERGY-DRIVEN if τ_EQ ≥ 0.6 on both AND median R_B ≤ 0.1. MIXED otherwise.
- **Content verdict** (absolute values, the sign is reported; for ViM the direction of a content effect is not defined in advance). CONTENT-RESPONSIVE if |C_m| ≥ 2 on ≥ 10 of 13 AND |Δ| ≥ 2 on ≥ 10 of 13. CONTENT-BLIND if |C_m| ≤ 1 on ≥ 10 of 13 AND |Δ| ≤ 1 on ≥ 10 of 13. PARTIAL otherwise.
- **Cross-model (descriptive, R5 format).** τ_b(S, SING order) on the estimable E12 / S11 / all-13 models with `tau_ci` (2000 model resamples, seed 0, percentile and BCa); TRACKS if τ ≥ 0.6 on both E12 and S11, DOES-NOT-TRACK if < 0.3 on both, else INCONCLUSIVE. τ_b(S, ρ) and the partial Spearman with SING's order given ρ (`partial_ci`) are reported. τ vs ρ is never a validity criterion (a small monotone bias gives |τ| = 1); the energy verdict comes from the interventions.
- **No kill for being energy-driven** (that is the measured outcome). A score is excluded from §4 only by G5. (All unit tests pass at this commit.)

## 4. Counting rule and pre-registered statements

Let n = number of A–D with a verdict (G5) and k = number labelled ENERGY-DRIVEN. NuSA never counts.

- **GENERAL** ("published null/residual content scores are generally confounded by discarded energy") iff k ≥ 3 with n = 4, or k ≥ 2 with n = 3.
- **DESIGN-DEPENDENT** iff n = 4 and k = 2; the paper names which designs escape.
- **SING-SPECIFIC** iff k ≤ 1 (n = 3 or 4). The paper then states that the energy confound is largely specific to SING's AS/IS, keeps its SING-centred title and presents the audit as a protocol that most past scores pass.
- **CONTENT IS MEASURABLE** is stated additionally iff ≥ 2 counted scores are both NOT ENERGY-DRIVEN and CONTENT-RESPONSIVE.
- If n ≤ 2, no family-level statement; per-score results only.
- The title and abstract of the paper may be broadened beyond SING only under GENERAL or DESIGN-DEPENDENT.

## 5. Reporting

Every per-model value, every statistic, every NA, the unit-test log and each verdict with CIs go to `r6/results/` (`models/*.json`, `worlds/*.json`, `R6_summary.json`), `r6/README.md`, `RESULTS.md` and the paper, whatever the outcome. Any deviation from this file is listed with its reason.
