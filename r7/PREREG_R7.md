# IC014 round 7 pre-registration: semantic edits and independent nuisance views (committed and pushed before any real feature is extracted)

Date: 2026-11-03 (IST). Adds round 7; `PREREG_R2.md`-`PREREG_R5.md` and `r6/PREREG_R6.md` are not edited. Plan and data
choice: `r7/PLAN.md`. Code: `r7/views_r7.py` (edits, controls, nuisance views), `r7/extract_r7.py` (features), `r7/stats_r7.py`
(statistics), `r7/analyze_r7.py` (per-model analysis and verdicts), `r7/synth_r7.py` (synthetic stability study); frozen unit
tests `r7/tests/test_r7.py` (10 tests, synthetic only). All are committed with this file.

## 0. What was known when this was written (plain disclosure)

- **No real feature of any round-7 image existed at this commit.** The IN-9 test images were downloaded and opened only to
  check pairing, masks and file names (§1), and the data pipeline was timed on two images without any model.
- CPU throughput was timed with randomly initialised networks on random tensors (no real image).
- The synthetic stability study (§4) was run before this file was finalised; its results are committed in
  `r7/results/synthetic_stability.json` and determined two choices, disclosed here: (a) the primary statistic is **T2**
  (first-order normaliser), not the reviewer-style **T** (fourth-order normaliser), because under low-rank noise T is
  undefined in 20-58 % of draws from noise-to-signal ratio nu = 2 (T* = 0.2), while T2 is never undefined; (b) the kill threshold
  nu_max >= 2 in K0 was written after seeing that T2 reaches nu = 2 and T only nu = 1 (T*=0.2, low-rank noise).
- The outside reviewer's own candidate (cumulant / trace-invariant comparison) was unstable at high noise with 5,000 samples;
  T is our version of that kind of fourth-order normaliser and shows the same failure in §4.
- Prior expectation (written now): a background edit is a large change of the input; we expect the null-space edit to be
  reproducible (H1), are unsure about alignment with the background's text embedding (H2), and expect the nuisance levels to
  change the edit content somewhat (strong crops can remove background), which can make H3 fail for a reason other than noise.

## 1. Data, edits, views, models

- **Images.** IN-9 Backgrounds Challenge test set (`PLAN.md`); for each of the 9 classes, 100 of the 450 foreground images
  (seeded choice, seed 100701, from the sorted `mixed_next` file list): N = 900.
- **Conditions.** X = `mixed_next` (background from the next IN-9 class; its fine ImageNet synset b_i from the file name);
  S = `mixed_same` (same file name; background from another image of the same class); Xs, Ss = X, S with the background
  pixels (fg_mask False) permuted among themselves by one permutation per image (seed [100710, i]), foreground untouched.
- **Views.** Level 0: clean (1 view). Levels 1-3 (weak / medium / strong): 4 independent views each; view (L, i, v) draws one
  parameter set from rng([100700, L, i, v]) (RandomResizedCrop scale (0.85|0.60|0.35, 1), flip, brightness / contrast /
  saturation +-(0.10|0.25|0.40), hue +-(0|0.02|0.05), grayscale p (0|0.05|0.10), Gaussian blur p (0|0.3|0.5) with sigma up
  to (0|1.5|2.5), JPEG quality (85-95|60-95|30-95)) and applies it identically to X, S, Xs, Ss. Then each model's own
  evaluation preprocessing. 52 forward passes per image.
- **Models.** resnet, efficientnet, resnext (CPU) and dinov2 (GPU job, same code); SING registry at commit e47b37d, the
  classifier-input hook used for every cached feature. Null basis Vn and row basis Vr from `null_split(W)` (tol 1e-5) as in
  rounds 4-6.
- **Edit vectors.** Delta_{i,v} = Vn^T (f(X_{i,v}) - f(S_{i,v})); artefact control Delta~ = Vn^T (f(Xs) - f(Ss)); nuisance-only
  copies N1 = Vn^T (f(S_{i,0}) - f(S_{i,1})), N2 = Vn^T (f(S_{i,2}) - f(S_{i,3})). Row-space analogues with Vr (descriptive).
- **Reference.** R_i = SING's CLIP ViT-L/14 text embedding of ImageNet class b_i (templates averaged, unit norm, as in round 6).
  Group g_i = the foreground IN-9 class. All statistics are **within group** (they ignore class-level structure of the
  foreground and use only the image-level variation of the edit and of the background class within each foreground class).

## 2. Statistics (`stats_r7.py`)

- **Reliability** (H1): within-group centred split-half reliability of the edit, views {0,1} vs {2,3}:
  Rel = sum_i <A_i, B_i> / sqrt(sum ||A_i||^2 sum ||B_i||^2).
- **T2** (primary): sum_g HSIC_u(Gbar_g, K_g) / (sum_g tr_u(Gbar_g)/n_g * sqrt(sum_g HSIC_u(K_g, K_g))), with Gbar the
  cross-view Gram (mean over ordered view pairs a != b of Delta_a Delta_b^T), K = R R^T, HSIC_u Song et al.'s unbiased
  estimator (off-diagonal entries only) and tr_u the unbiased within-group trace of Cov(mu) from Gbar (diagonal included,
  which is unbiased across views). Population value sum_g ||Cov_g(mu, R)||^2 / (sum_g tr Cov_g(mu) * sqrt(sum_g ||Cov_g(R)||^2)):
  the share of the within-class edit variance aligned with the reference. Numerator and normaliser are unbiased whatever the
  covariance of noise that is independent across views and images; rescaling the null part leaves T2 unchanged exactly.
- **T** (secondary, reviewer-style): same numerator over sqrt(HSIC_u(S_ab, S_cd) * HSIC_u(K, K)), averaged over the 3 pairings
  of the 4 views (an unbiased fourth-order normaliser); undefined if that is <= 0.
- **T_naive** (manipulation check): classic same-view linear CKA, within group, averaged over views (noise-attenuated).
- **nu** = (mean per-view within-group edit energy - cross-view energy) / cross-view energy (per-view noise-to-signal ratio).
- **Inference.** Within-group permutation of the reference rows, 1000 permutations (seed 100720 + L); within-group image
  bootstrap, 500 resamples (seed 100730), percentile 95 % CIs for T, T2, Rel and the paired differences T2(edit) - T2(artefact)
  and T2(edit) - nuisance-only (control numerator on the edit's T2 scale).

## 3. Hypotheses and verdicts (per model; primary level L = 2)

- **Estimability.** A statistic is estimable at level L if nu_L <= nu_max (from §4: T2 2, T 1) and, for T, at most 5 % of
  bootstrap resamples are undefined.
- **H1 (reproducible null-space edit).** REPRODUCIBLE iff the bootstrap CI lower bound of Rel (null space, L2) > 0.10.
- **H2 (alignment with the outside reference, above controls), for T2.** ALIGNED iff permutation p < 0.01 AND the CI of
  T2(edit) - T2(artefact) excludes 0 from above AND the CI of T2(edit) - nuisance-only excludes 0 from above.
  NOT ALIGNED iff p >= 0.05. ARTEFACT iff p < 0.01 but the artefact difference CI includes 0 or is negative. PARTIAL otherwise.
  NOT ESTIMABLE if T2 is not estimable at L2. (T: the permutation test only, descriptive.)
- **H3 (noise cancellation in a world-C analogue built from real views).** Tested only if H2 = ALIGNED. Manipulation check:
  nu_3 / nu_1 >= 1.5, else INCONCLUSIVE. NOT ESTIMABLE if T2 is not estimable at any level. CANCELS iff T2 > 0 with
  permutation p < 0.01 at all three levels and max / min of T2 over the three levels <= 1.5 (the round-5/6 world-C bar);
  DOES NOT CANCEL otherwise. T_naive over the levels and the cross-view (signal) energy per level are reported beside it,
  because a drop in signal energy with stronger crops would mean content changed, not only noise.
- **Per-model summary.** "Real views support noise cancellation" iff H1 REPRODUCIBLE, H2 ALIGNED and H3 CANCELS.
- **Family statement** (only once all 4 models are in): SUPPORTED iff >= 3 of 4 models support it; NOT SUPPORTED iff <= 1;
  MIXED otherwise. With only the CPU models in, results are reported per model and labelled interim.
- **Descriptive (no verdict).** Row-space T2 / Rel / nu; global-mode T2 (includes foreground-class structure); clean-view
  checks (top-1 agreement of X and S, null energy share of the edit); a numerical world-B check (T2 after rescaling the null
  edit by 0.1 and 10).

## 4. Synthetic stability study and kill rules

- **Study** (`synth_r7.py`, seed 100740, 40 repetitions per cell): N = 900 in 9 groups of 100, k = 1048, 4 views, a 768-d
  reference with 30 fine classes per group; noise-free edits with T* in {0.1, 0.2, 0.4}; per-view noise with nu in
  {0.5, 1, 2, 4, 8, 16, 32}, isotropic, low-rank (power-law j^-1.5 spectrum) or heavy-tailed low-rank. Stable iff <= 5 %
  undefined, |median - target| <= 20 % and SD <= 25 % of the target.
- **Result (committed before any real feature).** T2 is never undefined; its median / target is 0.96-1.04 (isotropic),
  0.75-1.11 (low-rank) and 0.47-1.03 (heavy-tailed) for nu <= 16, and its spread grows with nu: stable up to nu = 2 for
  T* = 0.2 under low-rank noise (4 for T* = 0.4, 1 for T* = 0.1; 16-32 under isotropic noise; 1-4 heavy-tailed). T is
  undefined in 20-58 % of draws from nu = 2 under low-rank or heavy-tailed noise and is stable only up to nu = 0.5-1 there.
  Over each stable range max / min across nu is <= 1.10 for both, while T_naive falls by a factor 2.5-3 from nu = 0.5 to 32
  under low-rank and heavy-tailed noise (only 1.04-1.15 under isotropic noise, which spreads over all 1048 dimensions).
- **K0 (synthetic kill).** If T2's stable range under low-rank noise at T* = 0.2 did not reach nu = 2, stop before real data.
  It reaches nu = 2: no kill. (T alone would have been killed; it is kept as a descriptive secondary.)
- **K1 (no signal).** If no CPU model is H1 REPRODUCIBLE, stop after the CPU models and report; no H2/H3 verdicts, and the
  GPU job is not requested.
- **No other kill.** Every outcome, including NOT ESTIMABLE, is reported.

## 5. Reporting

Per-model aggregates to `r7/results/<model>.json` (no per-image values), the synthetic study to
`r7/results/synthetic_stability.json`, the unit-test log to `r7/results/unit_tests_prereg.log`, and a summary in
`r7/README.md`. Features and images stay private (`_ext/`). Any deviation from this file is listed with its reason.
The paper is not changed in this round.
