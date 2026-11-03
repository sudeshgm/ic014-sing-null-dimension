# IC014 round 8 pre-registration: a target-first T1 score, θ_sem

Date: 2026-11-03 (SGT). Frozen and pushed **before** any θ_sem, θ_cls, J_inn, shadow/innovation, self-duality or head-swap
number was computed on a real model. Adds round 8 to `PREREG_R2`–`PREREG_R7`; none of those is edited. Plan: `r8/PLAN.md`.
Code: `r8/core_r8.py` (estimators, worlds, descriptives), `r8/run_r8.py` (the registered run), `r8/k0_r8.py` (kill rule K0,
synthetic only), `r8/synth_r8.py` (synthetic generator), `r8/tests/test_r8.py` (unit tests, synthetic only).
Source of the score: an outside review we commissioned ("ideas K", 3 Nov 2026), whose synthetic Appendix B we replicated.

Defined as in `PREREG_R4.md`–`PREREG_R6.md`: model sets (all 13; E12 = ρ ≥ 0.005; S11 = E12 minus densenet and bitresnet);
SING's order (digitised IS/AS ratio, same orientation as rounds 4–6: `lk` = −ratio); ρ; null basis (absolute singular-value
tolerance 1e-5); `tau_ci`, `partial_ci` (B = 2000, seed 0); calibration bases ResNet50, DINOv2-L, ResNeXt101; the round-5
residual permutation (seed 99000) used to build worlds (`r4/calibration.py` convention).

## 0. Target (stated before anything else)

- **T1 (primary): amount of text-predictable class signal in ker W, on a row-space yardstick.** The truth T1 must
  rise in worlds M/D (content up), stay flat in world C′ (null noise up, class content fixed against the row space), stay flat under a
  constant null offset, go to 0 under class derangement and on content-free surrogates, and be **equivariant** (not invariant) in world B:
  θ(β) = β²S(Z)/(β²S(Z)+S(S)).
- **T2 (secondary, descriptive, not gated): information discarded by the head** (Fisher information of the innovation).
- **Not targeted:** T3 (arrangement; SO_null/XCC, round 5).
- **Old gates.** In the round-4 world C (L fixed per unit of total energy) the T1 truth rises 1/(1−ρ) = 1.81× across E12. World C is
  computed and reported for θ_sem and θ_cls but **does not gate** round 8. Kendall τ over synthetic world rows is not a gate
  (max/min and slope CIs only).

## 1. What was known when this was written (plain disclosure)

- **θ_sem has been seen on synthetic data only.** That covers the ideas-K Appendix B simulator re-run by us (floor −0.0001..0.0000, C′ max/min
  1.01, old C 1.64, offset 1.00, derangement 0.0000 ± 0.0003, plants +0.095 / −0.0006, world B closed form) and our own synthetic
  generator (`synth_r8.py`, unit tests). The pipeline `run_r8.py` was also smoke-tested end to end on *fake* features
  (random matrices with the real file layout, B = 40), only to check that the code runs. No θ_sem, θ_cls, J_inn,
  shadow/innovation, self-duality or head-swap number exists for any real model.
- **Post hoc, on the 13 real models (3 Nov 2026, before this text), the ρ decomposition** is known: ρ = ρ_offset + ρ_between + ρ_within.
  The offset share of ρ is DeiT3-L 92%, ResNet50 85%, DenseNet121 83%, ResNeXt101 33%, and ≤ 9% for all others.
- **Post hoc, on the 13 real models, the offset and centring results** are known:
  - τ with SING's order on all 13 models: uncentred ρ 0.82, **centred ρ (ρ − ρ_offset) 0.56** (E12 0.48, S11 0.49).
  - Content-free surrogates rescaled to the *centred* null energy still give AS_surr τ 0.67 / 0.61 / 0.64 (all13 / E12 / S11).
  - The constant offset alone accounts for 83% of AS for both DeiT3-L and ResNet50 (EVA02-L 55%, DenseNet121 87%, other models 5–25%).
  - By centred ρ, DeiT3-L (SING rank 1) moves from 3rd to 8th and ResNet50 from 7th to 9th.
  
  These results motivate a centred, content-based score. They were obtained before the bars below were fixed. No real-model θ_sem
  was used to set any bar.
- **Known from earlier rounds:**
  - AS_surr τ with SING 0.69; ρ τ 0.82; AS τ 0.82.
  - Round-4 constant-mean surrogate AS (DeiT3-L 4.22° vs AS 5.05°; ResNet50 1.39° vs 1.68°).
  - Round 5's η² per model; M1 τ 0.667; round-7 results.
- **Lessons carried from round 7 (binding here):**
  - **The resampling must never pair an image (or a class) with itself.** θ_sem's train and test folds never share a class,
    including inside every resampling draw: classes are resampled *within* each fold. This is unit-tested.
  - Ê_b pairs images 0–4 with 5–9 of the same class, never an image with itself.
  - **θ_sem's CI uses a class-level bootstrap or half-sampling, chosen by synthetic coverage (K0)** before any real run.

## 2. Score definitions (`core_r8.py`)

Per model: F (10 000 × d, 10 images per class), y, W. Vn = null basis of W (right singular vectors with singular value ≤ 1e-5);
Vr = the complementary right singular vectors (row basis). Z = F·Vn, S = F·Vr (uncentred; centring happens inside the score).

- **Texts.** SING's prompt-averaged CLIP ViT-L/14 class text embeddings (each prompt unit-normalised, averaged, normalised again, as in
  `r4/retrained_a4.text_emb`), centred across the 1000 classes: T (1000 × 768).
- **Class folds.** A random 500/500 class split, seed **100800** (primary). Sensitivity: a second split, seed 100801 (reported, not
  gating).
- **Class means.** m_c is the mean over all 10 images of class c. Within each fold, centre the class means by the fold mean.
- **Text map (train fold only).** Ĝ = M_trᵀ T_tr (T_trᵀT_tr + λI)⁻¹ with **λ = tr(T_trᵀT_tr)/768**, the mean eigenvalue. λ is set from the texts only
  and never tuned on features.
- **Test-fold score.** S_te(X) = Σ_{c∈te} ⟨Ĝ t_c, m_c − m̄_te⟩. Swap the folds; S(X) = (S_A→B + S_B→A)/(|A|+|B|). It is computed in the
  equivalent kernel form Σ H ⊙ G_c[tr, te], with H = (T_trT_trᵀ+λI)⁻¹T_trT_teᵀ and G_c the class-mean Gram double-centred within each fold.
  The tests check this against the direct formula.
- **θ_sem = S(Z) / (S(Z) + S(S)).** It is not estimable if S(Z)+S(S) ≤ 0. S(Z) is always reported with its CI (it can be negative).
- **Ê_b(X).** Split-half (images 0–4 vs 5–9 in file order), centred, cross-fitted between-class energy:
  mean_c ⟨m_c^A − m̄^A, m_c^B − m̄^B⟩.
- **θ_cls (descriptive):** Ê_b(Z)/(Ê_b(Z)+Ê_b(S)), with a class bootstrap CI (B = 2000, seed 100806).
- **Factorisation (descriptive):** S(Z) = [S(Z)/Ê_b(Z)] × Ê_b(Z).
- **Uncertainty.** Classes are resampled within each fold (B = 2000 draws, seed **100806**), and Ĝ is refit on the resampled train fold.
  Two methods are possible:
  - **'boot':** with replacement, percentile 95% CI.
  - **'half':** half of each fold without replacement; basic 95% CI θ̂ − q(θ*−θ̂). Without-replacement half samples have
    Var(θ* − θ̂) ≈ Var(θ̂), so no rescaling is needed and SE = SD of the draws.
  
  Which method is used is decided by K0 (§3). The SE is the resampling SD on the full-sample scale.

## 3. Gates for θ_sem (all must pass before the confirmatory τ is read)

Worlds are built on the three calibration bases with the round-5 residual permutation (seed 99000), with 12 rows at the E12 ρ targets
(the 12 real ρ values ≥ 0.005). Write z′ = a·μ_y + b·R, where μ is the uncentred class means of Z and R the permuted residuals; Er is the row energy and
h = ρ_min/2 as in round 4.
**World C′:** a² = h·(Er/(1−ρ_min))/Mm is fixed, so class-mean null energy is fixed against Er, and b is set so that ρ reaches each target.

**K0 (kill rule, synthetic only, before any real feature).**
- Implementation equivalence (Gram form vs direct ridge formula, invariances, world B closed form, the no-shared-class property of the
  resampling, and the zero floor) is covered by `tests/test_r8.py`, which must pass.
- **Coverage** (`k0_r8.py`): `synth_r8.synth` with K = 1000, in two scenarios: 'signal' (sz 0.3, ss 1, sb 0.5, sw 1; truth = the mean point
  θ_sem over 200 datasets, seeds 100820000+i) and 'zero' (no null class content, null offset 3; truth 0). Coverage of the 95%
  CI is computed over 100 datasets (seeds 100830000+i) with B = 300 draws (seed 100806), for both methods.
- **Rule:** use 'boot' if its coverage is in [0.90, 0.99] in both scenarios, else 'half' if it is, else **K0 fires**: round 8 stops and
  reports why. The real runs then use B = 2000 with the chosen method.

1. **G-floor (zero floor).** Two content-free surrogates at each of the 12 E12 energies on each base (72 cells): Gaussian with the
   base's null mean and covariance (class means drawn exactly in distribution, seed 100807), and an across-image permutation of
   Z (seed 100808). Add one cell of each kind per model, for all 13 models, at the model's own energy (26 cells). **Pass:** the 95% CI includes 0 in
   ≥ 90% of the 98 cells **and** max |θ_sem| over all cells ≤ 0.1 × max θ_sem over the 13 real models.
2. **G-noise (world C′).** A row is estimable if S(Z)+S(S) > 0, θ_sem > 0 and SE ≤ 0.5·θ_sem. **Pass on each of the 3 bases:**
   ≥ 8 of 12 rows estimable, max/min of θ_sem over the estimable rows ≤ **1.5**, and the 95% CI (t, n−2) of the log–log slope of θ_sem on ρ
   includes 0.
3. **G-offset / G-scale / G-rotation (exact invariances), all 13 models.** The relative change must be ≤ **1e-8** (float64) under each of:
   - a constant null offset of norm 10 × RMS‖z‖ (direction seed 100802);
   - F scaled by 7.3;
   - a Haar-random orthogonal map of the null coordinates together with an independent one of the row coordinates (seed 100803).
   
   **Offset-only world:** add c·ô (ô = o/‖o‖, o the real null mean) to C′ row 0 until ρ reaches each E12 target. This must give
   max relative deviation ≤ 1e-8 from C′ row 0 on all 3 bases.
4. **G-derange.** Class means are permuted across classes with no fixed point (20 draws, seed 100804), on each base's real data and on its C′ row 6
   (6 cells). **Pass:** mean ± 1.96·sd/√20 includes 0 **and** |mean| ≤ 0.1 × max real θ_sem, in all 6 cells.
5. **G-S (semantic vs arbitrary plant).** On C′ row 6 of each base, plant each of two class codes at 0.3 of null energy, then rescale back
   to the original null energy:
   - a semantic class code T·Pᵀ (P Gaussian d_null × 768);
   - an arbitrary centred Gaussian class code (both seed 100805).
   
   Δ is θ_sem(planted) − θ_sem(unplanted), with a paired CI over the same resampling draws. **Pass on each base:** Δ_sem's CI
   lower bound > 0 **and** Δ_arb's CI includes 0.
6. **G-B (world B equivariance).** Z is rescaled to the 12 E12 ρ targets on each base. Because S is linear in the class Gram and λ does not depend
   on features, the closed form is exact. **Pass:** relative error ≤ 1e-8 on all 36 rows.
7. **G-tol.** θ_sem with null tolerances 1e-4 and 1e-6. **Pass:** Kendall τ with the 13-model order at 1e-5 is ≥ **0.8** for both.
8. **Estimability.** Per model, SE ≤ **0.1 ×** (max − min of θ_sem over the 13 models). **Pass:** ≥ 11 of 13 models.
   - Models that fail are dropped from the τ sets.
   - If E12 or S11 then has fewer than 10 models, the confirmatory outcome is INCONCLUSIVE.

If any of gates 1–8 fails, θ_sem is reported descriptively as **NOT VALIDATED for T1**, with every gate number. The confirmatory τ
is then reported but not read.

## 4. Confirmatory test and reading rules (fixed now)

τ-b of θ_sem with SING's order on E12 and S11 (estimable models only), `tau_ci` (B = 2000, seed 0, percentile and BCa). Exactly one of:
- **TRACKS:** τ ≥ 0.6 **and** percentile CI lower bound > 0, on both E12 and S11. Statement: "SING's order is reproduced by a score
  with no content-free energy path (validated zero floor)."
- **DOES NOT TRACK:** τ < 0.3 on both. Statement (AS_surr's τ, already known, is 0.69): "SING's order is carried by content-free
  null energy; a validated T1 score does not reproduce it."
- **INCONCLUSIVE:** anything else. Only the factorisation and the descriptives are discussed.

Reported alongside (descriptive, never verdicts):
- τ(θ_sem, ρ) and τ(θ_sem, ρ_between);
- the partial association of θ_sem with SING given the content-free energy ρ_offset + ρ_within (`partial_ci`);
- the composition × amount factorisation;
- the result with the second fold seed.

## 5. Secondary descriptives (no verdicts)

- **ρ components** are recomputed inside r8 (`rho_offset`, `rho_between`, `rho_within`, plug-in).
- **Shadow/innovation split.**
  - H = Σ̂_w,zs (Σ̂_w,ss + γI)⁻¹ from within-class deviations, with γ = 0.01 × the mean eigenvalue of Σ̂_w,ss.
  - θ_sem is computed for the shadow H·s and for the innovation u = z − H·s.
- **J (Fisher, descriptive):** J_z, J_s and J_inn. Crossnobis between-class matrix from image halves, Ledoit–Wolf within covariance,
  restricted to the 64 leading PCs fitted on the other class fold (`fisher_J`, PC seed 100809).
- **θ_cls, self-duality index and anisotropy.**
  - Self-duality index: mean cosine of W's rows with the centred class means of F, in full space and in the row space.
  - Σ_w anisotropy: participation ratio of the within-class spectrum of Z and of S.
- **Head-swap controls** on the 3 bases: NCM and shrinkage-LDA (sklearn Ledoit–Wolf) heads fit on images 0–4 and evaluated on 5–9.
  For each head (and the released head) report d_null, ρ, AS (released translator, round-3 AS definition) and θ_sem with CI.
- **Old worlds C, M and D** (round-4 definitions): θ_sem and θ_cls point values per row. Non-gating.
- **τ with SING** on E12 and S11 (all models, no estimability filter) for θ_cls, shadow, innovation, J_inn, the self-duality index and the
  fold-2 θ_sem.
- **Standard-addition AS is not run in round 8** (dropped from the draft for time; noted as not run).

## 6. Seeds
- **100800** class folds (primary); **100801** sensitivity folds.
- **100802** offset direction; **100803** rotations; **100804** derangements; **100805** plants.
- **100806** resampling draws (and the θ_cls bootstrap).
- **100807** Gaussian surrogate; **100808** permutation surrogate.
- **100809** J PC/LW draws; **100810** reserved (head-swap uses the fixed image halves 0–4 / 5–9).
- **K0:** 100820000+i (truth datasets) and 100830000+i (coverage datasets).
- **99000** world residual permutation (unchanged from round 5).

## 7. Compute
CPU only, on cached features. Estimated at about 3–4 CPU-hours on 8 cores, dominated by the 2000 resampling draws (one K×K solve per draw)
and the world rows. Per-model JSON is cached in `r8/results/permodel/`, so the run can resume.

## 8. Reporting
- Every gate number is reported, pass or fail.
- The old world C sits next to C′.
- The ρ decomposition and the offset results are marked post hoc.
- Deviations from this text are listed in `r8/README.md`.
- Round 8 does not change any earlier verdict.
