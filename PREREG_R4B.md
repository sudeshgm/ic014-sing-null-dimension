# IC014 round-4 addendum pre-registration (committed and pushed before any addendum number is computed)

Date: 2026-11-02 (IST). This file adds five analyses (ideas 1, 2, 5, 3, 6 of an internal methods note) to `PREREG_R4.md`, which is not edited. Common definitions, sets, the model bootstrap and the fixed headline ("SING's ranking is indistinguishable from a null-energy ranking; the data cannot separate null energy from real null content") are as in `PREREG_R4.md` and `PREREG_R3.md`. Code goes in `r4/`, aggregate outputs in `r4/results/`, CPU only, no per-image output committed. Symmetric bars throughout: a score **tracks** SING at point τ ≥ 0.6, **does not reproduce** it at |τ| < 0.3; anything between is "intermediate". Partial association given ρ uses the rank-residual partial Spearman with its 2000× model-bootstrap percentile CI (`r4/common4.py`).

**Values already known when this was written (round-4 results computed under `PREREG_R4.md`).**
- S: the content-free AS_perm and the content excess D_perm both track (τ 0.69/0.64/0.64 and 0.62/0.67/0.71, all 13/E12/S11). The outcome is "cannot apportion". Surrogate τ spans 0.49–0.72; τ(AS_gauss, κ) = 0.92; KILL-R4-S did not fire.
- J: retrained A1 reaches the KILL-R3 level (τ 0.61 E12, 0.75 S11); released A1 gives ENERGY-ACCOUNTS. S11 without DinoViT/EVA02 gives 0.61. τ(A4_ret) = 0.52/0.49 (E12/S11).
- DN: KILL-DN did not fire. ρ = 0.189 (ViT-B/16) and 0.437 (ViT-S/16); AS_ret 0.514 / 0.652. Q-DN1 = SING-inconsistent and Q-DN2 = mixed (B/16 inside the PI, S/16 below it). The AugReg ViT-B/16 head has an empty null space.
- C6: passes under class resampling.
- Planted: α* > 0.3 for all three core models.

## 1. Class-mean permutation surrogate — `r4/class_perm.py`
- **Construction.**
  - Null coordinates are z_i = V_nᵀf_i = μ_{y_i} + r_i, where μ_c is the mean over the 10 images of class c and r_i is the within-class residual.
  - We draw a derangement π of the 1000 classes (5 draws, seed 94000+r) and set z′_i = μ_{π(y_i)} + r_i and f′_i = f_i − V_n z_i + V_n z′_i. The row-space part, and so every logit, is unchanged.
  - Every class has 10 images, so total null energy and the uncentred null second-moment matrix are preserved exactly.
  - Assertions: relative energy difference < 1e-9 and ‖ΔM‖_F/‖M‖_F < 1e-9. If either fails, the analysis stops.
- **Scores** (class text):
  - AS_real;
  - AS_cperm, the mean over draws;
  - Δ_align = AS_real − AS_cperm;
  - share s = Δ_align/AS_real.
  - We also report κ and its relative change under the surrogate.
- **Translators.**
  - Primary: SING released, on all 10k images.
  - Second run: the round-3 ridge translators, on the test half.
  - For each: τ against SING on all 13, E12 and S11, with percentile and BCa CIs, and the partial association given ρ.
- **Outcome (per translator family; E12 and S11):**
  - **CONTENT-CARRIES:** τ(Δ_align) ≥ 0.6 on both sets, and τ(AS_cperm) < 0.3 on both.
  - **ENERGY-CARRIES:** τ(AS_cperm) ≥ 0.6 on both, and the median s over the set is < 0.2 on both.
  - **UNAPPORTIONED:** anything else.
- **KILL-R4B-1:** if CONTENT-CARRIES fires under either translator family, we stop before any paper rewriting and report.
- **Sanity check (reported):** per model, whether AS_perm (round-4 S) ≤ AS_cperm.

## 2. Semi-synthetic calibration in worlds C, M and D — `r4/calibration.py`
- **Bases:** ResNet50, DINOv2-L and ResNeXt101, each with its released translator and class text. In each base we write z = μ_y + r, and rescale the class part by a and the residual by b. With M = Σ‖μ_y‖², Rr = Σ‖r‖² and E_row the row-space energy:
  - null energy E_n = a²M + b²Rr;
  - ρ = E_n/(E_row + E_n);
  - ground-truth class-aligned leakage L = a²M/(E_row + E_n).
- **Synthetic models:** 12 per world. Their target ρ values are the 12 E12 ρ values; let ρ_min be the smallest.
  - **World C (confound):** L is fixed at L_C = 0.5·ρ_min and b sets ρ. L is constant, so the content-blind test is against ρ: a content-diagnostic estimator must not follow ρ.
  - **World M (mediation):** the residual fraction b²Rr/(E_row + E_n) is fixed at 0.5·ρ_min and a sets ρ, so L = ρ − 0.5·ρ_min rises with ρ.
  - **World D (dissociation):** ρ is fixed at the median of the E12 ρ values, ρ_D. L takes the values ρ_D·(j + 0.5)/12, j = 0…11.
- **Estimators:**
  - ρ;
  - AS;
  - A1 (ρ₀ = 0.005);
  - AS_gauss (Gaussian surrogate of the synthetic z, 2 draws, seed 70000+r);
  - X = AS/AS_gauss;
  - Δ_align (idea 1, 2 derangement draws, seed 94000+r);
  - NCM_null (idea 5);
  - R (idea 3, k = 20, 5 random draws).
- **Rule:** an estimator is **content-diagnostic** iff, on all 3 bases, all three hold:
  - |τ(est, ρ)| ≤ 0.2 in World C;
  - τ(est, L) ≥ 0.8 in World M;
  - τ(est, L) ≥ 0.6 in World D.
  - An estimator that is constant within a world (τ undefined) fails that world.
- **Prediction stated in advance (not a criterion):** ρ and AS fail C, A1 fails M, and Δ_align passes. If Δ_align is not content-diagnostic, the paper makes no interpretive claim from idea 1 beyond its raw numbers.
- **Exponent calibration:** the OLS slope a of log AS on log ρ, with its 95 % CI, per base and world. It is set against â = 0.431 (the E1 pooled fit) and against the 13-model cross-model slope. The World-C slope is reported as the exponent from geometry alone. This is descriptive.

## 5. Translator-free null decodability — `r4/ncm_null.py`
- **Construction.**
  - Per model, u_i = Πf_i/‖Πf_i‖ in null coordinates, on the 10k images.
  - Leave-one-out nearest-class-mean classification over 1000 classes, by Euclidean distance to class means of u. The own-class mean excludes image i.
  - We report top-1 and top-5 accuracy.
  - **Control task:** the same with labels permuted across images (seed 95000).
  - **Score:** NCM_null = top-1 (real labels) − top-1 (control).
  - **NC1_null** = tr(Σ_W Σ_B⁺)/1000 on u, descriptive.
  - **Sanity check:** on the round-3 Gaussian surrogate (1 draw, seed 70000), top-1 must be within 0.5 points of the control.
  - DenseNet is excluded (‖Πf‖ is at float precision), so the sets are E12 and S11.
- **Rule:**
  - **NCM-CONTENT:** τ(NCM_null, SING) ≥ 0.6 on E12 and S11, and the partial-given-ρ CI excludes 0 on both. Independent content evidence. **KILL-R4B-5:** if it fires, we stop before rewriting and report.
  - **NCM-NULL:** |τ(NCM_null, SING)| < 0.3 on both sets. The energy-free score does not reproduce the order.
  - Otherwise intermediate.
  - τ(NCM_null, ρ) is reported in every case.

## 3. Class-discriminative versus energy-matched random null removal — `r4/selectivity.py`
- **Split halves by image within class:** half A is the first 5 images of each class in subset order, and half B is the other 5.
- **Removal directions:** U_k = the top-k eigenvectors of the between-class scatter of null coordinates, estimated on half A. k ∈ {5, 20, 50}, capped at ⌊d_null/2⌋. k = 20 is primary.
- **On half B:**
  - AS_lda = AS after removing V_n U_k U_kᵀ z, with removed energy e_lda;
  - AS_rand = AS after removing the same energy from a Haar-random rotation of the null basis (`equal_energy.energy_match`, 20 draws, seed 96000+j), averaged over draws;
  - selectivity R = AS_lda/AS_rand;
  - image-bootstrap CI (1000×, seed 0, resampling half-B images within class; the draw-mean AS_rand is fixed per image).
- **Within-model rule:** content sensitivity is shown if the CI of R excludes 1 on ≥ 9 of the 12 E12 models. We report the direction count.
- **Cross-model rule:**
  - **R-TRACKS:** τ(R, SING) ≥ 0.6 on E12 and S11. **KILL-R4B-3:** if it fires, we stop before rewriting and report.
  - **R-NULL:** |τ| < 0.3 on both. The order is not about discriminative null content.
  - Otherwise intermediate.
- This replaces the PC-band test as the semantic-direction control. The PC-band results stay in the appendix.

## 6. Out-of-sample prediction for true DINO ViT-B/16 and ViT-S/16 (folded into `r4/dino.py` results)
- The energy-law PI test (Q-DN2) and the rank test (Q-DN1) were registered in `PREREG_R4.md` and have run (see above).
- This addendum adds, under the ridge translators and the scripts of ideas 1 and 5:
  - Δ_align_ret and NCM_null for both DINO models;
  - a 95 % PI for each, from the OLS fit of the score on log ρ over the E12 models (Δ_align_ret on the test half; NCM_null on all 10k);
  - the rank of each DINO model among the 15 by ρ, AS_ret, Δ_align_ret and NCM_null.
- **Rule:** a score's cross-model law generalises iff both DINO models fall inside its PI. "Least leaking" holds for a score iff both DINO models rank in the lowest four of 15. Descriptive; no kill.

## Not run
- Ideas 4 (translator multiverse) and 7–9 are not run in this round.

## Stop rules
- KILL-R4B-1, -3 and -5 above, and any result that would change the headline beyond "the data cannot separate energy from content", stop the round before paper rewriting.
- Every outcome is reported.
- Nothing in this file changes after the first addendum number is computed.
