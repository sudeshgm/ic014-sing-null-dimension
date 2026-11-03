# IC014 round 5 pre-registration: class-level organisation of null content (committed and pushed before any confirmatory number is computed)

Date: 2026-11-03 (IST). Adds round 5 to `PREREG_R4.md`, `PREREG_R4B.md` and `PREREG_R4C.md`; none of them is edited. Definitions of the model sets (all 13, E12, S11), SING's order (the digitised IS/AS ratio, leakage = −ratio) and the model bootstrap (`tau_ci`, 2000 resamples, seed 0, percentile and BCa) are as there. Code: `r5/organisation.py` (CPU, numpy). Aggregate outputs only, at most per model, in `r5/results/`. No per-image output is committed.

## 0. What was known when this was written (plain disclosure)

- **Thresholds were chosen after seeing scratch numbers on a few models.** Two internal idea notes computed both scores on a few models, with no cross-model τ against SING's order or against ρ:
  - the SO-type score on ResNet50, DINOv2-L, VGG16, DenseNet121 and BiT-R50x1;
  - the XCC-type score on ResNet50, DINOv2-L and ResNeXt101.
  - Those notes also ran plants, worlds and derangements on these models.
  - The bars below (reliability floor 0.10, plant strength α = 0.3, drift ratio ≤ 1.5, self-term z ≥ 5, partial ≤ 0.05) were set with those numbers in view. For example, α = 0.3 replaced α = 0.1 because α = 0.1 raised the scratch score by only about 1 null SD on DINOv2-L and VGG16.
- **Smoke test of `r5/organisation.py` on ResNet50 (already seen in scratch).**
  - Under the round-4 class-mean derangement (§2 G2), the scratch definition of SO (Spearman of the raw crossnobis RDM) kept 0.10–0.13 of its real value 0.15 on ResNet50, while dropping to ≈0.01–0.06 on DINOv2-L and VGG16.
  - The cause: with in-sample class means, fold residuals sum to zero, so the crossnobis diagonal carries class-specific spread terms, which act as additive per-class (row/column) effects in the RDM. Exact class relabelling gives ≈0.
  - **Because of this, the primary SO_null is defined on the double-centred RDM** (§1). Under the derangement it gives |SO| ≤ 0.004 on these 3 models, against real values 0.31 / 0.21 / 0.31 (ResNet50 / DINOv2-L / VGG16).
  - The smoke test also gave ResNet50's XCC_24 (0.169) and its G1–G3 outcomes (all pass). No other model has been run with the final code.
- **The calibration fix of round 5** (`r4/calibration.py --resid perm`, residuals permuted across images, seed 99000) was started before this file was committed, and the world-level τ values of its first base (ResNet50) have been seen. It is a correction of the round-4 calibration, not a test of the new score.
- **Nothing in §3 (confirmatory) has been computed for any score of this round.**

## 1. Scores

All scores use the null coordinates z = V_nᵀ f, with SING's null basis at tolerance 1e-5, on the fixed 10k subset (10 images per class). They use no translator. The class texts are SING's prompt-averaged CLIP ImageNet class embeddings (`imagenet_text_embeddings_fp16.npz`, each prompt l2-normalised, averaged, re-normalised).

**Primary: SO_null (semantic organisation).**
1. **Folds.** Shuffle each class's images (seed 99200) and assign them round-robin to 4 folds.
2. **Cross-fitted class-mean distance** for folds (a, b): D_ab(c, c′) = (m_a(c) − m_a(c′)) · (m_b(c) − m_b(c′)), over the 499,500 class pairs.
3. **Double-centring.** Each RDM is double-centred (row and column means removed, grand mean added back) before ranking. So is the text RDM, 1 − cos(t_c, t_c′).
4. **Reliability.** rel = Spearman(D_01, D_23).
5. **Score.** SO_null = ½[Spearman(D_01, T) + Spearman(D_23, T)] / √rel.
6. **Estimable** iff rel ≥ 0.10 (G5).
7. **Permutation null.** 200 joint class permutations of D_01 and D_23 (seed 99300), in SO_null units. This gives the null SD used in G2 and G3.

**Secondary: XCC_24 (cross-fitted class-geometry congruence, fixed k = 24).**
1. **Halves.** First 5 / last 5 images of each class in file order (`estimators.halves`). Half class means are centred by their half's grand mean.
2. **Grams with the diagonal removed:**
   - cross Gram K_AB = ½(M_A M_Bᵀ + M_B M_Aᵀ);
   - self Grams K_A = M_A M_Aᵀ and K_B = M_B M_Bᵀ;
   - text Gram G = T_c T_cᵀ, with T_c the centred class texts.
3. **Congruence.** XCC = ⟨K_AB, G⟩ / √(⟨K_A, K_B⟩ ⟨G, G⟩). It is undefined if ⟨K_A, K_B⟩ ≤ 0.
4. **XCC_24** is the mean of XCC over 50 Haar 24-frames of the null coordinates (seeds 99100 + s).
5. **Estimable** iff the self-term z ≥ 5. z is computed on the full null: the self-term ⟨K_A, K_B⟩ is split over 10 class folds (seed 99600), and z = mean/SD·√10.
6. **Permutation null.** 200 class permutations of G (seed 99300), applied to the frame-averaged normalised K_AB.

**Reported alongside (descriptive):**
- full-null XCC;
- SO and XCC of the row-space coordinates;
- the cross-fitted centred class-mean share η²;
- ρ;
- F1-partial (§4).

## 2. Gates (applied to each score separately, on all 13 models unless stated)

- **G1, exact invariance.**
  - Requirement: |score(βz) − score(z)| ≤ 1e−10 for β ∈ {0.1, 10}, for SO_null and XCC_24.
  - The same holds under a random orthogonal map of the null coordinates (seed 99500) for SO_null and for the full-null XCC. XCC_24's Haar frames are rotation-invariant only in distribution.
  - **Kill the score on any violation.**
- **G2, class-mean derangement** (the round-4 construction: z′ = μ_π(y) + r with in-sample class means, π a derangement, 5 draws, seeds 94000 + r).
  - Pass if |mean derangement score| ≤ 0.10·|real score|, or ≤ 2·null SD/√5.
  - **Kill if it fails on more than 2 estimable models.**
- **G3, world S** (semantic vs arbitrary class-code plant, at fixed null energy).
  - Semantic code: centred class texts × a Gaussian 768 × d_null matrix. Arbitrary code: Gaussian 1000 × d_null. Seed 99400 for both.
  - Each code is scaled to the null energy. z′ = z + α·u_y, rescaled back to the original null energy.
  - Pass on a model if, at α = 0.3, the semantic plant raises the score by ≥ 2 null SDs and the arbitrary plant raises it by ≤ 1 null SD. α = 0.1 is reported.
  - **Kill unless it passes on ≥ 11 of 13.**
- **G4, corrected world C** (worlds of `r4/calibration.py --resid perm`, rebuilt in `r5/organisation.py` with the same construction and the same residual permutation, seed 99000).
  - Bases: ResNet50, DINOv2-L and ResNeXt101. The 12 E12 ρ values are the targets, with L = ½·min ρ.
  - Over the world-C rows where the score is estimable (G5 rule), with at least 3 such rows, max/min ≤ 1.5. Any value ≤ 0 counts as a fail.
  - A base with fewer than 3 estimable rows is NOT ESTIMABLE.
  - **Kill if it fails on ≥ 2 bases.** If fewer than 2 bases are testable, G4 is reported as not testable, which is not a kill.
  - "τ vs ρ < 0.3" is not used: a small monotone bias gives |τ| = 1.
  - Worlds B (whole null part rescaled to each ρ target; truth: exactly flat), M and D (truth for an organisation score: flat) are reported descriptively.
- **G5, estimability.**
  - SO_null needs rel ≥ 0.10 and XCC_24 needs z ≥ 5. Models below these floors are NA for that score.
  - There is no confirmatory test for a score if fewer than 9 of E12 or 8 of S11 are estimable.
- **Kill rule.**
  - **K1:** if SO_null is killed by G1–G4, round 5 stops after the gates. No confirmatory τ is computed for either score, and the gate results are reported as the outcome.
  - **K2:** if only XCC_24 is killed, it is dropped and SO_null continues.

## 3. Confirmatory test (only for scores alive after the gates)

- **Test.** Kendall τ_b(score, SING's leakage order) on the estimable models of E12 and S11, reported also on all 13, with `tau_ci` (2000 model resamples, seed 0, percentile and BCa).
- **Bars** (symmetric, point estimates; CIs reported):
  - **ORGANISATION-TRACKS** if τ ≥ 0.6 on both E12 and S11;
  - **ORGANISATION-DOES-NOT-TRACK** if τ < 0.3 on both, so a strongly negative τ is reported as "does not track", with the sign stated;
  - **INCONCLUSIVE** otherwise.
- **Which label is the headline.** SO_null's label is the headline. XCC_24's label is secondary.
- **Power.** With 11–13 models only |τ| of about 0.45 or more is distinguishable from 0. INCONCLUSIVE is an expected, acceptable outcome.
- **Descriptive only:**
  - τ(score, ρ) and τ(score, η²);
  - the partial Spearman with SING's order given ρ (`partial_ci`);
  - the row-space and full-null variants;
  - the two true-DINO models (values only).

## 4. Inheritance diagnostic (pre-registered statement)

- **F1-partial** is the full-null XCC with the model's own cross-fitted row-space class Gram projected out of both the null Gram and the text Gram.
- **INHERITED** ("null text congruence is inherited from the row geometry") is stated if F1-partial ≤ 0.05 on ≥ 75 % of the XCC-estimable models (13 models).
- **Plant check.** On the 3 calibration bases, planted text geometry at a share α = 0.05 of the class-mean energy (seed 99401) must raise F1-partial to ≥ 0.1. If it does not, the diagnostic is UNINFORMATIVE.
- Otherwise the statement is NOT INHERITED.

## 5. Reporting

Every gate result per model, any kill, and the confirmatory label with CIs go to `r5/results/`, the README and RESULTS, and the paper, whatever the outcome.
