# IC014 round-4 second addendum (committed and pushed before any number below is computed)

Date: 2026-11-02 (IST). Adds items B1, B3 and B4 of an internal scope note to `PREREG_R4.md` and `PREREG_R4B.md`; neither is edited. Definitions, sets, bootstrap and the fixed headline as there. Code in `r4/`, aggregate outputs in `r4/results/`, CPU only, no per-image output committed.

**Values already known when this was written.**
- **(a) PREREG_R4B idea 1** has run. The outcome is ENERGY-CARRIES under both translator families:
  - released translators: τ(AS_cperm) = 0.69 / 0.64 / 0.64 (all 13 / E12 / S11), τ(Δ_align) = 0.64 / 0.64 / 0.64, median share 0.05 / 0.10 / 0.14;
  - ridge translators: τ(AS_cperm) = 0.74 / 0.70 / 0.67, τ(Δ_align) ≈ 0.1;
  - AS_perm ≤ AS_cperm on only 6 of 13 models.
- **(b) PREREG_R4B idea 2** (calibration) is running; some world-level τ values have been seen.
- **(c) Exploratory scratch analyses, post hoc, by a separate note** on the 13 models with released translators. This was the full decomposition seen on all 13. These numbers may appear in the paper only labelled exploratory:
  - the out-of-plane-only removal share of AS is 0.20–0.98, and τ(AS_oop, SING) = 0.72;
  - the closed-form first-order Gaussian AS_gauss has Spearman 0.995 with the Monte-Carlo AS_surr;
  - the signed in-plane shift has τ 0.18;
  - the per-PC gain slope over all null PCs predicts a = (1+s)/2 of 0.40–0.44 on VGG16, ResNet50 and ResNeXt101, and 0.29 on DINOv2-L, which is outside the E1 CI [0.388, 0.485].
- **(d)** Nothing in B1 has been computed on DINO ViT-B/16 or ViT-S/16, or on a synthetic model. B3 (null zero-shot accuracy) and B4 (IS against row energy) have not been computed on any model.

## B1. Geometric account of AS — `r4/geometry.py`
Notation for one image: Z = T(f), δ = AΠf (the translated null part SING removes) and class text t. δ splits into δ_in, the part in span{Z, t}, and δ_oop, the part orthogonal to it.
- AS_in and AS_oop are the AS obtained by removing only δ_in or only δ_oop.
- The out-of-plane closed form is ½·cotθ·‖δ_oop‖²/‖Z‖², in degrees.
- The first-order Gaussian closed form AS_gauss is E|N(gᵀAVμ, gᵀAVΣVᵀAᵀg)| with g = ∂θ/∂Z.
- **Gain slope s** is the OLS slope of log mean squared angular gain per unit null-PC coordinate against log PC variance (uncentred PCs), over all null PCs with variance > 1e-8 × the top PC. The predicted exponent is a_pred = (1+s)/2.
- **Fitted exponent a_fit** follows the E1 cell design, part (a) (`e1/analyze_e1.py`):
  - cells are, for d ∈ {24, 100, 280} (capped at d_null), the top-d uncentred null PCs, the bottom-d PCs, and 5 Haar-random d-subspaces (seed 1000·d + s);
  - each cell records ρ_cell and AS_cell, for removal of that subspace;
  - a_fit is the coefficient of log ρ in the OLS log AS = a·log ρ + b·log d + c, fitted per model.
- **G1 (13 models, released translators; exploratory, post hoc).** We report AS_in, AS_oop, the oop share, the oop closed form, AS_gauss, s, a_pred and a_fit, and the τ of AS_in, AS_oop and AS_gauss against SING and against ρ. This is not confirmatory. It is labelled exploratory in the paper, including DINOv2's miss.
- **G2 (confirmatory).**
  - Models: DINO ViT-B/16, DINO ViT-S/16 and timm ViT-B/16 AugReg.
  - Translators: the ridge translators of `PREREG_R4.md` DN, on the test half.
  - The AugReg head has an empty null space (already known), so G2 cannot be tested on it and the rules apply to the two DINO models. "≥ 2 of 3" therefore becomes "both".
  - Criteria:
    - (i) Spearman(AS_gauss, AS_surr) over ρ_t ∈ {0.01, 0.05, 0.2, 0.45} is ≥ 0.9. AS_surr is the Gaussian surrogate rescaled to ρ_t (2 draws, seed 70000+r); AS_gauss is the closed form at the same scale.
    - (ii) The out-of-plane-only removal gives a mean signed shift whose 1000× image-bootstrap 95 % CI (seed 0) is > 0.
    - (iii) Among images with ‖δ_oop‖/‖Z‖ < 0.5, |mean closed form − mean measured oop shift| / mean measured is ≤ 25 %.
    - (iv) The exponent: |a_fit − a_pred| ≤ 0.1.
  - **Pass:** (i)–(iii) hold on both DINO models. (iv) is reported as "exponent predicted" (both models within 0.1), "not predicted" (both off by more than 0.1) or "mixed".
  - **G2 kill:** (ii) fails on both models, or (i) is < 0.7 on both. If it fires, the decomposition section is dropped and the failure is reported.
- **G3 (synthetic calibration of a; part of the calibration family of `PREREG_R4B` idea 2).**
  - Setup:
    - Gaussian null coordinates, d_null = 300, with variances λ_k ∝ k⁻¹;
    - N = 5000 images and 20 classes;
    - a 64-d CLIP-like space with class texts drawn at random;
    - a row-space part Z_row with cos(Z_row, t_y) ≈ 0.3;
    - a translator null block whose column k has norm ∝ λ_k^{s/2} (random unit directions, seed 98000), scaled so that the mean ‖δ‖/‖Z_row‖ = 0.3;
    - the null energy fraction is 0.3.
  - Same cell design and fit as G2. s ∈ {−0.4, −0.2, 0, 0.2}.
  - **Pass:** a_fit is within ±0.05 of (1+s)/2 for all 4 values. **Fail:** off by more than 0.1 for 2 or more. Otherwise partial.
  - The world-C/M exponents of idea 2 are reported next to it and are not duplicated.
- **Reading.**
  - If G2 passes, the paper may present the decomposition as a mechanism by which AS responds to null energy without regard to the class-text direction: the out-of-plane part is content-blind with respect to the class text. It must not be presented as "no semantics".
  - If G3 fails, the paper does not use a = (1+s)/2.

## B3. Energy-free score: null zero-shot accuracy — `r4/nzs.py`
- **NZS.** Classify each image by argmax_c cos(AΠf, t_c) over the 1000 class texts, using the null part only and no bias. We report top-1 and top-5.
  - Control: the same with the null coordinates permuted across images (seed 97000).
  - Score: NZS = top-1 − control top-1.
  - Translators: released (13 models); ridge for the DINO models (descriptive).
- **Validity gate (applied first, before any cross-model number is looked at):**
  - (a) **Rescaling invariance.** NZS changes by ≤ 5 % under Πf → βΠf for β ∈ {0.1, 1, 10}, on the 3 base models.
  - (b) **Planted content at fixed energy.** On each base (ResNet50, ResNeXt101, DINOv2-L), the null coordinates are replaced by the round-3 Gaussian surrogate (seed 70000) plus a planted share p of the null energy along u_y = normalise(AVᵀt_y), the null direction that maps most onto the class text:
    - c′ = √(1−p)·c̃ + √p·√E_n·u_y, rescaled to the model's own null energy;
    - p ∈ {0, 0.01, 0.03, 0.1, 0.3};
    - criterion: τ(NZS, p) ≥ 0.6 on all 3 bases.
  - The same gate is reported for NCM_null (idea 5), descriptively.
- If NZS fails the gate, it is dropped. The backups follow in order:
  - a whitened linear probe on Πf (5-fold, 1000-way logistic, fixed top-24 null PCs whitened, against a shuffled-label control);
  - then a LEACE-style erasure gap.
  - Each is gated the same way. If none passes, the paper says "no validated energy-free score" and proposes none.
- **For the score that passes the gate, on E12 and S11:**
  - **Energy test.** "Energy-free" if |τ(score, ρ)| < 0.3 on both sets.
  - **SING test**, reported both ways:
    - "SING's order survives an energy-free score" if τ(score, SING) ≥ 0.6 on both;
    - "SING's order is not recovered once energy is removed" if τ < 0.3 on both;
    - "inconclusive" otherwise.
  - **Stop:** if either decisive SING outcome occurs, we stop before rewriting and report to the owner. Both would move the headline away from "cannot separate".

## B4. SING's own IS against row-space energy — `r4/is_energy.py`
- IS is our measured invariant-space shift (`fix2_models.csv`). We report τ(IS, 1−ρ) and τ(IS, κ_row), where κ_row = E‖A R f‖²/E‖Tf‖² with R the row-space projector, on all 13, E12 and S11.
- IS counts as an energy proxy if |τ(IS, 1−ρ)| ≥ 0.6 on all three sets. Descriptive.

## Not run
- B2 (the ridge-shrinkage sweep) is not run unless time remains. If it runs, it is labelled post hoc.
- Not run at all: the OOD-score audit, the Rezaei score and OSF registration.

## Stop rules
- The G2 kill, and the B3 stop above.
- Any result that would change the headline beyond "the data cannot separate energy from content" stops the round before paper rewriting.
- Everything is reported, pass or fail.
