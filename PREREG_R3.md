# IC014 round-3 pre-registration (committed before any round-3 number is computed)

Date: 2026-11-02 (IST). Responds to two reviews of commit `25ba426`: Internal reviewer round 2 (fixes N1–N11) and an independent fresh review (PR2, points W1–W11, missing experiments 1–9, fixes 1–13). Plan and point-by-point mapping: `ROUND3_PLAN.md`. Target venue: TMLR.
Where a reviewer gave a pass/fail criterion we adopt it as written; additions are marked (ours). Code goes in `r3/`, outputs in `r3/results/`. CPU only.

**Disclosure of values already known when this was written.** The reviewers reported, from committed files: τ(ρ)=τ(AS_SING)=0.821; τ(ρ, AS_SING)=0.69; paired bootstrap CI of τ(ρ)−τ(AS_SING) [−0.36, 0.29]; partial Kendall τ 0.61 both ways; partial Spearman 0.63 (AS | ρ) and 0.72 (ρ | AS); like-for-like τ(IS/|AS|, SING ratio)=0.744; the fix-13 G ratios and the energy-law predictions 0.78–0.91; the fix-14 shuffled-pair floors (0.676/0.633/0.615/0.608, retrieval ≤0.2 %). We also know from round 2 that the first-order normaliser AS/√ρ gives τ=+0.26 against SING on all 13 models; experiment A below is the exact (not first-order) version of that normalisation, so its outcome is partly anticipated. Everything else below (the intervention itself at fixed ρ₀, PC and random-direction variants, CLIP-space energy, translator-gain control, leak-free surrogates, retrained translators, the â-benchmarked PC bands, BCa CIs, the --fast holdout) has not been computed.

## Common definitions
- Data: the 13 SING models on the fixed 10k ImageNet-val subset (`scripts/subset_files_10k.txt`), SING `e47b37d` released translators T(f)=Af+b, head W, null basis V_n from the SVD of W at absolute tolerance 1e-5 (as `r2/e2_controls.py`), Π = V_nV_nᵀ, ρ = E‖Πf‖²/E‖f‖². AS target = class text (normalised mean of SING's 6 prompts) unless stated. Per-image AS = |θ(Tf, t) − θ(T(f−r), t)| for a removed component r; the score is the image mean.
- SING reference = digitised Supp. Fig. 12b IS/AS bars (`exp0/fig6b_digitized.csv`); leakage order = −ratio. τ = Kendall τ-b of a leakage score against the leakage order.
- Sets: **all 13**; **E12** = the models with ρ ≥ ρ₀ (where an experiment needs it; at ρ₀=0.005 we expect all but DenseNet, decided from the committed `fix2_models.csv` ρ values); **S11** = the pre-declared round-2 screened set (excludes DenseNet and BiT).
- Model bootstrap: 2000× resampling of models, seed 0, resamples with < 4 distinct models skipped; report percentile and BCa (jackknife over models) CIs. Paired comparisons use the same resamples for both scores. Permutation p: 20000 random permutations of the score, seed 1, one-sided for τ ≥ observed.

## A. Equal null-energy intervention (PR2 missing 1, 2; W3, W4) — `r3/equal_energy.py`
- **A1 (primary).** Remove βΠf with β = √(ρ₀/ρ), ρ₀ = 0.005, so every model loses the same fraction of feature energy from its own null space; score AS_eq = mean |AS|. Sensitivity A1′: ρ₀ = 0.001.
- **A2.** Remove the top-variance null PCs (uncentred second moment of V_nᵀf), whole PCs in descending order with the last one scaled, to energy ρ₀·E‖f‖² (the `energy_match` rule of round 2).
- **A3.** Same energy, removed from a Haar-random rotation of the null basis (20 draws, seed 80000+j), mean over draws.
- **CLIP-space energy.** κ = E‖AΠf‖² / E‖Tf‖² and translator null gain γ = κ/ρ, per model; τ vs SING for ρ, κ, γ; paired bootstrap τ(κ)−τ(ρ).
- **A4 (translator-gain control).** Remove β′Πf with β′²κ = κ₀, κ₀ = min κ over E12 (so β′ ≤ 1); every model then loses the same fraction of CLIP-space energy.
- **Outcome (A1 at ρ₀ = 0.005, evaluated on E12 and on S11):**
  - **ENERGY-ACCOUNTS:** point τ(AS_eq) < 0.3 and the paired CI of τ(AS_SING) − τ(AS_eq) has lower bound > 0, on both sets.
  - **KILL-R3:** point τ(AS_eq) ≥ 0.6 on both sets. AS still matches SING's order after energy is equalised, so energy does not account for the ranking. The energy-ranking headline is dropped; we report this and stop before rewriting the paper.
  - **PARTIAL:** anything else. The headline may say only that SING's ranking is indistinguishable from an energy ranking, and must state that equalising energy weakens but does not remove the agreement (with the numbers).
  - If A1 does not give KILL-R3 but A2, A3 or A1′ gives τ ≥ 0.6 on both sets, the abstract must say that the result depends on which null directions carry the matched energy.
- **Translator gain (ours, reading rule).** If τ(A4) < 0.3 on both sets, the paper may say that AS at matched CLIP-space energy does not reproduce SING's order. If the paired CI of τ(κ)−τ(ρ) excludes 0, the paper names the better-agreeing variable instead of ρ.

## B. Leak-free null and planted leakage (N4; PR2 W4) — `r3/leakfree.py`
- **Leak-free surrogate.** Replace each image's null coordinates c_i = V_nᵀf_i by c̃_i ~ N(μ, Σ), the mean and covariance of c over the 10k images, drawn independently of the image (5 draws, seed 70000+r). The row-space part, and so every logit, is unchanged; c̃ carries no image-specific (and so no class-specific) content.
- **B-core (N4).** Models ResNet50, ResNeXt101, DINOv2-L (the screened null-1048 models). Concepts = the fix-10 holdout {water, green, thin, knitted}; target = concept text t_c; plant direction u_c = normalised null projection of A⁺t_c and concept scores s_i = standardised CLIP concept z-scores, exactly as the fix-10 generator (`e9/analyze_e9.py`), σ = √E‖f‖² of the original features. Features: f′ = f_row + λc̃ (in null coordinates) + ασs_iu_c, with λ set so the leak-free part has null energy fraction ρ_t ∈ {0.01, 0.05, 0.2, 0.45}, and α ∈ {0, 0.1, 0.3}. AS = mean |θ(Tf′, t_c) − θ(T(f′ − Πf′), t_c)|, averaged over draws and concepts.
- **N4 pass (reviewer, applied to each model):** at α = 0, AS increases with ρ_t with Spearman ≥ 0.8 over the 4 levels, and the OLS slope of log AS on log ρ_t (4 levels × 4 concepts) has a 1000× image-bootstrap 95 % CI (seed 0) excluding 0. **Pass requires all 3 models.** If it passes, the paper may say that AS responds to null energy that carries no image-specific content (a non-semantic path), and "confound" may be used only in that qualified sense. Otherwise the paper says "mediator or confound, undetermined".
- Also reported: AS(α)/AS(0) at ρ_t = 0.2 and at each model's own ρ (sensitivity to planted leakage at fixed energy).
- **B-rank (descriptive, ours).** On all 13 models with class text: AS_surr = AS when each model's null content is replaced by its own leak-free surrogate at its own ρ (5 draws). Report τ(AS_surr, SING), Spearman(AS_surr, AS_SING), and the excess X = AS_SING / AS_surr with its τ vs SING. Reading rule: if τ(AS_surr) ≥ 0.6 on both all 13 and S11, the paper states that leak-free null content with each model's own energy and covariance reproduces SING's order. No claim from X unless its τ CI lies entirely below 0 or entirely above 0.6 on both sets.

## C. Fix-13 re-benchmark against the fitted energy law (N2; PR2 W2, fix 2) — `r3/pc_bands_r3.py`
- Same bands, seeds and image bootstrap as fix 13 (top-variance null PCs vs 20 random null sub-subspaces, d ∈ {24, 100}, 1000×, seed 0). Statistic H = (AS_top/ρ_top^â) / (AS_rand/ρ_rand^â), â = 0.431 (E1 pooled fit), with sensitivity at â ∈ {0.388, 0.485} (E1 cell-bootstrap CI ends). Equivalent to the observed G ratio divided by the energy-law prediction (ρ_top/ρ_rand)^(â−0.5).
- Counted models: ResNet50, ResNeXt101, DINOv2-L. BiT is reported and not counted (degeneracy screen).
- Labels at d = 24 (d = 100 reported): CI covers 1 on ≥ 2/3 → "consistent with the energy law"; CI > 1 on ≥ 2/3 → "top PCs carry more AS than energy predicts"; CI < 1 on ≥ 2/3 → "top PCs carry less AS than energy predicts"; otherwise "mixed". A label is called robust only if it is the same at all three â.
- The round-2 label is reported verbatim ("top PCs carry less AS per unit energy", G statistic) with its screened count (2/3 at d = 24) and per-model d = 100 values.

## D. Fix-14 chance floor (N3) — `r3/fix14_floor.py`
- Same 1000 images and cached candidate features. Per candidate: mean cos(T(f), CLIP_img); shuffled-pair cosine (image permutation, seed 0); centred cosine (both sides mean-centred over the 1000 images); retrieval top-1 (argmax over the 1000 CLIP images).
- Pass (reviewer): every non-matching candidate has |cos − shuffled| < 0.01; the paper then says "the used checkpoint matches; the others are at chance", and states that the 84.0 % vs 85.85 % top-1 gap is unexplained. If a candidate is above its floor by ≥ 0.01, it is reported as partially matching.

## E. Corrected claim CIs and de-duplicated claims audit (N6, N7) — `r2/e2_claims_ci.py` (revised)
- Same effects, bootstrap (1000×, seed 0, images within class) and verdict rules as fix 8, with **BCa** intervals (acceleration from a delete-one-image jackknife over the 10k + 200 images). Percentile CIs are kept in the CSV for comparison. The script asserts, for every effect, that the point estimate lies inside its BCa CI; any violation is written as a flag and reported.
- **N7 pass:** no CI in `fix8_claims_ci.csv` excludes its point estimate.
- **N6:** C1 and C11 are one computation (the Table-1 pool equals the Fig.-3 pool) and become one row "C1/C11"; C3 and the C6 class-mean difference are one quantity and become one row (C6 keeps only the class-SD part). Under AS_fair, C1 and C4 coincide numerically because the same model attains the minimum; this is footnoted and counted once. Tallies are recounted. **Pass:** no two table rows share a computation.

## F. Paired statistics on n = 13 (N1; PR2 W1, W5, W10, fix 1, 5, 6) — `r3/paired_stats.py`
- Paired bootstrap of τ(ρ) − τ(AS_SING) (percentile and BCa); τ(ρ, AS_SING); partial Kendall τ both ways; partial Spearman both ways (rank residualisation); permutation p for τ(ρ) and τ(AS_SING); log-linear residual τ both ways; τ under the DenseNet/DinoViT near-tie swap.
- Like-for-like reproduction: τ(our IS/|AS|, SING's IS/AS) and the per-model ratio SING/ours.
- **Reading rule (N1, ours):** the paper may say ρ and AS_SING are "indistinguishable" predictors of SING's order if the paired CI covers 0; it may not use "explains" for ρ in the title, abstract or conclusion in any case. Both partial values and the paired CI are printed in the main text.

## G. Retrained translators (PR2 TMLR item 2, missing 3; ours) — `r3/retrain_translators.py`
- For each of the 13 models, fit a ridge translator f → CLIP image embedding (`clip_img.npz`, same 10k images) on a fixed train half (permutation seed 0, first 5000 images), intercept by centring, λ from the grid 10^{−3..3}·tr(XᵀX)/(n·m) chosen by mean cosine on a 4000/1000 split of the train half, then refit on all 5000. Evaluate on the other 5000 images: fidelity (mean cosine), AS_SING with the retrained translator (class text), and A1 at ρ₀ = 0.005.
- Descriptive. Reading rule: if τ(AS_SING-retrained, SING) < 0.3 on both all 13 and S11, the paper says SING's order depends on the released translators; if ≥ 0.6 on both, it says the order survives retraining. τ(AS_SING-retrained, AS_SING) is reported.

## H. A second null-space score (PR2 missing 8; ours) — in `r3/equal_energy.py`
- NuSA-style null norm: mean_i ‖Πf_i‖/‖f_i‖ per model. τ vs SING and vs ρ, descriptive. Expected to track √ρ; reported to show the energy control generalises to a norm-based score, not as a new finding.

## N10. CPU-bounded holdout — `r2/e9_holdout.py --fast`
- Fast mode, fixed now: random full-space draws for AS_fair NDRAW = 5 (seeds 10000–10004, the first 5 of the full run), nested sub-null bases NSUB = 2 (seeds 20000, 20001), candidates M1, M2, S10, M5, S4, AS_SING, AS_fair (S9, the ridge + KSG candidate, omitted), 50 trials at seed 1, linear and nonlinear plants, same models and concepts. Separate cache.
- **Pass (reviewer):** wall time < 30 min on 8 CPU cores; Kendall means of M1, AS_SING and AS_fair (linear and nonlinear) and the two M1 − AS_fair paired mean differences within ±0.03 of the committed full run (`r2/results/fix10_e9_holdout.json`). The full-run time is reported in the README. If the box is shared during the timed run, we say so.

## N8, N9, repo
- N8 pass (reviewer): `bash scripts/reproduce.sh && git diff --exit-code` succeeds in a scratch clone after deleting the generated `.tex` tables; `main.tex` `\input`s every numerical table.
- N9 pass (reviewer): `python -m venv v && v/bin/pip install -r requirements.txt` succeeds in a fresh venv; the file is project-only and includes timm, torchvision and transformers (PR2).
- Makefile: no machine-specific default feature path (a missing `IC014_FEATS` stops with an error). Private per-image outputs default to a git-ignored repo-local directory, never the feature store.

## Kill and stop rules
- **KILL-R3** (A above). If it fires we stop before rewriting the paper and report.
- Every outcome above is reported whether it passes or fails. No criterion, seed, threshold or definition in this file may change after the first round-3 number is computed; anything added later is labelled post hoc.
- If the results point to a change of direction beyond what the reading rules above allow, we stop and report rather than make the change.
