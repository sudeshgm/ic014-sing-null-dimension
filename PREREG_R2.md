# IC014 round-2 pre-registration (committed before any round-2 number is computed)

Date: 2026-11-01 (IST). Responds to `paper_workshop/reviews/review_20261101_v1.md`. Plan: `ROUND2_PLAN.md`.
Where the reviewer gave a pass/fail criterion we adopt it as written; additions are marked (ours).
Disclosure: the reviewer already reported several point values computed from committed files (normaliser τ's, AS_fair–ρ Spearman, E1 cell CI, symmetric E3 τ, DenseNet/BiT tolerance sensitivity). The E1 high/low/random cell means are also already in `e1/results/e1_cells.csv`. These criteria were therefore written knowing those values; the new information is the CIs, the two new control families, the tolerance sweep, the claim CIs, the synthetic test, the PC-band CIs and the E9 holdout.
Data: cached features for the 13 SING models on the fixed 10k ImageNet-val subset (10/class; identical `subset_files.txt` for E1/E2/E3), SING `e47b37d` translators, projectors (tol 1e-5 unless swept) and text embeddings. Target for AS = class text (normalised mean of SING's 6 prompts) unless stated. SING reference order = digitised Supp. Fig. 12b IS/AS bars (`exp0/fig6b_digitized.json`, higher = less leakage). τ = Kendall τ-b.

## Definitions
- **Family A (existing, AS_fair):** energy-matched Haar subspaces of the full feature space, seed 10000+r, 20 draws (`e2/analyze_e2.py`).
- **Family B (ours, low-gain row space):** W = UΣVᵀ at the tolerance in use; row-space directions sorted by σ ascending. K = the smallest K ≥ 24 such that the bottom-K row directions hold ≥ 1.5·E‖V_nᵀf‖² of feature energy, capped at rank(W) (cap recorded). Haar basis inside that K-dim span (seed 50000+r, 20 draws), energy-matched to the null exactly as in family A. We report max |Δlogit| / SD(logits).
- **Family C (ours, d-matched null):** numerator = mean over j = 0..19 of mean_x|AS| after removing a random 24-dim sub-subspace B_j of the null (Haar, seed 30000+j; for null dim 24 this is the full null). Denominator = mean over j of a family-B control energy-matched to E‖B_jᵀf‖² (one draw per j, seed 60000+j, K rule applied to that energy). Removing B_j changes no logit.
- **Normalisers (fix 3), 8 in total:** N1 AS/ρ^0.25, N2 AS/ρ^0.43, N3 AS/ρ^0.5, N4 AS/ρ, N5 AS/d_null, N6 AS_fair-A, N7 AS_fair-B, N8 AS_fair-C. AS = AS_SING (full-null class-text mean|AS|).
- **Model bootstrap:** resample the models with replacement, 2000×, seed 0, τ-b vs the SING order; resamples with < 4 distinct models are skipped. Leave-one-out = the range over the 13 single-model drops.

## Criteria
**Fix 2 (reviewer).** Claim "the size-corrected ranking disagrees with SING" only if the point τ vs SING is < 0.3 under ≥ 2 of the 3 families A, B, C. Otherwise drop the claim.

**Fix 3 (reviewer + ours).** D-weak: the τ 95% CI excludes AS_SING's τ (0.821) from below under ≥ 5 of the 8 normalisers. D-strong ("reverses", ours): the τ CI upper bound < 0 under ≥ 5 of 8.

**KILL-R (ours, headline kill).** If fix 2 fails, or D-weak fails, the reversal/disagreement claim is dropped from the title, abstract and conclusions. The paper is then reframed around the energy confound: null energy ρ alone reproduces SING's ranking. AS_fair is kept only as one diagnostic among several. If fix 2 and D-weak pass but D-strong fails, the paper may say "differs from" but never "reverses". Each criterion must hold on all 13 models and on the fix-5 screened set; if it holds on one only, that counts as failing. If KILL-R fires, we search no further normalisers or controls to rescue the claim. Fixes 4–13 still run, because they are reporting obligations that do not depend on the reversal claim.

**Fix 4 (reviewer).** Spearman(AS_fair-A, ρ_null) with a 2000× model-bootstrap CI, on all 13 and without DenseNet and BiT. "Removes the trend" may be written only if |Spearman| < 0.3 and the CI covers 0. Otherwise the Fig. 8 title and caption say what the data show.

**Fix 5 (reviewer).** Absolute tolerances 1e-6, 1e-5, 1e-4, 1e-3 and relative tolerances {1e-5, 1e-4, 1e-3}·σ_max. Scores: null dim, ρ, AS_SING, AS_fair-A. Degeneracy screen (pre-declared, at tol 1e-5): exclude a model if ρ_null < 1e-4, or if a retained direction with σ < 1e-3 carries > 5% of E‖f‖². Pass: τ between the rankings at adjacent absolute tolerances ≥ 0.9 for every score. We report this on all 13 and on the screened set, and every headline number is given both ways.

**Fix 6 (reviewer).** A from 2000× cell-level and 2000× model-cluster bootstraps (seed 0), plus leave-one-model-out and per-model a and b. Pass: the stated a CI is at least as wide as the cell bootstrap, and the text claims only "a ≈ 0.4–0.5".

**Fix 7 (reviewer).** Both scores are α=0-subtracted per (model, concept): AS_fair(α) − AS_fair(0). Kendall over 8 models per trial and concept, concept-averaged per trial, with a 10000× bootstrap CI (seed 1), as in E3. Both versions are reported.

**Fix 8 (reviewer).** Each testable claim gets a signed effect, positive = SING's direction, and a 95% bootstrap CI (1000×, seed 0, resampling images within class, the same indices for every model). Holds = CI > 0; reverses = CI < 0; not confirmed = CI covers 0. C9, a "no correlation" claim: holds if the CI covers 0, contradicted if not.
- C1/C4: min over the other models of S, minus S(DinoViT) (C1 over the 5 Fig. 3 models, C4 over all 13).
- C2: S(ResNeXt) − max over the other 4.
- C3: S(ResNet) − S(DinoViT).
- C5: S(ResNeXt) − second-largest over the other 4.
- C6: the class-mean difference and the class-SD difference, ResNet − DinoViT. Both must agree for holds/reverses.
- C7: the fraction of DinoViT classes with |AS| < 1°, minus 0.5. AS_SING only; N/A for AS_fair, which is unitless.
- C8: class percentile − 90, for each of porcupine and sports car.
- C10: jellyfish − camel.
- C11: min over the other Table-1 models, minus S(DinoViT).
- C12: τ vs the SING order.
- S ∈ {AS_SING, AS_fair-A}; lower S = less leakage. C8 and C10 are flagged as possible re-implementation discrepancies, because they fail under both scores. C13 and C14 are untestable. The abstract counts only CI-backed reversals.

**Fix 9 (reviewer + ours).** Synthetic data, pure numpy:
- m = 400, 100 classes (null dim 300), power-law feature spectrum, random 64×400 translator, n = 10000.
- Linear plant at α ∈ {0.1, 1}: (i) u in the full null; (ii) u confined to a fixed 24-dim null sub-subspace.
- Subspaces: d ∈ {24, 50, 100, 200, 300}, 5 nested seeds.
- Oracle O(B) = α‖Bᵀu‖, the planted amplitude actually inside B, noise-free. We also compute M1.
- The E8/E9 rule is max/min over d ≤ 1.5. If O fails it in setting (i), the rule is declared ill-posed.
- Revised rule (ours, fixed now): max/min over d of score(d)/O(d) ≤ 1.5. That is, a score must track the planted signal present in B.
- Re-adjudication: E8 NCP and E9 candidates are re-judged under the revised rule from the saved DEV flatness values. O(d) = α·mean_j‖B_jᵀu_c‖ uses the same sub-null bases (seed 20000+j). This is post hoc: it cannot turn E8 or E9 into a PASS, and it is reported as exploratory.

**Fix 10 (reviewer; post hoc by construction).** The E9 holdout is evaluated on {water, green, thin, knitted}, 50 trials at seed 1, with linear and nonlinear plants. Candidates: M1, M2, S9, S10, M5, S4, AS_SING (raw), AS_fair-A (α=0-subtracted).
- E9 step-3 rules apply as written: τ ≥ 0.8, CI lower ≥ 0.7, flat on ≥ 3/4 at both α, nonlinear τ ≥ 0.6. We also apply the fix-9 revised flatness.
- Labelled post hoc: E9 stays KILL as pre-registered.
- M1 vs AS_fair: a paired bootstrap over trials (1000×, seed 0) of the τ difference.
- E2 under M1 (descriptive, reported regardless): M1 at α=0 on the full null, averaged over the 173 SING concepts, for the 13 models, τ vs the SING order.
- Compute guard: any part over 3 CPU-hours is skipped and reported as not run.

**Fix 13 (reviewer + ours).** Models: ResNet50, ResNeXt101, DINOv2-L, BiT (null dim 1048). Null PCs come from the uncentred second moment of V_nᵀf. Bands at d ∈ {24, 100}: top, middle (centred on PC 524), bottom, and random (20 Haar draws, seed 40000+j). G = AS/√ρ per band, with a 1000× image bootstrap of G_top/G_rand.
- Energy is a nuisance: the CI covers 1 on ≥ 3/4 models at d = 24.
- Energy is partly signal: CI lower > 1 on ≥ 3/4.
- Top PCs carry less AS per unit energy: CI upper < 1 on ≥ 3/4.
- Otherwise: mixed.
- BiT is flagged for translator conditioning.

**Fix 12 (reviewer).** Pass: a fresh clone plus the cached features regenerates Tables 1–4 and the figures in under 30 min on CPU. We also test a fresh clone with no features (from the committed CSVs only). Features are not redistributed (ImageNet terms), so the feature part of the criterion can only be met by users who re-extract them.

**Fix 14 (reviewer).** GPU/ImageNet job (spec in `gpu_jobs/`). Pass: the checkpoint we used scores highest on cos(T(f), CLIP) by ≥ 0.05. Reported either way.

No criterion, seed or definition above may change after the first round-2 number is computed. Anything extra is labelled post hoc.
