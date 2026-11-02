# IC014 round-4 pre-registration (committed and pushed before any round-4 number is computed)

Date: 2026-11-02 (IST). This round responds to two reviews of commit `1566633`: Internal reviewer round 3 (fixes 1–13) and internal reviewer 2 round 2 (W1–W8, missing experiments 1–8, requested changes 1–12). The point-by-point mapping is in `ROUND4_PLAN.md`. Code goes in `r4/` and outputs in `r4/results/`. Everything runs on CPU. The true-DINO features were extracted earlier on a GPU (job spec `gpu_jobs/r3_dino_vitb16.md`). They are private, ImageNet-derived, and never committed.
The headline for this round is fixed in advance: **SING's cross-model ranking is indistinguishable from a null-energy ranking, and the data cannot separate null energy from real null content.** The rules below decide what the paper may add to that headline, or must withdraw from it. `PREREG_R3.md` and earlier files are not edited. Definitions not restated here (data, null basis at absolute tolerance 1e-5, ρ, AS, the sets all 13 / E12 / S11, the model bootstrap with 2000 resamples at seed 0 giving percentile and BCa intervals, and the leakage order = −SING ratio) are as in `PREREG_R3.md`, "Common definitions".

**Values already known when this was written.** These come from the committed round-3 files or from the reviewers' own checks.
- A1 against the normaliser: τ(A1, AS/√ρ) = 0.79 (E12) and 0.82 (S11); Spearman 0.92 / 0.93.
- A1 influence:
  - leave-two-out τ(A1) without DinoViT and EVA02 = 0.38 (E12) and 0.61 (S11);
  - the single-model jackknife stays below 0.3.
- A1 across translators:
  - τ(A1 released, A1 retrained) = 0.06;
  - τ(A1 retrained, ρ) = 0.64 / 0.58 / 0.67 (all 13 / E12 / S11);
  - paired CI of τ(AS) − τ(A1 retrained): [−0.23, 0.63] (E12) and [−0.29, 0.50] (S11).
- Content-free and content-bearing scores (reviewers' re-implementation):
  - τ(AS_surr, κ) = 0.92;
  - X: τ 0.54 [0.14, 0.83] (all 13) and 0.60 [0.24, 0.91] (S11);
  - AS − AS_surr: τ 0.59 (all 13) and 0.67 (S11);
  - AS − AS_perm: τ 0.62 (all 13) and 0.71 (S11);
  - surrogate variants, τ on all 13: across-image permutation ≈ 0.69, within-class permutation ≈ 0.80, isotropic ≈ 0.46, mean-only ≈ 0.49, zero-mean Gaussian ≈ 0.72;
  - partial Spearman of X given ρ ≈ 0.73 (E12) and 0.79 (S11).
- Planted concepts: the B-core own-ρ planted ratios are in `r3/results/B_core.json`.
- Claims: C10 is percentile [−0.206, +0.001] and BCa [−0.210, −0.006]. The C6-SD (AS_fair) point lies outside both of its CIs.
- Fast holdout: N10 took 36.3 min, with ΔAS_fair (nl) = 0.057.
- True DINO models: GPU runner's extraction log reports top-1 accuracy of the official linear heads as 78.06 % (ViT-B/16) and 76.91 % (ViT-S/16) on the 50k images, and 84.54 % for the timm AugReg ViT-B/16. No other number on these features has been computed. In particular, no ρ, translator or AS value exists yet.

## S. Surrogate family and the content-free / content-bearing split (R1 fixes 3, 5; PR2 W3, missing 1, change 3) — `r4/surrogates.py`
- **Scope:** all 13 models, class text, SING's released translators. In each variant the null coordinates c_i = V_nᵀf_i are replaced by c̃_i and rescaled so that the model keeps its own null energy (as in `leakfree.rank`). The score is mean |θ(T(f_row + V_n c̃), t) − θ(T f_row, t)|.
- **Content-free variants:**
  - `gauss`: N(μ, Σ), 5 draws, seed 70000+r. This is the round-3 AS_surr, re-used from `B_rank_permodel.json`.
  - `zm`: N(0, Σ).
  - `iso`: N(0, I).
  - `mean`: μ for every image.
  - `perm`: the real c permuted across all images.
  - Except for `gauss`, each variant uses 3 draws with seed 92000+r; `mean` needs no draw.
- **Content-bearing variant:** `permc`, the real c permuted among the images of the same class (3 draws, seed 93000+r). It keeps class-level null content and breaks the link to the individual image.
- **Excess scores:**
  - X = AS/AS_gauss;
  - D_gauss = AS − AS_gauss;
  - D_perm = AS − AS_perm;
  - Q = AS_permc/AS_perm;
  - class share = (AS_permc − AS_perm)/(AS − AS_perm), per model, descriptive.
- **Statistics:** τ against SING on all 13, E12 and S11, with percentile and BCa CIs. Partial Spearman given ρ (rank residualisation), with a model-bootstrap percentile CI. τ(AS_gauss, κ).
- **Symmetric reading rule.** This is one bar, applied identically to a content-free score and to a content excess. A score **tracks SING** iff its point τ ≥ 0.6 on both all 13 and S11 and its percentile CI lower bound is > 0 on both.
  - The primary content-free score is AS_perm and the primary content excess is D_perm. The same rule is reported for AS_gauss and X/D_gauss.
  - Outcome:
    - Both primary scores track → the paper says that the content-free and the content-bearing parts of AS both track SING's order, and that the data cannot apportion the order between them.
    - Only AS_perm tracks → the paper says that content-free null energy alone tracks the order, and that the content excess does not reach the same bar.
    - Only D_perm tracks → the mirror statement.
    - Neither tracks → the paper says that neither part alone reaches the bar.
  - The round-3 bars were asymmetric: a point τ ≥ 0.6 for AS_surr, against a whole CI above 0.6 for X. The paper states this.
- **Surrogate dependence:** if τ over the five content-free variants (all 13) spans more than 0.15, the paper must say that AS_surr depends on the choice of surrogate. It must then quote the range.
- **KILL-R4-S:** if τ(AS_perm) < 0.3 on both all 13 and S11, the round-3 content-free result does not hold for a non-parametric surrogate. In that case we withdraw the "content-free path reproduces the order" sentence, stop, and report before rewriting.

## J. A1 as a per-energy statistic, its influence, and both translator families (R1 fixes 1, 2, 4; PR2 W1, W2, missing 2–4) — `r4/a1_stats.py` and `r4/retrained_a4.py`
- **Normaliser link** (from committed per-model files):
  - Spearman and τ of A1 against AS/√ρ and against AS/ρ^â (â = 0.431), on E12 and S11;
  - the per-model ratio A1/(β·AS).
- **Influence:**
  - the single-model jackknife range of τ(A1);
  - every leave-two-out τ(A1), with the minimum, maximum and the share of pairs whose removal gives ≥ 0.6;
  - the named pair {DinoViT, EVA02};
  - all of this on E12 and S11, for the released and the retrained A1.
- **Across translators:**
  - τ(A1 released, A1 retrained);
  - τ(A1 retrained, ρ);
  - partial Spearman of A1 retrained given ρ;
  - paired CIs of τ(AS_SING) − τ(A1 retrained) and of τ(AS_retrained) − τ(A1 retrained), on E12 and S11.
- **The KILL-R3 rule applied to the retrained translators (reading only):** the round-3 outcome rules for A (ENERGY-ACCOUNTS / KILL-R3 level / PARTIAL) are evaluated on A1 retrained. Whatever the result, the abstract states it in the rule's own terms, next to the released-translator outcome. KILL-R3 was registered for the released translators, so no stop follows from this. The paper's headline already does not rest on A1.
- **A4, κ and γ under retrained ridge translators** (`r4/retrained_a4.py`):
  - The G pipeline is refit unchanged (same split, grid and selection). From it we compute κ_ret = E‖W_ret Πf‖²/E‖Z‖² on the test half, γ_ret = κ_ret/ρ, and A4_ret = AS after removing β′Πf with β′²κ_ret = κ0_ret = min over E12.
  - We report τ against SING with CIs, as a descriptive result.
  - If τ(A4_ret) < 0.3 on both E12 and S11, the paper says that matched CLIP-space energy removes the agreement under both translator families. Otherwise it says that this holds for the released translators only.
- **Influence sentence:** if any leave-two-out τ(A1 released) on S11 is ≥ 0.6, the text says that the S11 value without that pair reaches the KILL-R3 level.

## DN. True DINO ViT-B/16 and ViT-S/16, with a supervised ViT-B/16 control (PR2 W5; R1 §1) — `r4/dino.py`
- **Features** (private): `$IC014_FEATS/r3_dino_feats/{dino_vitb16, dino_vits16, timm_vitb16_augreg}_10k.npz`. These hold F, W, b, pred, files and labels, with the 10k rows in `subset_files_10k.txt` order (the same order as `clip_img.npz`). The order is checked by comparing `files` against the subset list; a mismatch stops the analysis.
- **KILL-DN (feature validity).**
  - For each model, argmax(FWᵀ + b) must agree with the stored `pred` on ≥ 99.9 % of the 10k images.
  - For the two DINO models, the stored 50k top-1 must be within 1.0 point of the documented 78.2 % / 77.0 %.
  - If either fails, the DINO analysis is reported as invalid and stops, and no DINO result enters the paper.
- **Null space:** SVD of W at absolute tolerance 1e-5. If a model has null dimension 0 (expected for the 768-d AugReg ViT-B/16 with 1000 classes), we report ρ = 0 and AS = 0 by construction, and that model takes no further part.
- **Translator:** SING released no translators for these models, so each gets a ridge translator from the G pipeline, unchanged. The CLIP target is `clip_img.npz`, with the same split, grid and selection. We report fidelity on the test half.
- **Scores** (test half, class text):
  - ρ (all 10k);
  - AS_ret;
  - A1_ret at ρ₀ = 0.005;
  - κ_ret and γ_ret;
  - AS_perm_ret and AS_permc_ret (3 draws, seeds 92000+r / 93000+r).
- **Comparison set:** the 13 models' retrained values from the same pipeline (`r3/results/G_retrain_permodel.json` for AS_ret and A1_ret; κ_ret from `r4/retrained_a4.py`).
- **Q-DN1, the SING narrative "self-supervised models leak least".**
  - "SING-consistent" if both DINO models rank among the lowest four AS_ret of the 15 models (13 + 2).
  - "SING-inconsistent" if both lie above the median AS_ret of the 13.
  - "Mixed" otherwise.
- **Q-DN2, the energy reading** (the same form of test). We fit log AS_ret = a + b·log ρ on the E12 models by OLS and form the 95 % prediction interval at each DINO model's ρ.
  - "Energy-consistent" if both DINO log AS_ret values fall inside it.
  - "Energy-inconsistent" if both fall outside.
  - "Mixed" otherwise.
  - The same test is applied to the supervised ViT-B/16 if its null dimension is > 0.
- **Reading:** the paper reports both labels. Neither label licenses a statement about why a model leaks. If the DINO features fail KILL-DN, §5 says plainly that the paper makes no claim about DINO models.

## P. Planted content against energy (R1 fix 10; PR2 W6c, change 7) — `r4/planted.py` (from `B_core.json`; no new feature computation)
- **Full planted table:** AS(α)/AS(0) at ρ_t = 0.2 and at each model's own ρ, for α ∈ {0.1, 0.3}, on the 3 core models.
- **Energy-equivalent α** per model. Let r_E = AS(ρ_75)/AS(ρ_25) at α = 0, where ρ_25 and ρ_75 are the 25th and 75th percentiles of ρ over the E12 models. AS(ρ) at α = 0 is interpolated log-log across the 4 registered levels and the model's own ρ.
  - α* is the α at which AS(α)/AS(0), at ρ_t = 0.2, equals r_E. It comes from linear interpolation over α ∈ {0, 0.1, 0.3}.
  - If the planted ratio at α = 0.3 is below r_E, we report α* > 0.3 (outside the tested range).
- **Reading:** the paper replaces "moves less than with a change of energy" with the α* values. If the sentence cannot be supported by α*, it is deleted.

## C6. Class-SD claim bootstrap (R1 fix 6; N7) — `r2/e2_claims_ci.py` (revised)
- For C6-SD only, under both scores, the CI resamples **classes** rather than images: 1000 resamples of the 1000 per-class means with replacement, seed 0. The BCa acceleration comes from a delete-one-class jackknife. The point estimate is unchanged.
- **Sensitivity (reported, not used for the verdict):** the variance-corrected SD, √max(SD² − mean within-class SE², 0), with the within-class SE of the per-class mean AS. Under AS_SING this is the SE of the class mean of AS. Under AS_fair it is the delta-method SE of the class ratio ΣAS/Σrand.
- **Pass:** the C6-SD point lies inside both its percentile and its BCa CI for both scores, and `N7_flags` is empty. Otherwise we report a fail.
- All other claims keep the round-3 image bootstrap and BCa. **C10:** a footnote gives both intervals and the round-2 verdict. The tally says "3 reverse (one borderline)" if C10 still reverses.

## N10 close-out (R1 fix 8)
- We do not re-time. The accuracy criterion fails on a deterministic quantity (AS_fair nonlinear Δ = 0.057 > 0.03 at NDRAW = 5), so a quieter box cannot make N10 pass.
- README and paper state: "fast mode fails the AS_fair tolerance; use the full run (1.95 h)".
- On contention, the round-3 timed run did not record the load average at start and end. An earlier attempt was aborted because other jobs were running (`r3_e9_fast_contended_aborted.log`). During the timed run the job itself used about 7.3 of the 8 cores. The box is shared by several agents, so contention cannot be excluded.

## Repo and presentation (R1 fixes 7, 9, 11–13; PR2 changes 4, 6, 8–12)
- **Family-C table:** generated by `r3/make_tables.py` from `r2/results/fix2_models.csv`. Pass: `grep -c 'begin{tabular}' paper_workshop/main.tex` returns 0, and `bash scripts/reproduce.sh && git diff --exit-code` stays clean.
- **Sign convention:** every correlation is against the leakage order (−SING ratio) unless a sentence says otherwise. d_null is reported as Spearman +0.46 (p = 0.12) against the leakage order.
- **Pinned environment:** `requirements-pinned.txt` lists exact versions. Pass: a fresh venv install plus reproduce gives a clean diff.
- **Double-blind archive:** a `git archive` of HEAD, stripped of author names, account names, GitHub URLs and box paths, with no git metadata. It is built outside version control (`_local/`, git-ignored). Pass: a grep of the unpacked archive finds none of these strings. The public repo is not changed.
- **Wording:**
  - restrict the recommendation to AS-type translated-angle scores;
  - call the NuSA-style norm an energy proxy;
  - note the possible in-sample advantage of the released translators;
  - fix the §4.3 "artefact" contradiction;
  - use point-estimate language for CIs at n = 13;
  - label DINOv2's H = 0.98 as a negligible effect;
  - add citations: Theiler et al. 1992, Hastie et al. 2009 §3.4, Bansal et al. 2021, Lenc & Vedaldi 2015;
  - note external timestamping: this file is pushed to GitHub on its own before any round-4 computation.

## Kill and stop rules
- **KILL-R4-S** (S) and **KILL-DN** (DN) as above. If KILL-R4-S fires, we stop before rewriting and report. KILL-DN stops only the DINO part.
- If a result points to a change of direction beyond the headline fixed above and these reading rules, we stop and report rather than make the change.
- Every outcome is reported, pass or fail. No criterion, seed, threshold or definition here changes after the first round-4 number is computed. Anything added later is labelled post hoc.
