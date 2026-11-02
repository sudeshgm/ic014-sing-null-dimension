# IC014 results: SING model comparison and null dimension

## Headline (current paper framing, round 3)
SING's cross-model ranking is indistinguishable from a null-energy ranking. With SING's released translators, energy accounts for the agreement; with retrained translators, it accounts for part of it. Round-3 numbers are in `r3/results/` (see `r3/README.md`); earlier rounds below.
* **ρ and AS_SING are indistinguishable predictors of SING's order** (F). Kendall τ vs Supp. Fig. 12b: ρ 0.821 [0.56, 1], AS_SING 0.821 [0.56, 1]; paired difference CI [−0.36, 0.29] (BCa [−0.37, 0.28]); permutation p = 5e−5 for both; partial Kendall τ 0.61 both ways; partial Spearman 0.63 (AS | ρ), 0.72 (ρ | AS). Like-for-like ratio τ (our IS/|AS| vs SING IS/AS) 0.744 [0.33, 1.00]; SING/ours 0.98–3.35 (largest: DinoViT 3.35, EVA02 2.51).
* **Equal-energy intervention (A): ENERGY-ACCOUNTS, no KILL-R3.** A1 (remove √(ρ₀/ρ)·Πf, ρ₀ = 0.005): τ −0.03 [−0.54, 0.61] (E12), 0.09 [−0.45, 0.78] (S11); paired drop CI [0.23, 1.39] / [0.08, 1.30]. A2 (top PCs) 0.03 / 0.13; A3 (random rotation) 0.03 / 0.16; A4 (matched CLIP energy) −0.15 / −0.20. κ 0.67 (E12), κ − ρ CI [−0.41, 0.28]; γ = κ/ρ −0.06. NuSA-style null norm 0.85 (all 13), τ(NuSA, ρ) 0.97.
* **Leak-free null (B): N4 passes on all 3 models** (Spearman 1, log-log slopes 0.38–0.69, CIs exclude 0): AS responds to null energy with no image-specific content; "confound" only in this qualified sense. Planted concept at ρ_t = 0.2: ×1.04–1.09 (α = 0.1), ×1.35–1.51 (α = 0.3). Leak-free scores reproduce the order: τ(AS_surr) 0.69 [0.39, 0.92] (all 13), 0.64 (S11); Spearman(AS_surr, AS_SING) 0.96. No claim from the excess X (τ 0.54 [0.14, 0.83]).
* **Retrained translators (G):** fidelity 0.79–0.84 (released 0.81–0.87 on the same test half). τ(AS retrained) 0.85 (all 13), 0.82 (E12, S11): order survives retraining; τ(retrained AS, released AS) 0.72. **Caveat:** A1 with retrained translators keeps part of the agreement, 0.61 [0.15, 0.94] (E12) and 0.75 [0.32, 1.00] (S11). The registered ridge grid put every model at its edge; post hoc extended grid: 0.39 / 0.53 (only VGG16/19 and DenseNet change λ, so fragile); λ = 10^6 everywhere: ≈ 0.79–0.82.
* **PC bands vs the energy law (C):** d = 24, â = 0.431: ResNet50 1.50, ResNeXt101 0.74, DINOv2-L 0.98 [0.97, 0.99] → "top PCs carry less AS than energy predicts" (2/3), not robust (at â = 0.388: "more"). d = 100 ratios: 1.18, 0.97, 0.91. BiT-R50x1 (0.009) is reported but not counted.
* **Within-model growth (Exp 1) and energy law (E1):** |AS| rises about 10× from d = 24 to d = 1048 on 3 of 4 models (BiT collapses); |AS| ∝ ρ^a, a = 0.431 [0.388, 0.485]; E1 flatness part fails (mixed).
* **DinoViT checkpoint (Exp 0b, fix 14, D):** SING's `dinovit1` translator matches timm AugReg ViT-L/16 (cos 0.848, shuffled floor 0.575, retrieval top-1 78.8 %); the other four candidates sit at their shuffled floors (gap ≤ 0.002, retrieval ≤ 0.2 %). No candidate gives 84.0 %; the 84.0 vs 85.85 % gap is unexplained (protocol unknown). SING describes the model as pretrained beyond ImageNet. A true DINO model is a GPU job spec (`gpu_jobs/r3_dino_vitb16.md`), not run.
* **Size-corrected rankings differ but do not reverse** (round 2). D-weak passes at the threshold on the screened set (5/8); D-strong fails, and its single screened pass comes from family B, which is not logit-preserving (max Δlogit/SD up to 74). AS_fair over-corrects; rankings are unstable across null tolerances.
* **Claims (E, BCa, de-duplicated):** under AS_SING 7 hold, 3 reverse (C4, C8, C10), 1 contradicted (C9). N7 flag: AS_fair C6 SD point −0.203 outside BCa [−0.2642, −0.2641] (degenerate bootstrap), reported, not used.
* **Failed or not-passed pre-registered checks:** E8 KILL, E9 KILL (holdout: no candidate passes as registered), E3 not passed, E1 flatness part fails, tolerance stability fails, D-strong fails, AS_fair "removes the trend" fails, flatness rule ill-posed, N7 (one BCa flag), PC-band label not robust in â.

## 1. DinoViT identity (Exp 0b)
Source: `exp0b/results/results_top1.json`, full ImageNet-val (50k).
| Candidate | native | r224 |
|---|---|---|
| torchvision ViT_L_16 IMAGENET1K_V1 | 79.670 | 79.692 |
| timm vit_large_patch16_224.augreg_in21k_ft_in1k (the one SING's models.yaml loads) | 85.854 | 85.826 |
| torchvision SWAG_LINEAR_V1 (CC BY-NC) | 85.140 | 85.012 |
| torchvision SWAG_E2E_V1 (CC BY-NC; native 512) | 88.066 | 76.502 |
| HF google/vit-large-patch16-224 (Apache-2.0) | 81.976 | 82.874 |

No candidate gives 84.0 ± 0.3 (Supp. Table 2). All of them are supervised or SWAG weakly supervised, and none is DINO.

## 2. Exp 0 (`exp0/`)
* SING Supp. Fig. 12b (called Fig. 6b in our file names) plots only IS/AS (13 bars). Our digitized values, at ±0.38: DenseNet 70.71, DinoViT 70.16, EVA02-L 54.32, BiT 47.21, ConvNeXt 37.38, EffNet 36.28, Swin-L 29.18, ResNet50 24.81, VGG16 24.26, VGG19 19.89, DINOv2 16.61, ResNeXt101 13.88, DeiT 9.51.
* The null dims match the plan except for **DenseNet121 = 27** (rank 997 at tol 1e-5).
* IS/AS vs null dim, all 13, DenseNet = 27: Spearman −0.456 (perm p 0.124), Kendall −0.379 (0.105). Without DinoViT: −0.381 (0.224) / −0.329 (0.187). With DenseNet = 24: −0.517 (0.077) / −0.449 (0.054).

## 3. Exp 1 (`exp1/results/`; 10,000 val images, 10 per class; 20 seeds)
Mean |AS| in degrees (seed SD), with ρ_d underneath:

| d | ResNet50 | ResNeXt101 | BiT-R50x1 | DINOv2-L |
|---|---|---|---|---|
| 24 | 0.166 (0.035) | 0.175 (0.034) | 9.30 (0.23) | 0.143 (0.011) |
| 48 | 0.273 (0.086) | 0.251 (0.037) | 10.64 (0.34) | 0.215 (0.013) |
| 100 | 0.393 (0.112) | 0.416 (0.075) | 11.89 (0.18) | 0.322 (0.015) |
| 280 | 0.736 (0.278) | 0.843 (0.172) | 12.76 (0.21) | 0.644 (0.050) |
| 536 | 1.055 (0.290) | 1.338 (0.160) | 13.10 (0.21) | 1.027 (0.044) |
| 1048 (full) | 1.682 | 1.999 | 0.556 | 1.499 |
| ρ at 24 / 1048 | 0.0033 / 0.146 | 0.0107 / 0.451 | 0.0011 / 0.046 | 0.0074 / 0.322 |

Fits, given as R² of AS vs d / vs ρ_d / vs log d, followed by the log-d slope:

| Model | R² vs d | R² vs ρ_d | R² vs log d | log-d slope |
|---|---|---|---|---|
| ResNet50 | 0.983 | 0.983 | 0.900 | 0.375 |
| ResNeXt101 | 0.977 | 0.977 | 0.906 | 0.468 |
| DINOv2-L | 0.974 | 0.974 | 0.908 | 0.349 |
| BiT-R50x1 | 0.512 | 0.512 | 0.140 | −1.21 |

r = AS(1048)/AS(24) with its 2σ band (recomputed from the raw files and confirmed):

| Model | r | 2σ band |
|---|---|---|
| ResNet50 | 10.15 | [5.83, 14.48] |
| ResNeXt101 | 11.44 | [6.93, 15.95] |
| DINOv2-L | 10.48 | [8.85, 12.10] |
| BiT-R50x1 | 0.060 | [0.057, 0.063] |

d=24 comparison over 8 models:
* By |AS| (low first): densenet, convnext, dinov2, resnet, resnext, eva02, deit, bitresnet.
* Supp. Fig. 12b reference order: densenet, eva02, bitresnet, convnext, resnet, dinov2, resnext, deit.
* Kendall τ vs Supp. Fig. 12b: 0.286 (p 0.40) for |AS|; 0.357 (p 0.28) for IS/|AS|.

Boring baseline over 13 models: Spearman(ρ_full, |AS|_full) = 0.868 (p 1.2e-4); Spearman(ρ_full, Supp. Fig. 12b ratio) = −0.940 (p 1.9e-6).

## 4. Exp 1 verdict (from the project lead)
GO under the pre-registered rule: r ≥ 3 on 3 of the 4 models and τ ≤ 0.5.

## 5. Caveats
* Energy vs dimension. AS vs ρ_d fits just as well as AS vs d (R² ≈ 0.98 for both), since ρ_d is linear in d for random subspaces. Exp 1 cannot separate the two.
* Boring baseline. On its own, the null-energy fraction ρ_full gives Spearman −0.94 against SING's ratio over 13 models.
* BiT anomaly (`exp1/diag_bit.py`, `exp1/results/diag_bit.json`; 100 Imagenette images on CPU). The code path is the same as for the other models, and the collapse comes from translator conditioning:
  * The full null space and the random subspaces go through identical code: the same `v_null` from SING's projectors, the same pre-logit hook, no normalization step. The random basis is `Vn @ qr(randn)`.
  * On the classifier null space, BiT's linear translator has a very large gain. The top singular values of A·V_null are 284, 179 and 159 (median 0.16), while the mean gain on principal directions is only 0.14. For ResNet50 the top singular value is 5.6.
  * We split the null space into 8 random orthogonal blocks. The translator shifts from the blocks add up exactly to the full-null shift (|Δz| 3.72 vs 3.72), yet their root-sum-square is 285.6. For ResNet50 the same comparison gives 8.0 vs 9.3.
  * For real BiT features, the full null component nearly cancels out through the translator, and any partial subspace breaks that cancellation. So BiT's AS(d < 1048) reflects how the translator is conditioned. None of the other models shows this.
* DenseNet121. ρ_full = 1.8e-8 and |AS| ≈ 2e-4, i.e. the features have essentially no energy in the head's null space.
* Subset count. `wc -l` reports 9999 lines for `subset_files.txt` only because the script writes no trailing newline. Every result row has n = 10,000, and the per-image labels are exactly 10 per class.

## 6. Round 2 (response to `review_20261101_v1`; criteria in `PREREG_R2.md`)
All numbers come from `r2/results/`; CIs are 95%. Unless noted, bootstraps resample models (2000×, seed 0), and LOO is leave-one-model-out.
* **Energy explains the published order (headline).** τ vs Supp. Fig. 12b over 13 models: ρ alone 0.821 [0.56, 1]; AS_SING 0.821 [0.56, 1]; M1 (post hoc) 0.667 [0.29, 0.94].
* **KILL-R did not fire.** Fix 2 and D-weak hold on all 13 and on the screened set, and D-strong does not, so the allowed wording is "differs from" and "reverses" is ruled out. This is a secondary finding.
* **Fix 3, size-fair scores (all 13).** τ for each normaliser:

  | Normaliser | τ | 95% CI |
  |---|---|---|
  | AS/ρ^0.25 | 0.72 | [0.40, 1] |
  | AS/ρ^0.43 | 0.33 | [−0.15, 0.77] |
  | AS/ρ^0.5 | 0.26 | [−0.24, 0.71] |
  | AS/ρ | −0.41 | [−0.86, 0.15] |
  | AS/d | 0.05 | [−0.54, 0.59] |
  | AS_fair A | −0.18 | [−0.78, 0.35] |
  | AS_fair B | −0.41 | [−0.83, 0.11] |
  | AS_fair C | −0.13 | [−0.63, 0.42] |

  Screened set (no DenseNet/BiT): AS_fair A −0.56 [−1, 0.00], B −0.60 [−0.89, −0.23], C −0.27 [−0.72, 0.29]. File: `fix3_normalisers.csv`.
* **Fix 4.** Spearman(AS_fair, ρ): −0.26 [−0.96, 0.41] over all 13, and −0.79 [−1, −0.23] screened. AS_fair over-corrects, so we do not claim it "removes the trend".
* **Fix 5, tolerance (failed).** The minimum adjacent-tolerance τ falls below the pre-registered 0.9 for AS_fair (0.87 all-13; 0.82 screened) and for AS_SING (0.87 all-13; 0.96 screened). Null dim (0.96) and ρ (0.92) pass. The degeneracy screen excludes DenseNet and BiT.
* **Fix 6, E1 CIs.** a = 0.431. Cell bootstrap gives [0.388, 0.485], cluster bootstrap [0.382, 0.464], LOMO 0.408–0.452, and per-model a 0.38–0.46. The CI for b covers 0.
* **Fix 7, symmetric E3.** α0-subtracted: AS_SING 0.386 [0.324, 0.448] and AS_fair 0.109 [0.046, 0.171]. Without DenseNet: 0.343 and 0.135.
* **Fix 8, claims.** See `fix8_claims_ci.csv`. Under AS_SING, C4 and C8 reverse, C10 is not confirmed, and the rest hold. C12 under AS_fair is not confirmed.
* **Fix 9, flatness.** The oracle score itself has a max/min ratio of 3.55 (full null) and 3.48 (fixed 24-d), so the registered rule (≤ 1.5) is ill-posed. Under the revised rule (score/oracle ratio ≤ 1.5), only M1 qualifies, at 4/4 and 4/4 with M1/oracle 1.00–1.03. E8 revised: 0/4.
* **Fix 10, E9 holdout (post hoc).** `fix10_e9_holdout.json`; water, green, thin, knitted; 50 trials, seed 1; 8 models. No candidate passes the E9 rule as registered, so E9 stays KILL.

  | Score | lin. τ [95% CI] | nl. τ | flat (α 0.1/1) | revised flat | pass |
  |---|---|---|---|---|---|
  | M1 | 0.932 [0.914, 0.949] | 0.968 | 0/0 | 4/4 | revised rule only |
  | M2 | 0.438 [0.388, 0.485] | 0.463 | 0/0 | 0/0 | no |
  | S9 | 0.454 [0.401, 0.509] | 0.439 | 0/1 | 2/0 | no |
  | S10 | 0.372 [0.327, 0.417] | 0.518 | 3/4 | 0/0 | no |
  | M5 (AS/√ρ) | 0.223 [0.175, 0.273] | 0.145 | 0/0 | 0/0 | no |
  | S4 (AS/ρ) | −0.081 [−0.151, −0.014] | −0.144 | 0/0 | 0/0 | no |
  | AS_SING (raw) | 0.332 [0.268, 0.399] | 0.362 | 0/0 | 0/0 | no |
  | AS_fair (α0-subtracted) | 0.167 [0.113, 0.221] | −0.045 | 0/0 | 0/0 | no |

  M1 meets both τ thresholds (τ ≥ 0.8, CI lower ≥ 0.7, nl. τ ≥ 0.6) but is flat on 0/4 models under the registered rule; it passes only under the revised (post-hoc) flatness rule. S10 is the only candidate flat under the registered rule, and its τ is too low. Paired over trials (1000×, seed 0), τ(M1) − τ(AS_fair) = 0.765 [0.714, 0.820] linear and 1.013 [0.959, 1.065] nonlinear. M1 is given the concept score that defines the plant and the AS-family scores are not, so this comparison favours M1 by construction. Runtime 1.95 CPU-h (under the 3 h guard).
* **Fix 10, E2 under M1 (post hoc, descriptive).** `r2/e2_m1.py` → `fix10_e2_m1.json`. M1 at α=0, full null, mean over SING's 173 concepts. τ vs Supp. Fig. 12b: 0.667 [0.29, 0.94] on all 13 (LOO 0.61–0.79) and 0.600 [0.06, 0.92] screened. Both CIs include AS_SING's 0.821. M1's order tracks ρ: τ 0.79, Spearman 0.90 (screened 0.75, 0.85). M1 therefore does not give a ranking that differs from SING's.
* **Fix 13, PC bands.** G = AS/√ρ is larger for bottom-variance null PCs than for top-variance ones. Top/random < 1 on 3 of 4 models at d=24 (ResNet 1.16). So energy-matched size acts as a confound rather than as a signal.
* **Fix 14, translator fidelity (pass).** `fix14_fidelity.json`. Mean cos(T(f), CLIP_img) under SING's `dinovit1` translator, first 1000 subset images, native transforms: timm AugReg (used) 0.848, torchvision IMAGENET1K_V1 0.677, HF google/vit-large-patch16-224 0.631, SWAG linear 0.615, SWAG E2E 0.608. Margin 0.171 ≥ 0.05. The re-extracted AugReg features equal the cached ones exactly. Features from a GPU run on Kaggle (2 × T4); they are ImageNet-derived and not in the repo.
