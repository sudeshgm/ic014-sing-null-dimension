# IC014 results: SING model comparison and null dimension

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
* Fig. 6b plots only IS/AS (13 bars). Our digitized values, at ±0.38: DenseNet 70.71, DinoViT 70.16, EVA02-L 54.32, BiT 47.21, ConvNeXt 37.38, EffNet 36.28, Swin-L 29.18, ResNet50 24.81, VGG16 24.26, VGG19 19.89, DINOv2 16.61, ResNeXt101 13.88, DeiT 9.51.
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
* Fig. 6b reference order: densenet, eva02, bitresnet, convnext, resnet, dinov2, resnext, deit.
* Kendall τ vs Fig. 6b: 0.286 (p 0.40) for |AS|; 0.357 (p 0.28) for IS/|AS|.

Boring baseline over 13 models: Spearman(ρ_full, |AS|_full) = 0.868 (p 1.2e-4); Spearman(ρ_full, Fig. 6b ratio) = −0.940 (p 1.9e-6).

## 4. Verdict (from the IC014 Idea Agent)
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
* **KILL-R did not fire.** Fix 2 and D-weak hold under both random-control families, D-strong does not, so the allowed wording is "differs from", not "reverses".
* **Energy explains the published order.** τ vs Supp. Fig. 12b over 13 models: ρ alone 0.821 [0.56, 1]; AS_SING 0.821 [0.56, 1]; M1 (post hoc) 0.667 [0.29, 0.94].
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
* **Fix 10, E9 holdout.** `fix10_e9_holdout.json`; see the paper (`paper_workshop/holdout_numbers.tex`).
* **Fix 13, PC bands.** G = AS/√ρ is larger for bottom-variance null PCs than for top-variance ones. Top/random < 1 on 3 of 4 models at d=24 (ResNet 1.16). Energy-matched size is a confound, not a signal.
* **Fix 14.** Needs ImageNet images; spec in `gpu_jobs/fix14_vitl_translator_fidelity.md`. Not run.
