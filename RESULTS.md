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

No candidate gives 84.0 ± 0.3 (Supp. Table 2). Every candidate is supervised or SWAG weakly supervised; none is DINO.

## 2. Exp 0 (`exp0/`)
* Fig. 6b plots only IS/AS (13 bars), digitized at ±0.38: DenseNet 70.71, DinoViT 70.16, EVA02-L 54.32, BiT 47.21, ConvNeXt 37.38, EffNet 36.28, Swin-L 29.18, ResNet50 24.81, VGG16 24.26, VGG19 19.89, DINOv2 16.61, ResNeXt101 13.88, DeiT 9.51.
* Null dims match the plan, except **DenseNet121 = 27** (rank 997 at tol 1e-5).
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

Fits, as R² of AS vs d / vs ρ_d / vs log d, then the log-d slope:

| Model | R² vs d | R² vs ρ_d | R² vs log d | log-d slope |
|---|---|---|---|---|
| ResNet50 | 0.983 | 0.983 | 0.900 | 0.375 |
| ResNeXt101 | 0.977 | 0.977 | 0.906 | 0.468 |
| DINOv2-L | 0.974 | 0.974 | 0.908 | 0.349 |
| BiT-R50x1 | 0.512 | 0.512 | 0.140 | −1.21 |

r = AS(1048)/AS(24), with 2σ band (recomputed from the raw files, confirmed):

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
* Energy vs dimension. AS vs ρ_d fits as well as AS vs d (R² ≈ 0.98 for both), because ρ_d is linear in d for random subspaces. Exp 1 does not separate the two.
* Boring baseline. The null-energy fraction ρ_full alone gives Spearman −0.94 against SING's ratio over 13 models.
* BiT anomaly (`exp1/diag_bit.py`, `exp1/results/diag_bit.json`; 100 Imagenette images on CPU). Same code path as the other models; the collapse is translator conditioning:
  * The full null space and the random subspaces run through the same code: the same `v_null` from SING's projectors, the same pre-logit hook, and no normalization step. The random basis is `Vn @ qr(randn)`.
  * BiT's linear translator has a huge gain on the classifier null space: the top singular values of A·V_null are 284, 179 and 159, with a median of 0.16. The mean gain on principal directions is only 0.14. For ResNet50 the top singular value is 5.6.
  * Split the null space into 8 random orthogonal blocks. The translator shifts from the blocks sum exactly to the full-null shift (|Δz| 3.72 vs 3.72), but the blocks' root-sum-square is 285.6. For ResNet50 the same comparison gives 8.0 vs 9.3.
  * The full null component of real BiT features nearly cancels through the translator; any partial subspace breaks that cancellation. BiT's AS(d < 1048) therefore reflects the translator's conditioning. The other models do not show this.
* DenseNet121. ρ_full = 1.8e-8 and |AS| ≈ 2e-4: the features have essentially no energy in the head's null space.
* Subset count. `subset_files.txt` shows 9999 lines in `wc -l` only because the script writes no trailing newline. n = 10,000 in every result row, and the per-image labels are exactly 10 per class.
