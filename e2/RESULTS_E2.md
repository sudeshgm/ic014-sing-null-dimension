# E2 results: all 13 SING models (raw; no verdict)

**Setup**
- Features:
  - E3 cache (8 models) plus E2 (5 models) on the same 10k subset (`subset_files.txt` identical)
  - claim-class set: 200 images, all 50 val images each of classes 107, 334, 354 and 817, for 13 models plus CLIP. Its first 10 images per class reproduce the 10k-subset features exactly (max diff 0).
- The features are private and not in this repo.
- AS_fair uses the definition registered for E3, at α=0, with 20 random subspaces. Target is the class text.
- The bootstrap uses 1000 resamples over images.

## Kendall τ vs the Fig. 6b IS/AS ordering (exp0 digitization)
| score | all 13 | without DenseNet (12) |
|---|---|---|
| AS_SING (class \|AS\|) | 0.821 (p 1.3e-5) | 0.788 (p 1.1e-4) |
| AS_fair (class) | −0.179 (p 0.435) | −0.364 (p 0.116) |
| IS/\|AS\| (ours) | 0.744 (p 1.5e-4) | 0.697 (p 9.7e-4) |
| AS_SING, mean of 173 concepts | 0.641 (p 0.0016) | — |
| AS_fair, mean of 173 concepts | −0.026 (p 0.952) | −0.182 (p 0.459) |

## Rankings (least leakage first)
- **Fig. 6b:** densenet, dinovit1, eva02, bitresnet, convnext, efficientnet, swin_large, resnet, vgg16, vgg19, dinov2, resnext, deit
- **AS_SING:** densenet, convnext, dinovit1, eva02, efficientnet, bitresnet, swin_large, vgg16, vgg19, dinov2, resnet, resnext, deit
- **AS_fair:** bitresnet, densenet, resnext, vgg16, dinov2, vgg19, swin_large, resnet, efficientnet, convnext, deit, dinovit1, eva02
- **AS_fair without DenseNet:** bitresnet, resnext, vgg16, dinov2, vgg19, swin_large, resnet, efficientnet, convnext, deit, dinovit1, eva02

## Claims
The per-claim table with values and CIs is in `results/e2_claims.md` and `results/e2_claims.json`. Per-model values are in `results/e2_models.csv`.
| id | claim | source | AS_SING | AS_fair |
|---|---|---|---|---|
| C1 | DinoViT has the best IS/AS trade-off (highest ratio) of the 5 models | Fig. 3b / Sec. 4.2 / abstract | holds | reverses |
| C2 | ResNeXt101 has the lowest IS/AS ratio of the 5 models | Fig. 3b caption | holds | reverses |
| C3 | ResNet50 leaks more class semantics into its null space than DinoViT | abstract / Fig. 1 / Fig. 4 | holds | reverses |
| C4 | DinoViT has the least class-relevant leakage among the examined networks | Introduction | holds | reverses |
| C5 | ResNeXt101 shows high AS (top-2 of the 5) | Sec. 4.2 | holds | reverses |
| C6 | ResNet50 has larger and more variable per-class AS than DinoViT | Fig. 4 / Sec. 4.3 | holds | reverses |
| C7 | DinoViT per-class |AS| is typically < 1 | Sec. 4.3 | holds | reverses |
| C8 | ResNet50 classes Porcupine and Sports-Car leak substantially more than other classes (>= 90th percentile) | Fig. 4 caption | reverses | reverses |
| C9 | No significant correlation between ResNet50 and DinoViT per-class AS orderings | Sec. 4.3 | holds | reverses |
| C10 | DinoViT: Jellyfish has substantially larger AS than Arabian Camel (proxy: class-text AS on all 50 val imgs/class when --claim-feats given; SING used 30 concepts, ~1300 imgs) | Fig. 5 / Sec. 4.3 | reverses | reverses |
| C11 | Table 1: DinoViT least steerable; ResNet50/EfficientNet/ResNeXt101 susceptible (proxy: null-removal AS, not IS=40 deg text-gradient steps) | Table 1 / Sec. 4.4 | holds | reverses |
| C12 | Fig. 6b / supp. Fig. 12b 13-model IS/AS ordering (holds = Kendall tau > 0 and p < 0.05) | 13-model IS/AS bar plot (digitized in exp0 as Fig. 6b; supp. Fig. 12b) | holds | reverses |
| C13 | EVA02 as target space preserves the 5-model ordering | Sec. 4.2 / supp. Fig. 12a | not testable (no EVA02-space translators released) | not testable (no EVA02-space translators released) |
| C14 | Translator robustness: Pearson 0.972 between principal and translated-principal head accuracy | Sec. 4.2 / supp. Fig. 12c | not a comparative leakage claim; not tested | not a comparative leakage claim; not tested |

C8 (ResNet50 class percentile, 95% bootstrap CI over 50 images):

| class | AS_SING | AS_fair |
|---|---|---|
| porcupine | 84.8 [80.1, 87.7] | 68.4 [64.5, 71.5] |
| sports car | 61.0 [56.1, 66.2] | 67.2 [63.6, 71.4] |

C10 (DinoViT, jellyfish − camel, 95% CI):
- AS_SING: 0.346 − 0.453, CI [−0.205, −0.010]
- AS_fair: 1.381 − 2.321, CI [−1.403, −0.458]

## Notes (raw)
- **DenseNet:** null energy ρ=1.8e-8, so a matched random subspace is about 0.004 of one direction and AS_fair is degenerate.
- **BiT:** AS_fair = 0.042, because random subspaces move |AS| by 13.2°.
- **DinoViT:** our IS/|AS| is 20.95, against 70.16 in SING's Fig. 6b.
- **C9 under AS_fair:** Spearman = −0.086, p = 0.006.
- **C1/C2:** these are IS/AS-ratio claims, but they are tested here on |AS|. Our IS/|AS| ranking is in `results/e2_rankings.json`.
