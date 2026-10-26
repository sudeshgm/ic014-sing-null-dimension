# E2: all 13 SING models, AS_SING vs AS_fair, and a re-test of SING's claims
E2 uses the AS_fair definition registered for E3, unchanged, at α = 0 (no plant).

* **Cached from E3** (`/workspace/gpubot/ic014/e3/ret/e3_feats`, private): resnet, resnext, dinov2, bitresnet, deit, eva02_large, convnext and densenet.
* **Missing**, i.e. SING's 13 registry models (`translators/registry.yaml`) minus those 8: vgg16, vgg19, efficientnet (EfficientNet-B0 NS), swin_large and dinovit1.
* **DinoViT** is SING's own registry entry `dinovit1`: timm `vit_large_patch16_224`, default tag `augreg_in21k_ft_in1k`. This is the model SING's released translator was trained for. Exp 0b found that no candidate matches the paper's 84.0 top-1; this entry scores 85.85.

1. **GPU** `extract_e2_gpu.py` runs the E3 extractor on the 5 missing models: same subset, same `<model>_full.npz` format, and no CLIP (E3's `clip_img.npz` covers the same subset). The outputs are ImageNet-derived and private. Never commit them.
2. **CPU** `analyze_e2.py <out> --feats <e3_feats> <e2_feats> --sing-root <sing>` writes:
   * `e2_models.csv`: per model, class-text |AS|, IS, IS/|AS|, AS_fair, ρ_null, and the means over the 173 concepts
   * `e2_concepts.csv`: model × concept AS_SING and AS_fair
   * `e2_perclass.csv`
   * `e2_rankings.json`: rankings, plus Kendall τ vs the Fig. 6b digitization from exp0
   * `e2_claims.{json,md}`: SING's comparative claims, each marked holds / reverses under AS_SING and under AS_fair. Lower |AS| or AS_fair counts as less leakage.

`results_partial8/` is the same analysis on the 8 cached models only. Claims that need missing models show as "not testable yet".
Smoke test: the 5 missing models were extracted on 12 fake images, and the analysis ran on all 13 fake models.
