# E3: planted leakage (8 models; no E2)
1. GPU pass (`extract_e3_gpu.py`): same 10k subset as Exp1/E1. Saves:
   * full penultimate features F, the head (W, b) and the SING linear translator (A, Ab) for resnet, resnext, dinov2, bitresnet, deit (DeiT3-L), eva02_large, convnext (ConvNeXt-B) and densenet (DenseNet121)
   * CLIP image embeddings from the Karlo-v1-alpha image encoder. This is CLIP ViT-L/14, the encoder inside the UnCLIP model SING uses (`kakaobrain/karlo-v1-alpha-image-variations`), so it shares the 768-d space of SING's text embeddings.
   
   The outputs are private and ImageNet-derived. Do not commit them.
2. CPU analysis (`analyze_e3.py <feats> <results> --private <dir> --sing-root <sing>`). Everything is exact linear algebra on CPU.

## Definitions
* Concepts: `red`, `striped` and `water`, taken from SING's attribute prompt set (`misc/embeddings/attribute_embeddings_broden_style_fp16.npz`, prompt entry `mean`). t_c is that embedding, L2-normalised.
* Plant: f' = f + α·σ·s_c(x)·u_c, where:
  * σ = sqrt(E‖f‖²) over the unplanted images
  * α ∈ {0.01, 0.03, 0.1, 0.3, 1}
  * u_c = normalise(V_n V_nᵀ A⁺ t_c). A⁺ is the pseudo-inverse of the translator weight, and V_n is SING's null basis (tol 1e-5).
  * s_c(x) = z-score over the N images of cos(CLIP_img(x), t_c)
  
  The max |logit change| at each α is recorded; on fake images it is about 1e-6.
* AS for concept c, per image: θ(T(f'), t_c) − θ(T(f̃'), t_c), in degrees. The metrics average |AS| over images.
* AS_SING(α) = mean|AS| for full null-space removal (f̃' = f' − V_nV_nᵀf') with the released translator, at α, minus the same quantity at α = 0.
* AS_fair(α) = mean|AS_null(α)| / ( (1/R) Σ_r mean|AS_{R_r}(α)| ), with R = 20. R_r is an energy-matched random subspace of the full m-dim feature space:
  * Take a Haar-random orthonormal basis q_1..q_m (seed 10000 + r).
  * Remove q_1..q_{k−1} fully and q_k by a fraction β ∈ (0, 1].
  * k and β are chosen so that E‖f' − f̃'‖² = E‖V_nᵀ f'‖² exactly, at the same α.
  * The expectation is over the N images. The numerator is the same null removal as AS_SING, without the α = 0 subtraction.
* **Ground truth**:
  * 50 trials (seed 0). In each trial, α_m ~ logUniform[0.01, 1], drawn independently for each of the 8 models; the truth is the α ranking.
  * For each trial and concept: Kendall τ(metric_m(α_m), α_m) over the 8 models, with both metrics computed exactly at the drawn α.
  * Reported: the mean τ with a 95% bootstrap CI of the mean over trials (10k resamples), plus the 2.5–97.5 percentiles of the per-trial τ. These are given per concept and averaged over concepts.

## Outputs
* **Small, committed after the run**:
  * `results/e3_table.csv`: (model, concept, α) rows with AS_SING, AS_fair, ρ_null, the random-subspace |AS| mean and SD, k, the max logit change and the fraction of u kept by the null projection
  * `results/e3_trials.csv`: per-trial α, metric values and τ
  * `results/e3_summary.json`
* **Private** (`--private`), for E8 (NCP):
  * `<model>_null.npz`: null coordinates c = F V_n and V_n, so Π_null f = c V_nᵀ
  * `clip_concepts.npz`: raw and z-scored CLIP scores for all 173 SING attribute concepts

Smoke test: the full pipeline ran on 12 fake images for all 8 models plus CLIP (3 draws, 5 trials).
