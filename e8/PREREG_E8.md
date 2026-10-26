# E8 pre-registration: Null-space Concept Probe (NCP)

This file is committed before any E8 computation. The spec was fixed by the IC014 Idea Agent on 2026-10-26 (option a).

## Data
- The private E3 features from `e3/extract_e3_gpu.py`, for 8 models: resnet, resnext, dinov2, bitresnet, deit, eva02_large, convnext and densenet.
- N = 10000 ImageNet val images.
- CLIP concept z-scores, plus SING's V_n (tol 1e-5).
- CPU only. No downloads, and no features are committed.

## Planting (identical to E3)
- f' = f + α·σ·s_c(x)·u_c
- u_c = normalise(P_null A⁺ t_c)
- s_c = the CLIP z-score of concept c
- Concepts: red, striped and water.

## NCP(model, concept, α, S)
For a subspace S with orthonormal basis B (d columns):
1. **Probe input:** X = F'B. The target is y = s_c(x).
2. **Split:** 80/20 train/test, using a fixed permutation (seed 0) that is the same for every fit.
3. **Ridge:** standardise X and y on the train split. Choose λ from a log grid of 10^-3 to 10^5 (17 points) by exact leave-one-out CV on train. Report the test R².
4. **Random baselines:** 20 energy-matched random subspaces of the full feature space, each a Haar basis with seed 10000+r, exactly as in E3's AS_fair:
   - remove q_1..q_{k-1} fully and scale the coordinate on q_k by √β;
   - choose k and β so that E‖coef‖² = E‖B B^T f'‖², at the same α.
   - The probe input is those k coordinates (the last one scaled).
5. **NCP** = (R²_S − mean R²_rand) / SD R²_rand, where SD is the population SD over the 20 draws.

## Kendall (planted ranking): the E3 trials protocol
- 50 trials, seed 0, with α_m ~ logUniform[0.01, 1] drawn per model (the same draws as E3).
- In each trial, compute NCP for each model at its drawn α_m with S = the full null space.
- Take Kendall τ(NCP_m, α_m) over the 8 models, per trial and per concept.
- Report:
  - the mean τ over trials, per concept;
  - the concept-averaged per-trial τ;
  - a 95% bootstrap CI over trials (1000 resamples, seed 0);
  - the per-trial 2.5–97.5 percentiles.

## Flatness across d (option a)
- α is fixed at the median of the log-uniform range: α = √(0.01·1) = 0.1. The plant always goes into the full null space.
- d ∈ {24, 100, 280, 536, full null dim}. Any d larger than the model's null dim is dropped.
- For each d, draw 5 nested random sub-null subspaces: B = V_n R[:, :d], where R is a Haar random nd×nd matrix with seed 20000+j, j = 0..4.
  - For a given j, the subspaces are nested across d.
  - At d = full, B spans the whole null space.
- At each d, NCP is computed as above (20 energy-matched random draws, matched to E‖B Bᵀ f'‖²) and then averaged over the 5 draws.
- Per model and concept, the ratio is max_d NCP / min_d NCP.
  - The ratio is undefined if min ≤ 0; that case is reported and counted as not flat.
  - The per-model ratio uses the NCP averaged over the 3 concepts. Per-concept ratios are also reported.
- **Flat** means a per-model ratio ≤ 1.5 on at least 3 of the 4 eligible models. Eligible models are those with null dim ≥ 1048: resnet, resnext, dinov2 and bitresnet.
- Models with null dim 24 or 27 have only one d value, so their ratio is 1 trivially. They are reported but are not eligible.

## Rule (applied by the Idea Agent, not by the CPU agent)
- PASS if mean Kendall ≥ 0.8 AND flat.
- KILL if mean Kendall < 0.6.
