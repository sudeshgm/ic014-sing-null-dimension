# Round 7: controlled semantic edits and independent nuisance views (results)

Pre-registration: `r7/PREREG_R7.md`, committed and pushed in 4babcfb ("Add round 7 pre-registration") before any real feature
existed. Plan and data choice: `r7/PLAN.md`. Per-model aggregates: `r7/results/<model>.json` (pre-registered pipeline) and
`r7/results/<model>_sens.json` (post-hoc corrected CIs, §3). No per-image values, features or images are published.

## 1. What was run

- Data: IN-9 Backgrounds Challenge test set (sha256 2d66f571...), N = 900 foreground images (100 per IN-9 class, seed 100701).
  Edit X = `mixed_next` (background from the next class, fine ImageNet synset known) vs S = `mixed_same`; artefact control
  Xs, Ss = the same pair with background pixels permuted inside the background mask; 4 independent random nuisance views at
  each of 3 levels (seeds from 100700), plus a clean view. 52 forward passes per image and model.
- Models (SING registry at e47b37d, classifier-input features, null space from `null_split(W)`):
  - resnet (torchvision resnet50 IMAGENET1K_V1) and efficientnet (timm tf_efficientnet_b0.ns_jft_in1k): CPU, torch 2.14.1.
  - resnext (timm resnext101_32x8d.fb_wsl_ig1b_ft_in1k, HF revision 95470a4c, model.safetensors sha256 8d77b419... matched):
    GPU (Tesla T4), torch 2.11.0+cu130, same frozen code at 4babcfb, fp32. A partial CPU run (torch 2.14.1, first 240 of 900
    items) agrees with the GPU features to 1.5e-6 relative (Frobenius; max per-image 8e-6) with identical head weights; the
    CPU run was then stopped and the GPU copy analysed.
  - dinov2 (facebook/dinov2-large-imagenet1k-1-layer, HF revision 79cf9d02): GPU (Tesla T4), torch 2.11.0+cu130, same code, fp32.
  - Deviation (environment, not analysis): GPU runs used torch 2.11.0 / torchvision 0.26.0 instead of the CPU env's torch
    2.14.1 / torchvision 0.29.1; timm 1.0.30, transformers 5.20.2, numpy 2.5.2, Pillow 12.3.0 (JPEG views) were identical.
    items_r7.json sha256 8c8619ec... matched for every run; all GPU files matched their SHA256SUMS on arrival.
- Synthetic stability check (K0) passed before any real data (see PREREG §4): T2 stable to nu = 2 under low-rank noise.

## 2. Results against PREREG_R7 (primary level L2 unless stated)

| model | d_null | Rel L2 [95% CI] | H1 | nu L1/L2/L3 | T2 edit L1/L2/L3 (perm p) | T2 artefact L2 (p) | dT2 edit-artefact L2, prereg boot CI | same, corrected CI | dT2 edit-nuisance L2, corrected CI | H2 (T2) prereg | H2 corrected | H3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet | 1048 | 0.720 [0.711, 0.729] | REPRODUCIBLE | 0.35/0.78/1.65 | 0.00156/0.00174/0.00164 (0.001/0.001/0.001) | 0.00162 (0.002) | [-0.00138, 0.00178] | [-0.00130, 0.00153] | [0.00032, 0.00295] | ARTEFACT | ARTEFACT | NOT TESTED (H2 not ALIGNED) |
| efficientnet | 280 | 0.753 [0.746, 0.761] | REPRODUCIBLE | 0.22/0.66/1.62 | 0.00070/0.00100/0.00152 (0.001/0.001/0.001) | 0.00087 (0.001) | [-0.00121, 0.00136] | [-0.00089, 0.00106] | [0.00032, 0.00181] | ARTEFACT | ARTEFACT | NOT TESTED (H2 not ALIGNED) |
| resnext | 1048 | 0.748 [0.732, 0.762] | REPRODUCIBLE | 0.33/0.69/1.59 | 0.00109/0.00103/0.00126 (0.001/0.001/0.001) | 0.00087 (0.004) | [-0.00173, 0.00233] | [-0.00087, 0.00122] | [0.00008, 0.00179] | ARTEFACT | ARTEFACT | NOT TESTED (H2 not ALIGNED) |
| dinov2 | 1048 | 0.821 [0.814, 0.827] | REPRODUCIBLE | 0.18/0.45/1.07 | 0.00154/0.00158/0.00154 (0.001/0.001/0.001) | 0.00137 (0.001) | [-0.00143, 0.00199] | [-0.00095, 0.00141] | [0.00069, 0.00233] | ARTEFACT | ARTEFACT | NOT TESTED (H2 not ALIGNED) |

| model | clean top-1 agree X,S | null share of clean edit | T2 global L2 | T (secondary) L1/L2/L3 | T_naive L1/L2/L3 | cross-view edit energy tr L1/L2/L3 | row-space T2 L2 (p) / Rel L2 | world-B rescale diff |
|---|---|---|---|---|---|---|---|---|
| resnet | 0.657 | 0.064 | 0.0164 | 0.0314/0.0325/0.0264 | 0.229/0.229/0.222 | 87.9/66.2/45.0 | 0.00270 (0.005) / 0.825 | 1.2e-17 |
| efficientnet | 0.668 | 0.137 | 0.0083 | 0.0215/0.0295/0.0426 | 0.243/0.246/0.251 | 36.9/29.1/20.8 | 0.00140 (0.005) / 0.784 | 2.8e-17 |
| resnext | 0.791 | 0.372 | 0.0156 | 0.0264/0.0244/0.0266 | 0.193/0.203/0.214 | 1705.7/1430.8/1082.1 | 0.00144 (0.005) / 0.778 | 6.1e-17 |
| dinov2 | 0.782 | 0.360 | 0.0114 | 0.0500/0.0488/0.0440 | 0.266/0.263/0.261 | 3764.0/3029.8/2185.1 | 0.00235 (0.005) / 0.834 | 2.1e-17 |

Reading the tables:
- **H1 (reproducible null-space edit): REPRODUCIBLE in all four models** (split-half reliability of the within-class null-space
  edit 0.72-0.82 at L2, CI lower bounds far above 0.10; 0.55-0.65 at L3). **K1 did not fire.**
- **Estimability:** nu <= 2 at every level for every model, so T2 is estimable everywhere; T (secondary) is not estimable at L3
  (nu > 1) in any model.
- **H2 (alignment with the outside reference above controls): ARTEFACT in all four models.** The null-space edit's within-class
  Gram is significantly aligned with the CLIP text embedding of the inserted background class (permutation p = 0.001, the
  floor, at every level), and well above the nuisance-only control. But the artefact control (same images with the background
  pixels randomly permuted, which keeps the background's colour histogram and destroys its spatial / semantic layout) is
  aligned just as strongly: T2(edit) - T2(artefact) is ~0 with CIs straddling 0 in every model and level, under both the
  pre-registered and the corrected CIs. The detectable alignment is therefore carried by low-level background statistics
  (colour / intensity distribution), not by the semantic content of the edit.
- **H3 (noise cancellation in a world-C analogue): NOT TESTED** in every model, as pre-registered (H2 not ALIGNED).
- **Per-model summary: "real views support noise cancellation" = false for all four. Family statement: NOT SUPPORTED (0 of 4).**
- **Kill rules:** K0 (synthetic) did not fire; K1 (no reproducible null-space edit) did not fire.
- The automated verdict file also prints an "H2 = ALIGNED" line for the secondary statistic T. That is a code artefact:
  the pre-registration makes T descriptive (permutation test only) and the code fills the control conditions as passed for
  T. It is not a verdict and is ignored here.

## 3. Deviation found after the analysis (post hoc): the pre-registered bootstrap CIs for T and T2 are invalid

The pre-registered within-group bootstrap resamples images with replacement. HSIC_u only drops the true diagonal, so a
duplicated image forms an "off-diagonal" pair with itself and its self-similarity is counted as cross-image covariance. The
bootstrap distributions of T and T2 are then shifted far upwards (for example, resnet L2: point T2 0.0017, bootstrap CI
0.012-0.016, which does not contain the point estimate). The CIs for dT2(edit - artefact) and dT2(edit - nuisance) that H2
uses are affected in the same way. Reliability is not a U-statistic and its CI is unaffected; point estimates and permutation
tests never resample and are unaffected. The docstring claim in `stats_r7.parts` that duplicates are harmless was wrong. The
unit tests did not catch this because they never compared a bootstrap CI with its point estimate.

Correction (post hoc, `r7/sens_r7.py`, seed 100750): within-group half-sampling without replacement (500 draws) with basic
CIs. For half-samples Var(theta_half - theta_hat) ~ Var(theta_hat), so the CIs are on the full-sample scale. This correction
was not validated for coverage on synthetic data. **It changes no verdict.** The difference to the artefact control is ~0
either way, and the corrected CIs (table column "corrected") still straddle 0 in every model and level.

## 4. Descriptive (no verdict)

- **Manipulation check for the nuisance levels holds** (nu3/nu1 = 4.7 resnet, 7.4 efficientnet, 4.8 resnext, 5.9 dinov2).
  But the cross-view (signal) energy of the edit also falls by 37-49 % from L1 to L3: the stronger crops remove background, so
  the levels change content as well as noise. Even if H2 had passed, the levels were not a clean world-C.
- T2 across levels (not a test): flat for resnet (max/min 1.11), resnext (1.22) and dinov2 (1.02); rises 2.2x for efficientnet.
- **The row space (classifier-visible part) carries the same kind of alignment at a similar size** (row T2 0.0014-0.0027
  vs null T2 0.0010-0.0017). The null space is not special here.
- World-B check: rescaling the null edit by 0.1 or 10 changes T2 by < 1e-16 (exact invariance, as designed).
- The aligned share is tiny: T2 ~ 0.001-0.002 of the within-class edit variance. In global mode, which includes
  foreground-class structure, it is ~10x larger (0.008-0.016), so most of the apparent alignment in a pooled analysis would be
  between-class structure.

## 5. What this means for a score (clue only; the paper is not changed in this round)

- A null-space edit is real and reproducible across independent nuisance views (H1). The machinery works on real data:
  T2 is estimable at every level, and nuisance-only controls are near 0.
- But on this edit set, what links the null-space edit to an outside semantic reference is explained by low-level
  background pixel statistics. So the experiment gives **no evidence for semantic noise cancellation** in the null space.
- Practical requirements for any future null-space "semantic" score, learned from this round:
  - it needs independent views (a cross-view Gram), so it cannot be computed from cached single-pass features;
  - it needs a matched low-level control, such as a pixel-shuffle of the edited region;
  - it needs a within-group design;
  - it needs a bootstrap that never pairs an image with itself.
