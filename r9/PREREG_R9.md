# Pre-registration, round 9: the three-step check on scores other than SING

Date: 2026-11-04. Frozen and pushed together with its code (`r9/*.py`, `r9/gpu/extract_r9_gpu.py`) and synthetic
unit tests (`r9/tests/test_r9.py`) before any round 9 number was computed on a real model. Adds round 9 to
PREREG_R2 to PREREG_R8; none of those is edited. After the push, this file is not edited; corrections go to a
separate `r9/DEVIATIONS_R9.md`.

## 0. Question

The check in the paper (step 1 centring, step 2 content-free surrogate, step 3 head refit) has so far been run in full
only on SING. Round 9 asks whether it gives informative answers on other scores that read a discarded subspace, and
whether it can tell planted content from pure energy when the answer is known.

## 1. What is known before this round

* Real data, earlier rounds: SING AS, IS and rho results (rounds 2 to 8); round 6 ran ViM (p_virtual, alpha refit),
  ActSub, CORE S_mem, conditional V-information and a NuSA-style share on all 10k images of the 13 models (`r6/README.md`):
  ViM NOT ENERGY-DRIVEN, NuSA-style share MIXED with tau 0.82 against SING's order. Round 9 uses a different split
  (fit half / evaluation half), different surrogates (matched mean and covariance per model) and new controls, so
  those numbers do not carry over, but the authors have seen them.
* Synthetic data only, this round: the calibration run in `r9/results/synthetic/` (13 toy models, d = 160, K = 40).
  The negative control flags the energy reference in 40/40 datasets; the positive control at a = 0.6 is not flagged
  in 40/40 for the content reference (manipulation tau 0.97); PC-text is not flagged in 20/20 for SING AS
  (tau 0.81). ViM p_virtual and core_res have NC flag rate 0 on synthetic data: both are invariant to the overall
  scale of the residual, so the pure-energy control cannot move them (see §6, low power).
* No round 9 score has been computed on real features.

## 2. Models and data

* CPU arm (confirmatory): the 13 SING models with cached classifier-input features of the 10k ImageNet-val subset
  (first 10 sorted files per class; listing sha256 75cdd9a7...a42f): `r9/models13.txt`. Features are private and are
  not committed.
* Fit images: the first 5 images of each class in file order (5,000); evaluation images: the other 5,000. Anything
  fitted (class means, ViM basis and alpha, refitted heads, core prototypes) uses fit images only; every score and
  top-1 is computed on evaluation images only.
* GPU arm (extension): `r9/gpu_models.txt`, 71 timm ImageNet-1k classifiers with a linear or 1x1-conv head and
  classifier input d > 1000 (null(W) is empty for d <= 1000): 4 anchors with the same weights as four of the 13
  (BiT-R50x1, EfficientNet-B0 NS, ConvNeXt-B in22k, ResNeXt101 WSL) and 67 new backbones. Extraction protocol:
  `r9/gpu/extract_r9_gpu.py` (same 10k listing, timm eval transforms, input of `get_classifier()` by pre-hook,
  float32, W and b from the classifier). Optional train10k split (first 10 sorted train files per class).

## 3. Scores

Notation: f the classifier input, W (K, d), b the head, P_null the projector onto null(W) (SVD tolerance 1e-5).

| key | definition | role |
|---|---|---|
| as_sing | SING AS: mean over evaluation images of abs(angle(A f + Ab, t_y) - angle(A f_row + Ab, t_y)), t_y the normalised mean of the normalised prompt embeddings of the true class, f_row the row-space part of f | reference (the paper's case study) |
| nsp | NSP (He et al. 2025, arXiv 2506.17249, §3.3): mean of norm(P_null f) / norm(f + o), o = W^T (W W^T)^-1 b, so W o = b | score |
| nusa | NuSA (Cook et al. 2020): mean of norm(P_null f) / norm(f) | score |
| rho | null-energy share of round 2 | score |
| null_norm | mean norm(P_null f) | energy reference |
| core_null | mean cosine between the null coordinates of f and the null mean of the predicted class (fit images) | content reference |
| vim | ViM p_virtual, official recipe: u = -pinv(W) b; principal space of the uncentred covariance of f - u on fit images, D = 1000 if d >= 2048, 512 if d >= 768, else d/2; alpha = mean max logit / mean norm(res) on fit images; mean of sigmoid(alpha norm(res) - logsumexp(logits)) | score |
| vim_res | mean of norm(res) / norm(f - u) | score |
| res_norm | mean norm(res) | energy reference |
| core_res | as core_null in the ViM residual space | content reference |

k-NNuSA (IEEE ICTC 2024) is excluded: no public definition or code could be found, and we will not guess one.
The scores are reduced to one per-model value; the check is about the cross-model order of that value.

## 4. The check (per score, `sanity_check_general.py`)

1. Centre: split the subspace energy into offset, between-class and within-class parts; recompute the score with the
   offset only and with the offset removed. Reported, no verdict.
2. Surrogate: replace each model's subspace coordinates by Gaussian draws with that model's own mean and covariance
   (10 draws on real data, mean of the draws used), and by the same draws without the mean. tau_E = Kendall tau-b
   between real and surrogate per-model values; model-bootstrap BCa 95% CI, B = 2000, at least 4 distinct models per
   resample.
   * ENERGY-DRIVEN: lower bound of tau_E > 0.5.
   * CONTENT-READING: upper bound < 0.5 and var(real - surrogate) / var(real) >= 0.5.
   * MIXED: otherwise. With 13 models the intervals are wide; MIXED means not resolved.
   * Order-equivalent to the surrogate: BCa 90% lower bound of tau_E > 0.9 (|dtau| < 0.1).
   For the ViM family the principal basis is frozen at its real-data value during replacements and alpha is refitted
   (simplification; under head refits ViM is refitted in full).
3. Head: refit the head on fit images and recompute every score; report top-1 on evaluation images.
   * Accuracy-matched heads (confirmatory): a ridge head distilled from the released logits (lambda by 2-fold CV) and
     the released head with a random perturbation (largest eps in 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01 that keeps fit
     top-1 within 1 point). A head counts as matched on a model if its evaluation top-1 is within 1 point of the
     released head's.
   * Exploratory heads: nearest class mean, shrinkage LDA (Ledoit-Wolf). Reported, not used for a reading.
   * Reading per score and head: tau between the per-model score under the released head and under the refitted head
     over matched models only: HEAD-STABLE tau >= 0.8, HEAD-DEPENDENT tau <= 0.5, INTERMEDIATE otherwise; NOT ESTIMABLE
     with fewer than 8 matched models.

Against SING: tau between each score and SING's published order (`r9/sing_order.csv`, -IS/AS ratio from the paper's
Fig. 6b), with a paired CI for dtau = tau(real) - tau(surrogate); "same order as its surrogate with respect to SING"
if the 90% CI lies inside (-0.1, 0.1). With n = 13 this margin is only reachable when the two orders nearly coincide;
a failure to reach it is not evidence of a difference.

## 5. Controls (`controls_r9.py`)

Built from the real features of the 13 models by rewriting the subspace coordinates C (mean mu, covariance Sigma,
E_c = tr Sigma, G ~ N(0, Sigma), unit class prototype u_y):

* NC, pure energy: C' = o mu + sqrt(e) G, with e in {0.25, 0.5, 1, 2, 4} and o in {0, 0.5, 1, 1.5, 2} drawn per model.
  No content by construction; the cross-model order is set by (e, o) alone.
* PC, planted content: C' = mu + sqrt(e) (sqrt(1 - a^2) G + a sqrt(E_c) u_y). Six strengths a_max in {0, 0.15, 0.3,
  0.45, 0.6, 0.75}; within a dataset each model gets a_max times a permutation of linspace(0, 1, 13).
  PCm: e = 1 (content only). PCc: e drawn per model from the e grid (content crossed with energy).
  Prototypes: random unit vectors (rand) or, for SING AS, the translator-aligned text direction u_y ∝ A_null^T t_y (text).
* Streams and replicates: NC_null 40, PCm_null_rand 40 per level, PCc_null_rand 20, PCm_null_text 20, PCc_null_text 10,
  NC_vim 20, PCm_vim_rand 20, PCc_vim_rand 10; 3 surrogate draws refitted to each control dataset.
* Seeds: numpy default_rng([100900, stream, level, rep]) per dataset, [.., model] per model; real surrogate draws stream
  9, head perturbations stream 10; bootstrap seed 100900.
* Reported per score, stream and level: rate of ENERGY-DRIVEN, rate not ENERGY-DRIVEN, rate CONTENT-READING with
  Wilson 95% intervals, and mean tau of the score with a, e and o.

## 6. Pass and kill criteria

* K-ctrl (global, null family): fires if the NC flag rate of null_norm is below 0.8, or the not-flagged rate of
  core_null in PCm_null_rand at a = 0.6 is below 0.8, or the mean tau(core_null, a) there is below 0.6. If K-ctrl
  fires, it is reported and no score verdict is given (all STOPPED).
* ViM family: the same three criteria with res_norm and core_res on the vim streams; if they fail, ViM-family verdicts
  are withheld and the null-family verdicts stand.
* Per score: if the NC false CONTENT-READING rate exceeds 0.2, that score's verdict is WITHHELD. If its NC flag rate is
  below 0.8 (power below 0.8), a MIXED verdict is reported as "MIXED (low power)".
* SING AS: CONTENT-READING on real data is reported as such only if PC-text validates it (not-flagged rate >= 0.8 and
  abs tau(AS, a) >= 0.6 at a = 0.6); otherwise it is reported as MIXED (not validated).

## 7. Outputs

`r9/results/r9_controls.json`, `r9_real_checks.json`, `r9_head_swap.json`, `r9_per_model.json`, and `r9/README.md`
reporting every verdict, the control power and every failure.

## 8. GPU arm gating

The CPU arm does not wait for the GPU arm. GPU results are confirmatory only if (a) features arrive with the manifest
and checksums of `r9/gpu/extract_r9_gpu.py`, (b) the four anchors reproduce the cached features (median relative
feature difference below 1e-3 and top-1 within 0.5 point; otherwise the arm is reported but not confirmatory), and
(c) they are analysed with the same runner and criteria (no text scores: no translators exist for these models).
Otherwise they are reported as a pre-registered extension that was not completed. With more than 13 models the same
thresholds apply; no threshold is changed after the CPU results are seen.

## 9. Known deviations from SING's own setup (disclosed in advance)

* Heads are refitted on the fit half of the val subset, not on training images (no training features exist for the
  13 models on this machine). The GPU arm also extracts train10k; train-fitted heads there are secondary.
* SING AS is computed on the evaluation half only, whereas SING's published values used all images.
* ViM basis frozen during replacements (§4).
