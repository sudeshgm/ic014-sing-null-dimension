# IC014 E9: screen of replacement scores (PRE-REGISTERED 2026-10-26 19:58 IST, before any of these is computed)
Requested by the user via Idea Launcher. Runs after E2 and E8. CPU only, on the saved E3 data (/workspace/gpubot/ic014/e3/). No new downloads (disk is nearly full).

## Ground truth and data split (fixed now)
- The plant is as in E3. Every score is reported as score(alpha) - score(0), so natural leakage is subtracted.
- DEV (used for choosing): concepts red and striped; E3 trials seed 0 (50 trials).
- HOLDOUT (touched once, only for the finalists): concept water, plus 3 new concepts drawn now with numpy seed 2026 from the 173 saved CLIP concepts excluding the 3 used, and 50 new trials with seed 1. The CPU agent records the 3 drawn names in the repo before computing any score.
- Flatness: models with null dim 1048 (ResNet50, ResNeXt101, DINOv2-L, BiT), fixed alpha in {0.1, 1}. The score is computed on random d-dim subspaces of the null space, d in {24,100,280,536,1048}, 5 seeds each. Max/min of the mean across d.

## Candidates (fixed now)
- S1 NCP (= E8): held-out ridge R2 from null coords to s_c, minus the mean R2 over 20 random full-space subspaces matched in d and energy, as a z-score.
- S2 Kernel NCP: as S1 with RBF kernel ridge (Nystrom, 1000 landmarks, median-heuristic bandwidth).
- S3 Whitened-null probe: whiten null coords by their covariance (ridge 1e-3), then held-out linear R2 to s_c.
- S4 Energy-normalised AS: AS_SING / rho_null.
- S5 Dimension-normalised AS: AS_SING / d_null.
- S6 d-matched z-scored AS: (AS_null - mean AS_R) / sd AS_R, with 20 random full-space subspaces of the same d.
- S7 Translator-free CCA: first canonical correlation between null coords and s_c (held-out).
- S8 Debiased linear CKA between null coords and s_c, minus a d- and energy-matched random baseline.
- S9 KSG mutual information (k=5) between s_c and the held-out best 1-D null projection.
- S10 Rank score: Spearman between null coords projected on V_n V_n^T A^+ t_c and s_c.
Combinations: rank-averages of the 3 best DEV singletons (3 pairs + 1 triple) = S11-S14. 14 candidates in total. Reported regardless of outcome.
Train/test for probes: a fixed 50/50 image split (seed 0).

## Selection and rule (one rule, fixed now)
1. On DEV, compute mean Kendall tau vs planted alpha over the 8 models, with a 95% bootstrap CI over trials, and flatness for every candidate.
2. Finalists: at most 3, the highest DEV mean tau among those with DEV tau >= 0.8 AND flatness <= 1.5 on >= 3/4 models. If none qualify, there are no finalists and E9 = KILL.
3. HOLDOUT, finalists only, evaluated once: PASS if holdout mean tau >= 0.8 with the 95% CI lower bound >= 0.7, AND flatness <= 1.5 on >= 3/4 models at holdout concepts. Multiple-testing guard: the holdout is untouched by selection; with up to 3 finalists each must clear the CI bound on its own.
4. KILL: no finalist passes the holdout. Then the paper has no working replacement; it reports the screen as a negative result.
No candidate, parameter, split or threshold may change after the first DEV number is computed. Anything extra goes in as clearly labelled 'post hoc' and can't PASS.

## AMENDMENT 1 (2026-10-26 20:00 IST, user request "think mathematically"; still before any score is computed)
Theory: to first order AS is proportional to ||T P f||, which is about sqrt(rho) under isotropy, so the prediction is log AS ~ 0.5 log rho (E1 measured a=0.43). Planted truth alpha = ||E[P f s]|| / sqrt(E||f||^2) (+ natural leakage). The naive estimator has bias E||Pf||^2/n, which grows with null size.
Added candidates:
- M1 Debiased null leakage: L^2 = (||sum_i Pf_i s_i||^2 - sum_i ||Pf_i||^2 s_i^2) / (n(n-1) E||f||^2), with s standardised. Score = sqrt(max(L^2,0)).
- M2 CLIP-space leakage: M2 = M1 * |t_c^T T u_hat| / (||T V_n||_F / sqrt(d_null)), where u_hat = the unit direction of the debiased estimate of E[P f s]. This is the concept-direction translator gain relative to the translator's average null gain, which fixes BiT-style conditioning.
- M3 Held-out partial R2: R2(s | Wf, Pf) - R2(s | Wf), linear, adjusted. Pre-registered prediction: FAILS (ranks by SNR, not amount).
- M4 Gaussian conditional MI: -0.5 log(1 - partial R2).
- M5 Linearised SING / sqrt(rho_null).
- M6 Scale-free alignment: E[t_c^T T P f s] / sqrt(E||T P f||^2).
Combinations S11-S14 are now drawn from the best 3 DEV singletons among S1-S10 and M1-M6 (20 candidates in total).
Robustness check (required for PASS, added now): a nonlinear plant f' = f + alpha sigma g(s_c) u_c with g(s) = s^2 - 1 (zero-mean, uncorrelated with s), on HOLDOUT concepts only. A finalist must also reach tau >= 0.6 there, judged against the alpha ranking, using |s|-based variants as registered: for M1/M2 replace s by g(s). Scores that by construction can't see nonlinear plants are reported, and the paper states this limitation.
