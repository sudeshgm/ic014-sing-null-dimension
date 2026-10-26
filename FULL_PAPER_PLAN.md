# IC014 full-paper plan (PROPOSAL, awaiting user approval; nothing runs until approved)

Working title: "Size Matters: Null-Space Interpretability Scores Measure Subspace Energy, Not Semantics"
Thesis: scores that probe what a classifier ignores (its null space or other residual subspaces) are dominated by how much feature energy and how many dimensions they remove. Cross-model comparisons with these scores are confounded unless they are size-matched. SING is the case study, and the lesson is general.

Already established (short-note stage, RESULTS.md cfb4bfa):
- Within a model, AS rises ~10x from d=24 to d=1048 (3 of 4 models).
- At matched d=24, the ranking disagrees with SING's (tau 0.29).
- Null-energy fraction rho vs SING ratio over 13 models: Spearman -0.94.
- "DinoViT" is a supervised ViT-L; no candidate reproduces 84.0%.
- BiT shows full-null cancellation caused by translator conditioning.
Main limit: energy and dimension are collinear (R2 0.98 for both).

Note on cost: features and translators exist for the 13 SING models. A new model needs a new translator (est. 2-4 GPU-h each; to be measured in E0). All GPU runs go through GPU Bot on free Kaggle first.

## E0. Pilot: translator cost and reproducibility (1 GPU-h)
Q: Can we train one SING translator (ResNet50) ourselves and recover the released one's AS within 10%?
Why: every new-model experiment depends on it; reviewers will ask whether the results are reproducible.
PASS: |AS_ours - AS_released|/AS_released <= 0.10 and training time <= 4 GPU-h per model. KILL (for E4 new models only): either fails, so E4 drops to the 13 released models and E4b.

## E1. Separate energy from dimension (core; ~1.5 GPU-h, ~0.5 day CPU)
Q: At fixed d, does AS follow removed energy? At fixed energy, does it follow d?
Design (ResNet50, ResNeXt101, DINOv2-L, EVA02-L, VGG16):
(a) Fixed d in {24, 100, 280}: high-energy null directions (top PCs of features projected into the null space) vs low-energy ones vs random, 20 seeds.
(b) Fixed rho in {0.5%, 2%, 8%}: choose d by mixing high/low-energy directions to hit the same rho at d in {24, 100, 280, 536}.
Fit log AS = a*log rho + b*log d + c (models pooled, model fixed effects).
Why: this is the reviewers' first question, and it decides whether the fix is energy-matching or dim-matching.
Outcome rule (any decisive outcome is publishable):
- ENERGY-DRIVEN: AS(high)/AS(low) >= 3 at fixed d on >=4/5 models, and AS changes <= 1.5x across d at fixed rho.
- DIMENSION-DRIVEN: the reverse.
- BOTH: both effects >= 2x, with 95% CI of a and b excluding 0.
- KILL: both effects < 1.5x on >=3/5 models. That contradicts Exp 1, so stop and audit.

## E2. Size-fair SING and whether SING's conclusions survive (~2 GPU-h, 1 day CPU)
Q: Define AS_fair = AS at matched removed energy (or matched d, per E1), normalized by a random-subspace baseline of the same size drawn from the full feature space (a "non-null" control). Do SING's headline conclusions hold?
Re-test, on the 13 models, (i) the model ranking, (ii) SING's per-class / per-concept claims in its main figures, (iii) the "DINO leaks least" claim.
Why: this turns a critique into a corrected method; reviewers want to know what changes.
PASS: Kendall tau(AS_fair, SING ranking) <= 0.5, or >= 1 headline claim reverses. KILL: tau >= 0.8 and no claim changes. The paper then becomes "the confound exists but is harmless here", which weakens it to TMLR.

## E3. Ground truth: planted leakage (~1 GPU-h, 1 day CPU) [most important for acceptance]
Q: When we plant a known amount of semantic signal (a CLIP-text direction) into the null space of each model at controlled strengths, which metric recovers the true leakage ranking?
Design: 8 models x 5 planted strengths; head logits are unchanged by construction (null space), so it's a valid model.
Why: a known ground truth answers "which metric is right?" rather than just "they disagree".
PASS: Kendall(AS_fair, planted) >= 0.8 and Kendall(AS_SING, planted) <= 0.5. KILL: AS_SING >= 0.8 as well, which means SING already recovers the truth and the confound is cosmetic; downgrade.

## E4. Causal width manipulation on one backbone (~3 GPU-h)
Q: Same backbone (DINOv2-L and ResNet50 features), linear heads trained on random/PCA bottlenecks of width m in {128, 256, 512, 1024, 2048 (ResNet)}, plus a fixed translator. Does AS rise with m when the backbone is identical?
Why: it removes all architecture confounds; this is the cleanest causal evidence.
PASS: Spearman(m, AS_SING) >= 0.9 on both backbones, and AS_fair flat (max/min <= 1.5). KILL: Spearman(m, AS_SING) < 0.5.
E4b (needs E0 pass): +6 new models (e.g. ConvNeXt-L, ViT-B/16 supervised, real DINO ViT-B/16 + linear head, MAE ViT-L ft, CLIP ViT-L zero-shot head, RegNetY), ~15-20 GPU-h. This also tests the real "DINO vs supervised" question SING meant to ask. PASS: the cross-model Spearman(rho_full, AS_SING) holds |rho| >= 0.7 on all 19.

## E5. Generality: does the confound hit other projection/null-space scores? (~4 GPU-h, 2 days CPU)
Q: Across the same ~13-19 models, do these scores track subspace size/energy more than their intended property?
(1) ViM residual OOD score (ImageNet vs iNaturalist/OpenImage-O; ground truth = AUROC);
(2) Cook et al. null-space outlier score;
(3) Rezaei & Sabokrou null-space overfitting score (ground truth = train-test gap);
(4) LEACE/INLP concept-erasure "residual leakage" probe accuracy vs number of removed dims.
For each: partial Spearman of score with ground truth controlling for rho and d, and raw Spearman with rho.
Why: this turns one-paper criticism into a general lesson for a main track.
PASS: >= 2 of 4 methods have |Spearman(score, rho or d)| >= 0.6, and fair normalization raises their correlation with ground truth by >= 0.15. KILL: 0 of 4 confounded, so the paper stays SING-specific (TMLR).

## E6. Translator conditioning (BiT) as a second failure mode (CPU, 0.5 day)
Q: Across all models, does the translator's null-restricted condition number predict the gap between the sum of block-wise AS and full-null AS?
PASS: Spearman >= 0.7 over >= 13 models gives a diagnostic. KILL: < 0.4, so it's reported as a BiT anecdote only.

## E7. Tool + recommendation (CPU, 1-2 days)
A small MIT pip package `fairnull`: computes rho, d, AS_fair, a random-subspace baseline and a condition-number warning for any (features, head, probe) triple, plus a 5-item reporting checklist. Validate on E3 and E5. No pass/kill; it's the deliverable.

## Budget
- GPU: ~12 h core (E0 1, E1 1.5, E2 2, E3 1, E4 3, E5 4), plus 15-20 h for E4b. Free Kaggle first through GPU Bot.
- CPU: ~2 weeks.
Order: E1 then E3 then E2 (the core story; ~4.5 GPU-h). A go/no-go after those three, then E4, E5, E0/E4b, E6, E7.

## Overall gate
- Main track GO after the core trio: E1 decisive, E3 PASS, E2 PASS.
- If E3 KILL or E2 KILL, drop to TMLR / workshop note with the existing results.
- If E5 PASS as well, frame as general.

## Outline (8 pages)
1. Intro: null-space scores compare models; a hidden size confound; contributions.
2. Background: SING, ViM, null-space scores; the DinoViT identity.
3. The confound: within-model curves (Exp 1), energy vs dimension (E1), BiT cancellation (E6).
4. Ground truth: planted leakage (E3) and causal width (E4).
5. A size-fair score and what changes for SING (E2, E4b, including real DINO).
6. Beyond SING: four other scores (E5).
7. Recommendations + `fairnull` (E7). Limitations. Appendix: DinoViT accuracy table, licences.

## Venue
Primary: ICML 2027 main track (deadline expected around late Jan 2027; to be confirmed), which suits a measurement/methodology paper and gives ~3 months. Alternative: CVPR 2027 if its deadline (expected mid-Nov 2026; to be confirmed) is reachable after the core trio, which is tight. Fallback: TMLR (rolling) if only the SING-specific findings survive.
Licence: all research non-commercial (ImageNet, ResNeXt WSL, SWAG CC BY-NC).
