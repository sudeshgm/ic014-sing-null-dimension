# IC014 round 8 plan

Date: 2026-11-03 (SGT). Base: `67425f3` (round 7 reported). Source of the ideas: an outside review we commissioned ("ideas K",
3 Nov 2026; not in this repo). Companion: `r8/PREREG_R8.md`, which is binding where the two differ (it fixes the CI method by
K0 coverage, the G-S paired CI, the 1e-8 float64 tolerances, and drops standard-addition AS and the GPU phase from round 8).

## 0. Why a round 8 at all

Ideas K makes one structural point we accept: the gates gathered over rounds 4–6 (world B flat, world C flat, worlds M/D rise)
cannot all be met by any class-level score that is a function of (class-mean null energy E_b, residual null energy E_r, row
energy E_row, a scale-free composition κ), not even by an oracle (§2.1 of ideas K). Rounds 4–6 never stated which quantity a
replacement should measure. Round 8 fixes the target first and derives the gates from it.

Target, stated first:
- **T1 (primary): amount.** How much text-predictable class signal sits in ker W, measured against the row space of W (which a
  null-space intervention leaves alone). Truth under: world B (Πf rescaled) rises in closed form; world C′ (only null noise grows,
  class content fixed against the row space) flat; worlds M/D (content up) rise; offset-only flat; class derangement → 0;
  content-free surrogate → 0.
- **T2 (secondary, descriptive only): discarded information.** How much class information the head discards (Fisher information of
  the innovation u = z − H·s). Not gated; reported with its expected value ≈ 0 for near-stationary heads.
- **T3 (arrangement)** is what SO_null/XCC already measure (round 5); not re-run.

## 1. What we checked before writing this (all CPU; private scratch scripts, numbers reproduced here)

### 1.1 Ideas K's claims about our code
| Claim | Verdict | Evidence |
|---|---|---|
| (a) §2.2 world C holds L fixed per unit of *total* energy, so class-mean null energy rises 1/(1−ρ) ≈ 1.81× against the row space | **CONFIRMED** | `r4/calibration.py` L42–44: `Et = Er/(1-rho)`, `a2 = L*Et/Mm` ⇒ a²·Mm = L·Er/(1−ρ); (1−0.0054)/(1−0.451) = 1.812. The 1.81× rise is already disclosed (`main.tex` L294, macro `\QCalCAbsRise`=1.81). Not disclosed: that this fixes the yardstick (total energy incl. null noise), so a row-yardstick amount score must fail world C. |
| (b) §2.3 L and ρ are uncentred, so a global null mean offset counts as content | **CONFIRMED** | `calibration.py` L35 class means of uncentred C = F·Vn (`class_means` is a plain mean); ρ = E‖Πf‖²/E‖f‖² (`main.tex` L68). Not disclosed anywhere in the paper. On real data it is material (§1.3). |
| (c) §2.5 M1 description | **MOSTLY CONFIRMED; "offset cancels" REFUTED** | Formula, kill by the ill-posed flatness rule, post hoc pass of the revised rule, τ 0.79 with ρ and 0.667 with SING, E‖f‖² yardstick: all as stated (`main.tex` L712–730, `e9/analyze_e9.py` L66, L119–120). M1 is √max(L̂²,0), not the squared norm. The offset cancels only in the cross term; the removed diagonal Σ‖xᵢ‖²yᵢ² contains ‖μ‖², giving E[L̂²] bias −ρ_offset/n (simulated −1.10e-4 vs theory −1.00e-4). Rough size: ~10% of L̂² for DeiT3-L, ~27% for ResNet50. |
| (d) §3.1 stationarity identity | **CONFIRMED on the training set** | Own fit (bias, L2, label smoothing 0/0.1, seeds 100800–2): null block 4e-9–7e-8 (gradient-norm level), row block = λ‖W‖. On held-out draws the null block is 0.06–0.10, same order as the row block, so it does not carry over to val features at this n/p. It is the multinomial GLM score equation projected on ker W. |

### 1.2 Appendix B synthetic replication: **MATCHED**
The review's simulator was run verbatim under numpy 2.5.4 / scipy 1.18.1 / sklearn 1.9.1 (the review used
1.26.4 / 1.11.0 / 1.5.1). Part A identical to 3 dp. Part B: every max/min, every range and every gate (floor, C′ 1.01, C 1.64,
offset 1.00, derangement +0.0000 ± 0.0003, plants +0.0953 vs −0.0006, world B 119.6×) identical. Only Kendall τ of quantities
constant to float precision differ (e.g. θ_sem τ in the offset world −0.24 vs −0.13), i.e. tie-breaking noise. Part C identical
(CV 0.3% / 14.0% / 29.5%). Caveat (ideas K §8): the synthetic data are built so the estimator's assumptions hold.

### 1.3 Post hoc descriptive: ρ = ρ_offset + ρ_between + ρ_within (13 models; recomputed in `r8/run_r8.py` as a descriptive)
Exact plug-in identity (class weights n_c/N; error ≤ 4e-17). "Debiased" between subtracts the within-class noise carried by
10-image class means. Recomputed ρ matches `r2/results/fix2_models.csv` to ≤ 2.2e-9.

| Model | d_null | ρ | ρ_offset | ρ_between (plug-in) | ρ_between (debiased) | ρ_within | offset % of ρ | between % (plug-in / debiased) | within % |
|---|---|---|---|---|---|---|---|---|---|
| ResNeXt101 | 1048 | 0.4507 | 0.1492 | 0.2012 | 0.1901 | 0.1002 | 33.1 | 44.7 / 42.2 | 22.2 |
| VGG16 | 3096 | 0.3503 | 0.02878 | 0.1016 | 0.0772 | 0.2199 | 8.2 | 29.0 / 22.0 | 62.8 |
| DeiT3-L | 24 | 0.3485 | 0.3199 | 0.02235 | 0.02166 | 0.00625 | 91.8 | 6.4 / 6.2 | 1.8 |
| VGG19 | 3096 | 0.3386 | 0.02846 | 0.104 | 0.08113 | 0.2061 | 8.4 | 30.7 / 24.0 | 60.9 |
| DINOv2-L | 1048 | 0.3217 | 0.01194 | 0.1354 | 0.1161 | 0.1743 | 3.7 | 42.1 / 36.1 | 54.2 |
| Swin-L | 536 | 0.2744 | 0.02216 | 0.09271 | 0.07499 | 0.1595 | 8.1 | 33.8 / 27.3 | 58.1 |
| ResNet50 | 1048 | 0.1459 | 0.1244 | 0.005869 | 0.004133 | 0.01562 | 85.3 | 4.0 / 2.8 | 10.7 |
| EfficientNet | 280 | 0.1303 | 0.0008858 | 0.04213 | 0.03243 | 0.08731 | 0.7 | 32.3 / 24.9 | 67.0 |
| BiT-R50 | 1048 | 0.04641 | 0.001948 | 0.01407 | 0.01069 | 0.0304 | 4.2 | 30.3 / 23.0 | 65.5 |
| ConvNeXt | 24 | 0.01349 | 0.0004649 | 0.006258 | 0.005506 | 0.006765 | 3.4 | 46.4 / 40.8 | 50.2 |
| EVA02-L | 24 | 0.008885 | 0.0007625 | 0.003557 | 0.003049 | 0.004566 | 8.6 | 40.0 / 34.3 | 51.4 |
| DinoViT | 24 | 0.005351 | 0.0001386 | 0.001161 | 0.0007104 | 0.004051 | 2.6 | 21.7 / 13.3 | 75.7 |
| DenseNet121 | 27 | 1.831e-08 | 1.519e-08 | 9.11e-10 | 6.657e-10 | 2.208e-09 | 83.0 | 5.0 / 3.6 | 12.1 |

Reading (descriptive, post hoc, no verdict):
- **DeiT3-L:** 92% of ρ is a constant null mean; centred null energy is only 0.029 (debiased between share of the centred part
  0.76, matching round 5's η² 0.757). So ideas K's hint is right: DeiT3-L's large ρ and AS are mostly offset.
- **ResNet50 (SING's headline model):** 85% of ρ is offset as well; centred null energy is only 0.021 of total feature
  energy (debiased between part 0.004). Round 4's constant-mean surrogate already gave 1.39° vs AS 1.68° for ResNet50.
- ResNeXt101 33%, DenseNet121 83% (of a negligible ρ). All others ≤ 9%.
- So "null energy" in the paper is, for two of the five highest-AS models, mostly one constant vector per model. This is a
  descriptive fact about the existing analysis and should be reported (post hoc) whatever round 8 does.
- Not computed before the pre-registration: θ_sem, θ_cls, J_inn, shadow/innovation. Later the same day, still before the
  pre-registration, τ of centred ρ with SING (0.56) and centred-energy surrogates (AS_surr τ 0.67) were computed post hoc;
  see `PREREG_R8.md` §1.

## 2. Novelty check (Semantic Scholar + web)

| Idea | Verdict | Closest prior work |
|---|---|---|
| θ_sem (cross-fitted ridge text→class-mean map, null vs row share) | **PARTLY** | Ingredients standard: held-out-condition encoding models from semantic embeddings (Mitchell et al. 2008; Diedrichsen & Kriegeskorte, representational models, PLoS CB 2017; de Heer et al. 2017 variance partitioning); cross-validated inner products for unbiasedness (Walther et al. 2016; crossnobis, Diedrichsen et al. 2016 arXiv 1607.01371; PCM, Diedrichsen et al. 2018); predicting class means from text for held-out classes (Changpinyo et al. ICCV 2017 "Predicting visual exemplars"; arXiv 2311.14544 "Inferring latent class statistics from text"; ESZSL, Romera-Paredes & Torr 2015; DeViSE 2013); text↔feature linear maps (Text-to-Concept, Moayeri et al. ICML 2023; LiT 2022). The null/row variance split mirrors output-null vs output-potent analysis (Kaufman et al., Nat Neurosci 2014). Not found: a text-predictable null-vs-row class-signal *share* as a classifier null-leakage amount. |
| Shadow/innovation split, J(f) = J(s) + J(u) | **NOT NOVEL (math), PARTLY (use)** | Rao's test of additional information in discriminant analysis (Rao 1946/1948; U-statistic), multi-group redundancy (Fujikoshi 1982; McKay 1977): D²_full − D²_subset = information of the subset-residualised extra variables. Also redundancy/unique information in PID (Williams & Beer 2010). Applying it to explain the 69-idea ledger (amount scores see the shadow, conditional scores see the innovation) is new framing. |
| J_inn | **NOT NOVEL** | It is a Gaussian/Fisher version of conditional usable information beyond the logits (Hewitt et al. EMNLP 2021 conditional probing / V-information, already our round-6 score and P1) and of Rao's additional-information D². New only as an estimator choice. |
| Standard-addition AS | **PARTLY** | Standard addition is textbook analytical chemistry (incl. multivariate standard addition, Chemom. Intell. Lab. Syst. 2019); planted-signal calibration is already in our rounds 4–5 (world S, B-3a). Normalising SING's AS by each model's own measured slope to remove translator gain was not found. |
| Head-swap controls (NCM, shrinkage-LDA) | **PARTLY** | NCM heads (Mensink et al. TPAMI 2013), LDA, last-layer retraining (Kirichenko et al., DFR, arXiv 2204.02937, ICLR 2023) and NC3/NC4 self-duality (Papyan et al. PNAS 2020) are standard. Using alternative heads on the same backbone as a real-data calibration world for null-leakage scores was not found. |
| §3.1 stationarity identity | **NOT NOVEL** | First-order (score) condition of L2-regularised multinomial logistic regression, projected on ker W. Useful as a stated proposition, not as a contribution. |

Searches did not find θ_sem itself; "not found" is not proof of absence (S2 keyword search plus web, about 30 queries).

## 3. Steps

**CPU phase (all 13 models; cached private features, read only). Estimated a few CPU-hours.**
1. Freeze `PREREG_R8.md` (from the draft) and push it before any real-model θ_sem/θ_cls/J_inn/shadow number exists.
2. Rebuild the worlds (`r8/core_r8.WorldPieces`) on the three calibration bases (ResNet50, DINOv2-L, ResNeXt101): C′ (class-mean null
   energy fixed against E_row, centred; offset kept as a separate fixed component), offset-only, and the old world C reported as
   non-gating. Plus surrogates, derangement and plants (world S).
3. Synthetic dress rehearsal: unit tests (`r8/tests`) and the K0 coverage check (`r8/k0_r8.py`) before touching real features
   (kill rule K0 in the prereg; it also picks bootstrap vs half-sampling CIs).
4. θ_sem gates on the bases (§3 of the prereg), then θ_sem on 13 models and the confirmatory τ with SING (E12, S11).
5. Descriptives: ρ components (§1.3, already seen), θ_cls, θ_sem's composition × amount factorisation, shadow/innovation split,
   J_inn, a self-duality index (cosine of W rows with centred class means), Σ_w anisotropy.
6. Head-swap controls (NCM and shrinkage-LDA heads fit on one image half, evaluated on the other) on the three bases: AS and
   θ_sem per head; accuracies reported.
7. Standard-addition AS: not run in round 8 (deferred).

**GPU phase (not part of round 8; would need its own addendum, only if step 4 passes its gates).**
8. Power: features for 50–100 timm ImageNet classifiers on the same 10k subset (10 images/class), CLIP image embeddings
   already cached. Estimate: a few to ~10 GPU-hours on a T4/P100 class GPU (Exp 1 took 1.4 GPU-hours for its model set), dominated
   by large ViTs and data loading. Then CPU: ridge translators, AS, AS_surr, ρ components, θ_sem; model AS on content-free energy
   (ρ_offset + ρ_within) and θ_sem across ~100 models. Separate pre-registration addendum before extraction.
9. Input-space semantic blindness (ideas K §5.7): PGD under a logit-preservation penalty, judged by a held-out CLIP/SigLIP;
   13 models × a few hundred images ≈ a few GPU-hours. Exploratory only; not in the confirmatory set.

## 4. Paper changes (whatever the outcome)
- Report §1.3 (offset share) as post hoc; say L and ρ are uncentred; add the §2.1 impossibility argument as a short proposition
  (with the oracle check), and the stationarity / Fisher identities as propositions with the synthetic check.
- Replace τ-based "content-diagnostic" with max/min or slope CI for new rows; old rows stay as registered.
- Correct the M1 sentence if it ever claims offset invariance (it currently does not).

## 5. Risks / honest caveats
- Changing world C to C′ is a change of target. It is defensible only because T1 is stated first; reviewers may still read it as
  moving the goalposts. Old world C stays in every table.
- With 12/11 models, τ CIs are wide; an "in between" outcome is likely without the GPU power phase.
- θ_sem counts shadow content (redundant class signal); a high θ_sem does not mean usable hidden semantics.
- The large offset share means θ_sem (centred) and AS (uncentred) may disagree for reasons that are purely the offset; this is
  informative but must be read with §1.3 in view.
- Text references: SING's prompt-averaged CLIP ViT-L/14 class texts; the same family as SING's translator target, so CLIP-specific
  structure is shared between θ_sem and AS (not independent evidence of "semantics" in a model-agnostic sense).
