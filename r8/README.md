# IC014 round 8: θ_sem, a target-first T1 score (results)

Pre-registration `PREREG_R8.md`, frozen with the code and tests in commit **09d6e48**. That commit was pushed before any real-model
θ_sem was computed. Run: `python r8/run_r8.py --feats-root <cached features>` (CPU, B = 2000, seeds from 100800).
Outputs:
- `results/R8_verdicts.json`: all gates, confirmatory τ and descriptive τ;
- `results/permodel/*.json`: every per-model and per-world number;
- `results/R8_summary.json`: compact table;
- `results/K0_coverage.json`: K0;
- `results/unit_tests_prereg.log`: 11 unit tests, synthetic only, passing at freeze.

## Bottom line

**θ_sem is NOT VALIDATED for T1.** Two of the eight pre-registered gates failed: G-noise (world C′) and G-derange. By the
pre-registered rule the confirmatory τ is reported but **not read**.

For the record, the raw τ is 0.64 on both E12 and S11, with percentile CI lower bounds above 0. Had the gates passed, the raw rule would have
given TRACKS. We make no claim from this.

No kill: K0 did not fire. Class bootstrap coverage was 0.97 (signal scenario) and 0.99 (zero scenario), so the 'boot' method was used. Half-sampling
coverage was 0.98 / 0.99.

## Gates

| Gate | Result | Numbers |
|---|---|---|
| K0 coverage | pass → 'boot' | boot 0.97 / 0.99; half 0.98 / 0.99 (bar [0.90, 0.99]) |
| 1 G-floor | **pass** | CI includes 0 in 98/98 cells; max \|θ_surr\| 0.0026 ≤ 0.040 (0.1 × max real θ_sem 0.401) |
| 2 G-noise (C′) | **FAIL** | estimable rows: ResNet50 3/12, DINOv2-L 7/12, ResNeXt101 7/12 (need ≥ 8). max/min 1.01 / 1.12 / 1.25 (≤ 1.5 OK). Log–log slope CIs: ResNet50 [−0.092, 0.063] (OK), DINOv2-L [−0.041, −0.004], ResNeXt101 [+0.036, +0.067] (exclude 0) |
| 3 G-offset/scale/rotation + offset world | **pass** | max relative error 4e-16 (13 models); offset world deviation 0 on all 3 bases |
| 4 G-derange | **FAIL** | 5/6 cells pass. ResNet50 real: mean +1.1e-4, CI [3.5e-5, 1.9e-4] excludes 0. Magnitude 0.0003 × max θ_sem, so the 0.1× size bar holds; only the CI criterion fails |
| 5 G-S plants | **pass** | Δ_sem +0.37 / +0.35 / +0.29 (CI lower bounds > 0); Δ_arb −0.0004 / −0.0002 / −0.0002 (CIs include 0) |
| 6 G-B closed form | **pass** | max relative error 4e-16 over 36 rows |
| 7 G-tol | **pass** | τ = 1.00 at both 1e-4 and 1e-6 |
| 8 Estimability | **pass** | 13/13 (SE ≤ 0.040) |

**Reading of the failures** (descriptive; none of this changes the verdict):
- **G-noise.** C′ fixes class-mean null energy at h = ρ_min/2 of the row yardstick, so its class content is tiny: θ_sem ≈ 0.001 against real
  values up to 0.40. At the high-ρ rows the noise SE overtakes that, so most rows are not estimable (the bar was SE ≤ 0.5 θ). Where rows are
  estimable, θ_sem is nearly flat (max/min ≤ 1.25), but the DINOv2-L and ResNeXt101 slopes are small and significantly non-zero (−0.02 and
  +0.05 per log-unit). We did not foresee that the C′ content level makes this gate low-powered. We do not re-design it post hoc.
- **G-derange.** One cell has a tiny but significantly positive mean. Its 20 draws are very tight (sd 1.7e-4).

The old world C (non-gating) is flat for θ_sem as well:
- ResNet50: 0.0002 → −0.0006, all within about 1 SE.
- DINOv2-L: 0.0013 → 0.0020 (×1.5).
- ResNeXt101: 0.0012 → 0.0027 (×2.2).

World M rises about 200–300× and world D about 20–30×, in line with T1.

## Confirmatory τ (reported, not read: gates failed)

| Set | n | τ(θ_sem, SING) | percentile CI | BCa CI |
|---|---|---|---|---|
| E12 | 12 | 0.636 | [0.186, 0.932] | [0.100, 0.901] |
| S11 | 11 | 0.636 | [0.130, 0.960] | [0.042, 0.918] |

Sensitivity fold seed 100801: τ is identical (0.636 / 0.636); θ_sem per model changes by ≤ 2%.

## Per model

ρ components are plug-in. "comp" = S(Z)/Ê_b(Z), the composition factor; the amount factor is Ê_b(Z). Self-duality is the mean cosine of W's rows
with the centred class means.

| Model | θ_sem [95% CI] | θ_cls | comp | ρ | ρ_off | ρ_between | shadow | innovation | self-duality | J_inn |
|---|---|---|---|---|---|---|---|---|---|---|
| DeiT3-L | 0.401 [0.396, 0.429] | 0.129 | 0.700 | 0.348 | 0.320 | 0.022 | 0.208 | 0.176 | 0.641 | 77.6 |
| ResNeXt101 | 0.212 [0.201, 0.217] | 0.346 | 0.107 | 0.451 | 0.149 | 0.201 | 0.179 | 0.005 | 0.419 | 59.1 |
| DINOv2-L | 0.093 [0.083, 0.098] | 0.266 | 0.061 | 0.322 | 0.012 | 0.135 | 0.094 | 0.054 | 0.457 | 39.5 |
| VGG19 | 0.109 [0.104, 0.110] | 0.225 | 0.162 | 0.339 | 0.028 | 0.104 | 0.085 | −0.001 | 0.551 | 13.9 |
| VGG16 | 0.113 [0.108, 0.115] | 0.226 | 0.171 | 0.350 | 0.029 | 0.102 | 0.088 | −0.001 | 0.545 | 11.8 |
| ResNet50 | 0.0128 [0.0122, 0.0133] | 0.020 | 0.251 | 0.146 | 0.124 | 0.006 | 0.011 | 0.001 | 0.738 | 9.9 |
| Swin-L | 0.287 [0.287, 0.304] | 0.209 | 0.225 | 0.274 | 0.022 | 0.093 | 0.112 | 0.188 | 0.597 | 24.6 |
| EfficientNet | 0.092 [0.089, 0.095] | 0.103 | 0.195 | 0.130 | 0.001 | 0.042 | 0.055 | 0.028 | 0.569 | 16.9 |
| ConvNeXt | 0.0147 [0.0141, 0.0167] | 0.011 | 0.195 | 0.013 | 0.0005 | 0.006 | 0.007 | 0.005 | 0.658 | 19.8 |
| BiT-R50 | 0.033 [0.032, 0.035] | 0.041 | 0.264 | 0.046 | 0.002 | 0.014 | 0.024 | 0.006 | 0.718 | 20.1 |
| EVA02-L | 0.0044 [0.0037, 0.0055] | 0.005 | 0.074 | 0.009 | 0.0008 | 0.004 | 0.002 | 0.003 | 0.777 | 21.7 |
| DinoViT | 0.0006 [0.0005, 0.0007] | 0.001 | 0.155 | 0.005 | 0.0001 | 0.001 | 0.0005 | 0.0004 | 0.792 | 6.4 |
| DenseNet121 | 2.5e-9 | 3.4e-9 | 0.291 | 1.8e-8 | 1.5e-8 | 9e-10 | 2.4e-9 | 9e-11 | 0.700 | 3.1 |

Rows are ordered by SING's order (DeiT3-L first). The percentile CIs are often asymmetric around the point (e.g. Swin-L, DeiT3-L), which is the
small-sample bias of a ratio estimator. K0 coverage was nonetheless 0.97–0.99.

## Descriptives (no verdicts)

τ with SING (percentile CI) on E12 / S11:
- θ_cls: 0.70 [0.36, 1.00] / 0.71 [0.29, 1.00].
- Shadow θ_sem: 0.76 [0.40, 1.00] / 0.78 [0.39, 1.00].
- Innovation θ_sem: 0.21 [−0.29, 0.67] / 0.24 [−0.27, 0.70].
- J_inn: 0.36 [−0.26, 0.83] / 0.45 [−0.18, 0.88].
- Self-duality index: **−0.64 [−1.00, −0.22] / −0.60 [−0.96, −0.12]**. Models whose head is *less* aligned with its class means rank higher in SING.

Other associations:
- τ(θ_sem, ρ) = 0.73 / 0.71; τ(θ_sem, ρ_between) = 0.61 / 0.53.
- Partial association of θ_sem with SING given the content-free energy ρ_offset + ρ_within: r = −0.41 [−0.85, 0.18] / −0.43 [−0.88, 0.41].
  So, given content-free energy, θ_sem adds no positive association with SING's order.

Composition × amount:
- DeiT3-L's top θ_sem comes from composition (S(Z)/Ê_b(Z) = 0.70, the highest of the 13), not amount. Its centred between-class null energy is small (ρ_between 0.022).
- ResNeXt101 and DINOv2-L are the opposite case: large amount, low composition (0.11, 0.06).
- The constant offset that dominates DeiT3-L's and ResNet50's ρ does not enter θ_sem (gate 3, exact).

Shadow/innovation:
- For most models θ_sem is mostly shadow, i.e. null class signal linearly predictable from the row space.
- Swin-L (innovation 0.19 > shadow 0.11), DeiT3-L (0.18 vs 0.21) and EVA02-L carry a large innovation share.
- The innovation order does not track SING (above).

Head-swap (3 bases; fit on images 0–4, top-1 on 5–9; all heads have d_null 1048):

| Base | head | ρ | AS (°) | top-1 | θ_sem [CI] |
|---|---|---|---|---|---|
| ResNet50 | released | 0.146 | 1.68 | — | 0.013 [0.012, 0.013] |
| | NCM | 0.014 | 0.34 | 0.652 | 0.0002 [0.0002, 0.0002] |
| | LDA | 0.930 | 2.56 | 0.635 | 0.930 [0.928, 0.932] |
| DINOv2-L | released | 0.322 | 1.50 | — | 0.093 [0.083, 0.098] |
| | NCM | 0.107 | 0.48 | 0.745 | 0.001 [0.001, 0.002] |
| | LDA | 0.436 | 1.42 | 0.752 | 0.378 [0.373, 0.387] |
| ResNeXt101 | released | 0.451 | 2.00 | — | 0.212 [0.201, 0.217] |
| | NCM | 0.036 | 0.38 | 0.773 | 0.0002 [0.0001, 0.0003] |
| | LDA | 0.662 | 2.46 | 0.779 | 0.676 [0.667, 0.692] |

At near-equal accuracy, the head choice moves ρ, AS and θ_sem by orders of magnitude. The NCM head spans the class means, so almost no class
signal is left in its null space. The shrinkage-LDA head leaves most of it there.

Within-class participation ratio (null / row) is in `R8_summary.json`. DeiT3-L 9.9 / 323 and DenseNet121 11 / 228 have very anisotropic null blocks.

## Deviations and notes

- No deviation from `PREREG_R8.md` in the code that was run; the code is unchanged since 09d6e48.
- The pipeline was smoke-tested on fake random features before the freeze (disclosed in the prereg §1).
- A `ConstantInputWarning` from `partial_ci` arises in a few bootstrap draws: 17 of 2000 dropped for E12, 22 for S11 (`B_used`).
- The ρ decomposition and the offset/centring results (prereg §1) were known before the freeze and are post hoc.
- Standard-addition AS was not run (as registered).
- Round 8 does not change any earlier verdict.
