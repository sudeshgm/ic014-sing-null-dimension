# POST HOC: the constant null-mean offset and the energy headline

**Post hoc, descriptive, not pre-registered.** These analyses were run after the ρ decomposition had been seen and before
`r8/PREREG_R8.md` was frozen (09d6e48); the pre-registration lists them as known (§1). They change no pre-registered verdict.
CPU only; no θ_sem, θ_cls or J_inn is computed here.

Scripts (features via `IC014_FEATS`, SING checkout via `SING_ROOT`, default `_ext/sing`):
- `rho_decomp.py` → `rho_decomp.json`: exact uncentred plug-in identity ρ = ρ_offset + ρ_between + ρ_within per model
  (ρ_offset = ‖mean null coordinate‖² / E‖f‖²; centred ρ = ρ_between + ρ_within).
- `offset_tau.py` → `offset_tau.json`: Kendall τ_b of each component with SING's order (`r3/common.tau_ci`, B = 2000, seed 0;
  sets all 13, E12, S11 as in `r4/common4.sets`), with paired differences on the same resamples.
- `offset_surr.py` → `offset_surr.json`: SING AS (released translators) with the null part replaced by content-free surrogates at
  the centred energy (zero-mean Gaussian N(0, Σ), and the centred null coordinates permuted across images; r4 seeds 92000+r,
  3 draws), by the offset alone, and by the real null content with the offset removed.
- `offset_surr_tau.py` → `offset_surr_tau.json`: τ of those variants with SING's order.

## Results

τ with SING's order (percentile 95% CI):

| Score | all 13 | E12 | S11 |
|---|---|---|---|
| ρ, uncentred (paper) | 0.82 [0.56, 1.00] | 0.79 | 0.78 |
| ρ centred = between + within | 0.56 [0.11, 0.89] | 0.48 [−0.02, 0.86] | 0.49 [−0.08, 0.91] |
| ρ_offset | 0.74 | 0.70 | 0.71 |
| ρ_between | 0.72 | 0.67 | 0.67 |
| ρ_within | 0.49 | 0.39 | 0.38 |
| AS, zero-mean Gaussian surrogate at centred ρ | 0.67 [0.38, 0.92] | 0.61 | 0.64 |
| AS, centred null coordinates permuted across images | 0.67 | 0.61 | 0.64 |
| AS, real null content with the offset removed | 0.74 | 0.70 | 0.75 |
| AS, offset only | 0.56 | 0.48 | 0.49 |

- Paired τ(centred ρ) − τ(ρ), all 13: −0.26, percentile [−0.58, 0.00], BCa [−0.66, −0.03].
- Paired τ(zero-mean Gaussian at centred ρ) − τ(paper Gaussian surrogate at full ρ, 0.69), all 13: −0.03 [−0.31, 0.31].
- The offset is 92% of ρ for DeiT3-L (0.348 → centred 0.029), 85% for ResNet50 (0.146 → 0.021), 83% for DenseNet121, 33% for
  ResNeXt101 and at most 9% for the other nine models. DeiT3-L, SING's rank 1, is 3rd by uncentred ρ and 8th by centred ρ;
  ResNet50 moves from 7th to 9th.
- AS with the offset alone, as a share of AS (AS is an angle and does not add over components): DeiT3-L 4.18° of 5.05° (83%),
  ResNet50 1.38° of 1.68° (83%); EVA02-L 55%, DenseNet121 87% (AS negligible), ResNeXt101 14%, others 5–25%.

## Reading
- "ρ alone reproduces SING's order" holds for uncentred ρ only; centred ρ gives 0.56 (0.48 / 0.49 on E12 / S11, CIs include 0).
- Content-free null energy at the centred level, passed through SING's released translators, still gives τ 0.61–0.67, so the
  broader energy reading does not depend on the offset.
- For DeiT3-L and ResNet50 most of SING's AS comes from one constant null vector per model, which carries no image or class content.
