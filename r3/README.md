# Round-3 analyses
Criteria: `../PREREG_R3.md` (committed before any round-3 number was computed). Plan and review mapping: `../ROUND3_PLAN.md`. CPU only.
Model bootstrap: 2000×, seed 0, percentile and BCa CIs. Sets: all 13; E12 (ρ ≥ 0.005, drops DenseNet); S11 (round-2 screened set, drops DenseNet and BiT).

| Script | Prereg | Inputs | Output (`results/`) | Time (8 cores) |
|---|---|---|---|---|
| `equal_energy.py` | A, H (equal-energy A1–A4, κ, γ, NuSA-style norm) | private features, SING | `A_equal_energy.json` | ~4 min |
| `leakfree.py` | B (N4 leak-free test, B-rank) | private features, SING | `B_leakfree.json` | B, C and G together ~3.5 min |
| `pc_bands_r3.py` | C (PC bands vs ρ^â) | private features, SING | `C_pc_bands_r3.json` | (with B) |
| `fix14_floor.py` | D (chance floor for the DinoViT check) | private fix-14 features | `D_fix14_floor.json` | seconds |
| `../r2/e2_claims_ci.py` (revised) | E (BCa claim CIs, de-duplicated rows) | private per-image arrays | `../r2/results/fix8_claims_ci.{csv,json}` | under 1 min |
| `paired_stats.py` | F (paired bootstrap, partial τ, like-for-like ratio) | committed CSVs | `F_paired_stats.json` | seconds |
| `retrain_translators.py` | G (ridge translators, train/test halves) | private features, CLIP image embeddings | `G_retrain.json`, `G_retrain_permodel.json` | (with B) |
| `retrain_lambda_posthoc.py` | **post hoc** (extended ridge grid 10^3–10^6; the registered grid put every model at its edge) | as G | `G_lambda_posthoc.json`, `G_lambda_posthoc_permodel.json` | ~2 min |
| `make_tables.py` | N8 | committed results | `../paper_workshop/tables/*.tex` | ~1 s |
| `../r2/e9_holdout.py --fast` | N10 | private features, SING | `../r2/results/fix10_e9_holdout_fast.json` | see below |

`common.py` holds the shared loaders, Kendall τ-b and the model bootstrap.

## Outcomes
* **A (primary): ENERGY-ACCOUNTS.** τ(A1) = −0.03 [−0.54, 0.61] on E12 and 0.09 [−0.45, 0.78] on S11; the paired drop τ(AS_SING) − τ(A1) has CI [0.23, 1.39] (E12) and [0.08, 1.30] (S11). No KILL-R3. A2/A3/A1′ do not reach 0.6. A4 (matched CLIP energy) −0.15 / −0.20. κ − ρ paired CI [−0.41, 0.28] covers 0, so ρ stays the named variable; γ = κ/ρ does not track SING (−0.06).
* **B: N4 passes on all 3 models** (Spearman 1; log-log slopes 0.38–0.69, CIs exclude 0). "Confound" is allowed only in the qualified sense (AS responds to null energy that has no image-specific content). Planted ratio at ρ_t = 0.2: ×1.04–1.09 (α = 0.1), ×1.35–1.51 (α = 0.3). B-rank: τ(AS_surr) = 0.69 (all 13), 0.64 (S11): leak-free null content reproduces SING's order. No claim from X = AS_SING/AS_surr (τ 0.54 [0.14, 0.83]).
* **C:** at â = 0.431, d = 24: ResNet50 1.50, ResNeXt101 0.74, DINOv2-L 0.98 [0.97, 0.99] → "top PCs carry less AS than energy predicts" (2/3), **not robust** (at â = 0.388 the label is "more"). Round-2 label kept verbatim: "top PCs carry less AS per unit energy", 2/3 screened.
* **D:** non-matching candidates sit at their shuffled floor (|cos − shuffled| ≤ 0.002, retrieval ≤ 0.2 %); the used timm AugReg checkpoint 0.848 vs floor 0.575, retrieval 78.8 %. The 84.0 % vs 85.85 % top-1 gap is unexplained.
* **E:** N6 done (C1/C11 and C3/C6-mean merged). N7 **fails** with one flag: AS_fair C6 SD point −0.203 outside BCa [−0.2642, −0.2641] (degenerate bootstrap); reported, rule not changed. Tally under AS_SING: 7 hold, 3 reverse (C4, C8, C10), 1 contradicted (C9).
* **F:** τ(ρ) = τ(AS_SING) = 0.82; paired CI [−0.36, 0.29] → "indistinguishable". Partial Kendall 0.61 both ways; partial Spearman 0.63 (AS | ρ), 0.72 (ρ | AS). Like-for-like τ = 0.744 [0.33, 1.00] (BCa [0.12, 0.94]); SING/ours ratio 0.98–3.35.
* **G:** retrained fidelity 0.79–0.84 (released 0.81–0.87 on the same test half). τ(AS_SING retrained) 0.85 / 0.82 / 0.82 (all 13 / E12 / S11): "order survives retraining". A1 with retrained translators 0.61 (E12), 0.75 (S11): equalising energy removes the agreement only with the released translators.
* **G post hoc:** with λ on the extended grid, A1 = 0.49 / 0.39 / 0.53; λ = 10^6 for every model gives ≈ 0.79–0.82. Only VGG16/19 and DenseNet change λ; the result is fragile.
* **N10:** see `../README.md` (timed run and comparison with the full run).
