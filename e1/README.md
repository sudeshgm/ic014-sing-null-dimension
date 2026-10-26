# E1: energy vs dimension
1. **GPU** `extract_null_gpu.py`: same 10k subset as Exp 1, 5 models (resnet, resnext, dinov2, eva02_large, vgg16). Writes `<out>/<model>_nullfeat.npz` with the null coordinates, T(f), ||f||², labels, V_null and A·V_null. **These files are private and ImageNet-derived: never commit or publish them** (they're gitignored).
2. **CPU** `analyze_e1.py <feats_dir> <out_dir>`: energy PCs of the null coordinates (uncentered second moment), then:
   * (a) fixed d ∈ {24, 100, 280}: high / low / random (20 seeds);
   * (b) fixed ρ ∈ {0.5%, 2%, 8%} at d ∈ {24, 100, 280, 536}, built from top-k PCs + bottom PCs + one rotated vector, so ρ is hit exactly. Infeasible cells are marked.

   Outputs:
   * `e1_cells.csv`
   * `e1_fit.json`: pooled log|AS| = a·log ρ + b·log d + model FE, 95% CI from a 1000× image bootstrap.
   * `e1_ratios.json`: AS(high)/AS(low) per d, and max/min AS across d per ρ.

   This is exact because the translators are linear: T(f − V U Uᵀ c) = z − (cU)(AVn U)ᵀ.

EVA02-L (null dim 24) only gets the full-null cell at d=24. Every other (a) and (b) cell is infeasible for it.
Smoke-tested on 6 fake images: all 5 models extract, and every feasible (b) cell hits its target ρ to 1e-6.
