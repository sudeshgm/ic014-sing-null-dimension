# E1: energy vs dimension
1. GPU pass (`extract_null_gpu.py`): the same 10k subset as Exp 1, on 5 models (resnet, resnext, dinov2, eva02_large, vgg16). For each model it writes `<out>/<model>_nullfeat.npz`, holding the null coordinates, T(f), ||f||², labels, V_null and A·V_null. These files are derived from ImageNet and stay private (gitignored), so do not commit or publish them.
2. CPU analysis (`analyze_e1.py <feats_dir> <out_dir>`): takes energy PCs of the null coordinates (uncentered second moment) and then runs:
   * (a) fixed d ∈ {24, 100, 280}: high / low / random (20 seeds);
   * (b) fixed ρ ∈ {0.5%, 2%, 8%} at d ∈ {24, 100, 280, 536}. Each subspace is built from top-k PCs, bottom PCs and one rotated vector, which lets ρ be hit exactly. Infeasible cells are marked.

   Outputs:
   * `e1_cells.csv`
   * `e1_fit.json`: pooled log|AS| = a·log ρ + b·log d + model FE, with a 95% CI from a 1000× image bootstrap.
   * `e1_ratios.json`: AS(high)/AS(low) per d, and max/min AS across d per ρ.

   Because the translators are linear, the computation is exact: T(f − V U Uᵀ c) = z − (cU)(AVn U)ᵀ.

EVA02-L (null dim 24) only gets the full-null cell at d=24; all its other (a) and (b) cells are infeasible.
We smoke-tested on 6 fake images. All 5 models extract, and every feasible (b) cell hits its target ρ to within 1e-6.
