# E1 results (raw, no interpretation)

Run: `python e1/analyze_e1.py <private e1_feats> e1/results --sing-root /tmp/sing` (SING e47b37d).
Features come from ImageNet and stay private. They are not in this repo. 5 models, n=10000 images each (meta.json; subset_files.txt has 10000 lines).

Files:
- `results/e1_cells.csv`: one row per cell, with |AS|, IS and rho (means and SDs over seeds) and the infeasible cells marked
- `results/e1_fit.json`: the pooled fit
- `results/e1_ratios.json`: the ratios

Pooled fit, log|AS| = a log rho + b log d + model FE (bootstrap over 1000 images, seed 0):
- a = 0.4312 [0.4296, 0.4328]
- b = 0.0101 [0.0081, 0.0121]
- R² = 0.932, 76 cells, corr(log rho, log d) = 0.325, no zero-energy cells excluded

EVA02-L (null dim 24) is not evaluable. Only its d=24 full-null cell is reported: |AS| = 0.417, IS = 9.02, rho = 0.00889.
Fixed-rho cells are marked infeasible when the target rho is outside what the top-to-bottom PC range can reach at that d.

Sanity checks:
- The fixed-rho cells hit their targets exactly.
- The random-cell rho scales about linearly with d (ResNet: 0.00334 / 0.0138 / 0.0386 at d = 24 / 100 / 280).
