# Exp 0: CPU re-analysis of SING Supp. Fig. 6b

* `digitize_fig6b.py`: pixel digitization of Fig. 6b (supplement p.5 rendered at 800 dpi with `pdftoppm -r 800`, crop `fig6b_800dpi.png`). Axis calibrated on the 0/20/40/60 gridlines (linear fit, max residual 0.33). Reading error ±0.38 ratio units = 2.5 px edge/antialias uncertainty plus calibration residual, in quadrature. Output: `fig6b_digitized.{csv,json}`.
* Limit: Fig. 6b plots only the *image-to-attribute ratio* (IS/AS), with 13 bars (the caption says "12 models"). It shows neither IS nor AS on their own, so AS-vs-null-dim cannot be computed from it. Main-paper Fig. 3 gives IS and AS for 5 models only.
* `null_dims.py`: m − rank(W) with SVD at tol 1e-5 (as in `sing/core/projectors.py`). 1×1-conv heads are flattened, as in SING's wrapper. Output: `null_dims.{csv,json}`. `densenet_resnet_svcheck.py` → `sv_tolerance_check.json`.
* `stats.py` (seed 0, 10k permutations, two-sided): Spearman/Kendall of IS/AS vs null dim; all 13, without DinoViT, and leave-one-out. `stats.{csv,json}` uses the measured null dims; `stats_densenet24.{csv,json}` uses the plan's DenseNet = 24.
* Boring baseline (null-energy fraction ρ): not run. It needs ImageNet features from all 13 models, i.e. forward passes on ImageNet-val. There is no ImageNet on this box, and it is GPU-class work.
