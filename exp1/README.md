# Exp 1: within-model AS(d) curve (plan section 3)
GPU pass (`run_exp1_gpu.py`): 10k ImageNet-val subset (the first 10 file names per class, sorted), 13 SING models loaded through SING's own `load_model`, SING's linear translators, and SING's released CLIP text embeddings. Inside each model's own null space (SING projectors, tol 1e-5), random d-dim subspaces are drawn by QR of a Gaussian, generator seed 1000·d+s, for d in {24,48,100,280,536,1048} with 20 seeds for d < null dim; the full null space is also removed for every model; DenseNet121 (null dim 27) additionally gets 20 random 24-dim subspaces. Records mean|AS|, AS, IS, rho_d and per-prompt |AS| per (model, d, seed), plus the max logit change as a check.
* AS text target: the unit-normalized mean of SING's 6 prompt embeddings for the ground-truth class. The prompt texts are not shipped, so per-prompt |AS| is also stored.
* Outputs: `exp1_per_seed.json`, `exp1_full_null_per_image.npz`, and `subset_files.txt` (file names and labels only). No images. Raw features only with `--save-features`; those stay local.

CPU analysis (`analyze_exp1.py <per_seed.json> <outdir>`):
* `exp1_curves.csv`: mean/SD over seeds per (model, d).
* `exp1_fits.json`: linear fits of |AS| vs d, vs rho_d, and vs log d; r = AS(full 1048)/AS(24), with a 2σ band from the d=24 seed SD.
* `exp1_d24_ranking.json`: d=24 ranking over 8 models and Kendall τ vs the SING Fig. 6b order. There is no published AS table for these 8 models, so the reference is the Fig. 6b IS/AS order. Also has the boring baseline (Spearman of rho_full vs |AS|_full and vs the Fig. 6b ratio, 13 models).

Smoke-tested on CPU with 6 fake images, all 13 models. All 101/21/1 conditions ran; max logit change 5e-6.
