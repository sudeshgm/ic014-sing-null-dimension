# Exp 1: within-model AS(d) curve (plan section 3)
GPU pass (`run_exp1_gpu.py`). It uses a 10k ImageNet-val subset (the first 10 file names per class, sorted) and the 13 SING models, loaded through SING's own `load_model`, together with SING's linear translators and its released CLIP text embeddings. Inside each model's own null space (SING projectors, tol 1e-5), it draws random d-dim subspaces by QR of a Gaussian with generator seed 1000·d+s, for d in {24,48,100,280,536,1048}, using 20 seeds whenever d < null dim. The full null space is also removed for every model, and DenseNet121 (null dim 27) gets an extra 20 random 24-dim subspaces. For each (model, d, seed) the script records mean|AS|, AS, IS, rho_d and per-prompt |AS|, along with the max logit change as a check.
* AS text target: the unit-normalized mean of SING's 6 prompt embeddings for the ground-truth class. SING does not ship the prompt texts, so we store per-prompt |AS| too.
* Outputs: `exp1_per_seed.json`, `exp1_full_null_per_image.npz`, and `subset_files.txt` (file names and labels only). No images. Raw features are written only with `--save-features`, and those stay local.

CPU analysis (`analyze_exp1.py <per_seed.json> <outdir>`):
* `exp1_curves.csv`: mean/SD over seeds per (model, d).
* `exp1_fits.json`: linear fits of |AS| vs d, vs rho_d, and vs log d; r = AS(full 1048)/AS(24), with a 2σ band from the d=24 seed SD.
* `exp1_d24_ranking.json`: the d=24 ranking over 8 models and its Kendall τ against the SING Fig. 6b order. SING published no AS table for these 8 models, so we use the Fig. 6b IS/AS order as the reference. The file also holds the boring baseline (Spearman of rho_full vs |AS|_full and vs the Fig. 6b ratio, 13 models).

We smoke-tested on CPU with 6 fake images and all 13 models. All 101/21/1 conditions ran, with a max logit change of 5e-6.
