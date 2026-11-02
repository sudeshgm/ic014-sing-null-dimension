# GPU job: fix 14, DinoViT translator fidelity across ViT-L candidates

**Why:** internal reviewer round 1, fix 14 (`paper_workshop/reviews/review_20261101_v1.md`). It needs ImageNet-val images, which are not on the CPU box. CPU-only work is not possible here.

**Pre-registered pass (PREREG_R2.md, fix 14):** the checkpoint we used (`timm vit_large_patch16_224.augreg_in21k_ft_in1k`, SING's registry entry) has the highest mean cos(T(f), CLIP_img) under SING's released `dinovit1` translator, by a margin of at least 0.05. The result is reported either way.

## Step 1 (GPU box with ImageNet val, e.g. GPU runner's Kaggle copy)
```
pip install -r exp0b/requirements.txt          # torch, torchvision, timm, transformers, pillow
python gpu_jobs/fix14_extract_vitl.py \
  --val $IMNET/ILSVRC/Data/CLS-LOC/val \
  --subset <IC014_FEATS>/e3/ret/e3_feats/subset_files.txt \
  --out fix14_feats
```
- **Script:** `gpu_jobs/fix14_extract_vitl.py`. It reuses `exp0b/eval_top1.py:build()`, so the checkpoints and native transforms are identical to Exp 0b.
- **Inputs:**
  - The first 1000 lines of `subset_files.txt` (ImageNet-val file names).
  - The 5 candidates in `exp0b/eval_top1.py:CANDS`: torchvision `IMAGENET1K_V1`, `SWAG_LINEAR_V1` and `SWAG_E2E_V1`; HF `google/vit-large-patch16-224`; and timm augreg as the in-job control.
- **Outputs (private, ImageNet-derived, do not commit):**
  - `fix14_feats/<candidate>.npz` with keys `F` (1000×1024 pre-head CLS features), `pred` (argmax) and `files`.
  - `fix14_feats/meta.json`.
  - Total output is about 25 MB.
- **Expected runtime:**
  - About 1–2 min per candidate on one T4/P100; SWAG-E2E runs at 512 px and takes about 4 min.
  - Under 15 min in total, including weight downloads (about 1.2 GB each).
  - On 8 CPU cores, expect roughly 1–1.5 h.

## Step 2 (CPU, this repo)
```
python r2/fix14_fidelity.py --feats-root $IC014_FEATS --cand-dir fix14_feats
```
- **Output:** `r2/results/fix14_fidelity.json`, containing the mean cos per candidate and `_pass`.
- **Reference:** the cached augreg features give 0.8482 on these first 1000 images and 0.8482 on the full 10k, using SING's registry transform. The in-job timm row checks that the native transform reproduces this value.

## Status
Done 2026-11-02. Step 1 ran as Kaggle kaggle-user/ic014-fix14 v1 (2 × T4; about 1.9 h including downloads); step 2 ran on CPU. Features stay private and are not in the repo.
Result (`r2/results/fix14_fidelity.json`): timm AugReg 0.848 (equal to the cached reference, and the re-extracted features match the cached ones exactly), torchvision IMAGENET1K_V1 0.677, HF 0.631, SWAG linear 0.615, SWAG E2E 0.608. Margin 0.171 ≥ 0.05: **pass**.
