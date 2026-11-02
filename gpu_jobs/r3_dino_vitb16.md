# GPU job: a true DINO backbone with a linear head (round 3, not run)

**Why:** the second review (TMLR item 3, fix 8) asks for one genuine DINO model, so the self-supervision question that SING's "DinoViT" label raises can be tested on a model that is actually DINO. SING's own "DinoViT" translator matches a supervised timm AugReg ViT-L/16 (`r3/results/D_fix14_floor.json`). This job needs ImageNet images and a GPU, so it was not run in round 3 (CPU only). It is not pre-registered; if it is run, its reading rule must be written into a new pre-registration first.

## Models
- `dino_vitb16` backbone with the official DINO linear-evaluation head (`facebookresearch/dino`, `dino_vitb16_linearweights.pth`, reported 78.2 % top-1). Head input as in the DINO linear eval: last-block CLS token concatenated with the average patch token (2 × 768 = 1536-d). Check the dimension against the head's weight shape before running.
- Second DINO model: `dino_vits16` with its official linear head (reported 77.0 %); head input = CLS tokens of the last 4 blocks (4 × 384 = 1536-d).
- Optional supervised control at the same size: timm `vit_base_patch16_224.augreg_in21k_ft_in1k`.

## Step 1 (GPU box with ImageNet-val)
1. `pip install -r exp0b/requirements.txt` (torch, torchvision, timm, transformers, pillow).
2. For each model, extract the head-input features for the 10k subset (`scripts/subset_files_10k.txt`) and the 50k top-1 (native DINO eval transform: resize 256, centre crop 224).
3. Save `F` (n×m float32), `W`, `b` (the linear head), `pred`, `files` per model to the private feature store (never to the repo).
4. Also extract CLIP ViT-L/14 image embeddings for the same 10k images if they are not already cached (`clip_img.npz` exists for the 10k subset; reuse it).

Expected runtime: about 10 min per backbone on one T4 for the 10k subset; about 30 min per backbone for the 50k top-1. Under 2 GPU-hours in total including downloads.

## Step 2 (CPU, this repo)
SING has no released translator for these models, so they cannot be placed in SING's published order. Use the round-3 retrained-translator pipeline instead:
1. `r3/retrain_translators.py` on the new features (same train/test split, ridge grid, and the extended grid of `r3/retrain_lambda_posthoc.py`, labelled post hoc).
2. Report ρ, d_null, κ, AS with the retrained translator, A1 (equal energy), the leak-free surrogate AS, and top-1.
3. Compare DINO ViT-B/16 with the supervised ViT-B/16 control at matched ρ (A1) and at raw ρ.

## Status
Not run (round 3 is CPU only). No money was spent.
