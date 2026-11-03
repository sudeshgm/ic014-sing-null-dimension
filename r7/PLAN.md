# Round 7 plan: do real nuisance views support noise cancellation for null-space content?

## Why this round

Rounds 3-6 found that every class-level, scale-free null-space score tried so far passes world B (rescaling the null part)
but fails world C (fixed content, varying null noise): with one feature vector per image, a score that ignores the null
part's scale can only measure content relative to the noise around it. An outside reviewer pointed out two gaps in that
argument (higher-order statistics against an outside semantic reference can cancel independent additive noise; the class-mean
Gram holds more than eta^2, SO/XCC and rho) and recommended an experiment rather than another score: obtain controlled
semantic edits and independently generated nuisance views, and test whether their residual structure supports noise
cancellation. Repeated extraction of one deterministic image gives no such evidence, so every view here is a fresh random
augmentation draw.

## Data (free, already public, nothing generated)

**Chosen: ImageNet-9 Backgrounds Challenge test set** (Xiao, Engstrom, Ilyas, Madry, "Noise or Signal: The Role of Image
Backgrounds in Object Recognition", ICLR 2021, arXiv 2006.09994; github.com/MadryLab/backgrounds_challenge, release `data`,
`backgrounds_challenge_data.tar.gz`, 279.6 MB, sha256 2d66f571b0492986b347e37d2ee498ed34b58012af425d772dacda4422896fe7,
downloaded 2026-11-03). 9 coarse classes x 450 foreground images, each in 8 variants plus a foreground mask (224 x 224).
Licence: the repository has no licence file; the images are ImageNet-derived, so ImageNet's non-commercial research terms
apply. Images and features stay private (`_ext/`, git-excluded); only aggregates are committed.

Why it fits:
- `mixed_next` and `mixed_same` paste the **same foreground pixels at the same place** onto a background from the next IN-9
  class or from the same IN-9 class (checked: mean |X - S| inside the mask 1-2 grey levels, 50-115 outside). Their difference
  is a controlled semantic edit of the background with the object held fixed.
- `mixed_next` file names carry the **fine ImageNet synset of the background** (checked: 4050 / 4050 map to IN-9 class c+1).
  That gives an image-level external semantic reference (SING's CLIP text embedding of that ImageNet class) which varies
  within each foreground class (283 distinct background classes in our 900-image subset; 16-54 per group).
- The foreground masks allow a same-pixels artefact control (background pixels permuted among themselves).

Considered and not used:
- **ImageNet-E** (Li et al., CVPR 2023; alibaba/easyrobust): diffusion edits of background complexity, size, position and
  direction for 4,352 images. Its edits are graded nuisances without a named semantic target, so there is no external
  reference to align with; access is through a download script of the easyrobust repo (Apache-2.0 code; ImageNet-derived data).
- Paired attribute datasets (e.g. CelebA attributes, PUG:ImageNet renders) are not ImageNet photographs for these classifiers
  or need a new rendering pipeline; out of scope for a free CPU round.

## Design in one paragraph

For each foreground image i (100 per IN-9 class, 900 total, seed 100701) and each condition X (`mixed_next`), S (`mixed_same`),
Xs, Ss (the same two with background pixels permuted, foreground untouched), we extract the classifier-input features under
one clean view and 4 independent nuisance views at each of 3 strengths (crop, flip, colour jitter, hue, grayscale, blur, JPEG).
Within a view the same nuisance draw is applied to all four conditions, so the edit Delta = Pi f(X) - Pi f(S) is paired and the
nuisance enters only through its interaction with the edit; views are independent. The edit's within-class structure is
compared with the text embedding of the background class by an unbiased HSIC statistic whose numerator and normaliser are
estimated across independent views, so additive view noise has no expected effect (world-C analogue), and rescaling Pi f
leaves it unchanged exactly (world B).

## Models (4 of the 13)

ResNet50 (`resnet`, d_null 1048, rho 0.146), EfficientNet-B0 NS (`efficientnet`, 280, 0.130), ResNeXt101 32x8d WSL
(`resnext`, 1048, 0.451) on CPU; DINOv2-L (`dinov2`, 1048, 0.322) via a GPU job spec (CPU timing 5.5 img/s at 6 threads, about
2.5-3 h for 46,800 forward passes). Same weights and the same feature hook as the cached 10k features (SING registry at commit
e47b37d).

## Cost

46,800 forward passes per model (900 images x 52 slots). Measured on this box (6 threads, shared): ResNet50 39 img/s,
EfficientNet-B0 92, ResNeXt101 15.5, DINOv2-L 5.5. Data pipeline about 2.9 s per image (52 augmented crops), run in workers.
Features: float32, 52 x 900 x d, 0.2-0.4 GB per model in `_ext/r7_feats/`.

## Order of work

1. Synthetic stability study (`r7/synth_r7.py`) and frozen unit tests (`r7/tests/test_r7.py`).
2. Commit and push `PREREG_R7.md`, code, tests and the synthetic results before any real feature exists.
3. CPU extraction (resnet, efficientnet, resnext); GPU job spec for dinov2 in `_local/r7_gpu_job/`.
4. Pre-registered analysis per model (`r7/analyze_r7.py`); results in `r7/results/`.
