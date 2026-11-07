# Round 9: broader score check and controls

Aggregated 2026-11-07 (IST), from the completed 13-model CPU caches. Analysis checkout: `7030d60`.

## Primary outcome: global control failure

**K-ctrl fired. Every final score/reference verdict is STOPPED.** The null-family planted-content manipulation check failed: mean tau(core_null, a) = 0.392949 at a_max = 0.6, below the frozen 0.6 threshold. Its not-flagged rate and the negative-control energy flag rate both passed. This is a calibration failure, not a result that any tested score reads or fails to read content. No thresholds, frozen files or score code were changed.

The ViM-family controls passed their own criteria, but the global null-family kill still stops all score verdicts under PREREG_R9 §6. SING AS also failed PC-text validation independently (manipulation tau 0.224359 < 0.6).

## Scope and provenance

- Protocol: `PREREG_R9.md`, `PLAN.md`, `USAGE_general.md` in this directory; frozen files remain byte-identical to preregistration commit `eb0b880`.
- All 13 feature files matched `scripts/checksums.sha256`; each has 1,000 classes with 10 images each. Fit: first 5 per class; evaluation: remaining 5. Features remain private and outside version control.
- All 26 cache JSON files were present, finite, and complete: 780 control datasets per model, all 8 streams with their full fixed replicate counts, 10 real surrogate draws and 5 heads. All cache SHA256 values were unchanged after aggregation.
- Bootstrap: 2,000 model resamples, seed 100900, at least 4 distinct models per retained resample; BCa intervals. No reduced replicate scale or head suppression was used.
- `run_r9.py` is unchanged from `7030d60`. Relative to `eb0b880`, the existing `--only` fix preserves the full model count/index for parallel seeds; the standalone module changes only added toy data/CLI. Its scoring/statistical routines are unchanged.
- Old worker logs record an overlapping dinov2 computation in w1 and w4. No worker remained when this aggregation began; one existing cache pair per model was aggregated, and no new model run or duplicate process was started.
- The checkout was initially clean, and the remote master tip also resolved to `7030d60`. No local cleanup edits were found; that does not establish whether another bot has finished an unmerged cleanup branch.
- k-NNuSA is **excluded** by frozen §3 (no public definition/code found before registration); it was not implemented or measured.

## Controls at the registered kill level

| Stream / reference | Relevant rate (Wilson 95%) | Manipulation tau with a | Criterion |
|---|---:|---:|---|
| NC_null / null_norm | 40/40 = 1.000 [0.912, 1.000] | — | >= 0.8 |
| PCm_null_rand / core_null | 40/40 = 1.000 [0.912, 1.000] | 0.393 | rate >= 0.8 and tau >= 0.6 |
| NC_vim / res_norm | 20/20 = 1.000 [0.839, 1.000] | — | >= 0.8 |
| PCm_vim_rand / core_res | 20/20 = 1.000 [0.839, 1.000] | 0.994 | rate >= 0.8 and tau >= 0.6 |
| PCm_null_text / as_sing | 20/20 = 1.000 [0.839, 1.000] | 0.224 | rate >= 0.8 and abs(tau) >= 0.6 |

Full power curves for all levels, streams and scores, including mean tau with content/energy/offset, are in `results/r9_controls.json`. Wilson intervals for flagged, not-flagged and CONTENT-READING rates are all in the reporting supplement `results/r9_control_intervals.json` (the unchanged runner exports only the flagged-rate interval).

| Score/reference | NC flag count | NC false CONTENT-READING count | PC-text validated | Final verdict |
|---|---:|---:|---|---|
| nsp | 40/40 | 0/40 | not applicable | STOPPED |
| nusa | 40/40 | 0/40 | not applicable | STOPPED |
| null_norm | 40/40 | 0/40 | not applicable | STOPPED |
| rho | 40/40 | 0/40 | not applicable | STOPPED |
| core_null | 36/40 | 0/40 | not applicable | STOPPED |
| as_sing | 40/40 | 0/40 | False | STOPPED |
| vim | 20/20 | 0/20 | not applicable | STOPPED |
| vim_res | 20/20 | 0/20 | not applicable | STOPPED |
| res_norm | 20/20 | 0/20 | not applicable | STOPPED |
| core_res | 0/20 | 2/20 | not applicable | STOPPED |

`core_res` has low NC flag power (0/20) and NC false CONTENT-READING rate 2/20 = 0.1, below the 0.2 withholding threshold. No other score has a nonzero NC false CONTENT-READING rate. Global stopping supersedes those per-score diagnostics.

## Step 1–2 diagnostics (not validated score conclusions)

The following statistics are retained for transparency. They do not override K-ctrl or license ENERGY-DRIVEN / MIXED / CONTENT-READING claims. “Equivalent” flags below are raw numerical criterion checks only; they are not validated findings after stopping.

| Score/reference | tau_E (BCa 95%) | Gap variance share | 90% tau_E lower bound > 0.9 | Final verdict |
|---|---:|---:|---|---|
| nsp | 1.000 [1.000, 1.000] | 0.004 | True | STOPPED |
| nusa | 0.949 [0.673, 1.000] | 0.003 | False | STOPPED |
| null_norm | 1.000 [1.000, 1.000] | 0.005 | True | STOPPED |
| rho | 1.000 [1.000, 1.000] | 0.000 | True | STOPPED |
| core_null | 0.667 [0.104, 0.917] | 0.602 | False | STOPPED |
| as_sing | 0.846 [0.444, 1.000] | 0.136 | False | STOPPED |
| vim | 0.487 [0.082, 0.785] | 0.536 | False | STOPPED |
| vim_res | 0.949 [0.798, 1.000] | 0.004 | False | STOPPED |
| res_norm | 1.000 [1.000, 1.000] | 0.001 | True | STOPPED |
| core_res | -0.051 [-0.460, 0.504] | 0.521 | False | STOPPED |

The null-energy offset share ranges from 0.007 to 0.918 (median 0.082). Step-1 per-model offset/between/within decompositions and centring statistics are retained in the JSON. The raw offset-order criterion is true only for res_norm.

### Relation to the published SING order

The paired quantity is dtau = tau(real, published SING order) - tau(surrogate, published SING order). The frozen equivalence margin is (-0.1, 0.1), using the paired BCa 90% interval. Failure of this criterion is not evidence of a difference.

| Score/reference | dtau | BCa 90% | Numerically inside margin |
|---|---:|---:|---|
| nsp | 0.000 | [0.000, 0.000] | True |
| nusa | 0.051 | [0.000, 0.240] | False |
| null_norm | 0.000 | [0.000, 0.000] | True |
| rho | 0.000 | [0.000, 0.000] | True |
| core_null | 0.026 | [-0.329, 0.250] | False |
| as_sing | 0.154 | [0.027, 0.486] | False |
| vim | 0.000 | [-0.438, 0.429] | False |
| vim_res | 0.051 | [0.000, 0.167] | False |
| res_norm | 0.000 | [0.000, 0.000] | True |
| core_res | 0.026 | [-0.561, 0.667] | False |

## Accuracy-matched head diagnostics

Ridge and perturbed heads each matched evaluation top-1 within 1 percentage point on all 13 models. NCM matched only 1 and LDA only 2, so neither exploratory head has an estimable matched-head reading (minimum 8). The tables below are descriptive diagnostics under global stopping, not validated conclusions.

| Score/reference | Ridge tau (BCa 95%) / raw reading | Perturbed tau (BCa 95%) / raw reading |
|---|---|---|
| nsp | 0.974 [0.734, 1.000] / HEAD-STABLE | 0.821 [0.362, 0.973] / HEAD-STABLE |
| nusa | 0.923 [0.563, 1.000] / HEAD-STABLE | 0.718 [0.038, 0.971] / INTERMEDIATE |
| null_norm | 1.000 [1.000, 1.000] / HEAD-STABLE | 0.846 [0.470, 1.000] / HEAD-STABLE |
| rho | 0.949 [0.710, 1.000] / HEAD-STABLE | 0.769 [0.156, 1.000] / INTERMEDIATE |
| core_null | 1.000 [1.000, 1.000] / HEAD-STABLE | 0.718 [-0.113, 0.973] / INTERMEDIATE |
| as_sing | 1.000 [1.000, 1.000] / HEAD-STABLE | 0.538 [-0.031, 1.000] / INTERMEDIATE |
| vim | 0.949 [0.580, 1.000] / HEAD-STABLE | 0.949 [0.580, 1.000] / HEAD-STABLE |
| vim_res | 0.974 [0.741, 1.000] / HEAD-STABLE | 0.974 [0.741, 1.000] / HEAD-STABLE |
| res_norm | 0.692 [0.167, 1.000] / INTERMEDIATE | 0.692 [0.167, 1.000] / INTERMEDIATE |
| core_res | 0.795 [0.217, 1.000] / INTERMEDIATE | 0.769 [0.164, 1.000] / INTERMEDIATE |

| Model | Released top-1 (%) | Ridge (%) | Perturbed (%) | NCM (%) | LDA (%) |
|---|---:|---:|---:|---:|---:|
| vgg16 | 71.76 | 71.68 | 71.26 | 56.10 | 51.04 |
| vgg19 | 73.00 | 72.94 | 72.74 | 57.24 | 52.86 |
| densenet | 75.00 | 75.00 | 74.22 | 59.26 | 59.04 |
| resnet | 76.18 | 76.18 | 75.54 | 65.18 | 63.50 |
| dinovit1 | 85.88 | 85.88 | 85.52 | 82.46 | 82.76 |
| efficientnet | 78.54 | 78.54 | 78.00 | 67.82 | 68.96 |
| bitresnet | 80.34 | 80.36 | 80.02 | 70.72 | 70.26 |
| resnext | 82.76 | 82.76 | 82.16 | 77.28 | 77.88 |
| convnext | 85.88 | 85.86 | 85.58 | 83.88 | 83.82 |
| swin_large | 86.42 | 86.42 | 85.94 | 82.02 | 83.10 |
| dinov2 | 85.96 | 85.94 | 85.26 | 74.52 | 75.22 |
| deit | 85.08 | 85.08 | 84.50 | 82.98 | 84.44 |
| eva02_large | 89.74 | 89.74 | 89.46 | 89.62 | 89.52 |

## GPU feature arm: eligibility checked, score analysis not started

All 71 validation NPZ files match their manifest SHA256 values, have finite float32 features with the expected shapes, and the manifest reports exact feature/model argmax agreement for every model. The 10k listing hash matches the frozen listing. Extractor and model-list hashes match `JOB.md`; extraction used commit `eb0b880`, timm 1.0.30, torch 2.11.0+cu130 and a free Kaggle T4. Environment differences are disclosed, not concealed.

| Anchor | Median relative feature difference | Cached / new top-1 (%) | Gate |
|---|---:|---:|---|
| resnetv2_50x1_bit.goog_in21k_ft_in1k | 0 | 80.32 / 80.32 | PASS |
| tf_efficientnet_b0.ns_jft_in1k | 8.5514297e-07 | 79.03 / 79.03 | PASS |
| convnext_base.fb_in22k_ft_in1k | 5.2363987e-07 | 85.79 / 85.79 | PASS |
| resnext101_32x8d.fb_wsl_ig1b_ft_in1k | 0 | 82.60 / 82.60 | PASS |

Relative difference is the per-image norm of (new - cached) divided by the cached feature norm, then the median across images. All four are below 1e-3, and all top-1 differences are zero (threshold 0.5 percentage point). This satisfies provenance/anchor gates (a)–(b), not analysis gate (c). No GPU score result or confirmatory extension claim exists yet. See `results/gpu_eligibility.json`.

**No train download is needed for the primary GPU analysis.** Frozen §2 and §9 use the first 5 validation images per class to fit heads/bases and the remaining 5 for evaluation. Train-fitted heads are optional secondary work; no unavailable train data were substituted. All 71 train NPZ remain on private Kaggle. Their estimated local footprint is 5,809,772,018 bytes (same shapes/uncompressed layout as validation); current free space is about 4.7 GB, so keeping a 1.5 GB reserve would require about 2.6 GB additional free space. No download was attempted.

A same-order escaped model-name list and an unchanged-runner command were prepared under `_local/r9/gpu_models_tags.txt` and `_local/r9/gpu_run_command.txt`, but were not executed. Dots are represented as `__` solely to match extractor filenames, with no model reordering or replacement. The 71-model extension is not completed. Execution stops after the global CPU K-ctrl failure: no GPU score run, new experiment, paper edit or review is being launched. Training-data availability is not the reason for this stop.

## Reproduction and verification

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest r9/tests -q
# 19 passed in 29.78 s

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -u r9/run_r9.py \
  --feats-dirs data/ic014/e2/ret/e2_feats data/ic014/e3/ret/e3_feats \
  --models-file r9/models13.txt \
  --text-emb _ext/sing/misc/embeddings/imagenet_text_embeddings_fp16.npz \
  --sing-order r9/sing_order.csv --out r9/results --work _local/r9/work --tag r9 --only
# Exit 0. Empty --only is a cache-only aggregation: it cannot compute missing models.

sha256sum -c _local/r9/cache_before_aggregation.sha256
# 26/26 OK after aggregation
```

Private verification reports are `_local/r9/cpu_provenance_20261107.json`, `gpu_provenance_20261107.json`, `gpu_anchor_checks_20261107.json`; aggregation stdout is `_local/r9/aggregation_20261107.log`. Result JSON contains per-model scalar summaries, not feature arrays. During aggregation, no paper edit, cleanup, Humanize, reviewer send, commit, push, paid compute or external publication was performed. Recording these factual results in source control is a separate finalization step; it does not lift the stop on further research.
