# Round 9 plan

Goal: run the three-step check on scores other than SING (NSP, NuSA, ViM residual), validate it with planted-content
and pure-energy controls, release a standalone version of the check, and repeat the accuracy-matched head swap on more
backbones. Criteria are in `PREREG_R9.md`.

| step | what | where |
|---|---|---|
| A | pre-registration, code, synthetic tests, synthetic calibration; push before any real run | `PREREG_R9.md`, `*.py`, `tests/`, `results/synthetic/` |
| A | GPU extraction job for 71 timm models | `gpu/extract_r9_gpu.py`, `gpu_models.txt` |
| B | controls and all scores on the 13 cached models (CPU) | `run_r9.py` -> `results/r9_*.json`, `README.md` |
| C | standalone check for any user score | `sanity_check_general.py` |
| D | GPU arm, if features arrive (gated, §8 of the prereg) | `results/gpu_*.json` |

Commands:

    python -m pytest r9/tests -q
    python r9/run_r9.py --synthetic --out r9/results/synthetic --tag synthetic --syn-n 13 --syn-d 160 --syn-K 40
    python r9/run_r9.py --feats-dirs $FEATS/e2/ret/e2_feats $FEATS/e3/ret/e3_feats \
        --text-emb _ext/sing/misc/embeddings/imagenet_text_embeddings_fp16.npz --out r9/results --work $WORK
