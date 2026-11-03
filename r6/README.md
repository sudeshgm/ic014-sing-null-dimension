# Round 6: audit of published null-space / residual scores

Pre-registration: `r6/PREREG_R6.md`. Scores: `scores_r6.py` (ViM, ActSub S→, CORE S_mem, conditional V-information, NuSA-style reference). Tests and driver: `audit_r6.py`. Statistics and verdicts: `verdicts_r6.py`. Frozen synthetic unit tests: `tests/test_r6.py` (`python -m pytest -q r6/tests`).

Run (CPU; needs the private features):

    python r6/audit_r6.py --feats-root $IC014_FEATS --models resnet,dinov2,... --worlds resnet,dinov2,resnext --threads 2
    python r6/verdicts_r6.py

Results will be written to `r6/results/`.
