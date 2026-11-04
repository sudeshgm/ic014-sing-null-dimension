# Standalone three-step check (`sanity_check_general.py`)

One file; needs only numpy, scipy and scikit-learn. Copy it next to your code.

Python:

    from sanity_check_general import run_check
    models = {name: dict(F=F, y=y, W=W, b=b, fit=fit_mask), ...}   # one entry per model
    out = run_check(models, my_score)                               # my_score(model_dict) -> float or {name: float}
    out['verdict'], out['step1'], out['step2'], out['step3']

`my_score` must read only the dict it receives: the tool calls it on copies of each model with the subspace
coordinates replaced (steps 1 and 2) or with a refitted head (step 3). The default subspace is null(W); pass
`subspace_fn=lambda m: (V, origin)` for another one (e.g. a PCA residual).

Command line (one npz per model with F, y, W, b; optional boolean `fit`, otherwise the first half of each class):

    python sanity_check_general.py --npz m1.npz m2.npz ... --score my_score.py:score --out check.json
    python sanity_check_general.py --demo

Steps and verdicts:

1. centre: offset / between-class / within-class split of the subspace energy; the score with the offset only and
   without it (`offset_dominated` if the offset-only order matches, BCa lower bound of tau > 0.5).
2. surrogate: subspace coordinates replaced by Gaussian draws with each model's own mean and covariance.
   ENERGY-DRIVEN if the BCa 95% lower bound of tau(real, surrogate) > 0.5; CONTENT-READING if the upper bound < 0.5
   and the surrogate gap carries at least half the cross-model variance; MIXED otherwise.
3. head: nearest class mean, shrinkage LDA, ridge distilled from the given logits, and a perturbed copy of the given
   head; top-1 on the non-fit images, `matched` within 1 point; HEAD-STABLE (tau >= 0.8) / HEAD-DEPENDENT
   (tau <= 0.5) / INTERMEDIATE over matched models, NOT ESTIMABLE below 8 matched models.

With around ten models every interval is wide; read MIXED as unresolved. Tests: `tests/test_general.py`.
