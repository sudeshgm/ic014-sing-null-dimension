"""Post hoc synthetic check (r6/POSTHOC_R6.md), not part of the frozen PREREG_R6 tests.
The within-class permutation C2 keeps each class's null distribution exactly, but it can still remove real conditional information:
here the null part carries the image-level logit noise, so (logits, null) recover the class, and the conditional V-information gain
drops to about zero after C2. A large C2 effect is therefore not by itself evidence of estimator noise.
Run: python -m pytest -q r6/tests"""
import os, sys, numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H)
import scores_r6 as sc  # noqa: E402


def _coupled(seed=0, n=2000, C=5, k=6):
    rng = np.random.default_rng(seed); y = np.repeat(np.arange(C), n // C); eps = rng.standard_normal((n, C))
    L = 1.0 * np.eye(C)[y] + eps; Cn = rng.standard_normal((n, k)) * 0.1; Cn[:, :C] += eps
    return Cn, L, y, C


def _within_class_perm(y, seed):
    rng = np.random.default_rng(seed); p = np.arange(len(y))
    for c in np.unique(y):
        idx = np.where(y == c)[0]; p[idx] = rng.permutation(idx)
    return p


def test_within_class_permutation_can_remove_real_conditional_information():
    old = sc.NCLS
    try:
        Cn, L, y, C = _coupled(); sc.NCLS = C; fit = np.arange(len(y)) % 2 == 0
        def gain(Cm):
            return sc.vinfo_fit_eval(Cm[fit], L[fit], y[fit], Cm[~fit], L[~fit], y[~fit], 1e-2, 'label')[0].mean()
        p = _within_class_perm(y, 1); Cp = Cn[p]
        for c in range(C):     # classwise null distribution unchanged (same multiset of rows per class)
            a, b = np.sort(Cn[y == c], axis=0), np.sort(Cp[y == c], axis=0); assert np.array_equal(a, b)
        g0, g1 = gain(Cn), gain(Cp)
        assert g0 > 0.3 and abs(g1) < 0.05, (g0, g1)
    finally:
        sc.NCLS = old
