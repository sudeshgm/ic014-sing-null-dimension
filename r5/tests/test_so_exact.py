"""Post hoc synthetic check (r5/POSTHOC_R5.md): the round-5 dcent leaves part of an additive class effect; dcent_exact removes it."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from so_exact_r5 import dcent_exact  # noqa: E402
import organisation as O  # noqa: E402


def _rowsum(v, n):
    iu = np.triu_indices(n, 1); M = np.zeros((n, n)); M[iu] = v; M = M + M.T; return M.sum(1)


def test_additive_effect():
    n = O.NCLS; rng = np.random.default_rng(1); a = rng.lognormal(0, 1, n); iu = np.triu_indices(n, 1); A = (a[:, None] + a[None])[iu]
    assert np.abs(dcent_exact(A)).max() < 1e-9 and np.abs(_rowsum(dcent_exact(A), n)).max() < 1e-8
    assert np.abs(O.dcent(A)).max() > 1e-3 and np.abs(_rowsum(O.dcent(A), n)).max() > 1.0   # the committed round-5 centring keeps a residual


def test_keeps_non_additive_part():
    n = 50; rng = np.random.default_rng(2); X = rng.standard_normal((n, 5)); iu = np.triu_indices(n, 1); D = ((X[:, None] - X[None]) ** 2).sum(-1)[iu]
    a = rng.normal(0, 3, n); Da = D + (a[:, None] + a[None])[iu]
    assert np.allclose(dcent_exact(D, n), dcent_exact(Da, n)); assert np.abs(dcent_exact(D, n)).max() > 0.1
