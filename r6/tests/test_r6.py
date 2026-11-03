"""Frozen synthetic unit tests for r6/scores_r6.py and the interventions of r6/audit_r6.py (PREREG_R6 §1). Synthetic data only.
Run: python -m pytest -q r6/tests"""
import os, sys, numpy as np, torch
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H)
import scores_r6 as sc  # noqa: E402
from scipy.special import logsumexp, softmax  # noqa: E402


def toy(n=600, d=40, C=10, seed=0, null_extra=0.5):
    rng = np.random.default_rng(seed); W = rng.standard_normal((C, d)); b = rng.standard_normal(C) * 0.1
    y = np.repeat(np.arange(C), n // C); M = rng.standard_normal((C, d)) * 2
    F = M[y] + rng.standard_normal((n, d)); Vr, Vn = sc.null_split(W); F = F + null_extra * rng.standard_normal((n, Vn.shape[1])) @ Vn.T
    return F, y, W, b, F @ W.T + b, Vr, Vn


def test_vim_matches_official_port():
    F, y, W, b, L, _, _ = toy(); fit = np.arange(len(y)) % 2 == 0
    m = sc.vim_fit(F[fit], L[fit], W, b); v = sc.vim_eval(m, F[~fit], L[~fit])
    # port of official benchmark.py lines (u, EmpiricalCovariance(assume_centered), eig, NS, alpha, score)
    u = -np.matmul(np.linalg.pinv(W), b); X = F[fit] - u; cov = X.T @ X / len(X); ev, evec = np.linalg.eig(cov)
    NS = np.ascontiguousarray((evec.T[np.argsort(ev * -1)[sc.vim_dim(F.shape[1]):]]).T).real
    alpha = L[fit].max(-1).mean() / np.linalg.norm(X @ NS, axis=-1).mean()
    vl = np.linalg.norm((F[~fit] - u) @ NS, axis=-1) * alpha; s = -vl + logsumexp(L[~fit], axis=-1)
    assert np.allclose(v['s'], s, rtol=1e-8, atol=1e-8)
    p = softmax(np.c_[L[~fit], vl], axis=1)[:, -1]; assert np.allclose(v['p_virtual'], p, atol=1e-10)
    assert abs(m['alpha'] * np.linalg.norm(X @ m['NS'], axis=1).mean() - L[fit].max(1).mean()) < 1e-8


def test_vim_dim_rule():
    assert [sc.vim_dim(d) for d in (4096, 2048, 1536, 1280, 1024, 768, 512)] == [1000, 1000, 512, 512, 512, 512, 256]


def test_actsub_cidx_matches_official_loop():
    F, y, W, b, L, _, _ = toy(); A = torch.tensor(W); E = torch.tensor(F)
    _, _, Vt = torch.linalg.svd(A); ratios = []
    for idx in range(A.shape[1]):
        dec, ins = Vt.T[:, :idx] @ Vt[:idx, :], Vt.T[:, idx:] @ Vt[idx:, :]
        ratios.append((torch.norm(E @ dec.T, dim=1).mean() - torch.norm(E @ ins.T, dim=1).mean()).abs())
    ref = int(torch.tensor(ratios).argmin(0)); Vt_np = np.linalg.svd(W, full_matrices=True)[2]
    assert sc.actsub_cidx(Vt_np, F) == ref


def test_actsub_score_matches_official_formula():
    F, y, W, b, L, _, _ = toy(); Vt = np.linalg.svd(W, full_matrices=True)[2]; fit = np.arange(len(y)) % 2 == 0
    m = sc.actsub_fit(Vt, F[fit], 1); s = sc.actsub_eval(m, F[~fit])['S_insig']
    ins = torch.tensor(m['Vi'] @ m['Vi'].T); x = torch.tensor(F[~fit]) @ ins.T
    idx = np.random.default_rng(1).permutation(fit.sum())[: fit.sum() // 10]; refs = torch.tensor(F[fit][idx]) @ ins.T
    cs = torch.topk(torch.nn.functional.normalize(x, dim=1) @ torch.nn.functional.normalize(refs, dim=1).T, 10, dim=1)[0].mean(1)
    assert np.allclose(s, (-(1. - cs).log()).numpy(), atol=1e-9)


def test_core_definitions():
    F, y, W, b, L, _, _ = toy(); yh = L.argmax(1); Z = sc.core_perp(F, W, yh)
    assert np.abs((Z * W[yh]).sum(1)).max() < 1e-9                      # z_perp orthogonal to w_yhat
    # all fit residuals of a class share one direction -> cos = 1 for an evaluation sample with that direction
    rng = np.random.default_rng(3); d = W.shape[1]; Wn = W; dirs = rng.standard_normal((10, d))
    yy = np.repeat(np.arange(10), 4); Ff = 5 * Wn[yy] + (1 + rng.random((40, 1))) * dirs[yy]
    Lf = Ff @ Wn.T; m = sc.core_fit(Ff, Wn, Lf, Lf.argmax(1)); s = sc.core_eval(m, Ff, Wn, Lf)['S_mem']
    keep = Lf.argmax(1) == yy; Zf = sc.core_perp(Ff, Wn, Lf.argmax(1))
    assert keep.sum() > 20 and np.all(s[keep] <= 1 + 1e-12)
    # exactness: when every class's fit residuals are collinear, S_mem = 1 for those samples
    c = np.unique(Lf.argmax(1))[0]; mm = Lf.argmax(1) == c; u = Zf[mm][0] / np.linalg.norm(Zf[mm][0])
    m2 = dict(mu=np.tile(u, (1000, 1)), Vn=None); assert abs(sc.core_eval(m2, Ff[mm][:1], Wn, Lf[mm][:1])['S_mem'][0] - 1) < 1e-12


def _vinfo_data(plant, seed=0, n=2000, C=5, k=6):
    rng = np.random.default_rng(seed); y = rng.integers(0, C, n); L = rng.standard_normal((n, C)) + 1.0 * np.eye(C)[y]
    Cn = rng.standard_normal((n, k))
    if plant: Cn[:, :C] += 2.0 * np.eye(C)[y]
    return Cn, L, y, C


def test_vinfo_zero_when_independent_and_positive_when_planted():
    import scores_r6 as s2
    old = s2.NCLS
    try:
        for plant, ok in ((False, lambda g: abs(g) < 0.02), (True, lambda g: g > 0.3)):
            Cn, L, y, C = _vinfo_data(plant); s2.NCLS = C; fit = np.arange(len(y)) % 2 == 0
            g, lam, _ = s2.vinfo_fit_eval(Cn[fit], L[fit], y[fit], Cn[~fit], L[~fit], y[~fit], 1e-2, 'label')
            assert ok(g.mean()), (plant, g.mean())
    finally:
        s2.NCLS = old


def test_interventions_keep_logits_and_hit_targets():
    sys.path.insert(0, H); import audit_r6 as au
    F, y, W, b, L, _, Vn = toy(n=1000, d=1100, C=1000, seed=5)
    M = au.Model(F, y, W, b, Vn)
    for rho in (0.005, 0.2):
        Cm = M.to_rho(rho); assert abs((Cm ** 2).sum(1).mean() / ((Cm ** 2).sum(1).mean() + M.Er) - rho) < 1e-12
        assert np.abs(M.feats(Cm) @ W.T + b - M.L).max() < 1e-8
    p = np.random.default_rng(0).permutation(len(y)); assert abs((M.C[p] ** 2).sum() - (M.C ** 2).sum()) < 1e-8
    u = np.random.default_rng(1).standard_normal((1000, M.C.shape[1])); Cp = au.energy_fix(M.C + 0.3 * u[y], M.En)
    assert abs((Cp ** 2).sum(1).mean() - M.En) < 1e-9


def test_halves_stratified():
    y = np.repeat(np.arange(1000), 10); A = sc.halves(y, 100100); assert np.all(np.bincount(y[A], minlength=1000) == 5)


def test_vinfo_corr_target_runs_and_is_small_when_independent():
    rng = np.random.default_rng(2); n = 3000; C = 1000; y = rng.integers(0, C, n); L = rng.standard_normal((n, C)) + 4 * np.eye(C)[y]
    Cn = rng.standard_normal((n, 8)); fit = np.arange(n) % 2 == 0
    g, _, _ = sc.vinfo_fit_eval(Cn[fit], L[fit], y[fit], Cn[~fit], L[~fit], y[~fit], 1e-1, 'corr'); assert abs(g.mean()) < 0.02
