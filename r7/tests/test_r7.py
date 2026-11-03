"""Frozen synthetic unit tests for PREREG_R7 (no real data)."""
import os, sys, numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H)
import stats_r7 as st, views_r7 as vw, synth_r7 as sy  # noqa: E402


def _world(seed=1, N=180, k=60, nu=0.0, kind='iso', w=0.6):
    rng = np.random.default_rng(seed); g, R, sig, u, sh = sy.world(rng, N=N, G=9, k=k, q=32, nfine=8)
    mu = sy.mu_for(sig, u, sh, w); sE = float(np.mean([((mu[g == c] - mu[g == c].mean(0)) ** 2).sum(1).mean() for c in np.unique(g)]))
    basis = np.linalg.qr(rng.standard_normal((k, k)))[0]
    D = mu[None] + (sy.noise(rng, kind, N, k, 4, nu * sE, basis) if nu > 0 else np.zeros((4, N, k))); return D, R, g, mu, rng, basis, sE


def test_hsic_u_shift_invariant_and_unbiased_zero():
    rng = np.random.default_rng(0); x = rng.standard_normal((60, 5)); y = rng.standard_normal((60, 3)); c = rng.standard_normal(5)
    assert abs(st.hsic_u(x @ x.T, y @ y.T) - st.hsic_u((x + c) @ (x + c).T, y @ y.T)) < 1e-10
    vals = [st.hsic_u(*(lambda a, b: (a @ a.T, b @ b.T))(rng.standard_normal((40, 4)), rng.standard_normal((40, 4)))) for _ in range(400)]
    assert abs(np.mean(vals)) < 3 * np.std(vals) / np.sqrt(400)


def test_world_B_exact_rescale_invariance():
    D, R, g, *_ = _world(nu=1.0)
    for s in (1e-3, 0.37, 50.0):
        a, b = st.T_stat(D, R, g), st.T_stat(D * s, R, g)
        assert abs(a['T'] - b['T']) < 1e-9 and abs(a['T2'] - b['T2']) < 1e-9


def test_noise_free_views_give_population_value():
    D, R, g, mu, *_ = _world(nu=0.0); s = st.T_stat(D, R, g)
    assert np.isfinite(s['T']) and abs(s['T'] - sy.tstar(mu, R, g)) < 1e-12 and 0 < s['T2'] < 1


def test_worldC_noise_does_not_bias_T2_on_average():
    D0, R, g, mu, rng, basis, sE = _world(nu=0.0); t0 = st.T_stat(D0, R, g)['T2']; vals = []
    for r in range(150):
        D = mu[None] + sy.noise(rng, 'iso', len(g), mu.shape[1], 4, 4.0 * sE, basis); vals.append(st.T_stat(D, R, g)['T2'])
    assert abs(np.mean(vals) - t0) < 4 * np.std(vals) / np.sqrt(150) + 0.02 * t0


def test_naive_cka_attenuates_with_noise():
    D0, R, g, mu, rng, basis, sE = _world(nu=0.0)
    Dn = mu[None] + sy.noise(rng, 'iso', len(g), mu.shape[1], 4, 8.0 * sE, basis)
    assert st.T_naive(Dn, R, g) < 0.8 * st.T_naive(D0, R, g)


def test_within_group_removes_group_level_alignment():
    rng = np.random.default_rng(3); G, n, k = 9, 20, 30; g = np.repeat(np.arange(G), n); Rg = rng.standard_normal((G, 16)); R = Rg[g] + 0.0 * g[:, None]
    R = R + 1e-3 * rng.standard_normal(R.shape)  # within-group reference variation is pure jitter
    A = rng.standard_normal((16, k)); mu = Rg[g] @ A + rng.standard_normal((G * n, k)); D = mu[None] + 0.1 * rng.standard_normal((4, G * n, k))
    assert abs(st.T_stat(D, R, g, 'wc')['T']) < 0.1 and st.T_stat(D, R, g, 'gl')['T'] > 0.3


def test_perm_test_null_and_planted():
    D, R, g, *_ = _world(nu=1.0, w=0.8); assert st.perm_test(D, R, g, 200, 5)['p'] < 0.01
    D0, R0, g0, *_ = _world(nu=1.0, w=0.0, seed=7); assert st.perm_test(D0, R0, g0, 200, 6)['p'] > 0.01


def test_reliability_and_noise_ratio():
    D, R, g, *_ = _world(nu=0.0); assert abs(st.reliability(D, g) - 1) < 1e-12 and abs(st.noise_ratio(D, g)['nu']) < 1e-9
    D, R, g, *_ = _world(nu=2.0); r = st.noise_ratio(D, g)['nu']; assert 1.5 < r < 2.6
    rng = np.random.default_rng(9); Z = rng.standard_normal((4, 180, 40)); assert abs(st.reliability(Z, g)) < 0.05


def test_nuisance_only_control_centred_on_zero():
    rng = np.random.default_rng(11); D, R, g, *_ = _world(nu=1.0); s = st.T_stat(D, R, g); vals = []
    for _ in range(100):
        Sv = rng.standard_normal((4, len(g), D.shape[2])); Nn = np.stack([Sv[0] - Sv[1], Sv[2] - Sv[3]])
        vals.append(st.control_num(Nn, R, g) / (s['tr'] * np.sqrt(s['kk'])))
    assert abs(np.mean(vals)) < 4 * np.std(vals) / 10


def test_views_deterministic_shared_and_pixel_shuffle_exact():
    th1 = vw.draw(2, 5, 1); th2 = vw.draw(2, 5, 1); th3 = vw.draw(2, 5, 2); assert th1 == th2 and th1 != th3
    rng = np.random.default_rng(0); img = rng.integers(0, 256, (224, 224, 3), dtype=np.uint8); M = np.zeros((224, 224), bool); M[50:150, 60:170] = True
    out = vw.pixel_shuffle(img, M, rng)
    assert np.array_equal(out[M], img[M]) and not np.array_equal(out[~M], img[~M])
    assert np.array_equal(np.sort(out[~M].view([('', np.uint8)] * 3).ravel()), np.sort(img[~M].view([('', np.uint8)] * 3).ravel()))
    a = np.asarray(vw.apply(img, th1)); b = np.asarray(vw.apply(img, th1)); assert a.shape == (224, 224, 3) and np.array_equal(a, b)
