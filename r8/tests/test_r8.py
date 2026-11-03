"""Frozen synthetic unit tests for PREREG_R8 (no real data)."""
import os, sys, numpy as np
H = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, H)
import core_r8 as c, synth_r8 as sy  # noqa: E402

K = 200


def _setup(seed=1, **kw):
    Z, S, y, T = sy.synth(seed, K=K, q=128, **kw); hA = c.halves(y, K)
    return Z, S, y, T, hA, c.CM.from_images(Z, y, K, hA), c.CM.from_images(S, y, K, hA), c.fold_split(K, 100800)


def _direct_sem(M, T, iA, iB, lam):
    """Reference: the ideas-K formula (explicit ridge map, fold-centred class means)."""
    tot = 0.0
    for tr, te in ((iA, iB), (iB, iA)):
        mtr, mte = M[tr] - M[tr].mean(0), M[te] - M[te].mean(0); Ttr = T[tr]
        lm = c.lam_rule(Ttr) if lam is None else lam
        G = mtr.T @ Ttr @ np.linalg.inv(Ttr.T @ Ttr + lm * np.eye(Ttr.shape[1]))
        tot += float(np.sum((T[te] @ G.T) * mte))
    return tot / len(M)


def test_halves_split_each_class():
    y = np.repeat(np.arange(7), 10); hA = c.halves(y, 7)
    assert all(hA[y == k].sum() == 5 for k in range(7)) and hA[:5].all() and not hA[5:10].any()


def test_gram_form_equals_direct_ridge():
    Z, S, y, T, hA, cz, cs, f = _setup()
    for lam in (1.0, None):
        a = c.S_multi([cz.G], T, *f, lam)[0]; b = _direct_sem(cz.M, T, *f, lam)
        assert abs(a - b) <= 1e-10 * abs(b)


def test_exact_invariances():
    Z, S, y, T, hA, cz, cs, f = _setup(offset=0.0)
    th = c.theta_from_S(*c.S_multi([cz.G, cs.G], T, *f))
    rng = np.random.default_rng(100802); v = rng.standard_normal(Z.shape[1]); v *= 10 * np.sqrt(cz.En) / np.linalg.norm(v)
    Q1 = np.linalg.qr(rng.standard_normal((Z.shape[1],) * 2))[0]; Q2 = np.linalg.qr(rng.standard_normal((S.shape[1],) * 2))[0]
    for Z2, S2 in ((Z + v, S), (7.3 * Z, 7.3 * S), (Z @ Q1, S @ Q2)):
        a, b = c.CM.from_images(Z2, y, K, hA), c.CM.from_images(S2, y, K, hA)
        assert abs(c.theta_from_S(*c.S_multi([a.G, b.G], T, *f)) - th) <= 1e-8 * abs(th)


def test_world_B_closed_form():
    Z, S, y, T, hA, cz, cs, f = _setup(); Sz, Ss = c.S_multi([cz.G, cs.G], T, *f)
    for beta in (0.01, 0.4, 3.0, 25.0):
        t = c.theta_from_S(*c.S_multi([cz.scaled(beta).G, cs.G], T, *f))
        assert abs(t - beta ** 2 * Sz / (beta ** 2 * Sz + Ss)) <= 1e-8 * abs(t)


def test_resampling_never_shares_a_class_and_ci_contains_point():
    Z, S, y, T, hA, cz, cs, f = _setup(sz=0.5)
    for m in ('boot', 'half'):
        draws = c.resample_indices(*f, 200, 100806, m)
        assert all(len(np.intersect1d(a, b)) == 0 for a, b in draws)
        r, _ = c.theta_sem_with_ci(cz, cs, T, f, draws, m)
        assert r['ci'][0] < r['theta_sem'] < r['ci'][1]   # the r7 failure mode: CI not containing its own point estimate


def test_zero_content_floor():
    hits = 0
    for s in range(6):
        Z, S, y, T, hA, cz, cs, f = _setup(seed=10 + s, sz=0.0, offset=5.0)
        r, _ = c.theta_sem_with_ci(cz, cs, T, f, c.resample_indices(*f, 200, 100806, 'boot'), 'boot')
        hits += r['ci'][0] <= 0 <= r['ci'][1]
    assert hits >= 5


def test_derangement_removes_signal():
    Z, S, y, T, hA, cz, cs, f = _setup(sz=0.6); th = c.theta_from_S(*c.S_multi([cz.G, cs.G], T, *f))
    rng = np.random.default_rng(100804)
    v = [c.theta_from_S(*c.S_multi([cz.permuted(c.derangement(K, rng)).G, cs.G], T, *f)) for _ in range(20)]
    assert abs(np.mean(v)) < 0.1 * th


def test_Eb_unbiased_without_class_structure():
    rng = np.random.default_rng(3); y = np.repeat(np.arange(K), 10); hA = c.halves(y, K)
    v = [c.Eb(c.CM.from_images(rng.standard_normal((len(y), 20)), y, K, hA)) for _ in range(30)]
    assert abs(np.mean(v)) < 3 * np.std(v) / np.sqrt(30)


def test_world_Cprime_flat_and_offset_world_exact():
    Z, S, y, T, hA, cz, cs, f = _setup(sz=0.6)
    wp = c.WorldPieces(Z, y, K, hA, Er=float(np.mean(np.sum(S ** 2, 1))))
    targ = sorted([0.0054, 0.0089, 0.0135, 0.0464, 0.130, 0.146, 0.274, 0.322, 0.339, 0.349, 0.350, 0.451])
    rows = wp.world("C'", targ)
    assert all(abs(r['rho'] - r['rho_target']) < 2e-3 * r['rho_target'] + 1e-4 for r in rows)
    base = rows[0]['cm']; t0 = c.theta_from_S(*c.S_multi([base.G, cs.G], T, *f))
    for r in wp.offset_world(base, targ):
        assert abs(r['rho'] - r['rho_target']) < 1e-9
        assert abs(c.theta_from_S(*c.S_multi([r['cm'].G, cs.G], T, *f)) - t0) <= 1e-8 * abs(t0)


def test_innovation_empty_when_null_is_a_linear_image_of_row():
    rng = np.random.default_rng(5); Z, S, y, T, hA, cz, cs, f = _setup(sz=0.0)
    A = rng.standard_normal((Z.shape[1], S.shape[1])) / np.sqrt(S.shape[1]); Z2 = S @ A.T
    sh, inn, U = c.shadow_innovation(Z2, S, y, K, hA, *f)
    assert np.max(np.abs(inn.M)) < 0.03 * np.max(np.abs(sh.M)) and np.max(np.abs(U)) < 0.03 * np.max(np.abs(Z2))  # gamma = 1% ridge bias


def test_participation_ratio_and_self_duality():
    assert abs(c.participation_ratio(np.random.default_rng(0).standard_normal((5000, 10))) - 10) < 0.3
    W = np.eye(3, 5); Fcm = np.r_[np.eye(3, 5), -np.ones((1, 5)) * 0]
    assert c.self_duality(W, Fcm[:3] + 1.0, np.eye(5)[:, :3])['row'] > 0.5
