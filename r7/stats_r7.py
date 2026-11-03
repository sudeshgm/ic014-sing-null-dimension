"""PREREG_R7 statistics on null-space edit vectors from independent nuisance views.

D: array (V=4, N, k) of edit vectors Delta_{i,v} (view v, image i, k null coordinates). g: (N,) group (foreground IN-9 class).
R: (N, q) external semantic reference (text embedding of the background's ImageNet class).

Population target (per group, linear kernels): the RV / linear-CKA coefficient between the noise-free edits mu_i and R_i,
  T* = sum_g ||Cov_g(mu, R)||_F^2 / sqrt(sum_g ||Cov_g(mu, mu)||_F^2 * sum_g ||Cov_g(R, R)||_F^2).
Estimator. Every term is Song et al.'s (2012) unbiased HSIC U-statistic, which uses off-diagonal kernel entries only:
  numerator   HSIC_u(Gbar, K) with Gbar = mean over ordered view pairs a != b of D_a D_b^T (unbiased off the diagonal even
              for one view, since noise is independent across images);
  denominator HSIC_u(S_ab, S_cd), averaged over the 3 pairings {ab|cd} of the 4 views (S_ab symmetrised); two independent
              copies make E[S_ab,ij S_cd,ij] = (mu_i.mu_j)^2, so no noise term enters whatever the noise covariance;
  HSIC_u(K, K) for the reference.
T = num / sqrt(den * kk); undefined (nan) if den <= 0. Mode 'wc': sums over the 9 groups (within-group structure only);
'gl': all N images as one group. Rescaling every Delta by s > 0 leaves T unchanged exactly."""
import numpy as np

PAIRINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))


def hsic_u(K, L):
    """Song et al. (2012) unbiased HSIC; K, L symmetric n x n (diagonals ignored)"""
    n = K.shape[0]; K = K.copy(); L = L.copy(); np.fill_diagonal(K, 0); np.fill_diagonal(L, 0)
    one_K = K.sum(1); one_L = L.sum(1)
    return float((np.sum(K * L) + one_K.sum() * one_L.sum() / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * one_K @ one_L) / (n * (n - 3)))


def sym(A, B): G = A @ B.T; return 0.5 * (G + G.T)


def _groups(g, mode): return [np.arange(len(g))] if mode == 'gl' else [np.where(g == c)[0] for c in np.unique(g)]


def cross_gram(Dg):
    V = Dg.shape[0]; S = Dg.sum(0); return (S @ S.T - sum(Dg[v] @ Dg[v].T for v in range(V))) / (V * (V - 1))


def tr_u(G):
    """unbiased estimate of n x tr Cov(mu) from a Gram whose entries are unbiased for mu_i.mu_j (diagonal included)"""
    n = G.shape[0]; return float(n * (np.trace(G) - G.sum() / n) / (n - 1))


def parts(D, R, g, mode='wc', idx=None):
    """per-group sums of the numerator, both denominators and the reference term. idx: optional bootstrap row index.
    Bootstrap duplicates are harmless: the cross-view Gram is unbiased on its diagonal too."""
    if idx is not None: D, R, g = D[:, idx], R[idx], g[idx]
    num = den = kk = tr = 0.0
    for m in _groups(g, mode):
        Dg = D[:, m].astype(np.float64); K = R[m] @ R[m].T; G = cross_gram(Dg)
        num += hsic_u(G, K); kk += hsic_u(K, K); tr += tr_u(G) / len(m)
        if Dg.shape[0] == 4: den += float(np.mean([hsic_u(sym(Dg[a], Dg[b]), sym(Dg[c], Dg[d])) for (a, b), (c, d) in PAIRINGS]))
    return num, den, kk, tr


def T_stat(D, R, g, mode='wc', idx=None):
    """T  (RV form, 4th-order denominator) and T2 (first-order denominator):
    T2 = sum_g HSIC_u(Gbar, K) / (sum_g tr_u Cov_g(mu) * sqrt(sum_g HSIC_u(K, K))), population value
    sum_g ||Cov_g(mu, R)||^2 / (sum_g tr Cov_g(mu) * sqrt(sum_g ||Cov_g(R)||^2)), in [0, 1] up to the group pooling."""
    num, den, kk, tr = parts(D, R, g, mode, idx)
    return dict(T=num / np.sqrt(den * kk) if den > 0 and kk > 0 else np.nan, T2=num / (tr * np.sqrt(kk)) if tr > 0 and kk > 0 else np.nan,
                num=num, den=den, kk=kk, tr=tr)


def T_naive(D, R, g, mode='wc'):
    """classic same-view linear CKA with diagonals (double-centred, within group), averaged over views: noise-attenuated,
    used only as the manipulation check that a nuisance level really adds noise"""
    out = []
    for v in range(D.shape[0]):
        num = gg = kk = 0.0
        for m in _groups(g, mode):
            X = D[v, m] - D[v, m].mean(0); Y = R[m] - R[m].mean(0); G = X @ X.T; K = Y @ Y.T
            num += np.sum(G * K); gg += np.sum(G * G); kk += np.sum(K * K)
        out.append(num / np.sqrt(gg * kk))
    return float(np.mean(out))


def control_num(Dctl, R, g, mode='wc'):
    """control numerator (>= 2 independent copies, e.g. nuisance-only differences); put on the edit's scale by dividing
    by the edit's own T or T2 denominator"""
    return float(sum(hsic_u(cross_gram(Dctl[:, m].astype(np.float64)), R[m] @ R[m].T) for m in _groups(g, mode)))


def reliability(D, g, mode='wc'):
    """split-half reliability of the (group-centred) edit vectors: views {0,1} vs {2,3}"""
    A = D[:2].mean(0).astype(np.float64); B = D[2:].mean(0).astype(np.float64)
    for m in _groups(g, mode): A[m] -= A[m].mean(0); B[m] -= B[m].mean(0)
    return float(np.sum(A * B) / np.sqrt(np.sum(A * A) * np.sum(B * B)))


def noise_ratio(D, g, mode='wc'):
    """nu = (mean per-view centred energy - cross-view centred energy) / cross-view centred energy"""
    per = cross = 0.0
    for m in _groups(g, mode):
        Dg = D[:, m].astype(np.float64); Dg = Dg - Dg.mean(1, keepdims=True)
        per += float(np.mean([np.sum(Dg[v] ** 2) for v in range(Dg.shape[0])])); cross += float(np.trace(cross_gram(Dg)))
    return dict(nu=(per - cross) / cross if cross > 0 else np.inf, cross=cross, per=per)


def perm_within(g, rng):
    idx = np.arange(len(g))
    for c in np.unique(g):
        m = np.where(g == c)[0]; idx[m] = rng.permutation(m)
    return idx


def boot_within(g, rng):
    return np.concatenate([rng.choice(np.where(g == c)[0], int((g == c).sum()), replace=True) for c in np.unique(g)])


def perm_test(D, R, g, nperm, seed, mode='wc'):
    """within-group permutation of the reference rows (den and kk are permutation-invariant); one-sided p for the numerator"""
    rng = np.random.default_rng(seed); groups = _groups(g, mode); Gs = [cross_gram(D[:, m].astype(np.float64)) for m in groups]
    def num(Rp): return sum(hsic_u(G, Rp[m] @ Rp[m].T) for G, m in zip(Gs, groups))
    obs = num(R); null = np.array([num(R[perm_within(g, rng)]) for _ in range(nperm)])
    return dict(p=float((1 + np.sum(null >= obs)) / (1 + nperm)), z=float((obs - null.mean()) / null.std()))
