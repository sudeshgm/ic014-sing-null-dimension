"""Round 8 core (PREREG_R8.md): theta_sem, theta_cls, class-level resampling, worlds C', offset-only, old C/M/D, descriptives.
Pure numpy. Everything works on class-mean matrices (K x p) and K x K Gram matrices of class means, so resampling classes
never pairs a class (or an image) with itself: theta_sem only ever pairs a train-fold class with a disjoint test-fold class,
and theta_cls only pairs the two disjoint image halves of a class."""
import numpy as np
from scipy.stats import t as tdist
from scipy.linalg import cho_factor, cho_solve

# ---------------------------------------------------------------- class means and Grams

def class_means(X, y, K, mask=None):
    """Plain class means of the rows of X (optionally only rows with mask True)."""
    idx = np.ones(len(y), bool) if mask is None else mask
    m = np.zeros((K, X.shape[1])); c = np.zeros(K)
    np.add.at(m, y[idx], X[idx]); np.add.at(c, y[idx], 1)
    return m / c[:, None]


def halves(y, K):
    """Half A = the first floor(n_c/2) images of each class in file order; half B = the rest."""
    A = np.zeros(len(y), bool); seen = np.zeros(K, int); n = np.bincount(y, minlength=K)
    for i, c in enumerate(y):
        if seen[c] < n[c] // 2: A[i] = True
        seen[c] += 1
    return A


class CM:
    """Class-mean triple of one block of coordinates: M (all images), MA, MB (image halves), and the per-image energy En.
    Pre-centred by the unweighted mean over classes (algebraically a no-op for every statistic, which re-centre inside each
    class fold; it removes a constant offset before any product is formed, which keeps float error small)."""
    def __init__(self, M, MA, MB, En):
        self.M, self.MA, self.MB, self.En = M, MA, MB, float(En)
        self.Mc = M - M.mean(0); self.MAc = MA - MA.mean(0); self.MBc = MB - MB.mean(0)
        self._G = None; self._GAB = None

    @classmethod
    def from_images(cls, X, y, K, hA):
        return cls(class_means(X, y, K), class_means(X, y, K, hA), class_means(X, y, K, ~hA), float(np.mean(np.sum(X ** 2, 1))))

    @property
    def G(self):
        if self._G is None: self._G = self.Mc @ self.Mc.T
        return self._G

    @property
    def GAB(self):
        if self._GAB is None: self._GAB = self.MAc @ self.MBc.T
        return self._GAB

    def scaled(self, s):
        return CM(self.M * s, self.MA * s, self.MB * s, self.En * s * s)

    def combo(self, a, other, b, En):
        """a*self + b*other (class means are linear in the images)."""
        return CM(a * self.M + b * other.M, a * self.MA + b * other.MA, a * self.MB + b * other.MB, En)

    def plus_class_code(self, U):
        """Add a class-level code U (K x p) to every image of class c; energy updated exactly (equal n per class assumed)."""
        En = self.En + 2 * float(np.mean(np.sum(self.M * U, 1))) + float(np.mean(np.sum(U ** 2, 1)))
        return CM(self.M + U, self.MA + U, self.MB + U, En)

    def permuted(self, pi):
        return CM(self.M[pi], self.MA[pi], self.MB[pi], self.En)


# ---------------------------------------------------------------- theta_sem

def lam_rule(Ttr):
    """lambda = mean eigenvalue of Ttr^T Ttr = tr(Ttr^T Ttr) / q (texts only)."""
    return float(np.sum(Ttr ** 2)) / Ttr.shape[1]


def ridge_H(Ttr, Tte, lam=None, KK=None):
    """H = Ttr (Ttr^T Ttr + lam I)^-1 Tte^T = (Ttr Ttr^T + lam I)^-1 Ttr Tte^T  (|tr| x |te|), kernel form.
    KK = (Ttr Ttr^T, Ttr Tte^T) may be passed precomputed (rows of the full text Gram)."""
    Ktt, Kte = (Ttr @ Ttr.T, Ttr @ Tte.T) if KK is None else KK
    lam = float(np.trace(Ktt)) / Ttr.shape[1] if lam is None else lam   # = lam_rule(Ttr)
    return cho_solve(cho_factor(Ktt + lam * np.eye(len(Ktt))), Kte)


def _S_block(G, H, itr, ite):
    X = G[np.ix_(itr, ite)]
    X = X - X.mean(1, keepdims=True) - X.mean(0, keepdims=True) + X.mean()   # = <m_c' - mbar_tr, m_c - mbar_te>
    return float(np.sum(H * X))


def S_multi(Gs, T, iA, iB, lam=None):
    """S(X) for each Gram in Gs, cross-fitted over class folds A, B (index arrays; may contain repeats, never shared).
    S = [sum_{c in B} <G_A t_c, m_c - mbar_B> + sum_{c in A} <G_B t_c, m_c - mbar_A>] / (|A| + |B|)."""
    TT = _text_gram(T)
    HAB = ridge_H(T[iA], T[iB], lam, (TT[np.ix_(iA, iA)], TT[np.ix_(iA, iB)]))
    HBA = ridge_H(T[iB], T[iA], lam, (TT[np.ix_(iB, iB)], TT[np.ix_(iB, iA)])); n = len(iA) + len(iB)
    return np.array([(_S_block(G, HAB, iA, iB) + _S_block(G, HBA, iB, iA)) / n for G in Gs])


_TT = {}


def _text_gram(T):
    key = (id(T), T.shape)
    if key not in _TT: _TT.clear(); _TT[key] = T @ T.T
    return _TT[key]


def theta_from_S(Sz, Ss):
    den = Sz + Ss
    return Sz / den if den > 0 else np.nan


def fold_split(K, seed):
    p = np.random.default_rng(seed).permutation(K); return np.sort(p[:K // 2]), np.sort(p[K // 2:])


def resample_indices(iA, iB, B, seed, method):
    """Class-level resampling inside each fold. 'boot': with replacement, same size. 'half': half of each fold without
    replacement. A class never appears in both folds of one draw."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        if method == 'boot': out.append((rng.choice(iA, len(iA), True), rng.choice(iB, len(iB), True)))
        else: out.append((rng.choice(iA, len(iA) // 2, False), rng.choice(iB, len(iB) // 2, False)))
    return out


def resampled_S(Gs, T, draws, lam=None):
    return np.array([S_multi(Gs, T, a, b, lam) for a, b in draws])   # (B, len(Gs))


def ci_from(theta_hat, boots, method):
    b = np.asarray(boots, float); b = b[np.isfinite(b)]
    if len(b) < 10: return [np.nan, np.nan], np.nan
    if method == 'boot':
        return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], float(np.std(b, ddof=1))
    d = b - theta_hat   # half-sampling: Var(theta_half - theta_hat) ~ Var(theta_hat); basic interval on the full-sample scale
    return [float(theta_hat - np.percentile(d, 97.5)), float(theta_hat - np.percentile(d, 2.5))], float(np.std(b, ddof=1))


def theta_sem_with_ci(cz, cs, T, folds, draws, method, lam=None):
    """Point theta_sem, S(Z), S(S) and resampling CI / SE (draws from resample_indices)."""
    Sz, Ss = S_multi([cz.G, cs.G], T, *folds, lam)
    th = theta_from_S(Sz, Ss); R = resampled_S([cz.G, cs.G], T, draws, lam)
    tb = np.array([theta_from_S(a, b) for a, b in R])
    ci, se = ci_from(th, tb, method)
    sz_ci, sz_se = ci_from(Sz, R[:, 0], method)
    return dict(theta_sem=float(th), S_Z=float(Sz), S_S=float(Ss), ci=ci, se=se, S_Z_ci=sz_ci, S_Z_se=sz_se,
                n_valid=int(np.isfinite(tb).sum())), tb


# ---------------------------------------------------------------- theta_cls / split-half between energy

def Eb(cm, idx=None):
    """Cross-fitted (split-half) centred between-class energy: mean_c <mA_c - mbarA, mB_c - mbarB> over classes idx."""
    idx = np.arange(len(cm.M)) if idx is None else idx
    d = np.einsum('ij,ij->i', cm.MAc[idx], cm.MBc[idx])
    return float(d.mean() - cm.GAB[np.ix_(idx, idx)].mean())


def theta_cls(cz, cs, idx=None):
    a, b = Eb(cz, idx), Eb(cs, idx); return a / (a + b) if (a + b) > 0 else np.nan


# ---------------------------------------------------------------- helpers for gates

def loglog_slope(x, y):
    x, y = np.log(np.asarray(x, float)), np.log(np.asarray(y, float)); X = np.c_[np.ones(len(x)), x]
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; n = len(x)
    se = np.sqrt(r @ r / (n - 2) * np.linalg.inv(X.T @ X)[1, 1]); q = tdist.ppf(0.975, n - 2)
    return dict(slope=float(b[1]), ci=[float(b[1] - q * se), float(b[1] + q * se)])


def derangement(K, rng):
    while True:
        p = rng.permutation(K)
        if not np.any(p == np.arange(K)): return p


def scale_to_rho(cm, Er, rho):
    """Rescale a null block so that its share of per-image energy is rho (row energy Er fixed)."""
    return cm.scaled((rho / (1 - rho) * Er / cm.En) ** 0.5)


def gaussian_surrogate_cm(mu, Ls, K, n_per_class, rng):
    """Class means of i.i.d. N(mu, Sigma) images, drawn exactly in distribution: half means ~ N(mu, Sigma / (n/2)),
    full mean = average of the two half means (equal halves). Ls: Sigma = Ls Ls^T."""
    h = n_per_class // 2; k = len(mu)
    MA = mu + rng.standard_normal((K, Ls.shape[1])) @ Ls.T / np.sqrt(h)
    MB = mu + rng.standard_normal((K, Ls.shape[1])) @ Ls.T / np.sqrt(n_per_class - h)
    M = (h * MA + (n_per_class - h) * MB) / n_per_class
    En = float(mu @ mu + np.sum(Ls ** 2))
    return CM(M, MA, MB, En)


# ---------------------------------------------------------------- worlds (mirror r4/calibration.py; C' and offset as PREREG_R8)

class WorldPieces:
    """Null coordinates C (N x k) of a base; mu = uncentred class means (incl. the offset); R = residuals permuted across
    images (seed 99000, round 5); Er = per-image row-space energy."""
    def __init__(self, C, y, K, hA, Er, resid_seed=99_000):
        cmu = class_means(C, y, K); R = (C - cmu[y])[np.random.default_rng(resid_seed).permutation(len(y))]
        self.mu = CM(cmu, cmu, cmu, float(np.mean(np.sum(cmu[y] ** 2, 1))))   # a class-constant block: halves equal the mean
        self.R = CM.from_images(R, y, K, hA)
        self.cross = float(np.mean(np.sum(cmu[y] * R, 1)))
        self.Er, self.Mm, self.Rm = float(Er), self.mu.En, self.R.En
        self.o = C.mean(0)

    def row(self, a2, b2):
        a, b = a2 ** 0.5, b2 ** 0.5; En = a2 * self.Mm + 2 * a * b * self.cross + b2 * self.Rm
        return self.mu.combo(a, self.R, b, En)

    def world(self, W, targets):
        """Rows of world W in {'C', 'M', 'D', "C'"}; targets = sorted E12 rho values."""
        h = 0.5 * targets[0]; rD = float(np.median(targets)); rows = []
        for j in range(len(targets)):
            rho = rD if W == 'D' else targets[j]; Et = self.Er / (1 - rho)
            if W in ('C', 'M', 'D'):
                L = h if W == 'C' else (rho - h if W == 'M' else rD * (j + 0.5) / len(targets))
                a2, b2 = L * Et / self.Mm, (rho - L) * Et / self.Rm
            elif W == "C'":
                L = h; a2 = h * (self.Er / (1 - targets[0])) / self.Mm
                b2 = (rho / (1 - rho) * self.Er - a2 * self.Mm) / self.Rm
            cm = self.row(a2, b2)
            rows.append(dict(W=W, L=L, rho_target=rho, rho=cm.En / (cm.En + self.Er), cm=cm))
        return rows

    def offset_world(self, base_cm, targets):
        """Add c * o_hat (o = the real null mean) to every image of base_cm until the null share reaches each target above it."""
        o = self.o / np.linalg.norm(self.o); zbar = base_cm.M.mean(0)   # equal n per class: image mean = mean of class means
        r0 = base_cm.En / (base_cm.En + self.Er); rows = []
        for rt in targets:
            if rt <= r0 * (1 + 1e-9): continue
            need = rt / (1 - rt) * self.Er; bq = 2 * float(zbar @ o); cq = base_cm.En - need
            c = (-bq + np.sqrt(bq * bq - 4 * cq)) / 2
            U = np.tile(c * o, (len(base_cm.M), 1)); cm = base_cm.plus_class_code(U)
            rows.append(dict(W='offset', rho_target=rt, rho=cm.En / (cm.En + self.Er), c=float(c), cm=cm))
        return rows


# ---------------------------------------------------------------- descriptives

def within_dev(X, y, K):
    return X - class_means(X, y, K)[y]


def shadow_innovation(Z, S, y, K, hA, iA, iB, gamma_frac=0.01):
    """H = Sw_zs (Sw_ss + gamma I)^-1 from within-class deviations of the classes of one fold, applied to the other fold's
    class means (and halves). Returns CM blocks for the shadow H*m_s and the innovation m_z - H*m_s, and the full per-image
    innovation U = Z - S H^T assembled fold-wise (for J_inn)."""
    Dz, Ds = within_dev(Z, y, K), within_dev(S, y, K)
    inA = np.isin(y, iA); M = {}
    out_sh = [np.zeros((K, Z.shape[1])) for _ in range(3)]; out_in = [np.zeros((K, Z.shape[1])) for _ in range(3)]
    U = np.zeros_like(Z); czM = [class_means(Z, y, K), class_means(Z, y, K, hA), class_means(Z, y, K, ~hA)]
    csM = [class_means(S, y, K), class_means(S, y, K, hA), class_means(S, y, K, ~hA)]
    for fit_mask, app in ((inA, iB), (~inA, iA)):
        dz, ds = Dz[fit_mask], Ds[fit_mask]; n = len(dz)
        Sss = ds.T @ ds / n; Szs = dz.T @ ds / n; g = gamma_frac * float(np.trace(Sss)) / Sss.shape[0]
        Hm = np.linalg.solve(Sss + g * np.eye(len(Sss)), Szs.T).T   # k x r
        for j in range(3):
            sh = csM[j][app] @ Hm.T; out_sh[j][app] = sh; out_in[j][app] = czM[j][app] - sh
        rows = np.isin(y, app); U[rows] = Z[rows] - S[rows] @ Hm.T
    Esh = float(np.mean(np.sum(out_sh[0][y] ** 2, 1))); Ein = float(np.mean(np.sum(out_in[0][y] ** 2, 1)))
    return CM(*out_sh, Esh), CM(*out_in, Ein), U


def ledoit_wolf(X):
    """Ledoit-Wolf shrinkage covariance of centred rows X (n x p) towards mu*I."""
    n, p = X.shape; S = X.T @ X / n; mu = np.trace(S) / p
    d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
    b2 = min(d2, np.sum([np.sum((np.outer(x, x) - S) ** 2) for x in X]) / (n * n) / p)
    s = b2 / d2 if d2 > 0 else 0.0
    return s * mu * np.eye(p) + (1 - s) * S


def fisher_J(X, y, K, hA, iA, iB, kmax=64, rng=None):
    """Descriptive class information tr(Sigma_w^-1 B) in the kmax leading within-class PCs: PCs and Ledoit-Wolf Sigma_w
    from one fold's within-class deviations (subsampled to <= 2000 images for the LW term), B from the other fold's split-half
    class means (crossnobis-style, no image paired with itself); averaged over the two directions."""
    rng = np.random.default_rng(0) if rng is None else rng
    D = within_dev(X, y, K); MA, MB = class_means(X, y, K, hA), class_means(X, y, K, ~hA); vals = []
    for fit, app in ((iA, iB), (iB, iA)):
        d = D[np.isin(y, fit)]; C = d.T @ d / len(d); ev, V = np.linalg.eigh(C); V = V[:, ::-1][:, :min(kmax, X.shape[1])]
        dv = d @ V; sub = dv[rng.choice(len(dv), min(2000, len(dv)), False)]
        Sw = ledoit_wolf(sub - sub.mean(0))
        a = (MA[app] - MA[app].mean(0)) @ V; b = (MB[app] - MB[app].mean(0)) @ V; Bm = (a.T @ b + b.T @ a) / (2 * len(app))
        vals.append(float(np.trace(np.linalg.solve(Sw, Bm))))
    return float(np.mean(vals))


def self_duality(W, Fcm, Vr):
    """Mean over classes of cos(W_c, m_c - mbar) for the full class mean and for its row-space part."""
    Mc = Fcm - Fcm.mean(0); Mr = Mc @ Vr @ Vr.T
    cos = lambda A, B: np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-300)
    return dict(full=float(np.mean(cos(W, Mc))), row=float(np.mean(cos(W, Mr))))


def participation_ratio(D):
    """(tr S)^2 / tr(S^2) of the covariance of within-class deviations D (n x p), via the smaller Gram side."""
    n, p = D.shape
    S = (D.T @ D if p <= n else D @ D.T) / n
    return float(np.trace(S) ** 2 / np.sum(S * S))
