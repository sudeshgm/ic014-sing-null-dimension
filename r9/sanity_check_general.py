"""Sanity check for scores that compare models through a discarded subspace of their features
(the null space of a linear head, a PCA residual, ...). Standalone: numpy, scipy and scikit-learn only.

You supply
  models    {name: dict(F=(N,d) features, y=(N,) int labels, W=(K,d) head weight, b=(K,) head bias, fit=(N,) bool mask
            of images used for fitting, plus any extra keys your score needs)}
  score_fn  score_fn(model_dict) -> float, the per-model value of your score. It must read only the dict it is given;
            the tool passes modified copies of each model (features replaced in the subspace, or a different head).
  subspace_fn (optional) subspace_fn(model_dict) -> (V, origin): orthonormal basis (d,k) of the subspace your score reads
            and an origin (d,) or None. Default: null space of model['W'] (singular-value tolerance 1e-5), origin None.

The check has three steps.
  1  centre     split the subspace energy into offset, between-class and within-class parts, and recompute the score
                with the offset only and with the offset removed.
  2  surrogate  recompute the score with the subspace coordinates replaced by content-free Gaussian draws that keep
                each model's own mean and covariance (so its energy and offset), and by the same draws without the mean.
                If the cross-model order survives this, the order does not by itself show content.
  3  head       refit the head (nearest class mean, shrinkage LDA, a ridge head distilled from the given logits, and the
                given head with a small random perturbation chosen to stay within 1 point of its top-1), report top-1 on
                the images not used for fitting, and recompute the score with each head.

Verdict on step 2 (tau_E = Kendall tau-b between the real and the surrogate per-model values, model-bootstrap BCa 95% CI):
  ENERGY-DRIVEN     lower bound of tau_E > 0.5
  CONTENT-READING   upper bound of tau_E < 0.5 and the surrogate gap carries at least half the cross-model variance
  MIXED             otherwise
  plus 'order-equivalent to surrogate' when the BCa 90% lower bound of tau_E exceeds 0.9 (|1 - tau_E| < 0.1).
With few models these intervals are wide; read MIXED as 'not resolved', not as 'half and half'.

Example
  from sanity_check_general import run_check, null_share_score
  out = run_check(models, null_share_score)
  print(out['verdict'])
"""
import copy
import numpy as np
from scipy.stats import kendalltau, norm

TOL = 1e-5
THRESH = dict(energy_lo=0.5, content_hi=0.5, content_gap=0.5, equiv=0.9)


# ------------------------------------------------------------------ linear algebra
def null_basis(W, tol=TOL):
    """Orthonormal basis (d, d - rank) of the null space of W (K, d), absolute singular-value tolerance."""
    _, sv, vt = np.linalg.svd(np.asarray(W, np.float64), full_matrices=True)
    return vt[int((sv > tol).sum()):].T


def default_subspace(m):
    """null space of m['W']; uses m['Vn'] when the caller has cached it."""
    return (m['Vn'] if 'Vn' in m else null_basis(m['W'])), None


def coords(F, V, origin=None):
    """Coordinates of F - origin in the subspace V: an orthonormal basis (d, k), or a slice when the features are
    already expressed in a basis whose trailing columns span the subspace (fast path)."""
    X = F if origin is None else F - origin
    return X[:, V] if isinstance(V, slice) else X @ V


def replace(F, V, Cnew, origin=None):
    """F with its coordinates in span(V) (measured from origin) replaced by Cnew."""
    if isinstance(V, slice):
        G = np.array(F, copy=True); G[:, V] = Cnew if origin is None else Cnew + np.asarray(origin)[V]; return G
    return F - coords(F, V, origin) @ V.T + Cnew @ V.T


def moment_sqrt(C):
    """Mean and a square root L of the covariance of the rows of C (L @ L.T = cov)."""
    C = np.asarray(C); mu = C.mean(0); Cc = C - mu; S = (Cc.T @ Cc / len(C)).astype(np.float64)
    try:
        return mu, np.linalg.cholesky(S + 1e-12 * np.trace(S) / len(S) * np.eye(len(S))).astype(C.dtype)
    except np.linalg.LinAlgError:
        ev, U = np.linalg.eigh(S)
        return mu, (U * np.sqrt(np.clip(ev, 0, None))).astype(C.dtype)


def gaussian_draw(n, mu, L, rng, centred=False):
    z = rng.standard_normal((n, L.shape[1]), dtype=np.float32 if L.dtype == np.float32 else np.float64) @ L.T
    return z if centred else mu + z


# ------------------------------------------------------------------ statistics
def ktau(x, y):
    return kendalltau(x, y).statistic


def ktau_rows(X, Y):
    """Kendall tau-b for each row pair of X, Y (B, n); nan where a row is constant."""
    X, Y = np.atleast_2d(X), np.atleast_2d(Y); n = X.shape[1]; iu = np.triu_indices(n, 1)
    sx = np.sign(X[:, :, None] - X[:, None, :])[:, iu[0], iu[1]]; sy = np.sign(Y[:, :, None] - Y[:, None, :])[:, iu[0], iu[1]]
    den = np.sqrt((sx != 0).sum(1) * (sy != 0).sum(1)).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(den > 0, (sx * sy).sum(1) / den, np.nan)


def boot_idx(n, B=2000, seed=0, min_distinct=4):
    r = np.random.default_rng(seed); out = []
    for _ in range(B):
        i = r.integers(0, n, n)
        if len(set(i.tolist())) >= min_distinct: out.append(i)
    return out


def bca(theta, boots, jack, alpha=0.05):
    boots = np.asarray([b for b in boots if np.isfinite(b)]); jack = np.asarray(jack, float)
    p = np.mean(boots < theta) + 0.5 * np.mean(boots == theta)
    p = min(max(p, 1.0 / (len(boots) + 1)), 1 - 1.0 / (len(boots) + 1))
    z0 = norm.ppf(p); jm = np.nanmean(jack); num = np.nansum((jm - jack) ** 3); den = 6 * np.nansum((jm - jack) ** 2) ** 1.5
    a = num / den if den > 0 else 0.0
    out = []
    for z in (norm.ppf(alpha / 2), norm.ppf(1 - alpha / 2)):
        out.append(float(np.percentile(boots, 100 * norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))))))
    return out


def tau_ci(x, y, B=2000, seed=0):
    """Kendall tau-b with model-bootstrap percentile, BCa 95% and BCa 90% intervals."""
    x, y = np.asarray(x, float), np.asarray(y, float); n = len(x); t = ktau(x, y)
    I = np.array(boot_idx(n, B, seed)); bs = ktau_rows(x[I], y[I]); bs = bs[np.isfinite(bs)]
    J = np.array([np.delete(np.arange(n), j) for j in range(n)]); jk = ktau_rows(x[J], y[J])
    if not np.isfinite(t) or len(bs) < 10:
        return dict(tau=float(t), pct=[np.nan, np.nan], bca=[np.nan, np.nan], bca90=[np.nan, np.nan], n=n)
    return dict(tau=float(t), pct=[float(v) for v in np.percentile(bs, [2.5, 97.5])], bca=bca(t, bs, jk),
                bca90=bca(t, bs, jk, alpha=0.10), n=n)


def paired_ci(x1, x2, y, B=2000, seed=0):
    """tau(x1, y) - tau(x2, y) under the same model resamples: BCa 95% and 90% intervals."""
    x1, x2, y = (np.asarray(v, float) for v in (x1, x2, y)); n = len(y)
    d0 = ktau(x1, y) - ktau(x2, y); I = np.array(boot_idx(n, B, seed))
    D = ktau_rows(x1[I], y[I]) - ktau_rows(x2[I], y[I]); D = D[np.isfinite(D)]
    J = np.array([np.delete(np.arange(n), j) for j in range(n)]); jk = ktau_rows(x1[J], y[J]) - ktau_rows(x2[J], y[J])
    return dict(diff=float(d0), bca=bca(d0, D, jk), bca90=bca(d0, D, jk, alpha=0.10))


# ------------------------------------------------------------------ step 1: energy decomposition
def rho_decomposition(F, y, V, origin=None):
    """E||c||^2 / E||f||^2 for c = (f - origin) V, split exactly into offset + between-class + within-class parts."""
    F = np.asarray(F, np.float64); y = np.asarray(y).astype(int); Z = coords(F, V, origin)
    Etot = float(np.mean(np.sum(F ** 2, 1))); N = len(y); K = int(y.max()) + 1
    n = np.bincount(y, minlength=K).astype(float); M = class_means(Z, y, K)
    zbar = Z.mean(0); off = float(zbar @ zbar); btw = float(np.sum(n[:, None] * (M - zbar) ** 2) / N)
    wit = float(np.sum((Z - M[y]) ** 2) / N); En = float(np.mean(np.sum(Z ** 2, 1)))
    return dict(rho=En / Etot, k=int(Z.shape[1]), rho_offset=off / Etot, rho_between=btw / Etot, rho_within=wit / Etot,
                rho_centred=(btw + wit) / Etot, identity_abs_err=abs(En - off - btw - wit) / Etot,
                share_offset=off / En if En > 0 else float('nan'))


# ------------------------------------------------------------------ step 1 + 2: variants of one model
def with_features(m, F):
    o = dict(m); o['F'] = F; return o


def as_dict(v):
    """score_fn may return one number or a dict of named numbers."""
    return {k: float(x) for k, x in v.items()} if isinstance(v, dict) else {'score': float(v)}


def variants(m, score_fn, subspace_fn=default_subspace, draws=3, rng=None):
    """Score on the real features and on content-free variants of the subspace coordinates.
    Returns {score name: dict(real, offset_only, centred, surrogate, surrogate_centred, ...)}."""
    rng = np.random.default_rng(0) if rng is None else rng
    F = np.asarray(m['F'], np.float64); V, org = subspace_fn(m); C = coords(F, V, org); mu, L = moment_sqrt(C); N = len(F)
    real = as_dict(score_fn(m))
    off = as_dict(score_fn(with_features(m, replace(F, V, np.broadcast_to(mu, C.shape), org))))
    cen = as_dict(score_fn(with_features(m, replace(F, V, C - mu, org))))
    s, sc = [], []
    for _ in range(draws):
        s.append(as_dict(score_fn(with_features(m, replace(F, V, gaussian_draw(N, mu, L, rng), org)))))
        sc.append(as_dict(score_fn(with_features(m, replace(F, V, gaussian_draw(N, mu, L, rng, centred=True), org)))))
    out = {}
    for k in real:
        d = [x[k] for x in s]; dc = [x[k] for x in sc]
        out[k] = dict(real=real[k], offset_only=off[k], centred=cen[k], surrogate=float(np.mean(d)), surrogate_sd=float(np.std(d)),
                      surrogate_centred=float(np.mean(dc)), surrogate_centred_sd=float(np.std(dc)), surrogate_draws=d,
                      surrogate_centred_draws=dc)
    return out


def verdict_step2(real, surr, B=2000, seed=0, th=THRESH):
    real, surr = np.asarray(real, float), np.asarray(surr, float)
    t = tau_ci(real, surr, B, seed); vr = np.var(real)
    gap = float(np.var(real - surr) / vr) if vr > 0 else float('nan')
    lo, hi = t['bca']
    if not np.isfinite(lo):
        v = 'NOT ESTIMABLE'
    elif lo > th['energy_lo']:
        v = 'ENERGY-DRIVEN'
    elif hi < th['content_hi'] and gap >= th['content_gap']:
        v = 'CONTENT-READING'
    else:
        v = 'MIXED'
    return dict(verdict=v, tau_E=t, gap_share=gap, order_equivalent=bool(np.isfinite(t['bca90'][0]) and t['bca90'][0] > th['equiv']))


# ------------------------------------------------------------------ step 3: heads
def class_means(F, y, K):
    from scipy.sparse import csr_matrix
    y = np.asarray(y).astype(int); Oh = csr_matrix((np.ones(len(y)), (y, np.arange(len(y)))), shape=(K, len(y)))
    return np.asarray(Oh @ F) / np.maximum(np.bincount(y, minlength=K), 1)[:, None]


def top1(F, y, W, b):
    return float(np.mean(np.argmax(F @ W.T + b, 1) == y))


def fit_heads(m, kinds=('NCM', 'LDA', 'ridge', 'perturbed'), seed=0, eps_grid=(0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01),
              ridge_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0), match_pts=1.0):
    """Heads fitted on m['fit'] images; top-1 reported on the other images. Returns {name: dict(W, b, top1, ...)}."""
    from sklearn.covariance import ledoit_wolf
    F = np.asarray(m['F'], np.float64); y = np.asarray(m['y']).astype(int); A = np.asarray(m['fit'], bool); K = m['W'].shape[0]
    W0, b0 = np.asarray(m['W'], np.float64), np.asarray(m['b'], np.float64)
    out = dict(released=dict(W=W0, b=b0, top1=top1(F[~A], y[~A], W0, b0)))
    Mh = class_means(F[A], y[A], K)
    if 'NCM' in kinds:
        out['NCM'] = dict(W=Mh, b=-0.5 * np.sum(Mh ** 2, 1))
    if 'LDA' in kinds:
        D = F[A] - Mh[y[A]]; Sig, shrink = ledoit_wolf(D, assume_centered=True); Wl = np.linalg.solve(Sig, Mh.T).T
        out['LDA'] = dict(W=Wl, b=-0.5 * np.sum(Wl * Mh, 1), shrinkage=float(shrink))
    if 'ridge' in kinds:
        # ridge regression of the given logits on the features (fit images), lambda by 2-fold split of the fit images
        L0 = F @ W0.T + b0; idx = np.where(A)[0]; r = np.random.default_rng(seed); r.shuffle(idx); h = len(idx) // 2
        def ridge(ix, lam):
            X = F[ix]; xm = X.mean(0); Xc = X - xm; lm = L0[ix].mean(0); s = lam * np.trace(Xc.T @ Xc) / Xc.shape[1]
            Wr = np.linalg.solve(Xc.T @ Xc + s * np.eye(Xc.shape[1]), Xc.T @ (L0[ix] - lm)).T
            return Wr, lm - Wr @ xm
        err = {}
        for lam in ridge_grid:
            e = 0.0
            for a_, b_ in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
                Wr, br = ridge(a_, lam); e += float(np.mean((F[b_] @ Wr.T + br - L0[b_]) ** 2))
            err[lam] = e
        lam = min(err, key=err.get); Wr, br = ridge(idx, lam)
        out['ridge'] = dict(W=Wr, b=br, lam=lam)
    if 'perturbed' in kinds:
        # largest eps on the grid whose perturbed head keeps top-1 on the fit images within match_pts of the given head
        r = np.random.default_rng(seed); G = r.standard_normal(W0.shape); G *= np.linalg.norm(W0) / np.linalg.norm(G)
        a0 = top1(F[A], y[A], W0, b0); chosen = None
        for e in eps_grid:
            Wp = W0 + e * G
            if abs(top1(F[A], y[A], Wp, b0) - a0) * 100 <= match_pts: chosen = e; break
        if chosen is not None:
            out['perturbed'] = dict(W=W0 + chosen * G, b=b0, eps=chosen)
    for k, h in out.items():
        h['top1'] = top1(F[~A], y[~A], h['W'], h['b'])
        h['matched'] = bool(abs(h['top1'] - out['released']['top1']) * 100 <= match_pts)
    return out


def with_head(m, W, b):
    """model dict with a different head. The cached null basis 'Vn' and every key starting with '_' (caches that
    depend on the head) are dropped."""
    o = {k: v for k, v in m.items() if k != 'Vn' and not k.startswith('_')}; o['W'] = W; o['b'] = b; return o


# ------------------------------------------------------------------ the whole check
def run_check(models, score_fn, subspace_fn=default_subspace, draws=3, seed=0, heads=True, B=2000, th=THRESH,
              head_kinds=('NCM', 'LDA', 'ridge', 'perturbed'), log=None):
    """models: {name: model dict}. Returns {'scores': {score name: result}} (one entry per score score_fn returns),
    and for a single-number score_fn also the result of that score at the top level."""
    names = list(models); per = {}
    for i, n in enumerate(names):
        m = dict(models[n]); rng = np.random.default_rng([seed, i])
        if subspace_fn is default_subspace and 'Vn' not in m:
            m['Vn'] = null_basis(m['W'])
        V, org = subspace_fn(m)
        r = dict(variants=variants(m, score_fn, subspace_fn, draws, rng), decomp=rho_decomposition(m['F'], m['y'], V, org))
        if heads and 'fit' in m:
            H = fit_heads(m, head_kinds, seed=seed + i); r['heads'] = {}
            for k, h in H.items():
                sc = as_dict(score_fn(m if k == 'released' else with_head(m, h['W'], h['b'])))
                r['heads'][k] = dict(top1=h['top1'], matched=h['matched'], score=sc,
                                     **{q: h[q] for q in ('eps', 'lam', 'shrinkage') if q in h})
        per[n] = r
        if log: log(n, {k: v['real'] for k, v in r['variants'].items()})
    out = dict(models=names, per_model=per, scores={})
    for k in per[names[0]]['variants']:
        col = lambda q: np.array([per[n]['variants'][k][q] for n in names])
        o = dict(step2=verdict_step2(col('real'), col('surrogate'), B, seed, th),
                 step2_centred=dict(tau=tau_ci(col('real'), col('surrogate_centred'), B, seed)),
                 step1=dict(tau_offset_only=tau_ci(col('real'), col('offset_only'), B, seed),
                            tau_centred=tau_ci(col('real'), col('centred'), B, seed)))
        lo = o['step1']['tau_offset_only']['bca'][0]
        o['step1']['offset_dominated'] = bool(np.isfinite(lo) and lo > th['energy_lo'])
        if heads and all('heads' in per[n] for n in names):
            o['step3'] = head_summary(per, names, k, B, seed)
        o['verdict'] = o['step2']['verdict']
        out['scores'][k] = o
    if list(out['scores']) == ['score']:
        out.update(out['scores']['score'])
    return out


def head_summary(per, names, key='score', B=2000, seed=0, min_models=8):
    """tau between the per-model scores under the given head and under each refitted head. 'matched' uses only the
    models whose refitted head is within 1 top-1 point of the given head."""
    kinds = sorted({k for n in names for k in per[n]['heads'] if k != 'released'})
    o = {}
    for k in kinds:
        ms = [n for n in names if k in per[n]['heads']]
        x = np.array([per[n]['heads']['released']['score'][key] for n in ms]); z = np.array([per[n]['heads'][k]['score'][key] for n in ms])
        mm = [n for n in ms if per[n]['heads'][k]['matched']]
        d = dict(n=len(ms), tau_all=tau_ci(x, z, B, seed) if len(ms) >= 4 else None, n_matched=len(mm),
                 top1_gap_pts=[100 * (per[n]['heads'][k]['top1'] - per[n]['heads']['released']['top1']) for n in ms])
        if len(mm) >= min_models:
            t = tau_ci([per[n]['heads']['released']['score'][key] for n in mm], [per[n]['heads'][k]['score'][key] for n in mm], B, seed)
            d['tau_matched'] = t
            d['reading'] = 'HEAD-DEPENDENT' if t['tau'] <= 0.5 else ('HEAD-STABLE' if t['tau'] >= 0.8 else 'INTERMEDIATE')
        else:
            d['tau_matched'] = None; d['reading'] = 'NOT ESTIMABLE (fewer than %d matched backbones)' % min_models
        o[k] = d
    return o


# ------------------------------------------------------------------ example scores (usable as templates)
def null_share_score(m):
    """Mean ||P_null f|| / ||f|| over the non-fit images (NuSA-style null share, Cook et al. 2020)."""
    F = np.asarray(m['F'], np.float64); Vn = default_subspace(m)[0]; ev = ~np.asarray(m.get('fit', np.zeros(len(F), bool)), bool)
    return float(np.mean(np.linalg.norm(F[ev] @ Vn, axis=1) / np.linalg.norm(F[ev], axis=1)))


def null_energy_score(m):
    """rho = E||P_null f||^2 / E||f||^2 over the non-fit images."""
    F = np.asarray(m['F'], np.float64); Vn = default_subspace(m)[0]; ev = ~np.asarray(m.get('fit', np.zeros(len(F), bool)), bool)
    return float(np.mean(np.sum((F[ev] @ Vn) ** 2, 1)) / np.mean(np.sum(F[ev] ** 2, 1)))


def split_first_half(y, per_class=None):
    """fit mask: the first half of each class's images, in the given order."""
    y = np.asarray(y).astype(int); A = np.zeros(len(y), bool); seen = {}
    cnt = np.bincount(y)
    for i, c in enumerate(y):
        k = seen.get(c, 0); half = (per_class or cnt[c]) // 2
        if k < half: A[i] = True
        seen[c] = k + 1
    return A
