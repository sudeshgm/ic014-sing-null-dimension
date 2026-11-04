"""Round 9 scores. Each function takes one model dict (see sanity_check_general.py) and returns per-model values
averaged over the evaluation images (~m['fit']); fitted quantities use the fit images m['fit'].

Optional keys used as fast paths (all start with '_' so that a head swap drops them):
  _null_from  int s: the features are expressed in a basis whose columns s: span null(W) (rotated coordinates)
  _vim_frozen int D: the features are expressed in ViM's eigenbasis about u (principal space = first D columns);
              the basis is kept, alpha is refitted
  _L          cached logits (valid while only null(W) coordinates change)
  _as_row     cached (A f_row + Ab, angle to t_y) for the evaluation images (valid while only null(W) changes)
  _rown2, _rowo2  cached ||f_row||^2 and ||f_row + o||^2 (null fast path)
  _N          null coordinates given directly (controls); then F is not read by the null family
  _pn2, _Lp, _WR  ViM fast path: ||principal part||^2, logits from the principal part, head columns on the residual
  _R          ViM residual coordinates given directly (controls)
"""
import numpy as np
from scipy.special import expit, logsumexp
import sanity_check_general as G


def _ev(m):
    return ~np.asarray(m['fit'], bool)


def null_coords(m, X=None):
    X = m['F'] if X is None else X
    if '_null_from' in m:
        return X[:, m['_null_from']:]
    return X @ G.default_subspace(m)[0]


def head_offset(W, b):
    """o = pinv(W) b, the minimum-norm solution of W o = b (W has full row rank)."""
    W = np.asarray(W, np.float64)
    return W.T @ np.linalg.solve(W @ W.T, np.asarray(b, np.float64))


def logits(m):
    return m['_L'] if '_L' in m else m['F'] @ np.asarray(m['W']).T + m['b']


# ---------------------------------------------------------------- null(W) family
def null_family(m):
    """nsp, nusa, null_norm, rho, core_null (and as_sing when the translator A, Ab and class text T are given)."""
    ev = _ev(m); fit = ~ev; N = m['_N'] if '_N' in m else null_coords(m); nn = np.linalg.norm(N, axis=1)
    if '_rown2' in m:
        fn = np.sqrt(m['_rown2'] + nn ** 2); fo = np.sqrt(m['_rowo2'] + nn ** 2)
    else:
        F = m['F']; fn = np.linalg.norm(F, axis=1); fo = np.linalg.norm(F + head_offset(m['W'], m['b']), axis=1)
    out = dict(nsp=float(np.mean(nn[ev] / fo[ev])),                  # He et al. 2025: ||x^{W-perp}|| / ||x||, x offset by pinv(W)b
               nusa=float(np.mean(nn[ev] / fn[ev])),                 # Cook et al. 2020: null-space share
               null_norm=float(np.mean(nn[ev])),                      # pure energy (negative control score)
               rho=float(np.mean(nn[ev] ** 2) / np.mean(fn[ev] ** 2)))
    y = np.asarray(m['y']).astype(int); K = np.asarray(m['W']).shape[0]
    M = G.class_means(N[fit], y[fit], K); yh = np.argmax(logits(m)[ev], 1); Mm = M[yh]
    out['core_null'] = float(np.mean(np.sum(N[ev] * Mm, 1) / np.maximum(nn[ev] * np.linalg.norm(Mm, axis=1), 1e-300)))
    if 'A' in m:
        out['as_sing'] = as_sing(m, N)
    return out


def _ang(Z, T):
    c = np.sum(Z * T, 1) / np.maximum(np.linalg.norm(Z, axis=1), 1e-300)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def as_sing(m, N=None):
    """SING alignment shift: mean |theta(A f + Ab, t_y) - theta(A f_row + Ab, t_y)| over the evaluation images,
    t_y the normalised mean of the normalised class prompt embeddings."""
    ev = _ev(m); y = np.asarray(m['y']).astype(int)[ev]; T = m['T'][y]; A = np.asarray(m['A'], np.float64)
    if '_as_row' in m:
        Zrow, th_row = m['_as_row']
    else:
        Zrow = as_row(m)[ev]; th_row = _ang(Zrow, T)
    if '_null_from' in m:
        s = m['_null_from']; Nn = N[ev] if N is not None else m['F'][ev, s:]
        Zf = Zrow + Nn @ (m['_A_null'] if '_A_null' in m else A[:, s:]).T
    else:
        Zf = m['F'][ev] @ A.T + m['Ab']
    return float(np.mean(np.abs(_ang(Zf, T) - th_row)))


def as_row(m):
    A = np.asarray(m['A'], np.float64)
    if '_null_from' in m:
        s = m['_null_from']; return m['F'][:, :s] @ A[:, :s].T + m['Ab']
    Vn = G.default_subspace(m)[0]; F = m['F']; return (F - (F @ Vn) @ Vn.T) @ A.T + m['Ab']


# ---------------------------------------------------------------- ViM family (Wang et al. 2022, official benchmark.py)
def vim_dim(d):
    return 1000 if d >= 2048 else (512 if d >= 768 else d // 2)


def vim_fit(m):
    """u = -pinv(W) b; principal space = top vim_dim(d) eigenvectors of the uncentred covariance of F_fit - u."""
    u = -head_offset(m['W'], m['b']); X = m['F'][np.asarray(m['fit'], bool)] - u
    ev, U = np.linalg.eigh(X.T @ X / len(X)); U = U[:, np.argsort(-ev)]
    return u, U


def vim_family(m):
    """vim: mean virtual-logit probability sigmoid(alpha ||res|| - logsumexp(L)); vim_res: mean ||res|| / ||f - u||."""
    fit = np.asarray(m['fit'], bool); ev = ~fit
    if '_R' in m:                                   # fast path: residual coordinates given, principal part cached
        Rc = m['_R']; R = np.linalg.norm(Rc, axis=1); xn = np.sqrt(m['_pn2'] + R ** 2); L = m['_Lp'] + Rc @ m['_WR'].T
    elif '_vim_frozen' in m:                        # features already in ViM's basis about u (u = 0 there)
        D = m['_vim_frozen']; F = m['F']; X = F + head_offset(m['W'], m['b']); Rc = X[:, D:]
        R = np.linalg.norm(Rc, axis=1); xn = np.linalg.norm(X, axis=1); L = logits(m)
    else:
        F = m['F']; u, U = vim_fit(m); D = vim_dim(F.shape[1]); X = F - u; Rc = X @ U[:, D:]
        R = np.linalg.norm(Rc, axis=1); xn = np.linalg.norm(X, axis=1); L = logits(m)
    alpha = L[fit].max(1).mean() / R[fit].mean()
    out = dict(vim=float(np.mean(expit(alpha * R[ev] - logsumexp(L[ev], axis=1)))),
               vim_res=float(np.mean(R[ev] / xn[ev])),
               res_norm=float(np.mean(R[ev])))                        # pure energy reference in the ViM residual space
    # content reference in the ViM residual space: cosine with the predicted class's residual mean (fit images)
    y = np.asarray(m['y']).astype(int); M = G.class_means(Rc[fit], y[fit], np.asarray(m['W']).shape[0]); Mm = M[np.argmax(L[ev], 1)]
    out['core_res'] = float(np.mean(np.sum(Rc[ev] * Mm, 1) / np.maximum(R[ev] * np.linalg.norm(Mm, axis=1), 1e-300)))
    return out


def all_scores(m):
    o = null_family(m); o.update(vim_family(m)); return o


NULL_SCORES = ['nsp', 'nusa', 'null_norm', 'rho', 'core_null', 'as_sing']
VIM_SCORES = ['vim', 'vim_res', 'res_norm', 'core_res']
