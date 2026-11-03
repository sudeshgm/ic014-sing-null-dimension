"""PREREG_R6: published null-space / residual scores, implemented on penultimate features F (n x d), head W (C x d), bias b.
A ViM (Wang et al. 2022; official code github.com/haoqiwang/vim, benchmark.py), B ActSub S-> (Zongur et al. 2025; github.com/visinf/actsub,
methods.py), C CORE S_mem (Yang et al. 2026, arXiv 2603.18290, Eq. 1 and 4), D conditional V-information (Hewitt et al. 2021) of the
null coordinates given the head's logits; reference NuSA-style null share (Cook et al. 2020). Every fitted quantity is fit on one half
of the images and evaluated on the other half; `crossfit` swaps the halves and returns one per-image value for every image.
numpy / torch (CPU), float64."""
import numpy as np, torch
from scipy.special import logsumexp, expit

NCLS = 1000


def null_split(W, tol=1e-5):
    _, sv, vt = np.linalg.svd(W.astype(np.float64), full_matrices=True); r = int((sv > tol).sum())
    return vt[:r].T, vt[r:].T                    # row-space basis Vr (d x r), null basis Vn (d x d-r)


def halves(y, seed):
    """class-stratified random halves: for each class, half of its images (rounded down) in A"""
    rng = np.random.default_rng(seed); A = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = np.where(y == c)[0]; rng.shuffle(idx); A[idx[: len(idx) // 2]] = True
    return A


# ---------------- A. ViM
def vim_dim(d):
    """principal-space dimension as in the official benchmark.py"""
    return 1000 if d >= 2048 else (512 if d >= 768 else d // 2)


def vim_fit(F, L, W, b):
    u = -np.linalg.pinv(W) @ b
    X = F - u; cov = X.T @ X / len(X)            # EmpiricalCovariance(assume_centered=True)
    ev, evec = np.linalg.eigh(cov); order = np.argsort(-ev); NS = evec[:, order[vim_dim(F.shape[1]):]]
    R = np.linalg.norm(X @ NS, axis=1); alpha = L.max(1).mean() / R.mean()
    return dict(u=u, NS=NS, alpha=alpha)


def vim_eval(m, F, L):
    v = m['alpha'] * np.linalg.norm((F - m['u']) @ m['NS'], axis=1); E = logsumexp(L, axis=1)
    return dict(p_virtual=expit(v - E), s=E - v)     # p_virtual = softmax([L, v])[-1]; s = official score (energy - vlogit)


# ---------------- B. ActSub insignificant score
def actsub_cidx(Vt, Ffit):
    """official find_c_idx: argmin_idx | mean||dec|| - mean||insig|| | over idx in range(d)"""
    P2 = (Ffit @ Vt.T) ** 2; cs = np.cumsum(P2, axis=1); tot = cs[:, -1:]
    dec = np.concatenate([np.zeros((len(Ffit), 1)), cs[:, :-1]], axis=1)        # idx = 0..d-1 -> first idx directions
    ins = tot - dec
    r = np.abs(np.sqrt(np.maximum(dec, 0)).mean(0) - np.sqrt(np.maximum(ins, 0)).mean(0)); return int(np.argmin(r))


def actsub_fit(Vt, Ffit, ref_seed, frac=10):
    c = actsub_cidx(Vt, Ffit); Vi = Vt[c:].T
    idx = np.random.default_rng(ref_seed).permutation(len(Ffit))[: len(Ffit) // frac]
    Ref = Ffit[idx] @ Vi; Ref = Ref / np.maximum(np.linalg.norm(Ref, axis=1, keepdims=True), 1e-300)
    return dict(c_idx=c, Vi=Vi, Ref=Ref)


def actsub_eval(m, F, N=10):
    Z = F @ m['Vi']; Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300); S = Z @ m['Ref'].T
    top = -np.sort(-S, axis=1)[:, :N]; cs = top.mean(1)
    return dict(S_insig=-np.log(np.maximum(1 - cs, 1e-300)))


# ---------------- C. CORE membership
def core_perp(F, W, yhat):
    w = W[yhat]; return F - ((F * w).sum(1) / (w * w).sum(1))[:, None] * w


def core_fit(F, W, L, y, yhat_fit=None, ker_only_Vn=None):
    yh = L.argmax(1) if yhat_fit is None else yhat_fit
    Z = F @ ker_only_Vn if ker_only_Vn is not None else core_perp(F, W, yh)
    M = np.zeros((NCLS, Z.shape[1])); np.add.at(M, y, Z); M /= np.maximum(np.bincount(y, minlength=NCLS), 1)[:, None]
    return dict(mu=M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-300), Vn=ker_only_Vn)


def core_eval(m, F, W, L):
    yh = L.argmax(1); Z = F @ m['Vn'] if m['Vn'] is not None else core_perp(F, W, yh)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300); return dict(S_mem=(Z * m['mu'][yh]).sum(1))


# ---------------- D. conditional V-information (affine-softmax family; B enters through the head's own logits)
def _fit_probe(B, Phi, y, lam, ncls, iters=200):
    """minimise mean NLL of softmax(g(B) + Phi@U.T + c) + lam*||U||^2, with g(B) = t*B (scalar temperature on the head's own logits)
    when B has one column per class, else an unpenalised linear map B@G (U = 0 and not fitted if Phi is None).
    torch L-BFGS (strong Wolfe), float32 fitting; NLL is then evaluated in float64."""
    dt = torch.float32; Bt = torch.tensor(B, dtype=dt); yt = torch.tensor(y).long(); own = B.shape[1] == ncls
    g = torch.ones(1, dtype=dt, requires_grad=True) if own else torch.zeros((B.shape[1], ncls), dtype=dt, requires_grad=True)
    c = torch.zeros(ncls, dtype=dt, requires_grad=True); ps = [g, c]
    if Phi is not None:
        Pt = torch.tensor(Phi, dtype=dt); U = torch.zeros((ncls, Phi.shape[1]), dtype=dt, requires_grad=True); ps.append(U)
    opt = torch.optim.LBFGS(ps, lr=1, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(); z = (g * Bt if own else Bt @ g) + c
        if Phi is not None: z = z + Pt @ U.T
        loss = torch.nn.functional.cross_entropy(z, yt) + (lam * (U * U).sum() if Phi is not None else 0.0); loss.backward(); return loss
    opt.step(closure)
    d = lambda v: v.detach().numpy().astype(np.float64)
    return dict(own=own, g=d(g), c=d(c), U=None if Phi is None else d(U))


def _nll(p, B, Phi, y):
    z = (p['g'] * B if p['own'] else B @ p['g']) + p['c'] + (Phi @ p['U'].T if p['U'] is not None else 0.0)
    return logsumexp(z, axis=1) - z[np.arange(len(y)), y]


def vinfo_phi_fit(Cfit, k=64):
    mu = Cfit.mean(0); _, _, vt = np.linalg.svd(Cfit - mu, full_matrices=False); return dict(mu=mu, V=vt[: min(k, Cfit.shape[1])].T)


def vinfo_target(L, y, which):
    if which == 'label': return L, y, NCLS
    lse = logsumexp(L, axis=1); mx = L.max(1); srt = np.sort(L, axis=1); B = np.c_[mx, lse, np.exp(mx - lse), srt[:, -1] - srt[:, -2]]
    return B, (L.argmax(1) == y).astype(int), 2


_BASE = {}


def vinfo_fit_eval(Cf, Lf, yf, Ce, Le, ye, lam_rel, which='label', phi_scale=None, base_key=None):
    """returns per-image gain on the evaluation half (nats) and the absolute lambda used. lam = lam_rel / phi_scale, where phi_scale is
    mean ||phi||^2 on the REAL fit half (frozen across interventions); if phi_scale is None it is computed here."""
    pm = vinfo_phi_fit(Cf); Pf = (Cf - pm['mu']) @ pm['V']; Pe = (Ce - pm['mu']) @ pm['V']
    Bf, Yf, nc = vinfo_target(Lf, yf, which); Be, Ye, _ = vinfo_target(Le, ye, which)
    if which != 'label':
        mB, sB = Bf.mean(0), Bf.std(0) + 1e-12; Bf = (Bf - mB) / sB; Be = (Be - mB) / sB
    sc = float((Pf ** 2).sum(1).mean()) if phi_scale is None else phi_scale; lam = lam_rel / sc
    if base_key is not None and base_key in _BASE: p0 = _BASE[base_key]      # baseline [B;0] does not depend on the null part: cached
    else:
        p0 = _fit_probe(Bf, None, Yf, 0.0, nc)
        if base_key is not None: _BASE[base_key] = p0
    p1 = _fit_probe(Bf, Pf, Yf, lam, nc)
    return _nll(p0, Be, None, Ye) - _nll(p1, Be, Pe, Ye), lam, sc


def vinfo_choose_lam(Cf, Lf, yf, which, grid=(1e-4, 1e-3, 1e-2, 1e-1, 1.0), seed=100700):
    """inner 2-fold CV on the fit half (class-stratified), on real data only; returns lam_rel with the largest mean held-out gain"""
    A = halves(yf, seed); best = None
    if A.sum() == 0 or (~A).sum() == 0: raise ValueError('inner CV needs >= 2 images per class in the fit half')
    for lr in grid:
        g = []
        for m in (A, ~A):
            gi, _, _ = vinfo_fit_eval(Cf[m], Lf[m], yf[m], Cf[~m], Lf[~m], yf[~m], lr, which); g.append(gi.mean())
        v = float(np.mean(g))
        if np.isfinite(v) and (best is None or v > best[1]): best = (lr, v)
    return best[0]


# ---------------- reference: NuSA-style null share
def nusa(F, C):
    return np.linalg.norm(C, axis=1) / np.linalg.norm(F, axis=1)
