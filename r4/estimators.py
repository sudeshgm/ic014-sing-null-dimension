"""Shared estimators for the round-4 addendum (PREREG_R4B ideas 1, 2, 3, 5), all in null coordinates.
Zrow: row-space part in CLIP space (N x 768), C: null coordinates (N x k), AVt: null block of the translator (k x 768), y: labels,
tc: class-text targets (N x 768)."""
import os, sys, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from retrain_translators import ang
from equal_energy import energy_match
from class_perm import derangement, class_means
RHO0 = 0.005


def as_img(Zrow, D, tc, arow=None):
    """per-image |theta(Zrow + D) - theta(Zrow)|"""
    arow = ang(Zrow, tc) if arow is None else arow; return (ang(Zrow + D, tc) - arow).abs()


def AS(Zrow, C, AVt, tc): return float(as_img(Zrow, C @ AVt, tc).mean())


def A1(Zrow, C, AVt, tc, rho):
    Z = Zrow + C @ AVt; b = math.sqrt(RHO0 / rho); return float((ang(Z, tc) - ang(Z - b * (C @ AVt), tc)).abs().mean())


def AS_gauss(Zrow, C, AVt, tc, draws=2):
    mu = C.mean(0); Cc = C - mu; S = Cc.T @ Cc / len(C); ev, U = torch.linalg.eigh(S); Ls = U * ev.clamp_min(0).sqrt(); En = float(C.pow(2).sum(1).mean()); v = []
    for r in range(draws):
        g = torch.Generator().manual_seed(70_000 + r); Ct = mu + torch.randn(C.shape, generator=g, dtype=C.dtype) @ Ls.T
        Ct = Ct * (En / float(Ct.pow(2).sum(1).mean())) ** 0.5; v.append(AS(Zrow, Ct, AVt, tc))
    return float(np.mean(v))


def D_align(Zrow, C, AVt, y, tc, draws=2):
    mu = class_means(C, y); Rr = C - mu[y]; a = AS(Zrow, C, AVt, tc)
    v = [AS(Zrow, mu[torch.tensor(derangement(1000, 94_000 + r))][y] + Rr, AVt, tc) for r in range(draws)]
    return a - float(np.mean(v)), a, float(np.mean(v))


def ncm_loo(C, y, seed_ctrl=95_000, chunk=2000, ncls=1000):
    """LOO nearest-class-mean on l2-normalised null coordinates; returns top-1/top-5 (%), real and shuffled-label control."""
    nrm = C.norm(dim=1, keepdim=True); u = C / nrm.clamp_min(1e-300)
    def run(lab):
        Sc = torch.zeros(ncls, u.shape[1], dtype=u.dtype).index_add_(0, lab, u); n = torch.bincount(lab, minlength=ncls).to(u.dtype)
        M = Sc / n[:, None]; m2 = M.pow(2).sum(1); t1 = t5 = 0
        for s in range(0, len(u), chunk):
            ub = u[s:s + chunk]; lb = lab[s:s + chunk]; D = m2[None] - 2 * ub @ M.T
            own = (Sc[lb] - ub) / (n[lb] - 1)[:, None]; D[torch.arange(len(ub)), lb] = own.pow(2).sum(1) - 2 * (ub * own).sum(1)
            top = D.topk(5, dim=1, largest=False).indices; t1 += int((top[:, 0] == lb).sum()); t5 += int((top == lb[:, None]).any(1).sum())
        return 100 * t1 / len(u), 100 * t5 / len(u)
    lab_c = y[torch.tensor(np.random.default_rng(seed_ctrl).permutation(len(y)))]
    r1, r5 = run(y); c1, c5 = run(lab_c); return dict(top1=r1, top5=r5, ctrl_top1=c1, ctrl_top5=c5, NCM_null=r1 - c1)


def nc1(C, y, ncls=1000):
    u = C / C.norm(dim=1, keepdim=True).clamp_min(1e-300); mu = class_means(u, y); g = u.mean(0)
    W = (u - mu[y]); Sw = W.T @ W / len(u); B = mu - g; Sb = B.T @ B / ncls
    return float(torch.trace(Sw @ torch.linalg.pinv(Sb, hermitian=True)) / ncls)


def selectivity(Zrow, C, AVt, y, tc, k, ndraw=20, seed=96_000, nboot=1000, halfA=None):
    """U_k from between-class scatter on half A; AS of U_k removal vs energy-matched random null removal on half B."""
    A, Bi = halfA
    muA = class_means(C[A], y[A]); g = muA.mean(0); Sb = (muA - g).T @ (muA - g); ev, U = torch.linalg.eigh(Sb); Uk = U[:, torch.argsort(ev, descending=True)[:k]]
    Cb, Zb, tb, yb = C[Bi], Zrow[Bi] + C[Bi] @ AVt, tc[Bi], y[Bi]; a0 = ang(Zb, tb)
    rem = (Cb @ Uk) @ (AVt.T @ Uk).T; e = float((Cb @ Uk).pow(2).sum(1).mean()); a_lda = (a0 - ang(Zb - rem, tb)).abs()
    nd = C.shape[1]; a_r = torch.zeros(len(Bi), dtype=C.dtype)
    for j in range(ndraw):
        gg = torch.Generator().manual_seed(seed + j); Q = torch.linalg.qr(torch.randn(nd, nd, generator=gg, dtype=C.dtype))[0]
        P = Cb @ Q; coef, _, _ = energy_match(P, e); k1 = coef.shape[1]; a_r += (a0 - ang(Zb - coef @ (AVt.T @ Q[:, :k1]).T, tb)).abs() / ndraw
    R = float(a_lda.mean() / a_r.mean()); rng = np.random.default_rng(0); yb_np = yb.numpy(); groups = [np.where(yb_np == c)[0] for c in np.unique(yb_np)]
    al, ar = a_lda.numpy(), a_r.numpy(); bs = [R] if nboot == 0 else []
    for _ in range(nboot):
        ix = np.concatenate([gq[rng.integers(0, len(gq), len(gq))] for gq in groups]); bs.append(al[ix].mean() / ar[ix].mean())
    return dict(R=R, ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])], AS_lda=float(a_lda.mean()), AS_rand=float(a_r.mean()), e_lda=e, k=k)


def halves(y):
    yn = y.numpy(); A = []; B = []
    for c in np.unique(yn):
        ii = np.where(yn == c)[0]; A += list(ii[:5]); B += list(ii[5:])
    return torch.tensor(sorted(A)), torch.tensor(sorted(B))
