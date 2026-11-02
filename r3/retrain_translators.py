"""PREREG_R3 G: retrained translators. For each of the 13 models, ridge f -> CLIP image embedding (clip_img.npz, same 10k
images) on a fixed train half (permutation seed 0, first 5000), intercept by centring, lambda from 10^{-3..3} *
tr(X^T X)/(n m) chosen by mean cosine on a 4000/1000 split of the train half, refit on all 5000. On the other 5000:
fidelity (mean cosine), AS_SING with the retrained translator (class text), A1 at rho0 = 0.005 (beta = sqrt(rho0/rho)).
Usage: python r3/retrain_translators.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import R, RES, load_models, tau_ci, ktau, dump, RHO0, S11_DROP
torch.set_num_threads(os.cpu_count()); LAMS = [10.0 ** k for k in range(-3, 4)]


def ang(Z, T): return torch.rad2deg(torch.acos(torch.clamp((torch.nn.functional.normalize(Z, dim=-1) * T).sum(-1), -1, 1)))


def find(fr, name):
    for r in [f'{fr}/e2/ret/e2_feats', f'{fr}/e3/ret/e3_feats']:
        f = os.path.join(r, f'{name}_full.npz')
        if os.path.exists(f): return f


def fit(X, Y, lam_rel):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; G = Xc.T @ Xc; n, m = X.shape
    lam = lam_rel * float(torch.trace(G)) / (n * m); W = torch.linalg.solve(G + lam * torch.eye(m, dtype=G.dtype), Xc.T @ Yc)
    return W, my - mx @ W


def cosm(P, Y): return float((torch.nn.functional.normalize(P, dim=-1) * torch.nn.functional.normalize(Y, dim=-1)).sum(1).mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float64)), dim=-1).mean(1), dim=-1)
    Yall = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double()
    perm = np.random.default_rng(0).permutation(10000); tr = torch.tensor(perm[:5000]); te = torch.tensor(perm[5000:])
    ms, col, sing = load_models(); lk = -sing; AS_rel = col('AS_SING')
    cache = os.path.join(RES, 'G_retrain_permodel.json'); P = json.load(open(cache)) if os.path.exists(cache) else {}
    for name in ms:
        if name in P: continue
        t0 = time.time(); d = np.load(find(fr, name)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long()
        A0 = torch.tensor(d['A']).double(); b0 = torch.tensor(d['Ab']).double()
        W = torch.tensor(d['W']).double(); _, sv, vt = torch.linalg.svd(W, full_matrices=True); Vn = vt.T[:, int((sv > 1e-5).sum()):]
        Xtr, Ytr = F[tr], Yall[tr]; sel = {}
        for lr in LAMS:
            Wl, bl = fit(Xtr[:4000], Ytr[:4000], lr); sel[lr] = cosm(Xtr[4000:] @ Wl + bl, Ytr[4000:])
        lbest = max(sel, key=sel.get); Wt, bt = fit(Xtr, Ytr, lbest)
        Fe = F[te]; tc = TC[y[te]]; Ce = Fe @ Vn; Pi = Ce @ Vn.T
        rho = float((F @ Vn).pow(2).sum(1).mean() / F.pow(2).sum(1).mean()); beta = (RHO0 / rho) ** 0.5
        def scores(Wm, bm):
            Z = Fe @ Wm + bm; a0 = ang(Z, tc); Rn = Pi @ Wm
            return cosm(Z, Yall[te]), float((a0 - ang(Z - Rn, tc)).abs().mean()), float((a0 - ang(Z - beta * Rn, tc)).abs().mean())
        fid, AS, A1 = scores(Wt, bt); fid0, AS0, A10 = scores(A0.T, b0)
        P[name] = dict(lambda_rel=lbest, sel_cos=sel, fidelity=fid, AS_SING_retrained=AS, A1_retrained=A1, rho=rho, beta=beta,
                       released_fidelity_test=fid0, AS_SING_released_test=AS0, A1_released_test=A10, secs=time.time() - t0)
        json.dump(P, open(cache, 'w'), indent=1); print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P[name].items() if k != 'sel_cos'}, flush=True)
        del F, Xtr, Fe, Pi, Ce
    e12 = [n for n, r in zip(ms, col('rho')) if r >= RHO0]; s11 = [n for n in ms if n not in S11_DROP]
    V = lambda k, sub: np.array([P[n][k] for n in sub]); L = lambda sub: np.array([lk[ms.index(n)] for n in sub])
    res = dict(per_model=P, tau={})
    for sname, sub in [('all13', ms), ('E12', e12), ('S11', s11)]:
        res['tau'][sname] = {k: tau_ci(V(k, sub), L(sub)) for k in ['AS_SING_retrained', 'AS_SING_released_test', 'fidelity']}
        if sname != 'all13': res['tau'][sname]['A1_retrained'] = tau_ci(V('A1_retrained', sub), L(sub))
    res['tau_retrained_vs_released_AS'] = ktau(V('AS_SING_retrained', ms), AS_rel)
    t = [res['tau'][s]['AS_SING_retrained']['tau'] for s in ('all13', 'S11')]
    res['reading'] = 'order depends on the released translators' if all(x < 0.3 for x in t) else ('order survives retraining' if all(x >= 0.6 for x in t) else 'intermediate')
    dump('G_retrain.json', res)
    print({s: {k: round(v['tau'], 3) for k, v in res['tau'][s].items()} for s in res['tau']}, res['reading'], res['tau_retrained_vs_released_AS'])


if __name__ == '__main__':
    main()
