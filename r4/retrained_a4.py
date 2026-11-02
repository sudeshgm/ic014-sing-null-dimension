"""PREREG_R4 J (part): kappa, gamma and A4 under the retrained ridge translators (the r3 G pipeline refit unchanged).
kappa_ret = E||Pi f W||^2 / E||Z||^2 on the test half; gamma_ret = kappa_ret/rho; A4_ret = AS after removing beta' Pi f W,
beta'^2 kappa_ret = kappa0_ret = min over E12. Outputs r4/results/J_retrained_a4{_permodel}.json.
Usage: python r4/retrained_a4.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES3, RES4, load_models, tau_ci, ktau, sets, dump4, RHO0
from retrain_translators import fit, cosm, ang, find, LAMS
torch.set_num_threads(os.cpu_count())


def text_emb(sing_root):
    emb = np.load(os.path.join(sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    return torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float64)), dim=-1).mean(1), dim=-1)


def split():
    perm = np.random.default_rng(0).permutation(10000); return torch.tensor(perm[:5000]), torch.tensor(perm[5000:])


def retrain(F, Y, tr):
    Xtr, Ytr = F[tr], Y[tr]; sel = {}
    for lr in LAMS:
        Wl, bl = fit(Xtr[:4000], Ytr[:4000], lr); sel[lr] = cosm(Xtr[4000:] @ Wl + bl, Ytr[4000:])
    lb = max(sel, key=sel.get); Wt, bt = fit(Xtr, Ytr, lb); return Wt, bt, lb, sel


def null_basis(W):
    _, sv, vt = torch.linalg.svd(W.double(), full_matrices=True); return vt.T[:, int((sv > 1e-5).sum()):]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    TC = text_emb(a.sing_root); Y = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double(); tr, te = split()
    ms, col, sing = load_models(); lk = -sing; rho_c = col('rho'); G = json.load(open(os.path.join(RES3, 'G_retrain_permodel.json')))
    cache = os.path.join(RES4, 'J_retrained_a4_permodel.json'); os.makedirs(RES4, exist_ok=True); P = json.load(open(cache)) if os.path.exists(cache) else {}
    store = {}
    for name in ms:
        t0 = time.time(); d = np.load(find(fr, name)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long(); Vn = null_basis(torch.tensor(d['W']))
        Wt, bt, lb, _ = retrain(F, Y, tr); Fe = F[te]; tc = TC[y[te]]; Z = Fe @ Wt + bt; Rn = (Fe @ Vn) @ Vn.T @ Wt; a0 = ang(Z, tc)
        kappa = float(Rn.pow(2).sum(1).mean() / Z.pow(2).sum(1).mean()); rho = G[name]['rho']
        AS = float((a0 - ang(Z - Rn, tc)).abs().mean())
        store[name] = (a0, Z, Rn, tc)
        P[name] = dict(kappa_ret=kappa, gamma_ret=kappa / rho, rho=rho, lambda_rel=lb, AS_ret_check=AS, AS_ret_G=G[name]['AS_SING_retrained'], secs=time.time() - t0)
        print(name, {k: round(v, 5) if isinstance(v, float) else v for k, v in P[name].items()}, flush=True); del F, Fe
    S = sets(ms, rho_c); k0n = min(S['E12'], key=lambda n: P[n]['kappa_ret']); k0 = P[k0n]['kappa_ret']
    for n in ms:
        a0, Z, Rn, tc = store[n]; b = (k0 / P[n]['kappa_ret']) ** 0.5; P[n]['A4_beta'] = b; P[n]['A4_ret'] = float((a0 - ang(Z - b * Rn, tc)).abs().mean())
    json.dump(P, open(cache, 'w'), indent=1)
    V = lambda k, sub: np.array([P[n][k] for n in sub]); L = lambda sub: np.array([lk[ms.index(n)] for n in sub])
    res = dict(per_model=P, kappa0_ret=k0, kappa0_model=k0n, tau={}, max_rel_err_AS_vs_G=max(abs(P[n]['AS_ret_check'] - P[n]['AS_ret_G']) / max(P[n]['AS_ret_G'], 1e-12) for n in ms))
    for s, sub in S.items():
        res['tau'][s] = {k: tau_ci(V(k, sub), L(sub)) for k in ['kappa_ret', 'gamma_ret'] + (['A4_ret'] if s != 'all13' else [])}
        res['tau'][s]['kappa_ret_vs_rho'] = ktau(V('kappa_ret', sub), V('rho', sub))
    res['A4_ret_tau_lt_0p3_both'] = bool(all(res['tau'][s]['A4_ret']['tau'] < 0.3 for s in ('E12', 'S11')))
    dump4('J_retrained_a4.json', res)
    print({s: {k: (round(v['tau'], 3) if isinstance(v, dict) else round(v, 3)) for k, v in res['tau'][s].items()} for s in S}, res['A4_ret_tau_lt_0p3_both'], res['max_rel_err_AS_vs_G'])


if __name__ == '__main__':
    main()
