"""POST HOC (not in PREREG_R3): the pre-registered lambda grid of G (10^{-3..3}) selected its upper edge for every
model, and the selection cosine was still rising. This script extends the grid to 10^{3..6} (same split, same fit,
same scores as r3/retrain_translators.py) and reports, per lambda, fidelity, tau(AS_SING-retrained) and
tau(A1-retrained) against SING's order, plus the lambda chosen on the extended grid.
Usage: python r3/retrain_lambda_posthoc.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import R, RES, load_models, tau_ci, dump, RHO0, S11_DROP
from retrain_translators import find, fit, cosm, ang
torch.set_num_threads(os.cpu_count()); LAMS = [10.0 ** k for k in range(3, 7)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float64)), dim=-1).mean(1), dim=-1)
    Yall = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double()
    perm = np.random.default_rng(0).permutation(10000); tr = torch.tensor(perm[:5000]); te = torch.tensor(perm[5000:])
    ms, col, sing = load_models(); lk = -sing
    cache = os.path.join(RES, 'G_lambda_posthoc_permodel.json'); P = json.load(open(cache)) if os.path.exists(cache) else {}
    for name in ms:
        if name in P: continue
        d = np.load(find(fr, name)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long()
        W = torch.tensor(d['W']).double(); _, sv, vt = torch.linalg.svd(W, full_matrices=True); Vn = vt.T[:, int((sv > 1e-5).sum()):]
        Xtr, Ytr = F[tr], Yall[tr]; Fe = F[te]; tc = TC[y[te]]; Pi = (Fe @ Vn) @ Vn.T
        rho = float((F @ Vn).pow(2).sum(1).mean() / F.pow(2).sum(1).mean()); beta = (RHO0 / rho) ** 0.5; P[name] = {}
        for lr in LAMS:
            Wl, bl = fit(Xtr[:4000], Ytr[:4000], lr); sel = cosm(Xtr[4000:] @ Wl + bl, Ytr[4000:])
            Wt, bt = fit(Xtr, Ytr, lr); Z = Fe @ Wt + bt; a0 = ang(Z, tc); Rn = Pi @ Wt
            P[name][str(lr)] = dict(sel_cos=sel, fidelity=cosm(Z, Yall[te]), AS=float((a0 - ang(Z - Rn, tc)).abs().mean()),
                                    A1=float((a0 - ang(Z - beta * Rn, tc)).abs().mean()))
        json.dump(P, open(cache, 'w'), indent=1); print(name, {k: round(v['sel_cos'], 4) for k, v in P[name].items()}, flush=True)
        del F, Xtr, Fe, Pi
    e12 = [n for n, r in zip(ms, col('rho')) if r >= RHO0]; s11 = [n for n in ms if n not in S11_DROP]
    L = lambda sub: np.array([lk[ms.index(n)] for n in sub]); res = dict(per_model=P, per_lambda={}, label='POST HOC')
    for lr in map(str, LAMS):
        res['per_lambda'][lr] = {s: dict(AS=tau_ci(np.array([P[n][lr]['AS'] for n in sub]), L(sub))['tau'],
                                         A1=tau_ci(np.array([P[n][lr]['A1'] for n in sub]), L(sub))['tau'])
                                 for s, sub in [('all13', ms), ('E12', e12), ('S11', s11)]}
    best = {n: max(P[n], key=lambda k: P[n][k]['sel_cos']) for n in ms}; res['chosen_extended'] = best
    res['chosen_tau'] = {s: dict(AS=tau_ci(np.array([P[n][best[n]]['AS'] for n in sub]), L(sub)),
                                 A1=tau_ci(np.array([P[n][best[n]]['A1'] for n in sub]), L(sub)))
                         for s, sub in [('all13', ms), ('E12', e12), ('S11', s11)]}
    dump('G_lambda_posthoc.json', res); print(json.dumps(res['per_lambda'], indent=0), best)
    print({s: {k: (round(v['tau'], 3), v['pct'], v['bca']) for k, v in d.items()} for s, d in res['chosen_tau'].items()})


if __name__ == '__main__':
    main()
