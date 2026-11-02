"""PREREG_R4B idea 5: translator-free null decodability. LOO nearest-class-mean on l2-normalised null coordinates (1000 classes),
shuffled-label control (seed 95000), NC1_null, Gaussian-surrogate sanity check (seed 70000). Also the two true-DINO models (idea 6).
Outputs r4/results/I5_ncm_null.json. Usage: python r4/ncm_null.py --feats-root $IC014_FEATS"""
import argparse, os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import RES4, load_models, tau_ci, ktau, sets, partial_ci, dump4
from retrained_a4 import null_basis
from retrain_translators import find
from leakfree import setup, draw
import estimators as est
torch.set_num_threads(os.cpu_count())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); a = ap.parse_args(); fr = a.feats_root
    ms, col, sing = load_models(); lk = -sing; rho = col('rho'); S = sets(ms, rho); P = {}
    for name in S['E12']:
        t0 = time.time(); E = setup(fr, name); C = E['F'] @ E['Vn']; y = E['y']
        o = est.ncm_loo(C, y); o['NC1_null'] = est.nc1(C, y); g = est.ncm_loo(draw(E, 0), y); o['gauss_top1'] = g['top1']; o['gauss_ctrl_top1'] = g['ctrl_top1']
        o['gauss_sanity_pass'] = bool(abs(g['top1'] - o['ctrl_top1']) <= 0.5); o['rho'] = E['rho']; o['secs'] = time.time() - t0; P[name] = o
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items()}, flush=True); del E, C
    D = {}
    for name in ('dino_vitb16', 'dino_vits16'):
        d = np.load(f'{fr}/r3_dino_feats/{name}_10k.npz', allow_pickle=True); F = torch.tensor(d['F']).double(); y = torch.tensor(d['labels'].astype(np.int64)); Vn = null_basis(torch.tensor(d['W']))
        C = F @ Vn; o = est.ncm_loo(C, y); o['NC1_null'] = est.nc1(C, y); o['rho'] = float(C.pow(2).sum(1).mean() / F.pow(2).sum(1).mean()); D[name] = o
        print(name, {k: round(v, 4) for k, v in o.items()}, flush=True)
    L = lambda sub: np.array([lk[ms.index(n)] for n in sub]); V = lambda k, sub: np.array([P[n][k] for n in sub]); Rh = lambda sub: np.array([rho[ms.index(n)] for n in sub])
    res = dict(per_model=P, dino=D, tau={}, partial_given_rho={}, tau_vs_rho={})
    for s in ('E12', 'S11'):
        sub = S[s]; res['tau'][s] = {k: tau_ci(V(k, sub), L(sub)) for k in ['NCM_null', 'top1', 'top5', 'NC1_null']}
        res['partial_given_rho'][s] = partial_ci(V('NCM_null', sub), L(sub), Rh(sub)); res['tau_vs_rho'][s] = ktau(V('NCM_null', sub), Rh(sub))
    t = {s: res['tau'][s]['NCM_null']['tau'] for s in ('E12', 'S11')}; pc = {s: res['partial_given_rho'][s]['pct'] for s in ('E12', 'S11')}
    excl = lambda ci: ci[0] > 0 or ci[1] < 0
    res['outcome'] = ('NCM-CONTENT' if all(t[s] >= 0.6 and excl(pc[s]) for s in t) else 'NCM-NULL' if all(abs(t[s]) < 0.3 for s in t) else 'intermediate')
    res['KILL_R4B_5'] = res['outcome'] == 'NCM-CONTENT'; res['gauss_sanity_all_pass'] = all(P[n]['gauss_sanity_pass'] for n in P)
    dump4('I5_ncm_null.json', res); print({s: {k: round(v['tau'], 3) for k, v in res['tau'][s].items()} for s in res['tau']}, res['partial_given_rho'], res['tau_vs_rho'], res['outcome'], res['gauss_sanity_all_pass'])


if __name__ == '__main__':
    main()
