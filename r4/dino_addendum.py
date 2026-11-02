"""PREREG_R4B idea 6 (folded into DN): ridge Delta_align and NCM_null for DINO ViT-B/16 and ViT-S/16, each against a 95%
prediction interval from OLS of the score on log rho over E12 (ridge translators for Delta_align), and ranks among the 15 models.
Outputs r4/results/I6_dino_addendum.json. Usage: python r4/dino_addendum.py --feats-root $IC014_FEATS"""
import argparse, os, sys, json, numpy as np, torch
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES4, load_models, sets, dump4
from retrained_a4 import text_emb, split, retrain, null_basis
from retrain_translators import ang
from class_perm import derangement, class_means, ND
torch.set_num_threads(os.cpu_count())


def pi(x, y, x0):
    n = len(x); X = np.c_[np.ones(n), x]; b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; s2 = r @ r / (n - 2)
    x0v = np.array([1, x0]); se = np.sqrt(s2 * (1 + x0v @ np.linalg.inv(X.T @ X) @ x0v)); q = tdist.ppf(0.975, n - 2); p = float(x0v @ b)
    return dict(pred=p, pi=[p - q * se, p + q * se], slope=float(b[1]), intercept=float(b[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    TC = text_emb(a.sing_root); Y = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double(); tr, te = split()
    ms, col, sing = load_models(); rho = col('rho'); E12 = sets(ms, rho)['E12']
    I1 = json.load(open(os.path.join(RES4, 'I1_class_perm.json')))['per_model']['ridge']; I5 = json.load(open(os.path.join(RES4, 'I5_ncm_null.json')))
    seeds = [94_000 + r for r in range(ND)]; D = {}
    for name in ('dino_vitb16', 'dino_vits16'):
        d = np.load(f'{fr}/r3_dino_feats/{name}_10k.npz', allow_pickle=True); F = torch.tensor(d['F']).double(); y = torch.tensor(d['labels'].astype(np.int64)); Vn = null_basis(torch.tensor(d['W']))
        Wt, bt, _, _ = retrain(F, Y, tr); C = F @ Vn; mu = class_means(C, y); Rr = C - mu[y]; Fe = F[te]; tc = TC[y[te]]; Ce = C[te]
        Zrow = (Fe - Ce @ Vn.T) @ Wt + bt; AVt = Vn.T @ Wt; arow = ang(Zrow, tc); ASr = float((ang(Zrow + Ce @ AVt, tc) - arow).abs().mean()); v = []
        for s in seeds:
            p = torch.tensor(derangement(1000, s)); v.append(float((ang(Zrow + (mu[p][y] + Rr)[te] @ AVt, tc) - arow).abs().mean()))
        r0 = float(C.pow(2).sum(1).mean() / F.pow(2).sum(1).mean())
        D[name] = dict(rho=r0, AS_ret=ASr, AS_cperm_ret=float(np.mean(v)), D_align_ret=ASr - float(np.mean(v)), NCM_null=I5['dino'][name]['NCM_null'])
    lx = np.log([rho[ms.index(n)] for n in E12]); res = dict(per_model=D, pi={}, ranks={})
    for k, src in (('D_align_ret', [I1[n]['D_align'] for n in E12]), ('NCM_null', [I5['per_model'][n]['NCM_null'] for n in E12])):
        for name in D:
            q = pi(lx, np.array(src), np.log(D[name]['rho'])); q['obs'] = D[name][k]; q['inside'] = q['pi'][0] <= q['obs'] <= q['pi'][1]; res['pi'][f'{name}:{k}'] = q
        allv = {n: (I1[n]['D_align'] if k == 'D_align_ret' else I5['per_model'][n]['NCM_null']) for n in ms if (k == 'D_align_ret' or n in I5['per_model'])}; allv.update({n: D[n][k] for n in D})
        order = sorted(allv, key=lambda n: allv[n]); res['ranks'][k] = {n: order.index(n) + 1 for n in D}; res['ranks'][k]['n_ranked'] = len(order); res['ranks'][k]['median_released'] = float(np.median([allv[n] for n in ms if n in allv]))
    dump4('I6_dino_addendum.json', res); print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
