"""PREREG_R4B idea 1: class-mean permutation surrogate. z = mu_y + r; z' = mu_pi(y) + r with pi a derangement of the 1000 classes
(5 draws, seed 94000+r). Energy and the uncentred null second moment are preserved exactly (10 images per class). Released translators
(all 10k) and the round-3 ridge translators (test half). Also used by dino.py-addendum code (helpers). Outputs r4/results/I1_class_perm.json.
Usage: python r4/class_perm.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES3, RES4, load_models, tau_ci, ktau, sets, partial_ci, dump4
from retrained_a4 import text_emb, split, retrain, null_basis
from retrain_translators import ang, find
torch.set_num_threads(os.cpu_count()); ND = 5


def derangement(n, seed):
    g = np.random.default_rng(seed)
    while True:
        p = g.permutation(n)
        if not (p == np.arange(n)).any(): return p


def class_means(Z, y, ncls=1000):
    M = torch.zeros(ncls, Z.shape[1], dtype=Z.dtype).index_add_(0, y, Z); n = torch.bincount(y, minlength=ncls).clamp_min(1).to(Z.dtype)
    return M / n[:, None]


def cperm_scores(F, Vn, Wm, bm, y, tc, seeds, check=True):
    """AS_real and AS_cperm for translator f -> f Wm + bm (Wm: m x 768). Returns dict."""
    C = F @ Vn; mu = class_means(C, y); Rr = C - mu[y]; Frow = F - C @ Vn.T; Zrow = Frow @ Wm + bm; AVt = Vn.T @ Wm
    arow = ang(Zrow, tc); Z = Zrow + C @ AVt; AS = float((ang(Z, tc) - arow).abs().mean())
    Mreal = C.T @ C; En = float(C.pow(2).sum()); kap = float((C @ AVt).pow(2).sum(1).mean() / Z.pow(2).sum(1).mean()); v = []; kv = []; chk = []
    for s in seeds:
        p = torch.tensor(derangement(1000, s)); Cp = mu[p][y] + Rr; Zp = Zrow + Cp @ AVt
        if check:
            chk.append((abs(float(Cp.pow(2).sum()) - En) / En, float(torch.linalg.norm(Cp.T @ Cp - Mreal) / torch.linalg.norm(Mreal))))
        v.append(float((ang(Zp, tc) - arow).abs().mean())); kv.append(float((Cp @ AVt).pow(2).sum(1).mean() / Zp.pow(2).sum(1).mean()))
    return dict(AS=AS, AS_cperm=float(np.mean(v)), AS_cperm_sd=float(np.std(v)), kappa=kap, kappa_cperm=float(np.mean(kv)),
                kappa_rel_change=float(np.mean(kv)) / kap - 1 if kap > 0 else 0.0,
                max_energy_rel_err=max(c[0] for c in chk) if chk else None, max_M_rel_err=max(c[1] for c in chk) if chk else None)


def outcome(res, fam):
    t = res['tau'][fam]; ok = lambda k, cond: all(cond(t[s][k]['tau']) for s in ('E12', 'S11'))
    med = {s: res['median_share'][fam][s] for s in ('E12', 'S11')}
    if ok('D_align', lambda v: v >= 0.6) and ok('AS_cperm', lambda v: v < 0.3): return 'CONTENT-CARRIES'
    if ok('AS_cperm', lambda v: v >= 0.6) and all(m < 0.2 for m in med.values()): return 'ENERGY-CARRIES'
    return 'UNAPPORTIONED'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    TC = text_emb(a.sing_root); Y = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double(); tr, te = split()
    ms, col, sing = load_models(); lk = -sing; rho = col('rho'); S = sets(ms, rho); seeds = [94_000 + r for r in range(ND)]
    Ssur = json.load(open(os.path.join(RES4, 'S_surrogates_permodel.json')))
    P = {'released': {}, 'ridge': {}}
    for name in ms:
        t0 = time.time(); d = np.load(find(fr, name)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long(); Vn = null_basis(torch.tensor(d['W']))
        A = torch.tensor(d['A']).double(); Ab = torch.tensor(d['Ab']).double()
        P['released'][name] = cperm_scores(F, Vn, A.T, Ab, y, TC[y], seeds)
        Wt, bt, _, _ = retrain(F, Y, tr)
        # ridge: class means from all 10k (the surrogate is a property of the features); scores on the test half
        C = F @ Vn; mu = class_means(C, y); Rr = C - mu[y]; Fe = F[te]; ye = y[te]; tc = TC[ye]; Ce = C[te]; Frow = Fe - Ce @ Vn.T
        Zrow = Frow @ Wt + bt; AVt = Vn.T @ Wt; arow = ang(Zrow, tc); ASr = float((ang(Zrow + Ce @ AVt, tc) - arow).abs().mean()); v = []
        for s in seeds:
            p = torch.tensor(derangement(1000, s)); Cp = (mu[p][y] + Rr)[te]; v.append(float((ang(Zrow + Cp @ AVt, tc) - arow).abs().mean()))
        P['ridge'][name] = dict(AS=ASr, AS_cperm=float(np.mean(v)), AS_cperm_sd=float(np.std(v)))
        for fam in P:
            q = P[fam][name]; q['D_align'] = q['AS'] - q['AS_cperm']; q['share'] = q['D_align'] / q['AS'] if q['AS'] > 0 else float('nan')
        P['released'][name]['AS_perm_r4S'] = Ssur[name]['perm']; P['released'][name]['perm_le_cperm'] = bool(Ssur[name]['perm'] <= P['released'][name]['AS_cperm'])
        print(name, round(time.time() - t0, 1), {f: {k: (round(v, 5) if isinstance(v, float) else v) for k, v in P[f][name].items()} for f in P}, flush=True)
        del F, C
    errs = [(P['released'][n]['max_energy_rel_err'], P['released'][n]['max_M_rel_err']) for n in ms]
    assert all(e < 1e-9 and m < 1e-9 for e, m in errs), errs
    L = lambda sub: np.array([lk[ms.index(n)] for n in sub]); Rh = lambda sub: np.array([rho[ms.index(n)] for n in sub])
    res = dict(per_model=P, sets=S, tau={}, partial_given_rho={}, median_share={}, max_energy_rel_err=max(e for e, _ in errs), max_M_rel_err=max(m for _, m in errs))
    for fam in P:
        V = lambda k, sub: np.array([P[fam][n][k] for n in sub]); res['tau'][fam] = {}; res['partial_given_rho'][fam] = {}; res['median_share'][fam] = {}
        for s, sub in S.items():
            res['tau'][fam][s] = {k: tau_ci(V(k, sub), L(sub)) for k in ['AS', 'AS_cperm', 'D_align', 'share']}
            res['partial_given_rho'][fam][s] = {k: partial_ci(V(k, sub), L(sub), Rh(sub)) for k in ['AS_cperm', 'D_align']}
            res['median_share'][fam][s] = float(np.nanmedian(V('share', sub)))
        res.setdefault('outcome', {})[fam] = outcome(res, fam)
    res['KILL_R4B_1'] = any(v == 'CONTENT-CARRIES' for v in res['outcome'].values())
    res['perm_le_cperm_count'] = sum(P['released'][n]['perm_le_cperm'] for n in ms)
    dump4('I1_class_perm.json', res)
    for fam in P:
        print(fam, {s: {k: round(v['tau'], 3) for k, v in res['tau'][fam][s].items()} for s in S}, res['median_share'][fam])
    print(res['outcome'], 'KILL', res['KILL_R4B_1'], 'perm<=cperm', res['perm_le_cperm_count'], 'errs', res['max_energy_rel_err'], res['max_M_rel_err'])


if __name__ == '__main__':
    main()
