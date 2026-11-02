"""PREREG_R4 S: surrogate family for the null coordinates, all 13 models, class text, SING released translators.
Each surrogate is rescaled to the model's own null energy (as r3/leakfree.rank). Content-free: gauss (round-3 AS_surr, re-used),
zm N(0,Sigma), iso N(0,I), mean (mu for every image), perm (across images); content-bearing: permc (within class).
Draws: 3 per variant, seed 92000+r (zm, iso, perm) and 93000+r (permc). Outputs r4/results/S_surrogates{_permodel}.json.
Usage: python r4/surrogates.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import RES3, RES4, load_models, tau_ci, ktau, sets, partial_ci, dump4
from leakfree import setup, ang, lam_for
torch.set_num_threads(os.cpu_count()); ND = 3
FREE = ['gauss', 'zm', 'iso', 'mean', 'perm']


def variant(E, kind, r):
    N, k = E['F'].shape[0], E['Vn'].shape[1]; C = E['F'] @ E['Vn']
    if kind == 'mean': return E['mu'].expand(N, k).clone()
    if kind == 'permc':
        g = np.random.default_rng(93_000 + r); y = E['y'].numpy(); idx = np.arange(N)
        for c in np.unique(y):
            ii = np.where(y == c)[0]; idx[ii] = ii[g.permutation(len(ii))]
        return C[torch.tensor(idx)]
    if kind == 'perm': return C[torch.tensor(np.random.default_rng(92_000 + r).permutation(N))]
    g = torch.Generator().manual_seed(92_000 + r); z = torch.randn(N, k, generator=g, dtype=torch.float64)
    return z @ E['Ls'].T if kind == 'zm' else z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(os.path.dirname(RES3), '..', '_ext', 'sing'))); a = ap.parse_args()
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float64)), dim=-1).mean(1), dim=-1)
    ms, col, sing = load_models(); lk = -sing; rho = col('rho'); AS = col('AS_SING')
    Bp = json.load(open(os.path.join(RES3, 'B_rank_permodel.json'))); Ae = json.load(open(os.path.join(RES3, 'A_equal_energy_permodel.json')))
    cache = os.path.join(RES4, 'S_surrogates_permodel.json'); os.makedirs(RES4, exist_ok=True)
    P = json.load(open(cache)) if os.path.exists(cache) else {}
    for name in ms:
        if name in P: continue
        t0 = time.time(); E = setup(a.feats_root, name); tc = TC[E['y']]; a0 = ang(E['Zrow'], tc); out = {}
        for kind in ['zm', 'iso', 'mean', 'perm', 'permc']:
            v = []
            for r in range(1 if kind == 'mean' else ND):
                Ct = variant(E, kind, r); lam = lam_for(E, Ct, E['rho'])
                v.append(float((ang(E['Zrow'] + lam * (Ct @ E['AVn'].T), tc) - a0).abs().mean()))
            out[kind] = float(np.mean(v)); out[kind + '_sd'] = float(np.std(v))
        out['gauss'] = Bp[name]['AS_surr']; out['AS'] = float((ang(E['F'] @ E['A'].T + E['Ab'], tc) - a0).abs().mean()); out['rho'] = E['rho']
        out['secs'] = time.time() - t0; P[name] = out; json.dump(P, open(cache, 'w'), indent=1)
        print(name, {k: round(v, 4) for k, v in out.items()}, flush=True); del E
    S = sets(ms, rho); kap = {n: Ae[n]['kappa'] for n in ms}
    V = lambda f, sub: np.array([f(P[n], n) for n in sub]); L = lambda sub: np.array([lk[ms.index(n)] for n in sub])
    R_ = lambda sub: np.array([rho[ms.index(n)] for n in sub])
    SC = {k: (lambda kk: (lambda p, n: p[kk]))(k) for k in FREE + ['permc', 'AS']}
    SC.update(X=lambda p, n: p['AS'] / p['gauss'], D_gauss=lambda p, n: p['AS'] - p['gauss'], D_perm=lambda p, n: p['AS'] - p['perm'],
              Q=lambda p, n: p['permc'] / p['perm'], X_perm=lambda p, n: p['AS'] / p['perm'])
    res = dict(per_model=P, tau={}, partial_given_rho={}, sets=S)
    for s, sub in S.items():
        res['tau'][s] = {k: tau_ci(V(f, sub), L(sub)) for k, f in SC.items()}
        res['partial_given_rho'][s] = {k: partial_ci(V(SC[k], sub), L(sub), R_(sub)) for k in ['X', 'D_gauss', 'D_perm', 'Q', 'perm', 'gauss', 'AS']}
    res['class_share'] = {n: (P[n]['permc'] - P[n]['perm']) / (P[n]['AS'] - P[n]['perm']) for n in ms}
    res['tau_gauss_kappa'] = ktau(V(SC['gauss'], ms), [kap[n] for n in ms])
    res['tau_perm_kappa'] = ktau(V(SC['perm'], ms), [kap[n] for n in ms])
    res['AS_check_max_rel_err'] = max(abs(P[n]['AS'] - AS[ms.index(n)]) / AS[ms.index(n)] for n in ms)
    tracks = lambda k: all(res['tau'][s][k]['tau'] >= 0.6 and res['tau'][s][k]['pct'][0] > 0 for s in ('all13', 'S11'))
    res['tracks'] = {k: tracks(k) for k in SC}
    f, c = res['tracks']['perm'], res['tracks']['D_perm']
    res['outcome'] = ('both track: cannot apportion' if f and c else 'content-free only' if f else 'content excess only' if c else 'neither tracks')
    span = [res['tau']['all13'][k]['tau'] for k in FREE]; res['free_tau_range_all13'] = [min(span), max(span)]
    res['surrogate_dependence_must_be_stated'] = bool(max(span) - min(span) > 0.15)
    res['KILL_R4_S'] = bool(all(res['tau'][s]['perm']['tau'] < 0.3 for s in ('all13', 'S11')))
    dump4('S_surrogates.json', res)
    for s in S: print(s, {k: round(v['tau'], 3) for k, v in res['tau'][s].items()})
    print({k: res[k] for k in ['tracks', 'outcome', 'free_tau_range_all13', 'surrogate_dependence_must_be_stated', 'KILL_R4_S', 'tau_gauss_kappa', 'AS_check_max_rel_err']})


if __name__ == '__main__':
    main()
