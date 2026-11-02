"""PREREG_R4B idea 3: class-discriminative (between-class scatter, half A) vs energy-matched random null removal (half B).
k in {5, 20, 50} capped at d_null/2 (k = 20 primary), 20 random draws (seed 96000+j), 1000x within-class image bootstrap of R.
Outputs r4/results/I3_selectivity.json. Usage: python r4/selectivity.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES4, load_models, tau_ci, sets, dump4
from retrained_a4 import text_emb
from leakfree import setup
import estimators as est
torch.set_num_threads(os.cpu_count())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    TC = text_emb(a.sing_root); ms, col, sing = load_models(); lk = -sing; S = sets(ms, col('rho')); P = {}
    for name in S['E12']:
        t0 = time.time(); E = setup(fr, name); C = E['F'] @ E['Vn']; y = E['y']; hv = est.halves(y); nd = C.shape[1]; o = {}
        for k in (5, 20, 50):
            kk = min(k, nd // 2); o[f'k{k}'] = est.selectivity(E['Zrow'], C, E['AVn'].T, y, TC[y], kk, ndraw=20, nboot=1000, halfA=hv)
        o['null_dim'] = nd; o['secs'] = time.time() - t0; P[name] = o
        print(name, nd, {k: (round(v['R'], 3), [round(c, 3) for c in v['ci']], v['k']) for k, v in o.items() if k.startswith('k')}, round(o['secs'], 1), flush=True); del E, C
    L = lambda sub: np.array([lk[ms.index(n)] for n in sub]); res = dict(per_model=P, tau={}, within={})
    for k in ('k5', 'k20', 'k50'):
        res['tau'][k] = {s: tau_ci([P[n][k]['R'] for n in S[s]], L(S[s])) for s in ('E12', 'S11')}
        ci = [P[n][k]['ci'] for n in S['E12']]
        res['within'][k] = dict(n_excl_1=sum(c[0] > 1 or c[1] < 1 for c in ci), n_above=sum(c[0] > 1 for c in ci), n_below=sum(c[1] < 1 for c in ci), n=len(ci))
    t = {s: res['tau']['k20'][s]['tau'] for s in ('E12', 'S11')}
    res['outcome'] = 'R-TRACKS' if all(v >= 0.6 for v in t.values()) else ('R-NULL' if all(abs(v) < 0.3 for v in t.values()) else 'intermediate')
    res['within_model_content_sensitivity'] = res['within']['k20']['n_excl_1'] >= 9; res['KILL_R4B_3'] = res['outcome'] == 'R-TRACKS'
    dump4('I3_selectivity.json', res); print({k: {s: round(v['tau'], 3) for s, v in res['tau'][k].items()} for k in res['tau']}, res['within'], res['outcome'])


if __name__ == '__main__':
    main()
