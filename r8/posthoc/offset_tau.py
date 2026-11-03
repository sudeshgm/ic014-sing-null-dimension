"""POST HOC (3 Nov 2026, round-8 exploration, not pre-registered): does the constant null-mean offset change the energy headline?
tau-b against SING's order (lk = -SING IS/AS ratio, as r3/paired_stats.py) with r3/common.tau_ci and paired_ci (B=2000, seed 0,
percentile + BCa), on all13 / E12 / S11 (r4/common4.sets). Components from rho_decomp.json (uncentred plug-in identity)."""
import os, sys, json, numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(R, 'r3')); sys.path.insert(0, os.path.join(R, 'r4'))
from common import load_models, tau_ci, paired_ci, ktau
from common4 import sets
ms, col, sing = load_models(); lk = -sing; rho = col('rho'); AS = col('AS_SING')
D = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rho_decomp.json')))
g = lambda k: np.array([D[m][k] for m in ms])
X = {'rho (uncentred, paper)': rho, 'rho_centred = between+within': g('rho_between') + g('rho_within'),
     'rho_centred_debiased = between_db+within_db': g('rho') - g('rho_offset') ,  # placeholder replaced below
     'rho_offset': g('rho_offset'), 'rho_between': g('rho_between'), 'rho_between_debiased': g('rho_between_debiased'),
     'rho_within': g('rho_within'), 'AS_SING': AS}
del X['rho_centred_debiased = between_db+within_db']  # centred total is identical either way (debiasing only moves noise between terms)
S = sets(ms, rho); out = dict(note='POST HOC', sets=S, tau={}, paired={})
for s, sub in S.items():
    idx = [ms.index(n) for n in sub]; y = lk[idx]
    out['tau'][s] = {k: tau_ci(v[idx], y) for k, v in X.items()}
    out['paired'][s] = {f'{a} - {b}': paired_ci(X[a][idx], X[b][idx], y) for a, b in
                        [('rho_centred = between+within', 'rho (uncentred, paper)'), ('rho_centred = between+within', 'AS_SING'),
                         ('rho (uncentred, paper)', 'AS_SING'), ('rho_between', 'AS_SING'), ('rho_within', 'AS_SING'), ('rho_offset', 'AS_SING')]}
    out[f'tau_centred_vs_uncentred_{s}'] = ktau(X['rho_centred = between+within'][idx], rho[idx])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offset_tau.json'), 'w'), indent=1, default=float)
f = lambda d: f"{d['tau']:+.2f} pct [{d['pct'][0]:+.2f},{d['pct'][1]:+.2f}] BCa [{d['bca'][0]:+.2f},{d['bca'][1]:+.2f}]"
for s in S:
    print('==', s, len(S[s]), 'models')
    for k, d in out['tau'][s].items(): print(f'  {k:32s} {f(d)}')
    for k, d in out['paired'][s].items(): print(f"  paired {k:55s} {d['diff']:+.2f} pct [{d['pct'][0]:+.2f},{d['pct'][1]:+.2f}] BCa [{d['bca'][0]:+.2f},{d['bca'][1]:+.2f}]")
    print('  tau(centred, uncentred) =', round(out[f'tau_centred_vs_uncentred_{s}'], 3))
