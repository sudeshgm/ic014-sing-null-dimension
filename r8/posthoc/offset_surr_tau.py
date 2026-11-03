"""POST HOC: tau of the offset_surr.json AS variants vs SING (same tau_ci/paired_ci as the paper), plus r4 'zm'/'perm'/'mean' for comparison."""
import os, sys, json, numpy as np
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.join(R, 'r3')); sys.path.insert(0, os.path.join(R, 'r4'))
from common import load_models, tau_ci, paired_ci
from common4 import sets
H = os.path.dirname(os.path.abspath(__file__))
ms, col, sing = load_models(); lk = -sing; rho = col('rho')
P = json.load(open(os.path.join(H, 'offset_surr.json'))); S4 = json.load(open(os.path.join(R, 'r4/results/S_surrogates_permodel.json')))
B3 = json.load(open(os.path.join(R, 'r3/results/B_rank_permodel.json')))
V = {'AS (real, paper)': [P[m]['AS'] for m in ms], 'AS real centred (offset removed)': [P[m]['real_c'] for m in ms],
     'AS offset only (natural size)': [P[m]['off_nat'] for m in ms],
     'AS_surr gauss N(mu,Sigma) @ full rho (paper)': [B3[m]['AS_surr'] for m in ms],
     'r4 zm N(0,Sigma) @ full rho': [S4[m]['zm'] for m in ms], 'r4 perm @ full rho': [S4[m]['perm'] for m in ms],
     'r4 mean (offset rescaled to full rho)': [S4[m]['mean'] for m in ms],
     'zm_c N(0,Sigma) @ centred rho': [P[m]['zm_c'] for m in ms], 'perm_c (C-mu permuted) @ centred rho': [P[m]['perm_c'] for m in ms]}
V = {k: np.array(v) for k, v in V.items()}
out = dict(note='POST HOC', tau={}, paired={})
for s, sub in sets(ms, rho).items():
    idx = [ms.index(n) for n in sub]; y = lk[idx]
    out['tau'][s] = {k: tau_ci(v[idx], y) for k, v in V.items()}
    out['paired'][s] = {f'{a} - {b}': paired_ci(V[a][idx], V[b][idx], y) for a, b in
                        [('zm_c N(0,Sigma) @ centred rho', 'AS_surr gauss N(mu,Sigma) @ full rho (paper)'),
                         ('AS real centred (offset removed)', 'AS (real, paper)')]}
json.dump(out, open(os.path.join(H, 'offset_surr_tau.json'), 'w'), indent=1, default=float)
f = lambda d: f"{d['tau']:+.2f} pct [{d['pct'][0]:+.2f},{d['pct'][1]:+.2f}] BCa [{d['bca'][0]:+.2f},{d['bca'][1]:+.2f}]"
for s in out['tau']:
    print('==', s)
    for k, d in out['tau'][s].items(): print(f'  {k:46s} {f(d)}')
    for k, d in out['paired'][s].items(): print(f"  paired {k:80s} {d['diff']:+.2f} pct [{d['pct'][0]:+.2f},{d['pct'][1]:+.2f}] BCa [{d['bca'][0]:+.2f},{d['bca'][1]:+.2f}]")
