"""Post-hoc, descriptive checks on the committed round-5 aggregates (no private inputs, no new feature computation).
Writes r5/results/R5_posthoc.json: threshold sensitivity of the round-5 labels, the full-null / row-space scores against SING's
order, raw SO, the SO-XCC agreement, the coverage of the world-C gate and the largest calibration changes."""
import os, sys, json
import numpy as np
R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(R, 'r4')); sys.path.insert(0, os.path.join(R, 'r3'))
from common4 import load_models, tau_ci, ktau, sets  # noqa: E402
J = lambda p: json.load(open(os.path.join(R, p)))
G = J('r5/results/R5_gates.json'); C = J('r5/results/R5_confirm.json'); CMP = J('r5/results/R5_calibration_old_vs_new.json'); CAL = J('r4/results/I2_calibration_v2.json')
P = G['models']; ms, col, sing = load_models(); lk = -sing; rho = col('rho'); S = sets(ms, rho)
lab = lambda tE, tS: 'TRACKS' if tE >= 0.6 and tS >= 0.6 else ('DOES-NOT-TRACK' if tE < 0.3 and tS < 0.3 else 'INCONCLUSIVE')


def vs_sing(key, ok=lambda n: True):
    o = {}
    for s in ('all13', 'E12', 'S11'):
        sub = [n for n in S[s] if ok(n) and np.isfinite(P[n][key])]
        x = np.array([P[n][key] for n in sub]); yv = np.array([lk[ms.index(n)] for n in sub]); rh = np.array([rho[ms.index(n)] for n in sub])
        o[s] = dict(n=len(sub), tau=tau_ci(x, yv), tau_vs_rho=float(ktau(x, rh)))
    return o


out = {}
for key in ('XCC_full', 'SO_row', 'XCC_row', 'SO_raw'):
    out[key] = vs_sing(key, (lambda n: P[n]['SO_est']) if key.startswith('SO') else (lambda n: True))
n13 = list(P); out['tau_SO_XCC24_all13'] = float(ktau(np.array([P[n]['SO'] for n in n13]), np.array([P[n]['XCC24'] for n in n13])))
# SO label under other reliability floors
fl = {}
for f in (0.005, 0.02, 0.05, 0.10, 0.2, 0.3, 0.4):
    ok = lambda n, f=f: P[n]['SO_rel'] >= f; v = vs_sing('SO', ok)
    g4 = {}
    for b, w in G['worlds'].items():
        vals = [r['SO'] for r in w['rows']['C'] if r['SO_rel'] >= f and np.isfinite(r['SO'])]
        g4[b] = dict(n=len(vals), ratio=(float(max(vals) / min(vals)) if len(vals) >= 3 and min(vals) > 0 else None))
    fails = sum(1 for x in g4.values() if x['ratio'] is not None and x['ratio'] > 1.5)
    fl[str(f)] = dict(n_E12=v['E12']['n'], n_S11=v['S11']['n'], tau_E12=v['E12']['tau']['tau'], tau_S11=v['S11']['tau']['tau'],
                     label=lab(v['E12']['tau']['tau'], v['S11']['tau']['tau']), G4=g4, G4_testable=sum(x['ratio'] is not None for x in g4.values()), G4_kill=fails >= 2)
out['SO_floor_sensitivity'] = fl
# G3 at alpha = 0.1
g31 = {}
for k in ('SO', 'XCC24'):
    sd = 'SO_null_sd' if k == 'SO' else 'XCC_null_sd'; ps = 0
    for n in n13:
        ds = (P[n]['G3']['semantic_0.1'][k] - P[n][k]) / P[n][sd]; da = (P[n]['G3']['arbitrary_0.1'][k] - P[n][k]) / P[n][sd]
        ps += bool(np.isfinite(ds) and np.isfinite(da) and ds >= 2 and da <= 1)
    g31[k] = ps
out['G3_alpha0.1_pass'] = g31
# inheritance: bar sensitivity
I = C['inheritance']; fp = {n: P[n]['F1_partial'] for n in I['estimable']}
out['inheritance'] = dict(n=len(fp), scratch_max=0.040, scratch_plant_min=0.103, plant_min=float(min(I['plant_partial_005'].values())),
                          counts={str(b): int(sum(v <= b for v in fp.values())) for b in (0.05, 0.06, 0.10, 0.12)},
                          inherited_at={str(b): bool(sum(v <= b for v in fp.values()) >= 0.75 * len(fp)) for b in (0.05, 0.06, 0.10, 0.12)},
                          high_models=sorted([n for n, v in fp.items() if v > 0.05], key=lambda n: -fp[n]), max_partial=float(max(fp.values())))
# world-C coverage of the passing SO rows on DINOv2-L
rows = [r for r in G['worlds']['dinov2']['rows']['C'] if r['SO_est']]
out['G4_dinov2_rho_range'] = [float(min(r['rho'] for r in rows)), float(max(r['rho'] for r in rows))] if rows else None
out['densenet'] = dict(rho=P['densenet']['rho'], SO=P['densenet']['SO'], XCC24=P['densenet']['XCC24'])
out['NCM_worldC'] = {b: CAL['bases'][b]['worlds']['C']['tau']['NCM_null'] for b in CAL['bases']}
ch = [r for r in CMP['rows'] if r['changed'] and r['estimator'] != 'exponent_a' and r['old'] is not None and r['new'] is not None]
ch.sort(key=lambda r: -abs(r['new'] - r['old']))
out['calib_top_changes'] = [dict(r, abs_delta=abs(r['new'] - r['old'])) for r in ch[:6]]
out['worldD_tau_rho_L'] = {b: CAL['bases'][b]['worlds']['D']['tau'].get('rho') for b in CAL['bases']}
json.dump(out, open(os.path.join(R, 'r5', 'results', 'R5_posthoc.json'), 'w'), indent=1, default=float)
print(json.dumps({k: out[k] for k in ('tau_SO_XCC24_all13', 'G3_alpha0.1_pass', 'inheritance', 'G4_dinov2_rho_range', 'densenet', 'NCM_worldC', 'worldD_tau_rho_L')}, default=float, indent=0))
for k in ('XCC_full', 'SO_row', 'XCC_row', 'SO_raw'): print(k, {s: (out[k][s]['n'], round(out[k][s]['tau']['tau'], 3), [round(v, 2) for v in out[k][s]['tau']['pct']], round(out[k][s]['tau_vs_rho'], 2)) for s in out[k]})
for f, v in fl.items(): print(f, v['n_E12'], v['n_S11'], round(v['tau_E12'], 3), round(v['tau_S11'], 3), v['label'], v['G4_testable'], v['G4_kill'], {b: (x['n'], x['ratio'] and round(x['ratio'], 2)) for b, x in v['G4'].items()})
for r in out['calib_top_changes']: print(r['base'], r['world'], r['estimator'], round(r['old'], 3), round(r['new'], 3), round(r['abs_delta'], 3))
