"""PREREG_R6 §2-§4: test statistics, per-score verdicts, cross-model taus and the counting rule, from r6/results/models/*.json and
r6/results/worlds/*.json. Writes r6/results/R6_summary.json. CPU, seconds."""
import os, sys, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
sys.path.insert(0, os.path.join(ROOT, 'r4')); sys.path.insert(0, os.path.join(ROOT, 'r3'))
from common4 import load_models, tau_ci, ktau, sets, partial_ci  # noqa: E402
RES6 = os.path.join(ROOT, 'r6', 'results')
PRIMARY = ('ViM', 'ActSub', 'CORE', 'Vinfo'); DESCR = ('ViM_s', 'CORE_ker', 'Vinfo_corr', 'NuSA'); CAL_BASES = ('resnet', 'dinov2', 'resnext')


def tau(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return float('nan') if len(x) < 3 or np.ptp(x) <= 1e-12 * max(np.abs(x).max(), 1e-300) or np.ptp(y) == 0 else float(ktau(x, y))


def score_block(k, P, S, ms, rho_all, lk):
    names = list(P); real = {n: P[n]['real'][k] for n in names}; rng_ = float(np.ptp(list(real.values()))); sd = {n: P[n]['real']['sd_boot'][k] for n in names}
    est = [n for n in names if sd[n] <= 0.1 * rng_]; o = dict(real=real, sd_boot=sd, range13=rng_, estimable=est)
    o['n_E12'] = sum(n in est for n in S['E12']); o['n_S11'] = sum(n in est for n in S['S11']); o['verdict_possible'] = o['n_E12'] >= 9 and o['n_S11'] >= 8
    e1 = {n: P[n]['E1'][k] for n in names}; o['E1'] = e1
    o['tau_EQ'] = {s: tau([e1[n] for n in S[s] if n in est], [real[n] for n in S[s] if n in est]) for s in ('E12', 'S11')}
    o['R_B'] = {n: float(np.ptp([r[k] for r in P[n]['E2']])) / rng_ for n in names}; o['R_B_median'] = float(np.median([o['R_B'][n] for n in est])) if est else float('nan')
    teo = {}
    for b in CAL_BASES:
        curve = {round(r['target'], 12): r[k] for r in P[b]['E2']}; sub = [n for n in S['E12'] if n in est]
        teo[b] = tau([real[n] for n in sub], [curve[round(float(rho_all[ms.index(n)]), 12)] for n in sub])
    o['tau_EO'] = teo; o['tau_EO_median'] = float(np.nanmedian(list(teo.values()))) if any(np.isfinite(v) for v in teo.values()) else float('nan')
    o['C1'] = {n: (real[n] - np.mean([r[k] for r in P[n]['C1']])) / sd[n] for n in names}
    o['C2'] = {n: (np.mean([r[k] for r in P[n]['C2']]) - real[n]) / sd[n] for n in names}
    o['C3_arb'] = {n: {a: (P[n]['C3'][f'arbitrary_{a}'][k] - real[n]) / sd[n] for a in ('0.1', '0.3')} for n in names}
    o['C3_sem'] = {n: {a: (P[n]['C3'][f'semantic_{a}'][k] - real[n]) / sd[n] for a in ('0.1', '0.3')} for n in names}
    o['C2_noisy'] = sum(abs(o['C2'][n]) > 1 for n in names) > 3
    nC1_hi = sum(abs(o['C1'][n]) >= 2 for n in names); nC1_lo = sum(abs(o['C1'][n]) <= 1 for n in names)
    nP_hi = sum(abs(o['C3_arb'][n]['0.3']) >= 2 for n in names); nP_lo = sum(abs(o['C3_arb'][n]['0.3']) <= 1 for n in names)
    o['counts'] = dict(C1_ge2=nC1_hi, C1_le1=nC1_lo, plant_ge2=nP_hi, plant_le1=nP_lo)
    t = o['tau_EQ']
    if not o['verdict_possible']: ev = 'NO VERDICT (G5)'
    elif all(np.isfinite(v) and v < 0.3 for v in t.values()) and o['R_B_median'] >= 0.5: ev = 'ENERGY-DRIVEN'
    elif all(np.isfinite(v) and v >= 0.6 for v in t.values()) and o['R_B_median'] <= 0.1: ev = 'NOT ENERGY-DRIVEN'
    else: ev = 'MIXED'
    cv = 'CONTENT-RESPONSIVE' if (nC1_hi >= 10 and nP_hi >= 10) else ('CONTENT-BLIND' if (nC1_lo >= 10 and nP_lo >= 10) else 'PARTIAL')
    o['energy_verdict'] = ev; o['content_verdict'] = cv if o['verdict_possible'] else 'NO VERDICT (G5)'
    cm = {}
    for s in ('all13', 'E12', 'S11'):
        sub = [n for n in S[s] if n in est]; x = np.array([real[n] for n in sub]); yv = np.array([lk[ms.index(n)] for n in sub]); rh = np.array([rho_all[ms.index(n)] for n in sub])
        if len(sub) >= 5:
            cm[s] = dict(n=len(sub), tau_sing=tau_ci(x, yv), tau_rho=tau(x, rh), partial_given_rho=partial_ci(x, yv, rh))
    o['cross_model'] = cm
    if all(s in cm for s in ('E12', 'S11')):
        ts = [cm[s]['tau_sing']['tau'] for s in ('E12', 'S11')]
        o['tracks_sing'] = 'TRACKS' if all(v >= 0.6 for v in ts) else ('DOES-NOT-TRACK' if all(v < 0.3 for v in ts) else 'INCONCLUSIVE')
    return o


def worlds_block(k, Wd):
    o = {}
    for b, w in Wd.items():
        rows = w['worlds']; ex = [r[k] - r['floor'][k] for r in rows['C']]; v = np.asarray(ex, float)
        o[b] = dict(C_excess=ex, C_ratio=float(v.max() / v.min()) if v.min() > 0 else float('inf'),
                    C_pass=bool(v.min() > 0 and v.max() / v.min() <= 1.5), C_tau_rho=tau([r[k] for r in rows['C']], [r['rho'] for r in rows['C']]),
                    M_tau_L=tau([r[k] for r in rows['M']], [r['L'] for r in rows['M']]), D_tau_L=tau([r[k] for r in rows['D']], [r['L'] for r in rows['D']]))
    return o


def main():
    ms, col, sing = load_models(); lk = -sing; rho_all = col('rho'); S = sets(ms, rho_all)
    P = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES6, 'models', '*.json')))}
    P = {n: P[n] for n in ms if n in P}; Wd = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES6, 'worlds', '*.json')))}
    out = dict(models=list(P), max_dlogit=max(max(P[n]['real']['max_dlogit'], P[n]['E1']['max_dlogit'], max(r['max_dlogit'] for r in P[n]['E2'])) for n in P), scores={})
    for k in PRIMARY + DESCR:
        out['scores'][k] = score_block(k, P, S, ms, rho_all, lk); out['scores'][k]['worlds'] = worlds_block(k, Wd)
    counted = [k for k in PRIMARY if out['scores'][k]['verdict_possible']]; n = len(counted); kk = sum(out['scores'][k]['energy_verdict'] == 'ENERGY-DRIVEN' for k in counted)
    good = sum(out['scores'][k]['energy_verdict'] == 'NOT ENERGY-DRIVEN' and out['scores'][k]['content_verdict'] == 'CONTENT-RESPONSIVE' for k in counted)
    if n <= 2: claim = 'NO FAMILY-LEVEL STATEMENT (n <= 2)'
    elif (n == 4 and kk >= 3) or (n == 3 and kk >= 2): claim = 'GENERAL'
    elif n == 4 and kk == 2: claim = 'DESIGN-DEPENDENT'
    else: claim = 'SING-SPECIFIC'
    out['counting'] = dict(counted=counted, n=n, k=kk, claim=claim, content_measurable=bool(good >= 2), n_not_energy_and_responsive=good)
    json.dump(out, open(os.path.join(RES6, 'R6_summary.json'), 'w'), indent=1, default=float)
    for k in PRIMARY + DESCR:
        b = out['scores'][k]; print(k, b['energy_verdict'], b['content_verdict'], 'tauEQ', {s: round(v, 2) for s, v in b['tau_EQ'].items()}, 'R_B', round(b['R_B_median'], 3),
                                      'tauEO', round(b['tau_EO_median'], 2), b['counts'], 'est', len(b['estimable']), b.get('tracks_sing'))
    print('counting', out['counting'], 'max dlogit', out['max_dlogit'])


if __name__ == '__main__':
    main()
