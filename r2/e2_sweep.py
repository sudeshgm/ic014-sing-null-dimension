"""Round-2 fixes 2, 3, 4, 5 and the KILL-R check, from r2/results/fix2_models.csv and fix5_tolerance.csv (no features needed).
Lower score = less leakage; tau-b of (-score) vs the digitised SING Supp. Fig. 12b IS/AS bars (higher = better), as in E2.
Usage: python r2/e2_sweep.py"""
import os, sys, csv, json, numpy as np
from scipy.stats import kendalltau, spearmanr
R = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(R, '..', 'e2'))
FIG6B = {'densenet': 70.71, 'dinovit1': 70.16, 'eva02_large': 54.32, 'bitresnet': 47.21, 'convnext': 37.38, 'efficientnet': 36.28,
         'swin_large': 29.18, 'resnet': 24.81, 'vgg16': 24.26, 'vgg19': 19.89, 'dinov2': 16.61, 'resnext': 13.88, 'deit': 9.51}
TAU_SING = 0.8205128205128204   # AS_SING tau, E2 (pre-registered reference 0.821)
rows = {r['model']: r for r in csv.DictReader(open(os.path.join(R, 'results/fix2_models.csv')))}
f = lambda m, k: float(rows[m][k])
NORM = {'N1_AS/rho^0.25': lambda m: f(m, 'AS_SING') / f(m, 'rho') ** 0.25, 'N2_AS/rho^0.43': lambda m: f(m, 'AS_SING') / f(m, 'rho') ** 0.43,
        'N3_AS/rho^0.5': lambda m: f(m, 'AS_SING') / f(m, 'rho') ** 0.5, 'N4_AS/rho': lambda m: f(m, 'AS_SING') / f(m, 'rho'),
        'N5_AS/d_null': lambda m: f(m, 'AS_SING') / f(m, 'null_dim'), 'N6_AS_fair_A': lambda m: f(m, 'AS_fair_A'),
        'N7_AS_fair_B': lambda m: f(m, 'AS_fair_B'), 'N8_AS_fair_C': lambda m: f(m, 'AS_fair_C')}
EXTRA = {'AS_SING': lambda m: f(m, 'AS_SING'), 'rho_alone': lambda m: f(m, 'rho'), 'M1_concept_mean (fix 10, descriptive)': lambda m: f(m, 'M1_concept_mean')}

def tau(models, fn): return float(kendalltau([-fn(m) for m in models], [FIG6B[m] for m in models]).statistic)

def boot(models, fn, stat='tau', B=2000):
    rng = np.random.default_rng(0); out = []
    for _ in range(B):
        s = [models[i] for i in rng.integers(0, len(models), len(models))]
        if len(set(s)) < 4: continue
        if stat == 'tau': v = kendalltau([-fn(m) for m in s], [FIG6B[m] for m in s]).statistic
        else: v = spearmanr([fn(m) for m in s], [f(m, 'rho') for m in s]).statistic
        if np.isfinite(v): out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], len(out)

def evaluate(models):
    res = {}
    for k, fn in {**NORM, **EXTRA}.items():
        ci, nb = boot(models, fn); loo = [tau([x for x in models if x != m], fn) for m in models]
        res[k] = dict(tau=tau(models, fn), ci95=ci, n_boot=nb, loo_range=[min(loo), max(loo)])
    fam = {x: res[f'N{i}_AS_fair_{x}']['tau'] for i, x in zip([6, 7, 8], 'ABC')}
    fix2 = sum(v < 0.3 for v in fam.values()) >= 2
    dweak = sum(res[k]['ci95'][1] < TAU_SING for k in NORM); dstrong = sum(res[k]['ci95'][1] < 0 for k in NORM)
    return res, dict(family_tau=fam, fix2_pass=fix2, n_ci_excl_0821_below=dweak, D_weak_pass=dweak >= 5, n_ci_upper_below_0=dstrong, D_strong_pass=dstrong >= 5)

ALL = list(rows); SCR = [m for m in ALL if rows[m]['screen_exclude'] != 'True']
out = dict(models_all=ALL, screened=SCR, excluded=[m for m in ALL if m not in SCR])
for lab, ms in [('all13', ALL), ('screened', SCR)]:
    r, c = evaluate(ms); out[lab] = dict(scores=r, criteria=c)
c1, c2 = out['all13']['criteria'], out['screened']['criteria']
fix2_ok = c1['fix2_pass'] and c2['fix2_pass']; dweak_ok = c1['D_weak_pass'] and c2['D_weak_pass']; dstrong_ok = c1['D_strong_pass'] and c2['D_strong_pass']
out['KILL_R'] = dict(fires=not (fix2_ok and dweak_ok), fix2_both=fix2_ok, D_weak_both=dweak_ok, D_strong_both=dstrong_ok,
                     allowed_wording='none (claim dropped)' if not (fix2_ok and dweak_ok) else ('reverses' if dstrong_ok else 'differs from'))
# fix 4: Spearman(AS_fair_A, rho)
fx4 = {}
for lab, ms in [('all13', ALL), ('without_densenet_bit', [m for m in ALL if m not in ('densenet', 'bitresnet')])]:
    s = float(spearmanr([f(m, 'AS_fair_A') for m in ms], [f(m, 'rho') for m in ms]).statistic); ci, _ = boot(ms, lambda m: f(m, 'AS_fair_A'), 'spearman')
    fx4[lab] = dict(spearman=s, ci95=ci, removes_trend_allowed=abs(s) < 0.3 and ci[0] <= 0 <= ci[1])
out['fix4'] = fx4
# fix 5: tolerance stability
tr = list(csv.DictReader(open(os.path.join(R, 'results/fix5_tolerance.csv')))); fx5 = {}
for lab, ms in [('all13', ALL), ('screened', SCR)]:
    g = lambda kind, t, k: [float(next(x for x in tr if x['model'] == m and x['tol_kind'] == kind and float(x['tol']) == t)[k]) for m in ms]
    o = {}
    for k in ['null_dim', 'rho', 'AS_SING', 'AS_fair_A']:
        adj = {f'{a}-{b}': float(kendalltau(g('abs', a, k), g('abs', b, k)).statistic) for a, b in [(1e-6, 1e-5), (1e-5, 1e-4), (1e-4, 1e-3)]}
        rel = {f'rel{t}': float(kendalltau(g('abs', 1e-5, k), g('rel', t, k)).statistic) for t in [1e-5, 1e-4, 1e-3]}
        o[k] = dict(adjacent_abs=adj, vs_rel=rel, min_adjacent=min(v for v in adj.values() if np.isfinite(v)) if any(np.isfinite(v) for v in adj.values()) else float('nan'))
    o['pass'] = all(all((not np.isfinite(v)) or v >= 0.9 for v in o[k]['adjacent_abs'].values()) for k in ['null_dim', 'rho', 'AS_SING', 'AS_fair_A'])
    fx5[lab] = o
out['fix5'] = fx5
out['null_dims_by_tol'] = {m: {f"{x['tol_kind']}{x['tol']}": int(x['null_dim']) for x in tr if x['model'] == m} for m in ALL}
json.dump(out, open(os.path.join(R, 'results/fix2345_sweep.json'), 'w'), indent=1, default=float)
with open(os.path.join(R, 'results/fix3_normalisers.csv'), 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(['set', 'score', 'tau', 'ci_lo', 'ci_hi', 'loo_min', 'loo_max'])
    for lab in ['all13', 'screened']:
        for k, v in out[lab]['scores'].items(): w.writerow([lab, k, v['tau'], *v['ci95'], *v['loo_range']])
for lab in ['all13', 'screened']:
    print(lab, json.dumps(out[lab]['criteria']))
    for k, v in out[lab]['scores'].items(): print(f"  {k:40s} {v['tau']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] LOO [{v['loo_range'][0]:+.3f},{v['loo_range'][1]:+.3f}]")
print('KILL_R', out['KILL_R']); print('fix4', json.dumps(fx4)); print('fix5', json.dumps({k: (v['pass'], {s: v[s]['adjacent_abs'] for s in ['null_dim', 'rho', 'AS_SING', 'AS_fair_A']}) for k, v in fx5.items()}))
print('excluded', out['excluded'])
