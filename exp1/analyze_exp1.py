"""IC014 Exp 1 CPU analysis of exp1_per_seed.json (from run_exp1_gpu.py).
Outputs: exp1_curves.csv (mean/SD over seeds per model,d), exp1_fits.json (AS vs d, vs rho_d, vs log d; r with 2-sigma band),
exp1_d24_ranking.json (8-model ranking at d=24, Kendall tau vs SING Fig.6b order).
Reference ranking: SING publishes no AS table covering these 8 models (main Fig.3 / Table 1 cover 5 models), so the
reference is the Fig.6b IS/AS ratio order (higher ratio = better = rank 1) from ../exp0/fig6b_digitized.json.
Our ranking: ascending mean|AS| at d=24 (lower = better). tau computed between the two rank orders."""
import json, sys, csv, os, numpy as np
from scipy.stats import kendalltau, linregress
src = sys.argv[1] if len(sys.argv) > 1 else 'results/exp1_per_seed.json'
out = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(src)
rows = json.load(open(src))
EXP1 = ['resnet', 'resnext', 'bitresnet', 'dinov2']; MATCHED = ['deit', 'eva02_large', 'convnext', 'densenet']
FIG = {'densenet': 'DenseNet', 'eva02_large': 'EVA02Large', 'bitresnet': 'BitResNet', 'convnext': 'ConvNext', 'resnet': 'ResNet50',
       'dinov2': 'DinoV2', 'resnext': 'ResNext101', 'deit': 'DeiT', 'vgg16': 'VGG16', 'vgg19': 'VGG19', 'efficientnet': 'EfficientNet',
       'swin_large': 'SwinLarge', 'dinovit1': 'DinoViT'}
fig = {r['model']: r['ratio_IS_over_AS'] for r in json.load(open(os.path.join(os.path.dirname(__file__), '../exp0/fig6b_digitized.json')))['values']}
curves = {}
for r in rows:
    key = (r['model'], r['d'], r['full'])
    curves.setdefault(key, []).append(r)
cur = []
for (m, d, full), rs in sorted(curves.items()):
    A = np.array([x['abs_AS'] for x in rs]); I = np.array([x['IS'] for x in rs]); R = np.array([x['rho'] for x in rs])
    cur.append(dict(model=m, d=d, full_null=full, n_seeds=len(rs), absAS_mean=A.mean(), absAS_sd=A.std(ddof=1) if len(A) > 1 else 0.0,
                    IS_mean=I.mean(), IS_sd=I.std(ddof=1) if len(I) > 1 else 0.0, rho_mean=R.mean(), rho_sd=R.std(ddof=1) if len(R) > 1 else 0.0,
                    signedAS_mean=np.mean([x['AS'] for x in rs]), max_logit_change=max(x['max_logit_change'] for x in rs)))
with open(os.path.join(out, 'exp1_curves.csv'), 'w') as f:
    w = csv.DictWriter(f, fieldnames=list(cur[0])); w.writeheader(); w.writerows(cur)
get = lambda m, d: next((c for c in cur if c['model'] == m and c['d'] == d), None)
fits = {}
for m in EXP1:
    cs = sorted([c for c in cur if c['model'] == m], key=lambda c: c['d'])
    if len(cs) < 2: continue
    d = np.array([c['d'] for c in cs], float); A = np.array([c['absAS_mean'] for c in cs]); rho = np.array([c['rho_mean'] for c in cs])
    lin_d, lin_rho, lin_logd = linregress(d, A), linregress(rho, A), linregress(np.log(d), A)
    a24, a_full = get(m, 24), max(cs, key=lambda c: c['d'])
    r = a_full['absAS_mean'] / a24['absAS_mean']
    sr = r * a24['absAS_sd'] / a24['absAS_mean']   # full-null AS is deterministic; band from d=24 seed SD
    fits[m] = dict(d=d.tolist(), absAS=A.tolist(), rho=rho.tolist(),
                   fit_AS_vs_d=dict(slope=lin_d.slope, intercept=lin_d.intercept, r2=lin_d.rvalue ** 2),
                   fit_AS_vs_rho=dict(slope=lin_rho.slope, intercept=lin_rho.intercept, r2=lin_rho.rvalue ** 2),
                   fit_AS_vs_logd=dict(slope=lin_logd.slope, intercept=lin_logd.intercept, r2=lin_logd.rvalue ** 2),
                   r_AS1048_over_AS24=r, r_2sigma=[r - 2 * sr, r + 2 * sr], d_full=a_full['d'])
json.dump(fits, open(os.path.join(out, 'exp1_fits.json'), 'w'), indent=1)
# d=24 comparison over 8 models: Exp1 models + densenet at random 24-dim subspace; deit/eva02/convnext at full null (=24)
d24 = {}
for m in EXP1 + MATCHED:
    c = get(m, 24)
    if c: d24[m] = dict(absAS=c['absAS_mean'], sd=c['absAS_sd'], IS=c['IS_mean'], full_null=c['full_null'])
dn = get('densenet', 27)
if dn: d24['densenet_full27'] = dict(absAS=dn['absAS_mean'], IS=dn['IS_mean'])
ms = [m for m in EXP1 + MATCHED if m in d24]
ours = sorted(ms, key=lambda m: d24[m]['absAS']); ref = sorted(ms, key=lambda m: -fig[FIG[m]])
tau = kendalltau([ours.index(m) for m in ms], [ref.index(m) for m in ms])
# secondary: rank by IS/|AS| at d=24 (same quantity as Fig.6b) vs Fig.6b
ours_ratio = sorted(ms, key=lambda m: -d24[m]['IS'] / d24[m]['absAS'])
tau2 = kendalltau([ours_ratio.index(m) for m in ms], [ref.index(m) for m in ms])
res = dict(reference='SING Supp. Fig. 6b IS/AS ratio order (no published AS table for these 8 models)',
           values_d24=d24, ranking_by_absAS_d24=ours, ranking_by_ratio_d24=ours_ratio, reference_ranking=ref,
           kendall_tau_absAS_vs_fig6b=dict(tau=tau.statistic, p=tau.pvalue),
           kendall_tau_ratio_vs_fig6b=dict(tau=tau2.statistic, p=tau2.pvalue))
# boring baseline: Spearman(rho_full, AS_full) over all models with a full-null row
from scipy.stats import spearmanr
full = {c['model']: c for c in cur if c['full_null']}
if len(full) >= 3:
    mm = sorted(full)
    sp = spearmanr([full[m]['rho_mean'] for m in mm], [full[m]['absAS_mean'] for m in mm])
    sp2 = spearmanr([full[m]['rho_mean'] for m in mm], [fig[FIG[m]] for m in mm])
    res['boring_baseline'] = dict(models=mm, rho_full=[full[m]['rho_mean'] for m in mm], absAS_full=[full[m]['absAS_mean'] for m in mm],
                                  IS_full=[full[m]['IS_mean'] for m in mm],
                                  spearman_rho_vs_absAS=[sp.statistic, sp.pvalue], spearman_rho_vs_fig6b_ratio=[sp2.statistic, sp2.pvalue])
json.dump(res, open(os.path.join(out, 'exp1_d24_ranking.json'), 'w'), indent=1, default=float)
print(json.dumps(dict(r={m: (round(f['r_AS1048_over_AS24'], 3), [round(x, 3) for x in f['r_2sigma']]) for m, f in fits.items()},
                      ranking=ours, ref=ref, tau=tau.statistic, tau_ratio=tau2.statistic), default=float))
