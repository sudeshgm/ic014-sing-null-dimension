"""Regenerate every paper figure from committed result files (no features needed).
Usage: python r2/make_figs.py  -> paper_workshop/figs/*.pdf|png"""
import json, os, csv
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'); OUT = f'{R}/paper_workshop/figs'; os.makedirs(OUT, exist_ok=True)
NICE = dict(vgg16='VGG16', vgg19='VGG19', densenet='DenseNet', resnet='ResNet50', dinovit1='DinoViT', efficientnet='EffNet',
            bitresnet='BiT', resnext='ResNeXt', convnext='ConvNeXt', swin_large='Swin-L', dinov2='DINOv2', deit='DeiT3', eva02_large='EVA02')
rows = lambda p: list(csv.DictReader(open(p)))
def save(fig, name):  # no creation date in the PDF, so reruns are byte-identical
    fig.tight_layout(); fig.savefig(f'{OUT}/{name}.pdf', dpi=250, metadata={'CreationDate': None}); fig.savefig(f'{OUT}/{name}.png', dpi=250); plt.close(fig)

# Fig 1: Exp 0b
d = json.load(open(f'{R}/exp0b/results/results_top1.json'))
C = [('tv_vit_l_16_IMAGENET1K_V1', 'tv ViT-L/16 V1'), ('timm_vit_large_patch16_224.augreg_in21k_ft_in1k', 'timm AugReg (yaml)'),
     ('tv_vit_l_16_IMAGENET1K_SWAG_LINEAR_V1', 'SWAG linear'), ('tv_vit_l_16_IMAGENET1K_SWAG_E2E_V1', 'SWAG E2E'), ('hf_google/vit-large-patch16-224', 'HF google/vit-L')]
v = [d[f'{k}|native']['top1'] for k, _ in C]
fig, ax = plt.subplots(figsize=(4, 2.9)); b = ax.bar(range(5), v, color='#4c78a8')
for i, x in enumerate(v): ax.text(i, x + .2, f'{x:.1f}', ha='center', fontsize=8)
ax.axhline(84, color='tab:red', ls='--', label='SING reported 84.0%'); ax.axhspan(83.7, 84.3, color='tab:red', alpha=.15)
ax.set_ylim(75, 90); ax.set_xticks(range(5)); ax.set_xticklabels([n for _, n in C], rotation=25, ha='right'); ax.set_ylabel('ImageNet-val top-1 (%)')
ax.set_title('Exp 0b: no candidate matches 84.0%', fontsize=10); ax.legend(frameon=False, fontsize=8, loc='upper left'); save(fig, 'fig_exp0b')

# Fig 2: Exp 1 curves
cv = rows(f'{R}/exp1/results/exp1_curves.csv'); fig, axs = plt.subplots(1, 2, figsize=(8, 3))
for m, lab in [('resnet', 'ResNet50'), ('resnext', 'ResNeXt101'), ('dinov2', 'DINOv2-L'), ('bitresnet', 'BiT-R50x1')]:
    r = sorted([x for x in cv if x['model'] == m], key=lambda x: int(x['d']))
    dd = [int(x['d']) for x in r]; a = [float(x['absAS_mean']) for x in r]; s = [float(x['absAS_sd'] or 0) for x in r]; rh = [float(x['rho_mean']) for x in r]
    axs[0].errorbar(dd, a, yerr=s, marker='o', ms=4, capsize=2, label=lab); axs[1].plot(rh, a, marker='o', ms=4, label=lab)
for ax, t, xl in [(axs[0], '(a) AS vs null-subspace size', 'subspace dim $d$'), (axs[1], r'(b) AS vs $\rho_d$', r'removed energy fraction $\rho_d$')]:
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(t, fontsize=10); ax.set_xlabel(xl); ax.set_ylabel('mean |AS| (deg)'); ax.legend(frameon=False, fontsize=8)
save(fig, 'fig_exp1_curves')

# Fig 3: E1 energy
ce = [x for x in rows(f'{R}/e1/results/e1_cells.csv') if x['feasible'] == 'True' and float(x['absAS']) > 0]
fit = json.load(open(f'{R}/e1/results/e1_fit.json')); ci = json.load(open(f'{R}/r2/results/fix6_e1_ci.json'))
fig, ax = plt.subplots(figsize=(4, 3.1))
for m, lab, mk in [('resnet', 'ResNet50', 'o'), ('resnext', 'ResNeXt101', 's'), ('dinov2', 'DINOv2-L', '^'), ('vgg16', 'VGG16', 'D'), ('eva02_large', 'EVA02-L', 'v')]:
    r = [x for x in ce if x['model'] == m]
    if r: ax.scatter([float(x['rho']) for x in r], [float(x['absAS']) for x in r], marker=mk, s=18, alpha=.75, label=lab)
xs = np.logspace(-4, np.log10(.5)); c0 = np.mean(list(fit['c_model'].values()))
ax.plot(xs, np.exp(c0 + fit['b'] * np.log(100)) * xs ** fit['a'], 'k-', lw=1, alpha=.7, label=f"fit $a={fit['a']:.2f}$")
ax.plot(xs, np.exp(c0 + fit['b'] * np.log(100)) * xs ** .5 * 0.01 ** (fit['a'] - .5), 'k--', lw=1, label=r'theory $\propto\sqrt{\rho}$')
ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'removed energy fraction $\rho$'); ax.set_ylabel('mean |AS| (deg)')
ax.set_title(r'E1: |AS| tracks $\rho$ ($a\approx0.4$–$0.5$)', fontsize=10); ax.legend(frameon=False, fontsize=7); save(fig, 'fig_e1_energy')

# Fig 4/5: E2 ranking (1-based) and scatter, from r2/results/fix2_models.csv
f2 = {x['model']: x for x in rows(f'{R}/r2/results/fix2_models.csv')}
dig = json.load(open(f'{R}/e2/results/e2_rankings.json')); order = dig['fig6b_ranking_best_first']; sw = json.load(open(f'{R}/r2/results/fix2345_sweep.json'))['all13']['scores']
def rank(sc): s = sorted(order, key=lambda m: sc[m]); return {m: s.index(m) + 1 for m in order}
rs = rank({m: float(f2[m]['AS_SING']) for m in order}); rf = rank({m: float(f2[m]['AS_fair_A']) for m in order}); rg = {m: i + 1 for i, m in enumerate(order)}
rp = rank({m: float(f2[m]['rho']) for m in order})
fig, ax = plt.subplots(figsize=(8, 3.4)); x = np.arange(13); w = .21
for k, (rr, col, lab) in enumerate([(rg, '#bbbbbb', 'SING Supp. Fig. 12b (IS/AS)'), (rp, 'tab:green', r'$\rho$ alone'), (rs, 'tab:blue', r'AS$_{\rm SING}$'), (rf, 'tab:red', r'AS$_{\rm fair}$ (family A)')]):
    y = [rr[m] for m in order]; ax.bar(x + (k - 1.5) * w, y, w, color=col, label=lab)
    ax.scatter([xi + (k - 1.5) * w for xi, yi in zip(x, y) if yi == 1], [1] * sum(yi == 1 for yi in y), marker='*', s=60, color='k', zorder=3)
ax.set_xticks(x); ax.set_xticklabels([NICE[m] for m in order], rotation=35, ha='right'); ax.set_ylim(13.5, 0); ax.set_yticks([1, 4, 7, 10, 13])
ax.set_ylabel('rank (1 = least leakage)'); ax.legend(frameon=False, fontsize=8, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.32))
ax.set_title(rf"Ranks vs SING Supp. Fig. 12b ($\tau_\rho={sw['rho_alone']['tau']:.2f}$, $\tau_{{\rm SING}}={sw['AS_SING']['tau']:.2f}$, $\tau_{{\rm fair}}={sw['N6_AS_fair_A']['tau']:.2f}$; $\star$ = rank 1)", fontsize=10)
save(fig, 'fig_e2_ranking')
fig, axs = plt.subplots(1, 2, figsize=(8, 3)); rho = {m: float(f2[m]['rho']) for m in order}
for ax, key, col, t in [(axs[0], 'AS_SING', 'tab:blue', r'(a) AS$_{\rm SING}$ rises with $\rho_{\rm null}$'), (axs[1], 'AS_fair_A', 'tab:red', r'(b) AS$_{\rm fair}$ falls with $\rho_{\rm null}$ (over-correction)')]:
    for m in order:
        ax.scatter(rho[m], float(f2[m][key]), color=col, s=25, marker='x' if f2[m]['screen_exclude'] == 'True' else 'o')
        if m in ('densenet', 'dinovit1', 'eva02_large', 'bitresnet', 'deit', 'resnext'): ax.annotate(NICE[m], (rho[m], float(f2[m][key])), fontsize=7, xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log'); ax.set_xlabel(r'null energy $\rho_{\rm null}$'); ax.set_title(t, fontsize=10)
axs[0].set_yscale('log'); axs[0].set_ylabel(r'AS$_{\rm SING}$ (deg)'); axs[1].set_ylabel(r'AS$_{\rm fair}$'); axs[1].axhline(1, color='gray', ls=':', lw=.8)
save(fig, 'fig_e2_scatter')

# Fig 6 (new): normaliser sweep, tau vs SING with 2000x model bootstrap CIs
nr = rows(f'{R}/r2/results/fix3_normalisers.csv'); lab = {'N1_AS/rho^0.25': r'AS/$\rho^{.25}$', 'N2_AS/rho^0.43': r'AS/$\rho^{.43}$', 'N3_AS/rho^0.5': r'AS/$\rho^{.5}$',
      'N4_AS/rho': r'AS/$\rho$', 'N5_AS/d_null': 'AS/$d$', 'N6_AS_fair_A': 'fair A', 'N7_AS_fair_B': 'fair B', 'N8_AS_fair_C': 'fair C', 'AS_SING': r'AS$_{\rm SING}$', 'rho_alone': r'$\rho$ alone'}
keys = ['AS_SING', 'rho_alone', 'N1_AS/rho^0.25', 'N2_AS/rho^0.43', 'N3_AS/rho^0.5', 'N4_AS/rho', 'N5_AS/d_null', 'N6_AS_fair_A', 'N7_AS_fair_B', 'N8_AS_fair_C']
fig, ax = plt.subplots(figsize=(8, 2.8))
for j, (st, col) in enumerate([('all13', 'k'), ('screened', 'tab:orange')]):
    for i, k in enumerate(keys):
        r = [x for x in nr if x['set'] == st and x['score'] == k]
        if not r: continue
        r = r[0]; t, lo, hi = float(r['tau']), float(r['ci_lo']), float(r['ci_hi'])
        ax.errorbar(i + (j - .5) * .3, t, yerr=[[t - lo], [hi - t]], fmt='o', color=col, ms=4, capsize=2, label=('all 13' if st == 'all13' else 'screened (11)') if i == 0 else None)
ax.axhline(0, color='gray', lw=.6); ax.axhline(.3, color='gray', ls=':', lw=.8); ax.set_xticks(range(len(keys))); ax.set_xticklabels([lab[k] for k in keys], fontsize=8)
ax.set_ylabel(r'Kendall $\tau$ vs SING'); ax.set_ylim(-1.05, 1.05); ax.legend(frameon=False, fontsize=8, loc='lower left')
ax.set_title('Normaliser sweep (2000× model bootstrap, 95% CI)', fontsize=10); save(fig, 'fig_r2_normalisers')
print('figures written to', OUT)

# Fig (round 3): agreement with SING's order under energy controls and interventions (r3/results)
A3 = json.load(open(f'{R}/r3/results/A_equal_energy.json')); B3 = json.load(open(f'{R}/r3/results/B_leakfree.json')); G3 = json.load(open(f'{R}/r3/results/G_retrain.json'))
ent = [(r'$\rho$', A3['tau']['E12']['rho'], A3['tau']['S11']['rho']), (r'AS$_{\rm SING}$', A3['tau']['E12']['AS_SING'], A3['tau']['S11']['AS_SING']),
       (r'$\kappa$', A3['tau']['E12']['kappa'], A3['tau']['S11']['kappa']), ('null norm', A3['tau']['E12']['nusa'], A3['tau']['S11']['nusa']),
       (r'leak-free AS$_{\rm surr}$', None, B3['tau_AS_surr']['S11']), ('A1 equal energy', A3['tau']['E12']['A1'], A3['tau']['S11']['A1']),
       ('A2 top PCs', A3['tau']['E12']['A2'], A3['tau']['S11']['A2']), ('A3 random rot.', A3['tau']['E12']['A3'], A3['tau']['S11']['A3']),
       (r'A4 equal $\kappa$', A3['tau']['E12']['A4'], A3['tau']['S11']['A4']),
       ('AS, retrained T', G3['tau']['E12']['AS_SING_retrained'], G3['tau']['S11']['AS_SING_retrained']), ('A1, retrained T', G3['tau']['E12']['A1_retrained'], G3['tau']['S11']['A1_retrained'])]
fig, ax = plt.subplots(figsize=(8, 2.9))
for i, (lab_, e, s_) in enumerate(ent):
    for j, (d, col, nm) in enumerate([(e, 'k', 'E12 (12 models)'), (s_, 'tab:orange', 'S11 (screened)')]):
        if d is None: continue
        t = d['tau']; ax.errorbar(i + (j - .5) * .3, t, yerr=[[t - d['pct'][0]], [d['pct'][1] - t]], fmt='o', color=col, ms=4, capsize=2, label=nm if i == 0 else None)
ax.axhline(0, color='gray', lw=.6); ax.axhline(.3, color='gray', ls=':', lw=.8); ax.axhline(.6, color='gray', ls='--', lw=.6); ax.axvline(4.5, color='gray', lw=.5)
ax.set_xticks(range(len(ent))); ax.set_xticklabels([e[0] for e in ent], fontsize=7.5, rotation=25, ha='right')
ax.set_ylabel(r'Kendall $\tau$ vs SING'); ax.set_ylim(-1.05, 1.05); ax.legend(frameon=False, fontsize=8, loc='lower left')
save(fig, 'fig_r3_tau')
