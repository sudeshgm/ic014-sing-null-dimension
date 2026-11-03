"""Paper macros and tables for round 7, round 8 and the post hoc offset analysis, from committed aggregates only
(no private inputs). Reads r7/results/*.json, r8/results/*.json, r8/posthoc/*.json and r3/r4 committed tables via r3/common.
Writes paper_workshop/tables/numbers_r78.tex, tab_r8_models.tex, tab_r8_heads.tex and tab_prereg_r78.tex."""
import os, sys, json, numpy as np
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, 'r3'))
from common import load_models
OUT = os.path.join(R, 'paper_workshop', 'tables')
J = lambda *p: json.load(open(os.path.join(R, *p)))
mac = {}
f2 = lambda x: f'{x:.2f}'.replace('-', '{-}')
ci2 = lambda c: f'[{f2(c[0])},{f2(c[1])}]'
pct = lambda x: f'{100 * x:.0f}'

# ---------------- post hoc offset analysis (r8/posthoc) ----------------
T = J('r8', 'posthoc', 'offset_tau.json'); TS = J('r8', 'posthoc', 'offset_surr_tau.json')
P = J('r8', 'posthoc', 'offset_surr.json'); D = J('r8', 'posthoc', 'rho_decomp.json')
cen, unc = 'rho_centred = between+within', 'rho (uncentred, paper)'
for s, k in [('all13', 'A'), ('E12', 'E'), ('S11', 'S')]:
    t = T['tau'][s]
    mac[f'ZOTauRho{k}'] = f2(t[unc]['tau']); mac[f'ZOTauCen{k}'] = f2(t[cen]['tau']); mac[f'ZOTauCen{k}CI'] = ci2(t[cen]['pct'])
    mac[f'ZOTauOff{k}'] = f2(t['rho_offset']['tau']); mac[f'ZOTauBtw{k}'] = f2(t['rho_between']['tau']); mac[f'ZOTauWit{k}'] = f2(t['rho_within']['tau'])
    p = T['paired'][s][f'{cen} - {unc}']; mac[f'ZOPairCen{k}'] = f2(p['diff']); mac[f'ZOPairCen{k}CI'] = ci2(p['pct']); mac[f'ZOPairCen{k}BCa'] = ci2(p['bca'])
    u = TS['tau'][s]
    mac[f'ZOSurrCen{k}'] = f2(u['zm_c N(0,Sigma) @ centred rho']['tau']); mac[f'ZOSurrCen{k}CI'] = ci2(u['zm_c N(0,Sigma) @ centred rho']['pct'])
    mac[f'ZOPermCen{k}'] = f2(u['perm_c (C-mu permuted) @ centred rho']['tau'])
    mac[f'ZOSurrFull{k}'] = f2(u['AS_surr gauss N(mu,Sigma) @ full rho (paper)']['tau'])
    mac[f'ZOAsCen{k}'] = f2(u['AS real centred (offset removed)']['tau']); mac[f'ZOAsOff{k}'] = f2(u['AS offset only (natural size)']['tau'])
    mac[f'ZOTauCenUnc{k}'] = f2(T[f'tau_centred_vs_uncentred_{s}'])
pz = TS['paired']['all13']['zm_c N(0,Sigma) @ centred rho - AS_surr gauss N(mu,Sigma) @ full rho (paper)']
mac['ZOPairSurrA'] = f2(pz['diff']); mac['ZOPairSurrACI'] = ci2(pz['pct'])
ms, col, sing = load_models()
rho = np.array([D[m]['rho'] for m in ms]); rc = np.array([D[m]['rho_between'] + D[m]['rho_within'] for m in ms])
rk = lambda v, m: int(1 + np.sum(v > v[ms.index(m)]))  # rank 1 = most null energy (most leakage under the energy reading)
mac.update(ZORankDeitUnc=str(rk(rho, 'deit')), ZORankDeitCen=str(rk(rc, 'deit')), ZORankResUnc=str(rk(rho, 'resnet')), ZORankResCen=str(rk(rc, 'resnet')))
assert all(abs(D[m]['rho'] - D[m]['rho_table']) < 1e-6 for m in ms) and max(D[m]['identity_abs_err'] for m in ms) < 1e-12
so = {m: D[m]['share_offset'] for m in ms}
mac.update(ZORhoOffDeit=pct(so['deit']), ZORhoOffRes=pct(so['resnet']), ZORhoOffDense=pct(so['densenet']), ZORhoOffRnx=pct(so['resnext']),
           ZORhoOffOtherMax=pct(max(v for m, v in so.items() if m not in ('deit', 'resnet', 'densenet', 'resnext'))))
mac.update(ZODeitRho=f"{D['deit']['rho']:.3f}", ZODeitRhoC=f"{D['deit']['rho_between'] + D['deit']['rho_within']:.3f}",
           ZOResRho=f"{D['resnet']['rho']:.3f}", ZOResRhoC=f"{D['resnet']['rho_between'] + D['resnet']['rho_within']:.3f}")
sh = {m: P[m]['off_nat'] / P[m]['AS'] for m in ms}
mac.update(ZOShareDeit=pct(sh['deit']), ZOShareRes=pct(sh['resnet']), ZOShareEva=pct(sh['eva02_large']), ZOShareDense=pct(sh['densenet']),
           ZOShareRnx=pct(sh['resnext']))
oth = [sh[m] for m in ms if m not in ('deit', 'resnet', 'eva02_large', 'densenet')]
mac.update(ZOShareOtherLo=pct(min(oth)), ZOShareOtherHi=pct(max(oth)))
for m, k in [('deit', 'Deit'), ('resnet', 'Res')]:
    mac[f'ZOAS{k}'] = f"{P[m]['AS']:.2f}"; mac[f'ZOASOff{k}'] = f"{P[m]['off_nat']:.2f}"; mac[f'ZOASzm{k}'] = f"{P[m]['zm_c']:.2f}"
    mac[f'ZOASreal{k}'] = f"{P[m]['real_c']:.2f}"

# ---------------- round 7 ----------------
M7 = ['resnet', 'efficientnet', 'resnext', 'dinov2']
r7 = {m: J('r7', 'results', f'{m}.json') for m in M7}; s7 = {m: J('r7', 'results', f'{m}_sens.json') for m in M7}
mac['ZSNH'] = str(sum(r7[m]['verdicts']['H1'] == 'REPRODUCIBLE' for m in M7))
mac['ZSNArt'] = str(sum(r7[m]['verdicts']['by_stat']['T2']['H2'] == 'ARTEFACT' for m in M7))
mac['ZSNArtCorr'] = str(sum(s7[m]['H2_T2_corrected'] == 'ARTEFACT' for m in M7))
mac['ZSNSupp'] = str(sum(bool(r7[m]['verdicts']['supports_cancellation']) for m in M7))
rel = [r7[m]['levels']['2']['edit']['reliability'] for m in M7]; mac.update(ZSRelLo=f2(min(rel)), ZSRelHi=f2(max(rel)))
te = [r7[m]['levels']['2']['edit']['T2'] for m in M7]; ta = [r7[m]['levels']['2']['artefact']['T2'] for m in M7]
tr = [r7[m]['row']['2']['T2'] for m in M7]
f4 = lambda x: f'{x:.4f}'
mac.update(ZSTeLo=f4(min(te)), ZSTeHi=f4(max(te)), ZSTaLo=f4(min(ta)), ZSTaHi=f4(max(ta)), ZSTrLo=f4(min(tr)), ZSTrHi=f4(max(tr)))
pmax = max(r7[m]['levels'][L]['edit']['perm']['p'] for m in M7 for L in '123'); mac['ZSPermP'] = f'{pmax:.3f}'
assert all(s7[m]['levels'][L]['dT2_art']['ci'][0] < 0 < s7[m]['levels'][L]['dT2_art']['ci'][1] for m in M7 for L in '123')
assert all(r7[m]['levels'][L]['boot']['dT2_artefact_ci'][0] < 0 < r7[m]['levels'][L]['boot']['dT2_artefact_ci'][1] for m in M7 for L in '123')
b = r7['resnet']['levels']['2']; mac.update(ZSBootPt=f4(b['edit']['T2']), ZSBootLo=f'{b["boot"]["T2_ci"][0]:.3f}', ZSBootHi=f'{b["boot"]["T2_ci"][1]:.3f}')
drop = [1 - r7[m]['levels']['3']['edit']['tr'] / r7[m]['levels']['1']['edit']['tr'] for m in M7]
mac.update(ZSDropLo=pct(min(drop)), ZSDropHi=pct(max(drop)))
mac['ZSSeed'] = str(s7['resnet']['seed']); mac['ZSNrep'] = str(s7['resnet']['nrep'])

# ---------------- round 8 ----------------
V = J('r8', 'results', 'R8_verdicts.json'); S8 = J('r8', 'results', 'R8_summary.json'); K0 = J('r8', 'results', 'K0_coverage.json')
pm = {m: J('r8', 'results', 'permodel', f'{m}.json') for m in ms}
mac['ZENGatePass'] = str(sum(bool(v) for v in S8['gates'].values())); mac['ZENGate'] = str(len(S8['gates']))
assert S8['gates']['G_noise'] is False and S8['gates']['G_derange'] is False and not S8['all_gates_passed']
for s, k in [('E12', 'E'), ('S11', 'S')]:
    c = V['confirmatory'][s]['tau']; mac[f'ZETau{k}'] = f2(c['tau']); mac[f'ZETau{k}CI'] = ci2(c['pct']); mac[f'ZETau{k}BCa'] = ci2(c['bca'])
    d = V['descriptive'][s]; t = d['tau_SING']
    for q, n in [('theta_cls', 'Cls'), ('shadow', 'Shad'), ('innovation', 'Inn'), ('J_innovation', 'Jinn'), ('self_duality', 'Self')]:
        mac[f'ZE{n}{k}'] = f2(t[q]['tau']); mac[f'ZE{n}{k}CI'] = ci2(t[q]['pct'])
    mac[f'ZETauRho{k}'] = f2(d['tau_theta_rho']); mac[f'ZETauBtw{k}'] = f2(d['tau_theta_rho_between'])
    pr = d['partial_theta_SING_given_contentfree']; mac[f'ZEPart{k}'] = f2(pr['r']); mac[f'ZEPart{k}CI'] = ci2(pr['pct'])
    mac[f'ZEFoldTwo{k}'] = f2(t['theta_all_models_fold2']['tau'])
gn = V['G_noise']['per_base']; bases = ['resnet', 'dinov2', 'resnext']; BN = dict(resnet='Res', dinov2='Dino', resnext='Rnx')
for bb in bases:
    mac[f'ZENoiseN{BN[bb]}'] = str(gn[bb]['n_estimable']); mac[f'ZENoiseMM{BN[bb]}'] = f2(gn[bb]['max_over_min'])
    sl = gn[bb]['slope']; mac[f'ZENoiseSl{BN[bb]}'] = f2(sl['slope']); mac[f'ZENoiseSl{BN[bb]}CI'] = '[' + ', '.join(f'{x:.3f}'.replace('-', '{-}') for x in sl['ci']) + ']'
    mac[f'ZEPlant{BN[bb]}'] = f2(V['G_S']['per_base'][bb]['semantic']['delta'])
dr = V['G_derange']['cells']['resnet/real']
mac.update(ZEDerMean=f"{dr['mean'] * 1e4:.1f}", ZEDerLo=f"{dr['ci'][0] * 1e5:.1f}", ZEDerHi=f"{dr['ci'][1] * 1e4:.1f}",
           ZEDerPass=str(sum(c['passed'] for c in V['G_derange']['cells'].values())),
           ZEDerN=str(len(V['G_derange']['cells'])))
mac.update(ZEFloorMax=f"{V['G_floor']['max_abs_theta']:.4f}", ZEFloorBar=f"{V['G_floor']['bar']:.3f}", ZEFloorN=str(V['G_floor']['n_cells']),
           ZEThetaMax=f"{V['max_13']:.2f}", ZEInvErr='4{\\times}10^{-16}' if V['G_invariance']['max_rel_err_13'] < 5e-16 else 'X')
cp = [r['theta_sem'] for bb in bases for r in pm[bb]['base']["C'"]]; mac.update(ZECpLo=f'{min(cp):.4f}', ZECpHi=f'{max(cp):.4f}')
def ratio(rows):
    v = [r['theta_sem'] for r in rows]; return v[-1] / v[0]
mr = [ratio(pm[bb]['base']['M']) for bb in bases]; dd = [max(r['theta_sem'] for r in pm[bb]['base']['D']) / min(r['theta_sem'] for r in pm[bb]['base']['D']) for bb in bases]
mac.update(ZEMLo=f'{min(mr):.0f}', ZEMHi=f'{max(mr):.0f}', ZEDLo=f'{min(dd):.0f}', ZEDHi=f'{max(dd):.0f}')
oc = {bb: [r['theta_sem'] for r in pm[bb]['base']['C']] for bb in bases}
mac.update(ZEOldCDino=f"{oc['dinov2'][-1] / oc['dinov2'][0]:.1f}", ZEOldCRnx=f"{oc['resnext'][-1] / oc['resnext'][0]:.1f}")
mac.update(ZEKBootSig=f2(K0['res']['signal']['coverage']['boot']), ZEKBootZero=f2(K0['res']['zero']['coverage']['boot']))
mods = {r['model']: r for r in S8['models']}
mac.update(ZECompDeit=f2(mods['deit']['composition_SZ_over_EbZ']), ZECompRnx=f2(mods['resnext']['composition_SZ_over_EbZ']), ZECompDino=f2(mods['dinov2']['composition_SZ_over_EbZ']),
           ZEThDeit=f"{mods['deit']['theta_sem']:.2f}", ZEBtwDeit=f"{mods['deit']['rho_between']:.3f}")
assert max(mods, key=lambda m: mods[m]['composition_SZ_over_EbZ']) == 'deit'
mac['ZECompRankDeit'] = '1'
hs = S8['head_swap']
for bb in bases:
    for h in ['released', 'NCM', 'LDA']:
        mac[f'ZEH{BN[bb]}{h}Rho'] = f"{hs[bb][h]['rho']:.3f}"; mac[f'ZEH{BN[bb]}{h}Th'] = f"{hs[bb][h]['theta_sem']:.4f}" if hs[bb][h]['theta_sem'] < 0.01 else f"{hs[bb][h]['theta_sem']:.3f}"
open(os.path.join(OUT, 'numbers_r78.tex'), 'w').write('% generated by r8/make_tables_r8.py -- do not edit\n' + ''.join(f'\\newcommand{{\\{k}}}{{{v}}}\n' for k, v in sorted(mac.items()) if v != ''))

# per-model table (round 8, descriptive)
NAME = dict(vgg16='VGG16', vgg19='VGG19', densenet='DenseNet121', resnet='ResNet50', dinovit1='DinoViT', efficientnet='EfficientNet', bitresnet='BiT-R50',
            resnext='ResNeXt101', convnext='ConvNeXt', swin_large='Swin-L', dinov2='DINOv2-L', deit='DeiT3-L', eva02_large='EVA02-L')
def sci(x):
    if x != 0 and abs(x) < 1e-3:
        m, e = f'{x:.1e}'.split('e'); return f'${m}{{\\times}}10^{{{int(e)}}}$'.replace('-', '{-}', 1) if m.startswith('-') else f'${m}{{\\times}}10^{{{int(e)}}}$'
    return f'{x:.3f}'.replace('-', '$-$')
order = sorted(ms, key=lambda m: S8['models'][[r['model'] for r in S8['models']].index(m)]['sing_lk'], reverse=True)
L = ['% generated by r8/make_tables_r8.py -- do not edit', '\\begin{tabular}{@{}lrrrrrrrr@{}}', '\\toprule',
     'Model & $\\theta_{\\mathrm{sem}}$ & $\\theta_{\\mathrm{cls}}$ & comp. & $\\rho$ & $\\rho_{\\mathrm{off}}$ & shadow & innov. & self-dual. \\\\', '\\midrule']
for m in order:
    r = mods[m]
    L.append(f"{NAME[m]} & {sci(r['theta_sem'])} & {sci(r['theta_cls'])} & {r['composition_SZ_over_EbZ']:.2f} & {sci(r['rho'])} & {sci(r['rho_offset'])} & {sci(r['shadow'])} & {sci(r['innovation'])} & {r['self_duality_full']:.2f} \\\\")
L += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(OUT, 'tab_r8_models.tex'), 'w').write('\n'.join(L) + '\n')
L = ['% generated by r8/make_tables_r8.py -- do not edit', '\\begin{tabular}{@{}llrrrr@{}}', '\\toprule', 'Base & Head & $\\rho$ & AS ($^\\circ$) & top-1 & $\\theta_{\\mathrm{sem}}$ \\\\', '\\midrule']
for bb in bases:
    for i, h in enumerate(['released', 'NCM', 'LDA']):
        x = hs[bb][h]; t1 = '--' if x['top1_halfB'] is None else f"{x['top1_halfB']:.3f}"
        L.append(f"{NAME[bb] if i == 0 else ''} & {h} & {x['rho']:.3f} & {x['AS']:.2f} & {t1} & {sci(x['theta_sem'])} \\\\")
    if bb != bases[-1]: L.append('\\addlinespace')
L += ['\\bottomrule', '\\end{tabular}']
open(os.path.join(OUT, 'tab_r8_heads.tex'), 'w').write('\n'.join(L) + '\n')

# ledger rows, rounds 7 and 8
g8 = S8['gates']; ok = lambda b: 'pass' if b else '\\textbf{fail}'
rows = [
 ('R7 H1, reproducible edit', 'REPRODUCIBLE if CI lower bound of split-half reliability (null, L2) $>0.10$', f"REPRODUCIBLE, {mac['ZSNH']}/4 (Rel {mac['ZSRelLo']}--{mac['ZSRelHi']})"),
 ('R7 H2, alignment above controls', 'ALIGNED if $p<0.01$ and the CIs of T2(edit)$-$T2(artefact) and T2(edit)$-$nuisance exclude 0 from above; ARTEFACT if $p<0.01$ but the artefact CI includes 0', f"ARTEFACT, {mac['ZSNArt']}/4 (also with corrected CIs)"),
 ('R7 H3, noise cancellation', 'tested only if H2 ALIGNED', 'NOT TESTED, 4/4'),
 ('R7 family', 'SUPPORTED if $\\ge$3 of 4 models support cancellation; NOT if $\\le$1', f"NOT SUPPORTED ({mac['ZSNSupp']}/4)"),
 ('R7 K0, K1', 'synthetic stability; no reproducible edit', 'did not fire'),
 ('R8 K0', 'CI method with synthetic coverage in [0.90, 0.99]', 'did not fire; class bootstrap'),
 ('R8 G-floor', 'CI includes 0 in $\\ge$90\\% of 98 surrogate cells and max $|\\theta|\\le0.1\\times$ max real', ok(g8['G_floor'])),
 ('R8 G-noise (world C$^\\prime$)', '$\\ge$8/12 rows estimable, max/min $\\le1.5$, slope CI includes 0, each base', ok(g8['G_noise']) + f" ({mac['ZENoiseNRes']}, {mac['ZENoiseNDino']}, {mac['ZENoiseNRnx']} of 12 estimable)"),
 ('R8 invariances, offset world', 'relative change $\\le10^{-8}$', ok(g8['G_invariance'])),
 ('R8 G-derange', 'CI includes 0 and $|$mean$|\\le0.1\\times$ max real, 6 cells', ok(g8['G_derange']) + f" ({mac['ZEDerPass']}/{mac['ZEDerN']} cells)"),
 ('R8 G-S plants', '$\\Delta_{\\mathrm{sem}}$ CI $>0$ and $\\Delta_{\\mathrm{arb}}$ CI includes 0, each base', ok(g8['G_S'])),
 ('R8 G-B, G-tol, estimability', 'closed form $\\le10^{-8}$; $\\tau\\ge0.8$ across tolerances; $\\ge$11/13 estimable', ', '.join(ok(g8[k]) for k in ['G_B', 'G_tol', 'estimability'])),
 ('R8 confirmatory', 'read only if all gates pass; TRACKS if $\\tau\\ge0.6$ and CI $>0$ on E12 and S11', f"NOT VALIDATED; $\\tau$ {mac['ZETauE']} / {mac['ZETauS']} reported, not read"),
]
L = ['% generated by r8/make_tables_r8.py -- do not edit', '\\begin{tabular}{@{}p{0.22\\linewidth}p{0.47\\linewidth}p{0.25\\linewidth}@{}}', '\\toprule', 'Test & Criterion & Outcome \\\\', '\\midrule']
L += [f'{a} & {b} & {c} \\\\' for a, b, c in rows] + ['\\bottomrule', '\\end{tabular}']
open(os.path.join(OUT, 'tab_prereg_r78.tex'), 'w').write('\n'.join(L) + '\n')
print('wrote numbers_r78.tex', len(mac), 'macros')
