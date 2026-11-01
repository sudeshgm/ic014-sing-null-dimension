"""Round-2 fix 8: signed effect sizes with 95% bootstrap CIs for SING's comparative claims (C1-C12), under AS_SING and
AS_fair-A. Positive effect = SING's direction. Bootstrap: 1000x, seed 0, resample images within class, the same indices
for every model. Per-image inputs from r2/e2_controls.py (--private dir). Lower score = less leakage.
Usage: python r2/e2_claims_ci.py --private $IC014_FEATS/r2_private"""
import argparse, os, json, csv, numpy as np
from scipy.stats import kendalltau, spearmanr
R = os.path.dirname(os.path.abspath(__file__))
FIG6B = {'densenet': 70.71, 'dinovit1': 70.16, 'eva02_large': 54.32, 'bitresnet': 47.21, 'convnext': 37.38, 'efficientnet': 36.28,
         'swin_large': 29.18, 'resnet': 24.81, 'vgg16': 24.26, 'vgg19': 19.89, 'dinov2': 16.61, 'resnext': 13.88, 'deit': 9.51}
FIVE = ['dinovit1', 'resnet', 'resnext', 'efficientnet', 'bitresnet']; TABLE1 = FIVE
CLS = {'jellyfish': 107, 'porcupine': 334, 'arabian_camel': 354, 'sports_car': 817}; ALL = list(FIG6B)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--private', required=True); ap.add_argument('--boot', type=int, default=1000); a = ap.parse_args()
    D = {m: dict(np.load(os.path.join(a.private, f'{m}_r2img.npz'))) for m in ALL}
    y = D['resnet']['y']; yc = D['resnet']['yc']
    assert all((D[m]['y'] == y).all() and (D[m]['yc'] == yc).all() for m in ALL)
    groups = [np.where(y == c)[0] for c in range(1000)]; cg = {c: np.where(yc == c)[0] for c in np.unique(yc)}
    def stats(ix, ixc):   # ix: indices into the 10k subset; ixc: dict class -> indices into the claim-class set
        S = {'AS_SING': {}, 'AS_fair': {}}; PC = {'AS_SING': {}, 'AS_fair': {}}
        yy = y[ix]
        for m in ALL:
            s, q = D[m]['AS'][ix], D[m]['randA'][ix]; S['AS_SING'][m] = s.mean(); S['AS_fair'][m] = s.sum() / q.sum()
            if m in ('resnet', 'dinovit1'):
                ss = np.bincount(yy, s, 1000); qq = np.bincount(yy, q, 1000); n = np.bincount(yy, minlength=1000).astype(float)
                for c, ii in ixc.items(): ss[c] = D[m]['ASc'][ii].sum(); qq[c] = D[m]['randAc'][ii].sum(); n[c] = len(ii)
                with np.errstate(invalid='ignore', divide='ignore'): PC['AS_SING'][m] = ss / n; PC['AS_fair'][m] = ss / qq
        E = {}
        for k in ['AS_SING', 'AS_fair']:
            s, pc = S[k], PC[k]; o = lambda pool, ex: [s[m] for m in pool if m != ex]
            e = dict(C1=min(o(FIVE, 'dinovit1')) - s['dinovit1'], C2=s['resnext'] - max(o(FIVE, 'resnext')),
                     C3=s['resnet'] - s['dinovit1'], C4=min(o(ALL, 'dinovit1')) - s['dinovit1'],
                     C5=s['resnext'] - sorted(o(FIVE, 'resnext'))[-2],
                     C6_mean=np.nanmean(pc['resnet']) - np.nanmean(pc['dinovit1']), C6_sd=np.nanstd(pc['resnet']) - np.nanstd(pc['dinovit1']),
                     C11=min(o(TABLE1, 'dinovit1')) - s['dinovit1'], C12=kendalltau([-s[m] for m in ALL], [FIG6B[m] for m in ALL]).statistic)
            v = pc['resnet']; vv = v[~np.isnan(v)]
            for c in ['porcupine', 'sports_car']: e[f'C8_{c}'] = 100 * np.mean(vv <= v[CLS[c]]) - 90
            e['C9_spearman'] = spearmanr(pc['resnet'], pc['dinovit1'], nan_policy='omit').statistic
            e['C10'] = pc['dinovit1'][CLS['jellyfish']] - pc['dinovit1'][CLS['arabian_camel']]
            if k == 'AS_SING': d = pc['dinovit1']; d = d[~np.isnan(d)]; e['C7'] = np.mean(d < 1) - 0.5
            E[k] = {kk: float(vv_) for kk, vv_ in e.items()}
        return E
    point = stats(np.arange(len(y)), cg); rng = np.random.default_rng(0); B = []
    for _ in range(a.boot):
        ix = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups if len(g)])
        ixc = {c: g[rng.integers(0, len(g), len(g))] for c, g in cg.items()}; B.append(stats(ix, ixc))
    out = {}
    def verdict(lo, hi, nocorr=False):
        if nocorr: return 'holds' if lo <= 0 <= hi else 'contradicted'
        return 'holds' if lo > 0 else ('reverses' if hi < 0 else 'not confirmed')
    for k in ['AS_SING', 'AS_fair']:
        out[k] = {}
        for e, v in point[k].items():
            arr = np.array([b[k][e] for b in B]); lo, hi = np.nanpercentile(arr, [2.5, 97.5])
            out[k][e] = dict(effect=v, ci95=[float(lo), float(hi)], verdict=verdict(lo, hi, e == 'C9_spearman'))
        c6 = [out[k]['C6_mean']['verdict'], out[k]['C6_sd']['verdict']]
        out[k]['C6'] = dict(verdict=c6[0] if c6[0] == c6[1] else 'not confirmed (mean and SD disagree)')
        c8 = [out[k]['C8_porcupine']['verdict'], out[k]['C8_sports_car']['verdict']]
        out[k]['C8'] = dict(verdict='reverses' if 'reverses' in c8 else ('holds' if all(x == 'holds' for x in c8) else 'not confirmed'),
                            flag='possible re-implementation discrepancy (fails under both scores in E2)')
        out[k]['C10']['flag'] = 'possible re-implementation discrepancy (fails under both scores in E2)'
    out['AS_fair']['C7'] = dict(verdict='N/A (AS_fair is unitless)')
    out['C13'] = 'untestable (no EVA02-space translators released)'; out['C14'] = 'untestable / not a comparative leakage claim'
    for k in ['AS_SING', 'AS_fair']:
        out[f'n_reverses_{k}'] = sorted(e for e, v in out[k].items() if isinstance(v, dict) and v.get('verdict') == 'reverses' and not e.startswith(('C6_', 'C8_')))
    json.dump(out, open(os.path.join(R, 'results/fix8_claims_ci.json'), 'w'), indent=1)
    with open(os.path.join(R, 'results/fix8_claims_ci.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['score', 'effect_id', 'effect', 'ci_lo', 'ci_hi', 'verdict'])
        for k in ['AS_SING', 'AS_fair']:
            for e, v in out[k].items():
                if 'effect' in v: w.writerow([k, e, v['effect'], *v['ci95'], v['verdict']])
                else: w.writerow([k, e, '', '', '', v['verdict']])
    for k in ['AS_SING', 'AS_fair']:
        print(k); [print(f"  {e:16s}", (f"{v['effect']:+.3f} [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] " if 'effect' in v else ''), v['verdict']) for e, v in out[k].items()]
        print('  reversals:', out[f'n_reverses_{k}'])

if __name__ == '__main__': main()
