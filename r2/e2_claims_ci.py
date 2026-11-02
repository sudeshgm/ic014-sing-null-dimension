"""Round-2 fix 8: signed effect sizes with 95% bootstrap CIs for SING's comparative claims (C1-C12), under AS_SING and
AS_fair-A. Positive effect = SING's direction. Bootstrap: 1000x, seed 0, resample images within class, the same indices
for every model. Per-image inputs from r2/e2_controls.py (--private dir). Lower score = less leakage.
Round 4 (PREREG_R4.md C6): the C6-SD effect (difference of class-SDs of per-class mean AS) gets its CI from a class-resampling
bootstrap (1000x, seed 0, the 1000 per-class values resampled with replacement) with a delete-one-class jackknife for BCa; a
variance-corrected SD (sqrt(max(SD^2 - mean within-class SE^2, 0))) is reported as a sensitivity value. Other effects unchanged.
Round 3 (PREREG_R3.md E; N6, N7): CIs are BCa (acceleration from a delete-one-image jackknife over the 10k subset images
and the 200 claim-class images); percentile CIs are kept in extra columns. Every point estimate is checked against its
BCa CI and violations are flagged. Duplicate rows are merged: C1/C11 (Table-1 pool = Fig.-3 pool) and C3/C6-mean (one
quantity; C6 keeps only the class-SD part). If C1 and C4 coincide numerically (same minimising model) they are counted once.
Usage: python r2/e2_claims_ci.py [--private _private]"""
import argparse, os, json, csv, numpy as np
from scipy.stats import kendalltau, spearmanr, norm
R = os.path.dirname(os.path.abspath(__file__))
FIG6B = {'densenet': 70.71, 'dinovit1': 70.16, 'eva02_large': 54.32, 'bitresnet': 47.21, 'convnext': 37.38, 'efficientnet': 36.28,
         'swin_large': 29.18, 'resnet': 24.81, 'vgg16': 24.26, 'vgg19': 19.89, 'dinov2': 16.61, 'resnext': 13.88, 'deit': 9.51}
FIVE = ['dinovit1', 'resnet', 'resnext', 'efficientnet', 'bitresnet']; TABLE1 = FIVE
CLS = {'jellyfish': 107, 'porcupine': 334, 'arabian_camel': 354, 'sports_car': 817}; ALL = list(FIG6B)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--private', default=os.environ.get('IC014_PRIVATE', os.path.join(R, '..', '_private'))); ap.add_argument('--boot', type=int, default=1000); a = ap.parse_args()
    D = {m: dict(np.load(os.path.join(a.private, f'{m}_r2img.npz'))) for m in ALL}
    y = D['resnet']['y']; yc = D['resnet']['yc']
    assert all((D[m]['y'] == y).all() and (D[m]['yc'] == yc).all() for m in ALL)
    groups = [np.where(y == c)[0] for c in range(1000)]; cg = {c: np.where(yc == c)[0] for c in np.unique(yc)}
    def stats(ix, ixc, ret_pc=False):   # ix: indices into the 10k subset; ixc: dict class -> indices into the claim-class set
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
        return (E, PC) if ret_pc else E
    point, PC0 = stats(np.arange(len(y)), cg, ret_pc=True); rng = np.random.default_rng(0); B = []
    for _ in range(a.boot):
        ix = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups if len(g)])
        ixc = {c: g[rng.integers(0, len(g), len(g))] for c, g in cg.items()}; B.append(stats(ix, ixc))
    # delete-one-image jackknife for the BCa acceleration
    full = np.arange(len(y)); J = []
    for i in range(len(y)): J.append(stats(np.delete(full, i), cg))
    for c, g in cg.items():
        for t in range(len(g)): J.append(stats(full, {**cg, c: np.delete(g, t)}))
    def bca(theta, arr, jack):
        arr = arr[np.isfinite(arr)]; jack = jack[np.isfinite(jack)]
        if np.ptp(arr) == 0: return [float(arr[0]), float(arr[0])]
        p = np.mean(arr < theta) + 0.5 * np.mean(arr == theta); p = min(max(p, 1 / (len(arr) + 1)), 1 - 1 / (len(arr) + 1))
        z0 = norm.ppf(p); jm = jack.mean(); den = 6 * np.sum((jm - jack) ** 2) ** 1.5; acc = np.sum((jm - jack) ** 3) / den if den > 0 else 0.0
        q = [norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z))) for z in (norm.ppf(.025), norm.ppf(.975))]
        return [float(np.percentile(arr, 100 * q[0])), float(np.percentile(arr, 100 * q[1]))]
    # PREREG_R4 C6: class-resampling bootstrap and delete-one-class jackknife for C6_sd
    sdd = lambda k, j: float(np.nanstd(PC0[k]['resnet'][j]) - np.nanstd(PC0[k]['dinovit1'][j]))
    rc = np.random.default_rng(0); allc = np.arange(1000); C6B = {k: np.array([sdd(k, rc.integers(0, 1000, 1000)) for _ in range(a.boot)]) for k in PC0}
    C6J = {k: np.array([sdd(k, np.delete(allc, c)) for c in allc]) for k in PC0}
    def within_se2(m, k):   # per-class squared SE of the class value (mean AS, or ratio sum AS / sum rand by the delta method)
        se2 = np.full(1000, np.nan)
        for c in range(1000):
            if c in cg: s_, q_ = D[m]['ASc'][cg[c]], D[m]['randAc'][cg[c]]
            else: g_ = groups[c]; s_, q_ = D[m]['AS'][g_], D[m]['randA'][g_]
            n_ = len(s_)
            if n_ < 2: continue
            if k == 'AS_SING': se2[c] = s_.var(ddof=1) / n_
            else: r_ = s_.sum() / q_.sum(); se2[c] = (s_ - r_ * q_).var(ddof=1) / n_ / q_.mean() ** 2
        return se2
    C6VC = {}
    for k in PC0:
        vc = {}
        for m in ('resnet', 'dinovit1'):
            v_ = PC0[k][m]; vc[m] = float(np.sqrt(max(np.nanvar(v_) - np.nanmean(within_se2(m, k)), 0.0)))
        C6VC[k] = dict(raw_sd={m: float(np.nanstd(PC0[k][m])) for m in ('resnet', 'dinovit1')}, corrected_sd=vc, corrected_effect=vc['resnet'] - vc['dinovit1'])
    out = {}; flags = []
    def verdict(lo, hi, nocorr=False):
        if nocorr: return 'holds' if lo <= 0 <= hi else 'contradicted'
        return 'holds' if lo > 0 else ('reverses' if hi < 0 else 'not confirmed')
    MERGE = {'C1': 'C1/C11', 'C3': 'C3/C6_mean'}; DROP = {'C11', 'C6_mean'}
    for k in ['AS_SING', 'AS_fair']:
        out[k] = {}
        for e, v in point[k].items():
            if e in DROP: continue
            arr = np.array([b[k][e] for b in B]); jk = np.array([j[k][e] for j in J])
            if e == 'C6_sd': arr, jk = C6B[k], C6J[k]
            plo, phi = np.nanpercentile(arr, [2.5, 97.5]); lo, hi = bca(v, arr, jk)
            inside = bool(lo - 1e-12 <= v <= hi + 1e-12)
            if not inside: flags.append(f'{k}:{e} point {v:.4g} outside BCa [{lo:.4g},{hi:.4g}]')
            out[k][MERGE.get(e, e)] = dict(effect=v, ci95=[lo, hi], ci95_percentile=[float(plo), float(phi)], point_in_ci=inside,
                                           verdict=verdict(lo, hi, e == 'C9_spearman'))
        out[k]['C6_sd']['ci_method'] = 'class-resampling bootstrap (1000 classes), BCa from delete-one-class jackknife'
        out[k]['C6_sd']['variance_corrected'] = C6VC[k]
        out[k]['C6'] = dict(verdict=out[k]['C6_sd']['verdict'], note='C6 = class-SD part only; the class-mean part is C3; CI resamples classes')
        c8 = [out[k]['C8_porcupine']['verdict'], out[k]['C8_sports_car']['verdict']]
        out[k]['C8'] = dict(verdict='reverses' if 'reverses' in c8 else ('holds' if all(x == 'holds' for x in c8) else 'not confirmed'),
                            flag='possible re-implementation discrepancy (fails under both scores in E2)')
        out[k]['C10']['flag'] = 'possible re-implementation discrepancy (fails under both scores in E2)'
        if abs(point[k]['C1'] - point[k]['C4']) < 1e-9:
            out[k]['C4']['footnote'] = 'numerically identical to C1/C11 (the same model attains both minima); counted once'
    out['AS_fair']['C7'] = dict(verdict='N/A (AS_fair is unitless)')
    out['C13'] = 'untestable (no EVA02-space translators released)'; out['C14'] = 'untestable / not a comparative leakage claim'
    for k in ['AS_SING', 'AS_fair']:
        rows = {e: v for e, v in out[k].items() if isinstance(v, dict) and 'verdict' in v and not e.startswith(('C6_', 'C8_'))
                and 'footnote' not in v}
        out[f'tally_{k}'] = {vd: sorted(e for e, v in rows.items() if v['verdict'] == vd) for vd in sorted({v['verdict'] for v in rows.values()})}
        out[f'n_reverses_{k}'] = out[f'tally_{k}'].get('reverses', [])
    out['N7_flags'] = flags; out['N7_pass'] = not flags
    out['C6_R4_pass'] = bool(not flags and all(out[k]['C6_sd']['ci95'][0] <= out[k]['C6_sd']['effect'] <= out[k]['C6_sd']['ci95'][1] and
                                               out[k]['C6_sd']['ci95_percentile'][0] <= out[k]['C6_sd']['effect'] <= out[k]['C6_sd']['ci95_percentile'][1] for k in ('AS_SING', 'AS_fair'))); out['jackknife_n'] = len(J)
    json.dump(out, open(os.path.join(R, 'results/fix8_claims_ci.json'), 'w'), indent=1)
    with open(os.path.join(R, 'results/fix8_claims_ci.csv'), 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['score', 'effect_id', 'effect', 'ci_lo_bca', 'ci_hi_bca', 'ci_lo_pct', 'ci_hi_pct', 'verdict', 'note'])
        for k in ['AS_SING', 'AS_fair']:
            for e, v in out[k].items():
                note = v.get('footnote', v.get('note', ''))
                if 'effect' in v: w.writerow([k, e, v['effect'], *v['ci95'], *v['ci95_percentile'], v['verdict'], note])
                else: w.writerow([k, e, '', '', '', '', '', v['verdict'], note])
    for k in ['AS_SING', 'AS_fair']:
        print(k); [print(f"  {e:16s}", (f"{v['effect']:+.3f} BCa[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] " if 'effect' in v else ''), v['verdict'], v.get('footnote', '')) for e, v in out[k].items()]
        print('  tally:', out[f'tally_{k}'])
    print('N7 flags:', flags)

if __name__ == '__main__': main()
