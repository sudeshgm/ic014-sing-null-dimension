"""Post hoc (after PREREG_R6; see r6/POSTHOC_R6.md). From committed aggregates only (CPU, seconds):
(1) positive-control rows: SING's AS and IS (released and retrained translators, r6/results/posthoc/controls_*.json) and rho itself, through
    the unchanged statistics and verdict rule of verdicts_r6.score_block;
(2) ViM with alpha frozen at its real-data value (E1, E2);
(3) a magnitude statistic, not pre-registered: the share of the cross-model range left after E1, range(E1) / range(real), on the
    estimable models of E12 (and S11);
(4) the size of tau_EQ under independence (the tau part of the ENERGY-DRIVEN bar);
(5) V-information sensitivity: range and estimability without DenseNet121, non-positive gains, C2 vs C1, penalty choices;
(6) the synthetic check of r6/tests/test_posthoc_r6.py: a within-class permutation removing real image-level conditional information.
Writes r6/results/R6_posthoc.json."""
import os, sys, json, glob, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H); sys.path.insert(0, H)
import verdicts_r6 as vr  # noqa: E402
from verdicts_r6 import load_models, sets, tau, RES6  # noqa: E402
PH = os.path.join(RES6, 'posthoc')


def mag(real, e1, sub):
    return float(np.ptp([e1[n] for n in sub]) / np.ptp([real[n] for n in sub]))


def energy_only(real, e1, e2, sub_sets, est):
    """tau_EQ and R_B exactly as verdicts_r6.score_block, for a score given as dicts (no bootstrap / content tests needed)"""
    rng_ = float(np.ptp([real[n] for n in real])); t = {s: tau([e1[n] for n in sub_sets[s] if n in est], [real[n] for n in sub_sets[s] if n in est]) for s in ('E12', 'S11')}
    rb = {n: float(np.ptp(e2[n])) / rng_ for n in real}; med = float(np.median([rb[n] for n in est]))
    v = ('ENERGY-DRIVEN' if all(np.isfinite(x) and x < 0.3 for x in t.values()) and med >= 0.5 else
         'NOT ENERGY-DRIVEN' if all(np.isfinite(x) and x >= 0.6 for x in t.values()) and med <= 0.1 else 'MIXED')
    return dict(tau_EQ=t, R_B_median=med, energy_verdict=v, range=rng_)


def main():
    ms, col, sing = load_models(); lk = -sing; rho_all = col('rho'); S = sets(ms, rho_all)
    SUM = json.load(open(os.path.join(RES6, 'R6_summary.json')))['scores']
    P6 = {n: json.load(open(os.path.join(RES6, 'models', f'{n}.json'))) for n in ms}
    CT = {n: json.load(open(os.path.join(PH, f'controls_{n}.json'))) for n in ms}
    out = dict(controls={}, magnitude={}, vim_frozen={}, vinfo={}, tau_null={})
    # (1) SING's scores through the unchanged rule
    for tr in ('released', 'retrained'):
        P = {n: CT[n][tr] for n in ms}
        for k in ('AS', 'IS'):
            b = vr.score_block(k, P, S, ms, rho_all, lk)
            out['controls'][f'{k}_{tr}'] = {q: b[q] for q in ('real', 'sd_boot', 'range13', 'estimable', 'n_E12', 'n_S11', 'E1', 'tau_EQ', 'R_B', 'R_B_median', 'energy_verdict',
                                                               'content_verdict', 'counts', 'C2_noisy', 'cross_model')}
            out['controls'][f'{k}_{tr}']['R_B_median_E12'] = float(np.median([b['R_B'][n] for n in S['E12']]))
    # rho itself: after E1 every model has rho = 0.005, so tau_EQ is undefined (no order left) and the conjunctive rule returns MIXED
    real = {n: float(rho_all[ms.index(n)]) for n in ms}; e1 = {n: P6[n]['E1']['rho'] for n in ms}; e2 = {n: [r['rho'] for r in P6[n]['E2']] for n in ms}
    out['controls']['rho'] = energy_only(real, e1, e2, S, ms); out['controls']['rho']['E1_values'] = e1
    # (2) ViM, alpha frozen
    realv = {n: CT[n]['vim_alpha']['real'] for n in ms}; e1f = {n: CT[n]['vim_alpha']['E1_frozen'] for n in ms}; e2f = {n: CT[n]['vim_alpha']['E2_frozen'] for n in ms}
    est = SUM['ViM']['estimable']; vf = energy_only(realv, e1f, e2f, S, est)
    vf.update(real=realv, E1_frozen=e1f, E1_refit={n: CT[n]['vim_alpha']['E1_refit'] for n in ms}, alpha_ratio_E1={n: float(np.mean([CT[n]['vim_alpha']['alpha_E1'][h] / CT[n]['vim_alpha']['alpha_real'][h] for h in 'AB'])) for n in ms},
              max_abs_real_vs_audit=float(max(abs(realv[n] - P6[n]['real']['ViM']) for n in ms)), max_abs_E1refit_vs_audit=float(max(abs(CT[n]['vim_alpha']['E1_refit'] - P6[n]['E1']['ViM']) for n in ms)))
    out['vim_frozen'] = vf
    # (3) magnitude: share of cross-model range left after E1
    rows = {k: (SUM[k]['real'], SUM[k]['E1'], SUM[k]['estimable']) for k in SUM}
    for k in ('AS_released', 'AS_retrained', 'IS_released', 'IS_retrained'):
        c = out['controls'][k]; rows[k] = (c['real'], c['E1'], c['estimable'])
    rows['ViM_frozen'] = (realv, e1f, est)
    for k, (re, e, es) in rows.items():
        out['magnitude'][k] = {s: mag(re, e, [n for n in S[s] if n in es]) for s in ('E12', 'S11')}
    # (4) tau_EQ < 0.3 under independence (20000 random permutations, seed 0)
    rs = np.random.default_rng(0)
    for n in (12, 11):
        t = np.array([tau(rs.permutation(n).astype(float), np.arange(n, dtype=float)) for _ in range(20000)])
        out['tau_null'][str(n)] = dict(p_lt_03=float(np.mean(t < 0.3)), p_ge_06=float(np.mean(t >= 0.6)), draws=20000)
    # (5) V-information
    b = SUM['Vinfo']; rv = b['real']; sd = b['sd_boot']; no_d = [n for n in ms if n != 'densenet']; r12 = float(np.ptp([rv[n] for n in no_d]))
    est12 = [n for n in no_d if sd[n] <= 0.1 * r12]; rb12 = {n: float(np.ptp([r['Vinfo'] for r in P6[n]['E2']])) / r12 for n in no_d}
    t12 = {s: tau([b['E1'][n] for n in S[s] if n in est12], [rv[n] for n in S[s] if n in est12]) for s in ('E12', 'S11')}
    lam = {n: {h: P6[n]['vinfo_cal'][f'label_{h}']['lam_rel'] for h in 'AB'} for n in ms}
    lam_abs = {n: float(np.mean([P6[n]['real'][f'lam_label_{h}'] for h in 'AB'])) for n in ms}
    out['vinfo'] = dict(real=rv, range13=b['range13'], range_without_densenet=r12, densenet_share_of_range=float(rv['densenet'] / b['range13']),
                        estimable_without_densenet=est12, n_E12_without_densenet=sum(n in est12 for n in S['E12']), n_S11_without_densenet=sum(n in est12 for n in S['S11']),
                        verdict_possible_without_densenet=bool(sum(n in est12 for n in S['E12']) >= 9 and sum(n in est12 for n in S['S11']) >= 8),
                        R_B_median_without_densenet_all12=float(np.median(list(rb12.values()))), R_B_median_without_densenet_est=float(np.median([rb12[n] for n in est12])) if est12 else float('nan'),
                        tau_EQ_without_densenet_est=t12, nonpositive=[n for n in ms if rv[n] <= 0], below_1e4=[n for n in ms if rv[n] < 1e-4],
                        within_2sd_of_zero=[n for n in ms if abs(rv[n]) <= 2 * sd[n]], C2_gt_C1=[n for n in ms if abs(b['C2'][n]) > abs(b['C1'][n])],
                        lam_rel=lam, lam_abs_mean=lam_abs, lam_top=sum(v >= 1.0 for d in lam.values() for v in d.values()), lam_n=2 * len(ms))
    # (6) synthetic: within-class permutation vs real conditional information (cheap, seconds)
    sys.path.insert(0, os.path.join(H, 'tests')); import test_posthoc_r6 as tp; import scores_r6 as sc; old = sc.NCLS
    try:
        Cn, L, y, C = tp._coupled(); sc.NCLS = C; fit = np.arange(len(y)) % 2 == 0
        gain = lambda Cm: float(sc.vinfo_fit_eval(Cm[fit], L[fit], y[fit], Cm[~fit], L[~fit], y[~fit], 1e-2, 'label')[0].mean())
        out['c2_synthetic'] = dict(gain_real=gain(Cn), gain_within_class_perm=gain(Cn[tp._within_class_perm(y, 1)]))
    finally:
        sc.NCLS = old
    rp = os.path.join(PH, 'vinfo_repro_resnet.json')
    if os.path.exists(rp): out['vinfo']['repro_resnet'] = json.load(open(rp))
    json.dump(out, open(os.path.join(RES6, 'R6_posthoc.json'), 'w'), indent=1, default=float)
    for k, c in out['controls'].items(): print(k, c['energy_verdict'], {s: round(v, 3) for s, v in c['tau_EQ'].items()}, round(c['R_B_median'], 3), c.get('content_verdict'))
    print('ViM frozen', vf['energy_verdict'], {s: round(v, 3) for s, v in vf['tau_EQ'].items()}, round(vf['R_B_median'], 3), 'check', vf['max_abs_real_vs_audit'], vf['max_abs_E1refit_vs_audit'])
    print('magnitude', {k: round(v['E12'], 3) for k, v in out['magnitude'].items()}); print('tau null', out['tau_null'])
    print('vinfo', {k: v for k, v in out['vinfo'].items() if k not in ('real', 'lam_rel', 'lam_abs_mean', 'repro_resnet')})


if __name__ == '__main__':
    main()
