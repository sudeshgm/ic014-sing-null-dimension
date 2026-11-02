"""PREREG_R4 J (part): A1 as a per-energy statistic, its influence, and both translator families. Uses committed per-model files only
(r3/results/A_equal_energy_permodel.json, G_retrain_permodel.json) plus r4/results/retrained_kappa_permodel.json if present.
Usage: python r4/a1_stats.py"""
import os, sys, json, itertools, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common4 import RES3, RES4, load_models, tau_ci, paired_ci, ktau, sets, partial_ci, dump4, RHO0
AHAT = 0.431


def outcome(t, pc):
    if all(v >= 0.6 for v in t.values()): return 'KILL-R3 level'
    if all(t[s] < 0.3 and pc[s]['pct'][0] > 0 for s in t): return 'ENERGY-ACCOUNTS'
    return 'PARTIAL'


def main():
    ms, col, sing = load_models(); lk = -sing; rho = col('rho'); AS = col('AS_SING'); S = sets(ms, rho)
    A = json.load(open(os.path.join(RES3, 'A_equal_energy_permodel.json'))); G = json.load(open(os.path.join(RES3, 'G_retrain_permodel.json')))
    ix = {n: i for i, n in enumerate(ms)}
    v = dict(A1_rel={n: A[n]['A1'] for n in ms}, A1_ret={n: G[n]['A1_retrained'] for n in ms}, AS_ret={n: G[n]['AS_SING_retrained'] for n in ms},
             AS={n: AS[ix[n]] for n in ms}, rho={n: rho[ix[n]] for n in ms}, norm_sqrt={n: AS[ix[n]] / rho[ix[n]] ** 0.5 for n in ms},
             norm_ahat={n: AS[ix[n]] / rho[ix[n]] ** AHAT for n in ms})
    V = lambda k, sub: np.array([v[k][n] for n in sub]); L = lambda sub: np.array([lk[ix[n]] for n in sub]); res = dict(normaliser={}, influence={}, translators={})
    for s in ('E12', 'S11'):
        sub = S[s]
        res['normaliser'][s] = {k: dict(tau=ktau(V('A1_rel', sub), V(k, sub)), spearman=float(spearmanr(V('A1_rel', sub), V(k, sub)).statistic))
                                for k in ('norm_sqrt', 'norm_ahat')}
        res['normaliser'][s]['ratio_A1_over_beta_AS'] = {n: A[n]['A1'] / (A[n]['A1_beta'] * A[n]['AS_SING']) for n in sub}
        res['influence'][s] = {}
        for key in ('A1_rel', 'A1_ret'):
            jk = {n: ktau(V(key, [m for m in sub if m != n]), L([m for m in sub if m != n])) for n in sub}
            l2 = {f'{p}+{q}': ktau(V(key, [m for m in sub if m not in (p, q)]), L([m for m in sub if m not in (p, q)])) for p, q in itertools.combinations(sub, 2)}
            vals = np.array(list(l2.values()))
            res['influence'][s][key] = dict(tau=ktau(V(key, sub), L(sub)), jackknife=jk, jackknife_range=[min(jk.values()), max(jk.values())],
                                            l2o_min=float(vals.min()), l2o_max=float(vals.max()), l2o_argmax=max(l2, key=l2.get),
                                            l2o_share_ge_0p6=float(np.mean(vals >= 0.6)), l2o_n=len(vals),
                                            drop_dinovit1_eva02=l2.get('dinovit1+eva02_large', l2.get('eva02_large+dinovit1')))
        sub_r = S[s]
        res['translators'][s] = dict(
            tau_A1rel_A1ret=ktau(V('A1_rel', sub_r), V('A1_ret', sub_r)), tau_A1ret_rho=ktau(V('A1_ret', sub_r), V('rho', sub_r)),
            tau_A1ret=tau_ci(V('A1_ret', sub_r), L(sub_r)), tau_A1rel=tau_ci(V('A1_rel', sub_r), L(sub_r)),
            partial_A1ret_given_rho=partial_ci(V('A1_ret', sub_r), L(sub_r), V('rho', sub_r)),
            paired_AS_minus_A1ret=paired_ci(V('AS', sub_r), V('A1_ret', sub_r), L(sub_r)),
            paired_ASret_minus_A1ret=paired_ci(V('AS_ret', sub_r), V('A1_ret', sub_r), L(sub_r)),
            A1ret_range=[float(V('A1_ret', sub_r).min()), float(V('A1_ret', sub_r).max())], A1rel_range=[float(V('A1_rel', sub_r).min()), float(V('A1_rel', sub_r).max())])
    res['translators']['all13'] = dict(tau_A1ret_rho=ktau(V('A1_ret', ms), V('rho', ms)), tau_A1rel_A1ret=ktau(V('A1_rel', ms), V('A1_ret', ms)))
    t = {s: res['translators'][s]['tau_A1ret']['tau'] for s in ('E12', 'S11')}
    pc = {s: res['translators'][s]['paired_AS_minus_A1ret'] for s in ('E12', 'S11')}
    res['KILL_R3_rule_on_retrained'] = dict(outcome=outcome(t, pc), tau=t, paired_lower_pct={s: pc[s]['pct'][0] for s in pc})
    t0 = {s: res['influence'][s]['A1_rel']['tau'] for s in ('E12', 'S11')}
    res['KILL_R3_rule_on_released'] = dict(outcome=outcome(t0, {s: paired_ci(V('AS', S[s]), V('A1_rel', S[s]), L(S[s])) for s in ('E12', 'S11')}), tau=t0)
    res['S11_l2o_reaches_kill_level'] = bool(res['influence']['S11']['A1_rel']['l2o_max'] >= 0.6)
    dump4('J_a1_stats.json', res)
    print(json.dumps({k: res[k] for k in ['KILL_R3_rule_on_retrained', 'KILL_R3_rule_on_released', 'S11_l2o_reaches_kill_level']}, indent=1, default=float))
    for s in ('E12', 'S11'):
        print(s, {k: round(x['tau'], 3) for k, x in res['normaliser'][s].items() if isinstance(x, dict) and 'tau' in x},
              {k: round(x['spearman'], 3) for k, x in res['normaliser'][s].items() if isinstance(x, dict) and 'spearman' in x})
        for key in ('A1_rel', 'A1_ret'):
            d = res['influence'][s][key]; print(' ', key, round(d['tau'], 3), 'jk', [round(x, 3) for x in d['jackknife_range']], 'l2o', round(d['l2o_min'], 3), round(d['l2o_max'], 3), d['l2o_argmax'], round(d['l2o_share_ge_0p6'], 3), 'drop pair', d['drop_dinovit1_eva02'])
        tr = res['translators'][s]; print('  ', {k: (round(x, 3) if isinstance(x, float) else x) for k, x in tr.items() if not isinstance(x, dict)},
              'pairedAS-A1ret', tr['paired_AS_minus_A1ret']['diff'], tr['paired_AS_minus_A1ret']['pct'], 'partial', tr['partial_A1ret_given_rho'])


if __name__ == '__main__':
    main()
