"""PREREG_R4 P: planted content against energy, from r3/results/B_core.json (no feature computation).
r_E = AS(rho_75)/AS(rho_25) at alpha = 0 (log-log interpolation over the registered levels and own rho), rho_25/rho_75 = quartiles
of rho over E12; alpha* = alpha at which the planted ratio at rho_t = 0.2 equals r_E (linear in alpha over {0, 0.1, 0.3}).
Usage: python r4/planted.py"""
import os, sys, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common4 import RES3, load_models, sets, dump4


def main():
    ms, col, _ = load_models(); rho = col('rho'); S = sets(ms, rho); r = np.array([rho[ms.index(n)] for n in S['E12']])
    q25, q75 = (float(v) for v in np.percentile(r, [25, 75])); B = json.load(open(os.path.join(RES3, 'B_core.json'))); out = dict(rho_q25=q25, rho_q75=q75, models={})
    for m, d in B.items():
        lv = sorted((float(k), v['0.0']) for k, v in d['AS_by_level_alpha'].items()); x = np.log([a for a, _ in lv]); y = np.log([b for _, b in lv])
        f = lambda q: float(np.exp(np.interp(np.log(q), x, y))); rE = f(q75) / f(q25)
        pr = d['planted_ratio']['rho_t=0.2']; al = [0.0, 0.1, 0.3]; pv = [pr['0.0'], pr['0.1'], pr['0.3']]
        astar = float(np.interp(rE, pv, al)) if pv[-1] >= rE else None
        own = d['planted_ratio']['rho_t=own']; ok = f"{d['own_rho']:.4f}"
        out['models'][m] = dict(own_rho=d['own_rho'], r_E=rE, AS_q25=f(q25), AS_q75=f(q75), planted_ratio_rho02=pr, planted_ratio_own=own,
                                AS_own_abs={k: d['AS_by_level_alpha'][ok][k] for k in ('0.0', '0.1', '0.3')} if ok in d['AS_by_level_alpha'] else None,
                                alpha_star=astar, alpha_star_beyond_0p3=astar is None)
    dump4('P_planted.json', out); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
