"""Shared helpers for round-4 analyses (PREREG_R4.md). Re-uses r3/common.py."""
import os, sys, json, numpy as np
from scipy.stats import rankdata, pearsonr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common import R, RES as RES3, load_models, tau_ci, paired_ci, ktau, boot_idx, RHO0, S11_DROP, PRETTY  # noqa: F401
RES4 = os.path.join(R, 'r4', 'results')


def sets(ms, rho):
    return {'all13': list(ms), 'E12': [n for n, r in zip(ms, rho) if r >= RHO0], 'S11': [n for n in ms if n not in S11_DROP]}


def partial_spearman(x, y, z):
    rx, ry, rz = (rankdata(v) for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(pearsonr(res(rx), res(ry)).statistic)


def partial_ci(x, y, z, B=2000, seed=0):
    x, y, z = (np.asarray(v, float) for v in (x, y, z)); p = partial_spearman(x, y, z); bs = []
    for i in boot_idx(len(x), B, seed):
        if len(set(i.tolist())) < 5: continue
        with np.errstate(all='ignore'):
            try: v = partial_spearman(x[i], y[i], z[i])
            except Exception: continue
        if np.isfinite(v): bs.append(v)
    return dict(r=p, pct=[float(v) for v in np.percentile(bs, [2.5, 97.5])], B_used=len(bs))


def dump4(name, obj):
    os.makedirs(RES4, exist_ok=True); json.dump(obj, open(os.path.join(RES4, name), 'w'), indent=1, default=float)


def load_json(path): return json.load(open(path))
