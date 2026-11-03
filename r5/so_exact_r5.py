"""Round 5, post hoc (not pre-registered; see r5/POSTHOC_R5.md): SO_null with exact hollow double-centring.

r5/organisation.py dcent() subtracts off-diagonal row means of a hollow (zero-diagonal) matrix. That leaves 1/(N-1) of
each additive class effect a_i + a_j, and the row sums of its output are not centred. dcent_exact() below projects out
the additive span exactly. This script reruns SO_null with dcent_exact on both the distance and the text side, with the
same features, folds, text embeddings, permutation null, G2 derangements and G3 plants as organisation.py, next to the
original values (which it recomputes and checks against r5/results/R5_gates.json). organisation.py and PREREG_R5.md are
unchanged. Aggregates only.

Usage: python r5/so_exact_r5.py --feats-root $IC014_FEATS [--sing-root _ext/sing] [--models a,b]
       python r5/so_exact_r5.py --summary          # -> r5/results/posthoc/SO_exact_summary.json (no features needed)
"""
import os, sys, json, time, argparse, numpy as np
from scipy.stats import rankdata
R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in ('r5', 'r4', 'r3'): sys.path.insert(0, os.path.join(R, p))
import organisation as O  # noqa: E402
from organisation import cmean, xdist, corr, folds, load, energy_fix, NCLS, IU, PERM_SEED, NPERM, PLANT_SEED, DER_SEED, NDER, REL_FLOOR  # noqa: E402
OUT = os.path.join(R, 'r5', 'results', 'posthoc'); PER = os.path.join(OUT, 'SO_exact_models.json'); SUM = os.path.join(OUT, 'SO_exact_summary.json')
MODELS = ['vgg16', 'vgg19', 'densenet', 'resnet', 'dinovit1', 'efficientnet', 'bitresnet', 'resnext', 'convnext', 'swin_large', 'dinov2', 'deit', 'eva02_large']


def dcent_exact(v, n=NCLS):
    """remove a_i + a_j (and the constant) exactly from a hollow symmetric pair vector (upper triangle)"""
    iu = np.triu_indices(n, 1); M = np.zeros((n, n)); M[iu] = v; M = M + M.T; r = M.sum(1); a = (r - r.sum() / (2 * (n - 1))) / (n - 2)
    return (M - a[:, None] - a[None])[iu]


class TextX:
    def __init__(s, T): s.T = T; s.rt = rankdata(dcent_exact((1 - T @ T.T)[IU])); s.Tc = T - T.mean(0)


def so_exact(Z, y, g, tx, nperm=0, seed=PERM_SEED):
    D1 = rankdata(dcent_exact(xdist(Z, y, g == 0, g == 1))); D2 = rankdata(dcent_exact(xdist(Z, y, g == 2, g == 3)))
    r1, r2 = corr(D1, tx.rt), corr(D2, tx.rt); rel = corr(D1, D2)
    o = dict(rel=rel, raw_q=0.5 * (r1 + r2), dis=float(0.5 * (r1 + r2) / np.sqrt(rel)) if rel > 0 else float('nan'), estimable=bool(rel >= REL_FLOOR))
    if nperm:
        rng = np.random.default_rng(seed); M1 = np.zeros((NCLS, NCLS)); M2 = np.zeros((NCLS, NCLS)); M1[IU] = D1; M2[IU] = D2; M1 += M1.T; M2 += M2.T; nl = []
        for _ in range(nperm):
            p = rng.permutation(NCLS); nl.append(0.5 * (corr(M1[np.ix_(p, p)][IU], tx.rt) + corr(M2[np.ix_(p, p)][IU], tx.rt)) / np.sqrt(max(rel, 1e-12)))
        o.update(null_mean=float(np.mean(nl)), null_sd=float(np.std(nl, ddof=1)))
    return o


def additive_share(v):
    """share of the variance of a cross-fitted pair vector explained by additive class effects a_i + a_j"""
    vc = v - v.mean(); r = vc - dcent_exact(v); return float((r @ r) / (vc @ vc))


def run(fr, sing_root, names):
    from class_perm import derangement
    T = O.text(sing_root); tx5 = O.Text(T); txx = TextX(T); os.makedirs(OUT, exist_ok=True)
    res = json.load(open(PER)) if os.path.exists(PER) else {}
    for n in names:
        t0 = time.time(); F, y, Vr, Vn = load(fr, n); C = F @ Vn; del F, Vr; g = folds(y); d = C.shape[1]
        s5 = O.so(C, y, g, tx5); sx = so_exact(C, y, g, txx, NPERM)
        o = dict(model=n, SO_r5=s5['dis'], rel_r5=s5['rel'], SO_x=sx['dis'], rel_x=sx['rel'], raw_q_x=sx['raw_q'], est_x=sx['estimable'],
                 null_sd_x=sx['null_sd'], null_mean_x=sx['null_mean'], additive_share_D=additive_share(xdist(C, y, g < 2, g >= 2)))
        mu = cmean(C, y, np.ones(len(y), bool)); Rr = C - mu[y]
        dv = [so_exact(mu[derangement(NCLS, DER_SEED + r)][y] + Rr, y, g, txx)['dis'] for r in range(NDER)]
        m = float(np.nanmean(dv)); o['G2_x'] = dict(vals=dv, mean=m, pass_=bool(abs(m) <= 0.10 * abs(o['SO_x']) or abs(m) <= 2 * o['null_sd_x'] / np.sqrt(NDER)))
        rng = np.random.default_rng(PLANT_SEED); P = rng.standard_normal((T.shape[1], d)); us = tx5.Tc @ P; ua = rng.standard_normal((NCLS, d)); e2 = (C ** 2).sum(1).mean(); g3 = {}
        for lab, u in (('semantic', us), ('arbitrary', ua)):
            u = u / np.sqrt((u ** 2).sum(1).mean()) * np.sqrt(e2); g3[lab] = so_exact(energy_fix(C + 0.3 * u[y], e2), y, g, txx)['dis']
        ds = (g3['semantic'] - o['SO_x']) / o['null_sd_x']; da = (g3['arbitrary'] - o['SO_x']) / o['null_sd_x']
        o['G3_x'] = dict(sem=g3['semantic'], arb=g3['arbitrary'], sem_sd=float(ds), arb_sd=float(da), pass_=bool(np.isfinite(ds) and np.isfinite(da) and ds >= 2 and da <= 1))
        res[n] = o; json.dump(res, open(PER, 'w'), indent=1, default=float)
        print(n, round(o['SO_r5'], 4), round(o['SO_x'], 4), 'G2', o['G2_x']['pass_'], 'G3', o['G3_x']['pass_'], round(time.time() - t0, 1), flush=True)
        del C, Rr, mu


def summary():
    from common4 import load_models, tau_ci, ktau, sets, partial_ci
    ms, col, sing = load_models(); lk = -sing; rho = col('rho'); S = sets(ms, rho); P = json.load(open(PER))
    G5 = json.load(open(os.path.join(R, 'r5', 'results', 'R5_gates.json')))['models']; out = {}
    for key, ek in (('SO_x', 'est_x'), ('SO_r5', None)):
        out[key] = {}
        for s in ('all13', 'E12', 'S11'):
            sub = [n for n in S[s] if (ek is None or P[n][ek]) and np.isfinite(P[n][key])]
            x = np.array([P[n][key] for n in sub]); yv = np.array([lk[ms.index(n)] for n in sub]); rh = np.array([rho[ms.index(n)] for n in sub])
            out[key][s] = dict(n=len(sub), tau=tau_ci(x, yv), tau_vs_rho=float(ktau(x, rh)), tau_vs_eta2=float(ktau(x, np.array([G5[n]['eta2'] for n in sub]))), partial_given_rho=partial_ci(x, yv, rh))
        t = {s: out[key][s]['tau']['tau'] for s in ('E12', 'S11')}
        out[key]['label'] = 'ORGANISATION-TRACKS' if all(v >= 0.6 for v in t.values()) else 'ORGANISATION-DOES-NOT-TRACK' if all(v < 0.3 for v in t.values()) else 'INCONCLUSIVE'
    x5 = np.array([P[n]['SO_r5'] for n in ms]); xx = np.array([P[n]['SO_x'] for n in ms])
    out['tau_r5_vs_x_all13'] = float(ktau(x5, xx)); out['max_abs_diff'] = float(np.abs(x5 - xx).max())
    out['r5_reproduced_max_diff'] = float(max(abs(P[n]['SO_r5'] - G5[n]['SO']) for n in ms))
    out['G2_pass'] = int(sum(P[n]['G2_x']['pass_'] for n in ms)); out['G3_pass'] = int(sum(P[n]['G3_x']['pass_'] for n in ms)); out['n_est_x'] = int(sum(P[n]['est_x'] for n in ms))
    json.dump(out, open(SUM, 'w'), indent=1, default=float, sort_keys=True)
    print({k: v for k, v in out.items() if not isinstance(v, dict)}, out['SO_x']['label'], out['SO_r5']['label'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root'); ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing')))
    ap.add_argument('--models', default=','.join(MODELS)); ap.add_argument('--summary', action='store_true'); a = ap.parse_args()
    if a.feats_root: run(a.feats_root, a.sing_root, a.models.split(','))
    if a.summary or a.feats_root: summary()
