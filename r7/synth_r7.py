"""PREREG_R7 synthetic stability study (run and frozen BEFORE any real feature; kill rule K0).
World: N = 900 images in 9 groups of 100, k null coordinates, V = 4 views, a 768-d reference built like the real one
(per group 30 fine classes; reference = unit text-like vector of the image's fine class). Noise-free edit
mu_i = w * A r_i + (1 - w) * u_i + group shift (A: k x 768 Gaussian, u_i idiosyncratic Gaussian); w sets the population
target T* = the noise-free value of T (within-group RV coefficient of mu and R). Per-view noise e_{i,v} independent across views, with energy nu x the within-group
signal energy, of three kinds: 'iso' (isotropic Gaussian), 'lowrank' (power-law spectrum j^-1.5 in a random basis,
realistic nuisance), 'heavy' (lowrank with a per-image Student-t(3) scale).
For each (kind, nu, T*) and rep: T (cross-view), T_naive, defined or not. Rows summarised as median, SD, undefined rate.
T (4th-order denominator) and T2 (first-order denominator) are both evaluated.
Stable at (stat, kind, nu, T*) iff undefined <= 5 %, |median T - T*| <= 0.2 T*, SD <= 0.25 T*."""
import os, sys, json, time, argparse, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import stats_r7 as st  # noqa: E402

SYN_SEED = 100740


def world(rng, N=900, G=9, k=1048, q=768, nfine=30):
    g = np.repeat(np.arange(G), N // G); gv = rng.standard_normal((G, q))
    fine = gv[:, None, :] * 0.7 + rng.standard_normal((G, nfine, q)) * 0.7; fine /= np.linalg.norm(fine, axis=-1, keepdims=True)
    b = rng.integers(0, nfine, N); R = fine[g, b]
    A = rng.standard_normal((k, q)) / np.sqrt(q); sig = R @ A.T; u = rng.standard_normal((N, k)) / np.sqrt(k) * np.sqrt((sig ** 2).sum(1).mean())
    shift = rng.standard_normal((G, k)) * 0.5 * np.sqrt((sig ** 2).sum(1).mean() / k)
    return g, R, sig, u, shift[g]


def mu_for(sig, u, shift, w):
    return w * sig + (1 - w) * u + shift


def tstar(mu, R, g):
    """noise-free value of T on this sample (4 identical noise-free views)"""
    return st.T_stat(np.repeat(mu[None], 4, 0), R, g)['T']


def noise(rng, kind, N, k, V, energy, basis=None):
    if kind == 'iso': E = rng.standard_normal((V, N, k))
    else:
        lam = np.arange(1, k + 1) ** -1.5; E = (rng.standard_normal((V, N, k)) * np.sqrt(lam)) @ basis.T
        if kind == 'heavy': E = E * np.sqrt(3 / rng.chisquare(3, (V, N, 1)))
    return E * np.sqrt(energy / (E ** 2).sum(-1).mean())


def calibrate_w(g, R, sig, u, shift, target):
    lo, hi = 0.0, 1.0
    for _ in range(40):
        m = (lo + hi) / 2
        if tstar(mu_for(sig, u, shift, m), R, g) < target: lo = m
        else: hi = m
    return (lo + hi) / 2


def run(kinds=('iso', 'lowrank', 'heavy'), nus=(0.5, 1, 2, 4, 8, 16, 32), targets=(0.1, 0.2, 0.4), reps=40, k=1048, N=900, seed=SYN_SEED, log=None):
    rng = np.random.default_rng(seed); g, R, sig, u, shift = world(rng, N=N, k=k)
    basis = np.linalg.qr(rng.standard_normal((k, k)))[0]; rows = []
    for t in targets:
        w = calibrate_w(g, R, sig, u, shift, t); mu = mu_for(sig, u, shift, w); ts = tstar(mu, R, g); t2s = st.T_stat(np.repeat(mu[None], 4, 0), R, g)['T2']
        sE = float(np.mean([((mu[g == c] - mu[g == c].mean(0)) ** 2).sum(1).mean() for c in np.unique(g)]))
        for kind in kinds:
            for nu in nus:
                T, T2, Tn = [], [], []; t0 = time.time()
                for r in range(reps):
                    D = mu[None] + noise(rng, kind, N, k, 4, nu * sE, basis); s = st.T_stat(D, R, g); T.append(s['T']); T2.append(s['T2']); Tn.append(st.T_naive(D, R, g))
                for name, vals, star in (('T', T, ts), ('T2', T2, t2s)):
                    vals = np.array(vals); ok = np.isfinite(vals); med = float(np.median(vals[ok])) if ok.any() else np.nan; sd = float(np.std(vals[ok])) if ok.sum() > 1 else np.nan
                    row = dict(stat=name, target=t, star=star, kind=kind, nu=nu, undefined=float(1 - ok.mean()), median=med, sd=sd, naive=float(np.median(Tn)),
                               stable=bool((1 - ok.mean()) <= 0.05 and abs(med - star) <= 0.2 * star and sd <= 0.25 * star))
                    rows.append(row)
                    if log: print(json.dumps({**row, 'secs': round(time.time() - t0, 1)}), flush=True)
    return rows


def summarise(rows):
    out = {}
    for t, kind, stat in sorted({(r['target'], r['kind'], r['stat']) for r in rows}):
            rr = sorted([r for r in rows if r['target'] == t and r['kind'] == kind and r['stat'] == stat], key=lambda r: r['nu']); nu_ok = 0
            for r in rr:
                if r['stable']: nu_ok = r['nu']
                else: break
            meds = [r['median'] for r in rr if r['stable']]
            out[f'{stat}_{kind}_T{t}'] = dict(max_stable_nu=nu_ok, worldC_ratio_over_stable=(max(meds) / min(meds)) if meds and min(meds) > 0 else None,
                                       naive_ratio=rr[0]['naive'] / rr[-1]['naive'] if rr[-1]['naive'] > 0 else None)
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, default=40); ap.add_argument('--out', default=os.path.join(H, 'results', 'synthetic_stability.json'))
    ap.add_argument('--k', type=int, default=1048); a = ap.parse_args()
    rows = run(reps=a.reps, k=a.k, log=True); s = summarise(rows)
    json.dump(dict(rows=rows, summary=s, seed=SYN_SEED, reps=a.reps, k=a.k), open(a.out, 'w'), indent=1); print(json.dumps(s, indent=1))
