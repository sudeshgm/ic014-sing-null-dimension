"""PREREG_R8 K0 (synthetic only, run before any real-model theta_sem): coverage of the theta_sem resampling CIs.
Two scenarios at the real geometry (K = 1000 classes, 10 images, 768-d texts): 'signal' (truth = Monte-Carlo mean of the
point estimate over 200 fresh datasets) and 'zero' (no text-predictable null content, truth theta_sem = 0, with a constant
null offset). 100 coverage datasets per scenario, B = 300 draws each, for both class bootstrap ('boot') and half-sampling
('half'). Seeds: truth datasets 100820000+i, coverage datasets 100830000+i, draws 100806.
Rule (PREREG_R8 K0): use 'boot' if its coverage is in [0.90, 0.99] in both scenarios, else 'half' if it is, else K0 fires.
Output r8/results/K0_coverage.json. Usage: python r8/k0_r8.py"""
import os, sys, json, time, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H)
import core_r8 as c, synth_r8 as sy

K, NT, NC, B = 1000, 200, 100, 300
SCEN = {'signal': dict(sz=0.3, ss=1.0, sb=0.5, sw=1.0, offset=0.0), 'zero': dict(sz=0.0, ss=1.0, sb=0.5, sw=1.0, offset=3.0)}


def point(seed, kw):
    Z, S, y, T = sy.synth(seed, K=K, **kw); hA = c.halves(y, K); f = c.fold_split(K, 100800)
    cz, cs = c.CM.from_images(Z, y, K, hA), c.CM.from_images(S, y, K, hA)
    return cz, cs, T, f


def main():
    out = dict(K=K, n_truth=NT, n_cov=NC, B=B, scenarios=SCEN, res={}); t0 = time.time()
    for name, kw in SCEN.items():
        if name == 'signal':
            tr = []
            for i in range(NT):
                cz, cs, T, f = point(100_820_000 + i, kw); tr.append(c.theta_from_S(*c.S_multi([cz.G, cs.G], T, *f)))
            truth = float(np.mean(tr)); truth_sd = float(np.std(tr, ddof=1))
        else:
            truth, truth_sd = 0.0, None
        cov = {m: [] for m in ('boot', 'half')}; ses = {m: [] for m in ('boot', 'half')}; pts = []
        for i in range(NC):
            cz, cs, T, f = point(100_830_000 + i, kw)
            for m in ('boot', 'half'):
                r, _ = c.theta_sem_with_ci(cz, cs, T, f, c.resample_indices(*f, B, 100806, m), m)
                cov[m].append(bool(r['ci'][0] <= truth <= r['ci'][1])); ses[m].append(r['se'])
            pts.append(r['theta_sem'])
            if i % 20 == 0: print(name, i, round(time.time() - t0), flush=True)
        out['res'][name] = dict(truth=truth, truth_mc_sd=truth_sd, sd_point_over_datasets=float(np.std(pts, ddof=1)),
                                coverage={m: float(np.mean(v)) for m, v in cov.items()}, mean_se={m: float(np.mean(v)) for m, v in ses.items()})
        print(name, out['res'][name], flush=True)
    ok = lambda m: all(0.90 <= out['res'][s]['coverage'][m] <= 0.99 for s in SCEN)
    out['method'] = 'boot' if ok('boot') else ('half' if ok('half') else None); out['K0_fires'] = out['method'] is None
    out['secs'] = time.time() - t0
    json.dump(out, open(os.path.join(H, 'results', 'K0_coverage.json'), 'w'), indent=1); print('method', out['method'], 'K0 fires', out['K0_fires'])


if __name__ == '__main__':
    main()
