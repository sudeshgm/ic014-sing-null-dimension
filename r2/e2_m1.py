"""Round-2 fix 10, second part (descriptive, post hoc): the E2 13-model ranking under E9's M1 (debiased null leakage).
M1 per model = mean over the 173 SING concepts of M1 at alpha=0 on the full null (tol 1e-5), as computed by
r2/e2_controls.py (column M1_concept_mean of results/fix2_models.csv). With --private, the concept mean is recomputed
from the per-concept M1 arrays in <private>/<model>_r2img.npz as a cross-check.
tau = Kendall tau-b of (-M1) vs the digitised SING Supp. Fig. 12b order; 2000x model bootstrap (seed 0) and
leave-one-out range, as in r2/e2_sweep.py. Also reports agreement of M1's ranking with null energy rho and with AS_SING.
Usage: python r2/e2_m1.py [--private $IC014_FEATS/r2_private]"""
import os, csv, json, argparse, numpy as np
from scipy.stats import kendalltau, spearmanr
R = os.path.dirname(os.path.abspath(__file__))
FIG6B = {'densenet': 70.71, 'dinovit1': 70.16, 'eva02_large': 54.32, 'bitresnet': 47.21, 'convnext': 37.38, 'efficientnet': 36.28,
         'swin_large': 29.18, 'resnet': 24.81, 'vgg16': 24.26, 'vgg19': 19.89, 'dinov2': 16.61, 'resnext': 13.88, 'deit': 9.51}  # as in e2_sweep.py
TAU_SING = 0.8205128205128204
rows = {r['model']: r for r in csv.DictReader(open(os.path.join(R, 'results/fix2_models.csv')))}
f = lambda m, k: float(rows[m][k]); ALL = list(rows); SCR = [m for m in ALL if rows[m]['screen_exclude'] != 'True']

def tau(models, fn): return float(kendalltau([-fn(m) for m in models], [FIG6B[m] for m in models]).statistic)

def boot(models, fn, B=2000):   # same resampling as e2_sweep.boot (seed 0, skip < 4 distinct models)
    rng = np.random.default_rng(0); out = []
    for _ in range(B):
        s = [models[i] for i in rng.integers(0, len(models), len(models))]
        if len(set(s)) < 4: continue
        v = kendalltau([-fn(m) for m in s], [FIG6B[m] for m in s]).statistic
        if np.isfinite(v): out.append(v)
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], len(out)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--private', default=None); a = ap.parse_args()
    m1 = lambda m: f(m, 'M1_concept_mean')
    check = None
    if a.private:
        check = {m: float(np.load(os.path.join(a.private, f'{m}_r2img.npz'))['M1'].mean()) for m in ALL}
        assert all(abs(check[m] - m1(m)) <= 1e-9 * max(1.0, abs(m1(m))) for m in ALL), 'M1 cross-check failed'
    out = dict(label='post hoc, descriptive (PREREG_R2.md fix 10: E2 under M1, reported regardless)',
               tau_ref_AS_SING=TAU_SING, per_model={m: dict(M1=m1(m), rho=f(m, 'rho'), AS_SING=f(m, 'AS_SING'), fig12b=FIG6B[m]) for m in ALL},
               cross_check_from_private=check is not None)
    for lab, ms in [('all13', ALL), ('screened', SCR)]:
        ci, nb = boot(ms, m1); loo = [tau([x for x in ms if x != k], m1) for k in ms]
        out[lab] = dict(models=ms, tau_vs_SING=tau(ms, m1), ci95=ci, n_boot=nb, loo_range=[min(loo), max(loo)],
                        tau_M1_vs_rho=float(kendalltau([m1(m) for m in ms], [f(m, 'rho') for m in ms]).statistic),
                        spearman_M1_vs_rho=float(spearmanr([m1(m) for m in ms], [f(m, 'rho') for m in ms]).statistic),
                        tau_M1_vs_AS_SING=float(kendalltau([m1(m) for m in ms], [f(m, 'AS_SING') for m in ms]).statistic))
    p = os.path.join(R, 'results/fix10_e2_m1.json'); json.dump(out, open(p, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k in ('all13', 'screened')}, indent=1))

if __name__ == '__main__': main()
