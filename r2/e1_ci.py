"""Round-2 fix 6: E1 exponent CIs at the right unit (cells / models). Input: e1/results/e1_cells.csv (committed)."""
import csv, json, os, numpy as np
R = os.path.join(os.path.dirname(__file__), '..')
C = [r for r in csv.DictReader(open(os.path.join(R, 'e1/results/e1_cells.csv'))) if r['feasible'] == 'True']
y = np.log([float(r['absAS']) for r in C]); lr = np.log([float(r['rho']) for r in C]); ld = np.log([float(r['d']) for r in C])
mods = sorted(set(r['model'] for r in C)); mi = np.array([mods.index(r['model']) for r in C])
def fit(idx):
    ms = sorted(set(mi[idx])); X = np.column_stack([lr[idx], ld[idx]] + [(mi[idx] == m).astype(float) for m in ms])
    b, *_ = np.linalg.lstsq(X, y[idx], rcond=None); res = y[idx] - X @ b; return b[0], b[1], 1 - res.var() / y[idx].var()
allidx = np.arange(len(y)); a0, b0, r20 = fit(allidx)
rng = np.random.default_rng(0); cell = np.array([fit(rng.integers(0, len(y), len(y)))[:2] for _ in range(2000)])
rng = np.random.default_rng(0); clus = []
for _ in range(2000):
    pick = rng.integers(0, len(mods), len(mods)); idx = np.concatenate([np.where(mi == p)[0] for p in pick])
    if len(set(mi[idx])) < 2: continue
    clus.append(fit(idx)[:2])
clus = np.array(clus)
lomo = {mods[m]: dict(zip(['a', 'b'], map(float, fit(np.where(mi != m)[0])[:2]))) for m in range(len(mods))}
per = {}
for m in range(len(mods)):
    idx = np.where(mi == m)[0]
    if len(idx) > 3:
        X = np.column_stack([lr[idx], ld[idx], np.ones(len(idx))]); b, *_ = np.linalg.lstsq(X, y[idx], rcond=None)
        per[mods[m]] = dict(a=float(b[0]), b=float(b[1]), n_cells=int(len(idx)))
    else: per[mods[m]] = dict(n_cells=int(len(idx)), note='too few cells for a per-model fit')
pc = lambda A, j: [float(np.percentile(A[:, j], 2.5)), float(np.percentile(A[:, j], 97.5))]
out = dict(n_cells=len(y), models=mods, a=float(a0), b=float(b0), r2=float(r20),
           cell_boot=dict(a_ci95=pc(cell, 0), b_ci95=pc(cell, 1), n=2000),
           model_cluster_boot=dict(a_ci95=pc(clus, 0), b_ci95=pc(clus, 1), n=int(len(clus))),
           leave_one_model_out=lomo, per_model=per,
           lomo_a_range=[min(v['a'] for v in lomo.values()), max(v['a'] for v in lomo.values())])
json.dump(out, open(os.path.join(R, 'r2/results/fix6_e1_ci.json'), 'w'), indent=1); print(json.dumps(out, indent=1))
