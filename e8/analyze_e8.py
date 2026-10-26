"""IC014 E8: Null-space Concept Probe (NCP). The spec is in e8/PREREG_E8.md. CPU only, using the private E3 features.
Outputs (small): e8_trials.csv, e8_flat.csv, e8_summary.json, e8_alpha_grid.csv."""
import argparse, os, sys, json, csv, math, time, numpy as np, torch
from scipy.stats import kendalltau
MODELS = ['resnet', 'resnext', 'dinov2', 'bitresnet', 'deit', 'eva02_large', 'convnext', 'densenet']
CONCEPTS = ['red', 'striped', 'water']; DGRID = [24, 100, 280, 536]
LAMS = 10.0 ** np.arange(-3, 5.01, 0.5)
torch.set_default_dtype(torch.float64)

def ridge_r2(X, y, tr, te):
    Xt, yt = X[tr], y[tr]; mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-12); ym, ys = yt.mean(), yt.std()
    Xt = (Xt - mu) / sd; ytt = (yt - ym) / ys; Xe = (X[te] - mu) / sd; ye = (y[te] - ym) / ys
    e, V = torch.linalg.eigh(Xt.T @ Xt); e = e.clamp_min(0); XV = Xt @ V; b = XV.T @ ytt
    best = None
    for lam in LAMS:
        w = 1.0 / (e + lam); yh = XV @ (w * b); h = (XV ** 2) @ w
        loo = float((((ytt - yh) / (1 - h)) ** 2).mean())
        if best is None or loo < best[0]: best = (loo, lam)
    w = V @ (b / (e + best[1])); pred = Xe @ w
    return float(1 - ((ye - pred) ** 2).sum() / ((ye - ye.mean()) ** 2).sum()), float(best[1])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats', required=True); ap.add_argument('--private', required=True)
    ap.add_argument('--sing-root', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--models', nargs='*', default=MODELS); ap.add_argument('--trials', type=int, default=50)
    ap.add_argument('--draws', type=int, default=20); ap.add_argument('--subdraws', type=int, default=5)
    ap.add_argument('--skip-trials', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    att = np.load(os.path.join(a.sing_root, 'misc/embeddings/attribute_embeddings_broden_style_fp16.npz'), allow_pickle=True)
    names = [str(c) for c in att['classes']]; pi = [str(p) for p in att['prompts']].index('mean')
    TA = torch.nn.functional.normalize(torch.tensor(att['embeddings'][:, pi].astype(np.float64)), dim=-1)
    cc = np.load(os.path.join(a.private, 'clip_concepts.npz')); cn = [str(c) for c in cc['concepts']]
    rng = np.random.default_rng(0)  # same draws as E3 (8 models in the E3 order)
    trial_alpha = np.exp(rng.uniform(math.log(0.01), math.log(1.0), size=(a.trials, len(MODELS))))
    N = 10000; perm = np.random.default_rng(0).permutation(N); tr = torch.tensor(perm[:8000]); te = torch.tensor(perm[8000:])
    trial_ncp, flat_rows, mods, grid_rows = {}, [], {}, []
    for name in a.models:
        t0 = time.time(); mi = MODELS.index(name)
        d = np.load(os.path.join(a.feats, f'{name}_full.npz')); F = torch.tensor(d['F']).double(); A = torch.tensor(d['A']).double()
        Vn = torch.tensor(np.load(os.path.join(a.private, f'{name}_null.npz'))['Vn']).double(); m = F.shape[1]; nd = Vn.shape[1]
        sigma = float((F ** 2).sum(1).mean().sqrt()); Apinv = torch.linalg.pinv(A)
        FQ = []
        for r in range(a.draws):
            g = torch.Generator().manual_seed(10_000 + r); Q = torch.linalg.qr(torch.randn(m, m, generator=g))[0]; FQ.append((Q, (F @ Q).float()))
        subs = []
        for j in range(a.subdraws):
            g = torch.Generator().manual_seed(20_000 + j); R = torch.linalg.qr(torch.randn(nd, nd, generator=g))[0]; subs.append(Vn @ R)
        ds = [x for x in DGRID if x < nd] + [nd]
        mods[name] = dict(m=m, null_dim=nd, sigma=sigma, d_grid=ds)
        for c in CONCEPTS:
            t = TA[names.index(c)]; s = torch.tensor(cc['zscore'][:, cn.index(c)]).double()
            w = Apinv @ t; un = Vn @ (Vn.T @ w); u = un / un.norm()
            uQ = [Q.T @ u for Q, _ in FQ]
            def ncp(alpha, B):
                sc = alpha * sigma * s; X = F @ B + sc[:, None] * (B.T @ u)[None]
                r2s, lam = ridge_r2(X, s, tr, te); E = float((X ** 2).sum(1).mean()); rr, ks = [], []
                for (Q, fq), uq in zip(FQ, uQ):
                    P = fq.double() + sc[:, None] * uq[None]; cum = torch.cumsum((P ** 2).mean(0), 0)
                    k = min(int(torch.searchsorted(cum, torch.tensor(E)).item()), m - 1); prev = float(cum[k - 1]) if k > 0 else 0.0
                    beta = math.sqrt(max(E - prev, 0.0) / float((P[:, k] ** 2).mean())); Xr = P[:, :k + 1].clone(); Xr[:, k] *= beta
                    rr.append(ridge_r2(Xr, s, tr, te)[0]); ks.append(k + beta)
                rr = np.array(rr); sd = float(rr.std())
                return dict(ncp=(r2s - rr.mean()) / sd if sd > 0 else float('nan'), r2_null=r2s, r2_rand_mean=float(rr.mean()),
                            r2_rand_sd=sd, k_rand=float(np.mean(ks)), lam=lam)
            for al in [0.0, 0.01, 0.1, 1.0]:
                r = ncp(al, Vn); grid_rows.append(dict(model=name, concept=c, alpha=al, **r)); print(json.dumps(grid_rows[-1]), flush=True)
            for dd in ds:
                vals = [ncp(0.1, B[:, :dd]) for B in (subs if dd < nd else [Vn])]
                flat_rows.append(dict(model=name, concept=c, d=dd, ncp=float(np.mean([v['ncp'] for v in vals])),
                                      r2_sub=float(np.mean([v['r2_null'] for v in vals])), r2_rand=float(np.mean([v['r2_rand_mean'] for v in vals])),
                                      ncp_sd_over_subdraws=float(np.std([v['ncp'] for v in vals]))))
                print(json.dumps(flat_rows[-1]), flush=True)
            if not a.skip_trials:
                for ti in range(a.trials):
                    trial_ncp[(ti, c, name)] = ncp(float(trial_alpha[ti, mi]), Vn)['ncp']
        print(f'# {name} done {time.time()-t0:.0f}s', flush=True)
    W = lambda fn, rows: (lambda f: (lambda w: (w.writeheader(), w.writerows(rows)))(csv.DictWriter(f, fieldnames=list(rows[0]))))(open(os.path.join(a.out, fn), 'w', newline=''))
    W('e8_flat.csv', flat_rows); W('e8_alpha_grid.csv', grid_rows)
    summ = dict(models=mods, concepts=CONCEPTS, alpha_flat=0.1, lams=LAMS.tolist(), draws=a.draws, subdraws=a.subdraws)
    ratios = {}
    for name in a.models:
        rs = [r for r in flat_rows if r['model'] == name]; ds = mods[name]['d_grid']
        avg = [float(np.mean([r['ncp'] for r in rs if r['d'] == dd])) for dd in ds]
        pc = {c: (lambda v: max(v) / min(v) if min(v) > 0 else None)([r['ncp'] for r in rs if r['concept'] == c]) for c in CONCEPTS}
        ratios[name] = dict(d=ds, ncp_avg=avg, ratio=(max(avg) / min(avg) if min(avg) > 0 else None), per_concept_ratio=pc)
    elig = [n for n in a.models if mods[n]['null_dim'] >= 1048]
    nflat = sum(1 for n in elig if ratios[n]['ratio'] is not None and ratios[n]['ratio'] <= 1.5)
    summ.update(flatness=ratios, eligible=elig, n_flat=nflat)
    if not a.skip_trials:
        rows, per = [], {c: [] for c in CONCEPTS}
        for ti in range(a.trials):
            al = trial_alpha[ti, [MODELS.index(n) for n in a.models]]; row = dict(trial=ti)
            for c in CONCEPTS:
                v = [trial_ncp[(ti, c, n)] for n in a.models]; tau = float(kendalltau(v, al).statistic); per[c].append(tau)
                row[f'tau_{c}'] = tau; row.update({f'ncp_{c}_{n}': x for n, x in zip(a.models, v)})
            row['tau_mean'] = float(np.mean([row[f'tau_{c}'] for c in CONCEPTS])); rows.append(row)
        W('e8_trials.csv', rows); tm = np.array([r['tau_mean'] for r in rows]); b = np.random.default_rng(0)
        boot = [tm[b.integers(0, len(tm), len(tm))].mean() for _ in range(1000)]
        summ['kendall'] = dict(mean=float(tm.mean()), ci95_boot_mean=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                               pct2_5_97_5_trials=[float(np.percentile(tm, 2.5)), float(np.percentile(tm, 97.5))],
                               per_concept={c: float(np.mean(per[c])) for c in CONCEPTS}, n=len(tm))
    json.dump(summ, open(os.path.join(a.out, 'e8_summary.json'), 'w'), indent=1); print(json.dumps({k: summ.get(k) for k in ['kendall', 'n_flat']}))
main()
