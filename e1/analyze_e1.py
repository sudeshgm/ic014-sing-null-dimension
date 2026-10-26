"""IC014 E1 CPU analysis: energy vs dimension. Input: <feats>/<model>_nullfeat.npz from extract_null_gpu.py.
Energy PCs = eigenvectors of the UNCENTERED second moment M = c^T c / N of the null coordinates (so rho of a
subspace = tr(U^T M U) * N / sum ||f||^2 exactly).
(a) fixed d in {24,100,280}: high = top-d PCs, low = bottom-d PCs, random = 20 seeds (QR of Gaussian, seed 1000*d+s).
    If d == null dim all three equal the full null space (reported once as 'full'); d > null dim infeasible.
(b) fixed rho in {0.005,0.02,0.08} at d in {24,100,280,536}: top-k PCs + bottom (d-k-1) PCs + one vector rotating
    between PC k and the next-lowest unused PC, angle solved to hit rho exactly. Feasible iff
    rho_low(d) <= rho <= rho_high(d) and d <= null dim. Deterministic (seed SD = 0).
Outputs to <out>: e1_cells.csv, e1_fit.json (pooled log|AS| = a log rho + b log d + model FE, 1000-image-bootstrap 95% CI),
e1_ratios.json. Usage: python analyze_e1.py <feats_dir> <out_dir> [--sing-root /path/to/sing] [--boot 1000]"""
import argparse, os, json, csv, numpy as np, torch
MODELS = ['resnet', 'resnext', 'dinov2', 'eva02_large', 'vgg16']
DS_A = [24, 100, 280]; RHOS = [0.005, 0.02, 0.08]; DS_B = [24, 100, 280, 536]

def ang(a, b):
    a = torch.nn.functional.normalize(a, dim=-1); b = torch.nn.functional.normalize(b, dim=-1)
    return torch.rad2deg(torch.acos(torch.clamp((a * b).sum(-1), -1, 1)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('feats'); ap.add_argument('out')
    ap.add_argument('--sing-root', default='/tmp/sing'); ap.add_argument('--boot', type=int, default=1000)
    ap.add_argument('--seeds', type=int, default=20); ap.add_argument('--models', nargs='*', default=MODELS)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    Em = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float32)), dim=-1).mean(1), dim=-1)
    cells, perimg = [], []   # perimg: (model, cell_idx, |AS| per image, energy per image)
    for m in a.models:
        fn = os.path.join(a.feats, f'{m}_nullfeat.npz')
        if not os.path.exists(fn): print('missing', m); continue
        D = np.load(fn); c = torch.tensor(D['c']).double(); z = torch.tensor(D['z']).double(); AVn = torch.tensor(D['AVn']).double()
        fsq_sum = float(D['fsq'].astype(np.float64).sum()); ty = Em[torch.tensor(D['y']).long()].double(); N, nd = c.shape
        lam, E = torch.linalg.eigh(c.T @ c / N); lam, E = lam.flip(0), E.flip(1)   # descending
        k_rho = N / fsq_sum
        def run(U):
            cu = c @ U; zp = z - cu @ (AVn @ U).T
            AS = ang(z, ty) - ang(zp, ty); IS = ang(z, zp); e = (cu ** 2).sum(1)
            return AS.abs(), IS, e
        def add(kind, d, U_list, target=None):
            r = [run(U) for U in U_list]
            A = torch.stack([x[0] for x in r]); I = torch.stack([x[1] for x in r]); e = torch.stack([x[2] for x in r])
            per_seed_AS = A.mean(1); per_seed_IS = I.mean(1); per_seed_rho = e.sum(1) / fsq_sum
            cells.append(dict(model=m, null_dim=nd, block='a' if target is None else 'b', kind=kind, d=d, target_rho=target, n_seeds=len(U_list),
                              absAS=float(per_seed_AS.mean()), absAS_sd=float(per_seed_AS.std()) if len(U_list) > 1 else 0.0,
                              IS=float(per_seed_IS.mean()), IS_sd=float(per_seed_IS.std()) if len(U_list) > 1 else 0.0,
                              rho=float(per_seed_rho.mean()), rho_sd=float(per_seed_rho.std()) if len(U_list) > 1 else 0.0, feasible=True))
            perimg.append((len(cells) - 1, A.mean(0).float().numpy(), e.mean(0).float().numpy()))
        def infeasible(block, kind, d, target, why):
            cells.append(dict(model=m, null_dim=nd, block=block, kind=kind, d=d, target_rho=target, n_seeds=0, absAS=None, absAS_sd=None,
                              IS=None, IS_sd=None, rho=None, rho_sd=None, feasible=False, note=why))
        for d in DS_A:
            if d > nd:
                for kd in ['high', 'low', 'random']: infeasible('a', kd, d, None, f'd>{nd}')
            elif d == nd:
                add('full', d, [torch.eye(nd, dtype=torch.float64)])
            else:
                add('high', d, [E[:, :d]]); add('low', d, [E[:, -d:]])
                add('random', d, [torch.linalg.qr(torch.randn(nd, d, generator=torch.Generator().manual_seed(1000 * d + s)).double())[0] for s in range(a.seeds)])
        for t in RHOS:
            for d in DS_B:
                if d > nd: infeasible('b', 'mix', d, t, f'd>{nd}'); continue
                lo, hi = float(lam[-d:].sum()) * k_rho, float(lam[:d].sum()) * k_rho
                if not (lo <= t <= hi): infeasible('b', 'mix', d, t, f'rho range [{lo:.4g},{hi:.4g}]'); continue
                if d == nd: add('mix', d, [torch.eye(nd, dtype=torch.float64)], t); continue
                # rho(k) = top k + bottom d-k; find k with rho(k) <= t <= rho(k+1)
                for k in range(d):
                    base = float(lam[:k].sum() + lam[nd - (d - k - 1):].sum()) if d - k - 1 > 0 else float(lam[:k].sum())
                    lh, ll = float(lam[k]), float(lam[nd - (d - k)])
                    if (base + ll) * k_rho <= t <= (base + lh) * k_rho: break
                s2 = 0.0 if lh == ll else (lh - t / k_rho + base) / (lh - ll)   # sin^2 theta
                s2 = min(max(s2, 0.0), 1.0)
                u = np.sqrt(1 - s2) * E[:, k] + np.sqrt(s2) * E[:, nd - (d - k)]
                cols = [E[:, :k]] + ([E[:, nd - (d - k - 1):]] if d - k - 1 > 0 else []) + [u[:, None]]
                add('mix', d, [torch.cat(cols, 1)], t); cells[-1]['k_high'] = k
        print(m, 'done', flush=True)
    keys = sorted({k for c in cells for k in c}, key=lambda k: list(cells[0]).index(k) if k in cells[0] else 99)
    with open(os.path.join(a.out, 'e1_cells.csv'), 'w') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(cells)
    # pooled fit with model fixed effects
    perimg = [p for p in perimg if cells[p[0]]['absAS'] > 1e-12 and cells[p[0]]['rho'] > 1e-12]  # log fit needs >0; excluded cells noted
    fit_excluded = [cells[i]['model'] + ':' + cells[i]['kind'] + ':' + str(cells[i]['d']) for i in range(len(cells)) if cells[i]['feasible'] and i not in {p[0] for p in perimg}]
    idx = [i for i, _, _ in perimg]; ASm = np.stack([p[1] for p in perimg]).astype(np.float64); Em_ = np.stack([p[2] for p in perimg]).astype(np.float64)
    mods = sorted({cells[i]['model'] for i in idx}); dlog = np.log([cells[i]['d'] for i in idx])
    fsq_tot = {}
    for m in mods: fsq_tot[m] = np.load(os.path.join(a.feats, f'{m}_nullfeat.npz'))['fsq'].astype(np.float64)
    def fit(sel):
        y = np.log(ASm[:, sel].mean(1)); rho = np.array([Em_[j, sel].sum() / fsq_tot[cells[i]['model']][sel].sum() for j, i in enumerate(idx)])
        X = np.column_stack([np.log(rho), dlog] + [[cells[i]['model'] == mm for i in idx] for mm in mods]).astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
        return beta, 1 - res.var() / y.var()
    N = ASm.shape[1]; beta, r2 = fit(np.arange(N)); rng = np.random.default_rng(0)
    B = np.array([fit(rng.integers(0, N, N))[0][:2] for _ in range(a.boot)])
    fitres = dict(formula='log mean|AS| = a*log(rho) + b*log(d) + c_model', n_cells=len(idx), models=mods,
                  a=beta[0], a_ci95=np.percentile(B[:, 0], [2.5, 97.5]).tolist(), b=beta[1], b_ci95=np.percentile(B[:, 1], [2.5, 97.5]).tolist(),
                  c_model=dict(zip(mods, beta[2:].tolist())), r2=r2, n_boot=a.boot, excluded_zero_cells=fit_excluded, boot_seed=0, corr_logrho_logd=None)
    rho_all = np.array([cells[i]['rho'] for i in idx]); fitres['corr_logrho_logd'] = float(np.corrcoef(np.log(rho_all), dlog)[0, 1])
    json.dump(fitres, open(os.path.join(a.out, 'e1_fit.json'), 'w'), indent=1, default=float)
    rat = {}
    for m in mods:
        cm = [c for c in cells if c['model'] == m and c['feasible']]
        sd = lambda x, y: x / y if y > 1e-12 else None
        g = lambda kind, d: next((c for c in cm if c['kind'] == kind and c['d'] == d), None)
        rat[m] = dict(high_over_low={d: sd(g('high', d)['absAS'], g('low', d)['absAS']) for d in DS_A if g('high', d)},
                      high_over_random={d: sd(g('high', d)['absAS'], g('random', d)['absAS']) for d in DS_A if g('high', d)},
                      fixed_rho_max_over_min={})
        for t in RHOS:
            v = {c['d']: c['absAS'] for c in cm if c['block'] == 'b' and c['target_rho'] == t}
            rat[m]['fixed_rho_max_over_min'][t] = dict(ratio=sd(max(v.values()), min(v.values())) if len(v) >= 2 else None, absAS_by_d=v)
    json.dump(rat, open(os.path.join(a.out, 'e1_ratios.json'), 'w'), indent=1)
    print(json.dumps(dict(fit={k: fitres[k] for k in ['a', 'a_ci95', 'b', 'b_ci95', 'r2', 'n_cells', 'corr_logrho_logd']}), default=float))

if __name__ == '__main__':
    main()
