"""Round-2 fix 9: is the E8/E9 flatness rule (max/min over d <= 1.5) passable by a perfect score?
Part 1 (synthetic, numpy): known W, translator, plant; oracle O(B) = alpha*||B^T u|| (planted amplitude inside B, noise-free) and
M1 (E9 debiased null leakage). Part 2 (post hoc): re-judge E8 NCP and E9 DEV flatness with the revised rule
max/min_d [score(d)/O(d)] <= 1.5, O(d) = alpha * mean_j ||B_j^T u_c|| on the same sub-null bases (seed 20000+j) as E8/E9.
Usage: python r2/flatness_synth.py --feats-root $IC014_FEATS --sing-root /tmp/sing"""
import argparse, os, json, csv, numpy as np, torch
R = os.path.join(os.path.dirname(__file__), '..'); DS = [24, 50, 100, 200, 300]; ALPHAS = [0.1, 1.0]

def m1(X, s, sig2):
    n = len(s); Ss = X.T @ s; L2 = (Ss @ Ss - ((X ** 2).sum(1) * s ** 2).sum()) / (n * (n - 1) * sig2); return np.sqrt(max(L2, 0.0))

def synth():
    rng = np.random.default_rng(0); m, c, n, k = 400, 100, 10000, 64
    W = rng.standard_normal((c, m)); _, _, vt = np.linalg.svd(W); Vn = vt[c:].T                      # null dim 300
    O, _ = np.linalg.qr(rng.standard_normal((m, m))); lam = 1.0 / np.arange(1, m + 1)
    F = (rng.standard_normal((n, m)) * np.sqrt(lam)) @ O.T; A = rng.standard_normal((k, m)) / np.sqrt(m)
    t = rng.standard_normal(k); t /= np.linalg.norm(t); w = np.linalg.pinv(A) @ t
    s = rng.standard_normal(n); s = (s - s.mean()) / s.std(); sig2 = float((F ** 2).sum(1).mean()); sig = np.sqrt(sig2)
    R0 = np.linalg.qr(np.random.default_rng(7).standard_normal((300, 300)))[0]; S0 = Vn @ R0[:, :24]
    settings = {'i_full_null': Vn @ (Vn.T @ w), 'ii_fixed_24dim_subspace': S0 @ (S0.T @ w)}
    subs = [Vn @ np.linalg.qr(np.random.default_rng(20000 + j).standard_normal((300, 300)))[0] for j in range(5)]
    out = {}
    for name, un in settings.items():
        u = un / np.linalg.norm(un); res = {}
        for al in ALPHAS:
            Fp = F + al * sig * s[:, None] * u[None]; row = {}
            for dd in DS:
                o, mm = [], []
                for Bf in subs:
                    B = Bf[:, :dd]; o.append(al * np.linalg.norm(B.T @ u)); mm.append(m1(Fp @ B, s, sig2) - m1(F @ B, s, sig2))
                row[dd] = dict(oracle=float(np.mean(o)), M1=float(np.mean(mm)))
            ro = max(v['oracle'] for v in row.values()) / min(v['oracle'] for v in row.values())
            rm = max(v['M1'] for v in row.values()) / min(v['M1'] for v in row.values()) if min(v['M1'] for v in row.values()) > 0 else float('inf')
            rel = [v['M1'] / v['oracle'] for v in row.values()]
            res[str(al)] = dict(per_d=row, oracle_ratio=ro, oracle_flat=ro <= 1.5, M1_ratio=rm, M1_flat=rm <= 1.5,
                                M1_over_oracle_ratio=max(rel) / min(rel), M1_revised_flat=max(rel) / min(rel) <= 1.5)
        out[name] = res
    return out

def readjudicate(fr, sing):
    torch.set_default_dtype(torch.float64)   # E8/E9 built their sub-null bases in float64
    att = np.load(os.path.join(sing, 'misc/embeddings/attribute_embeddings_broden_style_fp16.npz'), allow_pickle=True)
    names = [str(x) for x in att['classes']]; pi = [str(p) for p in att['prompts']].index('mean')
    TA = torch.nn.functional.normalize(torch.tensor(att['embeddings'][:, pi].astype(np.float64)), dim=-1)
    BIG = ['resnet', 'resnext', 'dinov2', 'bitresnet']; DG = [24, 100, 280, 536]; frac = {}
    for nm in BIG:
        A = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/{nm}_full.npz')['A']).double(); Vn = torch.tensor(np.load(f'{fr}/e3/private/{nm}_null.npz')['Vn']).double()
        nd = Vn.shape[1]; Ap = torch.linalg.pinv(A)
        subs = [Vn @ torch.linalg.qr(torch.randn(nd, nd, generator=torch.Generator().manual_seed(20_000 + j)))[0] for j in range(5)]
        for c in ['red', 'striped', 'water']:
            w = Ap @ TA[names.index(c)]; un = Vn @ (Vn.T @ w); u = un / un.norm()
            for dd in DG: frac[(nm, c, dd)] = float(np.mean([float((S[:, :dd].T @ u).norm()) for S in subs]))
            frac[(nm, c, 1048)] = 1.0
    # E9 DEV
    e9 = json.load(open(os.path.join(R, 'e9/results/e9_dev.json'))); raw = {tuple(json.loads(k)): v for k, v in e9['dev_flat_raw'].items()}
    cands = list(next(iter(raw.values())).keys()); E9 = {}
    for sc in cands:
        r = {}
        for al in ALPHAS:
            rat = {}
            for nm in BIG:
                rs = []
                for c in ['red', 'striped']:
                    v = [raw[(nm, c, al, dd)][sc] / (al * frac[(nm, c, dd)]) for dd in DG + [1048]]
                    rs.append(max(v) / min(v) if min(v) > 0 else float('inf'))
                rat[nm] = float(np.mean(rs))
            r[str(al)] = dict(ratio=rat, n_flat=sum(x <= 1.5 for x in rat.values()))
        r['revised_flat'] = all(r[str(al)]['n_flat'] >= 3 for al in ALPHAS)
        r['dev_tau'] = e9['dev'][sc]['kendall']['mean']; r['original_flat'] = e9['dev'][sc]['flatness']['flat']; E9[sc] = r
    # E8 (NCP at alpha = 0.1, per-concept values in e8_flat.csv)
    rows = list(csv.DictReader(open(os.path.join(R, 'e8/results/e8_flat.csv')))); E8 = {}
    for nm in BIG:
        rs = []
        for c in ['red', 'striped', 'water']:
            v = {int(x['d']): float(x['ncp']) for x in rows if x['model'] == nm and x['concept'] == c}
            q = [v[dd] / (0.1 * frac[(nm, c, dd if dd in DG else 1048)]) for dd in sorted(v)]
            rs.append(max(q) / min(q) if min(q) > 0 else float('inf'))
        E8[nm] = float(np.mean(rs))
    return dict(E9=E9, E9_revised_qualifiers=[k for k, v in E9.items() if v['revised_flat'] and v['dev_tau'] >= 0.8],
                E8_revised_ratio=E8, E8_revised_n_flat=sum(x <= 1.5 for x in E8.values()),
                planted_fraction_in_B=dict((f'{k[0]}|{k[1]}|{k[2]}', v) for k, v in frac.items()))

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--sing-root', required=True); a = ap.parse_args()
    out = dict(synthetic=synth(), readjudication_post_hoc=readjudicate(a.feats_root, a.sing_root))
    json.dump(out, open(os.path.join(R, 'r2/results/fix9_flatness.json'), 'w'), indent=1, default=float)
    s = out['synthetic']; print(json.dumps({k: {al: {x: v[al][x] for x in ['oracle_ratio', 'oracle_flat', 'M1_ratio', 'M1_over_oracle_ratio', 'M1_revised_flat']} for al in v} for k, v in s.items()}, indent=1))
    r = out['readjudication_post_hoc']; print('E9 revised:', {k: (round(v['dev_tau'], 3), v['0.1']['n_flat'], v['1.0']['n_flat'], v['revised_flat']) for k, v in r['E9'].items()})
    print('E9 revised qualifiers (tau>=0.8 & revised flat):', r['E9_revised_qualifiers']); print('E8 revised:', r['E8_revised_ratio'], r['E8_revised_n_flat'])
