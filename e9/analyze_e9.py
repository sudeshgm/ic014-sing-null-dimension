"""IC014 E9 score screen. Spec: E9_SCORE_SCREEN_PREREG.md (+ amendments 1, 2). CPU only, on the private E3 features.
One automated run: DEV (all 20 candidates), then the selection rule, then the finalists-only holdout (linear and nonlinear plant).
Per-evaluation raw scores are checkpointed to --cache (jsonl) so an interrupted run resumes without recomputing."""
import argparse, os, sys, json, csv, math, time, itertools, numpy as np, torch
from scipy.stats import kendalltau, spearmanr, rankdata
from scipy.special import digamma
torch.set_default_dtype(torch.float64)
MODELS = ['resnet', 'resnext', 'dinov2', 'bitresnet', 'deit', 'eva02_large', 'convnext', 'densenet']
BIG = ['resnet', 'resnext', 'dinov2', 'bitresnet']
DEV_C = ['red', 'striped']; HOLD_C = ['water', 'green', 'thin', 'knitted']
DGRID = [24, 100, 280, 536]; FLAT_A = [0.1, 1.0]
SINGLE = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6']
LAMS = 10.0 ** np.arange(-3, 5.01, 0.5); NDRAW = 20; NSUB = 5

def ridge(X, y, tr, te, return_pred=False):
    Xt = X[tr]; mu = Xt.mean(0); sd = Xt.std(0).clamp_min(1e-12); yt = y[tr]; ym, ys = yt.mean(), yt.std()
    Xt = (Xt - mu) / sd; ytt = (yt - ym) / ys; Xe = (X[te] - mu) / sd; ye = (y[te] - ym) / ys
    e, V = torch.linalg.eigh(Xt.T @ Xt); e = e.clamp_min(0); XV = Xt @ V; b = XV.T @ ytt; best = None
    for lam in LAMS:
        w = 1.0 / (e + lam); h = (XV ** 2) @ w; yh = XV @ (w * b)
        loo = float((((ytt - yh) / (1 - h)) ** 2).mean())
        if best is None or loo < best[0]: best = (loo, lam)
    pred = Xe @ (V @ (b / (e + best[1])))
    r2 = float(1 - ((ye - pred) ** 2).sum() / ((ye - ye.mean()) ** 2).sum())
    return (r2, pred, ye) if return_pred else r2

def nystrom(X, Lidx):
    XL = X[Lidx]; sq = (X ** 2).sum(1); sl = sq[Lidx]
    D2 = (sq[:, None] + sl[None] - 2 * X @ XL.T).clamp_min(0)
    dl = D2[Lidx]; iu = torch.triu_indices(len(Lidx), len(Lidx), 1); h2 = float(dl[iu[0], iu[1]].sqrt().median()) ** 2
    K = torch.exp(-D2 / (2 * max(h2, 1e-12))); e, V = torch.linalg.eigh(K[Lidx]); e = e.clamp_min(e.max() * 1e-10)
    return K @ (V / e.sqrt()) @ V.T

def whiten_r2(X, y, tr, te):
    Xt = X[tr]; mu = Xt.mean(0); Xc = Xt - mu; d = X.shape[1]; S = Xc.T @ Xc / len(tr)
    e, V = torch.linalg.eigh(S + 1e-3 * float(torch.trace(S)) / d * torch.eye(d)); Wh = V / e.clamp_min(1e-300).sqrt() @ V.T
    yt = y[tr]; ym = yt.mean(); w = torch.linalg.lstsq(Xc @ Wh, (yt - ym)[:, None]).solution
    pred = (X[te] - mu) @ Wh @ w + ym; ye = y[te]
    return float(1 - ((ye - pred[:, 0]) ** 2).sum() / ((ye - ye.mean()) ** 2).sum())

def hsic_u(X, Y):
    n = X.shape[0]; nx = (X ** 2).sum(1); ny = (Y ** 2).sum(1)
    tr_ = float(((X.T @ Y) ** 2).sum() - (nx * ny).sum()); k1 = X @ X.sum(0) - nx; l1 = Y @ Y.sum(0) - ny
    return (tr_ + float(k1.sum()) * float(l1.sum()) / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * float(k1 @ l1)) / (n * (n - 3))

def cka_u(X, Y):
    a = hsic_u(X, X); b = hsic_u(Y, Y); return hsic_u(X, Y) / math.sqrt(a * b) if a > 0 and b > 0 else float('nan')

def ksg(x, y, k=5):  # Kraskov et al. 2004, algorithm 1, max-norm, on standardised x, y
    from scipy.spatial import cKDTree
    x = (x - x.mean()) / x.std(); y = (y - y.mean()) / y.std(); n = len(x); P = np.c_[x, y]
    dist, _ = cKDTree(P).query(P, k=k + 1, p=np.inf); eps = dist[:, -1]
    xs, ys = np.sort(x), np.sort(y)
    nx = np.searchsorted(xs, x + eps, 'left') - np.searchsorted(xs, x - eps, 'right') - 1
    ny = np.searchsorted(ys, y + eps, 'left') - np.searchsorted(ys, y - eps, 'right') - 1
    return float(digamma(k) + digamma(n) - np.mean(digamma(nx + 1) + digamma(ny + 1)))

def ang(Z, t): return torch.rad2deg(torch.arccos((torch.nn.functional.normalize(Z, dim=-1) @ t).clamp(-1, 1)))

class Model:
    def __init__(s, name, a, TA, names, cc):
        d = np.load(os.path.join(a.feats, f'{name}_full.npz')); s.name = name
        s.F = torch.tensor(d['F']).double(); s.A = torch.tensor(d['A']).double(); Ab = torch.tensor(d['Ab']).double()
        W = torch.tensor(d['W']).double(); s.L = s.F @ W.T + torch.tensor(d['bW']).double()
        s.Vn = torch.tensor(np.load(os.path.join(a.private, f'{name}_null.npz'))['Vn']).double()
        s.n, s.m = s.F.shape; s.nd = s.Vn.shape[1]; s.sigma = float((s.F ** 2).sum(1).mean().sqrt())
        s.Z = s.F @ s.A.T + Ab; Apinv = torch.linalg.pinv(s.A); s.EF2 = float((s.F ** 2).sum(1).mean())
        s.Q, s.FQ, s.AQ, s.colE = [], [], [], []
        for r in range(NDRAW):
            g = torch.Generator().manual_seed(10_000 + r); Q = torch.linalg.qr(torch.randn(s.m, s.m, generator=g))[0]
            FQ = s.F @ Q; s.Q.append(Q); s.FQ.append(FQ.float()); s.AQ.append(s.A @ Q); s.colE.append((FQ ** 2).mean(0))
        s.subs = []
        for j in range(NSUB):
            g = torch.Generator().manual_seed(20_000 + j); s.subs.append(s.Vn @ torch.linalg.qr(torch.randn(s.nd, s.nd, generator=g))[0])
        s.conc = {}
        for c in set(a.dev_concepts + HOLD_C):
            t = TA[names.index(c)]; w = Apinv @ t; un = s.Vn @ (s.Vn.T @ w); u = un / un.norm()
            s.conc[c] = dict(t=t, w=w, u=u, Au=s.A @ u, Fu=s.F @ u, uQ=[Q.T @ u for Q in s.Q])
    def B(s, key):
        if key == 'full': return s.Vn
        d, j = key; return s.subs[j][:, :d]

def evaluate(M, c, mode, alpha, bkey, s_raw, tr, te, Lidx, need, cache_red):
    C = M.conc[c]; B = M.B(bkey); d = B.shape[1]; gs = s_raw if mode == 'lin' else s_raw ** 2 - 1
    y = (gs - gs.mean()) / gs.std(); sc = alpha * M.sigma * gs
    X = M.F @ B + sc[:, None] * (B.T @ C['u'])[None]; EX = float((X ** 2).sum(1).mean())
    out = {}; AB = M.A @ B; t = C['t']
    if need & {'S1', 'S7', 'S9'}:
        r2, pred, ye = ridge(X, y, tr, te, True); out['_r2'] = r2
        out['S7'] = float(np.corrcoef(pred.numpy(), ye.numpy())[0, 1]); out['S9'] = ksg(pred.numpy(), ye.numpy())
    Xr = []
    if need & {'S1', 'S2', 'S8'}:
        for r in range(NDRAW):
            uq = C['uQ'][r]; FQ = M.FQ[r]; Fs = (FQ.double().T @ sc) / M.n
            cum = torch.cumsum(M.colE[r] + 2 * uq * Fs + float((sc ** 2).mean()) * uq ** 2, 0)
            k = min(int(torch.searchsorted(cum, torch.tensor(EX)).item()), M.m - 1); prev = float(cum[k - 1]) if k > 0 else 0.0
            P = FQ[:, :k + 1].double() + sc[:, None] * uq[None, :k + 1]
            beta = math.sqrt(max(EX - prev, 0.0) / float((P[:, k] ** 2).mean())); P[:, k] *= beta; Xr.append(P)
    if 'S1' in need:
        rr = np.array([ridge(P, y, tr, te) for P in Xr]); out['S1'] = (out['_r2'] - rr.mean()) / rr.std()
    if 'S2' in need:
        r2k = ridge(nystrom(X, Lidx), y, tr, te); rk = np.array([ridge(nystrom(P, Lidx), y, tr, te) for P in Xr])
        out['S2'] = (r2k - rk.mean()) / rk.std()
    if 'S8' in need:
        Y = y[:, None]; out['S8'] = cka_u(X, Y) - float(np.mean([cka_u(P, Y) for P in Xr]))
    if 'S3' in need: out['S3'] = whiten_r2(X, y, tr, te)
    if need & {'S4', 'S5', 'S6', 'M5'}:
        Zp = M.Z + sc[:, None] * C['Au'][None]; a0 = ang(Zp, t); asn = float((a0 - ang(Zp - X @ AB.T, t)).abs().mean())
        Ef = M.EF2 + 2 * alpha * M.sigma * float((gs * C['Fu']).mean()) + (alpha * M.sigma) ** 2 * float((gs ** 2).mean()); rho = EX / Ef
        out['S4'] = asn / rho; out['S5'] = asn / d; out['M5'] = asn / math.sqrt(rho)
        if 'S6' in need:
            ar = []
            for r in range(NDRAW):
                Xq = M.FQ[r][:, :d].double() + sc[:, None] * C['uQ'][r][None, :d]
                ar.append(float((a0 - ang(Zp - Xq @ M.AQ[r][:, :d].T, t)).abs().mean()))
            ar = np.array(ar); out['S6'] = (asn - ar.mean()) / ar.std()
    if 'S10' in need: out['S10'] = float(spearmanr((X @ (B.T @ C['w'])).numpy(), y.numpy()).statistic)
    if need & {'M1', 'M2'}:
        Ssum = X.T @ y; L2 = (float(Ssum @ Ssum) - float(((X ** 2).sum(1) * y ** 2).sum())) / (M.n * (M.n - 1) * M.sigma ** 2)
        out['M1'] = math.sqrt(max(L2, 0.0)); uh = Ssum / Ssum.norm()
        out['M2'] = out['M1'] * abs(float(t @ (AB @ uh))) / (float(AB.norm()) / math.sqrt(d))
    if 'M6' in need:
        out['M6'] = float((y * (X @ (AB.T @ t))).mean()) / math.sqrt(float(((X @ AB.T) ** 2).sum(1).mean()))
    if need & {'M3', 'M4'}:
        nte, p0 = len(te), M.L.shape[1]; adj = lambda r2, p: 1 - (1 - r2) * (nte - 1) / (nte - p - 1)
        key = (M.name, c, mode)
        if key not in cache_red: cache_red[key] = adj(ridge(M.L, y, tr, te), p0)
        m3 = adj(ridge(torch.cat([M.L, X], 1), y, tr, te), p0 + d) - cache_red[key]
        out['M3'] = m3; out['M4'] = -0.5 * math.log(1 - min(max(m3, 0.0), 0.999999))
    out.pop('_r2', None); return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats', required=True); ap.add_argument('--private', required=True)
    ap.add_argument('--sing-root', required=True); ap.add_argument('--out', required=True); ap.add_argument('--cache', required=True)
    ap.add_argument('--models', nargs='*', default=MODELS); ap.add_argument('--dev-concepts', nargs='*', default=DEV_C)
    ap.add_argument('--trials', type=int, default=50); ap.add_argument('--dev-only', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    att = np.load(os.path.join(a.sing_root, 'misc/embeddings/attribute_embeddings_broden_style_fp16.npz'), allow_pickle=True)
    names = [str(c) for c in att['classes']]; pi = [str(p) for p in att['prompts']].index('mean')
    TA = torch.nn.functional.normalize(torch.tensor(att['embeddings'][:, pi].astype(np.float64)), dim=-1)
    cc = np.load(os.path.join(a.private, 'clip_concepts.npz')); cn = [str(c) for c in cc['concepts']]
    S = {c: torch.tensor(cc['zscore'][:, cn.index(c)]).double() for c in set(a.dev_concepts + HOLD_C)}
    perm = np.random.default_rng(0).permutation(10000); tr = torch.tensor(perm[:5000]); te = torch.tensor(perm[5000:])
    Lidx = tr[torch.tensor(np.random.default_rng(0).choice(5000, 1000, replace=False))]
    alphas = {sd: np.exp(np.random.default_rng(sd).uniform(math.log(0.01), math.log(1.0), size=(50, len(MODELS))))[:a.trials] for sd in (0, 1)}
    done = {}
    if os.path.exists(a.cache):
        for line in open(a.cache): r = json.loads(line); done[r['key']] = r['v']
    cf = open(a.cache, 'a'); cache_red = {}; models = {}
    def get(name):
        if name not in models: models.clear(); models[name] = Model(name, a, TA, names, cc)
        return models[name]
    def run(phase, concepts, mode, seed, need):
        """Returns raw-difference scores: trials[(ti, c, model)] and flat[(model, c, alpha, d)] = mean over seeds."""
        trials, flat = {}, {}
        for name in a.models:
            t0 = time.time(); M = None
            jobs = [('full', 0.0)] + [('full', float(alphas[seed][ti, MODELS.index(name)])) for ti in range(a.trials)]
            if name in BIG:
                keys = [(dd, j) for dd in DGRID for j in range(NSUB)]
                for bk in keys: jobs += [(bk, 0.0)] + [(bk, al) for al in FLAT_A]
                jobs += [('full', al) for al in FLAT_A]
            for c in concepts:
                res = {}
                for bk, al in jobs:
                    key = json.dumps([phase, mode, name, c, bk if bk == 'full' else list(bk), al, sorted(need)])
                    if key not in done:
                        if M is None: M = get(name)
                        done[key] = evaluate(M, c, mode, al, bk, S[c], tr, te, Lidx, need, cache_red)
                        cf.write(json.dumps(dict(key=key, v=done[key])) + '\n'); cf.flush()
                    res[(bk, al)] = done[key]
                diff = lambda bk, al: {k: res[(bk, al)][k] - res[(bk, 0.0)][k] for k in res[(bk, al)]}
                for ti in range(a.trials): trials[(ti, c, name)] = diff('full', float(alphas[seed][ti, MODELS.index(name)]))
                if name in BIG:
                    for al in FLAT_A:
                        for dd in DGRID:
                            v = [diff((dd, j), al) for j in range(NSUB)]; flat[(name, c, al, dd)] = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
                        flat[(name, c, al, 1048)] = diff('full', al)
            print(f'# {phase} {mode} {name} {time.time()-t0:.0f}s', flush=True)
        return trials, flat
    def kendall(trials, concepts, seed, score_fn):
        per = {c: [] for c in concepts}
        for ti in range(a.trials):
            al = alphas[seed][ti, [MODELS.index(n) for n in a.models]]
            for c in concepts: per[c].append(float(kendalltau([score_fn(trials, ti, c, n) for n in a.models], al).statistic))
        tm = np.mean([per[c] for c in concepts], 0); b = np.random.default_rng(0)
        boot = [tm[b.integers(0, len(tm), len(tm))].mean() for _ in range(1000)]
        return dict(mean=float(tm.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                    per_concept={c: float(np.mean(per[c])) for c in concepts}, per_trial=tm.tolist())
    def flatness(flat, concepts, sc):
        outp = {}
        for al in FLAT_A:
            rat = {}
            for n in [x for x in BIG if x in a.models]:
                rs = []
                for c in concepts:
                    v = [flat[(n, c, al, dd)][sc] for dd in DGRID + [1048]]; rs.append(max(v) / min(v) if min(v) > 0 else float('inf'))
                rat[n] = float(np.mean(rs))
            outp[str(al)] = dict(ratio=rat, n_flat=sum(1 for v in rat.values() if v <= 1.5))
        outp['flat'] = all(outp[str(al)]['n_flat'] >= 3 for al in FLAT_A); return outp
    single_fn = lambda sc: (lambda T, ti, c, n: T[(ti, c, n)][sc])
    def combo_fn(parts):
        def f(T, ti, c, n):
            rk = [rankdata([T[(ti, c, m)][p] for m in a.models]) for p in parts]; return float(np.mean([r[a.models.index(n)] for r in rk]))
        return f
    # DEV
    need = set(SINGLE); T, Fl = run('dev', a.dev_concepts, 'lin', 0, need)
    dev = {}
    for sc in SINGLE: dev[sc] = dict(kendall=kendall(T, a.dev_concepts, 0, single_fn(sc)), flatness=flatness(Fl, a.dev_concepts, sc))
    top3 = sorted(SINGLE, key=lambda s: -dev[s]['kendall']['mean'])[:3]
    combos = {f'S{11+i}': list(p) for i, p in enumerate(list(itertools.combinations(top3, 2)) + [tuple(top3)])}
    for k, parts in combos.items():
        fl = {str(al): dict(ratio={n: max(dev[p]['flatness'][str(al)]['ratio'][n] for p in parts) for n in dev[parts[0]]['flatness'][str(al)]['ratio']}) for al in FLAT_A}
        for al in FLAT_A: fl[str(al)]['n_flat'] = sum(1 for v in fl[str(al)]['ratio'].values() if v <= 1.5)
        fl['flat'] = all(fl[str(al)]['n_flat'] >= 3 for al in FLAT_A)
        dev[k] = dict(parts=parts, kendall=kendall(T, a.dev_concepts, 0, combo_fn(parts)), flatness=fl)
    ok = [k for k in dev if dev[k]['kendall']['mean'] >= 0.8 and dev[k]['flatness']['flat']]
    finalists = sorted(ok, key=lambda k: -dev[k]['kendall']['mean'])[:3]
    json.dump(dict(dev=dev, top3_singletons=top3, combos=combos, qualifying=ok, finalists=finalists,
                   dev_flat_raw={json.dumps(k): v for k, v in Fl.items()}), open(os.path.join(a.out, 'e9_dev.json'), 'w'), indent=1)
    with open(os.path.join(a.out, 'e9_dev_table.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['cand', 'parts', 'tau_mean', 'ci_lo', 'ci_hi'] + [f'tau_{c}' for c in a.dev_concepts] + [f'nflat_a{al}' for al in FLAT_A] + ['flat'])
        for k, v in dev.items():
            w.writerow([k, '+'.join(v.get('parts', [])), v['kendall']['mean'], *v['kendall']['ci95'], *[v['kendall']['per_concept'][c] for c in a.dev_concepts],
                        *[v['flatness'][str(al)]['n_flat'] for al in FLAT_A], v['flatness']['flat']])
    print(json.dumps(dict(top3=top3, qualifying=ok, finalists=finalists)), flush=True)
    if a.dev_only or not finalists: print('# no holdout (no finalists or dev-only)', flush=True); return
    # HOLDOUT, finalists only, once
    need = set(itertools.chain.from_iterable(dev[k].get('parts', [k]) for k in finalists)); hold = {}
    TH, FH = run('hold', HOLD_C, 'lin', 1, need); TN, _ = run('holdnl', HOLD_C, 'nl', 1, need)
    for k in finalists:
        parts = dev[k].get('parts'); fn = combo_fn(parts) if parts else single_fn(k)
        if parts:
            fls = [flatness(FH, HOLD_C, p) for p in parts]
            fl = {str(al): dict(ratio={n: max(x[str(al)]['ratio'][n] for x in fls) for n in fls[0][str(al)]['ratio']}) for al in FLAT_A}
            for al in FLAT_A: fl[str(al)]['n_flat'] = sum(1 for v in fl[str(al)]['ratio'].values() if v <= 1.5)
            fl['flat'] = all(fl[str(al)]['n_flat'] >= 3 for al in FLAT_A)
        else: fl = flatness(FH, HOLD_C, k)
        hold[k] = dict(kendall=kendall(TH, HOLD_C, 1, fn), flatness=fl, kendall_nonlinear=kendall(TN, HOLD_C, 1, fn))
    json.dump(dict(finalists=finalists, holdout=hold), open(os.path.join(a.out, 'e9_holdout.json'), 'w'), indent=1)
    print(json.dumps({k: (v['kendall']['mean'], v['kendall']['ci95'], v['flatness']['flat'], v['kendall_nonlinear']['mean']) for k, v in hold.items()}), flush=True)
main()
