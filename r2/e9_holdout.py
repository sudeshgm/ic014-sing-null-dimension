"""Round-2 fix 10 (post hoc by construction; E9 stays KILL as pre-registered): evaluate the E9 holdout
{water, green, thin, knitted}, 50 trials at seed 1, linear and nonlinear plants, for M1, M2, S9, S10, M5, S4, AS_SING (raw)
and AS_fair-A (alpha=0-subtracted). Reuses e9/analyze_e9.py's Model/evaluate unchanged (imported without running its main).
Also: original and fix-9 revised flatness, paired bootstrap of tau(M1) - tau(AS_fair) over trials (1000x, seed 0).
Usage: python r2/e9_holdout.py --feats-root $IC014_FEATS [--sing-root _ext/sing] [--cache r2_private/e9_holdout_cache.jsonl]"""
import argparse, os, sys, json, math, time, numpy as np, torch
from scipy.stats import kendalltau
R = os.path.dirname(os.path.abspath(__file__)); src = open(os.path.join(R, '..', 'e9', 'analyze_e9.py')).read()
assert src.rstrip().endswith('main()'); E9 = {}; exec(compile(src.rstrip()[:-len('main()')], 'analyze_e9.py', 'exec'), E9)
torch.set_num_threads(os.cpu_count()); MODELS, BIG, HOLD_C, DGRID, FLAT_A, NDRAW, NSUB = (E9[k] for k in ['MODELS', 'BIG', 'HOLD_C', 'DGRID', 'FLAT_A', 'NDRAW', 'NSUB'])
CANDS = ['M1', 'M2', 'S9', 'S10', 'M5', 'S4']; ALLC = CANDS + ['AS_SING', 'AS_fair']; GUARD_S = 3 * 3600

def as_scores(M, c, mode, jobs, s_raw):
    """AS_SING (full-null or sub-null B removal, mean|AS|) and AS_fair-A (/ energy-matched full-space random, 20 draws) per job.
    Random removals are computed with an incremental k-prefix sum per draw (exact algebra, same k/beta rule as E2/E3)."""
    C = M.conc[c]; t = C['t']; gs = s_raw if mode == 'lin' else s_raw ** 2 - 1; out = {}
    info = []
    for bk, al in jobs:
        B = M.B(bk); sc = al * M.sigma * gs; X = M.F @ B + sc[:, None] * (B.T @ C['u'])[None]; EX = float((X ** 2).sum(1).mean())
        Zp = M.Z + sc[:, None] * C['Au'][None]; a0 = ang(Zp, t); asn = float((a0 - ang(Zp - X @ (M.A @ B).T, t)).abs().mean())
        info.append((bk, al, sc, EX, a0, asn))
    rnd = {i: [] for i in range(len(info))}
    for r in range(NDRAW):
        FQ = M.FQ[r]; AQ = M.AQ[r].float(); uq = C['uQ'][r]; plan = []
        for i, (bk, al, sc, EX, a0, asn) in enumerate(info):
            Fs = (FQ.double().T @ sc) / M.n; cum = torch.cumsum(M.colE[r] + 2 * uq * Fs + float((sc ** 2).mean()) * uq ** 2, 0)
            k = min(int(torch.searchsorted(cum, torch.tensor(EX)).item()), M.m - 1); prev = float(cum[k - 1]) if k > 0 else 0.0
            Pk = FQ[:, k].double() + sc * uq[k]; beta = math.sqrt(max(EX - prev, 0.0) / float((Pk ** 2).mean())); plan.append((k, beta, i))
        plan.sort(); G = torch.zeros_like(M.Z, dtype=torch.float32); kc = 0
        for k, beta, i in plan:
            if k > kc: G += FQ[:, kc:k] @ AQ[:, kc:k].T; kc = k
            bk, al, sc, EX, a0, asn = info[i]; scf = sc.float(); uqf = uq.float()
            rem = G + beta * FQ[:, k:k + 1] @ AQ[:, k:k + 1].T + scf[:, None] * (AQ[:, :k] @ uqf[:k] + beta * uqf[k] * AQ[:, k])[None]
            Zp = (M.Z + sc[:, None] * C['Au'][None]); rnd[i].append(float((a0 - ang((Zp - rem.double()), t)).abs().mean()))
    for i, (bk, al, sc, EX, a0, asn) in enumerate(info): out[(bk, al)] = dict(AS_SING=asn, AS_fair=asn / float(np.mean(rnd[i])))
    return out

ang = E9['ang']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_ext', 'sing')))
    ap.add_argument('--cache', default=None); a0 = ap.parse_args(); fr = a0.feats_root
    a = argparse.Namespace(feats=f'{fr}/e3/ret/e3_feats', private=f'{fr}/e3/private', dev_concepts=E9['DEV_C'], trials=50, models=MODELS)
    cache = a0.cache or f'{fr}/r2_private/e9_holdout_cache.jsonl'; os.makedirs(os.path.dirname(cache), exist_ok=True)
    att = np.load(os.path.join(a0.sing_root, 'misc/embeddings/attribute_embeddings_broden_style_fp16.npz'), allow_pickle=True)
    names = [str(c) for c in att['classes']]; pi = [str(p) for p in att['prompts']].index('mean')
    TA = torch.nn.functional.normalize(torch.tensor(att['embeddings'][:, pi].astype(np.float64)), dim=-1)
    cc = np.load(os.path.join(a.private, 'clip_concepts.npz')); cn = [str(c) for c in cc['concepts']]
    S = {c: torch.tensor(cc['zscore'][:, cn.index(c)]).double() for c in HOLD_C}
    perm = np.random.default_rng(0).permutation(10000); tr = torch.tensor(perm[:5000]); te = torch.tensor(perm[5000:])
    Lidx = tr[torch.tensor(np.random.default_rng(0).choice(5000, 1000, replace=False))]
    alphas = np.exp(np.random.default_rng(1).uniform(math.log(0.01), math.log(1.0), size=(50, len(MODELS))))
    done = {}
    if os.path.exists(cache):
        for line in open(cache): r = json.loads(line); done[r['key']] = r['v']
    cf = open(cache, 'a'); cache_red = {}; need = set(CANDS); T = {'lin': {}, 'nl': {}}; FL = {}; RAW = {'lin': {}, 'nl': {}}; frac = {}; elapsed = {}
    for mode in ['lin', 'nl']:
        t_mode = time.time()
        for name in MODELS:
            t0 = time.time(); M = E9['Model'](name, a, TA, names, cc)
            jobs = [('full', 0.0)] + [('full', float(alphas[ti, MODELS.index(name)])) for ti in range(50)]
            if name in BIG and mode == 'lin':
                jobs += [((dd, j), al) for dd in DGRID for j in range(NSUB) for al in [0.0] + FLAT_A] + [('full', al) for al in FLAT_A]
            for c in HOLD_C:
                key = json.dumps(['r2hold', mode, name, c])
                if key not in done:
                    res = {}
                    asx = as_scores(M, c, mode, jobs, S[c])
                    for bk, al in jobs:
                        v = E9['evaluate'](M, c, mode, al, bk, S[c], tr, te, Lidx, need, cache_red); v.update(asx[(bk, al)]); res[json.dumps([bk if bk == 'full' else list(bk), al])] = v
                    done[key] = res; cf.write(json.dumps(dict(key=key, v=res)) + '\n'); cf.flush()
                res = {tuple([x if isinstance(x, str) else tuple(x) for x in [json.loads(k)[0]]] + [json.loads(k)[1]]): v for k, v in done[key].items()}
                diff = lambda bk, al: {k: res[(bk, al)][k] - res[(bk, 0.0)][k] for k in res[(bk, al)]}
                for ti in range(50):
                    al = float(alphas[ti, MODELS.index(name)]); T[mode][(ti, c, name)] = diff('full', al); RAW[mode][(ti, c, name)] = res[('full', al)]
                if name in BIG and mode == 'lin':
                    for al in FLAT_A:
                        for dd in DGRID:
                            v = [diff((dd, j), al) for j in range(NSUB)]; FL[(name, c, al, dd)] = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
                            frac[(name, c, dd)] = float(np.mean([float((M.B((dd, j)).T @ M.conc[c]['u']).norm()) for j in range(NSUB)]))
                        FL[(name, c, al, 1048)] = diff('full', al); frac[(name, c, 1048)] = 1.0
            elapsed[f'{mode}_{name}'] = time.time() - t0; print(f'# {mode} {name} {time.time()-t0:.0f}s', flush=True); del M
            if time.time() - t_mode > GUARD_S: print('# guard hit', flush=True); break
    def kend(mode, sc):
        per = {c: [] for c in HOLD_C}
        for ti in range(50):
            al = alphas[ti]
            for c in HOLD_C:
                src_ = RAW[mode] if sc == 'AS_SING' else T[mode]; per[c].append(float(kendalltau([src_[(ti, c, n)][sc] for n in MODELS], al).statistic))
        tm = np.mean([per[c] for c in HOLD_C], 0); b = np.random.default_rng(0); boot = [tm[b.integers(0, 50, 50)].mean() for _ in range(1000)]
        return dict(mean=float(tm.mean()), ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], per_concept={c: float(np.mean(per[c])) for c in HOLD_C}, per_trial=tm.tolist())
    def flat(sc, revised):
        o = {}
        for al in FLAT_A:
            rat = {}
            for n in BIG:
                rs = []
                for c in HOLD_C:
                    v = [FL[(n, c, al, dd)][sc] / ((al * frac[(n, c, dd)]) if revised else 1.0) for dd in DGRID + [1048]]
                    rs.append(max(v) / min(v) if min(v) > 0 else float('inf'))
                rat[n] = float(np.mean(rs))
            o[str(al)] = dict(ratio=rat, n_flat=sum(x <= 1.5 for x in rat.values()))
        o['flat'] = all(o[str(al)]['n_flat'] >= 3 for al in FLAT_A); return o
    out = {}
    for sc in ALLC:
        kl, kn = kend('lin', sc), kend('nl', sc); f0, f1 = flat(sc, False), flat(sc, True)
        out[sc] = dict(kendall=kl, kendall_nonlinear=kn, flatness=f0, flatness_revised=f1,
                       pass_as_registered=kl['mean'] >= 0.8 and kl['ci95'][0] >= 0.7 and f0['flat'] and kn['mean'] >= 0.6,
                       pass_with_revised_flatness=kl['mean'] >= 0.8 and kl['ci95'][0] >= 0.7 and f1['flat'] and kn['mean'] >= 0.6)
    pm = {}
    for mode in ['lin', 'nl']:
        d = np.array((kend(mode, 'M1')['per_trial'])) - np.array(kend(mode, 'AS_fair')['per_trial']); b = np.random.default_rng(0)
        bs = [d[b.integers(0, 50, 50)].mean() for _ in range(1000)]; pm[mode] = dict(mean_diff=float(d.mean()), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
    res = dict(label='post hoc; E9 remains KILL as pre-registered', candidates=out, M1_minus_ASfair_paired=pm, elapsed_s=elapsed)
    json.dump(res, open(os.path.join(R, 'results/fix10_e9_holdout.json'), 'w'), indent=1)
    for sc, v in out.items():
        print(f"{sc:8s} lin {v['kendall']['mean']:+.3f} [{v['kendall']['ci95'][0]:+.3f},{v['kendall']['ci95'][1]:+.3f}] nl {v['kendall_nonlinear']['mean']:+.3f} "
              f"flat {v['flatness']['0.1']['n_flat']}/{v['flatness']['1.0']['n_flat']} rev {v['flatness_revised']['0.1']['n_flat']}/{v['flatness_revised']['1.0']['n_flat']} "
              f"pass {v['pass_as_registered']} pass_rev {v['pass_with_revised_flatness']}")
    print('M1-ASfair', pm)

if __name__ == '__main__': main()
