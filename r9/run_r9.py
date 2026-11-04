"""Round 9 runner: controls, the three-step check on every score, and the accuracy-matched head swap.

Inputs: one npz per model with F (N, d) features, y (N,) labels, W (K, d), b or bW (K,) head; optionally A (t, d) and
Ab (t,) (a SING translator into CLIP text space) and, through --text-emb, the class prompt embeddings.
Fit images = the first half of each class's images in file order; evaluation images = the rest.

  python r9/run_r9.py --feats-dirs DIR [DIR ...] --models-file r9/models13.txt --out r9/results --work _local/r9/work
  python r9/run_r9.py --synthetic --out /some/dir      (synthetic smoke run)
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sanity_check_general as G
import scores_r9 as S
import controls_r9 as C

HERE = os.path.dirname(os.path.abspath(__file__))
REAL_DRAWS = 10
CTRL_DRAWS = 3


# ------------------------------------------------------------------ loading
def load_npz(path, T=None, dtype=np.float32):
    d = np.load(path)
    m = dict(F=d['F'].astype(dtype), y=d['y'].astype(int), W=d['W'].astype(np.float64),
             b=(d['b'] if 'b' in d.files else d['bW']).astype(np.float64))
    if 'A' in d.files and T is not None:
        m['A'] = d['A'].astype(np.float64); m['Ab'] = d['Ab'].astype(np.float64); m['T'] = T
    return m


def class_text(path):
    e = np.load(path, allow_pickle=True)['embeddings'].astype(np.float64)
    e = e / np.linalg.norm(e, axis=-1, keepdims=True); t = e.mean(1); return t / np.linalg.norm(t, axis=1, keepdims=True)


def find(dirs, name):
    for r in dirs:
        for f in (f'{name}_full.npz', f'{name}.npz'):
            p = os.path.join(r, f)
            if os.path.exists(p): return p
    raise FileNotFoundError(name)


# ------------------------------------------------------------------ representations
def prepare(raw):
    """Null family: features in the basis [row(W) | null(W)] (null = columns r:). ViM family: features about u in the
    eigenbasis of the uncentred fit-image covariance (residual = columns D:)."""
    F, y, W, b = raw['F'], raw['y'], raw['W'], raw['b']; dt = F.dtype
    fit = G.split_first_half(y); K, d = W.shape
    _, sv, vt = np.linalg.svd(W, full_matrices=True); r = int((sv > G.TOL).sum()); Q = vt.T
    Z = (F.astype(np.float64) @ Q).astype(dt); Wq = W @ Q; Wq[:, r:] = 0.0
    mn = dict(F=Z, y=y, W=Wq, b=b, fit=fit, _null_from=r)
    if 'A' in raw:
        mn.update(A=raw['A'] @ Q, Ab=raw['Ab'], T=raw['T'])
    mn['_L'] = Z @ Wq.T.astype(dt) + b.astype(dt)
    oq = S.head_offset(Wq, b)[:r]
    mn['_rown2'] = np.sum(Z[:, :r].astype(np.float64) ** 2, 1); mn['_rowo2'] = np.sum((Z[:, :r].astype(np.float64) + oq) ** 2, 1)
    if 'A' in mn:
        ev = ~fit; Zrow = S.as_row(mn)[ev]; mn['_as_row'] = (Zrow, S._ang(Zrow, mn['T'][y[ev]])); mn['_A_null'] = mn['A'][:, r:].astype(dt)
    u, U = S.vim_fit(dict(F=F.astype(np.float64), W=W, b=b, fit=fit)); D = S.vim_dim(d)
    Zv = ((F.astype(np.float64) - u) @ U).astype(dt)
    mv = dict(F=Zv, y=y, W=W @ U, b=W @ u + b, fit=fit, _vim_frozen=D)
    mv['_pn2'] = np.sum(Zv[:, :D].astype(np.float64) ** 2, 1); mv['_Lp'] = Zv[:, :D] @ mv['W'][:, :D].T.astype(dt) + mv['b'].astype(dt)
    mv['_WR'] = mv['W'][:, D:].astype(dt)
    info = dict(d=d, K=K, rank=r, null_dim=d - r, vim_D=D, vim_res_dim=d - D, N=len(y), n_fit=int(fit.sum()))
    return mn, mv, info


def generic(raw):
    """model dict in the original coordinates for head refits (null basis recomputed for every head)."""
    m = dict(raw); m['fit'] = G.split_first_half(raw['y']); m['F'] = raw['F'].astype(np.float64); return m


def drop_text(m):
    return {k: v for k, v in m.items() if k not in ('A', 'Ab', 'T', '_as_row', '_A_null')}


# ------------------------------------------------------------------ controls for one model
def controls_one(i, n_models, mn, mv, streams=None, reps_scale=1.0, log=print):
    out = {}; K = mn['W'].shape[0]
    fam = dict(null=(mn, slice(mn['_null_from'], None), S.null_family), vim=(mv, slice(mv['_vim_frozen'], None), S.vim_family))
    base = {}
    for f, (m, sl, _) in fam.items():
        Cr = G.coords(m['F'], sl); base[f] = (Cr, *G.moment_sqrt(Cr))
    for name, (sid, f, kind, proto, reps) in C.STREAMS.items():
        if streams and name not in streams: continue
        m, sl, fn = fam[f]
        if f == 'null' and proto == 'text' and 'A' not in m: continue
        if f == 'null' and proto == 'rand': m = drop_text(m)
        Cr, mu, L = base[f]; k = Cr.shape[1]
        U = None
        if proto == 'rand':
            U = C.random_prototypes(K, k, C.model_rng(sid, 99, 0, i)).astype(Cr.dtype)
        elif proto == 'text':
            U = C.text_prototypes(m['A'][:, sl], m['T']).astype(Cr.dtype)
        levels = [0] if kind == 'NC' else range(len(C.A_LEVELS)); R = max(2, int(round(reps * reps_scale)))
        res = {}; t0 = time.time()
        for li in levels:
            for rep in range(R):
                p = C.dataset_params(sid, kind, li, rep, n_models); rng = C.model_rng(sid, li, rep, i)
                Cc = C.make_control(Cr, m['y'], kind, p['e'][i], p['o'][i], p['a'][i], U, rng, mu, L)
                key = '_N' if f == 'null' else '_R'
                mc = dict(m); mc[key] = Cc; real = fn(mc)
                mu2, L2 = G.moment_sqrt(Cc); sur = []
                for _ in range(CTRL_DRAWS):
                    ms = dict(m); ms[key] = G.gaussian_draw(len(Cc), mu2, L2, rng); sur.append(fn(ms))
                res[f'{li}_{rep}'] = {q: [real[q], float(np.mean([s[q] for s in sur]))] for q in real}
        out[name] = res; log(f'  {name}: {len(res)} datasets, {time.time() - t0:.0f}s')
    return out


# ------------------------------------------------------------------ real features for one model
def real_one(i, raw, mn, mv, heads=True, log=print):
    t0 = time.time()
    o = dict(null=G.variants(mn, S.null_family, lambda m: (slice(m['_null_from'], None), None), REAL_DRAWS,
                             np.random.default_rng([C.SEED, C.STREAM_REAL, 0, 0, i])),
             vim=G.variants(mv, S.vim_family, lambda m: (slice(m['_vim_frozen'], None), None), REAL_DRAWS,
                            np.random.default_rng([C.SEED, C.STREAM_REAL, 1, 0, i])))
    o['decomp'] = G.rho_decomposition(mn['F'], mn['y'], slice(mn['_null_from'], None))
    log(f'  variants {time.time() - t0:.0f}s')
    if heads:
        g = generic(raw); H = G.fit_heads(g, seed=int(C.SEED + 1000 * C.STREAM_HEAD + i)); o['heads'] = {}
        for k, h in H.items():
            if k == 'released':
                sc = {**S.null_family(mn), **S.vim_family(mv)}
            else:
                mh = G.with_head(g, h['W'], h['b']); mh['Vn'] = G.null_basis(h['W']); sc = S.all_scores(mh)
            o['heads'][k] = dict(top1=h['top1'], matched=h['matched'], score=sc, **{q: h[q] for q in ('eps', 'lam', 'shrinkage') if q in h})
        log(f'  heads {time.time() - t0:.0f}s ' + ' '.join(f"{k}:{v['top1']:.3f}" for k, v in o['heads'].items()))
    return o


# ------------------------------------------------------------------ aggregation
def verdict_of(real, sur):
    return G.verdict_step2(real, sur, B=2000, seed=C.BOOT_SEED)['verdict']


def aggregate_controls(names, per):
    """per[name][stream][dataset][score] = [real, surrogate]. Flag rates, power curves and manipulation checks."""
    n = len(names); out = {}
    for sname, (sid, f, kind, proto, reps) in C.STREAMS.items():
        have = [m for m in names if sname in per[m]]
        if len(have) < n: continue
        keys = sorted(per[names[0]][sname]); scores = list(per[names[0]][sname][keys[0]])
        res = {}
        for q in scores:
            byl = {}
            for key in keys:
                li, rep = map(int, key.split('_'))
                real = np.array([per[m][sname][key][q][0] for m in names]); sur = np.array([per[m][sname][key][q][1] for m in names])
                d = byl.setdefault(li, dict(flag=[], content=[], tau_a=[], tau_e=[], tau_o=[]))
                v = verdict_of(real, sur); d['flag'].append(v == 'ENERGY-DRIVEN'); d['content'].append(v == 'CONTENT-READING')
                p = C.dataset_params(sid, kind, li, rep, n)
                if kind != 'NC' and li > 0: d['tau_a'].append(float(G.ktau(real, p['a'])))
                d['tau_e'].append(float(G.ktau(real, p['e'])) if np.ptp(p['e']) > 0 else float('nan'))
                d['tau_o'].append(float(G.ktau(real, p['o'])) if np.ptp(p['o']) > 0 else float('nan'))
            res[q] = {str(C.A_LEVELS[li] if kind != 'NC' else 'NC'): dict(
                n_datasets=len(v['flag']), flag_rate=float(np.mean(v['flag'])), not_flag_rate=float(1 - np.mean(v['flag'])),
                flag_rate_ci95=wilson(sum(v['flag']), len(v['flag'])), content_rate=float(np.mean(v['content'])),
                mean_tau_score_a=float(np.mean(v['tau_a'])) if v['tau_a'] else None,
                mean_tau_score_e=float(np.nanmean(v['tau_e'])) if np.isfinite(v['tau_e']).any() else None,
                mean_tau_score_o=float(np.nanmean(v['tau_o'])) if np.isfinite(v['tau_o']).any() else None) for li, v in sorted(byl.items())}
        out[sname] = res
    return out


def wilson(k, n, z=1.96):
    if n == 0: return [float('nan')] * 2
    p = k / n; c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(c - h), float(c + h)]


def kill_checks(ctrl, has_text=True, has_vim=True):
    """Pre-registered control criteria (PREREG_R9.md section 5)."""
    g = lambda s, q, lv: ctrl.get(s, {}).get(q, {}).get(lv)
    a = str(C.A_KILL); k = {}

    def family(nc_stream, pc_stream, e_ref, c_ref):
        nc = g(nc_stream, e_ref, 'NC'); pc = g(pc_stream, c_ref, a)
        r = dict(energy_ref=e_ref, content_ref=c_ref, nc_flag_rate=nc['flag_rate'] if nc else None,
                 pc_not_flag_rate=pc['not_flag_rate'] if pc else None, pc_manipulation_tau=pc['mean_tau_score_a'] if pc else None)
        r['fired'] = bool(not nc or not pc or nc['flag_rate'] < 0.8 or pc['not_flag_rate'] < 0.8 or pc['mean_tau_score_a'] < 0.6)
        return r
    k['null_family'] = family('NC_null', 'PCm_null_rand', 'null_norm', 'core_null')
    k['K_ctrl_global_fired'] = k['null_family']['fired']
    if has_vim:
        k['vim_family'] = family('NC_vim', 'PCm_vim_rand', 'res_norm', 'core_res')
    per = {}
    for fam, nc_s, qs in (('null', 'NC_null', S.NULL_SCORES), ('vim', 'NC_vim', S.VIM_SCORES)):
        for q in qs:
            r = g(nc_s, q, 'NC')
            if not r: continue
            per[q] = dict(nc_flag_rate=r['flag_rate'], nc_false_content_rate=r['content_rate'], low_power=bool(r['flag_rate'] < 0.8),
                          withheld=bool(r['content_rate'] > 0.2 or (fam == 'vim' and k.get('vim_family', {}).get('fired', False))))
    if has_text and 'as_sing' in per:
        r = g('PCm_null_text', 'as_sing', a)
        per['as_sing'].update(pc_text_not_flag_rate=r['not_flag_rate'] if r else None,
                              pc_text_manipulation_tau=r['mean_tau_score_a'] if r else None,
                              pc_validated=bool(r and r['not_flag_rate'] >= 0.8 and abs(r['mean_tau_score_a']) >= 0.6))
    k['per_score'] = per
    return k


def aggregate_real(names, per, sing=None, B=2000):
    out = {}; seed = C.BOOT_SEED
    for f in ('null', 'vim'):
        for q in per[names[0]][f]:
            col = lambda key: np.array([per[m][f][q][key] for m in names])
            o = dict(values={m: per[m][f][q]['real'] for m in names},
                     step1=dict(tau_offset_only=G.tau_ci(col('real'), col('offset_only'), B, seed),
                                tau_centred=G.tau_ci(col('real'), col('centred'), B, seed)),
                     step2=G.verdict_step2(col('real'), col('surrogate'), B, seed),
                     step2_centred=G.tau_ci(col('real'), col('surrogate_centred'), B, seed))
            lo = o['step1']['tau_offset_only']['bca'][0]; o['step1']['offset_dominated'] = bool(np.isfinite(lo) and lo > 0.5)
            if sing is not None:
                o['tau_vs_sing_order'] = G.tau_ci(col('real'), sing, B, seed)
                o['dtau_real_minus_surrogate_vs_sing'] = G.paired_ci(col('real'), col('surrogate'), sing, B, seed)
                lo9, hi9 = o['dtau_real_minus_surrogate_vs_sing']['bca90']
                o['sing_order_equivalent'] = bool(np.isfinite(lo9) and lo9 > -0.1 and hi9 < 0.1)
            out[q] = o
    return out


def aggregate_heads(names, per, B=2000, min_models=8):
    out = {}; kinds = [k for k in ('NCM', 'LDA', 'ridge', 'perturbed') if any(k in per[m].get('heads', {}) for m in names)]
    if not all('heads' in per[m] for m in names): return out
    out['top1'] = {m: {k: v['top1'] for k, v in per[m]['heads'].items()} for m in names}
    out['matched'] = {m: {k: v['matched'] for k, v in per[m]['heads'].items()} for m in names}
    scores = list(per[names[0]]['heads']['released']['score'])
    for q in scores:
        sq = {}
        for k in kinds:
            ms = [m for m in names if k in per[m]['heads']]; mm = [m for m in ms if per[m]['heads'][k]['matched']]
            x = lambda L: np.array([per[m]['heads']['released']['score'][q] for m in L]); z = lambda L: np.array([per[m]['heads'][k]['score'][q] for m in L])
            d = dict(n=len(ms), n_matched=len(mm), tau_all=G.tau_ci(x(ms), z(ms), B, C.BOOT_SEED) if len(ms) >= 4 else None)
            if len(mm) >= min_models:
                t = G.tau_ci(x(mm), z(mm), B, C.BOOT_SEED); d['tau_matched'] = t
                d['reading'] = 'HEAD-DEPENDENT' if t['tau'] <= 0.5 else ('HEAD-STABLE' if t['tau'] >= 0.8 else 'INTERMEDIATE')
            else:
                d['tau_matched'] = None; d['reading'] = f'NOT ESTIMABLE (fewer than {min_models} matched backbones)'
            sq[k] = d
        out[q] = sq
    return out


def final_verdicts(real, kill):
    v = {}
    for q, o in real.items():
        w = kill['per_score'].get(q, {})
        if q in ('res_norm', 'core_res', 'null_norm', 'core_null'):
            role = 'reference'
        else:
            role = 'score'
        if kill['K_ctrl_global_fired']:
            v[q] = dict(verdict='STOPPED (K-ctrl fired)', role=role)
        elif w.get('withheld'):
            v[q] = dict(verdict='WITHHELD (controls failed for this score)', role=role, step2=o['step2']['verdict'])
        else:
            s = o['step2']['verdict']
            if q == 'as_sing' and not w.get('pc_validated', True) and s == 'CONTENT-READING':
                s = 'MIXED (content reading not validated by PC-text)'
            if s == 'MIXED' and w.get('low_power'):
                s = 'MIXED (low power: NC flag rate below 0.8 for this score)'
            v[q] = dict(verdict=s, role=role, order_equivalent=o['step2']['order_equivalent'],
                        tau_E=o['step2']['tau_E']['tau'], tau_E_bca95=o['step2']['tau_E']['bca'])
    return v


# ------------------------------------------------------------------ synthetic models (tests, smoke runs)
def synthetic_models(n=8, d=96, K=12, per_class=10, content=None, energy=None, seed=0, translator=True):
    """Toy models: the head reads class means in a K-dim row space; the null part has an offset, isotropic noise of
    energy energy[i] and class content of strength content[i] (0..1)."""
    r = np.random.default_rng(seed); content = np.zeros(n) if content is None else np.asarray(content, float)
    energy = np.linspace(0.5, 2, n) if energy is None else np.asarray(energy, float); out = {}; t = 16
    T = r.standard_normal((K, t)); T /= np.linalg.norm(T, axis=1, keepdims=True)
    for i in range(n):
        Qd, _ = np.linalg.qr(r.standard_normal((d, d))); Vr, Vn = Qd[:, :K], Qd[:, K:]
        y = np.repeat(np.arange(K), per_class); N = len(y); k = d - K
        Mr = 3 * r.standard_normal((K, K)); Mn = r.standard_normal((K, k)); Mn /= np.linalg.norm(Mn, axis=1, keepdims=True)
        off = r.standard_normal(k); off *= 0.5 / np.linalg.norm(off)
        nul = off + np.sqrt(1 - content[i] ** 2) * r.standard_normal((N, k)) / np.sqrt(k) + content[i] * Mn[y]
        F = (Mr[y] + r.standard_normal((N, K))) @ Vr.T + np.sqrt(energy[i]) * nul @ Vn.T
        m = dict(F=F.astype(np.float32), y=y, W=Mr @ Vr.T, b=-0.5 * np.sum(Mr ** 2, 1))
        if translator:
            m['A'] = r.standard_normal((t, d)) / np.sqrt(d); m['Ab'] = 0.1 * r.standard_normal(t); m['T'] = T
        out[f'm{i}'] = m
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats-dirs', nargs='*', default=[]); ap.add_argument('--models-file', default=os.path.join(HERE, 'models13.txt'))
    ap.add_argument('--text-emb', default=None); ap.add_argument('--sing-order', default=os.path.join(HERE, 'sing_order.csv'))
    ap.add_argument('--out', required=True); ap.add_argument('--work', default=None); ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--reps-scale', type=float, default=1.0); ap.add_argument('--no-heads', action='store_true')
    ap.add_argument('--tag', default='r9'); ap.add_argument('--syn-n', type=int, default=8); ap.add_argument('--syn-d', type=int, default=96)
    ap.add_argument('--syn-K', type=int, default=12)
    ap.add_argument('--only', nargs='*', help='compute only these models (parallel workers); model index and count stay those of the full list')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); work = a.work or os.path.join(a.out, 'work'); os.makedirs(work, exist_ok=True)
    T = class_text(a.text_emb) if a.text_emb else None
    if a.synthetic:
        syn = synthetic_models(n=a.syn_n, d=a.syn_d, K=a.syn_K); names = list(syn); get = lambda n: syn[n]; sing = None
    else:
        names = [l.split()[0] for l in open(a.models_file) if l.strip() and not l.startswith('#')]
        get = lambda n: load_npz(find(a.feats_dirs, n), T)
        sing = None
        if os.path.exists(a.sing_order) and a.text_emb:
            so = {l.split(',')[0]: float(l.split(',')[1]) for l in open(a.sing_order).read().split('\n')[1:] if l.strip()}
            if all(n in so for n in names): sing = -np.array([so[n] for n in names])   # higher = more leakage-free in SING's reading
    per_c, per_r, info = {}, {}, {}
    for i, n in enumerate(names):
        pc, pr = os.path.join(work, f'{n}_controls.json'), os.path.join(work, f'{n}_real.json')
        if os.path.exists(pc) and os.path.exists(pr):
            per_c[n] = json.load(open(pc)); per_r[n] = json.load(open(pr)); info[n] = per_r[n].pop('_info'); continue
        if a.only is not None and n not in a.only:
            continue
        t0 = time.time(); raw = get(n); mn, mv, info[n] = prepare(raw); print(n, info[n], flush=True)
        per_r[n] = real_one(i, raw, mn, mv, heads=not a.no_heads, log=lambda s: print(s, flush=True))
        per_c[n] = controls_one(i, len(names), mn, mv, reps_scale=a.reps_scale, log=lambda s: print(s, flush=True))
        json.dump(per_c[n], open(pc, 'w')); json.dump({**per_r[n], '_info': info[n]}, open(pr, 'w'))
        print(n, f'done {time.time() - t0:.0f}s', flush=True); del raw, mn, mv
    if len(per_r) < len(names):
        print('partial run: missing', [n for n in names if n not in per_r], '; no aggregate written', flush=True); return
    has_text = all('as_sing' in per_r[n]['null'] for n in names)
    ctrl = aggregate_controls(names, per_c); kill = kill_checks(ctrl, has_text=has_text)
    real = aggregate_real(names, per_r, sing); heads = aggregate_heads(names, per_r)
    dump = lambda f, o: json.dump(o, open(os.path.join(a.out, f), 'w'), indent=1, default=float)
    dump(f'{a.tag}_controls.json', dict(models=names, streams={k: v[:4] + (v[4],) for k, v in C.STREAMS.items()}, results=ctrl, kill=kill))
    dump(f'{a.tag}_real_checks.json', dict(models=names, info=info, results=real, verdicts=final_verdicts(real, kill), kill=kill))
    dump(f'{a.tag}_head_swap.json', dict(models=names, results=heads))
    dump(f'{a.tag}_per_model.json', dict(models=names, info=info, per_model={n: {k: v for k, v in per_r[n].items()} for n in names}))
    print(json.dumps(dict(kill=kill, verdicts=final_verdicts(real, kill)), indent=1, default=float))


if __name__ == '__main__':
    main()
