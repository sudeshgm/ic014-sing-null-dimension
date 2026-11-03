"""PREREG_R8: theta_sem on the 13 models, all gates, confirmatory tau, descriptives. CPU only.
Reads cached features (F, y, W, A, Ab) via r3/retrain_translators.find; SING class texts from _ext/sing.
Per-model results are cached in r8/results/permodel/<model>.json (resume-safe); verdicts in r8/results/R8_verdicts.json.
Requires r8/results/K0_coverage.json (K0 decides the resampling method). Usage:
  python r8/run_r8.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(H)
sys.path.insert(0, H); sys.path.insert(0, os.path.join(R, 'r3')); sys.path.insert(0, os.path.join(R, 'r4'))
import core_r8 as c
from common import load_models, tau_ci, ktau, S11_DROP, RHO0
from common4 import partial_ci
from retrain_translators import find

K = 1000; BASES = ['resnet', 'dinov2', 'resnext']; B = 2000
SEED = dict(fold=100800, fold2=100801, offset=100802, rot=100803, derange=100804, plant=100805, boot=100806, gauss=100807,
            perm=100808, jinn=100809, heads=100810)
RES = os.path.join(H, 'results'); PM = os.path.join(RES, 'permodel')


def svd_split(W):
    _, sv, vt = np.linalg.svd(W.astype(np.float64), full_matrices=True); return sv, vt


def bases_at(sv, vt, tol):
    r = int((sv > tol).sum()); return vt[r:].T, vt[:r].T


def ang(Zc, t):
    Zn = Zc / np.linalg.norm(Zc, axis=1, keepdims=True); return np.degrees(np.arccos(np.clip(np.sum(Zn * t, 1), -1, 1)))


def AS_of(F, Vn, A, Ab, tc):
    Frow = F - (F @ Vn) @ Vn.T; return float(np.mean(np.abs(ang(F @ A.T + Ab, tc) - ang(Frow @ A.T + Ab, tc))))


def rho_parts(Z, y, Etot):
    M = c.class_means(Z, y, K); zb = Z.mean(0)
    return dict(rho=float(np.mean(np.sum(Z ** 2, 1))) / Etot, rho_offset=float(zb @ zb) / Etot,
                rho_between=float(np.mean(np.sum((M[y] - zb) ** 2, 1))) / Etot, rho_within=float(np.mean(np.sum((Z - M[y]) ** 2, 1))) / Etot)


def theta_batch(cms, cs, T, folds, draws, method):
    """Point theta_sem, CI and SE for several null blocks against one row block, sharing the resampling draws."""
    Gs = [m.G for m in cms] + [cs.G]; P = c.S_multi(Gs, T, *folds); Rb = c.resampled_S(Gs, T, draws)
    out = []
    for j in range(len(cms)):
        th = c.theta_from_S(P[j], P[-1]); tb = np.array([c.theta_from_S(a, b) for a, b in zip(Rb[:, j], Rb[:, -1])])
        ci, se = c.ci_from(th, tb, method); sci, sse = c.ci_from(P[j], Rb[:, j], method)
        out.append(dict(theta_sem=float(th), S_Z=float(P[j]), S_S=float(P[-1]), ci=ci, se=se, S_Z_ci=sci, S_Z_se=sse,
                        rho=None, n_valid=int(np.isfinite(tb).sum())))
    return out, Rb


def gauss_pieces(Zc_all):
    mu = Zc_all.mean(0); X = Zc_all - mu; Sig = X.T @ X / len(X); ev, U = np.linalg.eigh(Sig)
    return mu, U * np.sqrt(np.clip(ev, 0, None))


def per_model(name, fr, T, TCn, folds, folds2, draws, method, targets, ctx):
    t0 = time.time(); d = np.load(find(fr, name)); F = d['F'].astype(np.float64); y = d['y'].astype(int); W = d['W'].astype(np.float64)
    A = d['A'].astype(np.float64); Ab = d['Ab'].astype(np.float64); hA = c.halves(y, K); Etot = float(np.mean(np.sum(F ** 2, 1)))
    sv, vt = svd_split(W); Vn, Vr = bases_at(sv, vt, 1e-5); Z, S = F @ Vn, F @ Vr
    cz, cs = c.CM.from_images(Z, y, K, hA), c.CM.from_images(S, y, K, hA); out = dict(model=name, d_null=int(Vn.shape[1]))
    out.update(rho_parts(Z, y, Etot))
    # content-free surrogates at the model's own energy (G-floor cells)
    mu, Ls = gauss_pieces(Z); g = c.gaussian_surrogate_cm(mu, Ls, K, 10, np.random.default_rng(SEED['gauss']))
    pz = c.CM.from_images(Z[np.random.default_rng(SEED['perm']).permutation(len(Z))], y, K, hA)
    # shadow / innovation
    sh, inn, U = c.shadow_innovation(Z, S, y, K, hA, *folds)
    res, _ = theta_batch([cz, g, pz, sh, inn], cs, T, folds, draws, method)
    out['theta'] = res[0]; out['surr_own'] = dict(gauss=res[1], perm=res[2]); out['shadow'] = res[3]; out['innovation'] = res[4]
    # theta_cls with class bootstrap over all classes (descriptive)
    rng = np.random.default_rng(SEED['boot']); tcb = [c.theta_cls(cz, cs, rng.integers(0, K, K)) for _ in range(B)]
    out['theta_cls'] = dict(point=float(c.theta_cls(cz, cs)), ci=[float(np.nanpercentile(tcb, 2.5)), float(np.nanpercentile(tcb, 97.5))])
    out['Eb_Z'], out['Eb_S'] = c.Eb(cz) / Etot, c.Eb(cs) / Etot
    out['composition_SZ_over_EbZ'] = float(res[0]['S_Z'] / c.Eb(cz)) if c.Eb(cz) > 0 else None
    # sensitivity fold, tolerances
    out['theta_fold2'] = float(c.theta_from_S(*c.S_multi([cz.G, cs.G], T, *folds2)))
    out['theta_tol'] = {}
    for tol in (1e-4, 1e-6):
        Vn2, Vr2 = bases_at(sv, vt, tol); a, b = c.CM.from_images(F @ Vn2, y, K, hA), c.CM.from_images(F @ Vr2, y, K, hA)
        out['theta_tol'][f'{tol:g}'] = dict(theta=float(c.theta_from_S(*c.S_multi([a.G, b.G], T, *folds))), d_null=int(Vn2.shape[1]))
    # exact invariances (G-offset / G-scale / G-rotation)
    th = res[0]['theta_sem']; r2 = np.random.default_rng(SEED['offset']); v = r2.standard_normal(Z.shape[1]); v *= 10 * np.sqrt(cz.En) / np.linalg.norm(v)
    rel = lambda a, b: float(abs(c.theta_from_S(*c.S_multi([a.G, b.G], T, *folds)) - th) / abs(th))
    r3 = np.random.default_rng(SEED['rot']); Q1 = np.linalg.qr(r3.standard_normal((Z.shape[1],) * 2))[0]; Q2 = np.linalg.qr(r3.standard_normal((S.shape[1],) * 2))[0]
    out['invariance_rel_err'] = dict(offset=rel(c.CM.from_images(Z + v, y, K, hA), cs), scale=rel(cz.scaled(7.3), cs.scaled(7.3)),
                                     rotation=rel(c.CM(cz.M @ Q1, cz.MA @ Q1, cz.MB @ Q1, cz.En), c.CM(cs.M @ Q2, cs.MA @ Q2, cs.MB @ Q2, cs.En)))
    del Q1, Q2
    # descriptives
    out['self_duality'] = c.self_duality(W, c.class_means(F, y, K), Vr)
    out['PR_within'] = dict(null=c.participation_ratio(c.within_dev(Z, y, K)), row=c.participation_ratio(c.within_dev(S, y, K)))
    rj = np.random.default_rng(SEED['jinn'])
    out['J'] = dict(null=c.fisher_J(Z, y, K, hA, *folds, rng=rj), row=c.fisher_J(S, y, K, hA, *folds, rng=rj), innovation=c.fisher_J(U, y, K, hA, *folds, rng=rj))
    del U
    if name in BASES: out['base'] = base_worlds(name, F, Z, S, y, hA, cz, cs, Vn, A, Ab, T, TCn, folds, draws, method, targets, mu, Ls, W)
    out['secs'] = time.time() - t0; return out


def base_worlds(name, F, Z, S, y, hA, cz, cs, Vn, A, Ab, T, TCn, folds, draws, method, targets, mu, Ls, W):
    wp = c.WorldPieces(Z, y, K, hA, Er=cs.En); o = {}
    Cp = wp.world("C'", targets)
    g = c.gaussian_surrogate_cm(mu, Ls, K, 10, np.random.default_rng(SEED['gauss']))
    pz = c.CM.from_images(Z[np.random.default_rng(SEED['perm']).permutation(len(Z))], y, K, hA)
    sg = [c.scale_to_rho(g, cs.En, r) for r in targets]; sp = [c.scale_to_rho(pz, cs.En, r) for r in targets]
    # plants on C' row 6
    r6 = Cp[6]['cm']; rp = np.random.default_rng(SEED['plant']); P = rp.standard_normal((Z.shape[1], T.shape[1]))
    sem = T @ P.T; arb = rp.standard_normal((K, Z.shape[1])); arb -= arb.mean(0); planted = {}
    for nm, U in (('semantic', sem), ('arbitrary', arb)):
        U = U * np.sqrt(0.3 * r6.En / float(np.mean(np.sum(U ** 2, 1)))); pc = r6.plus_class_code(U); planted[nm] = pc.scaled(np.sqrt(r6.En / pc.En))
    blocks = [r['cm'] for r in Cp] + sg + sp + [planted['semantic'], planted['arbitrary']]
    res, Rb = theta_batch(blocks, cs, T, folds, draws, method)
    o["C'"] = [dict(rho=r['rho'], rho_target=r['rho_target'], **{k: v for k, v in x.items() if k != 'rho'}) for r, x in zip(Cp, res[:12])]
    o['surr_gauss'] = [dict(rho_target=r, **{k: v for k, v in x.items() if k != 'rho'}) for r, x in zip(targets, res[12:24])]
    o['surr_perm'] = [dict(rho_target=r, **{k: v for k, v in x.items() if k != 'rho'}) for r, x in zip(targets, res[24:36])]
    # paired plant differences (same draws), CI by the K0 method
    th6 = lambda j: np.array([c.theta_from_S(a, b) for a, b in zip(Rb[:, j], Rb[:, -1])])
    base6 = th6(6); o['plants'] = {}
    for j, nm in ((36, 'semantic'), (37, 'arbitrary')):
        dlt = res[j]['theta_sem'] - res[6]['theta_sem']; ci, se = c.ci_from(dlt, th6(j) - base6, method)
        o['plants'][nm] = dict(delta=float(dlt), ci=ci, se=se, theta=res[j]['theta_sem'])
    o['plants']['se_unplanted'] = res[6]['se']
    # old worlds C, M, D (non-gating, point values) and the offset-only world
    pt = lambda cm: float(c.theta_from_S(*c.S_multi([cm.G, cs.G], T, *folds)))
    for Wn in ('C', 'M', 'D'):
        o[Wn] = [dict(L=r['L'], rho=r['rho'], theta_sem=pt(r['cm']), theta_cls=float(c.theta_cls(r['cm'], cs))) for r in wp.world(Wn, targets)]
    for r, x in zip(o["C'"], Cp): r['theta_cls'] = float(c.theta_cls(x['cm'], cs))
    o['offset'] = [dict(rho=r['rho'], c=r['c'], theta_sem=pt(r['cm'])) for r in wp.offset_world(Cp[0]['cm'], targets)]
    # world B (real data rescaled), closed form
    Sz, Ss = c.S_multi([cz.G, cs.G], T, *folds); o['B'] = []
    for r in targets:
        bc = c.scale_to_rho(cz, cs.En, r); beta2 = bc.En / cz.En; t = pt(bc); cf = beta2 * Sz / (beta2 * Sz + Ss)
        o['B'].append(dict(rho=r, theta_sem=t, closed_form=float(cf), rel_err=float(abs(t - cf) / abs(cf))))
    # derangements on real data and C' row 6
    o['derange'] = {}
    for lab, cm in (('real', cz), ("C'6", r6)):
        rd = np.random.default_rng(SEED['derange']); v = [pt(cm.permuted(c.derangement(K, rd))) for _ in range(20)]
        o['derange'][lab] = dict(values=v, mean=float(np.mean(v)), sd=float(np.std(v, ddof=1)), theta_undeformed=pt(cm))
    o['heads'] = head_swap(F, y, hA, A, Ab, T, TCn, folds, draws, method, W, Vn)
    return o


def head_swap(F, y, hA, A, Ab, T, TCn, folds, draws, method, W, Vn):
    """NCM and shrinkage-LDA heads fit on image half A, top-1 on half B; null split, rho, AS (released translator), theta_sem."""
    from sklearn.covariance import ledoit_wolf
    tc = TCn[y]; Mh = c.class_means(F, y, K, hA); D = F[hA] - Mh[y[hA]]; Sig, _ = ledoit_wolf(D, assume_centered=True)
    Wl = np.linalg.solve(Sig, Mh.T).T; heads = dict(released=(W, None), NCM=(Mh, -0.5 * np.sum(Mh ** 2, 1)), LDA=(Wl, -0.5 * np.sum(Wl * Mh, 1)))
    out = {}; Etot = float(np.mean(np.sum(F ** 2, 1)))
    for nm, (Wh, bh) in heads.items():
        sv, vt = svd_split(Wh); Vn2, Vr2 = bases_at(sv, vt, 1e-5); Z2, S2 = F @ Vn2, F @ Vr2
        a, b = c.CM.from_images(Z2, y, K, hA), c.CM.from_images(S2, y, K, hA); r, _ = theta_batch([a], b, T, folds, draws, method)
        acc = None if bh is None else float(np.mean(np.argmax(F[~hA] @ Wh.T + bh, 1) == y[~hA]))
        out[nm] = dict(d_null=int(Vn2.shape[1]), rho=float(np.mean(np.sum(Z2 ** 2, 1))) / Etot, AS=AS_of(F, Vn2, A, Ab, tc), top1_halfB=acc, theta=r[0])
    return out


def verdicts(P, ms, sing, targets):
    lk = -sing; th = {n: P[n]['theta']['theta_sem'] for n in ms}; se = {n: P[n]['theta']['se'] for n in ms}
    rng_ = np.nanmax(list(th.values())) - np.nanmin(list(th.values())); mx = np.nanmax(list(th.values())); V = dict(range_13=float(rng_), max_13=float(mx))
    # G-floor
    cells = []
    for b in BASES:
        for k in ('surr_gauss', 'surr_perm'): cells += [x for x in P[b]['base'][k]]
    for n in ms: cells += [P[n]['surr_own']['gauss'], P[n]['surr_own']['perm']]
    inc = [x['ci'][0] <= 0 <= x['ci'][1] for x in cells]; mabs = max(abs(x['theta_sem']) for x in cells)
    V['G_floor'] = dict(n_cells=len(cells), frac_ci_includes_0=float(np.mean(inc)), max_abs_theta=float(mabs), bar=0.1 * mx,
                        passed=bool(np.mean(inc) >= 0.90 and mabs <= 0.1 * mx))
    # G-noise
    gn = {}
    for b in BASES:
        rows = P[b]['base']["C'"]; est = [r for r in rows if np.isfinite(r['theta_sem']) and r['theta_sem'] > 0 and r['se'] <= 0.5 * abs(r['theta_sem'])]
        d = dict(n_estimable=len(est))
        if len(est) >= 3:
            v = [r['theta_sem'] for r in est]; d['max_over_min'] = float(max(v) / min(v)); d['slope'] = c.loglog_slope([r['rho'] for r in est], v)
            d['passed'] = bool(len(est) >= 8 and d['max_over_min'] <= 1.5 and d['slope']['ci'][0] <= 0 <= d['slope']['ci'][1])
        else: d['passed'] = False
        gn[b] = d
    V['G_noise'] = dict(per_base=gn, passed=all(gn[b]['passed'] for b in BASES))
    # exact invariances + offset world
    inv = {n: P[n]['invariance_rel_err'] for n in ms}; mi = max(max(v.values()) for v in inv.values())
    off = {b: max(abs(r['theta_sem'] / P[b]['base']["C'"][0]['theta_sem'] - 1) for r in P[b]['base']['offset']) for b in BASES}
    V['G_invariance'] = dict(max_rel_err_13=float(mi), offset_world_max_rel_dev=off, bar=1e-8, passed=bool(mi <= 1e-8 and max(off.values()) <= 1e-8))
    # G-derange
    gd = {}
    for b in BASES:
        for lab, x in P[b]['base']['derange'].items():
            hw = 1.96 * x['sd'] / np.sqrt(20); gd[f'{b}/{lab}'] = dict(mean=x['mean'], ci=[x['mean'] - hw, x['mean'] + hw],
                                                                     passed=bool(x['mean'] - hw <= 0 <= x['mean'] + hw and abs(x['mean']) <= 0.1 * mx))
    V['G_derange'] = dict(cells=gd, passed=all(v['passed'] for v in gd.values()))
    # G-S
    gs = {b: dict(semantic=P[b]['base']['plants']['semantic'], arbitrary=P[b]['base']['plants']['arbitrary']) for b in BASES}
    for b in BASES: gs[b]['passed'] = bool(gs[b]['semantic']['ci'][0] > 0 and gs[b]['arbitrary']['ci'][0] <= 0 <= gs[b]['arbitrary']['ci'][1])
    V['G_S'] = dict(per_base=gs, passed=all(gs[b]['passed'] for b in BASES))
    # G-B
    mb = max(r['rel_err'] for b in BASES for r in P[b]['base']['B']); V['G_B'] = dict(max_rel_err=float(mb), bar=1e-8, passed=bool(mb <= 1e-8))
    # G-tol
    base_v = [th[n] for n in ms]; tt = {t: float(ktau(base_v, [P[n]['theta_tol'][t]['theta'] for n in ms])) for t in ('0.0001', '1e-06')}
    V['G_tol'] = dict(tau=tt, passed=bool(min(tt.values()) >= 0.8))
    # estimability
    estm = [n for n in ms if np.isfinite(th[n]) and se[n] <= 0.1 * rng_]
    V['estimability'] = dict(estimable=estm, n=len(estm), bar_se=float(0.1 * rng_), passed=bool(len(estm) >= 11))
    gates = ['G_floor', 'G_noise', 'G_invariance', 'G_derange', 'G_S', 'G_B', 'G_tol', 'estimability']
    V['all_gates_passed'] = all(V[g]['passed'] for g in gates)
    # confirmatory tau on E12 and S11 (estimable models only)
    rho = {n: P[n]['rho'] for n in ms}; sets = dict(E12=[n for n in ms if rho[n] >= RHO0 and n in estm], S11=[n for n in ms if n not in S11_DROP and n in estm])
    L = {n: lk[ms.index(n)] for n in ms}; conf = {}
    for s, sub in sets.items():
        conf[s] = dict(models=sub, n=len(sub), tau=tau_ci([th[n] for n in sub], [L[n] for n in sub]) if len(sub) >= 4 else None)
    V['confirmatory'] = conf
    enough = all(conf[s]['n'] >= 10 for s in conf)
    if not enough: out = 'INCONCLUSIVE (fewer than 10 estimable models in E12 or S11)'
    else:
        t = {s: conf[s]['tau'] for s in conf}
        if all(t[s]['tau'] >= 0.6 and t[s]['pct'][0] > 0 for s in t): out = 'TRACKS'
        elif all(t[s]['tau'] < 0.3 for s in t): out = 'DOES NOT TRACK'
        else: out = 'INCONCLUSIVE'
    V['reading_rule_outcome'] = out if V['all_gates_passed'] else f'NOT VALIDATED (gates failed); tau reported, not read; raw rule would give: {out}'
    # descriptive associations
    desc = {}
    for s in ('E12', 'S11'):
        sub = [n for n in ms if (rho[n] >= RHO0 if s == 'E12' else n not in S11_DROP)]; y_ = [L[n] for n in sub]
        val = lambda f: [f(n) for n in sub]
        desc[s] = dict(
            tau_theta_rho=float(ktau(val(lambda n: th[n]), val(lambda n: rho[n]))),
            tau_theta_rho_between=float(ktau(val(lambda n: th[n]), val(lambda n: P[n]['rho_between']))),
            partial_theta_SING_given_contentfree=partial_ci(val(lambda n: th[n]), y_, val(lambda n: P[n]['rho_offset'] + P[n]['rho_within'])),
            tau_SING=dict(theta_cls=tau_ci(val(lambda n: P[n]['theta_cls']['point']), y_), shadow=tau_ci(val(lambda n: P[n]['shadow']['theta_sem']), y_),
                          innovation=tau_ci(val(lambda n: P[n]['innovation']['theta_sem']), y_), J_innovation=tau_ci(val(lambda n: P[n]['J']['innovation']), y_),
                          self_duality=tau_ci(val(lambda n: P[n]['self_duality']['full']), y_), theta_all_models_fold2=tau_ci(val(lambda n: P[n]['theta_fold2']), y_),
                          theta_all_models=tau_ci(val(lambda n: th[n]), y_)))
    V['descriptive'] = desc
    return V


def main():
    global RES, PM
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); ap.add_argument('--only', default=None)
    ap.add_argument('--res-dir', default=RES, help='smoke tests only; the registered run uses r8/results')
    ap.add_argument('--B', type=int, default=B, help='smoke tests only; the registered run uses 2000'); a = ap.parse_args()
    RES = a.res_dir; PM = os.path.join(RES, 'permodel'); os.makedirs(PM, exist_ok=True)
    k0 = json.load(open(os.path.join(RES, 'K0_coverage.json')))
    if k0['K0_fires']: sys.exit('K0 fired: no valid resampling CI; round 8 stops (PREREG_R8 K0).')
    method = k0['method']
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    E = emb['embeddings'].astype(np.float64); E /= np.linalg.norm(E, axis=-1, keepdims=True); TCn = E.mean(1); TCn /= np.linalg.norm(TCn, axis=1, keepdims=True)
    T = TCn - TCn.mean(0)
    ms, col, sing = load_models(); rho_tab = col('rho'); targets = sorted(float(r) for r in rho_tab if r >= RHO0)
    folds, folds2 = c.fold_split(K, SEED['fold']), c.fold_split(K, SEED['fold2']); draws = c.resample_indices(*folds, a.B, SEED['boot'], method)
    P = {}
    for n in ms:
        p = os.path.join(PM, f'{n}.json')
        if os.path.exists(p): P[n] = json.load(open(p)); continue
        if a.only and n not in a.only.split(','): continue
        P[n] = per_model(n, a.feats_root, T, TCn, folds, folds2, draws, method, targets, None)
        json.dump(P[n], open(p, 'w'), indent=1, default=float)
        print(n, round(P[n]['secs']), 'theta_sem', P[n]['theta']['theta_sem'], P[n]['theta']['ci'], flush=True)
    if len(P) < len(ms): return
    V = verdicts(P, ms, sing, targets); V['method'] = method; V['B'] = a.B; V['seeds'] = SEED
    json.dump(V, open(os.path.join(RES, 'R8_verdicts.json'), 'w'), indent=1, default=float)
    print(json.dumps({k: (V[k]['passed'] if isinstance(V[k], dict) and 'passed' in V[k] else None) for k in V}, default=str))
    print('outcome', V['reading_rule_outcome'])


if __name__ == '__main__':
    main()
