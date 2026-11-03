"""PREREG_R7 analysis of one model's view features (aggregates only to r7/results/<model>.json; no per-image values).
Usage: python r7/analyze_r7.py --feats _ext/r7_feats --sing-root _ext/sing --model resnet"""
import argparse, os, sys, json, time, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H); sys.path.insert(0, H); sys.path.insert(0, os.path.join(ROOT, 'r6'))
import stats_r7 as st  # noqa: E402
from scores_r6 import null_split  # noqa: E402

PERM_SEED, BOOT_SEED, NPERM, NBOOT = 100720, 100730, 1000, 500
PRIMARY_LEVEL = 2


def text(sing_root):
    e = np.load(os.path.join(sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)['embeddings'].astype(np.float64)
    e = e / np.linalg.norm(e, axis=-1, keepdims=True); t = e.mean(1); return t / np.linalg.norm(t, axis=1, keepdims=True)


def ci(x): x = np.asarray(x, float); x = x[np.isfinite(x)]; return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else [np.nan, np.nan]


def level_arrays(F, slot, L, B):
    """edit (X - S), artefact control (Xs - Ss) and nuisance-only (S_0 - S_1, S_2 - S_3) in basis B"""
    g = lambda c, v: F[slot[(L, v, c)]].astype(np.float64) @ B
    E = np.stack([g('X', v) - g('S', v) for v in range(4)]); A = np.stack([g('Xs', v) - g('Ss', v) for v in range(4)])
    S = [g('S', v) for v in range(4)]; Nn = np.stack([S[0] - S[1], S[2] - S[3]]); return E, A, Nn


def analyse_level(E, A, Nn, R, g, boot_idx, perm_seed):
    out = {}
    for name, D in (('edit', E), ('artefact', A)):
        s = st.T_stat(D, R, g); out[name] = dict(T=s['T'], T2=s['T2'], den=s['den'], tr=s['tr'], reliability=st.reliability(D, g),
                                                 nu=st.noise_ratio(D, g)['nu'], naive=st.T_naive(D, R, g), T2_gl=st.T_stat(D, R, g, 'gl')['T2'],
                                                 perm=st.perm_test(D, R, g, NPERM, perm_seed))
    s = st.T_stat(E, R, g); nn = st.control_num(Nn, R, g)
    out['nuisance_only'] = dict(T2_scale=nn / (s['tr'] * np.sqrt(s['kk'])), T_scale=nn / np.sqrt(s['den'] * s['kk']) if s['den'] > 0 else np.nan)
    bs = {k: [] for k in ('T', 'T2', 'rel', 'dT2_art', 'dT2_nui', 'T_undef')}
    for idx in boot_idx:
        se = st.T_stat(E, R, g, idx=idx); sa = st.T_stat(A, R, g, idx=idx); nb = st.control_num(Nn[:, idx], R[idx], g[idx])
        bs['T'].append(se['T']); bs['T2'].append(se['T2']); bs['rel'].append(st.reliability(E[:, idx], g[idx]))
        bs['dT2_art'].append(se['T2'] - sa['T2']); bs['dT2_nui'].append(se['T2'] - nb / (se['tr'] * np.sqrt(se['kk']))); bs['T_undef'].append(not np.isfinite(se['T']))
    out['boot'] = dict(T_ci=ci(bs['T']), T2_ci=ci(bs['T2']), T2_sd=float(np.nanstd(bs['T2'])), reliability_ci=ci(bs['rel']),
                       dT2_artefact_ci=ci(bs['dT2_art']), dT2_nuisance_ci=ci(bs['dT2_nui']), T_undefined_frac=float(np.mean(bs['T_undef'])))
    return out


def verdicts(res, numax):
    L2 = res['levels'][str(PRIMARY_LEVEL)]; v = {}
    v['H1'] = 'REPRODUCIBLE' if L2['boot']['reliability_ci'][0] > 0.10 else 'NOT REPRODUCIBLE'
    out = {}
    for stat in ('T2', 'T'):
        est = {L: (res['levels'][L]['edit']['nu'] <= numax[stat]) and (stat == 'T2' or res['levels'][L]['boot']['T_undefined_frac'] <= 0.05) and np.isfinite(res['levels'][L]['edit'][stat])
               for L in res['levels']}
        if not est[str(PRIMARY_LEVEL)]: h2 = 'NOT ESTIMABLE'
        else:
            p = L2['edit']['perm']['p']; art = L2['boot']['dT2_artefact_ci'][0] > 0; nui = L2['boot']['dT2_nuisance_ci'][0] > 0
            if stat == 'T': art, nui = True, True  # T uses the same controls via T2 (descriptive)
            h2 = 'ALIGNED' if p < 0.01 and art and nui else 'NOT ALIGNED' if p >= 0.05 else 'ARTEFACT' if p < 0.01 and not art else 'PARTIAL'
        nus = [res['levels'][L]['edit']['nu'] for L in ('1', '2', '3')]
        if h2 != 'ALIGNED': h3 = 'NOT TESTED (H2 not ALIGNED)'
        elif not (nus[2] / nus[0] >= 1.5): h3 = 'INCONCLUSIVE (nuisance levels do not add noise)'
        elif not all(est.values()): h3 = 'NOT ESTIMABLE'
        else:
            vals = [res['levels'][L]['edit'][stat] for L in ('1', '2', '3')]; ps = [res['levels'][L]['edit']['perm']['p'] for L in ('1', '2', '3')]
            h3 = 'CANCELS' if min(vals) > 0 and max(ps) < 0.01 and max(vals) / min(vals) <= 1.5 else 'DOES NOT CANCEL'
        out[stat] = dict(estimable=est, H2=h2, H3=h3)
    v['by_stat'] = out; v['supports_cancellation'] = v['H1'] == 'REPRODUCIBLE' and out['T2']['H2'] == 'ALIGNED' and out['T2']['H3'] == 'CANCELS'
    return v


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats', required=True); ap.add_argument('--sing-root', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--synthetic', default=os.path.join(H, 'results', 'synthetic_stability.json')); ap.add_argument('--out', default=os.path.join(H, 'results'))
    a = ap.parse_args(); t0 = time.time()
    meta = json.load(open(os.path.join(a.feats, 'items_r7.json'))); its = meta['items']; slot = {tuple(s): j for j, s in enumerate(meta['slots'])}
    prog = int(open(os.path.join(a.feats, f'{a.model}_r7.progress')).read()); assert prog >= len(its), 'extraction not finished'
    F = np.load(os.path.join(a.feats, f'{a.model}_r7.npy'), mmap_mode='r'); hd = np.load(os.path.join(a.feats, f'{a.model}_r7_head.npz'))
    W, b = hd['W'].astype(np.float64), hd['bW'].astype(np.float64); Vr, Vn = null_split(W)
    g = np.array([it['g'] for it in its]); R = text(a.sing_root)[[it['bg_idx'] for it in its]]
    syn = json.load(open(a.synthetic))['summary']; numax = {s: syn[f'{s}_lowrank_T0.2']['max_stable_nu'] for s in ('T2', 'T')}
    rng = np.random.default_rng(BOOT_SEED); boot_idx = [st.boot_within(g, rng) for _ in range(NBOOT)]
    res = dict(model=a.model, N=len(its), d=int(W.shape[1]), d_null=int(Vn.shape[1]), nu_max=numax, levels={}, row={}, checks={})
    X0 = F[slot[(0, 0, 'X')]].astype(np.float64); S0 = F[slot[(0, 0, 'S')]].astype(np.float64)
    LX, LS = X0 @ W.T + b, S0 @ W.T + b; yfg = None
    res['checks']['clean_top1_agree_X_S'] = float((LX.argmax(1) == LS.argmax(1)).mean())
    res['checks']['clean_null_energy_frac_X'] = float(((X0 @ Vn) ** 2).sum() / (X0 ** 2).sum())
    res['checks']['clean_edit_null_energy_frac'] = float((((X0 - S0) @ Vn) ** 2).sum() / ((X0 - S0) ** 2).sum())
    for L in (1, 2, 3):
        E, A, Nn = level_arrays(F, slot, L, Vn); res['levels'][str(L)] = analyse_level(E, A, Nn, R, g, boot_idx, PERM_SEED + L)
        Er, Ar, Nr = level_arrays(F, slot, L, Vr); se = st.T_stat(Er, R, g)
        res['row'][str(L)] = dict(T=se['T'], T2=se['T2'], reliability=st.reliability(Er, g), nu=st.noise_ratio(Er, g)['nu'], perm=st.perm_test(Er, R, g, 200, PERM_SEED + 10 + L))
        if L == PRIMARY_LEVEL:
            res['checks']['worldB_rescale'] = {str(s): st.T_stat(E * s, R, g)['T2'] - res['levels'][str(L)]['edit']['T2'] for s in (0.1, 10.0)}
        print(json.dumps({a.model: L, 'secs': round(time.time() - t0, 1)}), flush=True)
    res['verdicts'] = verdicts(res, numax); res['secs'] = round(time.time() - t0, 1)
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, f'{a.model}.json'), 'w'), indent=1, default=float)
    print(json.dumps(res['verdicts'], indent=1, default=str))


if __name__ == '__main__':
    main()
