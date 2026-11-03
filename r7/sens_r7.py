"""POST-HOC DEVIATION from PREREG_R7 (written after the resnet result; see r7/README.md).
Bug found: the pre-registered within-group bootstrap resamples images WITH replacement; a duplicated image forms an
off-diagonal pair with itself, so HSIC_u (which only drops the true diagonal) counts ||mu_i||^2 x ||R_i||^2 = self-similarity as
cross-image covariance. The bootstrap distribution of T2 (and T) is then biased far upward (resnet L2: point T2 0.0017, bootstrap
CI 0.012-0.016) and its CIs, including the dT2 control CIs used by H2, are invalid. Reliability is not a U-statistic and is unaffected.
Correction used here: within-group half-sampling WITHOUT replacement (no duplicates), 500 draws, seed 100750; for half-samples
Var(theta_half - theta_hat) ~= Var(theta_hat), so basic CIs [2 t - q97.5, 2 t - q2.5] are on the full-sample scale.
Point estimates and permutation tests are unchanged (they never resample). Writes r7/results/<model>_sens.json."""
import argparse, os, sys, json, time, numpy as np
H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, H); sys.path.insert(0, os.path.join(os.path.dirname(H), 'r6'))
import stats_r7 as st, analyze_r7 as an  # noqa: E402
from scores_r6 import null_split  # noqa: E402
SEED, NREP = 100750, 500


def t2_parts(D, R, g, idx=None):
    if idx is not None: D, R, g = D[:, idx], R[idx], g[idx]
    num = kk = tr = 0.0
    for m in st._groups(g, 'wc'):
        G = st.cross_gram(D[:, m]); K = R[m] @ R[m].T; num += st.hsic_u(G, K); kk += st.hsic_u(K, K); tr += st.tr_u(G) / len(m)
    return num, kk, tr


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats', required=True); ap.add_argument('--sing-root', required=True); ap.add_argument('--model', required=True)
    ap.add_argument('--out', default=os.path.join(H, 'results')); a = ap.parse_args(); t0 = time.time()
    meta = json.load(open(os.path.join(a.feats, 'items_r7.json'))); its = meta['items']; slot = {tuple(s): j for j, s in enumerate(meta['slots'])}
    F = np.load(os.path.join(a.feats, f'{a.model}_r7.npy'), mmap_mode='r'); W = np.load(os.path.join(a.feats, f'{a.model}_r7_head.npz'))['W'].astype(np.float64)
    _, Vn = null_split(W); g = np.array([it['g'] for it in its]); R = an.text(a.sing_root)[[it['bg_idx'] for it in its]]
    rng = np.random.default_rng(SEED)
    halves = [np.concatenate([rng.choice(np.where(g == c)[0], int((g == c).sum()) // 2, replace=False) for c in np.unique(g)]) for _ in range(NREP)]
    out = dict(model=a.model, method='within-group half-sampling without replacement, basic CI', nrep=NREP, seed=SEED, levels={})
    for L in (1, 2, 3):
        E, A, Nn = an.level_arrays(F, slot, L, Vn)
        def stats(idx):
            ne, kk, te = t2_parts(E, R, g, idx); na, _, ta = t2_parts(A, R, g, idx)
            nn = st.control_num(Nn if idx is None else Nn[:, idx], R if idx is None else R[idx], g if idx is None else g[idx])
            e = ne / (te * np.sqrt(kk)); return dict(T2_edit=e, T2_art=na / (ta * np.sqrt(kk)), dT2_art=e - na / (ta * np.sqrt(kk)), dT2_nui=e - nn / (te * np.sqrt(kk)))
        p = stats(None); bs = [stats(h) for h in halves]; r = {}
        for k in p:
            q = np.percentile([b[k] for b in bs], [2.5, 97.5]); r[k] = dict(point=float(p[k]), ci=[float(2 * p[k] - q[1]), float(2 * p[k] - q[0])],
                                                                             sd=float(np.std([b[k] for b in bs])))
        out['levels'][str(L)] = r; print(json.dumps({a.model: L, 'secs': round(time.time() - t0, 1)}), flush=True)
    L2 = out['levels'][str(an.PRIMARY_LEVEL)]; perm_p = json.load(open(os.path.join(a.out, f'{a.model}.json')))['levels'][str(an.PRIMARY_LEVEL)]['edit']['perm']['p']
    art, nui = L2['dT2_art']['ci'][0] > 0, L2['dT2_nui']['ci'][0] > 0
    out['H2_T2_corrected'] = 'ALIGNED' if perm_p < 0.01 and art and nui else 'NOT ALIGNED' if perm_p >= 0.05 else 'ARTEFACT' if perm_p < 0.01 and not art else 'PARTIAL'
    json.dump(out, open(os.path.join(a.out, f'{a.model}_sens.json'), 'w'), indent=1); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
