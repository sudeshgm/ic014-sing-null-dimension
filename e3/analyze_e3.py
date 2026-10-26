"""IC014 E3 (planted leakage) CPU analysis. Inputs: private features from e3/extract_e3_gpu.py.
Plant  f' = f + alpha * sigma * s_c(x) * u_c,  sigma = sqrt(E||f||^2) (unplanted),
  u_c = normalise(P_null A^+ t_c)  (A = SING linear translator weight, A^+ Moore-Penrose pinv, t_c = SING CLIP text
        embedding of concept c, prompt entry 'mean', L2-normalised; P_null = Vn Vn^T, Vn = SING projector, tol 1e-5),
  s_c(x) = z-score over the N images of cos(CLIP_img(x), t_c).
Translators are linear: T(f) = A f + Ab, so every perturbation is exact algebra on CPU.
Metrics per (model, concept, alpha), AS = angle(T(f'),t_c) - angle(T(f'~),t_c) in degrees, |AS| averaged over images:
  AS_SING = mean|AS| (full null removal, f'~ = f' - P_null f') at alpha  minus  the same at alpha = 0.
  AS_fair = mean|AS_null(alpha)| / mean_r mean|AS_R_r(alpha)|, R_r = energy-matched random subspace of the FULL
            feature space (see RESULTS/defn text in README): Haar-random orthonormal basis q_1..q_m (seed r), remove
            q_1..q_{k-1} fully and q_k with fraction beta in (0,1], k and beta chosen so that the removed energy
            E||f' - f'~||^2 equals E||P_null f'||^2 exactly (E = mean over the N images, at the same alpha).
Ground truth: n_trials trials; in each, alpha_m ~ logUniform[amin, amax] independently per model (seed); truth = the
  alpha ranking over models; per trial & concept Kendall tau(metric_m(alpha_m), alpha_m) over the models, metrics
  computed exactly at the drawn alpha. Outputs (results dir, small): e3_table.csv, e3_trials.csv, e3_summary.json.
Private (--private dir): <model>_null.npz (null coordinates c = F Vn, Vn) and clip_concepts.npz (all 173 SING
  attribute concepts: raw cosine and z-scored scores) for E8."""
import argparse, os, sys, json, csv, math, numpy as np, torch
from scipy.stats import kendalltau

MODELS = ['resnet', 'resnext', 'dinov2', 'bitresnet', 'deit', 'eva02_large', 'convnext', 'densenet']
CONCEPTS = ['red', 'striped', 'water']
ALPHAS = [0.01, 0.03, 0.1, 0.3, 1.0]

def ang(a, b):
    a = torch.nn.functional.normalize(a, dim=-1)
    return torch.rad2deg(torch.acos(torch.clamp(a @ b, -1, 1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('feats'); ap.add_argument('out'); ap.add_argument('--private', required=True)
    ap.add_argument('--sing-root', required=True); ap.add_argument('--models', nargs='*', default=MODELS)
    ap.add_argument('--concepts', nargs='*', default=CONCEPTS); ap.add_argument('--draws', type=int, default=20)
    ap.add_argument('--trials', type=int, default=50); ap.add_argument('--amin', type=float, default=0.01)
    ap.add_argument('--amax', type=float, default=1.0); ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(a.sing_root, 'src'))
    from sing.core.projectors import compute_projectors_from_weight
    torch.manual_seed(0); os.makedirs(a.out, exist_ok=True); os.makedirs(a.private, exist_ok=True)
    att = np.load(os.path.join(a.sing_root, 'misc/embeddings/attribute_embeddings_broden_style_fp16.npz'), allow_pickle=True)
    names = [str(c) for c in att['classes']]; pi = [str(p) for p in att['prompts']].index('mean')
    TA = torch.nn.functional.normalize(torch.tensor(att['embeddings'][:, pi].astype(np.float32)), dim=-1)  # 173x768
    e = torch.tensor(np.load(os.path.join(a.feats, 'clip_img.npz'))['e']).float()
    raw = e @ TA.T; zs = (raw - raw.mean(0)) / raw.std(0)
    np.savez(os.path.join(a.private, 'clip_concepts.npz'), concepts=np.array(names), raw_cos=raw.numpy(), zscore=zs.numpy())
    cidx = [names.index(c) for c in a.concepts]
    rng = np.random.default_rng(a.seed)
    trial_alpha = np.exp(rng.uniform(math.log(a.amin), math.log(a.amax), size=(a.trials, len(a.models))))
    table, trial_vals, mods = [], {}, {}
    for mi, name in enumerate(a.models):
        d = np.load(os.path.join(a.feats, f'{name}_full.npz'))
        F = torch.tensor(d['F']).float(); W = torch.tensor(d['W']).float(); A = torch.tensor(d['A']).float(); Ab = torch.tensor(d['Ab']).float()
        N, m = F.shape
        Vn = compute_projectors_from_weight(W.double()).v_null.float(); nd = Vn.shape[1]
        Z = F @ A.T + Ab; zerr = float((Z - torch.tensor(d['z16']).float()).abs().max())
        C = F @ Vn; AVn = A @ Vn
        np.savez(os.path.join(a.private, f'{name}_null.npz'), c=C.numpy(), Vn=Vn.numpy(), y=d['y'])
        sigma = float((F ** 2).sum(1).mean().sqrt()); Apinv = torch.linalg.pinv(A)
        Qs = []
        for r in range(a.draws):
            g = torch.Generator().manual_seed(10_000 + r)
            Q = torch.linalg.qr(torch.randn(m, m, generator=g))[0]
            Qs.append((Q, F @ Q, A @ Q))
        mods[name] = dict(m=m, null_dim=nd, sigma=sigma, z_recon_maxerr=zerr)
        for ci, c in zip(cidx, a.concepts):
            t = TA[ci]; s = zs[:, ci]
            w = Apinv @ t; un = Vn @ (Vn.T @ w); keep = float(un.norm() / w.norm()); u = un / un.norm()
            ucoef = Vn.T @ u; Au = A @ u; Wu = W @ u; uQ = [Q.T @ u for Q, _, _ in Qs]
            dlogit1 = float(Wu.abs().max() * sigma * s.abs().max())  # max |logit change| over images at alpha = 1
            logit_scale = float((F @ W.T + torch.tensor(d['bW']).float()).std())
            def evaluate(alpha):
                sc = alpha * sigma * s
                Zp = Z + sc[:, None] * Au[None]
                Cp = C + sc[:, None] * ucoef[None]
                Zt = Zp - Cp @ AVn.T
                a0 = ang(Zp, t); absAS = float((a0 - ang(Zt, t)).abs().mean()); sAS = float((a0 - ang(Zt, t)).mean())
                Enull = float((Cp ** 2).sum(1).mean()); Etot = float(((F + sc[:, None] * u[None]) ** 2).sum(1).mean())
                rr, ks = [], []
                for (Q, FQ, AQ), uq in zip(Qs, uQ):
                    P = FQ + sc[:, None] * uq[None]
                    cum = torch.cumsum((P ** 2).mean(0), 0)
                    k = int(torch.searchsorted(cum, torch.tensor(Enull)).item()); k = min(k, m - 1)
                    prev = float(cum[k - 1]) if k > 0 else 0.0
                    beta = math.sqrt(max(Enull - prev, 0.0) / float((P[:, k] ** 2).mean()))
                    coef = P[:, :k + 1].clone(); coef[:, k] *= beta
                    Zr = Zp - coef @ AQ[:, :k + 1].T
                    rr.append(float((a0 - ang(Zr, t)).abs().mean())); ks.append(k + beta)
                return dict(absAS_null=absAS, AS_null_signed=sAS, rho_null=Enull / Etot, absAS_rand=float(np.mean(rr)),
                            absAS_rand_sd=float(np.std(rr)), k_rand=float(np.mean(ks)), AS_fair=absAS / float(np.mean(rr)))
            base = evaluate(0.0)
            for al in ALPHAS:
                r = evaluate(al)
                table.append(dict(model=name, m=m, null_dim=nd, concept=c, alpha=al, AS_SING=r['absAS_null'] - base['absAS_null'],
                                  AS_SING_signed=r['AS_null_signed'] - base['AS_null_signed'], **r,
                                  absAS_null_alpha0=base['absAS_null'], AS_fair_alpha0=base['AS_fair'],
                                  max_logit_change=dlogit1 * al, logit_sd=logit_scale, u_null_fraction=keep))
            for ti in range(a.trials):
                r = evaluate(float(trial_alpha[ti, mi]))
                trial_vals[(ti, c, name)] = (r['absAS_null'] - base['absAS_null'], r['AS_fair'])
            print(json.dumps(dict(model=name, concept=c, null_dim=nd, u_null_fraction=keep, max_logit_change_alpha1=dlogit1,
                                  AS_SING_alpha1=table[-1]['AS_SING'], AS_fair_alpha1=table[-1]['AS_fair'])), flush=True)
    trows, summ = [], dict(models=mods, concepts=a.concepts, prompt_entry='mean', draws=a.draws, trials=a.trials,
                           alpha_range=[a.amin, a.amax], seed=a.seed, per_concept={})
    boot = np.random.default_rng(1)
    def ci(x):
        x = np.asarray(x, float); x = x[~np.isnan(x)]
        bs = boot.choice(x, size=(10000, len(x))).mean(1)
        return dict(mean=float(x.mean()), ci95_boot_mean=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                    pct2_5_97_5_trials=[float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))], n=int(len(x)))
    allt = {'AS_SING': [], 'AS_fair': []}
    for c in a.concepts:
        per = {'AS_SING': [], 'AS_fair': []}
        for ti in range(a.trials):
            al = trial_alpha[ti]; v = np.array([trial_vals[(ti, c, n)] for n in a.models])
            row = dict(trial=ti, concept=c, **{f'alpha_{n}': float(al[j]) for j, n in enumerate(a.models)})
            for k, key in enumerate(['AS_SING', 'AS_fair']):
                tau = float(kendalltau(v[:, k], al).statistic); per[key].append(tau); allt[key].append(tau); row[f'tau_{key}'] = tau
                row.update({f'{key}_{n}': float(v[j, k]) for j, n in enumerate(a.models)})
            trows.append(row)
        summ['per_concept'][c] = {k: ci(v) for k, v in per.items()}
    tri = np.array(allt['AS_SING']).reshape(len(a.concepts), a.trials).mean(0); trf = np.array(allt['AS_fair']).reshape(len(a.concepts), a.trials).mean(0)
    summ['concept_averaged_per_trial'] = {'AS_SING': ci(tri), 'AS_fair': ci(trf)}
    for fn, rows in [('e3_table.csv', table), ('e3_trials.csv', trows)]:
        with open(os.path.join(a.out, fn), 'w', newline='') as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump(summ, open(os.path.join(a.out, 'e3_summary.json'), 'w'), indent=1)
    print(json.dumps(summ['concept_averaged_per_trial']))

if __name__ == '__main__':
    main()
