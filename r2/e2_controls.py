"""Round-2 fixes 2, 5 (and inputs for 3, 4, 8, 10): recompute the 13-model E2 scores
 - at null tolerances abs {1e-6,1e-5,1e-4,1e-3} and rel {1e-5,1e-4,1e-3}*sigma_max: null dim, rho, AS_SING, AS_fair-A;
 - at tol 1e-5: control families B (low-gain row space) and C (d=24 null sub-subspaces), see PREREG_R2.md;
 - degeneracy-screen statistics; M1 (debiased null leakage, alpha=0, mean over 173 SING concepts).
Per-image arrays for fix 8 are written to --private (outside the repo).
Usage: python r2/e2_controls.py --feats-root $IC014_FEATS [--sing-root _ext/sing] --private $IC014_FEATS/r2_private"""
import argparse, os, sys, json, csv, math, time, numpy as np, torch
torch.set_num_threads(os.cpu_count())
ALL = ['vgg16', 'vgg19', 'densenet', 'resnet', 'dinovit1', 'efficientnet', 'bitresnet', 'resnext', 'convnext',
       'swin_large', 'dinov2', 'deit', 'eva02_large']
ABS = [1e-6, 1e-5, 1e-4, 1e-3]; REL = [1e-5, 1e-4, 1e-3]; DRAWS = 20

def ang_rows(Z, T):
    return torch.rad2deg(torch.acos(torch.clamp((torch.nn.functional.normalize(Z, dim=-1) * T).sum(-1), -1, 1)))

def energy_match(P, E):
    """P: coords (N,K) on an orthonormal basis; remove first k fully and k-th by beta so removed energy = E."""
    cum = torch.cumsum((P ** 2).mean(0), 0); K = P.shape[1]
    if float(cum[-1]) <= E: return P.clone(), float(K), True   # cannot reach E: remove everything (shortfall)
    k = min(int(torch.searchsorted(cum, torch.tensor(E, dtype=cum.dtype)).item()), K - 1)
    prev = float(cum[k - 1]) if k > 0 else 0.0
    beta = math.sqrt(max(E - prev, 0.0) / float((P[:, k] ** 2).mean()))
    coef = P[:, :k + 1].clone(); coef[:, k] *= beta; return coef, k + beta, False

def find(roots, name):
    for r in roots:
        f = os.path.join(r, f'{name}_full.npz')
        if os.path.exists(f): return f

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_ext', 'sing')))
    ap.add_argument('--private', required=True); ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'results'))
    ap.add_argument('--models', nargs='*', default=ALL)
    a = ap.parse_args(); os.makedirs(a.private, exist_ok=True); os.makedirs(a.out, exist_ok=True)
    import importlib.util  # load SING's projectors.py directly (the sing package __init__ pulls in diffusers)
    spec = importlib.util.spec_from_file_location('sing_projectors', os.path.join(a.sing_root, 'src/sing/core/projectors.py'))
    pj = importlib.util.module_from_spec(spec); sys.modules['sing_projectors'] = pj; spec.loader.exec_module(pj); compute_projectors_from_weight = pj.compute_projectors_from_weight
    fr = a.feats_root; roots = [f'{fr}/e2/ret/e2_feats', f'{fr}/e3/ret/e3_feats']; croot = [f'{fr}/e2/ret/e2_claim_feats']
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float32)), dim=-1).mean(1), dim=-1)
    S = torch.tensor(np.load(f'{fr}/e3/private/clip_concepts.npz')['zscore']).double()   # same 10k subset (checked)
    S = (S - S.mean(0)) / S.std(0)
    rows, tolrows = [], []
    for name in a.models:
        t0 = time.time(); d = np.load(find(roots, name)); dc = np.load(find(croot, name))
        F = torch.tensor(d['F']).float(); A = torch.tensor(d['A']).float(); Ab = torch.tensor(d['Ab']).float(); Wd = torch.tensor(d['W']).double()
        y = torch.tensor(d['y']).long(); N, m = F.shape; tc = TC[y]
        Fc = torch.tensor(dc['F']).float(); yc = torch.tensor(dc['y']).long(); tcc = TC[yc]
        Z = F @ A.T + Ab; Zc = Fc @ A.T + Ab; a0 = ang_rows(Z, tc); a0c = ang_rows(Zc, tcc)
        L = F.double() @ Wd.T + torch.tensor(d['bW']).double(); Lsd = float(L.std())
        _, sv, vt = torch.linalg.svd(Wd, full_matrices=True); V = vt.T.float(); sv = sv.numpy()
        ref = compute_projectors_from_weight(Wd); assert ref.v_null.shape[1] == m - int((sv > 1e-5).sum())
        Etot = float((F.double() ** 2).sum(1).mean()); ecol = ((F @ V).double() ** 2).mean(0).numpy()   # energy per right-singular dir
        tols = [('abs', t, t) for t in ABS] + [('rel', t, t * float(sv.max())) for t in REL]
        ranks = [int((sv > thr).sum()) for _, _, thr in tols]
        def asx(Zr, Zbase=Z, a00=a0, t=tc): return (a00 - ang_rows(Zr, t)).abs()
        T = {}
        for (kind, t, thr), r in zip(tols, ranks):
            Vn = V[:, r:]; C = F @ Vn; En = float((C.double() ** 2).sum(1).mean())
            Zt = Z - C @ (A @ Vn).T; T[(kind, t)] = dict(rank=r, null_dim=m - r, Enull=En, rho=En / Etot, AS=float(asx(Zt).mean()), rand=[], k=[])
            if kind == 'abs' and t == 1e-5:
                AS_img = asx(Zt); Cc = Fc @ Vn; AS_c = (a0c - ang_rows(Zc - Cc @ (A @ Vn).T, tcc)).abs()
                IS = float(ang_rows(Z, torch.nn.functional.normalize(Zt, dim=-1)).mean())
                Cd = C.double(); Ssum = Cd.T @ S; L2 = ((Ssum ** 2).sum(0) - ((Cd ** 2).sum(1)[:, None] * S ** 2).sum(0)) / (N * (N - 1) * Etot)
                M1 = L2.clamp_min(0).sqrt().numpy()
        randA_img = torch.zeros(N); randA_c = torch.zeros(len(Fc))
        for rr in range(DRAWS):
            g = torch.Generator().manual_seed(10_000 + rr); Q = torch.linalg.qr(torch.randn(m, m, generator=g))[0]
            P = F @ Q; AQ = A @ Q; Pc = None
            for key, v in T.items():
                coef, kk, _ = energy_match(P, v['Enull']); k1 = coef.shape[1]
                ai = asx(Z - coef @ AQ[:, :k1].T); v['rand'].append(float(ai.mean())); v['k'].append(kk)
                if key == ('abs', 1e-5):
                    randA_img += ai / DRAWS
                    cc = Fc @ Q[:, :k1]; 
                    if k1 > 0: cc[:, -1] *= coef[0, -1] / P[0, k1 - 1] if float(P[0, k1 - 1]) != 0 else 1.0
                    randA_c += (a0c - ang_rows(Zc - cc @ AQ[:, :k1].T, tcc)).abs() / DRAWS
            del Q, P, AQ
        # families B and C at tol 1e-5
        r5 = T[('abs', 1e-5)]['rank']; Vn = V[:, r5:]; En = T[('abs', 1e-5)]['Enull']; nd = m - r5
        bot = list(range(r5 - 1, -1, -1))  # row-space dirs, sigma ascending
        def famB(E, seed):
            cum = np.cumsum(ecol[bot]); K = int(np.searchsorted(cum, 1.5 * E) + 1); K = max(K, 24); capped = K >= r5; K = min(K, r5)
            Vb = V[:, bot[:K]]; g = torch.Generator().manual_seed(seed)
            Qb = Vb @ torch.linalg.qr(torch.randn(K, K, generator=g))[0]; P = F @ Qb
            coef, kk, short = energy_match(P, E); k1 = coef.shape[1]
            ai = asx(Z - coef @ (A @ Qb[:, :k1]).T); dl = (coef.double() @ (Wd @ Qb[:, :k1].double()).T).abs().max()
            return float(ai.mean()), K, capped, short, float(dl) / Lsd, kk
        B = [famB(En, 50_000 + rr) for rr in range(DRAWS)]
        Cn, Cd_ = [], []
        for j in range(DRAWS):
            g = torch.Generator().manual_seed(30_000 + j); Bj = Vn @ torch.linalg.qr(torch.randn(nd, nd, generator=g))[0][:, :24]
            cj = F @ Bj; Cn.append(float(asx(Z - cj @ (A @ Bj).T).mean())); Ej = float((cj.double() ** 2).sum(1).mean())
            Cd_.append(famB(Ej, 60_000 + j)[0])
        # degeneracy screen stats at tol 1e-5
        small = [i for i in range(r5) if sv[i] < 1e-3]; e_small = float(ecol[small].max() / Etot) if small else 0.0
        row = dict(model=name, m=m, null_dim=nd, rho=T[('abs', 1e-5)]['rho'], AS_SING=T[('abs', 1e-5)]['AS'], IS=IS,
                   AS_rand_A=float(np.mean(T[('abs', 1e-5)]['rand'])), AS_rand_B=float(np.mean([b[0] for b in B])),
                   K_B=B[0][1], B_capped=B[0][2], B_shortfall=bool(any(b[3] for b in B)), B_max_dlogit_over_sd=float(max(b[4] for b in B)),
                   k_B=float(np.mean([b[5] for b in B])), AS_null24=float(np.mean(Cn)), AS_rand_C=float(np.mean(Cd_)),
                   n_retained_sigma_lt_1e3=len(small), max_energy_frac_retained_sigma_lt_1e3=e_small, M1_concept_mean=float(M1.mean()))
        row['AS_fair_A'] = row['AS_SING'] / row['AS_rand_A']; row['AS_fair_B'] = row['AS_SING'] / row['AS_rand_B']; row['AS_fair_C'] = row['AS_null24'] / row['AS_rand_C']
        row['screen_exclude'] = bool(row['rho'] < 1e-4 or e_small > 0.05); rows.append(row)
        for (kind, t), v in T.items():
            tolrows.append(dict(model=name, tol_kind=kind, tol=t, rank=v['rank'], null_dim=v['null_dim'], rho=v['rho'], AS_SING=v['AS'],
                                AS_rand_A=float(np.mean(v['rand'])), AS_fair_A=v['AS'] / float(np.mean(v['rand'])), k_rand=float(np.mean(v['k']))))
        np.savez(os.path.join(a.private, f'{name}_r2img.npz'), y=y.numpy(), AS=AS_img.numpy(), randA=randA_img.numpy(),
                 yc=yc.numpy(), ASc=AS_c.numpy(), randAc=randA_c.numpy(), M1=M1)
        print(json.dumps({k: row[k] for k in ['model', 'null_dim', 'rho', 'AS_SING', 'AS_fair_A', 'AS_fair_B', 'AS_fair_C', 'K_B', 'B_capped', 'B_max_dlogit_over_sd', 'screen_exclude']}), f'{time.time()-t0:.0f}s', flush=True)
        for fn, rr_ in [('fix2_models.csv', rows), ('fix5_tolerance.csv', tolrows)]:
            with open(os.path.join(a.out, fn), 'w', newline='') as fh:
                w = csv.DictWriter(fh, fieldnames=list(rr_[0])); w.writeheader(); w.writerows(rr_)

if __name__ == '__main__':
    main()
