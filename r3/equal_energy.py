"""PREREG_R3 A + H: equal null-energy intervention, CLIP-space energy and translator gain, NuSA-style null norm.
Per model (13), class-text target, SING released translator T(f)=Af+b, null basis at abs tol 1e-5 (as r2/e2_controls.py):
  A1  remove beta*Pi f, beta = sqrt(rho0/rho), rho0 = 0.005 (A1' rho0 = 0.001)
  A2  remove top-variance null PCs (uncentred), energy-matched to rho0*E||f||^2
  A3  same energy from Haar-random rotations of the null basis (20 draws, seed 80000+j), mean over draws
  kappa = E||A Pi f||^2 / E||T f||^2, gamma = kappa/rho; A4 removes beta'*Pi f with beta'^2 kappa = kappa0 = min kappa over E12
  H   NuSA-style null norm mean_i ||Pi f_i|| / ||f_i||
Score: mean_i |theta(Tf_i, t) - theta(T(f_i - r_i), t)| (degrees). Outputs r3/results/A_equal_energy.json (+ per-model cache).
Usage: python r3/equal_energy.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, math, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import R, RES, load_models, tau_ci, paired_ci, ktau, dump, RHO0, RHO0_SENS, S11_DROP, PRETTY
torch.set_num_threads(os.cpu_count())
NA3 = 20


def ang_rows(Z, T):
    return torch.rad2deg(torch.acos(torch.clamp((torch.nn.functional.normalize(Z, dim=-1) * T).sum(-1), -1, 1)))


def energy_match(P, E):
    cum = torch.cumsum((P.double() ** 2).mean(0), 0); K = P.shape[1]
    if float(cum[-1]) <= E: return P.clone(), float(K), True
    k = min(int(torch.searchsorted(cum, torch.tensor(E, dtype=cum.dtype)).item()), K - 1)
    prev = float(cum[k - 1]) if k > 0 else 0.0
    beta = math.sqrt(max(E - prev, 0.0) / float((P[:, k].double() ** 2).mean()))
    coef = P[:, :k + 1].clone(); coef[:, k] *= beta; return coef, k + beta, False


def find(roots, name):
    for r in roots:
        f = os.path.join(r, f'{name}_full.npz')
        if os.path.exists(f): return f


def load(fr, sing_root, name, TC):
    d = np.load(find([f'{fr}/e2/ret/e2_feats', f'{fr}/e3/ret/e3_feats'], name))
    F = torch.tensor(d['F']).float(); A = torch.tensor(d['A']).float(); Ab = torch.tensor(d['Ab']).float(); Wd = torch.tensor(d['W']).double()
    y = torch.tensor(d['y']).long(); tc = TC[y]
    _, sv, vt = torch.linalg.svd(Wd, full_matrices=True); r = int((sv > 1e-5).sum()); Vn = vt.T[:, r:].float()
    return F, A, Ab, tc, Vn


def per_model(fr, sing_root, name, TC):
    F, A, Ab, tc, Vn = load(fr, sing_root, name, TC); N, m = F.shape; nd = Vn.shape[1]
    Z = F @ A.T + Ab; a0 = ang_rows(Z, tc); asx = lambda rem: float((a0 - ang_rows(Z - rem, tc)).abs().mean())
    Etot = float((F.double() ** 2).sum(1).mean()); C = F @ Vn; AV = A @ Vn; ANull = C @ AV.T
    En = float((C.double() ** 2).sum(1).mean()); rho = En / Etot
    kappa = float((ANull.double() ** 2).sum(1).mean() / (Z.double() ** 2).sum(1).mean())
    out = dict(model=name, m=m, null_dim=nd, rho=rho, kappa=kappa, gamma=kappa / rho,
               AS_SING=asx(ANull), nusa=float((C.norm(dim=1) / F.norm(dim=1)).mean()))
    for tag, r0 in [('A1', RHO0), ('A1p', RHO0_SENS)]:
        b = math.sqrt(r0 / rho); out[f'{tag}_beta'] = b; out[tag] = asx(b * ANull)
    # A2: top-variance (uncentred) null PCs
    M2 = (C.double().T @ C.double()) / N; ev, U = torch.linalg.eigh(M2); U = U[:, torch.argsort(ev, descending=True)].float()
    P = C @ U; coef, k2, short = energy_match(P, RHO0 * Etot); k1 = coef.shape[1]
    out['A2'] = asx(coef @ (AV @ U[:, :k1]).T); out['A2_k'] = k2; out['A2_short'] = short
    del P, M2
    a3 = []; k3 = []
    for j in range(NA3):
        g = torch.Generator().manual_seed(80_000 + j); Q = torch.linalg.qr(torch.randn(nd, nd, generator=g))[0]
        P = C @ Q; coef, kk, short = energy_match(P, RHO0 * Etot); k1 = coef.shape[1]
        a3.append(asx(coef @ (AV @ Q[:, :k1]).T)); k3.append(kk)
    out['A3'] = float(np.mean(a3)); out['A3_sd'] = float(np.std(a3)); out['A3_k'] = float(np.mean(k3))
    return out


def a4(fr, sing_root, name, TC, kappa0, kappa):
    F, A, Ab, tc, Vn = load(fr, sing_root, name, TC); Z = F @ A.T + Ab; a0 = ang_rows(Z, tc)
    b = math.sqrt(kappa0 / kappa); rem = b * ((F @ Vn) @ (A @ Vn).T)
    return float((a0 - ang_rows(Z - rem, tc)).abs().mean()), b


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args()
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float32)), dim=-1).mean(1), dim=-1)
    ms, col, sing = load_models(); lk = -sing; rho_c = col('rho'); AS_c = col('AS_SING')
    cache = os.path.join(RES, 'A_equal_energy_permodel.json'); P = json.load(open(cache)) if os.path.exists(cache) else {}
    for name in ms:
        if name in P: continue
        t0 = time.time(); P[name] = per_model(a.feats_root, a.sing_root, name, TC); P[name]['secs'] = time.time() - t0
        json.dump(P, open(cache, 'w'), indent=1); print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in P[name].items()}), flush=True)
    e12 = [n for n, r in zip(ms, rho_c) if r >= RHO0]
    k0 = min(P[n]['kappa'] for n in e12)
    for n in ms:
        if 'A4' not in P[n]:
            P[n]['A4'], P[n]['A4_beta'] = a4(a.feats_root, a.sing_root, n, TC, k0, P[n]['kappa']); json.dump(P, open(cache, 'w'), indent=1)
    # consistency with the committed round-2 table
    chk = {n: dict(rho_rel_err=abs(P[n]['rho'] - r) / r, AS_rel_err=abs(P[n]['AS_SING'] - s) / s) for n, r, s in zip(ms, rho_c, AS_c)}
    sets = {'all13': ms, 'E12': e12, 'S11': [n for n in ms if n not in S11_DROP]}
    V = lambda key, sub: np.array([P[n][key] for n in sub]); L = lambda sub: np.array([lk[ms.index(n)] for n in sub])
    res = dict(rho0=RHO0, rho0_sens=RHO0_SENS, kappa0=k0, kappa0_model=min(e12, key=lambda n: P[n]['kappa']), sets=sets, per_model=P,
               consistency_vs_fix2=chk, tau={}, paired={})
    for sname, sub in sets.items():
        res['tau'][sname] = {}; res['paired'][sname] = {}
        for key in ['rho', 'kappa', 'gamma', 'nusa', 'AS_SING', 'A1', 'A1p', 'A2', 'A3', 'A4']:
            if sname == 'all13' and key in ('A1', 'A1p', 'A2', 'A3', 'A4'): continue   # densenet beta>1 / not defined
            res['tau'][sname][key] = tau_ci(V(key, sub), L(sub))
        for key in ['A1', 'A1p', 'A2', 'A3', 'A4']:
            if sname != 'all13': res['paired'][sname][f'AS_SING_minus_{key}'] = paired_ci(V('AS_SING', sub), V(key, sub), L(sub))
        res['paired'][sname]['kappa_minus_rho'] = paired_ci(V('kappa', sub), V('rho', sub), L(sub))
        res['tau'][sname]['nusa_vs_rho'] = ktau(V('nusa', sub), V('rho', sub))
    t1 = {s: res['tau'][s]['A1']['tau'] for s in ('E12', 'S11')}; pc = {s: res['paired'][s]['AS_SING_minus_A1'] for s in ('E12', 'S11')}
    if all(v >= 0.6 for v in t1.values()): outcome = 'KILL-R3'
    elif all(t1[s] < 0.3 and pc[s]['pct'][0] > 0 for s in t1): outcome = 'ENERGY-ACCOUNTS'
    else: outcome = 'PARTIAL'
    sens = {k: all(res['tau'][s][k]['tau'] >= 0.6 for s in ('E12', 'S11')) for k in ('A2', 'A3', 'A1p')}
    res['outcome'] = dict(A1=outcome, tau_A1=t1, paired_lower_pct={s: pc[s]['pct'][0] for s in pc}, paired_lower_bca={s: pc[s]['bca'][0] for s in pc},
                          variants_tau_ge_0p6_both_sets=sens, direction_dependence_must_be_stated=bool(outcome != 'KILL-R3' and any(sens.values())),
                          A4_tau_lt_0p3_both=all(res['tau'][s]['A4']['tau'] < 0.3 for s in ('E12', 'S11')),
                          kappa_minus_rho_CI_excludes_0={s: not (res['paired'][s]['kappa_minus_rho']['pct'][0] <= 0 <= res['paired'][s]['kappa_minus_rho']['pct'][1]) for s in sets})
    dump('A_equal_energy.json', res)
    for s in sets: print(s, {k: round(v['tau'], 3) if isinstance(v, dict) else round(v, 3) for k, v in res['tau'][s].items()})
    print(json.dumps(res['outcome'], indent=1))


if __name__ == '__main__':
    main()
