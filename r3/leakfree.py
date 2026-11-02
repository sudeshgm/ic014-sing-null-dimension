"""PREREG_R3 B (N4): leak-free null surrogate and planted leakage.
Surrogate: each image's null coordinates c_i = V_n^T f_i are replaced by c~_i ~ N(mu, Sigma) (mean/covariance of c over
the 10k images), drawn independently of the image (5 draws, seed 70000+r). Row-space part (all logits) unchanged.
B-core: ResNet50, ResNeXt101, DINOv2-L; holdout concepts {water, green, thin, knitted}; target t_c (SING attribute text,
'mean' prompt); plant u_c = normalised null projection of A^+ t_c; s_i = standardised CLIP concept z-scores; sigma =
sqrt(E||f||^2). f' = f_row + lam*V_n c~ + alpha*sigma*s_i*u_c, lam so that the leak-free part has null energy fraction
rho_t in {0.01, 0.05, 0.2, 0.45} (+ the model's own rho), alpha in {0, 0.1, 0.3}. AS = mean|theta(Tf',t) - theta(T f_row,t)|.
N4 pass (each of the 3 models): at alpha=0, Spearman(AS, rho_t) >= 0.8 over the 4 levels, and the OLS slope of log AS on
log rho_t (4 levels x 4 concepts) has a 1000x image-bootstrap 95% CI (seed 0) excluding 0.
B-rank: all 13 models, class text, surrogate at own rho; tau vs SING, X = AS_SING/AS_surr.
Usage: python r3/leakfree.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np, torch
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import R, RES, load_models, tau_ci, ktau, dump, S11_DROP
torch.set_num_threads(os.cpu_count())
CORE = ['resnet', 'resnext', 'dinov2']; HOLD_C = ['water', 'green', 'thin', 'knitted']
LEVELS = [0.01, 0.05, 0.2, 0.45]; ALPHAS = [0.0, 0.1, 0.3]; NDRAWS = 5


def ang(Z, t): return torch.rad2deg(torch.arccos((torch.nn.functional.normalize(Z, dim=-1) * t).sum(-1).clamp(-1, 1)))


def find(fr, name):
    for r in [f'{fr}/e2/ret/e2_feats', f'{fr}/e3/ret/e3_feats']:
        f = os.path.join(r, f'{name}_full.npz')
        if os.path.exists(f): return f


def setup(fr, name):
    d = np.load(find(fr, name)); F = torch.tensor(d['F']).double(); A = torch.tensor(d['A']).double(); Ab = torch.tensor(d['Ab']).double()
    W = torch.tensor(d['W']).double(); _, sv, vt = torch.linalg.svd(W, full_matrices=True); r = int((sv > 1e-5).sum()); Vn = vt.T[:, r:]
    C = F @ Vn; Frow = F - C @ Vn.T; mu = C.mean(0); Cc = C - mu; S = Cc.T @ Cc / len(C)
    ev, U = torch.linalg.eigh(S); Ls = U * ev.clamp_min(0).sqrt()
    E = dict(F=F, A=A, Ab=Ab, Vn=Vn, Frow=Frow, mu=mu, Ls=Ls, y=torch.tensor(d['y']).long(), Etot=float((F ** 2).sum(1).mean()),
             Erow=float((Frow ** 2).sum(1).mean()), rho=float((C ** 2).sum(1).mean() / (F ** 2).sum(1).mean()), AVn=A @ Vn)
    E['Zrow'] = Frow @ A.T + Ab; E['sigma'] = E['Etot'] ** 0.5; return E


def draw(E, r):
    g = torch.Generator().manual_seed(70_000 + r); z = torch.randn(E['F'].shape[0], E['Vn'].shape[1], generator=g, dtype=torch.float64)
    return E['mu'] + z @ E['Ls'].T


def lam_for(E, Ct, rho_t):
    Ec = float((Ct ** 2).sum(1).mean()); return (rho_t / (1 - rho_t) * E['Erow'] / Ec) ** 0.5


def core(fr, sing_root, conc):
    att = np.load(os.path.join(sing_root, 'misc/embeddings/attribute_embeddings_broden_style_fp16.npz'), allow_pickle=True)
    names = [str(c) for c in att['classes']]; pi = [str(p) for p in att['prompts']].index('mean')
    TA = torch.nn.functional.normalize(torch.tensor(att['embeddings'][:, pi].astype(np.float64)), dim=-1)
    cn = [str(c) for c in conc['concepts']]; out = {}
    for name in CORE:
        t0 = time.time(); E = setup(fr, name); levels = LEVELS + [E['rho']]; Apinv = torch.linalg.pinv(E['A'])
        img = {}   # (level_idx, alpha, concept) -> per-image AS averaged over draws
        CT = {}
        for c in HOLD_C:
            t = TA[names.index(c)]; w = Apinv @ t; un = E['Vn'] @ (E['Vn'].T @ w); u = un / un.norm(); Au = E['A'] @ u
            s = torch.tensor(conc['zscore'][:, cn.index(c)]).double(); s = (s - s.mean()) / s.std()
            a0row = ang(E['Zrow'], t[None])
            CT[c] = (t, Au, s, a0row)
        for r in range(NDRAWS):
            Ct = draw(E, r); G = Ct @ E['AVn'].T
            for li, lv in enumerate(levels):
                lam = lam_for(E, Ct, lv)
                for al in ALPHAS:
                    for c in HOLD_C:
                        t, Au, s, a0row = CT[c]
                        Zp = E['Zrow'] + lam * G + (al * E['sigma'] * s)[:, None] * Au[None]
                        v = (ang(Zp, t[None]) - a0row).abs() / NDRAWS
                        img[(li, al, c)] = img.get((li, al, c), 0) + v
            del Ct, G
        # N4 at alpha = 0 on the 4 pre-registered levels
        X = np.stack([np.stack([img[(li, 0.0, c)].numpy() for c in HOLD_C]) for li in range(4)])   # (4 levels, 4 concepts, N)
        def slope(means):
            xs = np.repeat(np.log(LEVELS), len(HOLD_C)); ys = np.log(means.reshape(-1)); return float(np.polyfit(xs, ys, 1)[0])
        pt = X.mean(2); sl = slope(pt); rng = np.random.default_rng(0); N = X.shape[2]; bs = []
        for _ in range(1000):
            wts = np.bincount(rng.integers(0, N, N), minlength=N) / N; bs.append(slope(X @ wts))
        lvl_mean = pt.mean(1); sp = float(spearmanr(LEVELS, lvl_mean).statistic); ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        m = lambda li, al: float(np.mean([img[(li, al, c)].mean().item() for c in HOLD_C]))
        out[name] = dict(own_rho=E['rho'], AS_alpha0_by_level=dict(zip([str(l) for l in LEVELS], lvl_mean.tolist())),
                         AS_alpha0_own_rho=m(4, 0.0), per_concept_alpha0={c: pt[:, k].tolist() for k, c in enumerate(HOLD_C)},
                         spearman=sp, loglog_slope=sl, slope_ci95=ci, pass_model=bool(sp >= 0.8 and (ci[0] > 0 or ci[1] < 0)),
                         planted_ratio={f'rho_t={lab}': {str(al): m(li, al) / m(li, 0.0) for al in ALPHAS} for li, lab in [(2, '0.2'), (4, 'own')]},
                         AS_by_level_alpha={f'{levels[li]:.4g}': {str(al): m(li, al) for al in ALPHAS} for li in range(5)}, secs=time.time() - t0)
        print(name, json.dumps({k: out[name][k] for k in ['spearman', 'loglog_slope', 'slope_ci95', 'pass_model', 'planted_ratio']}), flush=True)
        del E, img
    return out


def rank(fr, TC, ms, P):
    for name in ms:
        if name in P: continue
        t0 = time.time(); E = setup(fr, name); tc = TC[E['y']]; a0row = ang(E['Zrow'], tc); a = 0.0
        for r in range(NDRAWS):
            Ct = draw(E, r); lam = lam_for(E, Ct, E['rho']); a += float((ang(E['Zrow'] + lam * (Ct @ E['AVn'].T), tc) - a0row).abs().mean()) / NDRAWS
        # the real null content for comparison (should reproduce AS_SING)
        real = float((ang(E['F'] @ E['A'].T + E['Ab'], tc) - a0row).abs().mean())
        P[name] = dict(AS_surr=a, AS_SING_check=real, rho=E['rho'], secs=time.time() - t0); print(name, P[name], flush=True); del E
        json.dump(P, open(os.path.join(RES, 'B_rank_permodel.json'), 'w'), indent=1)
    return P


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); ap.add_argument('--skip-core', action='store_true'); a = ap.parse_args()
    conc = np.load(f'{a.feats_root}/e3/private/clip_concepts.npz')
    cpath = os.path.join(RES, 'B_core.json')
    C = json.load(open(cpath)) if (a.skip_core and os.path.exists(cpath)) else core(a.feats_root, a.sing_root, conc)
    json.dump(C, open(cpath, 'w'), indent=1)
    emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float64)), dim=-1).mean(1), dim=-1)
    ms, col, sing = load_models(); lk = -sing; AS = col('AS_SING')
    pp = os.path.join(RES, 'B_rank_permodel.json'); P = rank(a.feats_root, TC, ms, json.load(open(pp)) if os.path.exists(pp) else {})
    sur = np.array([P[n]['AS_surr'] for n in ms]); X = AS / sur; sc = np.array([n not in S11_DROP for n in ms])
    res = dict(core=C, N4_pass=all(C[n]['pass_model'] for n in CORE), rank=P,
               tau_AS_surr=dict(all13=tau_ci(sur, lk), S11=tau_ci(sur[sc], lk[sc])),
               spearman_AS_surr_AS_SING=float(spearmanr(sur, AS).statistic), tau_AS_surr_AS_SING=ktau(sur, AS),
               tau_X=dict(all13=tau_ci(X, lk), S11=tau_ci(X[sc], lk[sc])), X={n: float(x) for n, x in zip(ms, X)})
    res['reading'] = dict(surr_reproduces_order=bool(res['tau_AS_surr']['all13']['tau'] >= 0.6 and res['tau_AS_surr']['S11']['tau'] >= 0.6),
                          X_claim_allowed=bool(all(res['tau_X'][s]['pct'][0] > 0.6 or res['tau_X'][s]['pct'][1] < 0 for s in ('all13', 'S11'))))
    dump('B_leakfree.json', res)
    print('N4_pass', res['N4_pass'], 'tau_surr', {k: round(v['tau'], 3) for k, v in res['tau_AS_surr'].items()},
          'tau_X', {k: (round(v['tau'], 3), [round(x, 2) for x in v['pct']]) for k, v in res['tau_X'].items()}, res['reading'])


if __name__ == '__main__':
    main()
