"""PREREG_R4C B1: geometric account of AS. In-plane / out-of-plane split of the removed translated null part, closed forms,
gain slope s -> a_pred = (1+s)/2, E1-style cell fit a_fit. G1: 13 models, released translators (exploratory, post hoc).
G2: DINO ViT-B/16, ViT-S/16 (+AugReg, empty null space) with ridge translators on the test half (confirmatory). G3: synthetic
calibration of a. Outputs r4/results/B1_geometry.json. Usage: python r4/geometry.py --feats-root $IC014_FEATS [--part G1|G2|G3|all]"""
import argparse, os, sys, json, math, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES4, load_models, tau_ci, ktau, sets, dump4
from retrained_a4 import text_emb, split, retrain, null_basis
from retrain_translators import ang, find
from scipy.stats import spearmanr
torch.set_num_threads(os.cpu_count()); nz = torch.nn.functional.normalize


def grad_theta(Z, tc):
    zn = Z.norm(dim=1, keepdim=True); zh = Z / zn; cos = (zh * tc).sum(1, keepdim=True).clamp(-1 + 1e-9, 1 - 1e-9); sin = (1 - cos ** 2).sqrt()
    return -(tc - cos * zh) / (zn * sin)


def split_plane(Z, D, tc):
    e1 = nz(Z, dim=1); e2 = nz(tc - (tc * e1).sum(1, keepdim=True) * e1, dim=1)
    Din = (D * e1).sum(1, keepdim=True) * e1 + (D * e2).sum(1, keepdim=True) * e2; return Din, D - Din


def decomposition(Z, D, tc, boot=1000):
    a0 = ang(Z, tc); Din, Doop = split_plane(Z, D, tc); f = lambda Rm: float((a0 - ang(Z - Rm, tc)).abs().mean())
    sgn = ang(Z - Doop, tc) - a0; cos = (nz(Z, dim=1) * tc).sum(1).clamp(-1 + 1e-9, 1 - 1e-9); cot = cos / (1 - cos ** 2).sqrt()
    cf = torch.rad2deg(0.5 * cot * Doop.pow(2).sum(1) / Z.pow(2).sum(1)); small = (Doop.norm(dim=1) / Z.norm(dim=1)) < 0.5
    rng = np.random.default_rng(0); s = sgn.numpy(); bs = [s[rng.integers(0, len(s), len(s))].mean() for _ in range(boot)]
    meas_s, cf_s = float(sgn[small].mean()), float(cf[small].mean())
    return dict(AS=f(D), AS_in=f(Din), AS_oop=f(Doop), oop_share=f(Doop) / max(f(D), 1e-30), signed_oop=float(sgn.mean()), signed_oop_ci=[float(v) for v in np.percentile(bs, [2.5, 97.5])],
                oop_closed_form=float(cf.mean()), frac_small=float(small.float().mean()), oop_meas_small=meas_s, oop_cf_small=cf_s,
                oop_rel_err_small=abs(cf_s - meas_s) / abs(meas_s) if meas_s != 0 else float('nan'))


def gauss_closed(Zrow, AVt, tc, mu, Sig, lam):
    G = grad_theta(Zrow, tc); H = G @ AVt.T; m = lam * (H @ mu); s = lam * ((H @ Sig) * H).sum(1).clamp_min(0).sqrt()
    fm = s * math.sqrt(2 / math.pi) * torch.exp(-m ** 2 / (2 * s ** 2 + 1e-300)) + m * (1 - 2 * torch.distributions.Normal(0, 1).cdf((-m / (s + 1e-300)).float()).double())
    return float(torch.rad2deg(fm).mean())


def gain_slope(Z, C, AVt, tc):
    G = grad_theta(Z, tc); H = G @ AVt.T; M2 = C.T @ C / len(C); ev, U = torch.linalg.eigh(M2); o = torch.argsort(ev, descending=True); ev, U = ev[o], U[:, o]
    gk = ((H @ U) ** 2).mean(0); keep = ev > ev[0] * 1e-8
    if keep.sum() <= 5: return float('nan'), int(keep.sum())
    return float(np.polyfit(torch.log(ev[keep]).numpy(), torch.log(gk[keep]).numpy(), 1)[0]), int(keep.sum())


def e1_cells(Z, C, AVt, tc, Etot, ds=(24, 100, 280), nrand=5):
    nd = C.shape[1]; M2 = C.T @ C / len(C); ev, U = torch.linalg.eigh(M2); U = U[:, torch.argsort(ev, descending=True)]; a0 = ang(Z, tc); cells = []
    def cell(Us, d, kind):
        P = C @ Us; rho = float(P.pow(2).sum(1).mean()) / Etot; asv = float((a0 - ang(Z - P @ (AVt.T @ Us).T, tc)).abs().mean()); cells.append(dict(d=d, kind=kind, rho=rho, AS=asv))
    for d in ds:
        if d >= nd: continue
        cell(U[:, :d], d, 'top'); cell(U[:, -d:], d, 'bottom')
        for s in range(nrand):
            g = torch.Generator().manual_seed(1000 * d + s); cell(torch.linalg.qr(torch.randn(nd, d, generator=g, dtype=C.dtype))[0], d, 'random')
    ok = [c for c in cells if c['rho'] > 0 and c['AS'] > 0]
    if len({c['d'] for c in ok}) < 1 or len(ok) < 4: return float('nan'), cells
    X = np.c_[np.log([c['rho'] for c in ok]), np.log([c['d'] for c in ok]), np.ones(len(ok))]
    if len({c['d'] for c in ok}) == 1: X = X[:, [0, 2]]
    b, *_ = np.linalg.lstsq(X, np.log([c['AS'] for c in ok]), rcond=None); return float(b[0]), cells


def model_block(Z, Zrow, C, AVt, tc, Etot, rho, with_surr=False):
    o = decomposition(Z, C @ AVt, tc); s, npc = gain_slope(Z, C, AVt, tc); o.update(gain_slope_s=s, n_pc=npc, a_pred=(1 + s) / 2)
    o['a_fit'], o['cells'] = e1_cells(Z, C, AVt, tc, Etot); mu = C.mean(0); Cc = C - mu; Sig = Cc.T @ Cc / len(C)
    o['AS_gauss_own'] = gauss_closed(Zrow, AVt, tc, mu, Sig, 1.0)
    if with_surr:
        ev, Uq = torch.linalg.eigh(Sig); Ls = Uq * ev.clamp_min(0).sqrt(); Erow = Etot * (1 - rho); arow = ang(Zrow, tc); lv = {}
        for rt in (0.01, 0.05, 0.2, 0.45):
            v = []; lams = []
            for r in range(2):
                g = torch.Generator().manual_seed(70_000 + r); Ct = mu + torch.randn(C.shape, generator=g, dtype=C.dtype) @ Ls.T
                lam = (rt / (1 - rt) * Erow / float(Ct.pow(2).sum(1).mean())) ** 0.5; lams.append(lam); v.append(float((ang(Zrow + lam * (Ct @ AVt), tc) - arow).abs().mean()))
            lv[str(rt)] = dict(AS_surr=float(np.mean(v)), AS_gauss=gauss_closed(Zrow, AVt, tc, mu, Sig, float(np.mean(lams))))
        o['levels'] = lv; o['spearman_gauss_surr'] = float(spearmanr([lv[k]['AS_gauss'] for k in lv], [lv[k]['AS_surr'] for k in lv]).statistic)
    return o


def g3():
    out = {}
    for s in (-0.4, -0.2, 0.0, 0.2):
        g = torch.Generator().manual_seed(98_000); N, nd, dc, K = 5000, 300, 64, 20
        y = torch.arange(N) % K; T = nz(torch.randn(K, dc, generator=g, dtype=torch.float64), dim=1); tc = T[y]
        w = torch.randn(N, dc, generator=g, dtype=torch.float64); w = nz(w - (w * tc).sum(1, keepdim=True) * tc, dim=1); Zrow = 0.3 * tc + math.sqrt(1 - 0.09) * w
        lam = 1.0 / torch.arange(1, nd + 1, dtype=torch.float64); C = torch.randn(N, nd, generator=g, dtype=torch.float64) * lam.sqrt()
        Q = nz(torch.randn(nd, dc, generator=g, dtype=torch.float64), dim=1) * (lam ** (s / 2))[:, None]
        sc = 0.3 / float((C @ Q).norm(dim=1).mean()); AVt = Q * sc; En = float(C.pow(2).sum(1).mean()); Etot = En / 0.3
        Z = Zrow + C @ AVt; s_hat, _ = gain_slope(Z, C, AVt, tc); a_fit, cells = e1_cells(Z, C, AVt, tc, Etot)
        out[str(s)] = dict(s=s, a_target=(1 + s) / 2, s_hat=s_hat, a_pred_hat=(1 + s_hat) / 2, a_fit=a_fit, err=a_fit - (1 + s) / 2, n_cells=len(cells))
        print('G3', out[str(s)], flush=True)
    errs = [abs(v['err']) for v in out.values()]
    return dict(per_s=out, verdict='pass' if all(e <= 0.05 for e in errs) else ('fail' if sum(e > 0.1 for e in errs) >= 2 else 'partial'))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--part', default='all')
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    TC = text_emb(a.sing_root); path = os.path.join(RES4, 'B1_geometry.json'); res = json.load(open(path)) if os.path.exists(path) else {}
    if a.part in ('G2', 'all'):
        Y = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double(); tr, te = split(); G2 = {}
        for name in ('dino_vitb16', 'dino_vits16', 'timm_vitb16_augreg'):
            d = np.load(f'{fr}/r3_dino_feats/{name}_10k.npz', allow_pickle=True); F = torch.tensor(d['F']).double(); y = torch.tensor(d['labels'].astype(np.int64)); Vn = null_basis(torch.tensor(d['W']))
            if Vn.shape[1] == 0: G2[name] = dict(null_dim=0, testable=False); print(name, 'empty null space', flush=True); continue
            Wt, bt, _, _ = retrain(F, Y, tr); Fe = F[te]; tc = TC[y[te]]; C = Fe @ Vn; AVt = Vn.T @ Wt; Z = Fe @ Wt + bt; Zrow = Z - C @ AVt
            Etot = float(Fe.pow(2).sum(1).mean()); rho = float(C.pow(2).sum(1).mean()) / Etot
            o = model_block(Z, Zrow, C, AVt, tc, Etot, rho, with_surr=True); o['rho_test'] = rho; o['testable'] = True
            o['i'] = o['spearman_gauss_surr'] >= 0.9; o['ii'] = o['signed_oop_ci'][0] > 0; o['iii'] = o['oop_rel_err_small'] <= 0.25; o['iv_err'] = o['a_fit'] - o['a_pred']
            G2[name] = o; print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in o.items() if k not in ('cells', 'levels')}, flush=True)
        T = [n for n in G2 if G2[n].get('testable')]
        G2['verdict'] = dict(pass_=all(G2[n]['i'] and G2[n]['ii'] and G2[n]['iii'] for n in T), kill=all(not G2[n]['ii'] for n in T) or all(G2[n]['spearman_gauss_surr'] < 0.7 for n in T),
                             exponent=('predicted' if all(abs(G2[n]['iv_err']) <= 0.1 for n in T) else 'not predicted' if all(abs(G2[n]['iv_err']) > 0.1 for n in T) else 'mixed'), tested=T)
        res['G2'] = G2; dump4('B1_geometry.json', res); print('G2 verdict', G2['verdict'])
    if a.part in ('G3', 'all'):
        res['G3'] = g3(); dump4('B1_geometry.json', res); print('G3 verdict', res['G3']['verdict'])
    if a.part in ('G1', 'all'):
        ms, col, sing = load_models(); lk = -sing; rho = col('rho'); S = sets(ms, rho); P = {}
        for name in ms:
            d = np.load(find(fr, name)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long(); Vn = null_basis(torch.tensor(d['W']))
            A = torch.tensor(d['A']).double(); Ab = torch.tensor(d['Ab']).double(); tc = TC[y]; C = F @ Vn; AVt = Vn.T @ A.T; Z = F @ A.T + Ab; Zrow = Z - C @ AVt
            Etot = float(F.pow(2).sum(1).mean()); P[name] = model_block(Z, Zrow, C, AVt, tc, Etot, rho[ms.index(name)])
            print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in P[name].items() if k != 'cells'}, flush=True); del F, C, Z
        L = lambda sub: np.array([lk[ms.index(n)] for n in sub]); V = lambda k, sub: np.array([P[n][k] for n in sub])
        G1 = dict(per_model=P, label='exploratory (post hoc)', tau={}, tau_vs_rho={})
        for s, sub in S.items():
            G1['tau'][s] = {k: tau_ci(V(k, sub), L(sub)) for k in ['AS_in', 'AS_oop', 'AS_gauss_own', 'oop_share']}
            G1['tau_vs_rho'][s] = {k: ktau(V(k, sub), np.array([rho[ms.index(n)] for n in sub])) for k in ['AS_in', 'AS_oop', 'AS_gauss_own']}
        res['G1'] = G1; dump4('B1_geometry.json', res); print({s: {k: round(v['tau'], 3) for k, v in G1['tau'][s].items()} for s in S})


if __name__ == '__main__':
    main()
