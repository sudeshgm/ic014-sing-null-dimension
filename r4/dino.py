"""PREREG_R4 DN: true DINO ViT-B/16 and ViT-S/16 (official linear heads) and a supervised timm ViT-B/16 (AugReg) control.
Features are private ($IC014_FEATS/r3_dino_feats/<model>_10k.npz: F, W, b, pred, files, labels). Ridge translator to CLIP image
embeddings with the r3 G pipeline unchanged; test-half scores AS_ret, A1_ret, kappa_ret, AS_perm_ret, AS_permc_ret; Q-DN1 / Q-DN2.
Only aggregate numbers are written (r4/results/DN_dino.json); no per-image output. Usage: python r4/dino.py --feats-root $IC014_FEATS"""
import argparse, os, sys, json, time, numpy as np, torch
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES3, RES4, load_models, sets, dump4, RHO0
from retrained_a4 import text_emb, split, retrain, null_basis
from retrain_translators import ang, cosm
torch.set_num_threads(os.cpu_count())
MODELS = {'dino_vitb16': 78.2, 'dino_vits16': 77.0, 'timm_vitb16_augreg': None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    dd = f'{fr}/r3_dino_feats'; meta = json.load(open(f'{dd}/meta.json'))
    subset = [l.split('\t')[0].strip() for l in open(os.path.join(R, 'scripts/subset_files_10k.txt')) if l.strip()]
    ylab = [int(l.split('\t')[1]) for l in open(os.path.join(R, 'scripts/subset_files_10k.txt')) if l.strip()]
    TC = text_emb(a.sing_root); Y = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double(); tr, te = split()
    out = {}; kill = {}
    for name, doc in MODELS.items():
        t0 = time.time(); d = np.load(f'{dd}/{name}_10k.npz', allow_pickle=True)
        files = [os.path.basename(str(f)).split('.')[0] for f in d['files']]
        if files != subset or [int(v) for v in d['labels']] != ylab: raise SystemExit(f'{name}: row order / labels do not match the subset list')
        F = torch.tensor(d['F']).double(); W = torch.tensor(d['W']).double(); b = torch.tensor(d['b']).double(); y = torch.tensor(d['labels'].astype(np.int64))
        agree = float(((F @ W.T + b).argmax(1).numpy() == d['pred'].astype(np.int64)).mean())
        top50 = meta['models'][name]['top1_50k']; ok = agree >= 0.999 and (doc is None or abs(top50 - doc) <= 1.0)
        kill[name] = dict(pred_agreement_10k=agree, top1_50k=top50, documented=doc, top1_10k=float((d['pred'].astype(np.int64) == y.numpy()).mean() * 100), pass_=ok)
        Vn = null_basis(W); nd = Vn.shape[1]; o = dict(m=F.shape[1], null_dim=nd, fidelity=None)
        o['rho'] = float((F @ Vn).pow(2).sum(1).mean() / F.pow(2).sum(1).mean()) if nd else 0.0
        if nd and ok:
            Wt, bt, lb, sel = retrain(F, Y, tr); Fe = F[te]; ye = y[te]; tc = TC[ye]; Z = Fe @ Wt + bt; C = Fe @ Vn; Frow = Fe - C @ Vn.T
            Rn = C @ Vn.T @ Wt; Zrow = Z - Rn; a0 = ang(Z, tc); arow = ang(Zrow, tc); AVt = Vn.T @ Wt
            beta = (RHO0 / o['rho']) ** 0.5; Erow = float(Frow.pow(2).sum(1).mean()); En = float(C.pow(2).sum(1).mean())
            o.update(lambda_rel=lb, fidelity=cosm(Z, Y[te]), AS_ret=float((a0 - arow).abs().mean()), A1_ret=float((a0 - ang(Z - beta * Rn, tc)).abs().mean()),
                     kappa_ret=float(Rn.pow(2).sum(1).mean() / Z.pow(2).sum(1).mean()))
            o['gamma_ret'] = o['kappa_ret'] / o['rho']
            for kind, seed in [('perm', 92_000), ('permc', 93_000)]:
                v = []
                for r in range(3):
                    g = np.random.default_rng(seed + r); idx = np.arange(len(te))
                    if kind == 'perm': idx = g.permutation(len(te))
                    else:
                        yy = ye.numpy()
                        for c in np.unique(yy):
                            ii = np.where(yy == c)[0]; idx[ii] = ii[g.permutation(len(ii))]
                    Ct = C[torch.tensor(idx)]; lam = (En / float(Ct.pow(2).sum(1).mean())) ** 0.5
                    v.append(float((ang(Zrow + lam * (Ct @ AVt), tc) - arow).abs().mean()))
                o[f'AS_{kind}_ret'] = float(np.mean(v))
        o['secs'] = time.time() - t0; out[name] = o; print(name, kill[name], {k: (round(v, 5) if isinstance(v, float) else v) for k, v in o.items()}, flush=True)
    # comparison set: the 13 models' retrained values
    ms, col, sing = load_models(); G = json.load(open(os.path.join(RES3, 'G_retrain_permodel.json'))); S = sets(ms, col('rho'))
    as13 = {n: G[n]['AS_SING_retrained'] for n in ms}; rho13 = {n: G[n]['rho'] for n in ms}
    dn = [n for n in ('dino_vitb16', 'dino_vits16') if out[n].get('AS_ret') is not None]
    res = dict(models=out, kill_dn=kill, KILL_DN=not all(kill[n]['pass_'] for n in ('dino_vitb16', 'dino_vits16')))
    if len(dn) == 2:
        allv = sorted(list(as13.values()) + [out[n]['AS_ret'] for n in dn]); med13 = float(np.median(list(as13.values())))
        rk = {n: int(sum(v < out[n]['AS_ret'] for v in allv)) + 1 for n in dn}
        q1 = ('SING-consistent' if all(rk[n] <= 4 for n in dn) else 'SING-inconsistent' if all(out[n]['AS_ret'] > med13 for n in dn) else 'mixed')
        x = np.log([rho13[n] for n in S['E12']]); yv = np.log([as13[n] for n in S['E12']]); n_ = len(x); X = np.c_[np.ones(n_), x]
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None); resid = yv - X @ beta; s2 = resid @ resid / (n_ - 2); XtXi = np.linalg.inv(X.T @ X); tq = tdist.ppf(0.975, n_ - 2)
        pi = {}
        for n in list(dn) + [m for m in ('timm_vitb16_augreg',) if out[m].get('AS_ret') is not None]:
            x0 = np.array([1, np.log(out[n]['rho'])]); yh = float(x0 @ beta); hw = float(tq * np.sqrt(s2 * (1 + x0 @ XtXi @ x0)))
            pi[n] = dict(log_AS_ret=float(np.log(out[n]['AS_ret'])), pred=yh, lo=yh - hw, hi=yh + hw, inside=bool(yh - hw <= np.log(out[n]['AS_ret']) <= yh + hw),
                         AS_pred_deg=float(np.exp(yh)), AS_pi_deg=[float(np.exp(yh - hw)), float(np.exp(yh + hw))])
        q2 = 'energy-consistent' if all(pi[n]['inside'] for n in dn) else 'energy-inconsistent' if not any(pi[n]['inside'] for n in dn) else 'mixed'
        res.update(rank_among_15=rk, median_AS_ret_13=med13, Q_DN1=q1, fit_E12=dict(a=float(beta[0]), b=float(beta[1]), s=float(np.sqrt(s2)), n=n_),
                   prediction=pi, Q_DN2=q2, AS_ret_13=as13, rho_13=rho13)
    dump4('DN_dino.json', res); print(json.dumps({k: res[k] for k in res if k not in ('models', 'AS_ret_13', 'rho_13')}, indent=1, default=float))


if __name__ == '__main__':
    main()
