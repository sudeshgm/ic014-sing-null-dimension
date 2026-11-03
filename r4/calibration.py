"""PREREG_R4B idea 2: semi-synthetic calibration of the estimators in worlds C (confound), M (mediation), D (dissociation).
Bases ResNet50, DINOv2-L, ResNeXt101 (released translators, class text). z = mu_y + r rescaled as C' = a mu_y + b r; 12 synthetic
models per world with the E12 rho values as targets.
Residuals (round 5 fix, PREREG_R5.md): with --resid perm (default) the residual r is permuted across all images (seed 99000) before scaling,
so within a class the residuals no longer sum to zero and split-half / leave-one-out class means are not anti-correlated. --resid insample
reproduces the round-4 construction (r = z - in-sample class mean). Outputs r4/results/I2_calibration_v2.json (perm) or
r4/results/I2_calibration.json (insample, round 4). Usage: python r4/calibration.py --feats-root $IC014_FEATS [--resid perm|insample]"""
import argparse, os, sys, json, time, numpy as np, torch
from scipy.stats import t as tdist
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES4, load_models, ktau, sets, dump4
from retrained_a4 import text_emb, null_basis
from retrain_translators import find
from class_perm import class_means
import estimators as est
torch.set_num_threads(os.cpu_count()); BASES = ['resnet', 'dinov2', 'resnext']
RESID_SEED = 99_000
ESTS = ['rho', 'AS', 'A1', 'AS_gauss', 'X', 'D_align', 'NCM_null', 'R']


def slope(x, y):
    x, y = np.log(x), np.log(y); X = np.c_[np.ones(len(x)), x]; b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; n = len(x)
    se = np.sqrt(r @ r / (n - 2) * np.linalg.inv(X.T @ X)[1, 1]); q = tdist.ppf(0.975, n - 2); return dict(a=float(b[1]), ci=[float(b[1] - q * se), float(b[1] + q * se)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--resid', choices=['perm', 'insample'], default='perm')
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    TC = text_emb(a.sing_root); ms, col, _ = load_models(); rho_all = col('rho'); S = sets(ms, rho_all)
    targ = sorted(float(rho_all[ms.index(n)]) for n in S['E12']); rmin = targ[0]; rD = float(np.median(targ)); h = 0.5 * rmin
    out = dict(targets=targ, rho_D=rD, L_C=h, resid_frac_M=h, resid=a.resid, resid_seed=RESID_SEED if a.resid == 'perm' else None, bases={})
    for base in BASES:
        d = np.load(find(fr, base)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long(); Vn = null_basis(torch.tensor(d['W']))
        A = torch.tensor(d['A']).double(); Ab = torch.tensor(d['Ab']).double(); tc = TC[y]
        C = F @ Vn; Frow = F - C @ Vn.T; Zrow = Frow @ A.T + Ab; AVt = Vn.T @ A.T; mu = class_means(C, y)[y]; Rr = C - mu
        if a.resid == 'perm': Rr = Rr[torch.tensor(np.random.default_rng(RESID_SEED).permutation(len(y)))]
        Er = float(Frow.pow(2).sum(1).mean()); Mm = float(mu.pow(2).sum(1).mean()); Rm = float(Rr.pow(2).sum(1).mean()); hv = est.halves(y); k = min(20, C.shape[1] // 2)
        worlds = {}
        for W in ['C', 'M', 'D']:
            rows = []
            for j in range(12):
                rho = rD if W == 'D' else targ[j]; Et = Er / (1 - rho)
                L = h if W == 'C' else (rho - h if W == 'M' else rD * (j + 0.5) / 12)
                a2 = L * Et / Mm; b2 = (rho - L) * Et / Rm; Cs = a2 ** 0.5 * mu + b2 ** 0.5 * Rr
                t0 = time.time(); v = dict(world=W, L=L, rho_target=rho)
                v['rho'] = float(Cs.pow(2).sum(1).mean()) / (Er + float(Cs.pow(2).sum(1).mean()))
                v['AS'] = est.AS(Zrow, Cs, AVt, tc); v['A1'] = est.A1(Zrow, Cs, AVt, tc, v['rho']); v['AS_gauss'] = est.AS_gauss(Zrow, Cs, AVt, tc, 2)
                v['X'] = v['AS'] / v['AS_gauss']; v['D_align'] = est.D_align(Zrow, Cs, AVt, y, tc, 2)[0]
                v['NCM_null'] = est.ncm_loo(Cs, y)['NCM_null']; v['R'] = est.selectivity(Zrow, Cs, AVt, y, tc, k, ndraw=5, nboot=0, halfA=hv)['R']
                v['secs'] = time.time() - t0; rows.append(v)
            # tau: world C against rho (L constant), M and D against L
            ref = 'rho' if W == 'C' else 'L'; tau = {}
            for e in ESTS:
                xs = [r_[e] for r_ in rows]; ys = [r_[ref] for r_ in rows]
                tau[e] = float('nan') if (np.ptp(xs) < 1e-12 * max(1e-300, np.abs(xs).max()) or np.ptp(ys) == 0) else float(ktau(xs, ys))
            worlds[W] = dict(rows=rows, tau_vs=ref, tau=tau, exponent=slope([r_['rho'] for r_ in rows], [r_['AS'] for r_ in rows]) if W != 'D' else None)
            print(base, W, {e: round(t, 3) for e, t in tau.items()}, worlds[W]['exponent'], round(sum(r_['secs'] for r_ in rows), 1), flush=True)
        out['bases'][base] = dict(worlds=worlds, E_row=Er, M=Mm, Rr=Rm, L_base=Mm / float(F.pow(2).sum(1).mean()), k=k)
    diag = {}
    for e in ESTS:
        ok = []
        for b in BASES:
            t = out['bases'][b]['worlds']
            c, m, dd = t['C']['tau'][e], t['M']['tau'][e], t['D']['tau'][e]
            ok.append(bool(np.isfinite(c) and abs(c) <= 0.2 and np.isfinite(m) and m >= 0.8 and np.isfinite(dd) and dd >= 0.6))
        diag[e] = all(ok)
    out['content_diagnostic'] = diag
    ASm = col('AS_SING'); e12 = S['E12']; out['cross_model_exponent_E12'] = slope([rho_all[ms.index(n)] for n in e12], [ASm[ms.index(n)] for n in e12]); out['ahat_E1'] = 0.431
    dump4('I2_calibration_v2.json' if a.resid == 'perm' else 'I2_calibration.json', out); print('content-diagnostic:', diag, 'cross-model exponent', out['cross_model_exponent_E12'])


if __name__ == '__main__':
    main()
