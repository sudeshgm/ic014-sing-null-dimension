"""PREREG_R4C B4: our measured IS (fix2_models.csv) against row-space energy 1-rho and kappa_row = E||A R f||^2 / E||Tf||^2.
Outputs r4/results/B4_is_energy.json. Usage: python r4/is_energy.py --feats-root $IC014_FEATS"""
import argparse, os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import load_models, tau_ci, sets, dump4
from retrained_a4 import null_basis
from retrain_translators import find
torch.set_num_threads(os.cpu_count())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); a = ap.parse_args()
    ms, col, sing = load_models(); rho = col('rho'); IS = col('IS'); S = sets(ms, rho); kr = {}
    for name in ms:
        d = np.load(find(a.feats_root, name)); F = torch.tensor(d['F']).double(); Vn = null_basis(torch.tensor(d['W'])); A = torch.tensor(d['A']).double(); Ab = torch.tensor(d['Ab']).double()
        Fr = F - (F @ Vn) @ Vn.T; kr[name] = float((Fr @ A.T).pow(2).sum(1).mean() / (F @ A.T + Ab).pow(2).sum(1).mean()); print(name, kr[name], flush=True)
    res = dict(per_model={n: dict(IS=float(IS[i]), one_minus_rho=float(1 - rho[i]), kappa_row=kr[n]) for i, n in enumerate(ms)}, tau_IS_rowenergy={}, tau_IS_kappa_row={}, tau_IS_sing={})
    for s, sub in S.items():
        ix = [ms.index(n) for n in sub]
        res['tau_IS_rowenergy'][s] = tau_ci(IS[ix], 1 - rho[ix]); res['tau_IS_kappa_row'][s] = tau_ci(IS[ix], np.array([kr[n] for n in sub])); res['tau_IS_sing'][s] = tau_ci(IS[ix], -sing[ix])
    res['energy_proxy'] = all(abs(res['tau_IS_rowenergy'][s]['tau']) >= 0.6 for s in S)
    dump4('B4_is_energy.json', res); print({k: v for k, v in res.items() if k != 'per_model'})


if __name__ == '__main__':
    main()
