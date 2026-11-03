"""Post hoc, descriptive (round-8 exploration, NOT pre-registered): rho = rho_offset + rho_between + rho_within on the 13 models.
Exact plug-in identity with weights n_c/N:  E||z||^2 = ||zbar||^2 + (1/N) sum_c n_c ||m_c - zbar||^2 + (1/N) sum_i ||z_i - m_{y_i}||^2,
z = Vn^T f (uncentred, as in rho and r4/calibration.py), each term divided by E||f||^2.
Also: rho_between_debiased = rho_between - (within-class noise contribution to plug-in class means), i.e. subtract
(1/N) sum_c n_c * tr(S_c)/(n_c) with S_c the unbiased within-class covariance (=> (1/N) sum_c tr(S_c)). No proposed round-8 score is computed.
Reads the private cached features (read only) from $IC014_FEATS."""
import os, sys, json, numpy as np, torch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); FR = os.environ['IC014_FEATS']  # private cached features
sys.path.insert(0, os.path.join(R, 'r3')); sys.path.insert(0, os.path.join(R, 'r4'))
from retrain_translators import find
from common import load_models
torch.set_num_threads(8)

def null_basis(W):  # identical to r4/retrained_a4.null_basis (absolute tolerance 1e-5)
    _, sv, vt = torch.linalg.svd(W.double(), full_matrices=True); return vt.T[:, int((sv > 1e-5).sum()):]

ms, col, sing = load_models(); rho_tab = dict(zip(ms, col('rho')))
out = {}
for name in ms:
    d = np.load(find(FR, name)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long()
    Vn = null_basis(torch.tensor(d['W'])); Z = F @ Vn; N = len(y); Etot = float(F.pow(2).sum(1).mean())
    K = int(y.max()) + 1; n = torch.bincount(y, minlength=K).double()
    M = torch.zeros(K, Z.shape[1], dtype=Z.dtype).index_add_(0, y, Z) / n.clamp_min(1)[:, None]
    zbar = Z.mean(0)
    off = float(zbar @ zbar); btw = float((n[:, None] * (M - zbar).pow(2)).sum() / N); wit = float((Z - M[y]).pow(2).sum() / N)
    En = float(Z.pow(2).sum(1).mean())
    # debias: plug-in class mean carries tr(Sigma_c)/n_c of noise; sum_c n_c * tr(S_c)/n_c / N, S_c unbiased (ddof 1)
    ssw = torch.zeros(K, dtype=Z.dtype).index_add_(0, y, (Z - M[y]).pow(2).sum(1))
    noise = float((ssw / (n - 1).clamp_min(1)).sum() / N)
    r = dict(rho=En / Etot, rho_table=float(rho_tab[name]), d_null=int(Vn.shape[1]), rho_offset=off / Etot, rho_between=btw / Etot,
             rho_within=wit / Etot, identity_abs_err=abs(En - off - btw - wit) / Etot, rho_between_debiased=(btw - noise) / Etot,
             share_offset=off / En if En > 0 else float('nan'), share_between=btw / En if En > 0 else float('nan'),
             share_within=wit / En if En > 0 else float('nan'), images_per_class=sorted(set(n.long().tolist())))
    out[name] = r; print(name, {k: (f'{v:.4g}' if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'rho_decomp.json'), 'w'), indent=1)
