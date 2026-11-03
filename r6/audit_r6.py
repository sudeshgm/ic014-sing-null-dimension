"""PREREG_R6 driver: the four published scores (ViM, ActSub S->, CORE S_mem, conditional V-information) plus descriptive variants and the
NuSA-style reference, on each model under the uniform tests E1 (equal energy), E2 (world B), C1 (across-image permutation of the null part),
C2 (within-class permutation), C3 (class-code plants), and on the synthetic worlds C / M / D of the three calibration bases.
Every intervention changes only the null-space component Pi f, so the logits are unchanged (checked). Aggregates only (per model) in
r6/results/. CPU, numpy + torch.
Usage: python r6/audit_r6.py --feats-root $IC014_FEATS --models resnet,dinov2 [--worlds resnet] [--threads 2]"""
import argparse, os, sys, json, time, itertools, numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
sys.path.insert(0, H); sys.path.insert(0, os.path.join(ROOT, 'r4')); sys.path.insert(0, os.path.join(ROOT, 'r3')); sys.path.insert(0, os.path.join(ROOT, 'r5'))
import scores_r6 as sc  # noqa: E402
from common4 import R, load_models, sets  # noqa: E402
from retrain_translators import find  # noqa: E402
RES6 = os.path.join(ROOT, 'r6', 'results'); NCLS = 1000
HALF_SEED, REF_SEED, PERM_SEED, WPERM_SEED, PLANT_SEED, WORLD_SEED, BOOT_SEED = 100100, 100150, 100200, 100250, 100400, 100500, 100600
RHO0, NPERM, NBOOT = 0.005, 5, 200
_UID = itertools.count()
CAL_BASES = ('resnet', 'dinov2', 'resnext')
PRIMARY = ('ViM', 'ActSub', 'CORE', 'Vinfo'); DESCR = ('ViM_s', 'CORE_ker', 'Vinfo_corr', 'NuSA')


def text(sing_root):
    e = np.load(os.path.join(sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)['embeddings'].astype(np.float64)
    e = e / np.linalg.norm(e, axis=-1, keepdims=True); t = e.mean(1); return t / np.linalg.norm(t, axis=1, keepdims=True)


def load(fr, name):
    d = np.load(find(fr, name)); F = d['F'].astype(np.float64); y = d['y'].astype(int); W = d['W'].astype(np.float64); b = d['bW'].astype(np.float64)
    Vr, Vn = sc.null_split(W); return F, y, W, b, Vr, Vn


class Model:
    def __init__(s, F, y, W, b, Vn):
        s.y, s.W, s.b, s.Vn = y, W, b, Vn; s.C = F @ Vn; s.Frow = F - s.C @ Vn.T; s.L = F @ W.T + b; s.yhat = s.L.argmax(1)
        s.A = sc.halves(y, HALF_SEED); s.Vt = np.linalg.svd(W, full_matrices=True)[2]
        s.En = float((s.C ** 2).sum(1).mean()); s.Er = float((s.Frow ** 2).sum(1).mean()); s.rho = s.En / (s.En + s.Er); s.vinfo = {}; s.uid = next(_UID)

    def feats(s, Cm): return s.Frow + Cm @ s.Vn.T

    def calibrate_vinfo(s):
        """choose lam_rel by inner CV and freeze phi_scale on REAL data, per fit half and target"""
        for which in ('label', 'corr'):
            for h, m in (('A', s.A), ('B', ~s.A)):
                lr = sc.vinfo_choose_lam(s.C[m], s.L[m], s.y[m], which); pm = sc.vinfo_phi_fit(s.C[m]); P = (s.C[m] - pm['mu']) @ pm['V']
                s.vinfo[(which, h)] = dict(lam_rel=lr, phi_scale=float((P ** 2).sum(1).mean()))

    def per_image(s, Cm, vinfo=True):
        """per-image score values; each image is evaluated by the statistics fitted on the other half"""
        F = s.feats(Cm); n = len(s.y); out = {k: np.full(n, np.nan) for k in PRIMARY + DESCR}; meta = {}
        dl = float(np.abs(F @ s.W.T + s.b - s.L).max()); meta['max_dlogit'] = dl
        for h, fit in (('A', s.A), ('B', ~s.A)):
            ev = ~fit
            m = sc.vim_fit(F[fit], s.L[fit], s.W, s.b); v = sc.vim_eval(m, F[ev], s.L[ev]); out['ViM'][ev] = v['p_virtual']; out['ViM_s'][ev] = v['s']
            m = sc.actsub_fit(s.Vt, F[fit], REF_SEED + (h == 'B')); out['ActSub'][ev] = sc.actsub_eval(m, F[ev])['S_insig']; meta[f'c_idx_{h}'] = m['c_idx']
            m = sc.core_fit(F[fit], s.W, s.L[fit], s.y[fit]); out['CORE'][ev] = sc.core_eval(m, F[ev], s.W, s.L[ev])['S_mem']
            m = sc.core_fit(F[fit], s.W, s.L[fit], s.y[fit], ker_only_Vn=s.Vn); out['CORE_ker'][ev] = sc.core_eval(m, F[ev], s.W, s.L[ev])['S_mem']
            if vinfo:
                for which, key in (('label', 'Vinfo'), ('corr', 'Vinfo_corr')):
                    p = s.vinfo[(which, h)]
                    g, lam, _ = sc.vinfo_fit_eval(Cm[fit], s.L[fit], s.y[fit], Cm[ev], s.L[ev], s.y[ev], p['lam_rel'], which, phi_scale=p['phi_scale'], base_key=(s.uid, which, h))
                    out[key][ev] = g; meta[f'lam_{which}_{h}'] = lam
        out['NuSA'] = sc.nusa(F, Cm); return out, meta

    def summary(s, Cm, vinfo=True, boot=False):
        pi, meta = s.per_image(Cm, vinfo); o = {k: float(np.mean(v)) for k, v in pi.items() if np.all(np.isfinite(v))}; o.update(meta)
        o['rho'] = float((Cm ** 2).sum(1).mean() / ((Cm ** 2).sum(1).mean() + s.Er))
        if boot:
            rng = np.random.default_rng(BOOT_SEED); idx = [rng.integers(0, len(s.y), len(s.y)) for _ in range(NBOOT)]
            o['sd_boot'] = {k: float(np.std([v[i].mean() for i in idx], ddof=1)) for k, v in pi.items() if np.all(np.isfinite(v))}
        return o

    def to_rho(s, rho): return s.C * np.sqrt(rho * (s.Er / (1 - rho)) / s.En)


def energy_fix(Cp, e2): return Cp * np.sqrt(e2 / (Cp ** 2).sum(1).mean())


def run_model(fr, name, tx, targ):
    t0 = time.time(); F, y, W, b, Vr, Vn = load(fr, name); M = Model(F, y, W, b, Vn); M.calibrate_vinfo(); C = M.C; d = C.shape[1]
    o = dict(model=name, d=F.shape[1], d_null=d, rho=M.rho, vim_dim=sc.vim_dim(F.shape[1]), vinfo_cal={f'{k[0]}_{k[1]}': v for k, v in M.vinfo.items()})
    o['real'] = M.summary(C, boot=True); print(name, 'real', {k: round(v, 5) for k, v in o['real'].items() if isinstance(v, float)}, round(time.time() - t0, 1), flush=True)
    o['E1'] = M.summary(M.to_rho(RHO0))
    o['E2'] = [dict(target=t, **M.summary(M.to_rho(t))) for t in targ]
    o['C1'] = [M.summary(C[np.random.default_rng(PERM_SEED + r).permutation(len(y))]) for r in range(NPERM)]
    c2 = []
    for r in range(NPERM):
        rng = np.random.default_rng(WPERM_SEED + r); p = np.arange(len(y))
        for c in range(NCLS):
            idx = np.where(y == c)[0]; p[idx] = rng.permutation(idx)
        c2.append(M.summary(C[p]))
    o['C2'] = c2
    rng = np.random.default_rng(PLANT_SEED); P = rng.standard_normal((tx.shape[1], d)); us = (tx - tx.mean(0)) @ P; ua = rng.standard_normal((NCLS, d)); e2 = M.En; g3 = {}
    for lab, u in (('arbitrary', ua), ('semantic', us)):
        u = u / np.sqrt((u ** 2).sum(1).mean()) * np.sqrt(e2)
        for al in (0.1, 0.3): g3[f'{lab}_{al}'] = M.summary(energy_fix(C + al * u[y], e2))
    o['C3'] = g3; o['secs'] = time.time() - t0; return o


def run_worlds(fr, base, targ, rD, h):
    t0 = time.time(); F, y, W, b, Vr, Vn = load(fr, base); M = Model(F, y, W, b, Vn); M.calibrate_vinfo(); C = M.C
    mu = np.zeros((NCLS, C.shape[1])); np.add.at(mu, y, C); mu = (mu / np.bincount(y, minlength=NCLS)[:, None])[y]
    Rr = (C - mu)[np.random.default_rng(WORLD_SEED).permutation(len(y))]; Mm = float((mu ** 2).sum(1).mean()); Rm = float((Rr ** 2).sum(1).mean()); Er = M.Er
    out = dict(base=base, M=Mm, Rr=Rm, worlds={})
    for Wd in ('C', 'M', 'D'):
        rows = []
        for j in range(12):
            rho = rD if Wd == 'D' else targ[j]; Et = Er / (1 - rho); L = h if Wd == 'C' else (rho - h if Wd == 'M' else rD * (j + 0.5) / 12)
            Cs = np.sqrt(L * Et / Mm) * mu + np.sqrt((rho - L) * Et / Rm) * Rr
            r_ = dict(L=L, rho_target=rho, **M.summary(Cs)); r_['floor'] = M.summary(Cs[np.random.default_rng(PERM_SEED).permutation(len(y))]); rows.append(r_)
        out['worlds'][Wd] = rows; print(base, 'world', Wd, round(time.time() - t0, 1), flush=True)
    out['secs'] = time.time() - t0; return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--models', default=''); ap.add_argument('--worlds', default='')
    ap.add_argument('--threads', type=int, default=2); ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args()
    torch.set_num_threads(a.threads); os.makedirs(os.path.join(RES6, 'models'), exist_ok=True); os.makedirs(os.path.join(RES6, 'worlds'), exist_ok=True)
    tx = text(a.sing_root); ms, col, _ = load_models(); rho_all = col('rho'); S = sets(ms, rho_all)
    targ = sorted(float(rho_all[ms.index(n)]) for n in S['E12']); rD = float(np.median(targ)); h = 0.5 * targ[0]
    for n in [m for m in a.models.split(',') if m]:
        p = os.path.join(RES6, 'models', f'{n}.json')
        if os.path.exists(p): print('skip', n); continue
        json.dump(run_model(a.feats_root, n, tx, targ), open(p, 'w'), indent=1, default=float)
    for bse in [m for m in a.worlds.split(',') if m]:
        p = os.path.join(RES6, 'worlds', f'{bse}.json')
        if os.path.exists(p): print('skip worlds', bse); continue
        json.dump(run_worlds(a.feats_root, bse, targ, rD, h), open(p, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
