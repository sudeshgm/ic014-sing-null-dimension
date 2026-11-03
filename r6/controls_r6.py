"""Post hoc (after PREREG_R6; listed in r6/POSTHOC_R6.md). Positive controls and a built-in-normalisation check for the round-6 audit.
(1) SING's own scores AS and IS (class-text target) under exactly the round-6 interventions of audit_r6.run_model: real (with the same
    200-resample bootstrap), E1 (rho = 0.005), E2 (the 12 E12 rho targets), C1, C2 and C3, with the released translator (all 10k images)
    and with the round-3 retrained ridge translator (refit on the fixed train half with the lambda recorded in
    r3/results/G_retrain_permodel.json; scored on the other half, as in r3/retrain_translators.py).
    AS = mean_i |theta(Tf_i, t) - theta(T(f_i - Pi f_i), t)|, IS = mean_i theta(Tf_i, T(f_i - Pi f_i)) (degrees).
(2) ViM with alpha frozen at its real-data value (per fit half) under E1 and E2, next to the refit-alpha value of audit_r6.
Aggregates only, written to r6/results/posthoc/controls_<model>.json. CPU, numpy + torch, float64.
Usage: python r6/controls_r6.py --feats-root $IC014_FEATS --models resnet,dinov2 [--threads 2] [--force]"""
import argparse, os, sys, json, time, numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
sys.path.insert(0, H)
import audit_r6 as au  # noqa: E402
import scores_r6 as sc  # noqa: E402
from audit_r6 import load_models, sets, R, NCLS  # noqa: E402
from retrain_translators import fit as ridge_fit  # noqa: E402
OUT = os.path.join(ROOT, 'r6', 'results', 'posthoc')


def ang(Z, T):
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True); return np.degrees(np.arccos(np.clip((Z * T).sum(1), -1, 1)))


class SingScores:
    """per-image AS / IS for null coordinates Cm, translator Z = F @ Wt + bt evaluated on rows `rows`"""
    def __init__(s, Frow, Vn, Wt, bt, tc, rows):
        s.rows = rows; s.Zr = Frow[rows] @ Wt + bt; s.VW = Vn.T @ Wt; s.tc = tc[rows]; s.a_r = ang(s.Zr, s.tc)
        s.Zrn = s.Zr / np.linalg.norm(s.Zr, axis=1, keepdims=True)

    def per_image(s, Cm):
        Z = s.Zr + Cm[s.rows] @ s.VW; return dict(AS=np.abs(ang(Z, s.tc) - s.a_r), IS=ang(Z, s.Zrn))   # IS as in r2/e2_controls.py: angle between T(f) and T(Pi_row f)


def summ(fn, Cm, boot_idx=None):
    pi = fn(Cm); o = {k: float(v.mean()) for k, v in pi.items()}
    if boot_idx is not None: o['sd_boot'] = {k: float(np.std([v[i].mean() for i in boot_idx], ddof=1)) for k, v in pi.items()}
    return o


def run_tests(fn, M, C, y, tx, targ, nrows):
    """the round-6 E1 / E2 / C1 / C2 / C3 interventions of audit_r6.run_model, same seeds, applied to the score function fn"""
    rng = np.random.default_rng(au.BOOT_SEED); bi = [rng.integers(0, nrows, nrows) for _ in range(au.NBOOT)]
    o = dict(real=summ(fn, C, bi), E1=summ(fn, M.to_rho(au.RHO0)), E2=[dict(target=t, **summ(fn, M.to_rho(t))) for t in targ])
    o['C1'] = [summ(fn, C[np.random.default_rng(au.PERM_SEED + r).permutation(len(y))]) for r in range(au.NPERM)]
    c2 = []
    for r in range(au.NPERM):
        rng = np.random.default_rng(au.WPERM_SEED + r); p = np.arange(len(y))
        for c in range(NCLS):
            idx = np.where(y == c)[0]; p[idx] = rng.permutation(idx)
        c2.append(summ(fn, C[p]))
    o['C2'] = c2; d = C.shape[1]
    rng = np.random.default_rng(au.PLANT_SEED); P = rng.standard_normal((tx.shape[1], d)); us = (tx - tx.mean(0)) @ P; ua = rng.standard_normal((NCLS, d)); e2 = M.En; g3 = {}
    for lab, u in (('arbitrary', ua), ('semantic', us)):
        u = u / np.sqrt((u ** 2).sum(1).mean()) * np.sqrt(e2)
        for al in (0.1, 0.3): g3[f'{lab}_{al}'] = summ(fn, au.energy_fix(C + al * u[y], e2))
    o['C3'] = g3; return o


def vim_fit_u(F, L, u):
    """scores_r6.vim_fit with the origin u = -pinv(W) b passed in (it depends only on the head), same arithmetic otherwise"""
    X = F - u; cov = X.T @ X / len(X)
    ev, evec = np.linalg.eigh(cov); order = np.argsort(-ev); NS = evec[:, order[sc.vim_dim(F.shape[1]):]]
    R_ = np.linalg.norm(X @ NS, axis=1); alpha = L.max(1).mean() / R_.mean()
    return dict(u=u, NS=NS, alpha=alpha)


def vim_frozen(M, F, L, W, b, targ):
    """mean p_virtual cross-fitted as in audit_r6.Model.per_image; refit alpha vs alpha frozen at the real-data fit of each half"""
    real_alpha = {}; u = -np.linalg.pinv(W) @ b
    def run(Cm, frozen):
        G = M.feats(Cm); v = np.full(len(M.y), np.nan); al = {}
        for h, fit in (('A', M.A), ('B', ~M.A)):
            m = vim_fit_u(G[fit], L[fit], u); al[h] = float(m['alpha'])
            if frozen: m['alpha'] = real_alpha[h]
            v[~fit] = sc.vim_eval(m, G[~fit], L[~fit])['p_virtual']
        return float(v.mean()), al
    r, al = run(M.C, False); real_alpha.update(al)
    e1r, a1 = run(M.to_rho(au.RHO0), False); e1f, _ = run(M.to_rho(au.RHO0), True)
    e2f = [run(M.to_rho(t), True)[0] for t in targ]      # the refit-alpha E2 values are those of audit_r6 (r6/results/models)
    return dict(real=r, alpha_real=real_alpha, E1_refit=e1r, E1_frozen=e1f, alpha_E1=a1, E2_frozen=e2f)


def run_model(fr, name, tx, TC, targ, Yclip, G, part='all', old=None):
    t0 = time.time(); F, y, W, b, Vr, Vn = au.load(fr, name); M = au.Model(F, y, W, b, Vn); C = M.C; L = M.L
    d = np.load(au.find(fr, name)); A0 = d['A'].astype(np.float64); b0 = d['Ab'].astype(np.float64); tc = TC[y]
    o = dict(model=name, rho=M.rho, n_released=len(y))
    if part == 'vim' or (part == 'sing' and old is not None):
        o.update({k: v for k, v in (old or {}).items() if k in (('vim_alpha',) if part == 'sing' else ('released', 'retrained_lambda_rel', 'n_retrained', 'retrained'))})
    if part == 'vim':
        o['vim_alpha'] = vim_frozen(M, F, L, W, b, targ); o['secs'] = time.time() - t0; return o
    rel = SingScores(M.Frow, Vn, A0.T, b0, tc, np.arange(len(y)))
    o['released'] = run_tests(rel.per_image, M, C, y, tx, targ, len(y)); print(name, 'released', round(o['released']['real']['AS'], 4), round(time.time() - t0, 1), flush=True)
    perm = np.random.default_rng(0).permutation(len(y)); tr, te = perm[:5000], perm[5000:]
    lam = G[name]['lambda_rel']; Wt, bt = ridge_fit(torch.tensor(F[tr]), torch.tensor(Yclip[tr]), lam); Wt, bt = Wt.numpy(), bt.numpy()
    ret = SingScores(M.Frow, Vn, Wt, bt, tc, te); o['retrained_lambda_rel'] = lam; o['n_retrained'] = len(te)
    o['retrained'] = run_tests(ret.per_image, M, C, y, tx, targ, len(te)); print(name, 'retrained', round(o['retrained']['real']['AS'], 4), round(time.time() - t0, 1), flush=True)
    if part == 'sing' and 'vim_alpha' in o: o['secs'] = time.time() - t0; return o
    o['vim_alpha'] = vim_frozen(M, F, L, W, b, targ); print(name, 'vim', {k: v for k, v in o['vim_alpha'].items() if k in ('real', 'E1_refit', 'E1_frozen')}, flush=True)
    o['secs'] = time.time() - t0; return o


KEYS = ('model', 'rho', 'n_released', 'released', 'retrained_lambda_rel', 'n_retrained', 'retrained', 'vim_alpha', 'secs')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--models', default=''); ap.add_argument('--threads', type=int, default=2)
    ap.add_argument('--force', action='store_true', help='recompute models whose output already exists')
    ap.add_argument('--part', default='all', choices=('all', 'sing', 'vim'), help='recompute only the SING (AS/IS) or only the ViM block of an existing output')
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args()
    torch.set_num_threads(a.threads); os.makedirs(OUT, exist_ok=True)
    tx = au.text(a.sing_root); TC = tx    # class-text target: mean of normalised prompt embeddings, renormalised (as r3/equal_energy.py)
    ms, col, _ = load_models(); rho_all = col('rho'); S = sets(ms, rho_all); targ = sorted(float(rho_all[ms.index(n)]) for n in S['E12'])
    Yclip = np.load(os.path.join(a.feats_root, 'e3/ret/e3_feats/clip_img.npz'))['e'].astype(np.float64)
    G = json.load(open(os.path.join(ROOT, 'r3', 'results', 'G_retrain_permodel.json')))
    for n in [m for m in a.models.split(',') if m] or ms:
        p = os.path.join(OUT, f'controls_{n}.json')
        if os.path.exists(p) and not a.force and a.part == 'all': print('skip', n); continue
        old = json.load(open(p)) if a.part != 'all' and os.path.exists(p) else None
        o = run_model(a.feats_root, n, tx, TC, targ, Yclip, G, a.part, old)
        o = {k: o[k] for k in KEYS if k in o}    # same key order whichever part was recomputed
        json.dump(o, open(p, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
