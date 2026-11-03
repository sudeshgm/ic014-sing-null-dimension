"""PREREG_R5: translator-free, energy-free scores of how the null-space class means are organised relative to CLIP's class-name geometry.
Primary SO_null (cross-fitted crossnobis class-mean distances, double-centred, Spearman against the double-centred 1 - cos of the
class-text embeddings, disattenuated by split reliability); secondary XCC_24 (cross-fitted class-mean Gram congruence with the centred text Gram, mean over 50 Haar 24-frames).
Gates G1-G5, worlds B / S / corrected C (and M, D descriptive), the confirmatory tau against SING's order and the inheritance diagnostic.
Aggregate outputs only (per model at most) in r5/results/. CPU, numpy.
Usage: python r5/organisation.py --feats-root $IC014_FEATS --part gates|confirm|all [--models a,b] [--sing-root _ext/sing]"""
import argparse, os, sys, json, time, numpy as np
from scipy.stats import rankdata
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H)
sys.path.insert(0, os.path.join(ROOT, 'r4')); sys.path.insert(0, os.path.join(ROOT, 'r3'))
from common4 import R, load_models, tau_ci, ktau, sets, partial_ci  # noqa: E402
from retrain_translators import find  # noqa: E402
from class_perm import derangement  # noqa: E402
RES5 = os.path.join(ROOT, 'r5', 'results')
NCLS, IU = 1000, np.triu_indices(1000, 1)
FOLD_SEED, PERM_SEED, HAAR_SEED, PLANT_SEED, ROT_SEED, ZFOLD_SEED, RESID_SEED, DER_SEED = 99200, 99300, 99100, 99400, 99500, 99600, 99000, 94000
NPERM, NHAAR, K, NDER = 200, 50, 24, 5
REL_FLOOR, Z_FLOOR = 0.10, 5.0
CAL_BASES = ('resnet', 'dinov2', 'resnext')


# ---------------- data
def text(sing_root):
    e = np.load(os.path.join(sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)['embeddings'].astype(np.float64)
    e = e / np.linalg.norm(e, axis=-1, keepdims=True); t = e.mean(1); return t / np.linalg.norm(t, axis=1, keepdims=True)


def load(fr, name):
    if name.startswith('dino_vit'):
        d = np.load(f'{fr}/r3_dino_feats/{name}_10k.npz', allow_pickle=True); F, y, W = d['F'], d['labels'].astype(int), d['W']
    else:
        d = np.load(find(fr, name)); F, y, W = d['F'], d['y'].astype(int), d['W']
    F = F.astype(np.float64); _, sv, vt = np.linalg.svd(W.astype(np.float64), full_matrices=True); r = int((sv > 1e-5).sum())
    return F, y, vt[:r].T, vt[r:].T


# ---------------- SO_null
def folds(y, seed=FOLD_SEED):
    rng = np.random.default_rng(seed); g = np.zeros(len(y), int)
    for c in range(NCLS):
        idx = np.where(y == c)[0]; rng.shuffle(idx); g[idx] = np.arange(len(idx)) % 4
    return g


def cmean(Z, y, m):
    M = np.zeros((NCLS, Z.shape[1])); np.add.at(M, y[m], Z[m]); return M / np.bincount(y[m], minlength=NCLS)[:, None]


def xdist(Z, y, ma, mb):
    """cross-fitted squared class-mean distance (m_a(c)-m_a(c')).(m_b(c)-m_b(c')), upper triangle"""
    a, b = cmean(Z, y, ma), cmean(Z, y, mb); M = a @ b.T; d = np.diag(M); return (d[:, None] + d[None] - M - M.T)[IU]


def dcent(v):
    """double-centre a symmetric pair vector (upper triangle, zero diagonal): removes additive per-class (row/column) effects"""
    M = np.zeros((NCLS, NCLS)); M[IU] = v; M = M + M.T; rm = M.sum(1) / (NCLS - 1); return (M - rm[:, None] - rm[None] + v.mean())[IU]


def corr(u, v): u = u - u.mean(); v = v - v.mean(); return float(u @ v / np.sqrt((u @ u) * (v @ v)))


class Text:
    def __init__(s, T):
        s.T = T; s.rt = rankdata(dcent((1 - T @ T.T)[IU])); Tc = T - T.mean(0); G = Tc @ Tc.T; np.fill_diagonal(G, 0); s.G = G; s.GG = float((G * G).sum()); s.Tc = Tc


def so(Z, y, g, tx, nperm=0, seed=PERM_SEED):
    D = dcent(xdist(Z, y, g < 2, g >= 2)); D1 = rankdata(dcent(xdist(Z, y, g == 0, g == 1))); D2 = rankdata(dcent(xdist(Z, y, g == 2, g == 3)))
    raw = corr(rankdata(D), tx.rt); r1, r2 = corr(D1, tx.rt), corr(D2, tx.rt); rel = corr(D1, D2)
    o = dict(raw=raw, rel=rel, raw_q=0.5 * (r1 + r2), dis=float(0.5 * (r1 + r2) / np.sqrt(rel)) if rel > 0 else float('nan'), estimable=bool(rel >= REL_FLOOR))
    if nperm:
        rng = np.random.default_rng(seed); M1 = np.zeros((NCLS, NCLS)); M2 = np.zeros((NCLS, NCLS)); M1[IU] = D1; M2[IU] = D2; M1 += M1.T; M2 += M2.T; nl = []
        for _ in range(nperm):
            p = rng.permutation(NCLS); nl.append(0.5 * (corr(M1[np.ix_(p, p)][IU], tx.rt) + corr(M2[np.ix_(p, p)][IU], tx.rt)) / np.sqrt(max(rel, 1e-12)))
        o.update(null_mean=float(np.mean(nl)), null_sd=float(np.std(nl, ddof=1)), null95=float(np.quantile(nl, .95)))
    return o


# ---------------- XCC
def halves_mask(y):
    A = np.zeros(len(y), bool); seen = {}
    for i, c in enumerate(y): k = seen.get(c, 0); A[i] = k < 5; seen[c] = k + 1
    return A


def half_means(X, y, A):
    out = []
    for m in (A, ~A):
        M = cmean(X, y, m); out.append(M - M.mean(0))
    return out


def grams(MA, MB):
    Kab = 0.5 * (MA @ MB.T + MB @ MA.T); KA = MA @ MA.T; KB = MB @ MB.T
    for G in (Kab, KA, KB): np.fill_diagonal(G, 0)
    return Kab, KA, KB


def xcc_parts(MA, MB, tx):
    Kab, KA, KB = grams(MA, MB); nK = float((KA * KB).sum()); return Kab, nK


def xcc(MA, MB, tx):
    Kab, nK = xcc_parts(MA, MB, tx); return float((Kab * tx.G).sum() / np.sqrt(nK * tx.GG)) if nK > 0 else float('nan')


def haar(d, s): return np.linalg.qr(np.random.default_rng(HAAR_SEED + s).standard_normal((d, K)))[0]


def xcc24(C, y, A, tx, frames=None, nperm=0, seed=PERM_SEED):
    frames = frames if frames is not None else [haar(C.shape[1], s) for s in range(NHAAR)]; vals = []; Kbar = np.zeros((NCLS, NCLS)); nf = 0
    for Q in frames:
        ma, mb = half_means(C @ Q, y, A); Kab, nK = xcc_parts(ma, mb, tx)
        if nK > 0: vals.append(float((Kab * tx.G).sum() / np.sqrt(nK * tx.GG))); Kbar += Kab / np.sqrt(nK); nf += 1
    o = dict(xcc24=float(np.mean(vals)) if vals else float('nan'), xcc24_sd=float(np.std(vals)) if vals else float('nan'), frames_ok=nf)
    if nperm and nf:
        Kbar /= nf; rng = np.random.default_rng(seed); nl = []
        for _ in range(nperm):
            p = rng.permutation(NCLS); nl.append(float((Kbar * tx.G[np.ix_(p, p)]).sum() / np.sqrt(tx.GG)))
        o.update(null_mean=float(np.mean(nl)), null_sd=float(np.std(nl, ddof=1)))
    return o


def selfterm_z(C, y, A):
    ma, mb = half_means(C, y, A); KA = ma @ ma.T; KB = mb @ mb.T; np.fill_diagonal(KA, 0); np.fill_diagonal(KB, 0); KK = KA * KB
    fold = np.random.default_rng(ZFOLD_SEED).integers(0, 10, NCLS); per = np.array([KK[fold == f].sum() for f in range(10)])
    return float(per.mean() / per.std(ddof=1) * np.sqrt(10))


def partial_xcc(MA, MB, RA, RB, tx):
    Kab, KA, KB = grams(MA, MB); Pab, PA, PB = grams(RA, RB); G = tx.G
    PP = (PA * PB).sum(); KP = 0.5 * ((KA * PB).sum() + (KB * PA).sum()); beta = KP / PP
    num = (Kab * G).sum() - beta * (Pab * G).sum(); PG = (Pab * G).sum(); G2 = tx.GG - PG ** 2 / PP; K2 = (KA * KB).sum() - KP ** 2 / PP
    return float(num / np.sqrt(K2 * G2)) if K2 > 0 and G2 > 0 else float('nan')


# ---------------- both scores on one null-coordinate matrix
def scores(C, y, g, A, tx, nperm=0, z=True):
    s = so(C, y, g, tx, nperm); x = xcc24(C, y, A, tx, nperm=nperm)
    o = dict(SO=s['dis'], SO_rel=s['rel'], SO_raw=s['raw'], SO_est=s['estimable'], XCC24=x['xcc24'])
    if nperm: o.update(SO_null_sd=s['null_sd'], SO_null_mean=s['null_mean'], XCC_null_sd=x['null_sd'], XCC_null_mean=x['null_mean'])
    if z: o['XCC_z'] = selfterm_z(C, y, A); o['XCC_est'] = bool(o['XCC_z'] >= Z_FLOOR)
    return o


def energy_fix(Cp, e2): return Cp * np.sqrt(e2 / (Cp ** 2).sum(1).mean())


def per_model(fr, name, tx, full=True):
    t0 = time.time(); F, y, Vr, Vn = load(fr, name); C = F @ Vn; Rw = F @ Vr; g = folds(y); A = halves_mask(y); d = C.shape[1]
    o = dict(model=name, d_null=d, rho=float((C ** 2).sum(1).mean() / (F ** 2).sum(1).mean()))
    a, b = cmean(C, y, g < 2), cmean(C, y, g >= 2); gm = C.mean(0); o['eta2'] = float(((a - gm) * (b - gm)).sum(1).mean() / ((C - gm) ** 2).sum(1).mean())
    o.update(scores(C, y, g, A, tx, NPERM))
    ma, mb = half_means(C, y, A); ra, rb = half_means(Rw, y, A); o['XCC_full'] = xcc(ma, mb, tx); o['F1_partial'] = partial_xcc(ma, mb, ra, rb, tx)
    o['SO_row'] = so(Rw, y, g, tx)['dis']; o['XCC_row'] = xcc(ra, rb, tx)
    if not full: o['secs'] = time.time() - t0; return o
    # G1 exact invariance: null rescaling (both scores) and a random orthogonal map of the null coordinates (SO_null, full-null XCC)
    Q = np.linalg.qr(np.random.default_rng(ROT_SEED).standard_normal((d, d)))[0]; g1 = {}
    for beta in (0.1, 10.0):
        s = so(beta * C, y, g, tx); g1[f'SO_beta{beta}'] = abs(s['dis'] - o['SO']); g1[f'XCC24_beta{beta}'] = abs(xcc24(beta * C, y, A, tx)['xcc24'] - o['XCC24'])
    CQ = C @ Q; g1['SO_rot'] = abs(so(CQ, y, g, tx)['dis'] - o['SO']); m1, m2 = half_means(CQ, y, A); g1['XCCfull_rot'] = abs(xcc(m1, m2, tx) - o['XCC_full'])
    o['G1'] = g1; o['G1_pass'] = bool(max(v for v in g1.values() if np.isfinite(v)) <= 1e-10) if any(np.isfinite(v) for v in g1.values()) else False
    # G2 class-mean derangement (energy preserved exactly): z' = mu_pi(y) + r, seeds 94000+r
    mu = cmean(C, y, np.ones(len(y), bool)); Rr = C - mu[y]; dso, dx = [], []
    for r in range(NDER):
        p = derangement(NCLS, DER_SEED + r); Cd = mu[p][y] + Rr; dso.append(so(Cd, y, g, tx)['dis']); dx.append(xcc24(Cd, y, A, tx)['xcc24'])
    o['G2'] = dict(SO_derange=dso, XCC24_derange=dx)
    for k, v, sd in (('SO', dso, o['SO_null_sd']), ('XCC24', dx, o['XCC_null_sd'])):
        m = float(np.mean(v)); o['G2'][f'{k}_mean'] = m
        o['G2'][f'{k}_pass'] = bool(abs(m) <= 0.10 * abs(o[k]) or abs(m) <= 2 * sd / np.sqrt(NDER))
    # G3 / world S: semantic vs arbitrary class-code plants at fixed null energy, alpha in {0.1, 0.3}
    rng = np.random.default_rng(PLANT_SEED); P = rng.standard_normal((tx.T.shape[1], d)); us = tx.Tc @ P; ua = rng.standard_normal((NCLS, d)); e2 = (C ** 2).sum(1).mean(); g3 = {}
    for lab, u in (('semantic', us), ('arbitrary', ua)):
        u = u / np.sqrt((u ** 2).sum(1).mean()) * np.sqrt(e2)
        for al in (0.1, 0.3):
            s = scores(energy_fix(C + al * u[y], e2), y, g, A, tx, z=False); g3[f'{lab}_{al}'] = dict(SO=s['SO'], XCC24=s['XCC24'])
    o['G3'] = g3
    for k, sd in (('SO', o['SO_null_sd']), ('XCC24', o['XCC_null_sd'])):
        ds = (g3['semantic_0.3'][k] - o[k]) / sd; da = (g3['arbitrary_0.3'][k] - o[k]) / sd
        o['G3'][f'{k}_sem_sd'] = float(ds); o['G3'][f'{k}_arb_sd'] = float(da); o['G3'][f'{k}_pass'] = bool(np.isfinite(ds) and np.isfinite(da) and ds >= 2 and da <= 1)
    # inheritance plant check (F): planted text geometry at fraction alpha of the class-mean energy
    if name in CAL_BASES:
        muc = mu - mu.mean(0); Pt = tx.Tc @ np.random.default_rng(PLANT_SEED + 1).standard_normal((tx.T.shape[1], d)); Pt *= np.sqrt((muc ** 2).sum() / (Pt ** 2).sum()); pr = {}
        for al in (0.05, 0.1, 0.2):
            m1, m2 = half_means(C + np.sqrt(al) * Pt[y], y, A); pr[str(al)] = partial_xcc(m1, m2, ra, rb, tx)
        o['inherit_plant_partial'] = pr
    o['secs'] = time.time() - t0; return o


# ---------------- worlds on the calibration bases: B (null rescaling), C* (corrected), M, D
def worlds(fr, base, tx, targ, rD, h):
    F, y, Vr, Vn = load(fr, base); C = F @ Vn; g = folds(y); A = halves_mask(y)
    mu = cmean(C, y, np.ones(len(y), bool))[y]; Rr = (C - mu)[np.random.default_rng(RESID_SEED).permutation(len(y))]
    Er = float(((F - C @ Vn.T) ** 2).sum(1).mean()); Mm = float((mu ** 2).sum(1).mean()); Rm = float((Rr ** 2).sum(1).mean()); En = float((C ** 2).sum(1).mean())
    out = {}
    for W in ('B', 'C', 'M', 'D'):
        rows = []
        for j in range(12):
            rho = rD if W == 'D' else targ[j]; Et = Er / (1 - rho)
            if W == 'B': Cs = C * np.sqrt(rho * Et / En); L = None
            else:
                L = h if W == 'C' else (rho - h if W == 'M' else rD * (j + 0.5) / 12)
                Cs = np.sqrt(L * Et / Mm) * mu + np.sqrt((rho - L) * Et / Rm) * Rr
            s = scores(Cs, y, g, A, tx); s.update(rho=float((Cs ** 2).sum(1).mean() / (Er + (Cs ** 2).sum(1).mean())), L=L); rows.append(s)
        out[W] = rows
    return out


def flat_ratio(vals):
    v = np.asarray(vals, float)
    if len(v) < 3: return None
    return float(v.max() / v.min()) if v.min() > 0 else float('inf')


def g4(wd):
    """corrected world C: max/min over estimable rows <= 1.5 (SO: rel >= 0.10; XCC: z >= 5); needs >= 3 estimable rows"""
    rows = wd['C']; o = {}
    for k, ek in (('SO', 'SO_est'), ('XCC24', 'XCC_est')):
        v = [r[k] for r in rows if r[ek] and np.isfinite(r[k])]; fr = flat_ratio(v)
        o[k] = dict(n_est=len(v), ratio=fr, status='NOT ESTIMABLE' if fr is None else ('pass' if fr <= 1.5 else 'fail'))
    return o


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--part', choices=['gates', 'confirm', 'all'], default='all')
    ap.add_argument('--models', default=''); ap.add_argument('--out', default='')
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    tx = Text(text(a.sing_root)); ms, col, sing = load_models(); lk = -sing; rho_all = col('rho'); S = sets(ms, rho_all); os.makedirs(RES5, exist_ok=True)
    names = a.models.split(',') if a.models else list(ms); outp = a.out or os.path.join(RES5, 'R5_gates.json')
    if a.part in ('gates', 'all'):
        G = dict(models={}, worlds={}); targ = sorted(float(rho_all[ms.index(n)]) for n in S['E12']); rD = float(np.median(targ)); h = 0.5 * targ[0]
        for n in names:
            G['models'][n] = per_model(fr, n, tx); m = G['models'][n]
            print(n, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if not isinstance(v, (dict, list))}, 'G1', m['G1_pass'], 'G2', m['G2']['SO_pass'], m['G2']['XCC24_pass'],
                  'G3', m['G3']['SO_pass'], m['G3']['XCC24_pass'], round(m['secs'], 1), flush=True)
        if not a.models:
            for b in CAL_BASES:
                t0 = time.time(); wd = worlds(fr, b, tx, targ, rD, h); G['worlds'][b] = dict(rows=wd, G4=g4(wd))
                print(b, 'G4', G['worlds'][b]['G4'], round(time.time() - t0, 1), flush=True)
            for n in ('dino_vitb16', 'dino_vits16'):
                G.setdefault('dino', {})[n] = per_model(fr, n, tx, full=False); print(n, {k: v for k, v in G['dino'][n].items()}, flush=True)
            G['summary'] = summarise(G, S)
        json.dump(G, open(outp, 'w'), indent=1, default=float); print('summary', json.dumps(G.get('summary'), indent=1))
    if a.part in ('confirm', 'all'):
        G = json.load(open(os.path.join(RES5, 'R5_gates.json'))); sm = G['summary']; C = dict(score_status=sm['score_status'])
        if sm['kill']:
            C['stopped'] = 'K1: SO_null killed at the gates; no confirmatory tau computed'; json.dump(C, open(os.path.join(RES5, 'R5_confirm.json'), 'w'), indent=1); print(C); return
        P = G['models']; lab = {}
        for k, ek in (('SO', 'SO_est'), ('XCC24', 'XCC_est')):
            if sm['score_status'][k] != 'alive': continue
            C[k] = {}
            for s in ('all13', 'E12', 'S11'):
                sub = [n for n in S[s] if P[n][ek] and np.isfinite(P[n][k])]; x = np.array([P[n][k] for n in sub]); yv = np.array([lk[ms.index(n)] for n in sub]); rh = np.array([rho_all[ms.index(n)] for n in sub])
                e2 = np.array([P[n]['eta2'] for n in sub])
                C[k][s] = dict(models=sub, n=len(sub), tau=tau_ci(x, yv), tau_vs_rho=float(ktau(x, rh)), tau_vs_eta2=float(ktau(x, e2)), partial_given_rho=partial_ci(x, yv, rh))
            enough = C[k]['E12']['n'] >= 9 and C[k]['S11']['n'] >= 8; t = {s: C[k][s]['tau']['tau'] for s in ('E12', 'S11')}
            lab[k] = ('NO CONFIRMATORY TEST (G5)' if not enough else 'ORGANISATION-TRACKS' if all(v >= 0.6 for v in t.values())
                      else 'ORGANISATION-DOES-NOT-TRACK' if all(v < 0.3 for v in t.values()) else 'INCONCLUSIVE')
        C['outcome'] = lab
        est_x = [n for n in ms if P[n]['XCC_est'] and np.isfinite(P[n]['F1_partial'])]; low = [n for n in est_x if P[n]['F1_partial'] <= 0.05]
        pl = {b: P[b].get('inherit_plant_partial', {}).get('0.05') for b in CAL_BASES}; plant_ok = all(v is not None and v >= 0.1 for v in pl.values())
        C['inheritance'] = dict(estimable=est_x, low=low, share=len(low) / max(1, len(est_x)), plant_partial_005=pl, plant_check=plant_ok,
                                statement=('INHERITED' if len(low) >= 0.75 * len(est_x) and plant_ok else 'NOT INHERITED' if plant_ok else 'UNINFORMATIVE (plant check failed)'))
        json.dump(C, open(os.path.join(RES5, 'R5_confirm.json'), 'w'), indent=1, default=float); print(json.dumps({k: C[k] for k in ('outcome', 'inheritance')}, indent=1, default=float))
        for k in ('SO', 'XCC24'):
            if k in C: print(k, {s: (round(C[k][s]['tau']['tau'], 3), [round(v, 2) for v in C[k][s]['tau']['pct']], C[k][s]['n']) for s in ('all13', 'E12', 'S11')})


def summarise(G, S):
    P = G['models']; ms13 = [n for n in P]; st = {}; det = {}
    for k, ek in (('SO', 'SO_est'), ('XCC24', 'XCC_est')):
        est = [n for n in ms13 if P[n][ek]]
        g1 = all(P[n]['G1'][f'{k}_beta{b}'] <= 1e-10 for n in ms13 for b in (0.1, 10.0)) and all(P[n]['G1']['SO_rot' if k == 'SO' else 'XCCfull_rot'] <= 1e-10 for n in ms13)
        g2_fail = [n for n in est if not P[n]['G2'][f'{k}_pass']]; g3_pass = [n for n in ms13 if P[n]['G3'][f'{k}_pass']]
        g4 = {b: G['worlds'][b]['G4'][k]['status'] for b in G['worlds']}; g4_fail = sum(v == 'fail' for v in g4.values()); g4_testable = sum(v != 'NOT ESTIMABLE' for v in g4.values())
        n_e12 = sum(1 for n in S['E12'] if P[n][ek]); n_s11 = sum(1 for n in S['S11'] if P[n][ek])
        killed = []
        if not g1: killed.append('G1')
        if len(g2_fail) > 2: killed.append('G2')
        if len(g3_pass) < 11: killed.append('G3')
        if g4_fail >= 2: killed.append('G4')
        det[k] = dict(G1=g1, G2_fail=g2_fail, G3_pass_n=len(g3_pass), G3_fail=[n for n in ms13 if n not in g3_pass], G4=g4, G4_testable=g4_testable,
                      G5_estimable=est, G5_E12=n_e12, G5_S11=n_s11, killed_by=killed)
        st[k] = 'killed' if killed else 'alive'
    return dict(gates=det, score_status=st, kill=st['SO'] == 'killed')


if __name__ == '__main__':
    main()
