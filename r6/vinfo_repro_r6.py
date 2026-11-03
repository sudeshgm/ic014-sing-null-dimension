"""Post hoc (r6/POSTHOC_R6.md): does the conditional V-information penalty selection reproduce? For one model, repeats the frozen
inner-CV selection of scores_r6.vinfo_choose_lam (class target, both fit halves) `--reps` times in this process and records the held-out
gain at every grid value, then evaluates the real-data V-information with lambda_rel in {0.1, 1} for each fit half. Writes r6/results/posthoc/vinfo_repro_<model>.json. CPU, torch L-BFGS.
Usage: python r6/vinfo_repro_r6.py --feats-root $IC014_FEATS --model resnet [--reps 3] [--threads 2]"""
import argparse, os, sys, json, time, numpy as np, torch
H = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(H); sys.path.insert(0, H)
import audit_r6 as au  # noqa: E402
import scores_r6 as sc  # noqa: E402
GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def cv_gains(Cf, Lf, yf, which='label', seed=100700):
    A = sc.halves(yf, seed); out = {}
    for lr in GRID:
        g = [sc.vinfo_fit_eval(Cf[m], Lf[m], yf[m], Cf[~m], Lf[~m], yf[~m], lr, which)[0].mean() for m in (A, ~A)]; out[lr] = float(np.mean(g))
    return out


def real_vinfo(M, lam_rel_by_half):
    v = np.full(len(M.y), np.nan)
    for h, fit in (('A', M.A), ('B', ~M.A)):
        ev = ~fit; pm = sc.vinfo_phi_fit(M.C[fit]); P = (M.C[fit] - pm['mu']) @ pm['V']; ps = float((P ** 2).sum(1).mean())
        v[ev] = sc.vinfo_fit_eval(M.C[fit], M.L[fit], M.y[fit], M.C[ev], M.L[ev], M.y[ev], lam_rel_by_half[h], 'label', phi_scale=ps)[0]
    return float(v.mean())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--model', default='resnet')
    ap.add_argument('--reps', type=int, default=3); ap.add_argument('--threads', type=int, default=2); a = ap.parse_args(); torch.set_num_threads(a.threads)
    t0 = time.time(); F, y, W, b, Vr, Vn = au.load(a.feats_root, a.model); M = au.Model(F, y, W, b, Vn)
    o = dict(model=a.model, threads=a.threads, grid=list(GRID), cv=[], chosen=[])
    for r in range(a.reps):
        g = {h: cv_gains(M.C[m], M.L[m], M.y[m]) for h, m in (('A', M.A), ('B', ~M.A))}
        o['cv'].append({h: {str(k): v for k, v in gh.items()} for h, gh in g.items()}); o['chosen'].append({h: max(gh, key=gh.get) for h, gh in g.items()})
        print(a.model, 'rep', r, o['chosen'][-1], {h: {k: round(v, 6) for k, v in gh.items()} for h, gh in g.items()}, round(time.time() - t0), flush=True)
    o['real_by_lam'] = {f'{la}|{lb}': real_vinfo(M, dict(A=la, B=lb)) for la in (0.1, 1.0) for lb in (0.1, 1.0)}
    print(a.model, 'real by lam', {k: round(v, 5) for k, v in o['real_by_lam'].items()}, flush=True)
    json.dump(o, open(os.path.join(ROOT, 'r6', 'results', 'posthoc', f'vinfo_repro_{a.model}.json'), 'w'), indent=1, default=float); print('secs', time.time() - t0)


if __name__ == '__main__':
    main()
