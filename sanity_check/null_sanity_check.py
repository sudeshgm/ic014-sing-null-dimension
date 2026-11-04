"""Sanity check for cross-model null-space or residual-energy scores (paper, Sec. 3).

Two modes, both CPU only:

  features  per feature file: the null-energy fraction rho, its split rho = rho_offset + rho_between + rho_within
            (same plug-in identity as r8/posthoc/rho_decomp.py), and optionally SING's AS under content-free variants
            (offset only, offset removed, zero-mean Gaussian at the centred energy, centred null coordinates permuted
            across images), computed with r3/leakfree.setup and ang as in r8/posthoc/offset_surr.py.
  rank      Kendall tau-b of each per-model column against a reference order, with model-bootstrap percentile and BCa
            CIs (r3/common.tau_ci, B=2000, seed 0), and paired CIs of every column against your score (r3/common.paired_ci).

Feature files are .npz with
  F  (N, d)  penultimate features, one row per image
  y  (N,)    integer labels
  W  (K, d)  the linear head (its null space is the complement of the row space of W)
and, for the AS variants only, A (K', d) and Ab (K',), the translator to the text space (SING's released translators).

Examples
  python sanity_check/null_sanity_check.py features --npz feats/*.npz --out per_model.json
  python sanity_check/null_sanity_check.py rank --per-model per_model.json --score my_score --reference ref.json --out tau.json
  python sanity_check/null_sanity_check.py rank --per-model per_model.json --score AS --reference sing --sets sing
"""
import argparse, json, os, sys
import numpy as np

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(R, 'r3')); sys.path.insert(0, os.path.join(R, 'r4'))
from common import load_models, tau_ci, paired_ci  # noqa: E402
from common4 import sets as sing_sets  # noqa: E402

DEFAULT_TEXT = os.path.join(os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing')), 'misc/embeddings/imagenet_text_embeddings_fp16.npz')


# ---------------------------------------------------------------- step 1: centre (rho decomposition)
def null_basis(W):
    """Orthonormal basis of the null space of W (absolute singular-value tolerance 1e-5, as r4/retrained_a4 and r3/leakfree)."""
    import torch
    _, sv, vt = torch.linalg.svd(W.double(), full_matrices=True)
    return vt.T[:, int((sv > 1e-5).sum()):]


def rho_decomposition(F, y, W):
    """rho = E||Vn^T f||^2 / E||f||^2 split into offset, between-class and within-class parts (r8/posthoc/rho_decomp.py)."""
    import torch
    F = torch.as_tensor(F).double(); y = torch.as_tensor(y).long(); W = torch.as_tensor(W)
    Vn = null_basis(W); Z = F @ Vn; N = len(y); Etot = float(F.pow(2).sum(1).mean())
    K = int(y.max()) + 1; n = torch.bincount(y, minlength=K).double()
    M = torch.zeros(K, Z.shape[1], dtype=Z.dtype).index_add_(0, y, Z) / n.clamp_min(1)[:, None]
    zbar = Z.mean(0)
    off = float(zbar @ zbar); btw = float((n[:, None] * (M - zbar).pow(2)).sum() / N); wit = float((Z - M[y]).pow(2).sum() / N)
    En = float(Z.pow(2).sum(1).mean())
    ssw = torch.zeros(K, dtype=Z.dtype).index_add_(0, y, (Z - M[y]).pow(2).sum(1))
    noise = float((ssw / (n - 1).clamp_min(1)).sum() / N)
    return dict(rho=En / Etot, d_null=int(Vn.shape[1]), rho_offset=off / Etot, rho_between=btw / Etot, rho_within=wit / Etot,
                rho_centred=(btw + wit) / Etot, identity_abs_err=abs(En - off - btw - wit) / Etot,
                rho_between_debiased=(btw - noise) / Etot,
                share_offset=off / En if En > 0 else float('nan'), share_between=btw / En if En > 0 else float('nan'),
                share_within=wit / En if En > 0 else float('nan'))


# ---------------------------------------------------------------- step 2: matched-energy content-free surrogates of AS
def text_targets(path=DEFAULT_TEXT):
    import torch
    emb = np.load(path, allow_pickle=True)
    nrm = torch.nn.functional.normalize
    return nrm(nrm(torch.tensor(emb['embeddings'].astype(np.float64)), dim=-1).mean(1), dim=-1)


def as_variants(npz, TC, draws=3, seed0=92_000):
    """SING's AS on the real features and on content-free variants of the null coordinates (r8/posthoc/offset_surr.py)."""
    import torch
    import leakfree
    old = leakfree.find; leakfree.find = lambda fr, name: npz          # point r3/leakfree.setup at this file
    try:
        E = leakfree.setup(None, None)
    finally:
        leakfree.find = old
    ang = leakfree.ang
    tc = TC[E['y']]; a0 = ang(E['Zrow'], tc); N = E['F'].shape[0]
    C = E['F'] @ E['Vn']; Cc = C - E['mu']; Ec = float((Cc ** 2).sum(1).mean())
    AS_of = lambda Ct: float((ang(E['Zrow'] + Ct @ E['AVn'].T, tc) - a0).abs().mean())
    o = dict(AS=AS_of(C), off_nat=AS_of(E['mu'].expand(N, -1)), real_c=AS_of(Cc))
    zm, pc = [], []
    for r in range(draws):
        g = torch.Generator().manual_seed(seed0 + r); z = torch.randn(N, C.shape[1], generator=g, dtype=torch.float64) @ E['Ls'].T
        zm.append(AS_of(z * (Ec / float((z ** 2).sum(1).mean())) ** 0.5))
        pc.append(AS_of(Cc[torch.tensor(np.random.default_rng(seed0 + r).permutation(N))]))
    o.update(zm_c=float(np.mean(zm)), zm_c_sd=float(np.std(zm)), perm_c=float(np.mean(pc)), perm_c_sd=float(np.std(pc)),
             rho_c=Ec / E['Etot'])
    return o


def run_features(paths, names=None, with_as=False, text=DEFAULT_TEXT, draws=3):
    import torch
    names = names or [os.path.basename(p).replace('_full.npz', '').replace('.npz', '') for p in paths]
    TC = text_targets(text) if with_as else None
    out = {}
    for name, p in zip(names, paths):
        d = np.load(p)
        r = rho_decomposition(torch.tensor(d['F']), torch.tensor(d['y']), torch.tensor(d['W']))
        if with_as:
            r.update(as_variants(p, TC, draws))
        out[name] = r
        print(name, {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    return out


# ---------------------------------------------------------------- comparison of rankings
COLUMNS = {'rho': 'rho (uncentred)', 'rho_centred': 'rho centred (offset removed)', 'rho_offset': 'rho offset only',
           'rho_between': 'rho between-class', 'rho_within': 'rho within-class', 'AS': 'AS (real)',
           'real_c': 'AS real centred (offset removed)', 'off_nat': 'AS offset only (natural size)',
           'zm_c': 'AS zero-mean Gaussian @ centred energy', 'perm_c': 'AS centred null permuted @ centred energy'}


def sing_reference():
    """SING's order: lk = -(SING's digitised IS/AS ratio), as in r3/paired_stats.py; plus the paper's rho column for the E12 screen."""
    ms, col, sing = load_models()
    return dict(zip(ms, (-sing).tolist())), dict(zip(ms, col('rho').tolist()))


def run_rank(per_model, score, reference, set_mode='all', set_rho=None, columns=None, B=2000, seed=0):
    """tau of every column against the reference order, and paired differences of each column against `score`."""
    ms = [m for m in reference if m in per_model]
    if len(ms) < 4:
        raise SystemExit('need at least 4 models present in both the per-model file and the reference')
    cols = columns or [c for c in [score] + [c for c in COLUMNS if c != score] if all(c in per_model[m] for m in ms)]
    X = {c: np.array([float(per_model[m][c]) for m in ms]) for c in cols}
    if set_mode == 'sing':
        rr = set_rho or {m: per_model[m]['rho'] for m in ms}
        S = sing_sets(ms, np.array([rr[m] for m in ms]))
    else:
        S = {'all': ms}
    out = dict(models=ms, score=score, sets=S, tau={}, paired={})
    for s, sub in S.items():
        idx = [ms.index(n) for n in sub]; y = np.array([reference[m] for m in ms])[idx]
        out['tau'][s] = {c: tau_ci(X[c][idx], y, B, seed) for c in cols}
        out['paired'][s] = {f'{c} - {score}': paired_ci(X[c][idx], X[score][idx], y, B, seed) for c in cols if c != score}
    return out


def _print_rank(out):
    f = lambda d: f"{d['tau']:+.2f} pct [{d['pct'][0]:+.2f},{d['pct'][1]:+.2f}] BCa [{d['bca'][0]:+.2f},{d['bca'][1]:+.2f}]"
    for s in out['tau']:
        print(f'== {s} ({len(out["sets"][s])} models), tau with the reference order')
        for c, d in out['tau'][s].items(): print(f'  {COLUMNS.get(c, c):44s} {f(d)}')
        for c, d in out['paired'][s].items():
            print(f"  paired {c:36s} {d['diff']:+.2f} pct [{d['pct'][0]:+.2f},{d['pct'][1]:+.2f}] BCa [{d['bca'][0]:+.2f},{d['bca'][1]:+.2f}]")
    print('Read: if rho, rho centred or the AS surrogates order the models about as well as your score, the ranking does not by '
          'itself show content. A paired interval covering zero means "not separated", not "equivalent".')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='mode', required=True)
    a = sp.add_parser('features', help='rho decomposition (and optional AS surrogates) per feature file')
    a.add_argument('--npz', nargs='+', required=True); a.add_argument('--names', nargs='*')
    a.add_argument('--as-variants', action='store_true', help='also compute SING AS and its content-free variants (needs A, Ab)')
    a.add_argument('--text', default=DEFAULT_TEXT, help='class text embeddings (.npz with key embeddings)')
    a.add_argument('--draws', type=int, default=3); a.add_argument('--out', required=True)
    b = sp.add_parser('rank', help='Kendall tau of each per-model column against a reference order')
    b.add_argument('--per-model', required=True, help='JSON {model: {column: value}}; add your own score as a column')
    b.add_argument('--score', required=True, help='column holding your score')
    b.add_argument('--reference', default='sing', help="'sing' or a JSON {model: value} (higher = ranked higher)")
    b.add_argument('--sets', choices=['all', 'sing'], default='all', help="'sing' = the paper's all13/E12/S11 sets")
    b.add_argument('--columns', nargs='*'); b.add_argument('--out')
    args = ap.parse_args(argv)
    if args.mode == 'features':
        out = run_features(args.npz, args.names, args.as_variants, args.text, args.draws)
        json.dump(out, open(args.out, 'w'), indent=1)
    else:
        P = json.load(open(args.per_model)); set_rho = None
        if args.reference == 'sing':
            ref, set_rho = sing_reference()
        else:
            ref = json.load(open(args.reference))
        out = run_rank(P, args.score, ref, args.sets, set_rho, args.columns)
        _print_rank(out)
        if args.out: json.dump(out, open(args.out, 'w'), indent=1, default=float)


if __name__ == '__main__':
    main()
