"""PREREG_R3 D (N3): chance floors for the fix-14 checkpoint audit. For each cached ViT-L candidate (first 1000 subset
images) through SING's released dinovit1 translator: mean cos(T(f), CLIP_img), shuffled-pair cos (image permutation,
seed 0), centred cos (both sides centred over the 1000 images), and retrieval top-1 among the 1000 CLIP images.
Usage: python r3/fix14_floor.py --feats-root $IC014_FEATS [--cand-dir fix14_feats]"""
import argparse, os, glob, numpy as np
from common import dump
ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--cand-dir', default=None); a = ap.parse_args()
cd = a.cand_dir or os.path.join(a.feats_root, 'fix14_feats')
d = np.load(f'{a.feats_root}/e2/ret/e2_feats/dinovit1_full.npz'); A, Ab = d['A'].astype(np.float64), d['Ab'].astype(np.float64)
E = np.load(f'{a.feats_root}/e3/ret/e3_feats/clip_img.npz')['e'][:1000].astype(np.float64)
En = E / np.linalg.norm(E, axis=1, keepdims=True); perm = np.random.default_rng(0).permutation(1000)
un = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)
def stats(F):
    Tn = un(F @ A.T + Ab); cos = float((Tn * En).sum(1).mean()); sh = float((Tn * En[perm]).sum(1).mean())
    cc = float((un(Tn - Tn.mean(0)) * un(En - En.mean(0))).sum(1).mean())
    r1 = float(((Tn @ En.T).argmax(1) == np.arange(1000)).mean())
    return dict(cos=cos, shuffled_pair_cos=sh, cos_minus_shuffled=cos - sh, centred_cos=cc, retrieval_top1=r1)
USED = 'timm_vit_large_patch16_224.augreg_in21k_ft_in1k'
res = {'cached_dinovit1 (used)': stats(d['F'][:1000].astype(np.float64))}
for fn in sorted(glob.glob(os.path.join(cd, '*.npz'))):
    res[os.path.basename(fn)[:-4]] = stats(np.load(fn)['F'].astype(np.float64))
non = {k: v for k, v in res.items() if k.startswith(('tv_', 'hf_'))}
partial = [k for k, v in non.items() if abs(v['cos_minus_shuffled']) >= 0.01]
out = dict(candidates=res, used=USED, non_matching=list(non), partially_matching=partial, pass_N3=not partial,
           chance_retrieval=1 / 1000)
dump('D_fix14_floor.json', out)
for k, v in res.items(): print(f'{k:50s}', {kk: round(vv, 4) for kk, vv in v.items()})
print('pass_N3', out['pass_N3'], 'partial', partial)
