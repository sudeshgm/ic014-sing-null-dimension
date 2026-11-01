"""Round-2 fix 14, CPU scoring step (run after gpu_jobs/fix14_extract_vitl.py): mean cos(T(f), CLIP_img) on the first
1000 subset images for each ViT-L candidate, T = SING's released dinovit1 linear translator (A, Ab as cached in
e2/ret/e2_feats/dinovit1_full.npz), CLIP_img = the cached clip_img.npz rows (same subset order).
Pass (PREREG_R2.md): the checkpoint we used (timm augreg_in21k_ft_in1k) scores highest by >= 0.05.
Usage: python r2/fix14_fidelity.py --feats-root $IC014_FEATS --cand-dir fix14_feats"""
import argparse, os, glob, json, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--cand-dir', required=True); a = ap.parse_args()
d = np.load(f'{a.feats_root}/e2/ret/e2_feats/dinovit1_full.npz'); A, Ab = d['A'].astype(np.float64), d['Ab'].astype(np.float64)
E = np.load(f'{a.feats_root}/e3/ret/e3_feats/clip_img.npz')['e'][:1000].astype(np.float64)
cos = lambda F: float(np.mean(np.sum((F @ A.T + Ab) / np.linalg.norm(F @ A.T + Ab, axis=1, keepdims=True) * E, 1)))
res = {'cached_dinovit1_first1000 (SING registry transform)': cos(d['F'][:1000].astype(np.float64))}
for fn in sorted(glob.glob(os.path.join(a.cand_dir, '*.npz'))):
    F = np.load(fn)['F'].astype(np.float64); res[os.path.basename(fn)[:-4]] = cos(F) if F.shape[1] == A.shape[1] else None
used = 'timm_vit_large_patch16_224.augreg_in21k_ft_in1k'; others = [v for k, v in res.items() if k.startswith(('tv_', 'hf_')) and v is not None]
res['_pass'] = bool(used in res and others and res[used] - max(others) >= 0.05)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results/fix14_fidelity.json'), 'w'), indent=1); print(json.dumps(res, indent=1))
