"""Round-2 fix 13: is null energy a nuisance or part of the signal? Remove d-dim bands of null principal components
(uncentred second moment of Vn^T f) and compare AS per unit sqrt-energy, G = AS/sqrt(rho), with random null subspaces.
Models: the four null-1048 models. Bands: top, middle (centred on PC 524), bottom, random (20 Haar draws in the null,
seed 40000+j). d in {24, 100}. Image bootstrap (1000x, seed 0) of G_top/G_rand.
Usage: python r2/pc_bands.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, json, numpy as np, torch
torch.set_num_threads(os.cpu_count()); R = os.path.join(os.path.dirname(__file__), '..')
BIG = ['resnet', 'resnext', 'dinov2', 'bitresnet']; DS = [24, 100]

def ang(Z, T): return torch.rad2deg(torch.acos(torch.clamp((torch.nn.functional.normalize(Z, dim=-1) * T).sum(-1), -1, 1)))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True); ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_ext', 'sing'))); a = ap.parse_args()
    fr = a.feats_root; emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float32)), dim=-1).mean(1), dim=-1)
    out = {}; rng = np.random.default_rng(0); N = 10000; BI = rng.integers(0, N, size=(1000, N))
    for nm in BIG:
        d = np.load(f'{fr}/e3/ret/e3_feats/{nm}_full.npz'); p = np.load(f'{fr}/e3/private/{nm}_null.npz')
        F = torch.tensor(d['F']).float(); A = torch.tensor(d['A']).float(); Ab = torch.tensor(d['Ab']).float(); y = torch.tensor(d['y']).long()
        Vn = torch.tensor(p['Vn']).float(); C = torch.tensor(p['c']).float(); nd = Vn.shape[1]; t = TC[y]
        Z = F @ A.T + Ab; a0 = ang(Z, t); Etot_i = (F.double() ** 2).sum(1); Etot = float(Etot_i.mean())
        ev, U = torch.linalg.eigh((C.double().T @ C.double()) / N); U = U[:, torch.argsort(ev, descending=True)].float()
        AVn = A @ Vn
        def band(Bc):   # Bc: nd x d orthonormal coords in the null
            cb = C @ Bc; asi = (a0 - ang(Z - cb @ (AVn @ Bc).T, t)).abs().double(); ei = (cb.double() ** 2).sum(1); return asi, ei
        res = {}
        for dd in DS:
            bands = {'top': U[:, :dd], 'mid': U[:, 524 - dd // 2: 524 - dd // 2 + dd], 'bottom': U[:, nd - dd:]}
            per = {}
            for k, Bc in bands.items():
                asi, ei = band(Bc); per[k] = (asi, ei)
            rnd = []
            for j in range(20):
                g = torch.Generator().manual_seed(40_000 + j); Bc = torch.linalg.qr(torch.randn(nd, nd, generator=g))[0][:, :dd]; rnd.append(band(Bc))
            def G(asi, ei, idx=None):
                if idx is None: return float(asi.mean() / torch.sqrt(ei.mean() / Etot))
                return float(asi[idx].mean() / torch.sqrt(ei[idx].mean() / Etot_i[idx].mean()))
            Gs = {k: G(*v) for k, v in per.items()}; Gs['random'] = float(np.mean([G(*v) for v in rnd]))
            rhos = {k: float(v[1].mean() / Etot) for k, v in per.items()}; rhos['random'] = float(np.mean([float(v[1].mean() / Etot) for v in rnd]))
            ASs = {k: float(v[0].mean()) for k, v in per.items()}; ASs['random'] = float(np.mean([float(v[0].mean()) for v in rnd]))
            RA = torch.stack([v[0] for v in rnd]); RE = torch.stack([v[1] for v in rnd]); bs = []
            for b in BI:
                idx = torch.tensor(b); den = Etot_i[idx].mean()
                gr = float((RA[:, idx].mean(1) / torch.sqrt(RE[:, idx].mean(1) / den)).mean())
                bs.append(G(*per['top'], idx) / gr)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            res[str(dd)] = dict(G=Gs, rho=rhos, AS=ASs, ratio_top_over_rand=Gs['top'] / Gs['random'], ci95=[float(lo), float(hi)],
                                verdict='covers1' if lo <= 1 <= hi else ('top>rand' if lo > 1 else 'top<rand'))
        out[nm] = res; print(nm, json.dumps({k: (round(v['ratio_top_over_rand'], 3), [round(x, 3) for x in v['ci95']], v['verdict']) for k, v in res.items()}), flush=True)
    v24 = [out[m]['24']['verdict'] for m in BIG]
    dec = ('energy is a nuisance' if v24.count('covers1') >= 3 else 'energy is partly signal' if v24.count('top>rand') >= 3
           else 'top PCs carry less AS per unit energy' if v24.count('top<rand') >= 3 else 'mixed')
    out['decision_d24'] = dec; out['note'] = 'BiT flagged: translator conditioning (E1/E2).'
    json.dump(out, open(os.path.join(R, 'r2/results/fix13_pc_bands.json'), 'w'), indent=1); print('decision:', dec)

if __name__ == '__main__': main()
