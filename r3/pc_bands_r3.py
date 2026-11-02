"""PREREG_R3 C (N2): fix-13 PC bands re-benchmarked against the fitted energy law AS ~ rho^a_hat.
Same bands, seeds and image bootstrap as r2/pc_bands.py (top-variance null PCs vs 20 random null sub-subspaces,
seed 40000+j, d in {24, 100}, 1000x image bootstrap seed 0). H = (AS_top/rho_top^a)/(AS_rand/rho_rand^a), a = 0.431
(E1 pooled fit), sensitivity a in {0.388, 0.485}. The random term is the mean over the 20 draws (as G in fix 13).
Counted models: resnet, resnext, dinov2; bitresnet reported only.
Usage: python r3/pc_bands_r3.py --feats-root $IC014_FEATS [--sing-root _ext/sing]"""
import argparse, os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import R, dump
torch.set_num_threads(os.cpu_count())
BIG = ['resnet', 'resnext', 'dinov2', 'bitresnet']; COUNTED = BIG[:3]; DS = [24, 100]; AH = [0.431, 0.388, 0.485]


def ang(Z, T): return torch.rad2deg(torch.acos(torch.clamp((torch.nn.functional.normalize(Z, dim=-1) * T).sum(-1), -1, 1)))


def label(vs):
    n = len(vs)
    if sum(v == 'covers1' for v in vs) * 3 >= 2 * n: return 'consistent with the energy law'
    if sum(v == 'top>rand' for v in vs) * 3 >= 2 * n: return 'top PCs carry more AS than energy predicts'
    if sum(v == 'top<rand' for v in vs) * 3 >= 2 * n: return 'top PCs carry less AS than energy predicts'
    return 'mixed'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args()
    fr = a.feats_root; emb = np.load(os.path.join(a.sing_root, 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
    TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float32)), dim=-1).mean(1), dim=-1)
    out = {}; rng = np.random.default_rng(0); N = 10000; BI = rng.integers(0, N, size=(1000, N))
    for nm in BIG:
        d = np.load(f'{fr}/e3/ret/e3_feats/{nm}_full.npz'); p = np.load(f'{fr}/e3/private/{nm}_null.npz')
        F = torch.tensor(d['F']).float(); A = torch.tensor(d['A']).float(); Ab = torch.tensor(d['Ab']).float(); y = torch.tensor(d['y']).long()
        Vn = torch.tensor(p['Vn']).float(); C = torch.tensor(p['c']).float(); nd = Vn.shape[1]; t = TC[y]
        Z = F @ A.T + Ab; a0 = ang(Z, t); Etot_i = (F.double() ** 2).sum(1); Etot = float(Etot_i.mean())
        ev, U = torch.linalg.eigh((C.double().T @ C.double()) / N); U = U[:, torch.argsort(ev, descending=True)].float(); AVn = A @ Vn
        def band(Bc):
            cb = C @ Bc; return (a0 - ang(Z - cb @ (AVn @ Bc).T, t)).abs().double(), (cb.double() ** 2).sum(1)
        res = {}
        for dd in DS:
            top = band(U[:, :dd]); rnd = []
            for j in range(20):
                g = torch.Generator().manual_seed(40_000 + j); rnd.append(band(torch.linalg.qr(torch.randn(nd, nd, generator=g))[0][:, :dd]))
            RA = torch.stack([v[0] for v in rnd]); RE = torch.stack([v[1] for v in rnd])
            rho_top = float(top[1].mean() / Etot); rho_r = [float(v[1].mean() / Etot) for v in rnd]
            AS_top = float(top[0].mean()); AS_r = [float(v[0].mean()) for v in rnd]
            G_obs = (AS_top / rho_top ** 0.5) / float(np.mean([x / r ** 0.5 for x, r in zip(AS_r, rho_r)]))
            res[str(dd)] = dict(rho_top=rho_top, rho_rand=float(np.mean(rho_r)), AS_top=AS_top, AS_rand=float(np.mean(AS_r)), G_round2=G_obs, H={})
            for ah in AH:
                H = (AS_top / rho_top ** ah) / float(np.mean([x / r ** ah for x, r in zip(AS_r, rho_r)])); bs = []
                for b in BI:
                    idx = torch.tensor(b); den = Etot_i[idx].mean()
                    ht = float(top[0][idx].mean() / (top[1][idx].mean() / den) ** ah)
                    hr = float((RA[:, idx].mean(1) / (RE[:, idx].mean(1) / den) ** ah).mean()); bs.append(ht / hr)
                lo, hi = np.percentile(bs, [2.5, 97.5])
                res[str(dd)]['H'][str(ah)] = dict(H=H, ci95=[float(lo), float(hi)], verdict='covers1' if lo <= 1 <= hi else ('top>rand' if lo > 1 else 'top<rand'),
                                                 law_prediction_of_G=(rho_top / float(np.mean(rho_r))) ** (ah - 0.5))
        out[nm] = res
        print(nm, json.dumps({dd: {ah: (round(v['H'][ah]['H'], 3), [round(x, 3) for x in v['H'][ah]['ci95']], v['H'][ah]['verdict']) for ah in v['H']} for dd, v in res.items()}), flush=True)
        del F, C, Z
    lab = {dd: {str(ah): label([out[m][dd]['H'][str(ah)]['verdict'] for m in COUNTED]) for ah in AH} for dd in map(str, DS)}
    r2 = json.load(open(os.path.join(R, 'r2/results/fix13_pc_bands.json')))
    summ = dict(labels=lab, primary_label_d24=lab['24']['0.431'], robust_d24=len(set(lab['24'].values())) == 1,
                round2_label_verbatim=r2.get('decision_d24'), round2_screened_count_d24=sum(r2[m]['24']['verdict'] == 'top<rand' for m in COUNTED),
                round2_d100_G={m: r2[m]['100']['ratio_top_over_rand'] for m in BIG}, counted=COUNTED, a_hat=AH)
    out['summary'] = summ; dump('C_pc_bands_r3.json', out); print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
