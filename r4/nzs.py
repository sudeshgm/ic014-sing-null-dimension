"""PREREG_R4C B3: null zero-shot accuracy (NZS) = top-1 of argmax_c cos(A Pi f, t_c), no bias, minus a control with null
coordinates permuted across images (seed 97000). Validity gate first: (a) rescaling invariance beta in {0.1,1,10};
(b) planted class-text content at fixed energy on ResNet50, ResNeXt101, DINOv2-L (tau(NZS, p) >= 0.6 on all 3); NCM_null gated
descriptively. Then cross-model tests (only if the gate passes). DINO ViT-B/16, ViT-S/16 with ridge translators (descriptive).
Outputs r4/results/B3_nzs.json. Usage: python r4/nzs.py --feats-root $IC014_FEATS"""
import argparse, os, sys, json, time, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'r3'))
from common4 import R, RES4, load_models, tau_ci, ktau, sets, partial_ci, dump4
from retrained_a4 import text_emb, split, retrain, null_basis
from retrain_translators import find
import estimators as est
torch.set_num_threads(os.cpu_count()); nz = torch.nn.functional.normalize
BASES = ['resnet', 'resnext', 'dinov2']; PS = [0.0, 0.01, 0.03, 0.1, 0.3]


def topk(D, TCn, y, chunk=2000):
    t1 = t5 = 0
    for i in range(0, len(D), chunk):
        S = nz(D[i:i + chunk], dim=1) @ TCn.T; k = S.topk(5, dim=1).indices; yy = y[i:i + chunk]
        t1 += int((k[:, 0] == yy).sum()); t5 += int((k == yy[:, None]).any(1).sum())
    return 100 * t1 / len(D), 100 * t5 / len(D)


def nzs(C, AVt, TCn, y):
    t1, t5 = topk(C @ AVt, TCn, y); g = torch.Generator().manual_seed(97_000); p = torch.randperm(len(C), generator=g)
    c1, c5 = topk(C[p] @ AVt, TCn, y); return dict(top1=t1, top5=t5, ctrl_top1=c1, ctrl_top5=c5, NZS=t1 - c1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--feats-root', required=True)
    ap.add_argument('--sing-root', default=os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing'))); a = ap.parse_args(); fr = a.feats_root
    TC = text_emb(a.sing_root); TCn = nz(TC, dim=1); ms, col, sing = load_models(); lk = -sing; rho = col('rho'); S = sets(ms, rho)
    gate = {}; P = {}
    for name in ms:
        t0 = time.time(); d = np.load(find(fr, name)); F = torch.tensor(d['F']).double(); y = torch.tensor(d['y']).long(); Vn = null_basis(torch.tensor(d['W']))
        A = torch.tensor(d['A']).double(); C = F @ Vn; AVt = Vn.T @ A.T
        if name in BASES:
            g = {'rescale': {}, 'planted': {}}
            for b in (0.1, 1.0, 10.0): g['rescale'][str(b)] = nzs(b * C, AVt, TCn, y)['NZS']
            b1 = g['rescale']['1.0']; g['rescale_max_rel_change'] = max(abs(v - b1) / max(abs(b1), 1e-12) for v in g['rescale'].values())
            En = float(C.pow(2).sum(1).mean()); mu = C.mean(0); Cc = C - mu; ev, U = torch.linalg.eigh(Cc.T @ Cc / len(C)); L = U * ev.clamp_min(0).sqrt()
            gg = torch.Generator().manual_seed(70_000); Ct = mu + torch.randn(C.shape, generator=gg, dtype=C.dtype) @ L.T; Ct = Ct * math.sqrt(En / float(Ct.pow(2).sum(1).mean()))
            u = nz(TC @ AVt.T, dim=1)  # (1000, nd): null direction mapping most onto each class text
            for p in PS:
                Cp = math.sqrt(1 - p) * Ct + math.sqrt(p * En) * u[y]; Cp = Cp * math.sqrt(En / float(Cp.pow(2).sum(1).mean()))
                o = nzs(Cp, AVt, TCn, y); o['NCM_null'] = est.ncm_loo(Cp, y)['NCM_null']; g['planted'][str(p)] = o
            g['tau_NZS_p'] = ktau(np.array([g['planted'][str(p)]['NZS'] for p in PS]), np.array(PS))
            g['tau_NCM_p'] = ktau(np.array([g['planted'][str(p)]['NCM_null'] for p in PS]), np.array(PS))
            gate[name] = g; print('gate', name, json.dumps(g)[:600], flush=True)
        P[name] = nzs(C, AVt, TCn, y); P[name]['rho'] = float(rho[ms.index(name)]); print(name, round(time.time() - t0, 1), P[name], flush=True); del F, C
    gpass = all(gate[b]['rescale_max_rel_change'] <= 0.05 and gate[b]['tau_NZS_p'] >= 0.6 for b in BASES)
    res = dict(gate=gate, gate_pass=gpass, per_model=P, sets=S, tau={}, tau_vs_rho={}, partial_given_rho={})
    if gpass:
        L = lambda sub: np.array([lk[ms.index(n)] for n in sub]); V = lambda sub: np.array([P[n]['NZS'] for n in sub]); Rh = lambda sub: np.array([rho[ms.index(n)] for n in sub])
        for s, sub in S.items():
            res['tau'][s] = tau_ci(V(sub), L(sub)); res['tau_vs_rho'][s] = tau_ci(V(sub), Rh(sub)); res['partial_given_rho'][s] = partial_ci(V(sub), L(sub), Rh(sub))
        ef = all(abs(res['tau_vs_rho'][s]['tau']) < 0.3 for s in ('E12', 'S11')); ts = [res['tau'][s]['tau'] for s in ('E12', 'S11')]
        res['energy_free'] = ef; res['sing_test'] = 'survives' if all(t >= 0.6 for t in ts) else ('not recovered' if all(t < 0.3 for t in ts) else 'inconclusive')
        res['stop'] = res['sing_test'] != 'inconclusive'
    Y = torch.tensor(np.load(f'{fr}/e3/ret/e3_feats/clip_img.npz')['e']).double(); tr, te = split(); res['dino'] = {}
    for name in ('dino_vitb16', 'dino_vits16'):
        d = np.load(f'{fr}/r3_dino_feats/{name}_10k.npz', allow_pickle=True); F = torch.tensor(d['F']).double(); y = torch.tensor(d['labels'].astype(np.int64)); Vn = null_basis(torch.tensor(d['W']))
        Wt, bt, _, _ = retrain(F, Y, tr); C = F[te] @ Vn; res['dino'][name] = nzs(C, Vn.T @ Wt, TCn, y[te]); print(name, res['dino'][name], flush=True)
    dump4('B3_nzs.json', res); print({k: res.get(k) for k in ('gate_pass', 'tau', 'tau_vs_rho', 'partial_given_rho', 'energy_free', 'sing_test', 'stop')})


if __name__ == '__main__':
    main()
