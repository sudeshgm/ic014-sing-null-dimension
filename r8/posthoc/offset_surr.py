"""POST HOC (3 Nov 2026): AS with content-free surrogates at CENTRED null energy, and AS of a pure offset. Re-uses r3/leakfree.setup/ang
and the r4/surrogates.py draws (seed 92000+r, 3 draws), SING released translators, class text, all 13 models. Variants (null coords C):
  off_nat : every image gets mu (the model's own null mean) at its natural size (no rescale) -> AS of the offset alone
  zm_c    : N(0, Sigma) rescaled to the centred null energy E||C-mu||^2 (r4 'zm' is rescaled to the full uncentred energy)
  perm_c  : (C-mu) permuted across images, natural size (= centred energy exactly)
  real_c  : the real centred null content C-mu (offset removed), natural size  [real model AS variant, not a round-8 score]
Writes offset_surr.json. No theta_sem / theta_cls computed."""
import os, sys, json, time, numpy as np, torch
R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); FR = os.environ['IC014_FEATS']; sys.path.insert(0, os.path.join(R, 'r3'))  # FR: private cached features
from leakfree import setup, ang
from common import load_models
torch.set_num_threads(6)
emb = np.load(os.path.join(os.environ.get('SING_ROOT', os.path.join(R, '_ext', 'sing')), 'misc/embeddings/imagenet_text_embeddings_fp16.npz'), allow_pickle=True)
TC = torch.nn.functional.normalize(torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float64)), dim=-1).mean(1), dim=-1)
ms, col, sing = load_models(); OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'offset_surr.json')
P = json.load(open(OUT)) if os.path.exists(OUT) else {}
for name in ms:
    if name in P: continue
    t0 = time.time(); E = setup(FR, name); tc = TC[E['y']]; a0 = ang(E['Zrow'], tc); N = E['F'].shape[0]
    C = E['F'] @ E['Vn']; Cc = C - E['mu']; Ec = float((Cc ** 2).sum(1).mean())
    AS_of = lambda Ct: float((ang(E['Zrow'] + Ct @ E['AVn'].T, tc) - a0).abs().mean())
    o = dict(AS=AS_of(C), off_nat=AS_of(E['mu'].expand(N, -1)), real_c=AS_of(Cc))
    zm, pc = [], []
    for r in range(3):
        g = torch.Generator().manual_seed(92_000 + r); z = torch.randn(N, C.shape[1], generator=g, dtype=torch.float64) @ E['Ls'].T
        zm.append(AS_of(z * (Ec / float((z ** 2).sum(1).mean())) ** 0.5))
        pc.append(AS_of(Cc[torch.tensor(np.random.default_rng(92_000 + r).permutation(N))]))
    o.update(zm_c=float(np.mean(zm)), zm_c_sd=float(np.std(zm)), perm_c=float(np.mean(pc)), perm_c_sd=float(np.std(pc)),
             rho=E['rho'], rho_c=Ec / E['Etot'], secs=time.time() - t0)
    P[name] = o; json.dump(P, open(OUT, 'w'), indent=1); print(name, {k: round(v, 4) for k, v in o.items()}, flush=True); del E, C, Cc
