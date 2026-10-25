"""CPU diagnosis of the BiT full-null vs random-subspace AS anomaly, on 100 Imagenette-160 val images (no ImageNet-scale work).
Same code path as run_exp1_gpu.py (sing load_model / translator / projectors)."""
import sys, os, glob, json, numpy as np, torch
sys.path.insert(0, '/tmp/sing/src')
from PIL import Image
from sing.models.registry import load_model
from sing.translators.registry import load_default_translator
from sing.core.projectors import compute_projectors_from_weight
from pathlib import Path
torch.manual_seed(0)
syn = sorted(l.split()[0] for l in open('/dev/stdin')) if False else None
root = '/workspace/ic013_data/imagenette2-160/val'
files = [f for c in sorted(os.listdir(root)) for f in sorted(glob.glob(f'{root}/{c}/*'))[:10]]
emb = np.load('/tmp/sing/misc/embeddings/imagenet_text_embeddings_fp16.npz', allow_pickle=True)
E = torch.nn.functional.normalize(torch.tensor(emb['embeddings'].astype(np.float32)), dim=-1)
Em = torch.nn.functional.normalize(E.mean(1), dim=-1)
def ang(a, b):
    a = torch.nn.functional.normalize(a, dim=-1); b = torch.nn.functional.normalize(b, dim=-1)
    return torch.rad2deg(torch.acos(torch.clamp((a * b).sum(-1), -1, 1)))
out = {}
for name in sys.argv[1:] or ['bitresnet', 'resnet']:
    lm = load_model(name, torch.device('cpu'))
    T = load_default_translator(translators_root=Path('/tmp/sing/translators'), registry_path=Path('/tmp/sing/translators/registry.yaml'), model_name=name, device='cpu').model.float()
    with torch.no_grad():
        F = torch.cat([lm.wrapper.extract_features(torch.stack([lm.preprocess(Image.open(f).convert('RGB')) for f in files[i:i+20]])) for i in range(0, len(files), 20)])
        logits = lm.wrapper.logits_from_features(F); Y = logits.argmax(1)  # predicted class as text target (no labels needed)
        W = lm.wrapper.classifier_weight.float(); P = compute_projectors_from_weight(W); Vn = P.v_null
        A = T.weight.float()
        nd = Vn.shape[1]; pn = F @ Vn  # null coordinates
        z = T(F); ty = Em[Y]
        res = dict(null_dim=nd, feat_norm=float(F.norm(dim=1).mean()), null_coord_mean_norm=float(pn.mean(0).norm()),
                   null_coord_rms=float(pn.norm(dim=1).mean()), null_coord_std_norm=float(pn.std(0).norm()),
                   A_gain_null_fro=float((A @ Vn).norm() / np.sqrt(nd)), A_gain_principal_fro=float((A @ P.v_principal).norm() / np.sqrt(W.shape[0])),
                   bias_norm=float(T.bias.norm()), z_norm=float(z.norm(dim=1).mean()))
        rows = {}
        for d in [24, 100, 536, nd]:
            vals = []
            for s in range(5 if d < nd else 1):
                Q = Vn if d == nd else Vn @ torch.linalg.qr(torch.randn(nd, d, generator=torch.Generator().manual_seed(1000 * d + s)))[0]
                dz = (F @ Q) @ Q.T @ A.T          # translator shift from removal
                zp = z - dz
                vals.append([float((ang(z, ty) - ang(zp, ty)).abs().mean()), float(ang(z, zp).mean()),
                             float(dz.norm(dim=1).mean()), float((dz.mean(0)).norm()), float((F @ Q).norm(dim=1).mean())])
            rows[d] = np.mean(vals, 0).round(4).tolist()
        res['per_d [absAS, IS, |dz|, |mean dz|, |proj f|]'] = rows
        # additivity check: partition the null space into 1048/ d blocks of a random rotation; dz sums exactly (linear) -> expected |dz_full|
        R = Vn @ torch.linalg.qr(torch.randn(nd, nd, generator=torch.Generator().manual_seed(7)))[0]
        blocks = [R[:, i:i + 131] for i in range(0, nd, 131)]
        dzs = [(F @ B) @ B.T @ A.T for B in blocks]
        res['additivity'] = dict(sum_block_dz_norm=float(sum(dzs).norm(dim=1).mean()), full_dz_norm=float(((F @ Vn) @ Vn.T @ A.T).norm(dim=1).mean()),
                                 sqrt_sum_sq_block_norms=float(torch.stack([x.norm(dim=1) ** 2 for x in dzs]).sum(0).sqrt().mean()))
        # spectrum of A restricted to null space and the share of the actual null component it sees
        sv = torch.linalg.svdvals(A @ Vn)
        res['A_null_sv_top5'] = sv[:5].tolist(); res['A_null_sv_median'] = float(sv.median())
    out[name] = res; print(name, json.dumps(res), flush=True)
json.dump(out, open('results/diag_bit.json', 'w'), indent=1)
